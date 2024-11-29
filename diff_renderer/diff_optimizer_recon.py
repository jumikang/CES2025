import os
import cv2
import torch
import numpy as np
import nvdiffrast.torch as dr
import pickle
import trimesh
import torchvision
from torchvision import transforms
import tempfile
torchvision.disable_beta_transforms_warning()

from PIL import Image
from torch import nn
from tqdm import tqdm

from smpl_optimizer.smpl_wrapper import BaseWrapper
from diff_renderer.normal_nds.nds.core import Camera
from diff_renderer.normal_nds.nds.losses import laplacian_loss
from diff_renderer.normal_nds.nds.core.mesh_ext import TexturedMesh
from diff_renderer.normal_nds.nds.core.mesh_smpl import SMPLMesh
from libs.one_click_densepose.un_unwrapping import DensePoser
from models.unet.deep_human_models import DeepHumanUVNet, DeepHumanColorNet
from rembg import remove
# from transformers import AutoModelForImageSegmentation

class Renderer:
    def __init__(self, params, res=1024, near=1, far=1000, orthographic=False, device='cuda'):

        self.max_mip_level = params['max_mip_level']
        self.angle_interval = params['angles']

        self.res = res
        self.device = device
        self.near = near
        self.far = far
        self.orthographic = orthographic

    def set_near_far(self, views, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
        """
        mins = []
        maxs = []
        for view in views:
            samples_projected = view.project(samples, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device)

    @staticmethod
    def ortho(left, right, bottom, top, zNear, zFar):
        # res = np.ones([4, 4], dtype=np.float32)
        res = np.identity(4, dtype=np.float32)
        res[0][0] = 2 / (right - left)
        res[1][1] = 2 / (top - bottom)
        res[2][2] = - 2 / (zFar - zNear)
        res[3][0] = - (right + left) / (right - left)
        res[3][1] = - (top + bottom) / (top - bottom)
        res[3][2] = - (zFar + zNear) / (zFar - zNear)
        return res.T

    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000, orthographic=False):
        if orthographic:
            z_near = 1
            z_far = -300
            ortho_ratio = 0.4 * (512 / resolution[0])
            # focal = np.sqrt(resolution * resolution + resolution * resolution)

            left = -resolution[1] * ortho_ratio / 2
            right = resolution[1] * ortho_ratio / 2
            bottom = -resolution[0] * ortho_ratio / 2
            top = resolution[0] * ortho_ratio / 2

            # int_mat = np.eye(3)
            # int_mat[0, 0] = focal
            # int_mat[1, 1] = focal
            # int_mat[0, 1] = 0
            # int_mat[0, 2] = resolution / 2
            # int_mat[1, 2] = resolution / 2

            res = np.identity(4, dtype=np.float32)
            res[0][0] = 2 / (right - left)
            res[1][1] = 2 / (top - bottom)
            res[2][2] = - 2 / (z_far - z_near)
            res[3][0] = - (right + left) / (right - left)
            res[3][1] = - (top + bottom) / (top - bottom)
            res[3][2] = - (z_far + z_near) / (z_far - z_near)
            projection_matrix = torch.tensor(res.T).to(camera.device)
            # projective = np.zeros([4, 4])
            # projective[:2, :2] = int_mat[:2, :2]
            # projective[:2, 2:3] = -int_mat[:2, 2:3]
            # projective[3, 2] = -1
            # projective[2, 2] = (z_near + z_far)
            # projective[2, 3] = (z_near * z_far)
            # projection_matrix = np.matmul(ndc, projective)

            # projection_matrix = torch.eye(4, device=camera.device)
            # projection_matrix[:3, :3] = camera.K
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  -1., 0,  0],
                                        [0,  0, 1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)
        else:
            projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                    fy=camera.K[1,1],
                                                    cx=camera.K[0,2],
                                                    cy=camera.K[1,2],
                                                    n=n,
                                                    f=f,
                                                    width=resolution[1],
                                                    height=resolution[0],
                                                    device=camera.device)
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  -1., 0,  0],
                                        [0,  0, 1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    def get_gl_camera(self, camera, resolution):
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        return P

    def transform_pos(self, mtx, pos):
        t_mtx = torch.from_numpy(mtx) if isinstance(mtx, np.ndarray) else mtx
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    def render(self, mesh, render_options,
               pos, pos_idx, pos_clip,
               rast, rast_db,
               verts_init=None,
               enable_mip=True):

        render_out = {}
        if render_options["color"] is True or render_options["offset"] is True:
            uv, tex, disp = mesh.uv_vts, mesh.tex, mesh.disp

        if render_options["color"]:
            if enable_mip:
                texc, texd = dr.interpolate(uv[None, ...], rast, pos_idx, rast_db=rast_db, diff_attrs='all')
                color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear',
                                   max_mip_level=self.max_mip_level)
            else:
                texc, _ = dr.interpolate(uv[None, ...], rast, pos_idx)
                color = dr.texture(tex[None, ...], texc, filter_mode='linear')[0]
            render_out["color"] = color * torch.clamp(rast[..., -1:], 0, 1)  # Mask out background.

        if render_options["mask"]:
            mask = torch.clamp(rast[..., -1:], 0, 1)
            mask = dr.antialias(mask, rast, pos_clip, pos_idx)[0] # if with_antialiasing else mask[0]
            render_out["mask"] = mask

        if render_options["normal"]:
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, pos_idx)
            render_out["normal"] = dr.antialias(normal, rast, pos_clip, pos_idx)[0] # if with_antialiasing else normal[0]

        if render_options["depth"]:
            position, _ = dr.interpolate(pos[None, ...], rast, pos_idx)
            render_out["depth"] = dr.antialias(position, rast, pos_clip, pos_idx)[0] # if with_antialiasing else position[0]
            # gbuffer["depth"] = view.project(gbuffer["position"], depth_as_distance=True)[..., 2:3]

        # for future use
        if render_options["offset"]:
            texc, _ = dr.interpolate(uv[None, ...], rast, pos_idx)
            render_out["disp_uv"] = dr.texture(disp[None, ...].contiguous(), texc, filter_mode='linear')[0]

            position, _ = dr.interpolate(pos[None, ...] - verts_init[None, ...], rast, pos_idx)
            render_out["disp_cv"] = position[0]

        return render_out  # texture map only

    def get_vert_visibility(self, glctx, mesh, camera, resolution=1024):
        vertices = mesh.vertices
        idx = mesh.indices.int()
        num_verts = len(vertices)

        with torch.no_grad():
            # for camera in cameras:
            vis_mask = torch.zeros(size=(num_verts,), device=self.device).bool()  # visibility mask
            P = Renderer.to_gl_camera(camera, [resolution, resolution],
                                      n=self.near, f=self.far,
                                      orthographic=self.orthographic)
            pos = Renderer.transform_pos(P, vertices)
            rast, rast_out_db = dr.rasterize(glctx, pos, idx, resolution=np.array([resolution, resolution]))

            # Do not support batch operation yet
            face_ids = rast[..., -1].long()
            masked_face_idxs_list = face_ids[face_ids != 0] - 1  # num_masked_face Tensor
            # masked_face_idxs_all = torch.unique(torch.cat(masked_face_idxs_list, dim=0))
            masked_verts_idxs = torch.unique(idx[masked_face_idxs_list].long())
            vis_mask[masked_verts_idxs] = 1
            vis_mask = vis_mask.bool().to(self.device)
            # vis_masks.append(vis_mask)
        return vis_mask

class Optimizer_recon(nn.Module):
    def __init__(self,
                 params,
                 orthographic=False,
                 device='cuda:0'):
        super(Optimizer_recon, self).__init__()

        # set params
        self.params = params
        self.cam_params = params['CAM']
        self.uv_config = self.params['UV_MAPPING']
        self.render_res = params['RENDER']['resolution_render']
        self.res = params['CAM']['width']
        self.use_opengl = True
        self.orthographic = orthographic
        self.device = device

        # set opt. and differentiable rendering options
        self.max_iter = params['OPT']['max_iter']
        self.lr = params['OPT']['lr_base']
        self.render = Renderer(params['RENDER'], orthographic=self.orthographic, device=device)
        self.weights = {'color': 1.0, 'per': 0.1, 'depth': 1e-1, 'disp': 1e-1, 'normal': 1.0, 'texture': 1e-1,
                        'seam': 50.0, 'smooth': 1e-4, 'laplacian': 10.0, 'symmetry': 0.0}
        self.log_interval = 10
        self.scheduler_interval = 100

        # unet prediction related
        self.densepose_predictor = DensePoser(params['DATA']['densepose_root'])
        self.model_uv = DeepHumanUVNet(params)
        self.model_color = DeepHumanColorNet(params)

        # load unet checkpoint
        check_point_pre = torch.load(params['DATA']['pretrained_model'])
        check_point_color_pre = torch.load(params['DATA']['pretrained_color_model'])
        self.model_uv.load_state_dict(check_point_pre['state_dict'])
        self.model_color.load_state_dict(check_point_color_pre['state_dict'])

        # set render params
        self.RGB_MAX = torch.Tensor(params['RENDER']['RGB_MAX'])
        self.RGB_MEAN = torch.Tensor(params['RENDER']['RGB_MEAN'])
        self.RGB_STD = torch.Tensor(params['RENDER']['RGB_STD'])
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=int(self.res/self.render_res))

        # set cameras
        self.glctx = dr.RasterizeGLContext() if self.use_opengl else dr.RasterizeCudaContext()
        self.cameras = self.set_cams_from_angles(orthographic=self.orthographic) # degree
        self.view_num = len(self.cameras)

        # set smplx params
        self.smpl_mesh = SMPLMesh(smpl_config=self.uv_config, device=self.device)
        with open(os.path.join(self.uv_config['smpl_root'], self.uv_config['lbs_weights']), 'rb') as f:
            self.lbs_weights = pickle.load(f)

        # set losses
        self.l2_loss = nn.MSELoss()

        # set render options
        self.render_options = {'color': True,
                               'depth': False,
                               'normal': True,
                               'mask': True,
                               'offset': True}

        # rmbg-2.0 model
        # self.rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0',
        #                                                                 trust_remote_code=True)
        # torch.set_float32_matmul_precision(['high', 'highest'][0])
        # self.rmbg_model.to(device)
        # self.rmbg_model.eval()

        # set mask kernels
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.close_kernel = np.ones((7, 7), np.uint8)

        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.normalize_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()
        ])

    def tv_loss(self, img):
        xv = img[1:, :, :] - img[:-1, :, :]
        yv = img[:, 1:, :] - img[:, :-1, :]
        loss = torch.mean(abs(xv)) + torch.mean(abs(yv))
        return loss

    def load_inputs(self, img, dense_uv):
        img_orig = img.copy()
        if not img.shape[1] == self.render_res:
            img = cv2.resize(img, dsize=(self.render_res, self.render_res), interpolation=cv2.INTER_AREA)

        if img.shape[2]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        image_tensor = torch.FloatTensor(img).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))

        image_tensor = torch.FloatTensor(img_orig).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_orig_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))

        if not dense_uv.shape[1] == self.render_res:
            dense_uv = cv2.resize(dense_uv, dsize=(self.render_res, self.render_res), interpolation=cv2.INTER_AREA)
        if dense_uv.shape[2]==4:
            dense_uv = cv2.cvtColor(dense_uv, cv2.COLOR_BGRA2RGB)
        else:
            dense_uv = cv2.cvtColor(dense_uv, cv2.COLOR_BGR2RGB)

        uv_tensor = torch.FloatTensor(dense_uv).permute(2, 0, 1)
        uv_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_uv_tensor =\
            ((uv_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))

        return {"image_1024":scaled_image_orig_tensor[None, ...],
                "image_512":scaled_image_tensor[None, ...],
                "dense_512": scaled_uv_tensor[None, ...]}

    # def rmbg(self, image):
    #     input_images = self.transform_image(image).unsqueeze(0).to(self.device)
    #
    #     # Prediction
    #     with torch.no_grad():
    #         preds = self.rmbg_model(input_images)[-1].sigmoid().cpu()
    #     pred = preds[0].squeeze()
    #     mask = pred[:, :, None].expand(pred.shape[0], pred.shape[1], 3)
    #     mask_np = mask.detach().cpu().numpy() * 255.0
    #     # mask_np = cv2.erode(mask_np, self.erode_kernel, cv2.BORDER_REFLECT, iterations=1)
    #     # mask_np[mask_np < 128] = 0
    #     # mask_np[mask_np > 128] = 255
    #     mask = torch.Tensor(mask_np/255).to(torch.uint8).to(self.device)
    #     # cv2.imwrite('tgt_mask.png', mask_np.astype(np.uint8))
    #     return mask

    def pred_uv_disp(self, input_var):
        self.input_var = input_var
        self.input_var["image_input"] = Image.open(self.input_var["input_path"])
        # self.input_var["image_pred"] = [Image.open(self.input_var["input_path"].replace('input', 'pred_color_f')),
        #                                 Image.open(self.input_var["input_path"].replace('input', 'pred_color_b'))]
        # self.input_var["normal_pred"] = [Image.open(self.input_var["input_path"].replace('input', 'pred_normal_f')),
        #                                  Image.open(self.input_var["input_path"].replace('input', 'pred_normal_b'))]

        image = np.array(input_var["image_input"])
        if not image.shape[1] == self.res:
            image = cv2.resize(image, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)

        # mask = self.rmbg(Image.fromarray(image[:, :, 0:3]))
        mask = remove(image)
        alpha_image = mask.copy()
        mask = torch.Tensor(mask[:, :, -1][:, :, None]).expand(mask.shape[0],
                                                               mask.shape[1], 3).to(torch.uint8).to(self.device)
        mask[mask<=128] = 0
        mask[mask>128] = 1

        input_dense_uv = self.densepose_predictor.get_dense_uv(alpha_image)
        input_data = self.load_inputs(alpha_image, input_dense_uv)
        input = torch.cat([input_data['image_512'], input_data['dense_512']], dim=1)
        self.input_var["uv_pred"] = self.model_uv.in_the_wild_step(input)
        self.input_var["color_pred"] = self.model_color.in_the_wild_step(input_data['image_1024'])
        self.input_var["color_pred"]['color'][0]  = (self.input_var["color_pred"]['color'][0]
                                                     * self.RGB_STD.view(3, 1, 1)
                                                     + self.RGB_MEAN.view(3, 1, 1))
        self.input_var["mask"] = mask

    def load_image_normal(self, image, normal=None):
        pred_color_f = None
        pred_color_np = (image[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        pred_color_b = self.normalize_image(Image.fromarray(pred_color_np))
        pred_color_b = pred_color_b.permute(1, 2, 0)

        if normal is not None:
            pred_normal_f = self.normalize_image(normal[0])
            pred_normal_b = self.normalize_image(normal[1])
            pred_normal_f = (pred_normal_f * 2 - 1).permute(1, 2, 0)
            pred_normal_b = (pred_normal_b * 2 - 1).permute(1, 2, 0)
            pred_normal_f = pred_normal_f[:, :, [2, 1, 0]]
            pred_normal_b = pred_normal_b[:, :, [2, 1, 0]]
        else:
            pred_normal_f = None
            pred_normal_b = None

        return pred_color_f, pred_color_b, pred_normal_f, pred_normal_b

    def set_cams_from_angles(self, device='cuda:0', orthographic=False):
        cameras = []
        for p in range(self.cam_params['pitch'][0],
                       self.cam_params['pitch'][1],
                       self.cam_params['pitch'][2]):
            for k in range(self.cam_params['view_angle'][0],
                           self.cam_params['view_angle'][1],
                           self.cam_params['view_angle'][2]):
                camera =\
                    Camera.perspective_camera_with_angle_recon(
                        view_angle=k, pitch=p, orthographic=orthographic,
                        cam_params=self.cam_params, device=device)
                cameras.append(camera)
        return cameras

    def init_ref_imgs(self):
        color_front = self.normalize_image(self.input_var["image_input"])[[2, 1, 0], :, :]

        if "normal_pred" in self.input_var:
            _, pred_color_b, pred_normal_f, pred_normal_b \
                = self.load_image_normal(self.input_var["color_pred"]["color"], self.input_var["normal_pred"])
            tgt_normals = [pred_normal_f.to(self.device), pred_normal_b.to(self.device)]
        else:
            _, pred_color_b, _, _ \
                = self.load_image_normal(self.input_var["color_pred"]["color"])
            tgt_normals = None
        tgt_images = [color_front.permute(1, 2, 0).to(self.device), pred_color_b.to(self.device)]

        return tgt_images, tgt_normals

    def smplx_base_recon(self):
        pred_disp_ = self.up_sample(self.input_var['uv_pred']['disp'])
        pred_uv_ = self.up_sample(self.input_var['uv_pred']['uv'])

        # set smpl-x params and deform canonical smplx
        smpl_params = BaseWrapper.load_params(self.input_var["input_smplx"])
        smpl_wrapper = BaseWrapper(self.uv_config, smpl_path=self.uv_config['smpl_root'])
        smpl_params = smpl_wrapper.to_gpu(smpl_params)

        self.smpl_mesh.set_canonical_smpl_in_uv(smpl_params)
        self.smpl_mesh.upsample()  # verts(N): 11313 -> 43525

        self.smpl_mesh.disp = pred_disp_[0].permute(1, 2, 0).to(self.device)
        disp = self.smpl_mesh.uv_sampling(mode='disp')
        tex_color = torch.flipud((pred_uv_[0].detach().cpu() * self.RGB_STD.view(3, 1, 1)
                                  + self.RGB_MEAN.view(3, 1, 1)).permute(1, 2, 0)).contiguous()
        # cv2.imwrite('pred_tex.png',
        #             ((torch.flipud(tex_color).detach().cpu().numpy()) * 255).astype(np.uint8))

        self.smpl_mesh.tex = tex_color
        self.smpl_mesh.update_seam(vertices=self.smpl_mesh.vertices)
        self.smpl_mesh.vertices += (disp / 100)
        self.smpl_mesh.lbs_weights = self.lbs_weights
        canonical_mesh = self.smpl_mesh.to_trimesh(with_texture=True)
        canonical_mesh.export('canon_recon.obj')

        # get posed smplx deformed model
        v_deformed = self.smpl_mesh.forward_skinning(smpl_params, self.smpl_mesh.vertices[None, :, :])
        self.mesh_pred = TexturedMesh(v_deformed,
                                      self.smpl_mesh.indices,
                                      self.smpl_mesh.uv_vts,
                                      self.smpl_mesh.seam,
                                      self.smpl_mesh.tex * 255,
                                      device=self.device)
        mesh_out = self.mesh_pred.to_trimesh(with_texture=True)
        mesh_out.export('pred_posed.obj')
        print('wait..')

    def pipeline(self):
        self()
        vertex_colors = self.mesh_pred.uv_sampling(mode='color')
        vertex_colors_new = torch.ones(vertex_colors.shape[0], 4)
        vertex_colors_new *= 255.0
        vertex_colors_new[:, 0:3] = vertex_colors[:, [2, 1, 0]] * 255

        opt_mesh = trimesh.Trimesh(vertices=(self.mesh_pred.vertices).detach().cpu().numpy(),
                                   faces=(self.mesh_pred.indices).detach().cpu().numpy(),
                                   vertex_colors=(vertex_colors_new[:, 0:3]).detach().numpy().astype(np.uint8))

        obj_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        obj_path = obj_file.name
        print(obj_path)
        opt_mesh.export(obj_path)

        return obj_path

    # update geometry, color, or both
    def forward(self):
        tgt_images, tgt_normals = self.init_ref_imgs()
        tgt_mask = self.input_var["mask"]

        # set optimize variable
        texture_init = self.mesh_pred.tex.clone()
        texture_opt = self.mesh_pred.tex
        texture_opt.requires_grad_()
        texture_opt.retain_grad()

        disp_opt = self.mesh_pred.disp.clone()
        disp_opt.requires_grad_()
        disp_opt.retain_grad()
        vertex_initial = self.mesh_pred.vertices.clone()
        vertex_normals = self.mesh_pred.vertex_normals.clone()

        # set optimize params
        opt_params = [texture_opt, disp_opt]
        optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        for k in tqdm(range(self.max_iter), 'opt. mesh:'):
            self.mesh_pred.disp = disp_opt
            self.mesh_pred.vertices = self.mesh_pred.uv_sampling(mode='disp') + vertex_initial
            pos, pos_idx = self.mesh_pred.vertices, self.mesh_pred.indices.int()
            render_pred = []
            loss = 0.0
            for v in range(2):
                mtx = self.render.to_gl_camera(self.cameras[v], [self.res, self.res],
                                               n=self.cam_params['near'], f=self.cam_params['far'])
                pos_clip = self.render.transform_pos(mtx, pos)

                with dr.DepthPeeler(self.glctx, pos_clip, pos_idx, [self.res, self.res]) as peeler:
                    for i in range(2):
                        rast, rast_db = peeler.rasterize_next_layer()
                        render_pred.append(self.render.render(self.mesh_pred,
                                                              self.render_options,
                                                              pos, pos_idx, pos_clip,
                                                              rast, rast_db,
                                                              verts_init=vertex_initial))
                # render_mask
                if v == 0:
                    pred_mask = render_pred[v]["mask"].expand(1024, 1024, 3)
                    mask = torch.zeros_like(pred_mask, dtype=torch.uint8).to(self.device)
                    mask[(tgt_mask + pred_mask) == 2] = 1

                # compute losses
                loss += self.l2_loss(render_pred[v]['color'][0] * mask, tgt_images[v] * mask) * self.weights["color"]
                loss += self.l2_loss(render_pred[v]["disp_uv"], render_pred[v]["disp_cv"]) * self.weights["disp"]
                if tgt_normals is not None and v==0:
                    loss += self.l2_loss(render_pred[v]['normal'] * mask, tgt_normals[v] * mask) * self.weights["normal"]

                # mesh smoothness
                loss += laplacian_loss(self.mesh_pred) * self.weights["laplacian"]

                # texture map smoothness
                loss += self.tv_loss(texture_opt) * self.weights["smooth"]

                # texture border consistency (important for uv-based diff. rendering)
                loss += self.l2_loss(self.mesh_pred.vertices[self.mesh_pred.seam[:, 0], :],
                                     self.mesh_pred.vertices[self.mesh_pred.seam[:, 1], :]) * self.weights["seam"]
                self.mesh_pred.tex = texture_opt
                vertex_colors = self.mesh_pred.uv_sampling(mode='color')
                loss += self.l2_loss(vertex_colors[self.mesh_pred.seam[:, 0], :],
                                     vertex_colors[self.mesh_pred.seam[:, 1], :]) * self.weights["seam"]
                loss += self.l2_loss(texture_opt, texture_init) * self.weights["texture"]

                loss += self.l2_loss(vertex_normals[self.mesh_pred.seam[:, 0], :],
                                     vertex_normals[self.mesh_pred.seam[:, 1], :])

                # displacement map smoothness
                loss += self.tv_loss(disp_opt) * self.weights["smooth"]

                if (k % self.scheduler_interval) == 0:
                    scheduler.step()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        self.mesh_pred.detach()