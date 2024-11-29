# import collections
import json
import os
import torch
import numpy as np
import random
import cv2
import nvdiffrast.torch as dr
from tqdm import tqdm
from torch import nn
from numpy.core.numeric import isscalar
from diff_renderer.normal_nds.nds.core import Camera
from diff_renderer.normal_nds.nds.losses import laplacian_loss, laplacian_loss_canonical


# import smplx
# import trimesh
# from torch.nn.functional import smooth_l1_loss
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# from apps.hand_replace import smpl_mesh
# from lbs_handler.utils import deform_vertices
# from diff_renderer.normal_nds.nds.core.mesh_textured import TexturedMesh
# from human_animator.animater_utils import batch_rigid_transform


class Renderer:
    def __init__(self, params, res=1024, near=1, far=1000, orthographic=False, device='cuda'):

        self.max_mip_level = params['max_mip_level']
        self.angle_interval = params['angles']

        self.res = res

        # self.glctx = dr.RasterizeGLContext()
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
            mins.append(samples_projected[..., 2].min())
            maxs.append(samples_projected[..., 2].max())

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
        return torch.tensor([[2.0 * fx / width, 0, 1.0 - 2.0 * cx / width, 0],
                             [0, 2.0 * fy / height, 1.0 - 2.0 * cy / height, 0],
                             [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                             [0, 0, -1, 0.0]], device=device)

    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000, orthographic=False):
        if orthographic:
            projection_matrix = torch.eye(4, device=camera.device)
            projection_matrix[:3, :3] = camera.K
            gl_transform = torch.tensor([[1., 0, 0, 0],
                                         [0, -1., 0, 0],
                                         [0, 0, -1., 0],
                                         [0, 0, 0, 1.]], device=camera.device)
        else:
            projection_matrix = Renderer.projection(fx=camera.K[0, 0],
                                                    fy=camera.K[1, 1],
                                                    cx=camera.K[0, 2],
                                                    cy=camera.K[1, 2],
                                                    n=n,
                                                    f=f,
                                                    width=resolution[1],
                                                    height=resolution[0],
                                                    device=camera.device)
            gl_transform = torch.tensor([[1., 0, 0, 0],
                                         [0, -1., 0, 0],
                                         [0, 0, 1., 0],
                                         [0, 0, 0, 1.]], device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    def get_gl_camera(self, camera, resolution):
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        return P

    def render(self, glctx, mesh, camera, render_options,
               resolution=1024,
               verts_init=None,
               enable_mip=True):

        render_out = {}
        if isscalar(resolution):
            resolution = [resolution, resolution]

        def transform_pos(mtx, pos):
            t_mtx = torch.from_numpy(mtx) if isinstance(mtx, np.ndarray) else mtx
            posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
            return torch.matmul(posw, t_mtx.t())[None, ...]

        pos, pos_idx = mesh.vertices, mesh.indices.int()
        if render_options["color"] is True or render_options["offset"] is True:
            uv, tex, disp = mesh.uv_vts, mesh.tex, mesh.disp

        mtx = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        pos_clip = transform_pos(mtx, pos)
        rast, rast_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution)
        # with dr.DepthPeeler(glctx, pos_clip, pos_idx, resolution) as peeler:
        #     rast, rast_db = peeler.rasterize_next_layer()

        # if render_options["visibility"]:
        #     with torch.no_grad():
        #         # Do not support batch operation yet
        #         face_ids = rast[..., -1].long()
        #         masked_face_idxs_list = face_ids[face_ids != 0] - 1  # num_masked_face Tensor
        #         masked_verts_idxs = torch.unique(mesh.indices[masked_face_idxs_list].long())
        #         vis_mask = torch.zeros(size=(pos.shape[0],), device=self.device).bool()  # visibility mask
        #         vis_mask[masked_verts_idxs] = 1
        #         render_out["vis_mask"] = vis_mask.bool().to(self.device)

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
            mask = dr.antialias(mask, rast, pos_clip, pos_idx)[0]  # if with_antialiasing else mask[0]
            render_out["mask"] = mask

        if render_options["normal"]:
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, pos_idx)
            render_out["normal"] = dr.antialias(normal, rast, pos_clip, pos_idx)[
                0]  # if with_antialiasing else normal[0]

        if render_options["depth"]:
            position, _ = dr.interpolate(pos[None, ...], rast, pos_idx)
            render_out["depth"] = dr.antialias(position, rast, pos_clip, pos_idx)[
                0]  # if with_antialiasing else position[0]
            # gbuffer["depth"] = view.project(gbuffer["position"], depth_as_distance=True)[..., 2:3]

        # for future use
        # if render_options["offset"]:
        #     texc, _ = dr.interpolate(uv[None, ...], rast, pos_idx)
        #     render_out["disp_uv"] = dr.texture(disp[None, ...], texc, filter_mode='linear')[0]
        #
        #     position, _ = dr.interpolate(pos[None, ...] - verts_init[None, ...], rast, pos_idx)
        #     render_out["disp_cv"] = position[0]
        return render_out  # texture map only

    def get_vert_visibility(self, glctx, mesh, camera, resolution=1024):
        if isscalar(resolution):
            resolution = [resolution, resolution]

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


class DiffOptimizer(nn.Module):
    def __init__(self,
                 params,
                 # init_uv=None,
                 device='cuda:0'):
        super(DiffOptimizer, self).__init__()

        self.cam_params = params['CAM']['DEFAULT']
        self.render_res = 1024
        self.use_opengl = True
        self.device = device

        path2label = os.path.join(params['SMPL']['smpl_root'], params['SMPL']['segmentation'])
        with open(path2label, 'r') as f:
            self.v_label = json.load(f)
        path2mapper = os.path.join(params['SMPL']['smpl_root'], params['SMPL']['uv_mapper'])
        with open(path2mapper, 'r') as f:
            self.uv_mapper = json.load(f)
        self.render = Renderer(params['RENDER'], device=device)

    def set_cams_from_angles(self, yaw_interval=30, pitch_interval=60, device='cuda:0'):
        cameras = []
        for j in range(-60, 60, pitch_interval):
            for k in range(0, 360, yaw_interval):
                camera = Camera.perspective_camera_with_angle(yaw=k, pitch=j,
                                                              cam_params=self.cam_params, device=device)
                cameras.append(camera)
        return cameras

    def set_cams_from_random_angles(self, cam_nums=1):
        cameras = []
        for _ in range(cam_nums):
            angle = random.randint(0, 359)
            camera = Camera.perspective_camera_with_angle(view_angle=angle, cam_params=self.cam_params)
            cameras.append(camera)
        return cameras

    def uv_sampling(self, tex, uv):
        '''

        :param displacement: [B, C, H, W] image features
        :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
        :return: [B, C, N] image features at the uv coordinates
        '''
        # uv is in [0, 1] so we need to convert to be in [-1, 1]
        uv = uv.unsqueeze(0).unsqueeze(2) * 2 - 1
        tex = tex.permute(2, 0, 1).unsqueeze(0)
        samples = torch.nn.functional.grid_sample(tex, uv, align_corners=True)  # [B, C, N, 1]
        return samples.squeeze().transpose(1, 0).contiguous()

    def pipeline(self, mesh_smpl, mesh_gt,
                 forward_skinning=False,
                 smpl_params=None,
                 update_smpl=False,
                 verbose=False):

        # coarse unwrapping (large displacements)
        weights = {'color': 1.0, 'depth': 1.0, 'normal': 0.1, 'seam': 100.0, 'smooth': 0.0001,
                   'laplacian': 10.0, 'symmetry': 30.0, 'penetration': 0.01, 'eye': 0.001, 'hands': 10.0}
        mesh_smpl = self(mesh_smpl, mesh_gt,
                         forward_skinning=forward_skinning,
                         weights=weights,
                         smpl_params=smpl_params,
                         update_smpl=update_smpl,
                         max_iter=10, lr=1e-5, decay=0.95, verbose=verbose)
        mesh_smpl = self(mesh_smpl, mesh_gt,
                         forward_skinning=forward_skinning,
                         weights=weights,
                         smpl_params=smpl_params,
                         update_smpl=update_smpl,
                         max_iter=1000, lr=1e-2, decay=0.95, verbose=verbose)

        # subdivision requires the update of the displacement map.
        mesh_smpl.subdivide()
        mesh_smpl = self.update_displacement_map(mesh_smpl, max_iter=500, verbose=verbose)

        # fine-unwrapping (details)
        weights = {'color': 1.0, 'depth': 1.0, 'normal': 0.1, 'seam': 10.0, 'smooth': 0.0001,
                   'laplacian': 5.0, 'symmetry': 0.01, 'penetration': 0.01, 'eye': 0.001, 'hands': 10.0}
        mesh_smpl = self(mesh_smpl, mesh_gt,
                         forward_skinning=forward_skinning,
                         weights=weights,
                         smpl_params=smpl_params,
                         update_smpl=update_smpl,
                         max_iter=1000, lr=1e-3, decay=0.95, verbose=verbose)
        return mesh_smpl

    def update_displacement_map(self, mesh_smpl, lr=1e-2, max_iter=1000, verbose=True):
        disp_opt = mesh_smpl.disp
        disp_opt.requires_grad_()

        scheduler_interval = 100
        l2_loss = nn.MSELoss()

        opt_params = [disp_opt]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        if hasattr(mesh_smpl, "disp_vector"):
            target_disp = mesh_smpl.disp_vector.detach().clone()
        else:
            disp = self.uv_sampling(mesh_smpl.disp, mesh_smpl.uv_vts)
            target_disp = disp.detach().clone()

        target_disp[mesh_smpl.idx2body['leftHand'], :] = 0.0
        target_disp[mesh_smpl.idx2body['rightHand'], :] = 0.0
        log_interval = 10

        for k in tqdm(range(max_iter), 'opt. disp:'):
            disp = self.uv_sampling(disp_opt, mesh_smpl.uv_vts)
            loss = l2_loss(disp, target_disp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k % scheduler_interval) == 0:
                scheduler.step()

            if (k % log_interval) == 0 and verbose:
                texture_map = disp_opt.squeeze(0).detach().cpu().numpy()
                cv2.imshow('disp_map', texture_map)
                cv2.waitKey(10)

        mesh_smpl.disp = disp_opt.detach().clone()
        return mesh_smpl.detach()

    @staticmethod
    def tv_loss(img):
        xv = img[1:, :, :] - img[:-1, :, :]
        yv = img[:, 1:, :] - img[:, :-1, :]
        loss = torch.mean(abs(xv)) + torch.mean(abs(yv))
        return loss

    def perceptual_loss(self, src, target):
        # self.lpip_loss = self.lpip_loss.to(src.device)
        loss = self.lpip_loss(src * 2.0 - 1.0, target * 2.0 - 1.0)
        return loss

    # update geometry, color, or both
    def forward(self, mesh_smpl, mesh_scan,
                smpl_params=None,
                update_smpl=False,
                forward_skinning=False,
                weights=None,
                max_iter=10000,
                lr=1e-2,
                decay=0.95,
                verbose=True):

        texture_opt = mesh_smpl.tex
        texture_opt.requires_grad_()
        texture_opt.retain_grad()

        texture_eye_mask = mesh_smpl.tex_eye_mask.clone()
        texture_init = mesh_smpl.tex_eye.clone() * texture_eye_mask

        disp_opt = mesh_smpl.disp
        disp_opt.requires_grad_()
        disp_opt.retain_grad()

        if forward_skinning:
            vertex_initial = mesh_smpl.v_posed.detach().clone().requires_grad_(False)
            mesh_smpl.compute_normals(canonical=True)
            vertex_normals = mesh_smpl.vertex_normals.detach().clone().requires_grad_(False)  # initial vertex normals
        else:
            vertex_initial = mesh_smpl.vertices.clone()
            vertex_normals = mesh_smpl.vertex_normals.clone()

        opt_params = [texture_opt, disp_opt]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay ** epoch)

        glctx = dr.RasterizeGLContext() if self.use_opengl else dr.RasterizeCudaContext()
        cameras = self.set_cams_from_angles(yaw_interval=10, pitch_interval=20)  # degree

        render_options = {'color': True,
                          'depth': True,
                          'normal': True,
                          'mask': False,
                          'visibility': False,
                          'offset': False}
        render_tgt = []
        for camera in cameras:
            render_tgt.append(self.render.render(glctx, mesh_scan, camera, render_options))

        view_num = len(cameras)
        log_interval = 10
        scheduler_interval = 100
        l2_loss = nn.MSELoss()

        render_options_src = {'color': True,
                              'depth': True,
                              'normal': True,
                              'mask': False,
                              'visibility': False,
                              'offset': False}

        for k in tqdm(range(max_iter), 'opt. mesh:'):
            v = random.randint(0, view_num - 1)
            disp = self.uv_sampling(disp_opt, mesh_smpl.uv_vts) / 100
            disp[mesh_smpl.idx2body['leftHand']] = 0
            disp[mesh_smpl.idx2body['rightHand']] = 0

            if forward_skinning:
                canonical_vertices = vertex_initial[None, ...] + disp[None, ...]
                mesh_smpl.vertices = mesh_smpl.forward_skinning(v_posed=canonical_vertices,
                                                                smpl_params=smpl_params,
                                                                update_smpl=update_smpl)
            else:
                canonical_vertices = None
                mesh_smpl.vertices = vertex_initial + disp

            render_src = self.render.render(glctx, mesh_smpl, cameras[v],
                                            render_options_src,
                                            verts_init=vertex_initial)

            loss = l2_loss(render_src["color"], render_tgt[v]["color"] * 0.9) * weights["color"]
            if weights["depth"]:
                loss += l2_loss(render_src["depth"], render_tgt[v]["depth"]) * weights["depth"]
            if weights["normal"]:
                loss += l2_loss(render_src["normal"], render_tgt[v]["normal"]) * weights["normal"]

            # anti-penetration loss (exclude this term in facial and hands regions)
            p = 1 - torch.sum(nn.functional.normalize(disp, dim=1) * vertex_normals, dim=1)
            for key in mesh_smpl.idx2body.keys():
                loss += torch.mean(p[mesh_smpl.idx2body[key]]) * weights['penetration']

            # (eyeballs move together)
            loss += l2_loss(disp[mesh_smpl.idx2body['eyeballs'], :],
                            torch.mean(disp[mesh_smpl.idx2body['eyeballs'], :], dim=0)) * 1.0

            # does not move hands
            loss += torch.mean(disp[mesh_smpl.idx2body['leftHand'], :] ** 2) * 1.0
            loss += torch.mean(disp[mesh_smpl.idx2body['rightHand'], :] ** 2) * 1.0

            # eye preserving loss (keep prior texture)
            if weights["eye"]:
                loss += l2_loss(texture_opt * texture_eye_mask, texture_init) * weights["eye"]

            # mesh smoothness exclude hands to oversimplification.
            if weights["laplacian"]:
                mask = mesh_smpl.idx2body['noHands'] if 'noHand' in mesh_smpl.idx2body else None
                loss += laplacian_loss(mesh_smpl, mask=mask) * weights["laplacian"]

            # texture map smoothness
            if weights["smooth"]:
                loss += self.tv_loss(texture_opt) * weights["smooth"]

            # make it smooth
            hand_texture = texture_opt[mesh_smpl.tex_hand_mask == 1, :]
            face_texture = torch.mean(texture_opt[mesh_smpl.tex_face_mask == 1, :], dim=0).detach()
            loss += l2_loss(hand_texture, face_texture[None, :]) * 0.01

            # texture border consistency (color / distance / normal consistency)
            if weights["seam"]:
                loss += l2_loss(mesh_smpl.vertices[mesh_smpl.seam[:, 0], :],
                                mesh_smpl.vertices[mesh_smpl.seam[:, 1], :]) * weights["seam"]

                uv_ones = torch.ones_like(mesh_smpl.uv_vts)
                uv_ones[:, 1] = 0
                vertex_colors = self.uv_sampling(texture_opt, uv_ones - mesh_smpl.uv_vts)
                loss += l2_loss(vertex_colors[mesh_smpl.seam[:, 0], :],
                                vertex_colors[mesh_smpl.seam[:, 1], :]) * weights["seam"]

            if weights["symmetry"] and forward_skinning and canonical_vertices is not None:
                # strong symmetry between canonical and flipped meshes.
                disp_flipped = disp.detach().clone()
                disp_flipped[:, 1] = -1.0 * disp_flipped[:, 1] * weights["symmetry"]
                loss += l2_loss(disp, disp_flipped)  # left-right symmetry

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (k % scheduler_interval) == 0:
                scheduler.step()

            if (k % log_interval) == 0 and verbose:
                texture_map = texture_opt.squeeze(0).detach().cpu().numpy()
                cv2.imshow('texture_map', texture_map)
                disp_map = disp_opt.squeeze(0).detach().cpu().numpy()
                cv2.imshow('disp_map', disp_map)
                cv2.waitKey(10)

        mesh_smpl.tex = texture_opt
        mesh_smpl.disp = disp_opt
        # mesh_smpl.v_posed = vertex_initial
        mesh_smpl.detach()
        return mesh_smpl