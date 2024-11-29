import math
import os.path

import cv2
import trimesh
import torch.utils.data
import math
import time
from torch.autograd import Function
from human_renderer.renderer.camera import Camera
from human_renderer.renderer.gl.normal_render import NormalRender
from basicsr.archs.rrdbnet_arch import RRDBNet
from depth_predictor.utils.loader_utils import *
from depth_predictor.utils.core.im_utils import get_plane_params
from torchmcubes import grid_interp
from reconstructor import models
import reconstructor.recon_utils as recon_utils
from lbs_handler.model import LBSModel
import smplx
from pysdf import SDF
from torchmcubes import marching_cubes, grid_interp
from typing import Tuple
import open3d as o3d
import kaolin as kal
from smpl_optimizer.smplify import SmplOptimizer
import copy
from sklearn.neighbors import KDTree
from trimesh import transformations

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3

os.environ["PYOPENGL_PLATFORM"] = "egl"

class HumanRecon(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='',
                 ckpt_path='',
                 color_ckpt_path='',
                 model_name='',
                 model_C_name='',
                 esr_path='',
                 lbs_ckpt='',
                 model_config='',
                 cam_params=None,
                 params=None,
                 eval_metrics=None,
                 device=torch.device('cuda:0')):
        super(HumanRecon, self).__init__()
        self._normal_render = NormalRender(width=512, height=512)
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.model_config = model_config
        self.res = cam_params['recon_res']
        self.voxel_size = cam_params['voxel_size']

        self.z_min = cam_params['real_dist'] - 64
        self.z_max = cam_params['real_dist'] + 64
        self.px = cam_params['px']
        self.py = cam_params['py']
        self.fx = cam_params['fx']
        self.fy = cam_params['fy']
        self.real_dist = cam_params['real_dist']
        if cam_params['cmax'][1] == cam_params['cmin'][1]:
            self.camera_height = cam_params['cmax'][1]
        else:
            self.camera_height = 0
        self.eval_metrics = eval_metrics
        self.device = torch.device(device)

        self.smpl_optimizer = SmplOptimizer(params,
                                            cam_params=cam_params,
                                            num_models=1,
                                            device=self.device)
        self.smpl_optimizer.init_semantic_labels(params.path2semantic)

        # load pre-trained model
        self.model = getattr(models, model_name)(split_last=True)
        self.model_C = getattr(models, model_C_name)(split_last=True)

        self.model.to(self.device)
        self.model_C.to(self.device)

        self.load_checkpoint([ckpt_path],
                             self.model,
                             is_evaluate=True,
                             device=device)

        self.load_checkpoint([color_ckpt_path],
                             self.model_C,
                             is_evaluate=True,
                             device=device)
        # Real-esrGAN
        esr_model_path = esr_path
        self.esr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        loadnet = torch.load(esr_model_path, map_location=device)
        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.esr_model.load_state_dict(loadnet[keyname], strict=True)
        self.esr_model.eval()
        self.esr_model = self.esr_model.to(device)

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        self.RGB_MG = [10.0, 10.0, 10.0]
        self.DEPTH_SCALE = 128.0
        self.DEPTH_MAX = 32767
        self.DEPTH_EPS = 0.5
        self.scale_factor = 1.0
        self.offset = 0
        x = np.reshape((np.linspace(0, self.res, self.res) - int(self.px)) / self.fx,
                       [1, 1, 1, -1])
        y = np.reshape((np.linspace(0, self.res, self.res) - int(self.res - self.py)) / self.fy,
                       [1, 1, -1, 1])

        x = np.tile(x, [1, 1, self.res, 1])
        y = np.tile(y, [1, 1, 1, self.res])
        self.xy = torch.Tensor(np.concatenate((y, x), axis=1)).cuda()
        self.coord = self.gen_volume_coordinate(xy=self.xy[0],
                                                z_min=self.z_min,
                                                z_max=self.z_max,
                                                voxel_size=self.voxel_size)
        # self.cosine_loss = nn.CosineSimilarity(dim=2)
        self.lbs_model = LBSModel().cuda()
        self.lbs_model.load_state_dict(torch.load(os.path.join(lbs_ckpt, 'best.tar'))['state_dict'])
        self.lbs_model.eval()

    @staticmethod
    def cosine_loss(x, y):
        return torch.mean(1.0 - torch.sum((x * y), dim=1))

    @staticmethod
    def load_checkpoint(model_paths, model,
                        is_evaluate=False, device=None):

        for model_path in model_paths:
            items = glob.glob(os.path.join(model_path, '*.pth.tar'))
            items.sort()

            if len(items) > 0:
                if is_evaluate is True:
                    model_path = os.path.join(model_path, 'model_best.pth.tar')
                else:
                    if len(items) == 1:
                        model_path = items[0]
                    else:
                        model_path = items[len(items) - 1]

                print(("=> loading checkpoint '{}'".format(model_path)))
                checkpoint = torch.load(model_path, map_location=device)
                start_epoch = checkpoint['epoch'] #+ 1

                if hasattr(model, 'module'):
                    model_state_dict = checkpoint['model_state_dict']
                else:
                    model_state_dict = collections.OrderedDict(
                        {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

                model.load_state_dict(model_state_dict, strict=False)

                print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
                return model

        print(("=> no checkpoint found at '{}'".format(model_path)))
        return model

    @staticmethod
    def gen_volume_coordinate(xy, z_min=120, z_max=320, voxel_size=512):
        grid = torch.ones((3, xy.shape[1], xy.shape[2], voxel_size))
        z_range = z_max - z_min
        slope = z_range / voxel_size
        ones = torch.ones_like(xy[0:1, :, :])
        for k in range(voxel_size):
            z = z_min + slope * k
            grid[:, :, :, k] = torch.cat((xy * z, ones * z), dim=0)
        return grid

    def cube_sdf(self, x_nx3):
        sdf_values = 0.5 - torch.abs(x_nx3)
        sdf_values = torch.clamp(sdf_values, min=0.0)
        sdf_values = sdf_values[:, 0] * sdf_values[:, 1] * sdf_values[:, 2]
        sdf_values = -1.0 * sdf_values

        return sdf_values

    def cube_sdf_gradient(self, x_nx3):
        gradients = []
        for i in range(x_nx3.shape[0]):
            x, y, z = x_nx3[i]
            grad_x, grad_y, grad_z = 0, 0, 0

            max_val = max(abs(x) - 0.5, abs(y) - 0.5, abs(z) - 0.5)

            if max_val == abs(x) - 0.5:
                grad_x = 1.0 if x > 0 else -1.0
            if max_val == abs(y) - 0.5:
                grad_y = 1.0 if y > 0 else -1.0
            if max_val == abs(z) - 0.5:
                grad_z = 1.0 if z > 0 else -1.0

            gradients.append(torch.tensor([grad_x, grad_y, grad_z]))

        return torch.stack(gradients).to(x_nx3.device)

    @staticmethod
    def get_mesh(volume, grid_coord, scale_factor=1.0):
        # mesh generation.
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        vertices, faces = marching_cubes(volume, 0.0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
        return new_mesh

    def _get_grid_coord_(self, v_min, v_max, res):
        x_ind, y_ind, z_ind = torch.meshgrid(torch.linspace(v_min[0], v_max[0], res[0]),
                                             torch.linspace(v_min[1], v_max[1], res[1]),
                                             torch.linspace(v_min[2], v_max[2], res[2]), indexing='ij')
        grid = torch.stack((x_ind, y_ind, z_ind), dim=0)
        grid = grid.float()
        pt = np.concatenate((np.asarray(x_ind).reshape(-1, 1),
                             np.asarray(y_ind).reshape(-1, 1),
                             np.asarray(z_ind).reshape(-1, 1)), axis=1)
        pt = pt.astype(float)
        return pt, grid.permute(0, 3, 2, 1).contiguous()

    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        # print("mse : ", mse)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    def euler_to_rot_mat(self, r_x, r_y, r_z):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r_x), -math.sin(r_x)],
                        [0, math.sin(r_x), math.cos(r_x)]
                        ])

        R_y = np.array([[math.cos(r_y), 0, math.sin(r_y)],
                        [0, 1, 0],
                        [-math.sin(r_y), 0, math.cos(r_y)]
                        ])

        R_z = np.array([[math.cos(r_z), -math.sin(r_z), 0],
                        [math.sin(r_z), math.cos(r_z), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R
    def make_rotation_matrix(self, rx, ry, rz):
        sinX = np.sin(rx)
        sinY = np.sin(ry)
        sinZ = np.sin(rz)

        cosX = np.cos(rx)
        cosY = np.cos(ry)
        cosZ = np.cos(rz)

        Rx = np.zeros((3, 3))
        Rx[0, 0] = 1.0
        Rx[1, 1] = cosX
        Rx[1, 2] = -sinX
        Rx[2, 1] = sinX
        Rx[2, 2] = cosX

        Ry = np.zeros((3, 3))
        Ry[0, 0] = cosY
        Ry[0, 2] = sinY
        Ry[1, 1] = 1.0
        Ry[2, 0] = -sinY
        Ry[2, 2] = cosY

        Rz = np.zeros((3, 3))
        Rz[0, 0] = cosZ
        Rz[0, 1] = -sinZ
        Rz[1, 0] = sinZ
        Rz[1, 1] = cosZ
        Rz[2, 2] = 1.0

        R = np.matmul(np.matmul(Rz, Ry), Rx)
        return R

    def align_mesh(self, mesh1, mesh2):
        # if isinstance(vertices, np.ndarray):  # differentiable if the input is tensor
        #     vertices = torch.Tensor(vertices).float().to(self.device)
        vts1 = mesh1.vertices
        center1 = mesh1.bounding_box.centroid
        scale1 = 2.0 / np.max(mesh1.bounding_box.extents)

        # vts2 = mesh2.vertices
        center2 = mesh2.bounding_box.centroid
        scale2 = 2.0 / np.max(mesh2.bounding_box.extents)

        new_vertices = (vts1 - center1) * scale1 / scale2 + center2
        mesh1.vertices = new_vertices
        return mesh1

    def eval_mesh(self, gt, pred, num_samples=10000):
        pred_surf_pts, _ = trimesh.sample.sample_surface(pred, num_samples)
        gt_surf_pts, _ = trimesh.sample.sample_surface(gt, num_samples)

        _, pred_gt_dist, _ = trimesh.proximity.closest_point(gt, pred_surf_pts)
        _, gt_pred_dist, _ = trimesh.proximity.closest_point(pred, gt_surf_pts)

        pred_gt_dist[np.isnan(pred_gt_dist)] = 0
        gt_pred_dist[np.isnan(gt_pred_dist)] = 0

        false_ratio_pred_gt = len(pred_gt_dist[pred_gt_dist > 3.0]) / num_samples
        false_ratio_gt_pred = len(gt_pred_dist[gt_pred_dist > 3.0]) / num_samples
        false_ratio = (false_ratio_pred_gt + false_ratio_gt_pred) / 2 * 100

        pred_gt_dist = pred_gt_dist.mean()
        gt_pred_dist = gt_pred_dist.mean()

        p2s = pred_gt_dist
        p2s_outlier = false_ratio_pred_gt
        chamfer = (pred_gt_dist + gt_pred_dist) / 2
        chamfer_outlier = false_ratio

        return p2s, chamfer, p2s_outlier, chamfer_outlier

    def volume_filter(self, volume, k_size=3, iter=3):
        # k_size must be an odd number
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        filters = torch.ones(1, 1, k_size, k_size, k_size) / (k_size*k_size*k_size)  # average filter
        volume = volume.unsqueeze(0)
        for _ in range(iter):
            volume = F.conv3d(volume, filters, padding=k_size//2)
        return volume.squeeze().detach().cpu().numpy()
    def init_variables(self, datum, device=None):
        input_color, input_mask, input_depth = datum['input']

        if 'label' in datum:
            target_color, target_depth = datum['label']
        else:
            target_color, target_depth = None, None

        if device is not None:
            if input_color is not None:
                input_color = input_color.unsqueeze(0).to(device)
            if input_depth is not None:
                input_depth = input_depth.unsqueeze(0).to(device)
            if input_mask is not None:
                input_mask = input_mask.unsqueeze(0).to(device)
            if target_color is not None:
                target_color = target_color.to(device)
            if target_depth is not None:
                target_depth = target_depth.to(device)

        if input_color is not None:
            input_color = torch.autograd.Variable(input_color)
        if input_depth is not None:
            input_depth = torch.autograd.Variable(input_depth)
        if input_mask is not None:
            input_mask = torch.autograd.Variable(input_mask)
        if target_depth is not None:
            target_depth = torch.autograd.Variable(target_depth)
        if target_color is not None:
            target_color = torch.autograd.Variable(target_color)

        input_var = (input_color, input_mask, input_depth)
        if target_color is not None and target_depth is not None:
            target_var = (target_color, target_depth)
        else:
            target_var = None

        return input_var, target_var

    def postprocess_mesh(self, mesh, num_faces=None):
        """Post processing mesh by removing small isolated pieces.

        Args:
            mesh (trimesh.Trimesh): input mesh to be processed
            num_faces (int, optional): min face num threshold. Defaults to 4096.
        """
        total_num_faces = len(mesh.faces)
        if num_faces is None:
            num_faces = total_num_faces // 100
        cc = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=3)
        mask = np.zeros(total_num_faces, dtype=bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh

    def canonicalization(self, mesh, color_mesh, param, pose=None, resource_path=None):
        if len(param['right_hand_pose'][0]) != 45:
            smplx_model = smplx.create(resource_path,
                                       model_type='smplx',
                                       gender='male',
                                       num_betas=10, ext='npz',
                                       use_face_contour=True,
                                       flat_hand_mean=False,
                                       use_pca=True,
                                       num_pca_comps=len(param['right_hand_pose'][0])).cuda()
        else:
            smplx_model = smplx.create(resource_path,
                                       model_type='smplx',
                                       gender='male',
                                       num_betas=10, ext='npz',
                                       use_face_contour=True,
                                       flat_hand_mean=False,
                                       use_pca=False).cuda()

        smpl_output, smpl_model = recon_utils.set_smpl_model(smplx_model, param, device='cuda')
        smpl_mesh = trimesh.Trimesh(
            smpl_output.vertices.detach().cpu().numpy().squeeze(),
            smplx_model.faces, process=False)

        v_pose_smpl = trimesh.Trimesh(smpl_model.v_template.cpu(),
                                      smpl_model.faces)
        canon_smpl_vertices = smpl_model.v_template + smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
        canon_smpl_mesh = trimesh.Trimesh(canon_smpl_vertices.detach().squeeze(0).detach().cpu().numpy(),
                                          smpl_model.faces, process=False)
        smpl_canon_vertices = torch.FloatTensor(canon_smpl_mesh.vertices).cuda()[None, ...]

        full_pose = torch.zeros((55, 3)).cuda()
        full_pose[0:1, :] = torch.FloatTensor(param['global_orient']).reshape(1, 3).cuda()
        full_pose[1:22, :] = torch.FloatTensor(param['body_pose']).reshape(21, 3).cuda()
        full_pose = full_pose[None, ...]

        if pose is not None:
            full_pose = pose
            full_pose[0:1, :] = torch.FloatTensor(param['global_orient']).reshape(1, 3).cuda()

        # real space to smpl space
        # verts = mesh.vertices / 100.0
        centroid_real = [0.0, 0.0, 0.0]
        scale_real = [0.011111111111111112]
        centroid_smpl = v_pose_smpl.bounding_box.centroid
        scale_smpl = 2.0 / np.max(v_pose_smpl.bounding_box.extents)

        # real space to smpl space
        verts = (mesh.vertices - centroid_real) * scale_real / scale_smpl + centroid_smpl

        lbs = mesh.visual.vertex_colors[:, :3]
        with torch.no_grad():
            full_lbs = self.lbs_model.decoder(torch.Tensor(lbs).cuda()/255)

        # canonicalization
        canon_verts = recon_utils.deform_vertices(torch.Tensor(verts).cuda().unsqueeze(0) / torch.Tensor(param['scale']).cuda() - torch.Tensor(param['transl']).cuda(),
                                             smpl_model, full_lbs,
                                             full_pose,
                                             inverse=True,
                                             return_vshape=False,
                                             device='cuda')

        canon_mesh = trimesh.Trimesh(canon_verts.squeeze().detach().cpu().numpy(), mesh.faces, process=False)
        return canon_mesh, canon_smpl_mesh, smpl_model.lbs_weights

    def get_face_length(self, vts, faces):
        # check faces.
        areas = []
        for k in range(faces.shape[0]):
            x, y, z = faces[k]
            a = sum((vts[x, :] - vts[y, :]) ** 2) ** 2
            b = sum((vts[y, :] - vts[z, :]) ** 2) ** 2
            c = sum((vts[x, :] - vts[z, :]) ** 2) ** 2
            s = a + b + c
            if s < 0.000001:
                areas.append(True)
            else:
                areas.append(False)
        return areas

    def make_rotate(self, rx, ry, rz):
        sinX = np.sin(rx)
        sinY = np.sin(ry)
        sinZ = np.sin(rz)

        cosX = np.cos(rx)
        cosY = np.cos(ry)
        cosZ = np.cos(rz)

        Rx = np.zeros((3, 3))
        Rx[0, 0] = 1.0
        Rx[1, 1] = cosX
        Rx[1, 2] = -sinX
        Rx[2, 1] = sinX
        Rx[2, 2] = cosX

        Ry = np.zeros((3, 3))
        Ry[0, 0] = cosY
        Ry[0, 2] = sinY
        Ry[1, 1] = 1.0
        Ry[2, 0] = -sinY
        Ry[2, 2] = cosY

        Rz = np.zeros((3, 3))
        Rz[0, 0] = cosZ
        Rz[0, 1] = -sinZ
        Rz[1, 0] = sinZ
        Rz[1, 1] = cosZ
        Rz[2, 2] = 1.0

        R = np.matmul(np.matmul(Rz, Ry), Rx)
        return R

    def point_mesh_distance(self, meshes, pcls, weighted=True):

        if len(meshes) != len(pcls):
            raise ValueError("meshes and pointclouds must be equal sized batches")

        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

        # point to face distance: shape (P,)
        point_to_face, idxs = _PointFaceDistance.apply(
            points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
        )

        if weighted:
            # weight each example by the inverse of number of points in the example
            point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
            num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
            weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
            weights_p = 1.0 / weights_p.float()
            point_to_face = torch.sqrt(point_to_face) * weights_p

        return point_to_face, idxs

    def _rand_barycentric_coords(self,
            size1, size2, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to generate random barycentric coordinates which are uniformly
        distributed over a triangle.

        Args:
            size1, size2: The number of coordinates generated will be size1*size2.
                          Output tensors will each be of shape (size1, size2).
            dtype: Datatype to generate.
            device: A torch.device object on which the outputs will be allocated.

        Returns:
            w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
                coordinates
        """
        uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
        u, v = uv[0], uv[1]
        u_sqrt = u.sqrt()
        w0 = 1.0 - u_sqrt
        w1 = u_sqrt * (1.0 - v)
        w2 = u_sqrt * v
        w = torch.cat([w0[..., None], w1[..., None], w2[..., None]], dim=2)

        return w

    def sample_points_from_meshes(self, meshes, num_samples: int = 10000):
        """
        Convert a batch of meshes to a batch of pointclouds by uniformly sampling
        points on the surface of the mesh with probability proportional to the
        face area.

        Args:
            meshes: A Meshes object with a batch of N meshes.
            num_samples: Integer giving the number of point samples per mesh.
            return_normals: If True, return normals for the sampled points.
            return_textures: If True, return textures for the sampled points.

        Returns:
            3-element tuple containing

            - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
              coordinates of sampled points for each mesh in the batch. For empty
              meshes the corresponding row in the samples array will be filled with 0.
            - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
              to each sampled point. Only returned if return_normals is True.
              For empty meshes the corresponding row in the normals array will
              be filled with 0.
            - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
              texture vector to each sampled point. Only returned if return_textures is True.
              For empty meshes the corresponding row in the textures array will
              be filled with 0.

            Note that in a future releases, we will replace the 3-element tuple output
            with a `Pointclouds` datastructure, as follows

            .. src-block:: python

                Pointclouds(samples, normals=normals, features=textures)
        """
        if meshes.isempty():
            raise ValueError("Meshes are empty.")

        verts = meshes.verts_packed()
        if not torch.isfinite(verts).all():
            raise ValueError("Meshes contain nan or inf.")

        faces = meshes.faces_packed()
        mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
        num_meshes = len(meshes)
        num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

        # Initialize samples tensor with fill value 0 for empty meshes.
        samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

        # Only compute samples for non empty meshes
        with torch.no_grad():
            areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
            max_faces = meshes.num_faces_per_mesh().max().item()
            areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)  # (N, F)

            # TODO (gkioxari) Confirm multinomial bug is not present with real data.
            samples_face_idxs = areas_padded.multinomial(
                num_samples, replacement=True
            )  # (N, num_samples)
            samples_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

        # Randomly generate barycentric coords.
        # w                 (N, num_samples, 3)
        # sample_face_idxs  (N, num_samples)
        # samples_verts     (N, num_samples, 3, 3)

        samples_bw = self._rand_barycentric_coords(num_valid_meshes, num_samples, verts.dtype, verts.device)
        sample_verts = verts[faces][samples_face_idxs]
        samples[meshes.valid] = (sample_verts * samples_bw[..., None]).sum(dim=-2)

        return samples, samples_face_idxs, samples_bw

    def calculate_chamfer_p2s(self, tgt_mesh, src_mesh, num_samples=1000):

        tgt_points = Pointclouds(self.sample_points_from_meshes(self.tgt_mesh, num_samples))
        src_points = Pointclouds(self.sample_points_from_meshes(self.src_mesh, num_samples))
        p2s_dist = self.point_mesh_distance(self.src_mesh, tgt_points) * 100.0
        chamfer_dist = (self.point_mesh_distance(self.tgt_mesh, src_points) * 100.0 + p2s_dist) * 0.5

        return chamfer_dist, p2s_dist

    def eval_normal(self, gt, pred, center, scale):
        import neural_renderer as nr

        gt_vertices = gt.vertices.copy()
        gt_verts = torch.Tensor(gt_vertices).cuda()
        gt_verts = gt_verts - torch.Tensor(center).cuda()
        gt_verts = gt_verts * scale
        gt_verts = gt_verts.unsqueeze(0)
        gt_faces = torch.tensor(gt.faces[None, :, :].copy()).float().cuda()
        gt_textr = torch.tensor(gt.visual.face_colors[None, :, -2:-5:-1].copy()).float().cuda() / 255.0
        gt_textr = gt_textr.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        pred_vertices = pred.vertices.copy()
        pred_verts = torch.Tensor(pred_vertices).cuda()
        pred_verts = pred_verts - torch.Tensor(center).cuda()
        pred_verts = pred_verts * scale
        pred_verts = pred_verts.unsqueeze(0)
        pred_faces = torch.tensor(pred.faces[None, :, :].copy()).float().cuda()
        pred_textr = torch.tensor(pred.visual.face_colors[None, :, -2:-5:-1].copy()).float().cuda() / 255.0
        pred_textr = pred_textr.unsqueeze(2).unsqueeze(2).unsqueeze(2)

        cam = Camera(width=512, height=512, projection='perspective')
        R, K, t, projection_matrix, model_view_matrix = cam.get_gl_matrix()
        K = torch.tensor(K[None:, :].copy()).float().cuda().unsqueeze(0)
        t = torch.tensor(t[None, :].copy()).float().cuda().unsqueeze(0)
        # t = torch.Tensor([0, 100, 300]).float().cuda().unsqueeze(0)
        # angles = [0, 90, -90, 180]
        angles = [0, 180]
        normal_error = 0

        gt_normals, pred_normals = [], []

        for a in angles:
            angle = self.make_rotate(0, math.radians(a), 0)
            Rot = np.matmul(R, angle)
            Rot = torch.tensor(Rot[None:, :].copy()).float().cuda().unsqueeze(0)

            renderer = nr.Renderer(image_size=cam.width, orig_size=cam.width,
                                   K=K, R=Rot, t=t,
                                   anti_aliasing=True,
                                   camera_direction=[0, 0, -1],
                                   camera_mode='projection',
                                   near=cam.near, far=cam.far)

            gt_images, gt_depths, gt_silhouettes = renderer(gt_verts, gt_faces, gt_textr)
            pred_images, pred_depths, pred_silhouettes = renderer(pred_verts, pred_faces, pred_textr)

            gt_depth = gt_depths.squeeze().detach().cpu().numpy()
            pred_depth = pred_depths.squeeze().detach().cpu().numpy()
            gt_silhouette = gt_silhouettes.squeeze().detach().cpu().numpy()
            pred_silhouette = pred_silhouettes.squeeze().detach().cpu().numpy()

            gt_normal = get_normal(torch.Tensor(gt_depth).unsqueeze(0))
            pred_normal = get_normal(torch.Tensor(pred_depth).unsqueeze(0))
            gt_normal[:, :, gt_silhouette == 0, :] = 0
            pred_normal[:, :, pred_silhouette == 0, :] = 0

            normal_error += self.cosine_loss(gt_normal, pred_normal)

            gt_normal_np = gt_normal.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            pred_normal_np = pred_normal.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            gt_normals.append(gt_normal_np * 255)
            pred_normals.append(pred_normal_np * 255)

        return float(normal_error / len(angles)), gt_normals, pred_normals
    def _render_normal(self, mesh, deg, data_flag=False, src_flag=False):
        vmin = mesh.vertices.min(0)
        vmax = mesh.vertices.max(0)
        up_axis = 1 #if (vmax - vmin).argmax() == 1 else 2

        if data_flag:
            center = (vmax + vmin) / 2.0
            mesh.vertices -= center
        else:
            center = np.median(np.asarray(mesh.vertices), 0)
            center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
            scale = 180 / (vmax[up_axis] - vmin[up_axis])
            # normalization
            mesh.vertices -= center
            mesh.vertices *= scale

        view_mat = np.identity(4)
        view_mat[:3, :3] *= 2 / 256
        rz = deg / 180. * np.pi
        model_mat = np.identity(4)
        model_mat[:3, :3] = self.euler_to_rot_mat(0, rz, 0)
        model_mat[1, 3] = self.offset
        view_mat[2, 2] *= -1
        if src_flag:
            # mesh.vertices[:, 2] *= (-1)
            # mesh.vertices[:, 0] *= (-1)
            # mesh.vertices[:, 0] *= (-1)
            # mesh.vertex_normals[:, 0:3] *= (-1)
            view_mat[2, 2] *= -1
        # elif src_flag and self.dir == 'back':
            # view_mat[2, 2] *= -1
        self._normal_render.set_matrices(view_mat, model_mat)
        self._normal_render.set_normal_mesh(self.scale_factor*mesh.vertices, mesh.faces, mesh.vertex_normals, mesh.faces)
        self._normal_render.draw()
        normal_img = self._normal_render.get_color()
        return normal_img

    def _get_reproj_normal_error(self, deg):
        tgt_normal = self._render_normal(self.tgt_mesh, deg, data_flag=False)
        src_normal = self._render_normal(self.src_mesh, deg, data_flag=False)

        error = ((src_normal[:, :, :3] - tgt_normal[:, :, :3]) ** 2).mean() * 3

        return error, src_normal, tgt_normal

    def get_reproj_normal_error(self, frontal=True, back=True, left=False, right=False, save_demo_img=None):
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")
        if self._normal_render is None:
            print("In order to use normal render, "
                  "you have to call init_gl() before initialing any evaluator objects.")
            return -1

        side_cnt = 0
        total_error = 0
        # demo_list = []
        src_list = []
        tgt_list = []
        if frontal:
            self.dir = 'frontal'
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(0)
            total_error += error
            src_list.append(src_normal)
            tgt_list.append(tgt_normal)
            # demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if back:
            self.dir = 'back'
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(180)
            total_error += error
            src_list.append(src_normal)
            tgt_list.append(tgt_normal)
            # demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if left:
            self.dir = 'left'
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(90)
            total_error += error
            src_list.append(src_normal)
            tgt_list.append(tgt_normal)
            # demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if right:
            self.dir = 'right'
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(270)
            total_error += error
            src_list.append(src_normal)
            tgt_list.append(tgt_normal)
            # demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if save_demo_img is not None:
            # res_array = np.concatenate(demo_list, axis=1)
            # res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            # res_img.save(save_demo_img)
            src_normal_f_img = Image.fromarray((src_list[0] * 255).astype(np.uint8))
            src_normal_b_img = Image.fromarray((src_list[1] * 255).astype(np.uint8))
            tgt_normal_f_img = Image.fromarray((tgt_list[0] * 255).astype(np.uint8))
            tgt_normal_b_img = Image.fromarray((tgt_list[1] * 255).astype(np.uint8))
            src_normal_f_img.save(save_demo_img.replace('.obj', '_pred_f.png'))
            src_normal_b_img.save(save_demo_img.replace('.obj', '_pred_b.png'))
            tgt_normal_f_img.save(save_demo_img.replace('.obj', '_gt_f.png'))
            tgt_normal_b_img.save(save_demo_img.replace('.obj', '_gt_b.png'))
        return total_error / side_cnt

    def remesh(self, scan_mesh, scale_factor=1.0):
        def _get_grid_coord_(v_min, v_max, res):
            x_ind, y_ind, z_ind = torch.meshgrid(torch.linspace(v_min[0], v_max[0], res[0]),
                                                 torch.linspace(v_min[1], v_max[1], res[1]),
                                                 torch.linspace(v_min[2], v_max[2], res[2]), indexing='ij')
            grid = torch.stack((x_ind, y_ind, z_ind), dim=0)
            grid = grid.float()
            pt = np.concatenate((np.asarray(x_ind).reshape(-1, 1),
                                 np.asarray(y_ind).reshape(-1, 1),
                                 np.asarray(z_ind).reshape(-1, 1)), axis=1)
            pt = pt.astype(float)
            return pt, grid.permute(0, 3, 2, 1).contiguous()

        def get_mesh(volume, grid_coord, scale_factor=1.0):
            # mesh generation.
            if isinstance(volume, np.ndarray):
                volume = torch.Tensor(volume)
            vertices, faces = marching_cubes(volume, 0.0)
            new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
            new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
            new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
            return new_mesh

        if scale_factor is not None:
            scan_mesh.vertices *= scale_factor

        v_margin = 3
        v_min = np.floor(np.min((scan_mesh.bounds[0], scan_mesh.bounds[0]), axis=0)).astype(int)
        v_max = np.ceil(np.max((scan_mesh.bounds[1], scan_mesh.bounds[1]), axis=0)).astype(int)
        v_min -= v_margin
        v_max += v_margin
        res = (v_max - v_min) * 2

        query_pts, grid_coord = _get_grid_coord_(v_min, v_max, res=res)
        sdf_gt = SDF(scan_mesh.vertices, scan_mesh.faces)
        volume_gt = sdf_gt(query_pts)
        mesh_remehed = get_mesh(volume_gt.reshape(res), grid_coord)

        kdtree = KDTree(scan_mesh.vertices, leaf_size=30, metric='euclidean')
        kd_idx = kdtree.query(mesh_remehed.vertices, k=1, return_distance=False)
        vertex_colors = scan_mesh.visual.vertex_colors[kd_idx.squeeze(), :]

        if scale_factor is not None:
            mesh_remehed.vertices = mesh_remehed.vertices / scale_factor

        remeshed_gt = trimesh.Trimesh(mesh_remehed.vertices,
                                      mesh_remehed.faces,
                                      vertex_colors=vertex_colors)
        return remeshed_gt
    def get_subdivided_smpl_mesh(self, vertices, faces, iter=1, remesh=False):
        if vertices is None:
            vertices = self.smpl_mesh.vertices.detach().squeeze(0).cpu().numpy()
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
        if iter > 1:
            vertices, faces = trimesh.remesh.subdivide(vertices, faces)
        smpl_vertices = vertices[:, :3]

        if remesh is False:
            return smpl_vertices, faces
        else:
            input_mesh = trimesh.smoothing.filter_laplacian(trimesh.Trimesh(smpl_vertices, faces, process=False))
            smpl_mesh = self.smpl_remesh(input_mesh)
            kdtree = KDTree(smpl_vertices, leaf_size=30, metric='euclidean')
            kd_idx = kdtree.query(smpl_mesh.vertices, k=1, return_distance=False)
            # return smpl_mesh.vertices, new_lbs, smpl_mesh.faces
            return smpl_mesh

    @torch.no_grad()
    def evaluate(self, input_var, model, model_C, data_name):
        model.eval()
        model_C.eval()

        start = time.time()
        pred_var = model(torch.cat([input_var[0], input_var[1], input_var[2]], dim=1))

        mask = input_var[1]

        res = mask.shape[2]
        batch_size = mask.shape[0]
        focal = np.sqrt(res * res + res * res)
        x = np.reshape((np.linspace(0, res, res) - int(res / 2)) / focal,
                       [1, 1, -1, 1])
        y = np.reshape((np.linspace(0, res, res) - int(res / 2)) / focal,
                       [1, 1, 1, -1])
        x = np.tile(x, [batch_size, 1, 1, res])
        y = np.tile(y, [batch_size, 1, res, 1])
        xy = torch.Tensor(np.concatenate((x, y), axis=1)).to(self.device)

        pred_df, pred_db = torch.chunk(pred_var['pred_depth'], chunks=pred_var['pred_depth'].shape[1], dim=1)
        predfd2n = get_plane_params(z=pred_df, xy=xy,
                                    pred_res=self.res, real_dist=self.real_dist,
                                    z_real=True, v_norm=True)
        predbd2n = get_plane_params(z=pred_db, xy=xy,
                                    pred_res=self.res, real_dist=self.real_dist,
                                    z_real=True, v_norm=True)
        pred_depth2normal = torch.cat([predfd2n[:, 0:3, :, :],
                                       predbd2n[:, 0:3, :, :]], dim=1)

        # input = torch.cat([input_var[0], input_var[1], pred_depth2normal], dim=1)#.detach()
        input = torch.cat([input_var[0], pred_depth2normal], dim=1)#.detach()
        pred_color = model_C(input)['pred_color']

        cf, cb = torch.chunk(pred_color, chunks=2, dim=1)
        cf = cf * mask
        cb = cb * mask
        cf_numpy = cf[0].permute(1, 2, 0).detach().cpu().numpy()
        cb_numpy = cb[0].permute(1, 2, 0).detach().cpu().numpy()
        cf_numpy = cf_numpy * self.RGB_STD + self.RGB_MEAN
        cb_numpy = cb_numpy * self.RGB_STD + self.RGB_MEAN
        cf_numpy[cf_numpy < 0] = 0
        cb_numpy[cb_numpy < 0] = 0

        input_img = input_var[0][0].permute(1, 2, 0).detach().cpu().numpy()
        input_img = input_img * self.RGB_STD + self.RGB_MEAN
        img_PSNR = self.psnr(input_img*255, cf_numpy*255)

        img_f = torch.from_numpy(np.transpose(cf_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_b = torch.from_numpy(np.transpose(cb_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float()
        imgf_LR = img_f.unsqueeze(0)
        imgf_LR = imgf_LR.to(self.device)
        imgb_LR = img_b.unsqueeze(0)
        imgb_LR = imgb_LR.to(self.device)

        with torch.no_grad():
            output_cf = self.esr_model(imgf_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_cb = self.esr_model(imgb_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output_cf = np.transpose(output_cf[[2, 1, 0], :, :], (1, 2, 0))
        output_cf = (output_cf * 255.0).round()
        output_cb = np.transpose(output_cb[[2, 1, 0], :, :], (1, 2, 0))
        output_cb = (output_cb * 255.0).round()

        output_cf = np.array(output_cf, dtype=np.uint8)
        output_cb = np.array(output_cb, dtype=np.uint8)

        # generate volume from depth map
        pred_depth = pred_var['pred_depth']
        df, db = torch.chunk(pred_depth, chunks=2, dim=1)
        df = (df - 0.5) * 128 + self.real_dist
        db = (db - 0.5) * 128 + self.real_dist
        # df[df < 0] = 0
        # db[db < 0] = 0

        df *= mask
        db *= mask

        volume = depth2occ_2view_torch(df, db, binarize=False,
                                       z_min=self.z_min, z_max=self.z_max,
                                       voxel_size=self.voxel_size)

        volume = self.volume_filter(volume, iter=1)
        end = time.time()

        if 'pred_lbs' in pred_var:
            pred_lbs = pred_var['pred_lbs']
            lf, lb = torch.chunk(pred_lbs, chunks=2, dim=1)
            lf *= mask
            lb *= mask
            lf_numpy = lf[0].permute(1, 2, 0).detach().cpu().numpy()
            lb_numpy = lb[0].permute(1, 2, 0).detach().cpu().numpy()
            # lf_numpy[lf_numpy < 0] = 0
            # lb_numpy[lb_numpy < 0] = 0

            output_lf = np.transpose(np.transpose(lf_numpy[:, :, [2, 1, 0]], (2, 0, 1)), (1, 2, 0))
            output_lb = np.transpose(np.transpose(lb_numpy[:, :, [2, 1, 0]], (2, 0, 1)), (1, 2, 0))
            output_lf = np.clip(output_lf, a_min=0, a_max=1)
            output_lb = np.clip(output_lb, a_min=0, a_max=1)

            lf_1024_f = Image.fromarray((output_lf*255).astype(np.uint8))
            lf_1024_b = Image.fromarray((output_lb*255).astype(np.uint8))
            output_lf = np.array(lf_1024_f.resize((1024, 1024)))
            output_lb = np.array(lf_1024_b.resize((1024, 1024)))

            lbs_color_mesh = colorize_model(volume, output_lf / 255.0, output_lb / 255.0,
                                            mask=mask.squeeze().detach().cpu().numpy(),
                                            texture_map=False)  # , volume_level=0.0)

            lbs_color_mesh = self.postprocess_mesh(lbs_color_mesh, num_faces=50000)

            new_vertices = grid_interp(self.coord, torch.Tensor(lbs_color_mesh.vertices))
            R = self.make_rotation_matrix(0, math.radians(0), math.radians(-90))
            vertices = np.matmul(np.asarray(new_vertices), R.transpose(1, 0))
            vertices[:, 2] *= (-1)
            vertices[:, 1] += self.camera_height  # 60.0 # 308.0/512.0/self.focal*220.0
            vertices[:, 2] += self.real_dist

            lbs_pred_mesh = trimesh.Trimesh(vertices=vertices,
                                            faces=lbs_color_mesh.faces, #lbs_color_mesh.faces[:, ::-1],
                                            visual=lbs_color_mesh.visual)

            # rot_ang = int(data_name.split('/')[-1].split('_')[1])
            #
            # R = self.make_rotation_matrix(0, math.radians(-rot_ang), 0)
            # vertices = np.matmul(np.asarray(lbs_pred_mesh.vertices), R.transpose(1, 0))
            #
            # lbs_pred_mesh_spin = trimesh.Trimesh(vertices=vertices,
            #                                 faces=lbs_pred_mesh.faces,
            #                                 visual=lbs_pred_mesh.visual)
            lbs_pred_mesh.fix_normals()
        else:
            lbs_pred_mesh = None
            lbs_pred_mesh_spin = None

        color_mesh = colorize_model(volume, output_cf / 255.0, output_cb / 255.0,
                                        mask=mask.squeeze().detach().cpu().numpy(),
                                        texture_map=False)  # , volume_level=0.0)

        color_mesh = self.postprocess_mesh(color_mesh, num_faces=50000)

        new_vertices = grid_interp(self.coord, torch.Tensor(color_mesh.vertices))
        R = self.make_rotation_matrix(0, math.radians(0), math.radians(-90))
        vertices = np.matmul(np.asarray(new_vertices), R.transpose(1, 0))
        vertices[:, 2] *= (-1)
        vertices[:, 1] += self.camera_height  # 60.0 # 308.0/512.0/self.focal*220.0
        vertices[:, 2] += self.real_dist
        color_pred_mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=color_mesh.faces, #color_mesh.faces[:, ::-1]
                                        visual=color_mesh.visual)

        # rot_ang = int(data_name.split('/')[-1].split('_')[1])
        #
        # R = self.make_rotation_matrix(0, math.radians(-rot_ang), 0)
        # vertices = np.matmul(np.asarray(color_pred_mesh.vertices), R.transpose(1, 0))
        #
        # color_pred_mesh = trimesh.Trimesh(vertices=vertices,
        #                                 faces=color_pred_mesh.faces,
        #                                 visual=color_pred_mesh.visual)

        color_pred_mesh.fix_normals()

        process_time = (end - start)
        return lbs_pred_mesh, color_pred_mesh, output_cf, output_cb, df, db, process_time

    def smpl_remesh(self, gt_mesh, scale_factor=1.0):
        if scale_factor is not None:
            gt_mesh.vertices *= scale_factor
        v_margin = 3
        v_min = np.floor(np.min((gt_mesh.bounds[0], gt_mesh.bounds[0]), axis=0)).astype(int)
        v_max = np.ceil(np.max((gt_mesh.bounds[1], gt_mesh.bounds[1]), axis=0)).astype(int)
        v_min -= v_margin
        v_max += v_margin
        res = (v_max - v_min) * 2

        query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
        sub_meshes = []
        total_num_faces = gt_mesh.faces.shape[0]
        cc = trimesh.graph.connected_components(gt_mesh.face_adjacency, min_len=3)
        for k in range(len(cc)):
            mask = np.zeros(total_num_faces, dtype=bool)
            tmp_mesh = gt_mesh.copy()
            mask[cc[k]] = True
            tmp_mesh.update_faces(mask)
            sub_meshes.append(tmp_mesh)

        sdf_gt = SDF(sub_meshes[0].vertices, sub_meshes[0].faces)
        sdf_misc1 = SDF(sub_meshes[1].vertices, sub_meshes[1].faces)
        sdf_misc2 = SDF(sub_meshes[2].vertices, sub_meshes[2].faces)
        volume_gt = sdf_gt(query_pts)
        volume_1 = sdf_misc1(query_pts)
        volume_2 = sdf_misc2(query_pts)
        volume_gt = np.maximum(np.maximum(volume_1, volume_2), volume_gt)
        mesh_merged = self.get_mesh(volume_gt.reshape(res), grid_coord)

        if scale_factor is not None:
            mesh_merged.vertices = mesh_merged.vertices / scale_factor

        remeshed_gt = trimesh.Trimesh(mesh_merged.vertices, mesh_merged.faces, process=True)
        return remeshed_gt

    def auto_rig(self, canon_mesh, smpl_mesh, smpl_lbs=None, scale_factor=128.0, v_margin=10, type='flow_subdiv'):
        if type == 'flow_subdiv':
            custom_mesh_scaled = trimesh.Trimesh(canon_mesh.vertices * scale_factor, canon_mesh.faces)
            custom_mesh_scaled = self.postprocess_mesh(custom_mesh_scaled, 10000)

            ref_vertices, ref_faces = \
                self.get_subdivided_smpl_mesh(vertices=smpl_mesh.vertices, faces=smpl_mesh.faces, iter=1)

            smpl_mesh_scaled = trimesh.Trimesh(ref_vertices * scale_factor, ref_faces, process=False)
            v_min = np.floor(np.min((smpl_mesh_scaled.bounds[0], smpl_mesh_scaled.bounds[0]), axis=0)).astype(int)
            v_max = np.ceil(np.max((smpl_mesh_scaled.bounds[1], smpl_mesh_scaled.bounds[1]), axis=0)).astype(int)
            v_min -= v_margin
            v_max += v_margin
            res = (v_max - v_min) * 2

            # show_meshes([smpl_mesh_scaled, custom_mesh_scaled])
            query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
            sdf_scan = SDF(custom_mesh_scaled.vertices, custom_mesh_scaled.faces)
            # use remeshed smpl sdf for smooth surface reconstruction.
            smpl_mesh4sdf = self.smpl_remesh(smpl_mesh_scaled, scale_factor=1.0)
            sdf_smpl = SDF(smpl_mesh4sdf.vertices, smpl_mesh4sdf.faces)
            volume_smpl = sdf_smpl(query_pts)

            # pixels in interests.
            canonical_vertices = smpl_mesh.vertices * scale_factor
            kdtree_s = KDTree(canonical_vertices, leaf_size=30, metric='euclidean')
            idx_s = kdtree_s.query(custom_mesh_scaled.vertices, k=1, return_distance=False)
            semantic_labels = np.zeros_like(canonical_vertices[:, 0:1])
            semantic_labels[self.smpl_optimizer.v_label['head']] = 1
            semantic_labels[self.smpl_optimizer.v_label['neck']] = 1
            semantic_labels[self.smpl_optimizer.v_label['leftArm']] = 2
            semantic_labels[self.smpl_optimizer.v_label['rightArm']] = 2
            semantic_labels[self.smpl_optimizer.v_label['leftForeArm']] = 3
            semantic_labels[self.smpl_optimizer.v_label['rightForeArm']] = 3
            semantic_labels[self.smpl_optimizer.v_label['leftHand']] = 4
            semantic_labels[self.smpl_optimizer.v_label['rightHand']] = 4
            semantic_labels[self.smpl_optimizer.v_label['leftFoot']] = 4
            semantic_labels[self.smpl_optimizer.v_label['leftToeBase']] = 4
            semantic_labels[self.smpl_optimizer.v_label['rightFoot']] = 4
            semantic_labels[self.smpl_optimizer.v_label['rightToeBase']] = 4
            canonical_labels = semantic_labels[idx_s.reshape(-1)].reshape(-1)

            # d_out = [6.0, 10.0, 2.0, 1.0, 0.1]
            # d_in = [2.0, 1.0, 1.0, 0.0, 0.1]  # tau should be smaller than max(d_out, d_in)
            # tau = [6.0, 10.0, 2.0, 1.0, 0.1]
            d_out = [12.0, 12.0, 2.0, 2.0, 0.5]
            d_in = [2.0, 1.0, 1.0, 0.0, 0.1]  # tau should be smaller than max(d_out, d_in)
            tau = [12.0, 12.0, 2.0, 2.0, 0.5]
            target_volume = copy.deepcopy(volume_smpl)
            denorm = np.ones_like(target_volume)

            for k in range(len(tau)):
                valid_voxels = np.where((volume_smpl > -d_out[k]) & (volume_smpl < d_in[k]))[0]
                valid_query = query_pts[valid_voxels, :]

                # assign semantic labels to custom_mesh
                kdtree_r = KDTree(custom_mesh_scaled.vertices[canonical_labels == k, :],
                                  leaf_size=30, metric='euclidean')
                dist_r, idx_r = kdtree_r.query(valid_query, k=1, return_distance=True)

                query_src = valid_query[dist_r.reshape(-1) < tau[k], :]
                query_idx = valid_voxels[dist_r.reshape(-1) < tau[k]]

                # put posed space signed distance to the canonical signed distance field.
                sdf_vals = sdf_scan(query_src)
                target_volume[query_idx] += sdf_vals
                denorm[query_idx] += 1.0

            target_volume = target_volume / denorm
            target_volume[denorm == 1] = volume_smpl[denorm == 1]
            target_volume = target_volume.reshape(res[0], res[1], res[2])

            b_range = 10
            volume_smpl = volume_smpl.reshape([res[0], res[1], res[2]])
            self.replace_hands = False  # always for single image input.

            if self.replace_hands:
                avg_left = np.mean(smpl_mesh_scaled.vertices[self.smpl_optimizer.left_wrist_idx, :], axis=0)
                avg_right = np.mean(smpl_mesh_scaled.vertices[self.smpl_optimizer.right_wrist_idx, :], axis=0)

                delta = np.abs(grid_coord[0, 0, 0, 1] - grid_coord[0, 0, 0, 0])
                left_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_left[0]) <= delta)
                right_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_right[0]) <= delta)

                offset_left = left_idx[0][-1]  # right-most value
                offset_right = right_idx[0][0]  # left-most value

                target_volume[offset_left:, :, :] = volume_smpl[offset_left:, :, :]
                target_volume[:offset_right, :, :] = volume_smpl[:offset_right, :, :]

                # linearly blending two volumes near the wrist
                for k in range(b_range):
                    alpha = (1 - k / (b_range - 1)) * 0.05
                    target_volume[offset_left - k, :, :] = \
                        target_volume[offset_left - k, :, :] * (1 - alpha) + \
                        volume_smpl[offset_left - k, :, :] * alpha
                    target_volume[offset_right + k, :, :] = \
                        target_volume[offset_right + k, :, :] * (1 - alpha) \
                        + volume_smpl[offset_right + k, :, :] * alpha

                target_volume[offset_right:offset_left, :, :] = \
                    self.volume_filter(target_volume[offset_right:offset_left, :, :], iter=3)

            mesh_merged = self.get_mesh(target_volume, grid_coord)
            mesh_merged = trimesh.smoothing.filter_laplacian(mesh_merged, lamb=0.3)
            mesh_merged = self.remesh(mesh_merged)
            mesh_merged = self.postprocess_mesh(mesh_merged, 10000)

            mesh_merged.vertices = mesh_merged.vertices / scale_factor

        else:
            scaled_canon_mesh = trimesh.Trimesh(canon_mesh.vertices * scale_factor, canon_mesh.faces)
            custom_mesh = self.postprocess_mesh(scaled_canon_mesh, 10000)
            sdf_func1 = SDF(custom_mesh.vertices, custom_mesh.faces)

            smpl_mesh = trimesh.Trimesh(smpl_mesh.vertices * scale_factor,
                                        smpl_mesh.faces)
            smpl_mesh = smpl_mesh.subdivide()
            sdf_func2 = SDF(smpl_mesh.vertices, smpl_mesh.faces)
            sdf_func2.sample_surface(50000)
            v_min = np.floor(np.min((custom_mesh.bounds[0], custom_mesh.bounds[0]), axis=0)).astype(int)
            v_max = np.ceil(np.max((custom_mesh.bounds[1], custom_mesh.bounds[1]), axis=0)).astype(int)
            v_min -= 10
            v_max += 10
            res = (v_max - v_min) * 2

            query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
            volume_scan1 = sdf_func1(query_pts).reshape((res[0], res[1], res[2]))
            volume_scan2 = sdf_func2(query_pts).reshape((res[0], res[1], res[2]))

            # remove artifacts by blending with smpl's sdf volume
            tau_min, tau_max = -2.0, 5.0
            boundary_condition = (volume_scan2 < tau_min) | (volume_scan2 > tau_max)
            volume_scan1[boundary_condition] = volume_scan2[boundary_condition]
            volume_scan1 = self.volume_filter(volume_scan1)

            mesh_merged = self.get_mesh(volume_scan1, grid_coord)
            mesh_merged.vertices /= scale_factor
        return mesh_merged

    def poisson_o3d(self, vertices, vertex_colors=None, vertex_normals=None, depth=9):
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(vertices)
        if vertex_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(vertex_colors / 255)
        if vertex_normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(vertex_normals)  # slower
        else:
            pcd.estimate_normals()

        # o3d.visualization.draw_geometries([pcd])
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        o3d.visualization.draw_geometries([mesh])
        return mesh


    def forward(self, images, depth_front, depth_back, masks, data_names=None, smpl_params=None, smpl_resource=None):
        lbs_pred_meshes = []
        color_pred_meshes = []
        pred_images_front = []
        pred_images_back = []
        canon_meshes = []
        canon_from_smpl_meshes = []


        for i in range(len(images)):
            image = images[i]
            image = torch.Tensor(image).permute(2, 0, 1).float()
            if torch.max(image) > 1.0:
                image = image / torch.Tensor(self.RGB_MAX).view(3, 1, 1)
            image_input = (image - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                          / torch.Tensor(self.RGB_STD).view(3, 1, 1)
            if masks is not None:
                mask = masks[i]
                # cv2.imwrite('mask.png', mask)
                mask_input = torch.Tensor(mask).permute(2, 0, 1).cuda().float()
                if torch.max(mask_input) > 1.0:
                    mask_input = mask_input / 255.0
            else:
                mask = torch.zeros((1, image.shape[1], image.shape[2]))

            front_depth = depth_front[i]
            back_depth = depth_back[i]
            front_depth = torch.Tensor(front_depth.copy()).permute(2, 0, 1).float().cuda()
            back_depth = torch.Tensor(back_depth.copy()).permute(2, 0, 1).float().cuda()
            depth_input = torch.cat([front_depth, back_depth], dim=0)

            datum = dict()
            datum['input'] = (image_input, mask_input, depth_input)
            input_var, target_var = self.init_variables(datum, device=self.device)

            lbs_pred_mesh, color_pred_mesh, image_front, image_back, ydf, ydb, process_time\
                = self.evaluate(input_var, self.model, self.model_C, data_names[i])

            lbs_pred_meshes.append(lbs_pred_mesh)
            color_pred_meshes.append(color_pred_mesh)
            pred_images_front.append(image_front)
            pred_images_back.append(image_back)

            if data_names is not None:
                file_name = data_names[i].split('/')[-1]
                dir_name = data_names[i].split('/')[-2]
                ext = '.' + file_name.split('.')[-1]  # can be any format (png, jpg, jpeg, etc.)
                save_dir = os.path.join(self.result_path, dir_name)
                # save_dir = os.path.join(self.result_path, dir_name, file_name[:-4])
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file_name.replace(ext, '.obj'))
                save_img_front = os.path.join(save_dir, file_name.replace(ext, '_front.png'))
                save_img_back = os.path.join(save_dir, file_name.replace(ext, '_back.png'))

                # GT가 있을때,
                gt_path = os.path.join('/'.join(save_path.replace('INIT_%s' % self.model_config, 'GT').split('/')[:-2]), dir_name, dir_name + '.obj')
                if os.path.isfile(gt_path):
                    pose_path = gt_path.replace('obj', 'pt')
                    pose = torch.load(pose_path)
                else:
                    pose = None

                if lbs_pred_mesh is not None:
                    self.src_lbs_mesh = lbs_pred_mesh
                else:
                    self.src_lbs_mesh = None
                self.src_mesh = color_pred_mesh

                if self.src_lbs_mesh is not None:
                    smpl_param = smpl_params[0]

                    canon_mesh, smpl_mesh, smpl_lbs = self.canonicalization(self.src_lbs_mesh, self.src_mesh,
                                                                            smpl_param, pose, smpl_resource)

                    confidence = self.get_face_length(canon_mesh.vertices, canon_mesh.faces)
                    canon_mesh.update_faces(confidence)
                    canon_mesh.remove_degenerate_faces()

                    canon_from_smpl = self.auto_rig(canon_mesh, smpl_mesh, smpl_lbs, scale_factor=128.0,
                                                    type='flow_subdiv')

                    # canon_from_smpl = trimesh.smoothing.filter_laplacian(canon_from_smpl)
                    canon_from_smpl = self.postprocess_mesh(canon_from_smpl, 50000)
                    canon_mesh.export(save_path.replace('.obj', '_canon.obj'))
                    canon_from_smpl.export(save_path.replace('.obj', '_canon_from_smpl.obj'))
                    self.src_lbs_mesh.export(save_path)

                self.src_mesh.export(save_path.replace('.obj', '_color.obj'))
                p2s = 0
                chamfer = 0
                p2s_outlier = 0
                chamfer_outlier = 0
                normal_loss = 0.0

                # p2s, chamfer, p2s_outlier, chamfer_outlier \
                #     = self.eval_mesh(self.tgt_mesh, self.src_mesh, num_samples=10000)
                # self._normal_render = NormalRender(width=512, height=512)
                # normal_loss = self.get_reproj_normal_error(save_demo_img=save_path)
            else:
                p2s = 0
                chamfer = 0
                p2s_outlier = 0
                chamfer_outlier = 0
                normal_loss = 0.0

            cv2.imwrite(save_img_front, image_front)
            cv2.imwrite(save_img_back, image_back)
        return (lbs_pred_meshes, color_pred_meshes, canon_meshes, canon_from_smpl_meshes, normal_loss*100, process_time, p2s, chamfer,
                p2s_outlier, chamfer_outlier)

if __name__=='__main__':
    recon = HumanRecon()
