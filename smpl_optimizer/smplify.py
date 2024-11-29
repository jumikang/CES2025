from __future__ import annotations

import numpy as np
import smplx
import torch.utils.data
import json
import trimesh
import yaml
from argparse import Namespace
from tqdm import tqdm
import os.path
# from pysdf import SDF
from utils.geometry import *
import torch.nn as nn
# from utils.loader_utils import *
# from utils.visualizer import show_meshes
from smpl_optimizer.model_fitter import SmplHandler, SmplParams
from apps.options import Configurator
from sklearn.neighbors import KDTree

import sys
sys.path.append(os.path.abspath(''))
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance
)


class MaxMixturePosePrior(nn.Module):
    def __init__(self, n_gaussians=8, prefix=0, device=torch.device('cpu')):
        '''
        Class for Preventing 'Candy-wrapper artifact' and 'Unnatural Motions' with Gaussian Mixture Models
        :param n_gaussians: the number of gaussian models
        :param prefix: prefix offset for input (usually 0 or 3)
        :param device: device
        '''
        super(MaxMixturePosePrior, self).__init__()

        self.prefix = prefix
        self.n_gaussians = n_gaussians
        self.smpl_model = 'smplx'
        self.precs = self.means = self.weights = []
        self.create_prior_from_cmu(device)

    def create_prior_from_cmu(self, device):
        """Load the gmm from the CMU motion database."""
        with open(os.path.join(os.path.dirname(__file__), '../resource/smpl_models/gmm_08.pkl'), 'rb') as f:
            gmm = pickle.load(f, encoding='bytes')
        precs = np.asarray([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm[b'covars']])
        means = np.asarray(gmm[b'means'])  # [8, 69]

        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm[b'covars']])
        const = (2 * np.pi) ** (69 / 2.)
        weights = np.asarray(gmm[b'weights'] / (const * (sqrdets / sqrdets.min())))

        self.precs = torch.from_numpy(precs).to(device)  # [8, 69, 69]
        self.means = torch.from_numpy(means).to(device)  # [8, 69]
        if self.smpl_model == 'smplx':
            self.precs = self.precs[:, :63, :63]  # to apply to SMPL-X (exclude pamls)
            self.means = self.means[:, :63]  # to apply to SMPL-X
        self.weights = torch.from_numpy(weights).to(device)

    @staticmethod
    def pose_angle_prior(theta):
        # prevent twisted joints for elbows (52/55), wrist (58/61), knees (9/12), thighs (2/5)
        loss = torch.mean(theta[:, 52] ** 2 + theta[:, 55] ** 2 + theta[:, 58] ** 2 + theta[:, 61] ** 2
                          + theta[:, 9] ** 2 + theta[:, 12] ** 2 + theta[:, 2] ** 2 + theta[:, 5] ** 2)
        # make spines straight as possible
        loss += torch.mean(theta[:, 6:9] ** 2) + torch.mean(theta[:, 15:18] ** 2) + torch.mean(theta[:, 24:27] ** 2)
        # make foot flat as possible (foot and ankles) - maybe ankles are not necessary.
        loss += torch.mean(theta[:, 27:33] ** 2)  # + torch.mean(theta[:, 18:24] ** 2)
        return loss

    def forward(self, x):
        theta = x[:, self.prefix:]
        batch, dim = theta.shape
        theta = theta.expand(self.n_gaussians, batch, dim).permute(1, 0, 2)
        theta = (theta - self.means[None])[:, :, None, :]
        log_likelihoods = np.sqrt(0.5) * torch.matmul(theta, self.precs.expand(batch, *self.precs.shape)).squeeze(2)
        results = (log_likelihoods * log_likelihoods).sum(-1) - self.weights.log()
        return results.min() + self.pose_angle_prior(x)


class SmplOptimizer(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 params,
                 cam_params=None,
                 input_path=None,
                 output_path=None,
                 path2obj=None,
                 num_models=1,
                 flip_y=False,
                 renderer=None,
                 device='cuda:0'):
        """
        Class for optimizing SMPL-X parameters from image(s) w/ or w/o meshes
        :param params:
        :param cam_params:
        :param input_path:
        :param output_path:
        :param path2obj:
        :param num_models:
        :param flip_y:
        :param device:
        """
        super(SmplOptimizer, self).__init__()

        # fetch options
        self.device = device
        self.model_path = params.smpl_path
        self.model_type = params.smpl_type
        self.semantic_labels = params.path2semantic
        self.gender = params.smpl_gender
        self.cam_params = cam_params
        self.age = params.age
        if 'hmin' in cam_params and 'hmax' in cam_params:
            self.average_height = (cam_params['hmin'] + cam_params['hmax'])/2
        else:
            self.average_height = 180.0

        self.centroid = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
        self.scale = 1 / (self.average_height / 2)

        # optimization parameters
        self.step_size = 1e-2
        self.init_iters = 200
        # self.main_iters = 300
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # path
        self.input_path = input_path
        self.save_path = output_path

        self.flip_y = flip_y
        if self.flip_y:  # opengl coordinate (graphics)
            cam_params['py'] = cam_params['recon_res'] - cam_params['py']
        else:  # image coordinate (computer vision)
            cam_params['py'] = cam_params['px']

        # to apply rendering loss (depth, silhouette, and color) - used for multiview input (slow!)
        # if renderer is None:
        #     self.renderer = HumanRenderer(input_path=input_path,
        #                                   save_root=output_path,
        #                                   path2obj=path2obj,
        #                                   cam_ext=cam_params,
        #                                   renderer='trimesh',
        #                                   rendering_mode='smpl_dual',
        #                                   view_idx=cam_params['view_idx'],
        #                                   device=self.device)
        # else:
        #     self.renderer = renderer

        # handling optimization variables
        self.smpl_handler = SmplHandler(model_path=self.model_path,
                                        model_type=self.model_type,
                                        use_pca=params.smpl_pca,
                                        num_pca_comp=params.smpl_num_pca_comp,
                                        num_models=num_models,
                                        num_betas=params.smpl_num_beta,
                                        gender=self.gender,
                                        flat_hands=params.smpl_flat_hand,
                                        age=self.age,
                                        pose_detector=params.pose_detector,
                                        device=self.device)

        self.v_label = None
        self.hand_idx = []
        self.left_hand_idx = []
        self.right_hand_idx = []
        self.non_hand_idx = []
        self.left_wrist_idx = []
        self.right_wrist_idx = []
        self.left_arm_idx = []
        self.right_arm_idx = []
        self.head_idx = []

    def init_semantic_labels(self, path2label=None):
        """
        Set semantic labels for SMPL(-X) vertices
        :param path2label: path to the semantic information (json file)
        :results are saved in instance variables
        """
        if path2label is None:
            path2label = self.semantic_labels
        # semantic labels for smplx vertices.
        if self.v_label is None and os.path.isfile(path2label):
            with open(path2label, "r") as json_file:
                self.v_label = json.load(json_file)
                self.v_label['leftWrist'], self.v_label['rightWrist'] = [], []
                for k in self.v_label['leftHand']:
                    if k in self.v_label['leftForeArm']:
                        self.left_wrist_idx.append(k)
                for k in self.v_label['rightHand']:
                    if k in self.v_label['rightForeArm']:
                        self.right_wrist_idx.append(k)
                for key in self.v_label.keys():
                    if 'leftHand' in key:
                        self.left_hand_idx += self.v_label[key]
                    elif 'rightHand' in key:
                        self.right_hand_idx += self.v_label[key]
                    else:
                        self.non_hand_idx += self.v_label[key]
                self.nonbody_idx = self.v_label['head'] + self.v_label['eyeballs'] + \
                                self.v_label['leftToeBase'] + self.v_label['rightToeBase'] + \
                                self.v_label['leftEye'] + self.v_label['rightEye'] + \
                                self.hand_idx
                self.nonbody_idx = np.asarray(list(set(self.nonbody_idx)))  # unique idx
                self.body_idx = np.asarray([i for i in range(0, 10475) if i not in self.nonbody_idx])

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        """
        Retrieve the bounding box for given vertices
        :param vertices: batch of input vertices
        :return: bounding boxes
        """
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    def smpl2real(self, vertices):
        """
        Warp input vertices to the real domain from SMPL coordinate
        :param vertices:
        :return:
        """
        if isinstance(vertices, np.ndarray):
            vertices = torch.Tensor(vertices).float().to(self.device)
        squeeze_back = False
        if vertices.size() == 3:
            vertices = vertices.squeeze(0)
            squeeze_back = True
        vertices = (vertices - self.smpl_handler.centroid_tensor) * self.smpl_handler.scale / self.scale + self.centroid
        if squeeze_back:
            vertices = vertices.unsqueeze(0)
        return vertices

    def real2smpl(self, vertices):
        if isinstance(vertices, np.ndarray):  # differentiable if the input is tensor
            vertices = torch.Tensor(vertices).float().to(self.device)
        squeeze_back = False
        if vertices.size() == 3:
            vertices = vertices.squeeze(0)
            squeeze_back = True
        vertices = (vertices - self.centroid) * self.scale / self.smpl_handler.scale + self.smpl_handler.centroid_tensor
        if squeeze_back:
            vertices = vertices.unsqueeze(0)
        return vertices

    @staticmethod
    def get_normalize_params(vertices, faces):  # must be trimesh.
        mesh = trimesh.Trimesh(vertices, faces.squeeze().cpu().numpy(), process=False)
        centroid_scan = mesh.bounding_box.centroid
        scale_scan = 2.0 / np.max(mesh.bounding_box.extents)
        return centroid_scan, scale_scan

    def set_initial_params(self, params, idx=0):
        self.smpl_handler.smpl_params.set_params(params, idx=idx)

    def joint_2d_loss(self, proj_joints, joints, option='all'):
        loss = 0.0

        if 'all' in option:
            mask = torch.ones_like(proj_joints).to(self.device)
            # make foot straight
            if 'foot' in option:
                mask[10:12, :] = 0.0  # index 10, 11
                mask[60:66, :] = 0.0  # index 60-65
                mask[7:9, :] = 0.0  # ankles (index 7-8)
        else:
            mask = torch.zeros_like(proj_joints).to(self.device)
            if 'hands' in option:
                mask[25:55, :] = 1.0
                mask[66:76, :] = 1.0

            if 'face' in option:
                mask[12:, :] = 1.0  # neck
                mask[15:, :] = 1.0  # head
                mask[23:25, :] = 1.0  # eyes
                mask[55:60, :] = 1.0  # nose, eye, ears
                mask[76:, :] = 1.0  # landmarks (contour)

            if 'body' in option:
                mask[:23, :] = 1.0

            if 'foot' in option:
                mask[10:12, :] = 1.0  # index 10, 11
                mask[60:66, :] = 1.0  # index 60-65
                mask[7:9, :] = 0.0  # ankles (index 7-8)

        loss += self.l2_loss(proj_joints * mask, joints * mask)
        return loss

    def visualize_smpl_joints(self, proj_joints, joints, image=None):
        width, height = 640, 640
        ratio_x = width / self.cam_params['width']
        ratio_y = height / self.cam_params['height']
        if image is not None:
            tmp = image.copy()
            tmp = cv2.resize(tmp, (640, 640))
        else:
            tmp = np.zeros((640, 640, 3))

        for i in range(proj_joints.shape[1]):
            pidx = (proj_joints[:, i, :])
            jidx = (joints[:, i, :])
            if int(pidx[0, 1]) < self.cam_params['width'] and int(pidx[0, 0]) < self.cam_params['height'] and \
                    int(jidx[0, 1]) < self.cam_params['width'] and int(jidx[0, 0]) < self.cam_params['height']:
                tmp = cv2.circle(tmp, (int(pidx[0, 0] * ratio_x), int(pidx[0, 1] * ratio_y)), 3,
                                 [0, 0, 255], 1)
                tmp = cv2.circle(tmp, (int(jidx[0, 0] * ratio_x), int(jidx[0, 1] * ratio_y)), 3,
                                 [0, 255, 0], 1)
        cv2.imshow('projected joints', tmp)
        cv2.waitKey(1)

    def optimize_smplx(self, scan_mesh, img_data, opt_options='base', loss_options='chamfer',
                       depth=None, K=None, R=None, t=None, keypoints=None, mask_gt=None, lr=0.01, iters=100,
                       model_constraints=None, pose_prior=None, keypoints3d=None, pred_faces=None):
        from pytorch3d.loss import (
            chamfer_distance
        )

        this_optimizer, this_scheduler = self.smpl_handler.set_optimizers(target=opt_options, lr=lr)

        smpl_output = self.smpl_handler()
        smpl_init = self.smpl_handler.get_smpl_params_torch(idx=0)
        betas_initial = smpl_output.betas.clone()
        with tqdm(range(iters), ncols=170) as pbar:
            for itr in pbar:
                loss_all = 0.0
                this_optimizer.zero_grad()
                smpl_output = self.smpl_handler(model_constraints=model_constraints)
                # smpl_output = self.smpl_handler.forward_test(smpl_init)
                if 'chamfer' in loss_options:
                    loss_cf, _ = chamfer_distance(smpl_output.vertices, scan_mesh)
                    loss_all += loss_cf * 0.5

                if 'prior' in loss_options:
                    loss_prior_pose = MaxMixturePosePrior.pose_angle_prior(smpl_output.body_pose)
                    loss_prior = torch.mean(smpl_output.expression ** 2) + torch.mean(smpl_output.betas**2)
                    loss_prior_beta = self.l1_loss(betas_initial, smpl_output.betas)
                    loss_all += loss_prior * 0.01 + loss_prior_pose * 0.01 + loss_prior_beta * 0.1

                if 'symmetry' in loss_options:
                    # can only be applied for symmetric input such as A-/T-posed humans
                    # fingers
                    loss_symm = self.l1_loss(smpl_output.left_hand_pose,
                                             smpl_output.right_hand_pose)
                    # elbows
                    loss_symm += self.l1_loss(smpl_output.body_pose[:, 51:54],
                                              smpl_output.body_pose[:, 54:57])
                    # wrists
                    loss_symm += self.l1_loss(smpl_output.body_pose[:, 57:60],
                                              smpl_output.body_pose[:, 60:63])
                    loss_all += loss_symm * 0.01

                # 2D joint loss
                if 'joint2d' in loss_options:
                    if isinstance(img_data, list) or isinstance(img_data, np.ndarray):
                        smpl_joints = self.smpl2real(smpl_output.joints)
                        proj_joints = perspective_projection(smpl_joints, R, t, K=K)
                        proj_joints[keypoints == 0] = 0
                        loss_jts = self.joint_2d_loss(proj_joints, keypoints, option=loss_options)
                        loss_all += loss_jts * 0.5
                        self.visualize_smpl_joints(proj_joints, keypoints)
                    elif isinstance(img_data, dict):
                        for k in range(len(img_data.keys())):
                            smpl_joints = self.smpl2real(smpl_output.joints)
                            proj_joints = perspective_projection(smpl_joints, R[k], t[k], K=K)
                            proj_joints[keypoints[k] == 0] = 0
                            loss_all += self.joint_2d_loss(proj_joints, keypoints[k], option=loss_options)

                if 'depth' in loss_options:
                    pass

                if 'joint3d' in loss_options:
                    smpl_joints = smpl_output.joints
                    mask = torch.ones_like(keypoints3d[:, :, 0:1])
                    mask[:, torch.sum(keypoints3d, dim=2).squeeze() == 0, :] = 0
                    joints_3d = self.real2smpl(keypoints3d)
                    loss_jts = self.l1_loss(smpl_joints * mask, joints_3d * mask)
                    loss_all += loss_jts * 0.2

                if 'sil' in loss_options:
                    vts_smpl = self.smpl2real(smpl_output.vertices)
                    vts_scan = self.smpl2real(scan_mesh)
                    mask = self.renderer(vts_smpl/5, self.smpl_handler.model.faces_tensor.unsqueeze(0),
                                         mode='silhouettes', K=K, R=R, t=t/5)
                    mask_gt = self.renderer(vts_scan/5, pred_faces,
                                            mode='silhouettes', K=K, R=R, t=t/5)
                    loss_all += self.l1_loss(mask, mask_gt) * 0.1
                    loss_all += torch.mean(torch.clip(mask - mask_gt, max=1, min=0)) * 0.1
                    R_new = self.get_random_rotation_matrix()
                    mask = self.renderer(vts_smpl / 5, self.smpl_handler.model.faces_tensor.unsqueeze(0),
                                         mode='silhouettes', K=K, R=R_new, t=t / 5)
                    mask_gt = self.renderer(vts_scan / 5, pred_faces,
                                            mode='silhouettes', K=K, R=R_new, t=t / 5)
                    loss_all += self.l1_loss(mask, mask_gt) * 0.1
                    loss_all += torch.mean(torch.clip(mask - mask_gt, max=1, min=0)) * 0.1  # shrink smpl mask

                # smpl_joints = self.smpl2real(smpl_output.joints)
                # proj_joints = perspective_projection(smpl_joints, R, t, K=K)
                # proj_joints[keypoints == 0] = 0
                # if itr % 50 == 0:
                #     self.visualize_smpl_joints(proj_joints, keypoints)
                # neural renderer
                pbar.set_description('optimization proc.: target:{0}, loss:{1}, iteration:{2}, loss:{3:0.5f}'
                                     .format(opt_options, loss_options, itr, loss_all))
                loss_all.backward()
                this_optimizer.step()
                this_scheduler.step(loss_all)

    def get_random_rotation_matrix(self):
        theta = int(np.random.uniform(0, 360, 1))
        phi = int(np.random.uniform(0, 360, 1))
        R_elev = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi)],
                           [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)]])
        R_azim = np.array([[np.cos(phi / 180 * np.pi), 0.0, np.sin(phi / 180 * np.pi)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(phi / 180 * np.pi), 0.0, np.cos(phi / 180 * np.pi)]])
        R = np.matmul(R_azim, R_elev)
        return torch.Tensor(R).float().unsqueeze(0).to(self.device)

    def parse_data(self, img_data):
        key = list(img_data.keys()).pop(0)
        if len(img_data[key]['cam_params']) == 0:
            return None, None, None, None, None
        K = torch.Tensor(img_data[key]['cam_params']['K']).to(self.device).unsqueeze(0)
        R, t, keypoints, mask_gt = [], [], [], []

        for key in img_data.keys():
            R.append(torch.Tensor(img_data[key]['cam_params']['R']).to(self.device).unsqueeze(0).float())
            t.append(torch.Tensor(img_data[key]['cam_params']['t']).to(self.device).unsqueeze(0).float())
            keypoints_openpose = img_data[key]['pose'].to(self.device)

            # ignore keypoints with low confidence
            confidence = keypoints_openpose[:, 2]
            keypoints_openpose[confidence < 0.5, :] = 0
            keypoints.append(self.smpl_handler.openpose2smpl(keypoints_openpose[:, 0:2]))
            if img_data[key]['mask'] is not None:
                mask_gt.append(torch.Tensor(img_data[key]['mask'][None, :, :, 0] / 255).to(self.device))

        # when???????? neural_renderer <-> trimesh
        # for R_ in R:
        #     R_[:, 1, 1] *= -1
        #     R_[:, 0, 2] *= -1
        #     R_[:, 2, 2] *= -1
        return K, R, t, keypoints, mask_gt

    @staticmethod
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

    def query_sdf(self, sdf_func, kdtree, query_pts, query_idx, n, return_sdf=True):
        dist = np.full(n, -100.0).astype(float)
        idx = np.full(n, -100.0).astype(float)

        d, i = kdtree.query(query_pts, k=1, return_distance=True)

        dist[query_idx] = d[:, 0]
        idx[query_idx] = i[:, 0]

        if return_sdf:
            sdf = np.full(n, -100.0).astype(float)
            sdf[query_idx] = sdf_func(query_pts)
            return dist, idx, sdf

        return dist, idx

    def canonical_remeshing(self, scan_mesh, lbs=None):
        from torchmcubes import marching_cubes, grid_interp
        sdf_scan = SDF(scan_mesh.vertices, scan_mesh.faces)

        v_min = scan_mesh.bounds[0].astype(np.int) - 1
        v_max = scan_mesh.bounds[1].astype(np.int) + 1
        res = (v_max - v_min) * 3

        query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
        kdtree = KDTree(scan_mesh.vertices, leaf_size=30, metric='euclidean')

        _, _, vals = self.query_sdf(sdf_scan, kdtree, query_pts, 10)
        sdf = vals.reshape((res[0], res[1], res[2]))

        vertices, faces = marching_cubes(torch.Tensor(sdf), 0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)

        idx = kdtree.query(new_vertices.detach().cpu().numpy(), k=1, return_distance=False)
        new_vertices = self.real2smpl(new_vertices.to(self.device))

        vertex_colors = scan_mesh.visual.vertex_colors[idx.squeeze(1), :]
        mesh = trimesh.Trimesh(vertices=new_vertices.detach().cpu().numpy(), faces=faces,
                               vertex_colors=vertex_colors, process=False)
        if lbs is not None:
            new_lbs = lbs[idx.squeeze(1), :]

        return mesh, new_lbs

    def replace_hands_canonical(self, smpl_mesh, scan_mesh):
        from torchmcubes import marching_cubes, grid_interp
        start_t = time.time()

        sdf_smpl = SDF(smpl_mesh.vertices, smpl_mesh.faces)
        sdf_scan = SDF(scan_mesh.vertices, scan_mesh.faces)

        v_min = np.floor(np.min((scan_mesh.bounds[0], smpl_mesh.bounds[0]), axis=0)).astype(np.int)
        v_max = np.ceil(np.max((scan_mesh.bounds[1], smpl_mesh.bounds[1]), axis=0)).astype(np.int)
        res = (v_max - v_min) * 3

        query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
        volume_smpl = sdf_smpl(query_pts).reshape((res[0], res[1], res[2]))
        volume_scan = sdf_scan(query_pts).reshape((res[0], res[1], res[2]))

        avg_left = np.mean(smpl_mesh.vertices[self.left_wrist_idx, :], axis=0)
        avg_right = np.mean(smpl_mesh.vertices[self.right_wrist_idx, :], axis=0)

        delta = np.abs(grid_coord[0, 0, 0, 1] - grid_coord[0, 0, 0, 0])
        left_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_left[0]) <= delta)
        right_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_right[0]) <= delta)

        offset_left = left_idx[0][-1]  # right-most value
        offset_right = right_idx[0][0]  # left-most value

        volume_scan[offset_left:, :, :] = volume_smpl[offset_left:, :, :]
        volume_scan[:offset_right, :, :] = volume_smpl[:offset_right, :, :]

        b_range = 20
        # linearly blending two volumes near the wrist
        for k in range(b_range):
            alpha = 1 - k/(b_range-1)
            volume_scan[offset_left - k, :, :] = \
                volume_scan[offset_left - k, :, :] * (1 - alpha) + volume_smpl[offset_left - k, :, :] * alpha
            volume_scan[offset_right + k, :, :] = \
                volume_scan[offset_right + k, :, :] * (1 - alpha) + volume_smpl[offset_right + k, :, :] * alpha

        # mesh generation.
        vertices, faces = marching_cubes(torch.Tensor(volume_scan), 0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        mesh = trimesh.Trimesh(new_vertices, faces)

        mesh = postprocess_mesh(mesh, num_faces=10000)
        kdtree_rgb = KDTree(scan_mesh.vertices, leaf_size=30, metric='euclidean')
        idx = kdtree_rgb.query(mesh.vertices, k=1, return_distance=False)
        vertex_colors = scan_mesh.visual.vertex_colors[idx.squeeze(1), :]
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.vertex_normals,
                               vertex_colors=vertex_colors, process=False)
        # mesh.show()
        print("> It took {:.2f}s seconds for changing full :-)".format(time.time() - start_t))
        return mesh

    def replace_hands(self, smpl_mesh, scan_mesh):
        from torchmcubes import marching_cubes, grid_interp
        start_t = time.time()

        sdf_smpl = SDF(smpl_mesh.vertices, smpl_mesh.faces)
        sdf_scan = SDF(scan_mesh.vertices, scan_mesh.faces)

        v_min = np.floor(np.min((scan_mesh.bounds[0], smpl_mesh.bounds[0]), axis=0)).astype(np.int)
        v_max = np.ceil(np.max((scan_mesh.bounds[1], smpl_mesh.bounds[1]), axis=0)).astype(np.int)
        res = (v_max - v_min) * 3

        query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
        query_pts_len = len(query_pts)

        query_pts_arr = np.array(query_pts)
        query_idx = np.where(np.abs(sdf_smpl(query_pts_arr)) < 15.0)[0]
        smpl_query_pts = np.array(query_pts_arr[query_idx, :])

        kdtree_hand = KDTree(smpl_mesh.vertices[self.hand_idx, :], leaf_size=30, metric='euclidean')
        kdtree_body = KDTree(smpl_mesh.vertices[self.non_hand_idx, :], leaf_size=30, metric='euclidean')
        # kdtree_wrist_left = KDTree(smpl_mesh.vertices[self.left_wrist_idx, :], leaf_size=30, metric='euclidean')
        # kdtree_wrist_right = KDTree(smpl_mesh.vertices[self.right_wrist_idx, :], leaf_size=30, metric='euclidean')

        dist_hand, idx_hand, sdf_hand = self.query_sdf(sdf_smpl, kdtree_hand, smpl_query_pts, query_idx, query_pts_len)
        dist_body, idx_body, sdf_body = self.query_sdf(sdf_scan, kdtree_body, smpl_query_pts, query_idx, query_pts_len)

        valid_indices = sdf_hand > -1.0
        alpha = dist_hand[valid_indices] / (dist_hand[valid_indices] + dist_body[valid_indices])
        sdf_body[valid_indices] = sdf_body[valid_indices] * alpha + sdf_hand[valid_indices] * (1 - alpha)

        sdf = sdf_body.reshape((res[0], res[1], res[2]))

        vertices, faces = marching_cubes(torch.Tensor(sdf), 0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        mesh = trimesh.Trimesh(new_vertices, faces)

        mesh = postprocess_mesh(mesh, num_faces=10000)
        kdtree_rgb = KDTree(scan_mesh.vertices, leaf_size=30, metric='euclidean')
        idx = kdtree_rgb.query(mesh.vertices, k=1, return_distance=False)
        vertex_colors = scan_mesh.visual.vertex_colors[idx.squeeze(1), :]
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.vertex_normals,
                               vertex_colors=vertex_colors, process=False)
        # mesh.show()
        print("> It took {:.2f}s seconds for changing full :-)".format(time.time() - start_t))

        return mesh

    def visualize_joints(self, joints):
        tmp = np.zeros((2048, 2048, 3))

        for i in range(joints.shape[0]):
            u, v = int(joints[i, 0]*4), int(joints[i, 1]*4)
            tmp = cv2.circle(tmp, (u, v), 3, [0, 0, 255], 1)
            tmp = cv2.circle(tmp, (u, v), 3, [0, 255, 0], 1)
            tmp = cv2.putText(tmp, str(i), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255) )
        cv2.imshow('projected joints', tmp)
        cv2.waitKey(0)

    def get_3D_keypoints(self, mesh, keypoints, cam_params, visualize=False):
        """
            Sample src for retrieve 3D joints from reconstructed human model.
            :param mesh: reconstructed mesh w.r.t. the camera center (origin)
            :param scene: openGL scene generated by the cam_params
            :param keypoints: SMPL-X keypoints converted from openpose or openpifpaf keypoints
            :param cam_params: camera parameters (intrinsic and extrinsic)
            :return: 144 x 3 matrix that contains 3D joints of 144 landmarks
        """
        # 0. update scene to the predefined camera parameters.
        scene = mesh.scene()
        scene.camera.focal = [cam_params.fx, cam_params.fy]
        scene.camera.resolution = [cam_params.width, cam_params.height]
        scene.camera.principal_x = cam_params.px
        scene.camera.principal_y = cam_params.py

        cam_center = np.asarray(cam_params.cmax)
        cam_trans = np.copy(scene.camera_transform)
        cam_trans[:3, 3] = np.asarray(cam_center)

        scene.camera_transform = cam_trans
        scene.camera.z_far = cam_params.far
        scene.camera.z_near = cam_params.near
        mesh.scene = scene

        # 1. make unit rays (z direction is negative)
        xyz = np.ones_like(keypoints[:, 0:3])
        xyz[:, 0] = (keypoints[:, 0] - cam_params.px) / cam_params.fx
        xyz[:, 1] = (keypoints[:, 1] - cam_params.py) / cam_params.fy
        xyz = -xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        xyz[:, 0] = -xyz[:, 0]
        origins = np.tile(cam_center, (keypoints.shape[0], 1))

        # 2. get depth from intersections
        pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(origins,
                                                                                   xyz,
                                                                                   multiple_hits=True)
        depth = trimesh.util.diagonal_dot(pers_points - origins[0].reshape(1, 3), xyz[pers_index_ray, :])

        depth_near = np.ones_like(keypoints[:, 0:1], dtype=float) * scene.camera.z_far
        depth_far = np.zeros_like(keypoints[:, 0:1], dtype=float)

        for i in range(len(pers_index_ray)):
            depth_near[pers_index_ray[i]] = min(depth_near[pers_index_ray[i]], depth[i])
            depth_far[pers_index_ray[i]] = max(depth_far[pers_index_ray[i]], depth[i])

        xyz_near, xyz_far = np.zeros_like(xyz), np.zeros_like(xyz)
        xyz_near[pers_index_ray, :] = xyz[pers_index_ray, :] * depth_near[pers_index_ray, :] + origins[0].reshape(-1, 3)
        xyz_far[pers_index_ray, :] = xyz[pers_index_ray, :] * depth_far[pers_index_ray, :] + origins[0].reshape(-1, 3)

        # 3. define 3D positions for keypoints (for openpifpaf)
        joints_3d = np.zeros_like(xyz)
        for i in pers_index_ray:
            if np.sum(xyz_near[i, :]) == cam_params.near:
                continue

            # we use 'smplx' indices for approximating 3D joint coordinates.
            if i <= 9:  # pelvis to spine
                joints_3d[i, :] = xyz_near[i, :]
                joints_3d[i, 2] = (xyz_far[i, 2] - xyz_near[i, 2]) / 2
            elif 10 <= i <= 11:  # foot
                joints_3d[i, :] = xyz_near[i, :]
            elif 13 <= i <= 14 or 16 <= i <= 21:  # collars and body
                joints_3d[i, :] = xyz_near[i, :]
                joints_3d[i, 2] = (xyz_far[i, 2] - xyz_near[i, 2]) / 2
            elif 22 <= i <= 24:  # jaw and eyes
                joints_3d[i, :] = xyz_near[i, :]
            elif i == 55:  # nose
                joints_3d[i, :] = xyz_near[i, :]
            elif 37 <= i <= 39 or 52 <= i <= 54:  # thumbs
                joints_3d[i, :] = xyz_near[i, :]
            elif 46 <= i <= 48 or 31 <= i <= 33:  # pinky
                joints_3d[i, :] = xyz_far[i, :]
            elif i == 62 or i == 65:  # heels
                joints_3d[i, :] = xyz_far[i, :]
            elif 60 <= i <= 61 or 63 <= i <= 64:  # toes
                joints_3d[i, :] = xyz_near[i, :]
            elif 76 <= i <= 143:  # face
                joints_3d[i, :] = xyz_near[i, :]

        if visualize:
            import open3d as o3d
            mesh_o3d = o3d.geometry.PointCloud()
            mesh_o3d.points = o3d.utility.Vector3dVector(joints_3d)
            o3d.visualization.draw_geometries([mesh_o3d])

        return torch.Tensor(joints_3d[None, :, :]).to(self.device)

    def get_camera_parameters(self):
        R, K, t, _, _ = self.renderer.cam.get_gl_matrix()
        P = np.eye(4, dtype=float)
        P[:3, :3] = R
        P[:3, 3] = t
        return K, R, t, P

    def smplify_alignment(self, mesh, smpl_params, keypoints2d,
                          K=None, R=None, t=None,
                          keypoints3d=None, loss_config='base', iters=500, lr=0.01, fix_all=False):
        self.smpl_handler.smpl_params.toggle_gradients(status=False)
        this_optimizer, this_scheduler = self.smpl_handler.set_optimizers(target=loss_config, lr=lr)

        src_vts = self.real2smpl(torch.Tensor(mesh.vertices).to(self.device))
        src_faces = torch.Tensor(self.smpl_handler.model.faces.astype(np.float64)).to(self.device)
        src_mesh = Meshes(verts=[src_vts], faces=[src_faces])

        if K is None or R is None or t is None:
            R_np, K_np, t_np, _, _ = self.renderer.cam.get_gl_matrix()
            K = torch.tensor(K_np[None, :, :]).float().to(self.device)
            R = torch.tensor(R_np[None, :, :]).float().to(self.device)
            t = torch.tensor(t_np[None, :]).float().to(self.device)
        else:
            K = torch.FloatTensor(K).to(self.device)
            R = torch.FloatTensor(R).to(self.device)
            t = torch.FloatTensor(t).to(self.device)

        keypoints2d = torch.tensor(keypoints2d).float().to(self.device)
        keypoints2d = self.smpl_handler.openpose2smpl(keypoints2d)
        if keypoints3d is not None:
            keypoints3d = torch.tensor(keypoints3d).float().to(self.device)
            keypoints3d = self.smpl_handler.openpose2smpl(keypoints3d)

        with tqdm(range(iters), ncols=120) as pbar:
            for itr in pbar:
                loss_all = 0.0
                this_optimizer.zero_grad()
                smpl_output = self.smpl_handler.forward_test(smpl_params, fix_all=fix_all)
                src_samples = sample_points_from_meshes(src_mesh, 50000, return_normals=False)
                loss_cf, _ = chamfer_distance(smpl_output.vertices, src_samples)

                if fix_all:
                    loss_all += loss_cf * 10.0
                else:
                    loss_all += loss_cf

                smpl_joints = smpl_output.joints
                mask = torch.ones_like(keypoints3d[:, :, 0:1])
                mask[:, torch.sum(keypoints3d, dim=2).squeeze() == 0, :] = 0
                joints_3d = self.real2smpl(keypoints3d)
                loss_jts = self.l1_loss(smpl_joints * mask, joints_3d * mask) * 0.05

                smpl_joints_real = self.smpl2real(smpl_joints)
                for k in range(R.shape[0]):
                    proj_joints = perspective_projection(smpl_joints_real,
                                                         R[k:k + 1, :, :], t[k:k + 1, :], K=K[k:k + 1, :, :])
                    gt_joints = perspective_projection(keypoints3d,
                                                       R[k:k + 1, :, :], t[k:k + 1, :], K=K[k:k + 1, :, :])
                    loss_jts += self.l1_loss(proj_joints * mask, gt_joints * mask)
                    if k == 10:
                        self.visualize_smpl_joints(proj_joints, gt_joints)
                loss_all += loss_jts * 0.1

                # smpl_joints = self.smpl2real(smpl_output.joints)
                # proj_joints = perspective_projection(smpl_joints, R, t, K=K)
                # proj_joints[keypoints2d == 0] = 0
                # loss_jts = self.joint_2d_loss(proj_joints, keypoints2d)
                # loss_all += loss_jts * 0.01
                # self.visualize_smpl_joints(proj_joints, keypoints2d)

                # neural renderer
                pbar.set_description('optimization proc.: iteration:{0}, loss:{1:0.5f}'.format(itr, loss_all))
                loss_all.backward()
                this_optimizer.step()
                this_scheduler.step(loss_all)

        smpl_output = self.smpl_handler.forward_test(smpl_params, fix_all=fix_all)
        smpl_vertices = self.smpl2real(smpl_output.vertices)
        smpl_mesh = trimesh.Trimesh(smpl_vertices.squeeze(0).detach().cpu().numpy(), self.smpl_handler.model.faces)

        return smpl_mesh, smpl_output

    def smplify_multiview(self, K_np, R_np, t_np, keypoints_2d, keypoints_3d, mesh,
                          loss_options='joint2d',
                          loss_config='base_shape_pose_hands_face',
                          smpl_params=None,
                          contraints='foot_spine', iters=500, lr=0.01):
        K = torch.tensor(K_np).float().to(self.device)
        R = torch.tensor(R_np).float().to(self.device)
        t = torch.tensor(t_np).float().to(self.device)

        # flip Rotation matrix.
        R_nr = R.clone().detach()
        # R_nr[:, 1, 1] *= -1
        # R_nr[:, 0, 2] *= -1
        # R_nr[:, 2, 2] *= -1
        # R_nr[:, 2, 0] *= -1

        # v-poser prior
        # from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
        # from human_body_prior.src.human_body_prior.tools.model_loader import load_model
        # from human_body_prior.src.human_body_prior.models.vposer_model import VPoser
        # bm = BodyModel(bm_fname='neutral').to(self.device)

        keypoints_2d = torch.tensor(keypoints_2d).float().to(self.device)
        keypoints2d = self.smpl_handler.openpose2smpl(keypoints_2d)
        if keypoints_3d is not None:
            keypoints_3d = torch.tensor(keypoints_3d).float().to(self.device)
            keypoints3d = self.smpl_handler.openpose2smpl(keypoints_3d)
            # left hand (for t-pose fitting with flat hands)
            # keypoints3d[0, 25:40, :] = 0
            # keypoints3d[0, 66:71, :] = 0
            # right hand
            # keypoints3d[0, 40:55, :] = 0
            # keypoints3d[0, 71:76, :] = 0

        this_optimizer, this_scheduler = self.smpl_handler.set_optimizers(target=loss_config, lr=lr)
        batch_size = keypoints2d.shape[0]
        src_vts = self.real2smpl(torch.Tensor(mesh.vertices).to(self.device))
        src_faces = torch.Tensor(mesh.faces.astype(np.float64)).to(self.device)
        src_mesh = Meshes(verts=[src_vts], faces=[src_faces])

        with tqdm(range(iters), ncols=120) as pbar:
            for itr in pbar:
                loss_all = 0.0
                this_optimizer.zero_grad()
                smpl_output = self.smpl_handler(model_constraints=contraints)
                # smpl_output = self.smpl_handler.forward_test(smpl_params)

                if 'chamfer' in loss_options:
                    src_samples = sample_points_from_meshes(src_mesh, 50000, return_normals=False)
                    loss_cf, _ = chamfer_distance(smpl_output.vertices, src_samples)
                    loss_all += loss_cf * 0.5

                if 'prior' in loss_options:
                    loss_prior_pose = MaxMixturePosePrior.pose_angle_prior(smpl_output.body_pose)
                    loss_prior_exp = torch.mean(smpl_output.expression**2)
                    loss_prior_beta = torch.mean(smpl_output.betas**2)
                    loss_all += loss_prior_beta * 0.8 + loss_prior_pose * 0.1 + loss_prior_exp * 0.1

                if 'symmetry' in loss_options:
                    # shoulder symmetry (to prevent distorting)
                    loss_symm = self.l1_loss(smpl_output.body_pose[:, 45:48], smpl_output.body_pose[:, 48:51])
                    # hands symmetry (for T-pose only)
                    # loss_symm += self.l1_loss(smpl_output.body_pose[:, 75:90], smpl_output.body_pose[:, 90:105])
                    loss_all += loss_symm * 0.3

                # multi-view 2D joint loss
                if 'joint2d' in loss_options:
                    smpl_joints = self.smpl2real(smpl_output.joints)
                    smpl_joints = smpl_joints.repeat([batch_size, 1, 1])
                    proj_joints = perspective_projection(smpl_joints, R_nr, t, K=K)
                    proj_joints[keypoints2d == 0] = 0
                    loss_jts = self.joint_2d_loss(proj_joints, keypoints2d)
                    loss_all += loss_jts * 0.1
                    # i = int(np.random.uniform(0, batch_size, 1))
                    # self.visualize_smpl_joints(proj_joints[i:i+1, :, :], keypoints2d[i:i+1, :, :])

                if 'joint3d' in loss_options and keypoints_3d is not None:
                    smpl_joints = smpl_output.joints
                    mask = torch.ones_like(keypoints3d[:, :, 0:1])
                    mask[:, torch.sum(keypoints3d, dim=2).squeeze() == 0, :] = 0
                    joints_3d = self.real2smpl(keypoints3d)
                    loss_jts = self.l1_loss(smpl_joints * mask, joints_3d * mask) * 0.05

                    smpl_joints_real = self.smpl2real(smpl_joints)
                    for k in range(R.shape[0]):
                        proj_joints = perspective_projection(smpl_joints_real,
                                                             R[k:k+1, :, :], t[k:k+1, :], K=K[k:k+1, :, :])
                        gt_joints = perspective_projection(keypoints3d,
                                                           R[k:k+1, :, :], t[k:k+1, :], K=K[k:k+1, :, :])
                        loss_jts += self.l1_loss(proj_joints * mask, gt_joints * mask)
                        if k == 0:
                            self.visualize_smpl_joints(proj_joints, gt_joints)
                    loss_all += loss_jts * 0.1

                if 'sil' in loss_options:
                    vertices_r = self.smpl2real(smpl_output.vertices)
                    vertices_g = torch.FloatTensor(mesh.vertices).unsqueeze(0).to(self.device)
                    faces_g = torch.FloatTensor(mesh.faces).unsqueeze(0).to(self.device)
                    faces_r = self.smpl_handler.model.faces_tensor.unsqueeze(0)

                    for k in range(10):
                        R_new = self.get_random_rotation_matrix()
                        mask = self.renderer(vertices_r, faces_r,
                                             K=K[0:1, :, :], R=R_new, t=-t[0:1, :],
                                             mode='silhouettes')
                        mask_gt = self.renderer(vertices_g,faces_g,
                                                K=K[0:1, :, :], R=R_new, t=-t[0:1, :],
                                                mode='silhouettes')
                        loss_all += self.bce_loss(mask, mask_gt)
                        # loss_all += self.l1_loss(mask, mask_gt)

                # neural renderer
                pbar.set_description('optimization proc.: iteration:{0}, loss:{1:0.5f}'.format(itr, loss_all))
                loss_all.backward()
                this_optimizer.step()
                this_scheduler.step(loss_all)

        smpl_output = self.smpl_handler()
        smpl_vertices = self.smpl2real(smpl_output.vertices)
        smpl_mesh = trimesh.Trimesh(smpl_vertices.squeeze(0).detach().cpu().numpy(), self.smpl_handler.model.faces)

        return smpl_mesh, smpl_output

    def forward_inference_single(self, img_data, keypoints=None, keypoints3d=None, init_params=None,
                                 path2save=None, pred_mesh=None, config='init'):
        """
            Returns smpl(-x) parameters and a mesh, gn
            :param img_data: (sample description)
            :param keypoints_openpose: (sample description)
            :param path2save: (sample description)
            :param pred_mesh: (sample description)
            :return: SMPL parameters and a mesh generated by the optimized parameters.
        """
        R_np, K_np, t_np, _, _ = self.renderer.cam.get_gl_matrix()
        K = torch.tensor(K_np[None, :, :]).float().to(self.device)
        R = torch.tensor(R_np[None, :, :]).float().to(self.device)
        t = torch.tensor(t_np[None, :]).float().to(self.device)

        if pred_mesh is not None:
            pred_vertices = torch.Tensor(pred_mesh.vertices[None, :, :]).to(self.device)
            pred_faces = torch.Tensor(pred_mesh.faces[None, :, :]).to(self.device)
        else:
            pred_vertices, pred_faces = None, None

        if keypoints is not None:
            keypoints = self.smpl_handler.openpose2smpl(keypoints[:, 0:2].to(self.device))

        if 'init' in config:
            # initial optimization options
            # lr = [0.05, 0.001, 0.001]
            # iters = [500, 500, 200]
            # opt_options = ['base_shape_pose',
            #                'base_pose',
            #                'pose']
            # loss_options = ['joint2d_all_prior',
            #                 'joint2d_all_prior',
            #                 'joint2d_all_prior']
            # if 'no_prior' in config:  # assume a standing person.
            #     model_constraints = ['spine_neck_knee_foot', '', '']
            # elif 'with_prior' in config:  # prior = pixie
            #     model_constraints = ['spine_neck_knee_foot', 'spine_neck_knee_foot', 'spine_neck_knee_foot']

            lr = [0.01, 0.001]
            iters = [500, 300]
            opt_options = ['base',
                           'base']
            loss_options = ['joint2d_all_prior',
                            'joint2d_all_prior']
            model_constraints = ['', '']
            # model_constraints = ['spine_neck_knee_foot', 'spine_neck_knee_foot', 'spine_neck_knee_foot']
        elif 'refine' in config:
            lr = [0.01, 0.001, 0.001]
            iters = [200, 300, 200]
            if 'no_prior' in config:  # without pixie
                opt_options = ['base_pose_shape',
                               'pose_shape_hands_face',
                               'pose']
                loss_options = ['chamfer_joint3d_symmetry',
                                'chamfer_joint3d_symmetry',
                                '']
                model_constraints = ['spine_neck_foot', '', '']
            elif 'with_prior' in config:  # prior = pixie
                # for refining 'RP', 'TH2.0' from GT labels
                # lr = [0.01, 0.001, 0.001]
                # iters = [200, 300, 200]
                # opt_options = ['base_pose_shape', 'pose_shape_hands_face', 'pose']
                # loss_options = ['chamfer_joint2d_joint3d',
                #                 'chamfer_joint2d_joint3d',
                #                 'chamfer_joint2d_joint3d']
                # model_constraints = ['foot', '', '']

                # for refining 'RP-D' from GT labels
                lr = [0.01, 0.0001]
                iters = [300, 100]
                opt_options = ['base', 'shape_pose']  # for refine
                loss_options = ['chamfer_joint2d_joint3d',
                                'chamfer_joint2d_joint3d']
                model_constraints = ['foot', '']
            pred_vertices = self.real2smpl(pred_vertices)

        pose_prior = MaxMixturePosePrior(device=self.device)
        if init_params is not None:
            if isinstance(init_params, dict):
                self.smpl_handler.smpl_params.set_params_from_dict(init_params, device=self.device)
            else:
                self.smpl_handler.smpl_params.set_params(init_params)
            # self.smpl_handler.smpl_params.set_params_motion(init_params)

        for k in range(len(opt_options)):
            self.optimize_smplx(pred_vertices, img_data, pred_faces=pred_faces,
                                opt_options=opt_options[k], loss_options=loss_options[k], K=K, R=R, t=t,
                                keypoints=keypoints, keypoints3d=keypoints3d, mask_gt=None, iters=iters[k],
                                model_constraints=model_constraints[k], pose_prior=pose_prior, lr=lr[k])
        # only float64 can be serialized and stored as json file
        smpl_params = self.smpl_handler.get_smpl_params()
        smpl_params['centroid_real'] = list(np.array(self.centroid.detach().cpu(), dtype=np.float64))
        smpl_params['scale_real'] = list(np.array([self.scale], dtype=np.float64))
        smpl_output = self.smpl_handler()
        vertices_real = self.smpl2real(smpl_output.vertices)
        smpl_trimesh = trimesh.Trimesh(vertices_real.detach().squeeze().cpu().numpy(),
                                       self.smpl_handler.model.faces, use_embree=True)
        # smpl_trimesh = trimesh.smoothing.filter_laplacian(smpl_trimesh.subdivide().subdivide())

        if path2save is not None:
            with open(path2save, 'w') as fp:
                json.dump(smpl_params, fp, indent="\t")

            smpl_trimesh.export(path2save.replace('.json', '.obj'))
        smpl_params = self.smpl_handler.get_smpl_params_torch(idx=0)
        smpl_params = Namespace(**smpl_params)
        return smpl_params, smpl_trimesh

    def visualize_meshes(self, vertices_s, mesh_data):
        smplout = self.smpl_handler()
        smpl_mesh = trimesh.Trimesh(smplout.vertices.detach().cpu().squeeze(0).numpy(), self.smpl_handler.model.faces)
        scan_mesh = trimesh.Trimesh(vertices_s.detach().cpu().numpy(),
                                    mesh_data['faces'].detach().squeeze().cpu().numpy())
        show_meshes([smpl_mesh, scan_mesh])

    def check_existence(self, data_name):
        filename = os.path.join(self.save_path, 'SMPLX', data_name)
        save_filename = os.path.join(filename, 'smplx_params.json')
        save_objname = save_filename.replace('_params.json', '_mesh.obj')
        if os.path.isfile(save_objname) and os.path.isfile(save_objname):
            print('skipping current images')
            return True
        else:
            return False

    # fit smpl for multiple images with gt scan.
    def forward(self, img_data, mesh_data, data_name=None, initial_params=None):
        from pytorch3d.structures import Meshes
        from pytorch3d.ops import sample_points_from_meshes

        n = self.smpl_handler.num_models
        if initial_params is not None and len(initial_params) == n:
            n = len(initial_params)
            for i in range(n):
                self.smpl_data.set_initial_params(initial_params[i], idx=i)

        filename = os.path.join(self.save_path, 'SMPLX', data_name)
        os.makedirs(filename, exist_ok=True)
        save_filename = os.path.join(filename, 'smplx_params.json')
        save_objname = save_filename.replace('_params.json', '_mesh.obj')

        K, R, t, keypoints, mask_gt = self.parse_data(img_data)
        if K is None:
            print('No camera parameters')
            return None

        centroid, scale = self.get_normalize_params(mesh_data['vertices'], mesh_data['faces'])
        centroid_tensor = torch.Tensor(centroid).to(self.device)

        vertices_s = self.real2smpl(mesh_data['vertices'], centroid_tensor, scale)  # smpl coordinate
        scan_mesh = Meshes(
            verts=[vertices_s],
            faces=[mesh_data['faces'].squeeze()])
        resampled_mesh = sample_points_from_meshes(scan_mesh, 10475)

        # initial optimization
        opt_options = 'base_pose_shape'
        loss_options = 'joints_all_chamfer'
        self.optimize_smplx(resampled_mesh, img_data, centroid_tensor, scale,
                            opt_options=opt_options, loss_options=loss_options,
                            K=K, R=R, t=t, keypoints=keypoints, mask_gt=mask_gt,
                            lr=0.01, iters=100)

        opt_options = 'base_pose_shape'
        loss_options = 'joints_all_chamfer_sil'
        self.optimize_smplx(resampled_mesh, img_data, centroid_tensor, scale,
                            opt_options=opt_options, loss_options=loss_options,
                            K=K, R=R, t=t, keypoints=keypoints, mask_gt=mask_gt,
                            lr=0.01, iters=10)

        # # initial optimization
        opt_options = 'base_pose_hands_face'
        loss_options = 'joints_all_chamfer_sil'
        self.optimize_smplx(resampled_mesh, img_data, centroid_tensor, scale,
                            opt_options=opt_options, loss_options=loss_options,
                            K=K, R=R, t=t, keypoints=keypoints, mask_gt=mask_gt,
                            lr=0.001, iters=100)

        # visualize fitted meshes.
        verbose = False
        if verbose:
            self.visualize_meshes(vertices_s, mesh_data)
        smpl_params = self.smpl_handler.get_smpl_params()  # fetch optimized parameters.
        smpl_params['centroid_real'] = list(centroid)
        smpl_params['scale_real'] = scale

        with open(save_filename, 'w') as fp:
            json.dump(smpl_params, fp, indent="\t")

        smpl_output = self.smpl_handler()
        vertices_r = self.smpl2real(smpl_output.vertices, centroid_tensor, scale)
        vertices_r = vertices_r / mesh_data['scale'] + torch.Tensor(mesh_data['center']).to(self.device).reshape(-1, 3)
        smpl_trimesh = trimesh.Trimesh(vertices_r.squeeze().detach().cpu().numpy(), self.smpl_handler.model.faces)
        # if self.path2save is not None:
        #     smpl_trimesh.export(save_objname)


def postprocess_mesh(mesh, num_faces=None):
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


if __name__ == '__main__':
    dataset = 'Famoz2023'
    input_path = '/data/DATASET_Canonical/M96'
    save_path = '/data/DATASET_Canonical/M96_OUT'

    with open('./camera_config.yaml') as f:
        cam_param = yaml.load(f, Loader=yaml.FullLoader)
        cam_param = cam_param[dataset]

    config = Configurator()
    params = config.parse()

    smpl_optimizer = SmplOptimizer(cam_param=cam_param,
                                   params=params)
