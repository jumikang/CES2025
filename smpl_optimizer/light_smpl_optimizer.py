# from __future__ import annotations
import os
import cv2
import smplx
import torch
import yaml
import json
import trimesh
import pickle
import numpy as np
from pytorch3d.loss import chamfer_distance
try:
    from pytorch3d.loss import chamfer_distance
except ModuleNotFoundError:
    need_pytorch3d=True

# from chamferdist import ChamferDistance
from src.smpl_optimizer.keypoints_mapper import dset_to_body_model
from src.lib.utils.geometry import perspective_projection
from torch import nn
from tqdm import tqdm
from src.smpl_optimizer.smpl_utils import (keypoint_loader,
                                           init_semantic_labels,
                                           get_3d_from_2d_keypoints,
                                           get_near_and_far_points)
from src.lib.utils.visualizer import trimesh2o3d


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


class LightSMPLOptimizer(nn.Module):
    def __init__(self,
                 cam_params=None,
                 path2config='confs/smpl_config.yaml',
                 smpl_root='./data/smplx',
                 gender=None,
                 device='cuda:0'):
        super(LightSMPLOptimizer, self).__init__()

        # set general parameters
        self.device = device

        # set smpl related parameters
        with open(path2config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            smpl_config = config['DEFAULT']

        self.smpl_conf = smpl_config
        if gender is not None:
            self.smpl_conf['gender'] = gender

        self.model_path = smpl_root
        self.regressor = os.path.join(smpl_root, smpl_config['regressor'])

        use_semantic = True
        if use_semantic:
            path2semantic = os.path.join(smpl_root, smpl_config['segmentation'])
            if os.path.exists(path2semantic):
                self.v_label = init_semantic_labels(path2semantic)

        # set cam parameters
        self.K = np.eye(3)
        self.K[0, 0], self.K[1, 1] = cam_params['fx'], cam_params['fy']
        self.K[0, 2], self.K[1, 2] = cam_params['px'], cam_params['py']
        self.R = np.eye(3)
        self.R[1, 1] *= -1.0
        self.R[2, 2] *= -1.0
        self.t = np.array(cam_params['cam_center']) / 100.0
        self.K = torch.FloatTensor(self.K[None, :, :]).to(self.device)
        self.R = torch.FloatTensor(self.R[None, :, :]).to(self.device)
        self.t = torch.FloatTensor(self.t[None, :]).to(self.device)
        self.width = cam_params['width']
        self.height = cam_params['height']

        # set 2d-pose converter (default is openpose <-> smplx)
        self.pose_format = smpl_config['pose_format']
        if self.pose_format == 'openpose':
            self.pose_idx, self.smpl_idx = dset_to_body_model(
                dset='openpose25+hands+face', model_type='smplx',
                use_hands=True, use_face=True, use_face_contour=True)
        elif self.pose_format == 'openpifpaf':
            self.pose_idx, self.smpl_idx = dset_to_body_model(
                dset='openpifpaf+wholebody', model_type='smplx',
                use_hands=True, use_face=True, use_face_contour=True)

        self.smpl_model = self.init_smpl_model().to(self.device)

    def init_smpl_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        return smplx.create(self.model_path,
                            model_type=self.smpl_conf['model_type'],
                            gender=self.smpl_conf['gender'],
                            num_betas=self.smpl_conf['num_betas'],
                            ext=self.smpl_conf['ext'],
                            use_face_contour=self.smpl_conf['use_face_contour'],
                            flat_hand_mean=self.smpl_conf['use_flat_hand'],
                            use_pca=self.smpl_conf['use_pca'],
                            num_pca_comps=self.smpl_conf['num_pca_comp']
                            )

    @staticmethod
    def get_empty_params():
        return {'global_orient': torch.zeros(1, 3),
                'expression': torch.zeros(1, 10),
                'betas': torch.zeros(1, 10),
                'jaw_pose': torch.zeros(1, 3),
                'body_pose': torch.zeros(1, 63),
                'left_hand_pose': torch.zeros(1, 45),
                'right_hand_pose': torch.zeros(1, 45),
                'transl': torch.zeros(1, 3),
                'scale': torch.FloatTensor([[1.1]])
                }

    def openpose2smpl(self, keypoints):
        """
            mapping 2d keypoints to the smpl-x format
            :param keypoints: 2d or 3d keypoints (support openpose and mediapifpaf formats)
                              torch.Tensor
            :return torch.Tensor of size 1x144x3: converted keypoints to the SMPL-X format
        """
        if len(keypoints.shape) == 2:
            d = keypoints.shape[1]
            keypoints_new = torch.zeros((144, d), device=keypoints.device)
            keypoints_new[self.smpl_idx, :] = keypoints[self.pose_idx, :]
            return keypoints_new.unsqueeze(0)
        elif len(keypoints.shape) == 3:
            d = keypoints.shape[2]
            keypoints_new = torch.zeros((keypoints.shape[0], 144, d), device=keypoints.device)
            keypoints_new[:, self.smpl_idx, :] = keypoints[:, self.pose_idx, :]
            return keypoints_new
        else:
            assert "check dimension"

    def set_openpose2smpl_keypoints(self, path2pose=None, keypoints=None):
        if keypoints is None:
            keypoints = keypoint_loader(path2pose, pose_detector=self.pose_format)
        keypoints_smplx = self.openpose2smpl(keypoints.to(self.device))

        # thresholding (ignore uncertain joints?)
        return keypoints_smplx

    def visualize_smpl_joints(self, proj_joints, joints, image=None):
        width, height = 640, 640
        ratio_x = width / self.width
        ratio_y = height / self.height
        if image is not None:
            tmp = image.copy()
            tmp = cv2.resize(tmp, (640, 640))
        else:
            tmp = np.zeros((640, 640, 3))

        for i in range(proj_joints.shape[1]):
            pidx = (proj_joints[:, i, :])
            jidx = (joints[:, i, :])
            if int(pidx[0, 1]) < self.width and int(pidx[0, 0]) < self.height and \
                    int(jidx[0, 1]) < self.width and int(jidx[0, 0]) < self.height:
                tmp = cv2.circle(tmp, (int(pidx[0, 0] * ratio_x), int(pidx[0, 1] * ratio_y)), 3,
                                 [0, 0, 255], 1)
                tmp = cv2.circle(tmp, (int(jidx[0, 0] * ratio_x), int(jidx[0, 1] * ratio_y)), 3,
                                 [0, 255, 0], 1)
        cv2.imshow('projected joints', tmp)
        cv2.waitKey(1)

    def set_pixie(self, params):
        output = self.get_empty_params()
        for key in params:
            output[key] = params[key].clone()
        for key in output:
            output[key] = output[key].to(self.device)
            output[key].requires_grad = True
        return output

    def get_3d_keypoints(self, mesh, keypoints):
        joints_3d = get_3d_from_2d_keypoints(keypoints[0], mesh, self.K[0], self.R[0], self.t[0])
        joints_3d = joints_3d.to(self.device)
        mask = torch.ones_like(joints_3d[:, :, 0:1]).to(self.device)
        mask[:, torch.sum(joints_3d, dim=2).squeeze() == 0, :] = 0
        return joints_3d * mask

    def get_3d_keypoints_near_far(self, mesh, keypoints):
        joints_near, joints_far = get_near_and_far_points(keypoints, mesh, self.K[0], self.R[0], self.t[0])

        mask = np.ones_like(joints_far)
        mask[torch.sum(keypoints.cpu(), dim=1) == 0, :] = 0
        return joints_near * mask, joints_far * mask

    def pipeline_data_generation(self, mesh, pose=None, pixie=None, gt_smpl=None, image=None, save_results=None, keypoints=None):
        if pixie is not None and image is not None:
            pixie_output = pixie(image)
        elif gt_smpl is not None:
            pixie_output = gt_smpl
        else:
            pixie_output = None

        vertices = mesh.vertices
        keypoints = self.set_openpose2smpl_keypoints(path2pose=pose, keypoints=keypoints)
        keypoints_3d = self.get_3d_keypoints(mesh, keypoints)
        smpl_params = self.smpl_optimizer(pixie=pixie_output,
                                          opt_config='phase3',
                                          keypoints3d=keypoints_3d,
                                          vertices=torch.FloatTensor(vertices).unsqueeze(0).cuda(),
                                          keypoints=keypoints, iters=300, lr=0.1, verbose=True)

        smpl_params = self.smpl_optimizer(pixie=smpl_params,
                                          opt_config='phase4',
                                          keypoints3d=keypoints_3d,
                                          vertices=torch.FloatTensor(vertices).unsqueeze(0).cuda(),
                                          keypoints=keypoints, iters=300, lr=0.1, verbose=True)
        _, smpl_mesh = self.forward(smpl_params, return_mesh=True)

        if save_results is not None:
            smpl_mesh.export(save_results["mesh"])
            for key in smpl_params:
                smpl_params[key] = smpl_params[key].detach().cpu().numpy().tolist()
            with open(save_results["params"], 'w') as f:
                json.dump(smpl_params, f, indent=4)

        return smpl_params, smpl_mesh

    def pipeline(self, mesh, pose=None, pixie=None, gt_smpl=None, image=None, save_results=None, keypoints=None):
        if pixie is not None and image is not None:
            pixie_output = pixie(image)
        elif gt_smpl is not None:
            pixie_output = gt_smpl
        else:
            pixie_output = None

        vertices = mesh.vertices
        keypoints = self.set_openpose2smpl_keypoints(path2pose=pose, keypoints=keypoints)
        keypoints_3d = self.get_3d_keypoints(mesh, keypoints)
        smpl_params = self.smpl_optimizer(pixie=pixie_output,
                                          opt_config='phase1',
                                          keypoints3d=keypoints_3d,
                                          vertices=torch.FloatTensor(vertices).unsqueeze(0).cuda(),
                                          keypoints=keypoints, iters=70, lr=0.1, verbose=True)
        smpl_params = self.smpl_optimizer(smpl_params=smpl_params,
                                          opt_config='phase2',
                                          keypoints3d=keypoints_3d,
                                          vertices=torch.FloatTensor(vertices).unsqueeze(0).cuda(),
                                          keypoints=keypoints, iters=100, lr=0.1, verbose=True)

        _, smpl_mesh = self.forward(smpl_params, return_mesh=True)

        if save_results is not None:
            smpl_mesh.export(save_results["mesh"])
            for key in smpl_params:
                smpl_params[key] = smpl_params[key].detach().cpu().numpy().tolist()
            with open(save_results["params"], 'w') as f:
                json.dump(smpl_params, f, indent=4)

        return smpl_params, smpl_mesh

    def smpl_optimizer(self,
                       smpl_params=None,
                       pixie=None,
                       keypoints=None,
                       keypoints3d=None,
                       opt_config='global_body_pose',
                       vertices=None,
                       iters=500,
                       lr=0.01,
                       verbose=False):

        if pixie is not None:
            smpl_params = self.set_pixie(pixie)
        elif smpl_params is not None:
            for key in smpl_params:
                smpl_params[key].requires_grad = True
        else:
            smpl_params = self.get_empty_params()
            for key in smpl_params:
                smpl_params[key] = smpl_params[key].to(self.device)
                smpl_params[key].requires_grad = True

        l2_loss = nn.MSELoss()

        opt_params = list()
        if opt_config == 'phase1':
            opt_params.append(smpl_params['global_orient'])
            opt_params.append(smpl_params['transl'])
            opt_params.append(smpl_params['scale'])
            opt_params.append(smpl_params['body_pose'])
            constraint = 'knee_foot_spine_wrist'
        elif opt_config == 'phase2':
            opt_params.append(smpl_params['global_orient'])
            opt_params.append(smpl_params['body_pose'])
            opt_params.append(smpl_params['transl'])
            opt_params.append(smpl_params['scale'])
            # opt_params.append(smpl_params['betas'])
            constraint = 'knee_foot_spine_wrist'
        elif opt_config == 'phase3':  # no contraint
            opt_params.append(smpl_params['global_orient'])
            opt_params.append(smpl_params['transl'])
            opt_params.append(smpl_params['scale'])
            constraint = ''
        elif opt_config == 'phase4':  # no contraint
            opt_params.append(smpl_params['global_orient'])
            opt_params.append(smpl_params['body_pose'])
            opt_params.append(smpl_params['transl'])
            opt_params.append(smpl_params['scale'])
            opt_params.append(smpl_params['betas'])
            constraint = ''

        # constraint = ''
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # chamfer_distance = ChamferDistance()

        with tqdm(range(iters), ncols=170) as pbar:
            for itr in pbar:
                optimizer.zero_grad()
                smpl_output = self.forward(smpl_params, model_constraints=constraint)
                proj_joints = perspective_projection(smpl_output.joints, self.R, self.t, K=self.K)
                proj_joints[keypoints[:, :, :2] == 0] = 0

                # l2 loss for 2d keypoints
                loss = l2_loss(proj_joints, keypoints[:, :, :2]) * 0.001  # image space is way larger
                if keypoints3d is not None:
                    mask = torch.ones_like(keypoints3d)
                    mask[keypoints3d == 0] = 0
                    loss += l2_loss(smpl_output.joints * mask, keypoints3d) * 0.05

                # chamfer distance
                if vertices is not None:
                    loss_chamfer, loss_normal = chamfer_distance(smpl_output.vertices, vertices)
                    # loss_chamfer = chamfer_distance(smpl_output.vertices, vertices)
                    loss += loss_chamfer * 100.0

                # loss += torch.mean(smpl_params['betas']) * 0.01

                loss += l2_loss(smpl_output.left_hand_pose, smpl_output.right_hand_pose) * 0.1
                loss_prior_pose = MaxMixturePosePrior.pose_angle_prior(smpl_output.body_pose)
                loss += loss_prior_pose * 0.1

                pbar.set_description('iter:{0}, loss:{1:0.5f}'.format(itr, loss))
                if verbose:
                    self.visualize_smpl_joints(proj_joints, keypoints[:, :, :2])

                loss.backward()
                optimizer.step()
                scheduler.step()

        for key in smpl_params.keys():
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].detach()

        # to use in the following modules.
        smpl_params['full_pose'] = smpl_output.full_pose.detach().clone()  # to canonicalize the mesh
        smpl_params['joints'] = smpl_output.joints.detach().clone()  # to refine postures.
        smpl_params['joints_2d'] = proj_joints.detach().clone()
        return smpl_params

    def get_idx(self, target='arms'):
        # return static vertices e.g., not in arms
        pass

    def optimize_arap(self, mesh, target='arms', max_iter=50):
        # where to optimize
        import open3d as o3d
        mesho3d = trimesh2o3d(mesh)
        vertices = np.asarray(mesh.vertices)
        static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
        static_pos = []
        for id in static_ids:
            static_pos.append(vertices[id])

        handle_ids = [2490]  # keypoints
        handle_pos = [vertices[2490] + np.array((-40, -40, -40))]
        constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
        constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh_prime = mesho3d.deform_as_rigid_as_possible(constraint_ids,
                                                             constraint_pos,
                                                             max_iter=max_iter)
        return mesh_prime

    def save_smpl_params(self, path2save, save_mesh=False):
        pass

    def forward(self, smpl_params, return_mesh=False, model_constraints=''):
        mask = torch.ones_like(smpl_params['body_pose']).to(self.device)
        mask.requires_grad_(False)

        if model_constraints is not None:
            if 'spine' in model_constraints:
                mask[:, 6:9] = 0  # spine 1
                mask[:, 15:18] = 0  # spine 2
                mask[:, 24:27] = 0  # spine 3
            if 'neck' in model_constraints:
                mask[:, 33:36] = 0  # neck
            if 'knee' in model_constraints:
                mask[:, 9:15] = 0  # knees
            if 'foot' in model_constraints:
                mask[:, 27:33] = 0  # foot
                mask[:, 18:24] = 0  # ankle
            if 'elbow' in model_constraints:
                mask[:, 51:57] = 0  # elbow
            if 'wrist' in model_constraints:
                mask[:, 57:63] = 0  # foot

        smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                      expression=smpl_params['expression'],
                                      body_pose=smpl_params['body_pose'] * mask,
                                      betas=smpl_params['betas'],
                                      global_orient=smpl_params['global_orient'],
                                      jaw_pose=smpl_params['jaw_pose'],
                                      left_hand_pose=smpl_params['left_hand_pose'],
                                      right_hand_pose=smpl_params['right_hand_pose'],
                                      return_full_pose=True,
                                      return_verts=True)

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            smpl_output.joints = smpl_output.joints * smpl_params['scale']
            smpl_output.vertices = smpl_output.vertices * smpl_params['scale']

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(),
                                        faces=self.smpl_model.faces, process=False)
            return smpl_output, smpl_mesh
        else:
            return smpl_output
