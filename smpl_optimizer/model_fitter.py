from __future__ import annotations
import smplx
import json
import numpy as np
import torch
import trimesh
import os
from torch import nn
from smpl_optimizer.keypoints_mapper import dset_to_body_model


class SmplParams:
    def __init__(self):
        self.betas = []
        self.body_pose = []
        self.expression = []
        self.global_orient = []
        self.left_hand_pose = []
        self.right_hand_pose = []
        self.jaw_pose = []
        self.transl = []
        self.scale = []
        self.num_models = 0
        self.prior = 'refine'
        # self.use_expression = []
        # self.use_pca = []
        # self.use_contour = []
        # self.flat_hands = []
        # currently we consider full models (with contour, hands, face)
        # self.use_face_contour = True

    def add_params(self, params, height_prior=0.0, scale_prior=0.9):
        """
        :param params:
        :param height_prior:
        :param scale_prior:
        """
        self.betas.append(nn.Parameter(torch.zeros_like(params.betas)))
        self.body_pose.append(nn.Parameter(torch.zeros_like(params.body_pose)))
        self.expression.append(nn.Parameter(torch.zeros_like(params.expression)))
        self.global_orient.append(nn.Parameter(torch.zeros_like(params.global_orient)))
        self.left_hand_pose.append(nn.Parameter(torch.zeros_like(params.left_hand_pose)))
        self.right_hand_pose.append(nn.Parameter(torch.zeros_like(params.right_hand_pose)))

        if self.betas[0].get_device() >= 0:
            device = 'cuda:' + str(self.betas[0].get_device())
        else:
            device = 'cpu'
        self.transl.append(nn.Parameter(torch.tensor([[0.0, height_prior, 0.0]]).float().to(device)))
        self.jaw_pose.append(nn.Parameter(torch.zeros_like(params.jaw_pose)))
        self.scale.append(nn.Parameter(torch.tensor(scale_prior).float().to(device)))
        self.num_models += 1

    def toggle_gradients(self, status=False):
        """
        toggle all the parameters to be optimizable or not (change required_grad value)
        :param status: True or False value
        """
        for i in range(self.num_models):
            self.betas[i].requires_grad = status
            self.body_pose[i].requires_grad = status
            self.expression[i].requires_grad = status
            self.global_orient[i].requires_grad = status
            self.left_hand_pose[i].requires_grad = status
            self.right_hand_pose[i].requires_grad = status
            self.transl[i].requires_grad = status
            self.jaw_pose[i].requires_grad = status
            self.scale[i].requires_grad = status

    def reset_for_rigging(self, idx=0):
        """
        :param idx:
        """
        # assert idx >= self.num_models, 'idx exceeds the number of initialized SMPL models'

        self.body_pose[idx] = self.body_pose[idx].detach()
        self.expression[idx] = self.expression[idx].detach()
        self.jaw_pose[idx] = self.jaw_pose[idx].detach()
        self.left_hand_pose[idx] = self.left_hand_pose[idx].detach()
        self.right_hand_pose[idx] = self.right_hand_pose[idx].detach()
        self.transl[idx] = self.transl[idx].detach()
        self.global_orient[idx] = self.global_orient[idx].detach()
        self.body_pose[idx][:] = 0
        self.expression[idx][:] = 0
        self.jaw_pose[idx][:] = 0
        self.left_hand_pose[idx][:] = 0
        self.right_hand_pose[idx][:] = 0
        self.scale[idx] = nn.Parameter(torch.tensor(1.0).float())
        self.transl[idx][:] = 0
        self.global_orient[idx][:] = 0
        # self.betas[idx][:]
        # self.global_orient[idx]  unchanged

    def set_params(self, params, idx=0, use_pca=True):
        # assert idx >= self.num_models, 'idx exceeds the number of initialized SMPL models'
        if 'betas' in params:
            self.betas[idx] = params.betas.detach()
        if 'body_pose' in params:
            self.body_pose[idx] = params.body_pose.detach()
        if 'expression' in params:
            self.expression[idx] = params.expression.detach()
        if 'global_orient' in params:
            self.global_orient[idx] = params.global_orient.detach()
        if 'left_hand_pose' in params:
            self.left_hand_pose[idx] = params.left_hand_pose.detach()
        if 'right_hand_pose' in params:
            self.right_hand_pose[idx] = params.right_hand_pose.detach()
        if 'jaw_pose' in params:
            self.jaw_pose[idx] = params.jaw_pose.detach()
        if 'transl' in params:
            self.transl[idx] = params.transl.detach()
        if 'scale' in params:
            self.scale[idx] = params.scale.detach()

    def set_params_from_dict(self, params, idx=0, device='cuda'):
        # assert idx >= self.num_models, 'idx exceeds the number of initialized SMPL models'
        if 'betas' in params:
            self.betas[idx] = torch.FloatTensor(params['betas']).to(device)
        if 'body_pose' in params:
            self.body_pose[idx] = torch.FloatTensor(params['body_pose']).to(device)
        if 'expression' in params:
            self.expression[idx] = torch.FloatTensor(params['expression']).to(device)
        if 'global_orient' in params:
            self.global_orient[idx] = torch.FloatTensor(params['global_orient']).to(device)
        if 'left_hand_pose' in params:
            self.left_hand_pose[idx] = torch.FloatTensor(params['left_hand_pose']).to(device)
        else:
            params.left_hand_pose = nn.Parameter(torch.zeros(1, 45)).to(device)
        if 'right_hand_pose' in params:
            self.right_hand_pose[idx] = torch.FloatTensor(params['right_hand_pose']).to(device)
        else:
            params.right_hand_pose = nn.Parameter(torch.zeros(1, 45)).to(device)

        if 'transl' in params:
            self.transl[idx] = torch.FloatTensor(params['transl']).unsqueeze(0).to(device)
        if 'jaw_pose' in params:
            self.jaw_pose[idx] = torch.FloatTensor(params['jaw_pose']).to(device)
        if 'scale' in params:
            self.scale[idx] = torch.FloatTensor(params['scale']).to(device)

    def set_params_motion(self, motion_dict, idx=0):
        """
        :param motion_dict:
        :param idx:
        :return:
        """
        if 'body_pose' in motion_dict:
            self.body_pose[idx] = motion_dict['body_pose']
        if 'global_orient' in motion_dict:
            self.global_orient[idx] = motion_dict['global_orient']
        if 'left_hand_pose' in motion_dict:
            self.left_hand_pose[idx] = motion_dict['left_hand_pose']
        if 'right_hand_pose' in motion_dict:
            self.right_hand_pose[idx] = motion_dict['right_hand_pose']
        if 'transl' in motion_dict:
            self.transl[idx] = motion_dict['transl']
        # self.expression[idx] = nn.Parameter(torch.tensor([[5.5, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        # self.left_hand_pose[idx] = nn.Parameter(torch.tensor([[-0.5, -0.5, -0.1, -1.0, -0.5, -0.1]]))
        # self.right_hand_pose[idx] = nn.Parameter(torch.tensor([[-0.5, -0.5, -0.1, -1.0, -0.5, -0.1]]))

    def remove_params(self, idx=0):
        """
        :param idx:
        :return:
        """
        self.betas.pop(idx)
        self.body_pose.pop(idx)
        self.expression.pop(idx)
        self.global_orient.pop(idx)
        self.left_hand_pose.pop(idx)
        self.right_hand_pose.pop(idx)
        self.transl.pop(idx)
        self.jaw_pose.pop(idx)
        self.scale.pop(idx)
        self.num_models -= 1


# Constraints should be added.
class SmplHandler(nn.Module):
    def __init__(self, model_path,
                 model_type='smplx',
                 gender='neutral',
                 age='adult',
                 use_pca=False,
                 num_pca_comp=6,
                 num_models=1,
                 flat_hands=True,
                 num_betas=10,
                 pose_detector='openpose',
                 device='cuda:0'):
        super(SmplHandler, self).__init__()
        self.device = device
        self.smpl_count = 0
        self.smpl_params = SmplParams()
        self.pose_format = pose_detector
        self.use_pca = use_pca
        self.flat_hands = flat_hands
        self.num_pca_comp = num_pca_comp
        self.age = age

        self.num_betas = num_betas
        self.model_path = model_path
        self.model_type = model_type
        self.gender = gender
        self.model = self.set_smpl_model()

        self.num_models = num_models
        for _ in range(self.num_models):
            self.smpl_params.add_params(self.model)

        # set center and scale parameters for the template mesh.
        v_pose_smpl = trimesh.Trimesh(self.model.v_template.cpu(),
                                      self.model.faces)
        self.centroid = v_pose_smpl.bounding_box.centroid
        self.centroid_tensor = torch.Tensor(self.centroid.copy()).to(self.device).float()
        self.scale = 2.0 / np.max(v_pose_smpl.bounding_box.extents)

        # self.average_height = 1.0
        # self.centroid_tensor = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
        # self.scale = 1 / (self.average_height / 2)

        # semantic labels of SMPL model
        self.use_semantic = False
        if self.use_semantic:
            self.body_seg_path = 'path/to/file'
            with open(self.body_seg_path, "r") as f:
                self.body_seg_data = json.load(f)
            self.seg_items = self.segIdx_set(self.body_seg_data)

        # keypoints to smpl joints mapper
        if self.pose_format == 'openpose':
            self.pose_idx, self.smpl_idx = dset_to_body_model(
                dset='openpose25+hands+face', model_type='smplx',
                use_hands=True, use_face=True, use_face_contour=True)
        elif self.pose_format == 'openpifpaf':
            self.pose_idx, self.smpl_idx = dset_to_body_model(
                dset='openpifpaf+wholebody', model_type='smplx',
                use_hands=True, use_face=True, use_face_contour=True)

    def set_smpl_model(self):
        """
            create smpl-x instance
            :return: a smpl instance
        """
        if self.age == 'adult':
            return smplx.create(self.model_path,
                                model_type=self.model_type,
                                gender=self.gender,
                                num_betas=self.num_betas,
                                ext='npz',
                                use_face_contour=True,
                                flat_hand_mean=self.flat_hands,
                                use_pca=self.use_pca,
                                num_pca_comps=self.num_pca_comp).to(self.device)
        else:  # 'kid'
            return smplx.create(self.model_path,
                                model_type=self.model_type,
                                gender=self.gender,
                                num_betas=self.num_betas,
                                ext='npz',
                                use_face_contour=True,
                                flat_hand_mean=True,
                                age='kid',
                                kid_template_path=os.path.join(self.model_path,
                                                               self.model_type,
                                                               self.model_type + '_kid_template.npy'),
                                use_pca=self.use_pca,
                                num_pca_comps=self.num_pca_comp).to(self.device)

    def add_params(self):
        """
            add a new set of parameters (it increases the number of avatars internally)
        """
        self.smpl_params.add_params(self.model)

    def reset_params(self, height_prior=0.0, scale_prior=1.2):
        """
            add a new set of parameters (it increases the number of avatars internally)
            :param height_prior: a prior position (0.0 means that the camera is looking at the center of mass)
                   increase or decrease this value to locate the initial position depending on the position of actual cameras
            :param scale_prior: estimate the scale of humans in addition to the shape parameters
        """
        self.smpl_params.remove_params()
        self.model = self.set_smpl_model()
        self.smpl_params.add_params(self.model, height_prior=height_prior, scale_prior=scale_prior)

    def get_faces(self):
        """
            a getter function for SMPL-X model faces
            :return: SMPL-X faces
        """
        return self.model.faces

    def set_params_from_dict(self, smpl_params, idx=0):
        self.smpl_params.set_params_from_dict(smpl_params, idx=idx, device=self.device)

    def set_params(self, smpl_params, idx=0, use_pca=True):
        self.smpl_params.set_params(smpl_params, idx, use_pca=use_pca)

    def openpose2smpl(self, keypoints):
        """
            mapping 2d keypoints to the smpl-x format
            :param keypoints: 2d or 3d keypoints (support openpose and mediapifpaf formats)
                              torch.Tensor
            :return torch.Tensor of size 1x144x3: converted keypoints to the SMPL-X format
        """
        if len(keypoints.shape) == 2:
            d = keypoints.shape[1]
            keypoints_new = torch.zeros((144, d), device=self.device)
            keypoints_new[self.smpl_idx, :] = keypoints[self.pose_idx, :]
            # keypoints_new[65:, :] = 0
            return keypoints_new.unsqueeze(0)
        elif len(keypoints.shape) == 3:
            d = keypoints.shape[2]
            keypoints_new = torch.zeros((keypoints.shape[0], 144, d), device=self.device)
            keypoints_new[:, self.smpl_idx, :] = keypoints[:, self.pose_idx, :]
            return keypoints_new
        else:
            assert "check dimension"

    def smpl2openpose(self, keypoints):
        """
            mapping 2d keypoints to the smpl-x format
            :param keypoints: 2d or 3d keypoints (support openpose and mediapifpaf formats)
                              torch.Tensor
            :return torch.Tensor of size 1x144x3: converted keypoints to the SMPL-X format
        """
        d = keypoints.shape[1]
        keypoints_new = torch.zeros((133, d), device=self.device)
        keypoints[self.pose_idx, :] = keypoints_new[self.smpl_idx, :]
        return keypoints_new.unsqueeze(0)

    def set_optimizers(self, target='base', lr=0.01):
        """
            [description]
            :param [name]: [description]
            :return: [description]
        """
        opt_params = []
        self.smpl_params.toggle_gradients()

        if 'base' in target:
            for i in range(self.num_models):
                self.smpl_params.global_orient[i].requires_grad = True
                self.smpl_params.transl[i].requires_grad = True
                opt_params.append(self.smpl_params.global_orient[i])
                opt_params.append(self.smpl_params.transl[i])
        if 'shape':  # scale affects the result !!
            for i in range(self.num_models):
                self.smpl_params.betas[i].requires_grad = True
                opt_params.append(self.smpl_params.betas[i])
        if 'base' in target or 'shape' in target:
            for i in range(self.num_models):
                self.smpl_params.scale[i].requires_grad = True
                opt_params.append(self.smpl_params.scale[i])
        if 'pose':
            for i in range(self.num_models):
                self.smpl_params.body_pose[i].requires_grad = True
                opt_params.append(self.smpl_params.body_pose[i])
        if 'hands' in target:
            for i in range(self.num_models):
                self.smpl_params.left_hand_pose[i].requires_grad = True
                self.smpl_params.right_hand_pose[i].requires_grad = True
                opt_params.append(self.smpl_params.left_hand_pose[i])
                opt_params.append(self.smpl_params.right_hand_pose[i])
        if 'face' in target:
            for i in range(self.num_models):
                self.smpl_params.expression[i].requires_grad = True
                opt_params.append(self.smpl_params.expression[i])
                self.smpl_params.jaw_pose[i].requires_grad = True
                opt_params.append(self.smpl_params.jaw_pose[i])

        optimizer = torch.optim.Adam(opt_params, lr=lr)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5
        )
        return optimizer, scheduler

    def get_smpl_params(self):
        output = dict()
        output['global_orient'] = list(self.smpl_params.global_orient[0].
                                       detach().cpu().numpy().reshape(-1).astype(float))
        output['betas'] = list(self.smpl_params.betas[0]
                               .detach().cpu().numpy().reshape(-1).astype(float))
        output['body_pose'] = list(self.smpl_params.body_pose[0]
                               .detach().cpu().numpy().reshape(-1).astype(float))
        output['left_hand_pose'] = list(self.smpl_params.left_hand_pose[0]
                                        .detach().cpu().numpy().reshape(-1).astype(float))
        output['right_hand_pose'] = list(self.smpl_params.right_hand_pose[0]
                                         .detach().cpu().numpy().reshape(-1).astype(float))
        output['transl'] = list(self.smpl_params.transl[0]
                                .detach().cpu().numpy().reshape(-1).astype(float))
        output['jaw_pose'] = list(self.smpl_params.jaw_pose[0]
                                  .detach().cpu().numpy().reshape(-1).astype(float))
        output['expression'] = list(self.smpl_params.expression[0]
                                    .detach().cpu().numpy().reshape(-1).astype(float))
        output['scale'] = list(self.smpl_params.scale[0]
                                    .detach().cpu().numpy().reshape(-1).astype(float))
        return output

    def get_smpl_params_torch(self, idx=0):
        output = dict()
        output['global_orient'] = self.smpl_params.global_orient[idx]
        output['betas'] = self.smpl_params.betas[idx]
        output['body_pose'] = self.smpl_params.body_pose[idx]
        output['left_hand_pose'] = self.smpl_params.left_hand_pose[idx]
        output['right_hand_pose'] = self.smpl_params.right_hand_pose[idx]
        output['transl'] = self.smpl_params.transl[idx]
        output['jaw_pose'] = self.smpl_params.jaw_pose[idx]
        output['expression'] = self.smpl_params.expression[idx]
        output['scale'] = self.smpl_params.scale[idx]

        return output

    @staticmethod
    def set_sdf(vertices, faces):  # for smpl-based signed distance prediction
        from pysdf import SDF
        sdf_func = SDF(vertices, faces)
        return sdf_func

    @staticmethod
    def set_semantic(data):
        body_idx, face_idx, hand_idx = [], [], []
        for item in data:
            if 'Eye' in item or 'Head' in item:
                face_idx.append(torch.Tensor(data[item]))
            if 'Shoulder' in item or \
                    'Arm' in item or \
                    'neck' in item or \
                    'spin' in item or \
                    'Leg' in item or \
                    'Foot' in item or \
                    'Toe' in item or \
                    'hips' in item:
                body_idx.append(torch.Tensor(data[item]))
            if 'Hand' in item:
                hand_idx.append(torch.Tensor(data[item]))
        return body_idx, face_idx, hand_idx

    def forward_test(self, smpl_params, idx=0, device='cuda:0', fix_all=False):
        smpl_output = self.model(transl=self.smpl_params.transl[idx],
                                 expression=smpl_params['expression'].detach(),
                                 body_pose=smpl_params['body_pose'].detach(),
                                 betas=smpl_params['betas'].detach(),
                                 global_orient=self.smpl_params.global_orient[idx],
                                 jaw_pose=smpl_params['jaw_pose'].detach(),
                                 left_hand_pose=smpl_params['left_hand_pose'].detach(),
                                 right_hand_pose=smpl_params['right_hand_pose'].detach(),
                                 return_full_pose=True,
                                 return_verts=True
                                 )

        smpl_output.joints = smpl_output.joints * self.smpl_params.scale[idx]
        smpl_output.vertices = smpl_output.vertices * self.smpl_params.scale[idx]
        return smpl_output

    def forward(self, idx=0, model_constraints=None):
        mask = torch.ones_like(self.smpl_params.body_pose[idx]).to(self.device)
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

        smpl_output = self.model(transl=self.smpl_params.transl[idx].to(self.device),
                                 expression=self.smpl_params.expression[idx].to(self.device),
                                 body_pose=self.smpl_params.body_pose[idx].to(self.device)*mask,
                                 betas=self.smpl_params.betas[idx].to(self.device),
                                 global_orient=self.smpl_params.global_orient[idx].to(self.device),
                                 jaw_pose=self.smpl_params.jaw_pose[idx].to(self.device),
                                 left_hand_pose=self.smpl_params.left_hand_pose[idx].to(self.device),
                                 right_hand_pose=self.smpl_params.right_hand_pose[idx].to(self.device),
                                 return_full_pose=True,
                                 return_verts=True
                                 )

        smpl_output.joints = smpl_output.joints * self.smpl_params.scale[idx].to(self.device)
        smpl_output.vertices = smpl_output.vertices * self.smpl_params.scale[idx].to(self.device)

        return smpl_output
