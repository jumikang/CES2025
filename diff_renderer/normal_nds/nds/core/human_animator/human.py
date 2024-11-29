from __future__ import annotations
import smplx
import torch.utils.data
import json
import os
import glob
import cv2
import trimesh
import numpy as np
from utils.visualizer import *
import albumentations as albu
from pylab import imshow
import tqdm
import warnings
import people_segmentation
from people_segmentation.pre_trained_models import create_model
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image


class HumanModel():
    def __init__(self,
                 images=None,
                 target_meshes=None,
                 affine_params=None,
                 smpl_params=None,
                 smpl_path=None,
                 smpl_part=None,
                 model_type='smplx',
                 gender='neutral'):
        super(HumanModel, self).__init__()
        # 1. everything begins with images.
        self.images, self.res = [], []
        self.set_images(images)
        # setting cameras and 2D features.
        self.camera_params, self.landmarks = [], []
        self.custom_colors = None
        self.affine_params = affine_params

        # 2. reconstruct and set target meshes.
        # 3. align meshes with the smpl-x model.
        self.meshes, self.centroid, self.scale = [], [], []
        for param in self.affine_params:
            self.centroid.append(param['scan_centroid'])
            self.scale.append(param['scan_scale'])
        self.smpl_centroid, self.smpl_scale = [], []
        self.smpl_params, self.smpl_meshes, self.smpl_model = [], [], None
        self.full_pose = []
        self.model_path = smpl_path
        self.model_type = model_type
        self.model_gender = gender
        self.smpl_colors = None
        self.set_smpl(smpl_params)
        self.set_meshes(target_meshes)
        self.smpl_custom_colors = None
        self.t_posed_smpl = []

        # 4. set position and sizes.
        # template mesh and lbs weights (v_num x 55 or 24)
        self.v_custom, self.l_custom = [], []
        self.v_nums = []  # number of custom vertices.
        # canonical vertices (same as v_custom if the number of frames is 1)
        self.v_canonical, self.f_canonical = [], []
        self.v_posed = []

        # 5. set per vertex semantic label.
        if smpl_part is not None and os.path.isfile(smpl_part):
            with open(smpl_part, "r") as json_file:
                self.v_label = json.load(json_file)
        else:
            self.v_label = None

        # set wrist labels from the existing labels.
        self.v_label['leftWrist'], self.v_label['rightWrist'] = [], []
        for k in self.v_label['leftHand']:
            if k in self.v_label['leftForeArm']:
                self.v_label['leftWrist'].append(k)
        for k in self.v_label['rightHand']:
            if k in self.v_label['rightForeArm']:
                self.v_label['rightWrist'].append(k)

        self.human_segmentation_model = None
        self.mask = []

    def get_smpl_model(self, smpl_params):
        # return None if self.smpl_model is None:
        return self.smpl_model(transl=smpl_params['transl'],
                               betas=smpl_params['betas'],
                               body_pose=smpl_params['body_pose'],
                               # global_orient=smpl_params['global_orient'],
                               jaw_pose=smpl_params['jaw_pose'],
                               joints=smpl_params['joints'],
                               expression=smpl_params['expression'],
                               left_hand_pose=smpl_params['left_hand_pose'],
                               right_hand_pose=smpl_params['right_hand_pose'],
                               return_verts=True)

    def set_smpl(self, smpl_params):
        if smpl_params is None:
            return None
        self.smpl_params = smpl_params
        self.smpl_model = smplx.create(self.model_path,
                                       model_type=self.model_type,
                                       gender=self.model_gender,
                                       num_betas=10,
                                       ext='npz',
                                       use_pca=False).cuda()

        # reset data.
        self.smpl_meshes, self.smpl_centroid, self.smpl_scale = [], [], []
        for smpl in smpl_params:
            # set foot and toes to be zeros (alleviate perspective distortion)
            # smpl['body_pose'][:, 18:24] = 0
            # smpl['body_pose'][:, 27:33] = 0
            self.smpl_meshes.append(self.smpl_model(transl=smpl['transl'],
                                                    betas=smpl['betas'],
                                                    body_pose=smpl['body_pose'],
                                                    global_orient=smpl['global_orient'],
                                                    jaw_pose=smpl['jaw_pose'],
                                                    joints=smpl['joints'],
                                                    expression=smpl['expression'],
                                                    left_hand_pose=smpl['left_hand_pose'],
                                                    right_hand_pose=smpl['right_hand_pose'],
                                                    return_verts=True))
            smpl_mesh_tmp = to_trimesh(self.smpl_meshes[-1].vertices, self.smpl_model.faces)
            self.smpl_centroid.append(smpl_mesh_tmp.bounding_box.centroid)
            self.smpl_scale.append(2.0 / np.max(smpl_mesh_tmp.bounding_box.extents))

            full_pose = torch.zeros(1, 165)
            if 'global_orient' in smpl and smpl['global_orient'] is not None:
                full_pose[0, :3] = smpl['global_orient'].reshape(-1, 3)
            if 'body_pose' in smpl and smpl['body_pose'] is not None:
                full_pose[0, 3:66] = smpl['body_pose'].reshape(-1, 63)
            if 'jaw_pose' in smpl and smpl['jaw_pose'] is not None:
                full_pose[0, 66:69] = smpl['jaw_pose'].reshape(-1, 3)
            if 'left_eye' in smpl and smpl['left_eye'] is not None:
                full_pose[0, 69:72] = smpl['left_eye'].reshape(-1, 3)
            if 'right_eye' in smpl and smpl['right_eye'] is not None:
                full_pose[0, 72:75] = smpl['right_eye'].reshape(-1, 3)
            if 'left_hand_pose' in smpl and smpl['left_hand_pose'] is not None:
                full_pose[0, 75:120] = smpl['left_hand_pose'].reshape(-1, 45)
            if 'right_hand_pose' in smpl and smpl['right_hand_pose'] is not None:
                full_pose[0, 120:165] = smpl['right_hand_pose'].reshape(-1, 45)
            self.full_pose.append(full_pose.cuda())

    def get_meshes(self, custom_color=False):
        t_posed, smpl_meshes = [], []
        for k in range(len(self.v_posed)):
            if custom_color:
                t_posed.append(to_trimesh(self.v_posed[k], self.meshes[k].faces,
                               vertex_colors=self.custom_color[k]))
            else:
                t_posed.append(to_trimesh(self.v_posed[k], self.meshes[k].faces,
                               vertex_colors=self.meshes[k].visual.vertex_colors))
        for k in range(len(self.smpl_meshes)):
            if custom_color:
                smpl_meshes.append(to_trimesh(self.smpl_meshes[k].vertices, self.smpl_model.faces,
                                              vertex_colors=self.smpl_colors[k]))
            else:
                smpl_meshes.append(to_trimesh(self.smpl_meshes[k].vertices, self.smpl_model.faces,
                                              vertex_colors=self.smpl_colors[k]))
                # smpl_meshes.append(to_trimesh(self.smpl_meshes[k].vertices, self.smpl_model.faces))
        return t_posed, smpl_meshes

    def set_meshes(self, meshes):
        if meshes is None:
            return None
        self.meshes = meshes
        for k in range(len(meshes)):
            self.meshes[k].vertices = (self.meshes[k].vertices - self.centroid[k]) * self.scale[k]
            self.meshes[k].vertices = self.meshes[k].vertices / self.smpl_scale[k] + self.smpl_centroid[k]

    def set_images(self, images):
        if images is None:
            return None
        self.images = images
        self.res = self.images[0].shape[1]

    def deform(self, idx, global_pose=None, body_pose=None, pose2rot=True, inverse=False, is_smpl=False, joints=None):
        if inverse:
            if is_smpl:
                v_posed = self.smpl_meshes[idx].vertices.double()
            else:
                v_posed = self.meshes[idx].vertices
                v_posed = torch.Tensor(v_posed).double().unsqueeze(0)
        else:
            if is_smpl:
                v_posed = self.smpl_model.v_template + \
                   smplx.lbs.blend_shapes(self.smpl_model.betas, self.smpl_model.shapedirs)
                v_posed = v_posed.double() # .unsqueeze(0)
            else:
                v_posed = self.v_posed[idx]
                # v_posed = self.scan_vshaped[idx]

        v_shaped = self.smpl_model.v_template + \
                   smplx.lbs.blend_shapes(self.smpl_model.betas, self.smpl_model.shapedirs)
        if joints is None:
            J, B = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, v_shaped), 1
        else:
            J, B = joints, 1

        if is_smpl:
            new_lbs = self.smpl_model.lbs_weights
        else:
            new_lbs = self.new_lbs[idx]
        full_pose = self.full_pose[idx]

        if global_pose is not None:
            full_pose[0, :3] = global_pose
        if body_pose is not None:
            full_pose[0, 3:66] = body_pose

        if pose2rot:
            rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([B, -1, 3, 3])
        else:
            rot_mats = full_pose.view(B, -1, 3, 3)

        J_transformed, A = smplx.lbs.batch_rigid_transform(rot_mats, J, self.smpl_model.parents, dtype=torch.float32)
        W = new_lbs.unsqueeze(dim=0).expand([B, -1, -1])
        num_joints = self.smpl_model.J_regressor.shape[0]
        T = torch.matmul(W, A.view(B, num_joints, 16)).view(B, -1, 4, 4)

        homogen_coord = torch.ones([B, v_posed.shape[1], 1], dtype=torch.float32).cuda()
        v_posed_homo = torch.cat([v_posed.cuda(), homogen_coord], dim=2)
        if inverse:
            T = torch.inverse(T.squeeze()).unsqueeze(0)

        v_homo = torch.matmul(T.double(), torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]
        return verts


def simple_loader(path2mesh=None, path2data=None, model_type='smplx'):
    mesh_list = glob.glob(path2mesh + '/mesh_scan_*.obj')
    npz_list = glob.glob(path2mesh + '/*.npz')

    ext = ['png', 'jpg', 'gif']
    image_list = []
    [image_list.extend(glob.glob(path2data + '/image_front_*.' + e)) for e in ext]

    image_list_back = []
    [image_list_back.extend(glob.glob(path2data + '/image_back_*.' + e)) for e in ext]

    npz_list = sorted(npz_list)
    mesh_list = sorted(mesh_list)
    image_list = sorted(image_list)
    image_list_back = sorted(image_list_back)

    meshes = []
    smpl_params = []
    images = []
    images_back = []
    affine_params = []

    for image in image_list:
        images.append(cv2.imread(image, 1))

    for image in image_list_back:
        images_back.append(cv2.imread(image, 1))

    for mesh in mesh_list:
        meshes.append(trimesh.load(mesh, processing=False, maintain_order=True))  # scanned model.

    for opt in npz_list:
        data = np.load(opt)
        smpl_param = {}

        if 'betas' in data:
            smpl_param['betas'] = torch.tensor(np.reshape(data.f.betas, [1, -1])).cuda()
        else:
            smpl_param['betas'] = None
        if 'global_orient' in data:
            smpl_param['global_orient'] = torch.tensor(np.reshape(data.f.global_orient, [1, -1])).float().cuda()
        else:
            smpl_param['global_orient'] = None
        if 'body_pose' in data:
            smpl_param['body_pose'] = torch.tensor(np.reshape(data.f.body_pose, [1, -1])).cuda()
        else:
            smpl_param['body_pose'] = None
        if 'joints' in data:
            smpl_param['joints'] = torch.tensor(np.reshape(data.f.joints, [1, -1])).cuda()
        else:
            smpl_param['joints'] = None
        if 'jaw_pose' in data:
            smpl_param['jaw_pose'] = torch.tensor(np.reshape(data.f.jaw_pose, [1, -1])).cuda()
        else:
            smpl_param['jaw_pose'] = None
        if 'left_hand' in data:
            smpl_param['left_hand'] = torch.tensor(np.reshape(data.f.left_hand_pose[0, 7], [1, -1])).cuda()
        else:
            smpl_param['left_hand'] = None
        if 'left_hand_pose' in data:
            smpl_param['left_hand_pose'] = torch.tensor(np.reshape(data.f.left_hand_pose, [1, -1])).cuda()
        else:
            smpl_param['left_hand_pose'] = None
        if 'right_hand' in data:
            smpl_param['right_hand'] = torch.tensor(np.reshape(data.f.right_hand_pose[0, 7], [1, -1])).cuda()
        else:
            smpl_param['right_hand'] = None
        if 'right_hand_pose' in data:
            smpl_param['right_hand_pose'] = torch.tensor(np.reshape(data.f.right_hand_pose, [1, -1])).cuda()
        else:
            smpl_param['right_hand_pose'] = None
        if 'expression' in data:
            smpl_param['expression'] = None
        else:
            smpl_param['expression'] = None
        if 'transl' in data:
            smpl_param['transl'] = torch.tensor(np.reshape(data.f.transl, [1, -1])).float().cuda()
        else:
            smpl_param['transl'] = None

        if 'vertices' in data:
            smpl_param['vertices'] = data.f.vertices

        if model_type == 'smpl':
            smpl_param['body_pose'] = torch.cat(
                (smpl_param['body_pose'], smpl_param['left_hand'], smpl_param['right_hand']), dim=1)

        affine = {}
        affine['smpl_centroid'] = np.reshape(data.f.affine_smpl_centroid, [1, -1])
        affine['smpl_scale'] = np.reshape(data.f.affine_smpl_scale, [1, -1])
        affine['scan_centroid'] = np.reshape(data.f.affine_scan_centroid, [1, -1])
        affine['scan_scale'] = np.reshape(data.f.affine_scan_scale, [1, -1])

        smpl_params.append(smpl_param)
        affine_params.append(affine)

    if path2data is None:
        return meshes, smpl_params, affine_params
    else:
        return meshes, smpl_params, affine_params, images, images_back


