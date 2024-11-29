import os
import cv2
import numpy as np
import trimesh
import torch
import glob
import collections
from PIL import Image

import json
import torch
import numpy as np
from torch import nn
from argparse import Namespace


def keypoint_loader(filename, pose_detector='openpose'):
    with open(filename, 'r') as f:
        pose_info = json.load(f)
        if pose_detector == 'openpose':
            if 'people' in pose_info.keys():
                keypoint = torch.concat((torch.Tensor(pose_info['people'][0]['pose_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['people'][0]['face_keypoints_2d']).reshape(-1, 3)),
                                        dim=0)
            else:
                num_human = (len(pose_info['pose_keypoints_2d']) // 63)
                if 'hand_left_keypoints_2d' not in pose_info:
                    pose_info['hand_left_keypoints_2d'] = torch.zeros((21, 3), dtype=float)
                if 'hand_right_keypoints_2d' not in pose_info:
                    pose_info['hand_right_keypoints_2d'] = torch.zeros((21, 3), dtype=float)

                if num_human == 1:
                    pose_flame = torch.Tensor(pose_info['face_keypoints_2d']).reshape(-1, 3)
                    pose_flame = pose_flame[17:17+51, :]  # only 51 keypoints are compatible with flame
                    keypoint = torch.concat((torch.Tensor(pose_info['pose_keypoints_2d']).reshape(-1, 3),
                                             torch.Tensor(pose_info['hand_left_keypoints_2d']).reshape(-1, 3),
                                             torch.Tensor(pose_info['hand_right_keypoints_2d']).reshape(-1, 3),
                                             pose_flame), dim=0)
                else:
                    pos = torch.mean(torch.Tensor(pose_info['pose_keypoints_2d']).reshape(-1, 25, 3), dim=1)
                    idx = np.argmin(np.abs(pos[:, 0] - 256.0))
                    keypoint = torch.concat((torch.Tensor(pose_info['pose_keypoints_2d']).reshape(-1, 25, 3),
                                             torch.Tensor(pose_info['hand_left_keypoints_2d']).reshape(-1, 21, 3),
                                             torch.Tensor(pose_info['hand_right_keypoints_2d']).reshape(-1, 21, 3),
                                             torch.Tensor(pose_info['face_keypoints_2d']).reshape(-1, 70, 3)),
                                             dim=1)
                    keypoint = keypoint[idx, :, :]
                # keypoint[25:, :] = 0

        elif pose_detector == 'openpifpaf':
            keypoint = torch.Tensor(pose_info['full_pose']).reshape(-1, 3)
        return keypoint


# load initilized smpl parameters (if you want to start from this point)
def load_smpl_from_json(filename, device='cpu'):
    with open(filename, 'r') as fp:
        smpl_params = json.load(fp)
        for key in smpl_params.keys():
            if isinstance(smpl_params[key], list):
                smpl_params[key] = nn.Parameter(torch.Tensor(smpl_params[key]).reshape(1, -1)).to(device)
        smpl_params = Namespace(**smpl_params)
    return smpl_params

def load_textured_mesh(path2mesh, image_only=False):
    """
    Load textured mesh
    Initially, find filename.bmp/jpg/tif/jpeg/png as a texture map.
     > if there are multiple images, take the first one.
     > otherwise, find material_0.bmp/jpe/tif/jpeg/png as a texture map.
    :param path2mesh: path to the textured mesh (.obj file only)
    :param filename: name of the current mesh
    :return: mesh with texture (texture will not be defined, if the texture map does not exist)
    """
    filename = path2mesh.split('/')[-1][:-4]

    exts = ['.tif', '.bmp', '.jpg', '.jpeg', '.png', '_0.png']
    # text_file = os.path.join(self.path2obj, filename, filename)
    text_file = [path2mesh.replace('.obj', ext) for ext in exts
                 if os.path.isfile(path2mesh.replace('.obj', ext))]
    if len(text_file) == 0:
        # text_file = os.path.join(self.path2obj, filename, 'material_0')
        text_file = path2mesh.replace(filename + '.obj', 'material_0')
        text_file = [text_file + ext for ext in exts if os.path.isfile(text_file + ext)]

    # for RP_T dataset
    if len(text_file) == 0:
        obj = path2mesh.split('/')[-1]
        # text_file = path2mesh.replace(filename + '.obj', '_0.png')
        text_file = path2mesh.replace('.obj', '_0.png')
        # text_file = os.path.join(path2mesh.replace(obj, ''), 'tex', filename.replace('FBX', 'dif.jpg'))
        if os.path.isfile(text_file):
            text_file = [text_file]
        else:
            text_file = []
    if len(text_file) > 0:
        im = Image.open(text_file[0])
        texture_image = np.array(im)
    else:
        texture_image = None

    if image_only:
        return texture_image
    else:
        mesh = trimesh.load(path2mesh, process=False)  # normal mesh
        return mesh, texture_image

def load_gt_data(path2obj, height=180.0):
    """
    NOT OPTIMIZED YET
    :param data:
    :param height:
    :return:
    """
    # input data loading
    obj_path = os.path.join(path2obj)

    if len(obj_path) > 0:
        if isinstance(obj_path, list):
            obj_path = obj_path[0]

        m, texture_image = load_textured_mesh(obj_path)
        if texture_image is None:
            print('could not find texture map')
        else:
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            texture_image = np.flip(texture_image, axis=0)

        # for standing mesh (general)
        vertices = m.vertices
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        up_axis = 1
        center = np.median(vertices, 0)
        center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

        scale = height / (vmax[up_axis] - vmin[up_axis])

        # without library (comment below lines for non-agisoft results)
        vertices = (vertices - center) * scale

        # agisoft results (stability_ai data)
        # rot = np.eye(3)
        # rot[1, 1] *= -1.0
        # rot[2, 2] *= -1.0
        # vertices = np.matmul(vertices, rot)

        return vertices, m.faces, m.visual.uv, texture_image
