import json
import torch
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
                if 'hand_left_keypoints_2d' not in pose_info:
                    pose_info['hand_left_keypoints_2d'] = torch.zeros((21, 3), dtype=float)
                if 'hand_right_keypoints_2d' not in pose_info:
                    pose_info['hand_right_keypoints_2d'] = torch.zeros((21, 3), dtype=float)
                keypoint = torch.concat((torch.Tensor(pose_info['pose_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['hand_left_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['hand_right_keypoints_2d']).reshape(-1, 3),
                                         torch.Tensor(pose_info['face_keypoints_2d']).reshape(-1, 3)),
                                         dim=0)
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