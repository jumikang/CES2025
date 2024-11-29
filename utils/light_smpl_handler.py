# from __future__ import annotations
import os

import numpy as np
import smplx
import json
import torch
import trimesh
from torch import nn
from tqdm import tqdm

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


class LightSMPLWrapper(nn.Module):
    def __init__(self,
                 smpl_conf,  # dictionary
                 smpl_path='./path/to/smpl/models',
                 # regressor='./data/smplx/J_regressor_body25_smplx.txt',
                 device='cuda:0'):
        super(LightSMPLWrapper, self).__init__()
        self.device = device
        self.model_path = smpl_path
        self.smpl_conf = smpl_conf
        # self.regressor = regressor
        self.model = self.set_smpl_model()

        self.use_ezmocap = False
        self.vertex_mapper = [[332, 9120],
                             [6260, 9929],
                             [2800, 9448],
                             [4071, 616],
                             [583, 6],
                             # [6191, 8079],
                             # [5782, 7669],
                             # [5905, 7794],
                             # [6016, 7905],
                             # [6133, 8022],
                             # [2746, 5361],
                             # [2319, 4933],
                             # [2445, 5058],
                             # [2556, 5169],
                             # [2673, 5286],
                             [3216, 5770],
                             [3226, 5780],
                             [3387, 8846],
                             [6617, 8463],
                             [6624, 8474],
                             [6787, 8635]]
        self.vertex_mapper = np.asarray(self.vertex_mapper)

    def set_smpl_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        if self.smpl_conf['model_type'] == 'smplx':
            return smplx.create(self.model_path,
                                model_type=self.smpl_conf['model_type'],
                                gender=self.smpl_conf['gender'],
                                num_betas=self.smpl_conf['num_betas'],
                                ext=self.smpl_conf['ext'],
                                use_face_contour=self.smpl_conf['use_face_contour'],
                                flat_hand_mean=self.smpl_conf['use_flat_hand'],
                                use_pca=self.smpl_conf['use_pca'],
                                num_pca_comps=self.smpl_conf['num_pca_comp']
                                ).to(self.device)
        elif self.smpl_conf['model_type'] == 'smpl':
            return smplx.create(self.model_path,
                                model_type=self.smpl_conf['model_type'],
                                gender=self.smpl_conf['gender'],
                                num_betas=self.smpl_conf['num_betas'],
                                ext=self.smpl_conf['ext']
                                ).to(self.device)

    def to_tensor(self, data):
        return torch.FloatTensor(data).view(1, -1).to(self.device)

    def set_params_smplx(self, smpl_params):
        output = self.get_empty_params()
        output['global_orient'] = self.to_tensor(smpl_params['global_orient'])
        output['transl'] = self.to_tensor(smpl_params['transl'])
        output['body_pose'] = self.to_tensor(smpl_params['body_pose'])
        output['betas'] = self.to_tensor(smpl_params['betas'])
        output['scale'] = self.to_tensor([smpl_params['scale']])
        if self.smpl_conf['model_type'] == 'smplx':
            output['left_hand_pose'] = self.to_tensor([smpl_params['left_hand_pose']])
            output['right_hand_pose'] = self.to_tensor([smpl_params['right_hand_pose']])
            output['jaw_pose'] = self.to_tensor([smpl_params['jaw_pose']])
            output['expression'] = self.to_tensor([smpl_params['expression']])
        if 'gender' in smpl_params:
            output['gender'] = smpl_params['gender']
        return output

    def set_params_ez_mocap(self, path2json):
        assert os.path.exists(path2json), path2json
        with open(path2json, "r") as f:
            smpl_params = json.load(f)

        output = self.get_empty_params()
        output['global_orient'] = None
        output['transl'] = None
        output['ezmocap_global'] = torch.FloatTensor(smpl_params[0]['Rh']).to(self.device)
        output['ezmocap_transl'] = torch.FloatTensor(smpl_params[0]['Th']).to(self.device)
        output['poses'] = torch.FloatTensor(smpl_params[0]['poses']).to(self.device)
        output['body_pose'] = output['poses'][:, 3:66]
        output['left_hand_pose'] = output['poses'][:, 66:72]  # 1 x 6 (pca)
        output['right_hand_pose'] = output['poses'][:, 72:78]
        output['jaw_pose'] = output['poses'][:, 78:81]
        output['betas'] = torch.FloatTensor(smpl_params[0]['shapes']).to(self.device)
        output['scale'] = torch.FloatTensor(smpl_params[0]['scale']).to(self.device)
        output['expression'] = torch.FloatTensor(smpl_params[0]['expression']).to(self.device)
        if 'gender' in smpl_params[0]:
            output['gender'] = smpl_params[0]['gender']
        return output

    @staticmethod
    def get_empty_params():
        return {'global_orient': None, 'body_pose': None,
                'expression': None, 'betas': None,
                'jaw_pose': None, 'left_hand_pose': None, 'right_hand_pose': None,
                'transl': None, 'scale': None}

    def smpl_converter(self, smpl_params, smplx_params,
                       iters=500, lr=0.01, opt_pose=False):

        opt_params = []
        if opt_pose:
            smpl_params['body_pose'].requires_grad = True
            opt_params.append(smpl_params['body_pose'])
        else:
            smpl_params['body_pose'].requires_grad = True
            opt_params.append(smpl_params['body_pose'])
            smpl_params['transl'].requires_grad = True
            smpl_params['scale'].requires_grad = True
            smpl_params['betas'].requires_grad = True
            opt_params.append(smpl_params['scale'])
            opt_params.append(smpl_params['transl'])
            opt_params.append(smpl_params['betas'])

        l2_loss = nn.MSELoss()
        smplx_vertices = smplx_params.vertices.detach()
        smplx_joints = smplx_params.joints.detach()

        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        with tqdm(range(iters), ncols=170) as pbar:
            for itr in pbar:

                smpl_output = self.forward(smpl_params)
                loss = l2_loss(smpl_output.vertices[:, self.vertex_mapper[:, 0], :],
                               smplx_vertices[:, self.vertex_mapper[:, 1], :])
                loss += l2_loss(smpl_output.joints[:, :22, :],
                                smplx_joints[:, :22, :])
                pbar.set_description('iter:{0}, loss:{1:0.5f}'.format(itr, loss * 100000 ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % 100 == 0:
                    scheduler.step()

        # smpl_output = self.forward(smpl_params)
        for key in smpl_params.keys():
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].detach()
        return smpl_params

    def simple_optimizer(self, smpl_params, smpl_vertices, iters=500, lr=0.1):
        for key in smpl_params.keys():
            if isinstance(smpl_params[key], list):
                smpl_params[key] = torch.FloatTensor(smpl_params[key]).to(self.device)

            if isinstance(smpl_params[key], np.ndarray):
                smpl_params[key] = torch.FloatTensor(smpl_params[key]).to(self.device)
        smpl_params['global_orient'] = torch.tensor([[0., 0., 0.]],
                                                    requires_grad=True,
                                                    device=self.device)
        smpl_params['transl'] = smpl_params['Th'].clone().detach()
        smpl_params['betas'] = smpl_params['shapes'].clone().detach()
        smpl_params['body_pose'] = smpl_params['poses'][:, 3:66].clone().detach()
        smpl_params['jaw_pose'] = smpl_params['poses'][:, 66:69].clone().detach()
        smpl_params['jaw_pose'] /= 2.0
        smpl_params['left_hand_pose'] = smpl_params['poses'][:, 75:81].clone().detach()
        smpl_params['right_hand_pose'] = smpl_params['poses'][:, 81:88].clone().detach()
        smpl_params['transl'].requires_grad = True
        smpl_params['scale'].requires_grad = True
        smpl_params['jaw_pose'].requires_grad = True
        smpl_params['left_hand_pose'].requires_grad = True
        smpl_params['right_hand_pose'].requires_grad = True
        smpl_params.pop('poses')
        smpl_params.pop('Th')
        smpl_params.pop('Rh')
        smpl_params.pop('shapes')

        l2_loss = nn.MSELoss()
        opt_params = []
        opt_params.append(smpl_params['global_orient'])
        opt_params.append(smpl_params['transl'])
        opt_params.append(smpl_params['scale'])
        opt_params.append(smpl_params['jaw_pose'])
        opt_params.append(smpl_params['left_hand_pose'])
        opt_params.append(smpl_params['right_hand_pose'])
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        if isinstance(smpl_vertices, np.ndarray):
            smpl_vertices = torch.FloatTensor(smpl_vertices).to(self.device)
        with tqdm(range(iters), ncols=170) as pbar:
            for itr in pbar:
                optimizer.zero_grad()
                smpl_output = self.forward(smpl_params)
                loss = l2_loss(smpl_output.vertices, smpl_vertices)
                pbar.set_description('iter:{0}, loss:{1:0.5f}'.format(itr, loss * 100000 ))
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

        smpl_output, smpl_mesh = self.forward(smpl_params, return_mesh=True)
        for key in smpl_params.keys():
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].detach()
        return smpl_params, smpl_mesh

    def save_smpl_params(self, path2save, save_mesh=False):
        pass

    def forward(self, smpl_params, return_mesh=False, use_ezmocap=False):
        if self.smpl_conf['model_type'] == 'smplx':
            smpl_output = self.model(transl=smpl_params['transl'],
                                     expression=smpl_params['expression'],
                                     body_pose=smpl_params['body_pose'],
                                     betas=smpl_params['betas'],
                                     global_orient=smpl_params['global_orient'],
                                     jaw_pose=smpl_params['jaw_pose'],
                                     left_hand_pose=smpl_params['left_hand_pose'],
                                     right_hand_pose=smpl_params['right_hand_pose'],
                                     return_full_pose=True,
                                     return_verts=True
                                     )
        elif self.smpl_conf['model_type'] == 'smpl':
            if smpl_params['body_pose'].shape[1] == 63:
                smpl_params['body_pose'] = torch.concatenate((smpl_params['body_pose'],
                                                              smpl_params['body_pose'][:, :6]), dim=1)
            smpl_output = self.model(transl=smpl_params['transl'],
                                     body_pose=smpl_params['body_pose'],
                                     betas=smpl_params['betas'],
                                     global_orient=smpl_params['global_orient'],
                                     return_full_pose=True,
                                     return_verts=True
                                     )
        if use_ezmocap:
            # eazy mocap format (https://github.com/zju3dv/EasyMocap)
            # global_orient is different from ezmocap_global
            rot = batch_rodrigues(smpl_params['ezmocap_global'])
            smpl_output.vertices = (torch.matmul(smpl_output.vertices, rot.transpose(1, 2)) +
                                    smpl_params['ezmocap_transl'])

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            smpl_output.joints = smpl_output.joints * smpl_params['scale']
            smpl_output.vertices = smpl_output.vertices * smpl_params['scale']

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(),
                                        faces=self.model.faces, process=False, maintain_order=True)
            smpl_mesh.visual.vertex_colors = ((smpl_mesh.vertex_normals + 1.0) / 2.0 * 255.0).astype(np.uint8)

            return smpl_output, smpl_mesh
        else:
            return smpl_output


