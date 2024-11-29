from __future__ import annotations
import os
import numpy as np
import smplx
import json
import torch
import copy
import trimesh
from torch import nn
from PIL import Image

class BaseWrapper(nn.Module):
    def __init__(self,
                 smpl_config,  # dictionary
                 smpl_path='./path/to/smpl/models',
                 device='cuda:0'):
        super(BaseWrapper, self).__init__()
        self.device = device
        self.model_path = smpl_path
        self.smpl_config = smpl_config
        self.model = self.set_model()

        if self.smpl_config['use_uv']:
            self.uv_mapping, self.uv_faces, self.uv_pos, self.uv_texture = self.set_smpl_uv(smpl_path)

    def set_smpl_uv(self, smpl_path):
        mapping_table, smpl_uv_faces, smpl_uv_vt, smpl_texture = None, None, None, None
        path2uv_table = os.path.join(smpl_path, 'smplx_textures', 'smpl_uv_table.json')
        if os.path.exists(path2uv_table):
            with open(path2uv_table, 'r') as f:
                mapping_table = json.load(f)
        path2uv_mesh = os.path.join(smpl_path, 'smplx_uv', 'smplx_uv.obj')
        if os.path.exists(path2uv_mesh):
            smpl_uv_mesh = trimesh.load_mesh(path2uv_mesh, process=False)
            smpl_uv_faces = smpl_uv_mesh.faces
            smpl_uv_vt = smpl_uv_mesh.visual.uv
        # path2uv_texture = os.path.join(smpl_path, 'smplx_uv', 'texture_' + self.smpl_config['gender'] + '.png')
        path2uv_texture = os.path.join(smpl_path, 'smplx_uv', 'texture_female.png')
        if os.path.exists(path2uv_texture):
            smpl_texture = Image.open(path2uv_texture)

        return mapping_table, smpl_uv_faces, smpl_uv_vt, smpl_texture

    def set_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        return smplx.create(self.model_path,
                            model_type=self.smpl_config['model_type'],
                            gender=self.smpl_config['gender'],
                            num_betas=self.smpl_config['num_betas'],
                            ext=self.smpl_config['ext'],
                            use_face_contour=self.smpl_config['use_face_contour'],
                            flat_hand_mean=self.smpl_config['use_flat_hand'],
                            use_pca=self.smpl_config['use_pca'],
                            num_pca_comps=self.smpl_config['num_pca_comp']
                            ).to(self.device)

    @staticmethod
    def get_empty_params():
        return {'global_orient': None, 'body_pose': None,
                'expression': None, 'betas': None,
                'jaw_pose': None, 'left_hand_pose': None, 'right_hand_pose': None,
                'reye_pose': None, 'leye_pose': None,
                'transl': None, 'scale': None}

    @staticmethod
    def load_params(path2json):
        assert os.path.exists(path2json), path2json
        with open(path2json, "r") as f:
            smpl_params = json.load(f)

        output = BaseWrapper.get_empty_params()
        for key in smpl_params.keys():
            # only when params are array (not string or scalar)
            if isinstance(smpl_params[key], str) or np.isscalar(smpl_params[key]) or smpl_params[key] is None:
                output[key] = smpl_params[key]
            elif torch.is_tensor(smpl_params[key]):
                output[key] = smpl_params[key].reshape(1, -1)
            else:
                output[key] = torch.FloatTensor(smpl_params[key]).reshape(1, -1)
        return output

    def to_gpu(self, smpl_params):
        for key in smpl_params.keys():
            # only when params are array (not string or scalar)
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].to(self.device)
        return smpl_params

    def to_cpu(self, smpl_params):
        pass

    def to_numpy(self, smpl_params):
        pass

    def export_params(self, path2save, smpl_params, smpl_mesh=None):
        smpl_params_to_save = copy.deepcopy(smpl_params)
        for key in smpl_params_to_save:
            if (smpl_params_to_save[key] is not None and not isinstance(smpl_params_to_save[key], str) and
                    not np.isscalar(smpl_params_to_save[key])):
                smpl_params_to_save[key] = smpl_params_to_save[key].detach().cpu().numpy().tolist()
        with open(path2save, 'w') as f:
            json.dump(smpl_params_to_save, f, indent=4)
        if smpl_mesh is not None:
            smpl_mesh.export(path2save.replace('.json', '.obj'))

    def convert2smpl_uv(self, smpl_mesh, exclude_texture=True):
        if smpl_mesh.vertices.shape[0] > 10475:
            return smpl_mesh

        if exclude_texture:
            texture_visual = trimesh.visual.TextureVisuals(uv=self.uv_pos)
        else:
            texture_visual = trimesh.visual.TextureVisuals(uv=self.uv_pos, image=self.uv_texture)
        smpl_mesh_uv = trimesh.Trimesh(vertices=smpl_mesh.vertices[self.uv_mapping['smplx_uv'], :],
                                       faces=self.uv_faces, visual=texture_visual, process=False)
        return smpl_mesh_uv

    def forward(self, smpl_params, return_mesh=False):
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

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            smpl_output.joints = smpl_output.joints * smpl_params['scale']
            smpl_output.vertices = smpl_output.vertices * smpl_params['scale']

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(),
                                        faces=self.model.faces, process=False)
            return smpl_output, smpl_mesh
        else:
            return smpl_output
