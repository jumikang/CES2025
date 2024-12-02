import os
import yaml
import torch
import pickle
import trimesh
from torch import nn
from smpl_optimizer.smpl_utils import load_smpl_info
from smpl_optimizer.light_animator import LightHumanAnimator
from diff_renderer.normal_nds.mesh2mesh_optimizer import ARAPOptimizer

import warnings
warnings.filterwarnings(action='ignore')

class FBX_Generator(nn.Module):
    def __init__(self,
                 device='cuda:0'):
        super(FBX_Generator, self).__init__()
        self.device = device

    def forward(self, input_dict):
        # load canonical mesh
        canon_mesh_path = os.path.join(input_dict["save_path"], input_dict["data_name"],
                                       'opt_recon_canon_%s.obj' % input_dict["data_name"])
        canonical_mesh = trimesh.load(canon_mesh_path)

        # load lbs
        lbs_path = os.path.join(input_dict["save_path"], input_dict["data_name"],
                                '%s_lbs.pickle' % input_dict["data_name"])
        with open(lbs_path, 'rb') as f:
            canonical_lbs = pickle.load(f)

        # load joints
        # with open(input_dict["input_joints"], 'rb') as f:
        #     joints_data = pickle.load(f)
        # joints_data = torch.FloatTensor(joints_data).to(self.device)

        # animation and fbx generation!
        animator = LightHumanAnimator(smpl_config='./config/smpl_config.yaml',
                                      smpl_root='./resource/smpl_models',
                                      smpl_model=None)
        animator.set_motion_list()
        render_mode = 'open3d'
        bake_fbx = True
        if render_mode == 'open3d':
            from smpl_optimizer.light_open3d_renderer import LightOpen3DRenderer
            renderer = LightOpen3DRenderer()
        elif render_mode == 'pytorch3d':
            from smpl_optimizer.light_pytorch3d_renderer import LightPytorch3DRenderer
            renderer = LightPytorch3DRenderer(res=1024)
            # renderer.render_ploty(canonical_mesh)
        else:
            assert 'set proper renderers...'

        keypoints = None
        # test_dataset = HumanDataset()

        # while True:
        # input, mask = test_dataset.load_custom_data(img_list[k], mask_list[k], depth_list[k])
        # keypoints = pose_detector.openpifpaf_detector(input[:3].unsqueeze(0))


        # save for ETRI (OMP project)
        # canonical_vts, canonical_lbs, canonical_mesh
        canonical_vertices = torch.FloatTensor(canonical_mesh.vertices[None, :, :]).to(self.device)
        if render_mode == 'open3d':
            # for _ in range(5):
            while True:
                renderer.set_mesh()
                animator.fetch_motion(path2motion=input_dict["input_motion"])
                animator.animate_open3d(canonical_vertices,
                                        canonical_lbs,
                                        canonical_mesh,
                                        renderer)
        elif render_mode == 'pytorch3d':
            # for k in range(100):
            k = 0
            while True:
                render_normal = True if k % 2 == 0 else False
                k += 1
                # animator.fetch_motion(path2motion=os.path.join(src_root, "data/sample_motion/test_motion.pkl"))
                animator.fetch_motion()
                output = animator.animate_pytorch3d(canonical_mesh.vertices,
                                                    canonical_lbs,
                                                    canonical_mesh,
                                                    renderer, render_normal=render_normal)
                # send to ui server
                animator.save2video(output)
                print('send output to polygom UI')
        return output