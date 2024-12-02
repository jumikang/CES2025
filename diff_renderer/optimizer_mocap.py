import yaml
import numpy as np
import torch
import glob
import os
import cv2
from torch import nn
from libs.pixie.options_pixie import ConfiguratorPixie
from libs.pixie.pixie_module import Pixie
from libs.ezmocap_light.smpl_ezmocap import EzMocapSMPL
from libs.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from utils.mesh_utils import keypoint_loader
from utils.light_smpl_handler import LightSMPLWrapper

import warnings
warnings.filterwarnings('ignore')

class Optimizer_mocap(nn.Module):
    def __init__(self,
                 params,
                 device='cuda:0'):
        super(Optimizer_mocap, self).__init__()

        # set params
        self.params = params
        self.device = device
        self.res = self.params['CAM']['DEFAULT']['width']
        self.model_type = self.params['SMPL']['model_type'].upper()
        self.skip_exist = False
        self.use_pixie = self.params['MOCAP']['use_pixie']
        self.use_depth_anything = self.params['MOCAP']['use_depth']
        self.use_gender = self.params['MOCAP']['use_gender']
        self.is_standing = self.params['MOCAP']['standing']

    def forward(self, input_dict):
        # setup pixie
        if self.use_pixie:
            config_pixie = ConfiguratorPixie()
            params_pixie = config_pixie.parse()
            params_pixie.inputpath = input_dict['input_path']
            params_pixie.out_dir = input_dict['save_path']
            params_pixie.savefolder = input_dict['save_path']
            pixie_net = Pixie(params_pixie, self.device)

        # setup depth anything v2
        if self.use_depth_anything:
            depth_anything = DepthAnythingV2()
            path2ckpt = os.path.join(self.params['MOCAP']['project_root'], self.params['MOCAP']['ckpt_depth'])
            depth_anything.load_state_dict(torch.load(path2ckpt, map_location='cpu'))
            depth_anything = depth_anything.to(self.device).eval()

        path2image = [input_dict["input_path"]]
        for idx, path in enumerate(path2image):
            path2save = input_dict["save_path"]
            dataname = input_dict["data_name"]
            os.makedirs(input_dict["save_path"], exist_ok=True)

            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image.shape[1] != self.res:
                image = cv2.resize(image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC)
            if image.shape[2] == 4:
                mask = image[:, :, -1] / 255.0
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                mask = np.clip(np.sum(image, axis=2), a_min=0, a_max=1)

            guide_smpl, no_people_flag = pixie_net(image) if self.params['MOCAP']['use_pixie'] else None
            guide_normal, guide_depth = depth_anything.get_normal(image.copy()) if self.params['MOCAP']['use_depth'] else None

            # 여기서부터 openpose 가져와서 해야함
            keypoint = keypoint_loader(input_dict['input_pose'])
            mocap = EzMocapSMPL(self.params, res=self.res, device=self.device)
            smpl_params, smpl_mesh, rendered, vertices = mocap.pipeline_single2d(keypoint.detach().cpu().numpy(),
                                                                                 mask=mask[:, :, None],
                                                                                 normal=guide_normal,
                                                                                 init_params=guide_smpl,
                                                                                 vis_smpl=True)
            # save results
            os.makedirs(os.path.join(input_dict["save_path"], input_dict["data_name"]), exist_ok=True)
            smpl_params['gender'] = self.params['SMPL']['gender']
            if self.params['MOCAP']['save_eazymocap']:
                mocap.save_smpl_params(smpl_params.copy(), smpl_mesh, f"{path2save}/{dataname}/ezmocap_{input_dict['file_name']}")

            if self.params['MOCAP']['save_standard']:
                # convert to standard SMPL parameters.
                smpl_handler = LightSMPLWrapper(self.params['SMPL'],
                                                smpl_path=self.params['SMPL']['smpl_root']+'/smpl_models')
                smpl_params_standard, smpl_mesh_standard = smpl_handler.simple_optimizer(smpl_params, vertices, iters=300)
                mocap.save_smpl_params(smpl_params_standard, smpl_mesh_standard, f"{path2save}/{dataname}/standard_{input_dict['file_name']}")

            # visualize smpl_model
            if self.params['MOCAP']['save_render']:
                if image.shape[2] == 3:
                    image = image + np.repeat((1 - mask[:, :, None])*255, repeats=3, axis=2)
                cv2.imwrite(f"{path2save}/{dataname}/{input_dict['file_name']}", image)
                # cv2.imwrite(f"{path2save}/{input_dict['file_name'].replace('.png', '_rendered_normal.png')}",
                #             (rendered['normal'].detach().cpu().numpy()+1)/2*255.0)


        obj_path = os.path.join(input_dict["save_path"], input_dict["data_name"],
                                'standard_%s.obj' % input_dict["data_name"])
        smpl_mesh_standard.vertices *= 30
        smpl_mesh_standard.export(obj_path)

        return obj_path