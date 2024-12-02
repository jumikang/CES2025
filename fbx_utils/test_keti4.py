import os
import cv2
import glob
import yaml
import torch
import hydra
import trimesh
import numpy as np
from misc import load_data
from src.fbx_wrapper import FBXExporter
from src.smpl_optimizer.smpl_utils import load_smpl_info
from smpl_optimizer.light_animator import LightHumanAnimator
from mesh2mesh_optimizer import ARAPOptimizer
# from smpl_optimizer.light_smpl_optimizer import LightSMPLOptimizer
# from lib.utils.visualizer import show_meshes
# from lib.datasets.HumanDataset import HumanDataset
# from smpl_optimizer.human_pose_estimator import BodyPoseFacade
import warnings
warnings.filterwarnings(action='ignore')


@hydra.main(config_path="confs", config_name="human_recon_base")
def main(opt):
    print("Working dir:", os.getcwd())

    src_root = "/home/mpark/code/DepthTrainerPL/src"
    img_list, mask_list, depth_list, pose_list, path2save = load_data(is_train=False, dataset='SET1')
    path2smpl = path2save.replace('MESH', 'SMPLX')
    os.makedirs(path2smpl, exist_ok=True)

    cam_config = os.path.join(src_root, "confs/cam_config.yaml")
    with open(cam_config) as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)
        cam_params = cam_params["CAM_512"]

    path2semantic = os.path.join(src_root, 'data/smplx/body_segmentation/smplx/smplx_vert_segmentation.json', )
    mesh_optimizer = ARAPOptimizer(cam_params=cam_params, path2semantic=path2semantic)

    # animation and fbx generation!
    animator = LightHumanAnimator(smpl_config=os.path.join(src_root, 'confs/smpl_config.yaml'),
                                  smpl_root=os.path.join(src_root, 'data/smplx'),
                                  smpl_model=None)
    animator.set_motion_list()

    fbx_baker = FBXExporter(animator.smpl_model)

    for k in range(1, 2):

        save2file = os.path.join(path2save,
                                 img_list[k].split('/')[-1].replace('.png', '.obj'))
        pred_mesh = trimesh.load(save2file)

        save2smpl_mesh = os.path.join(path2smpl, img_list[k].split('/')[-1].replace('.png', '.obj'))
        save2smpl_params = os.path.join(path2smpl, img_list[k].split('/')[-1].replace('.png', '.json'))
        save_dirs = {"folder": path2smpl, "mesh": save2smpl_mesh, "params": save2smpl_params}

        smpl_params, smpl_mesh = (
            load_smpl_info(save_dirs["params"], path2mesh=save_dirs["mesh"]))

        motion = None
        canonical_vts, canonical_lbs, canonical_mesh, canon_smpl_mesh, canon_joints = animator.rig(smpl_mesh,
                                                                                                   pred_mesh,
                                                                                                   smpl_params,
                                                                                                   motion=motion,
                                                                                                   return_mesh=True,
                                                                                                   colorize_normal=False,
                                                                                                   v_label=mesh_optimizer.v_label)

        save2fbx = 'animated_' + str(k) + '.fbx'
        # animator.fetch_motion(path2motion=os.path.join(src_root, "data/sample_motion/t2m-gpt-motion.pkl"))
        animator.fetch_motion()
        motion_batch = animator.save_motion()

        # animator.bake_fbx(canonical_lbs, canonical_mesh, canon_joints, motion_batch, filename=save2fbx)
        fbx_baker.bake_fbx(canonical_lbs, canonical_mesh, canon_joints, motion_batch, filename=save2fbx)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
