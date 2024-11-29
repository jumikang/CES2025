from __future__ import annotations
import pickle
import io
import cv2
import numpy as np
import numpy.matlib
from utils.visualizer import *
from utils.loader_utils import *
from expose.expose import ExPose
from utils.expose_utils import *
from smplify.smplify import SMPLifyLegacy
from humanimate.animate import AnimaterBasic, AnimaterLBS, CanonicalFusion
from humanimate.human import HumanModel, simple_loader
from humanimate.rig import RigAlignedHuman
from humanimate.recon import recon_wrapper
from humanimate.visualizer import Visualizer
# import albumentations as albu
# from pylab import imshow
import tqdm
import warnings
# import people_segmentation
# from people_segmentation.pre_trained_models import create_model
warnings.filterwarnings(action='ignore')
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
cudnn.fastest = True


if __name__ == '__main__':
    # parameters and options.
    idx_obj = 1
    input_type = 'single'
    # idx_obj = 10  # dslr example
    # input_type = 'ioys'

    # print(torch.cuda.is_available())
    smpl_params = dict()
    smpl_params['type'] = 'smplx'
    smpl_params['gender'] = 'neutral'
    smpl_params['regressor'] = 'h36m'  # for future use.

    # paths.
    save_folder = 'data{}'.format(idx_obj)
    paths = dict()
    # absolute paths (change 'data' to your own)
    paths['data'] = './data'
    paths['cfg'] = os.path.join(paths['data'], 'conf.yaml')
    paths['motion'] = '/data/samples/motions/*.pkl'
    paths['smpl'] = os.path.join(paths['data'], 'models')
    paths['semantic'] = os.path.join(paths['data'], 'body_segmentation/smplx/smplx_vert_segmentation.json')
    paths['root'] = '/data/samples'
    paths['folder'] = save_folder
    paths['input'] = os.path.join(paths['root'], 'image_' + input_type, paths['folder'])
    paths['scans'] = os.path.join(paths['root'], 'mesh', paths['folder'])
    paths['ckpt'] = os.path.join(paths['data'], 'checkpoints/CVPRW')

    path2save = dict()
    path2save['root'] = '/data/samples/results_' + input_type
    path2save['pred_color'] = os.path.join(path2save['root'], 'render_color', save_folder)
    path2save['pred_mesh'] = os.path.join(path2save['root'], 'render_mesh', save_folder)
    path2save['smplify'] = os.path.join(path2save['root'], 'simplify_results', save_folder)
    path2save['expose'] = os.path.join(path2save['root'], 'expose_results', save_folder)
    path2save['autorig'] = os.path.join(path2save['root'], 'rigged_results', save_folder)
    # path2save['fusion'] = os.path.join(path2save['root'], 'fused_results', save_folder)
    set_save_dirs(path2save)

    # reconstruct 3D human model
    save_data = run_align = True
    run_recon = False
    if run_recon:
        meshes, images, images_back = recon_wrapper(path2image=paths['input'],
                                                    path2mesh=paths['scans'],
                                                    path2checkpoints=paths['ckpt'])
        if save_data:
            save_results(meshes, path2save=path2save['pred_mesh'], type='mesh')
            save_results(images, path2save=path2save['pred_color'], type='image_front')
            save_results(images_back, path2save=path2save['pred_color'], type='image_back')

    if run_align:
        if run_recon is False:
            meshes, images, images_back = data_loader(path2save['pred_color'], path2save['pred_mesh'])
        expose = ExPose(data_path=path2save['root'], cfg_path=paths['cfg'], gender=smpl_params['gender'])
        smplify = SMPLifyLegacy(data_path=path2save['root'],
                                model_path=paths['smpl'],
                                body_seg_path=paths['semantic'],
                                model_name=smpl_params['type'],
                                gender=smpl_params['gender'],
                                cam_iters=200,
                                pose_iters=300,
                                res=256)

        expose_meshes, scan_meshes, expose_params = expose(meshes, images, idx_obj)
        # smplify_meshes, scan_meshes, opt_params = smplify(params=expose_params)
        smplify_meshes, scan_meshes, opt_params = smplify(params=expose_params,
                                                          meshes=scan_meshes)

        if save_data:
            save_results(smplify_meshes, path2save=path2save['smplify'], type='mesh_smpl')
            save_results(scan_meshes, path2save=path2save['smplify'], type='mesh_scan')
            save_results(opt_params, path2save=path2save['smplify'], type='opt')

    # load stored data.
    if run_align:
        exit(0)

    mesh_pred, smpl_preds, affine_params, images, images_back = simple_loader(path2mesh=path2save['smplify'],
                                                                              path2data=path2save['pred_color'])

    # initialize human model from SMPLify results.
    avatars = HumanModel(smpl_params=smpl_preds,
                         affine_params=affine_params,
                         images=images,
                         target_meshes=mesh_pred,
                         smpl_path=paths['smpl'],
                         smpl_part=paths['semantic'],  # per-vertex segmentation label
                         model_type=smpl_params['type'],
                         gender=smpl_params['gender'])

    smpl_mesh = to_trimesh(avatars.smpl_meshes[0].vertices, avatars.smpl_model.faces)

    # multi-view fusion can be placed here.
    rigger = RigAlignedHuman()
    is_seq = False
    is_multiview = False

    avatars.segment_humans()

    smplifier = AnimaterBasic(num_iters=200, voxel_res=512, is_multiview=is_multiview, is_seq=is_seq)
    # fuser = CanonicalFusion()

    recover_hands = save_data = run_rigging = True
    recover_hands = False
    save_data = True
    if run_rigging:
        avatars = rigger(avatars, interpolate=False)  # mesh to rigged model.
        # mesh = trimesh.Trimesh(avatars.v_posed[0].squeeze().detach().cpu().numpy(), faces=mesh_pred[0].faces,
        #                        vertex_colors=vertex_colors[:, 1:3], process=False)

        avatars = smplifier(avatars)  # refine alignment
        avatars = rigger(avatars, interpolate=True)
        # smpl_mesh = to_trimesh(avatars.smpl_meshes[0].vertices, avatars.smpl_model.faces)
        # show_meshes([avatars.meshes[0], smpl_mesh])
        avatars, affine_params = smplifier.canonicalize(avatars, affine_params)  # get t-posed meshes (normalized)

        # smpl_mesh = to_trimesh(avatars.smpl_meshes[0].vertices, avatars.smpl_model.faces)
        # show_meshes([avatars.meshes[0], smpl_mesh])
        # smpl_mesh = list()
        # smpl_mesh.append(to_trimesh(smplifier.scan_vshaped[0], avatars.meshes[0].faces))
        # smpl_mesh.append(to_trimesh(smplifier.scan_vshaped[1], avatars.meshes[1].faces))\
        # smpl_mesh.append(to_trimesh(smplifier.scan_vshaped[2], avatars.meshes[2].faces))
        # show_meshes(smpl_mesh)

        # avatars = fuser(avatars)  # fuse multiple predictions (views) - do nothing if the input is single image.
        # voxel_fusion() # t-posed -
        # avatars = smplifier.align_models(avatars)  # align models in the smpl coordinate
        # smpl_mesh = to_trimesh(avatars.smpl_meshes[0].vertices, avatars.smpl_model.faces)
        # # show_meshes([avatars.meshes[1], smpl_mesh])
        # show_meshes([avatars.meshes[0], smpl_mesh])
        # mesh = to_trimesh(avatars.v_posed[0].squeeze(0).detach().cpu().numpy(), avatars.meshes[0].faces, process=False)
        if recover_hands:
            avatars.meshes = smplifier.replace_hands([smpl_mesh],
                                                     avatars.meshes,
                                                     image_front=images,
                                                     image_back=images_back)
            avatars = rigger(avatars, interpolate=True)

        if save_data:
            save_results(avatars, path2save=path2save['autorig'], type='pickle')
    else:
        npz_avatar = glob.glob(path2save['autorig'] + '/*.pkl')
        with open(npz_avatar[0], "rb") as f:
            avatars = pickle.load(f)

    # pose augmentation for consistency checking.
    animater = AnimaterLBS(num_iters=500)
    idx_motion = 3
    visualizer = Visualizer(paths['motion'], seq=idx_motion)

    run_refine = False
    if run_refine:
        avatars = animater(avatars, visualizer)  # further optimize joints with random motion warping.

    visualize = True
    if visualize:
        for k in range(15):
            visualizer.get_cur_frame(avatars, vis=True)

    save_video = False
    if save_video:
        visualizer.save_as_video(avatars, idx_motion, idx_obj)

    # for debugging.
    # smpl_mesh1 = to_trimesh(avatars.smpl_meshes[0].vertices, avatars.smpl_model.faces)
    # smpl_mesh2 = to_trimesh(avatars.smpl_meshes[1].vertices, avatars.smpl_model.faces)
    # smpl_mesh3 = to_trimesh(avatars.smpl_meshes[2].vertices, avatars.smpl_model.faces)
    # show_meshes([smpl_mesh1, smpl_mesh2, smpl_mesh3])
    # show_meshes([avatars.meshes[0], avatars.meshes[1], avatars.meshes[2]])

    # cv2.imshow("test", avatars.mask[0])
    # cv2.waitKey(0)
    # cv2.imshow("test", avatars.mask[1])
    # cv2.waitKey(0)
    # cv2.imshow("test", avatars.mask[2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
