from __future__ import annotations
import smplx
import torch.utils.data
import json
import os.path
from pysdf import SDF
from utils.geometry import *
from utils.loader_utils import *
# from chamferdist import ChamferDistance
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.visualizer import to_mesh, to_trimesh, show_meshes
from humanimate.animater_utils import deform_vertices
from sklearn.neighbors import KDTree
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


# abstract class to define multiple SMPLify modules for different tasks.
# Started on May 24, 2022, by M.Park
# Last modified on May 27, 2022, by M.Park
class AnimaterAbstract(ABC):
    def __init__(self,
                 num_iters=100,
                 voxel_res=256,
                 is_multiview=False,
                 is_seq=False):
        self.scan_joints = None
        self.smpl_joints = None
        self.num_iters = num_iters
        self.optimizer = None
        self.scheduler = None
        self.device = 'cuda'
        self.chamfer_loss = ChamferDistance()
        self.is_multiview = is_multiview
        self.is_seq = is_seq

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.smpl_model, self.smpl_meshes, self.smpl_params = [], [], []
        # self.pred_meshes, self.

        # set grid and corresponding points.
        v_min, v_max = -1.0, 1.0
        self.res = voxel_res
        self.x_ind, self.y_ind, self.z_ind = torch.meshgrid(torch.linspace(v_min, v_max, self.res),
                                                            torch.linspace(v_min, v_max, self.res),
                                                            torch.linspace(v_min, v_max, self.res), indexing='ij')
        self.grid = torch.stack((self.x_ind, self.y_ind, self.z_ind), dim=0).cuda().unsqueeze(0)
        self.grid = self.grid.permute(0, 2, 3, 4, 1).float()
        self.pt = np.concatenate((np.asarray(self.x_ind).reshape(-1, 1),
                                  np.asarray(self.y_ind).reshape(-1, 1),
                                  np.asarray(self.z_ind).reshape(-1, 1)), axis=1)
        self.pt = self.pt.astype(float)

    @abstractmethod
    def set_experimentation(self):
        "Set your own experiment"

    def set_full_poses(self, smpl_params, smpl_model):
        self.full_pose, self.smpl_vshaped = [], []
        for smpl in smpl_params:
            full_pose = torch.zeros(1, 165, dtype=torch.float32).to(self.device)
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

            smpl_vshaped = smpl_model.v_template + \
                           smplx.lbs.blend_shapes(smpl['betas'], smpl_model.shapedirs)
            self.full_pose.append(full_pose)
            self.smpl_vshaped.append(smpl_vshaped)

    def get_smpl_model(self, smpl_params, global_orient=None):
        # return None if self.smpl_model is None:
        if global_orient is not None:
            return self.smpl_model(transl=smpl_params['transl'],
                                   betas=smpl_params['betas'],
                                   body_pose=smpl_params['body_pose'],
                                   global_orient=global_orient,
                                   jaw_pose=smpl_params['jaw_pose'],
                                   joints=smpl_params['joints'],
                                   # expression=smpl_params['expression'],
                                   left_hand_pose=smpl_params['left_hand_pose'],
                                   right_hand_pose=smpl_params['right_hand_pose'],
                                   return_verts=True)
        else:
            return self.smpl_model(transl=smpl_params['transl'],
                                   betas=smpl_params['betas'],
                                   body_pose=smpl_params['body_pose'],
                                   global_orient=smpl_params['global_orient'],
                                   jaw_pose=smpl_params['jaw_pose'],
                                   joints=smpl_params['joints'],
                                   # expression=smpl_params['expression'],
                                   left_hand_pose=smpl_params['left_hand_pose'],
                                   right_hand_pose=smpl_params['right_hand_pose'],
                                   return_verts=True)

    def get_normalize_params(self, mesh):
        # assuming trimesh is given as input.
        centroid = mesh.bounding_box.centroid
        scale = 2.0 / np.max(mesh.bounding_box.extents)
        # mesh.vertices = (mesh.vertices - centroid) * scale
        return centroid, scale

    def denormalize_mesh(self, mesh, centroid, scale):
        mesh.vertices = mesh.vertices / scale + centroid
        return mesh

    # for understanding people.
    def visualize_wrists(self, avatars):
        pass
        # v_color = np.zeros_like(avatars.smpl_colors[0].detach().cpu().numpy())
        # for k in avatars.v_label['leftHand']:
        #     if k in avatars.v_label['leftForeArm']:
        #         v_color[k, :] = [255, 0, 0]
        # for k in avatars.v_label['rightHand']:
        #     if k in avatars.v_label['rightForeArm']:
        #         v_color[k, :] = [255, 0, 0]
        # t_posed, smpl_meshes = avatars.get_meshes(custom_color=True)
        # t_posed[0].show()

    def replace_hands(self, smpl_mesh, scan_mesh, image_front=None,
                      image_back=None, verbose=True):
        for k in range(len(scan_mesh)):
            if image_front is None or image_back is None:
                scan_mesh[k] = self.replace_hand(smpl_mesh[k], scan_mesh[k],
                                                 verbose=verbose, idx=k)
            else:
                scan_mesh[k] = self.replace_hand(smpl_mesh[k], scan_mesh[k],
                                                 image_front=image_front[k], image_back=image_back[k],
                                                 verbose=verbose, idx=k)
        return scan_mesh

    def replace_hand(self, smpl_mesh, scan_mesh, image_front=None, image_back=None, verbose=True, idx=0):
        start_time = time.time()
        interval = 2.0 / self.res
        # smpl_mesh = to_trimesh(smpl_mesh.vertices, self.smpl_model.faces)
        # set to the smpl coordinate.
        # show_meshes([smpl_mesh, scan_mesh])

        # scan_centroid_local, scan_scale_local = self.get_normalize_params(scan_mesh)
        # scan_mesh.vertices = (scan_mesh.vertices - scan_centroid_local) * scan_scale_local
        smpl_centroid_local, smpl_scale_local = self.get_normalize_params(smpl_mesh)
        scan_mesh.vertices = (scan_mesh.vertices - smpl_centroid_local) * smpl_scale_local
        smpl_mesh.vertices = (smpl_mesh.vertices - smpl_centroid_local) * smpl_scale_local

        # show_meshes([smpl_mesh, scan_mesh])
        sdf_smpl = SDF(smpl_mesh.vertices, smpl_mesh.faces)
        sdf_scan = SDF(scan_mesh.vertices, scan_mesh.faces)

        hand_idx = []
        non_hand_idx = []
        for key in self.v_label.keys():
            if 'leftHand' in key or 'rightHand' in key:
                hand_idx += self.v_label[key]
            else:
                non_hand_idx += self.v_label[key]

        kdtree1 = KDTree(smpl_mesh.vertices[hand_idx, :], leaf_size=30, metric='euclidean')
        kdtree2 = KDTree(smpl_mesh.vertices[non_hand_idx, :], leaf_size=30, metric='euclidean')

        dist1, idx1 = kdtree1.query(self.pt, k=1, return_distance=True)
        dist2, idx2 = kdtree2.query(self.pt, k=1, return_distance=True)
        sdf1 = sdf_smpl(self.pt)
        sdf2 = sdf_scan(self.pt)
        #
        offset = 0.2
        for k in range(len(dist1)):
            if dist2[k] > dist1[k] - offset and abs(sdf1[k]) < 1:
                alpha = dist1[k] / (dist1[k] + dist2[k])
                sdf2[k] = alpha * sdf2[k] + sdf1[k] * (1 - alpha)

        sdf = sdf2.reshape((self.res, self.res, self.res)) * -1.0

        #
        mesh = to_mesh(sdf)
        mesh.vertices = mesh.vertices * interval - 1.0
        mesh.vertices = mesh.vertices / smpl_scale_local + smpl_centroid_local
        mesh.vertices = (mesh.vertices - self.smpl_centroid[idx]) * self.smpl_scale[idx]
        mesh.vertices = mesh.vertices / self.scale[idx] + self.centroid[idx]

        if image_front is not None and image_back is not None:
            mesh, vertex_colors = colorize_model2(mesh, image_front, image_back)

        mesh.vertices = (mesh.vertices - self.centroid[idx]) * self.scale[idx]
        mesh.vertices = mesh.vertices/self.smpl_scale[idx] + self.smpl_centroid[idx]

        if verbose:
            print("> It took {:.2f}s seconds for changing hands :-)".format(time.time() - start_time))
        return mesh

    def canonicalize(self, human_model, affine_params):

        # self.smpl_lbs = self.smpl_model.lbs_weights
        self.smpl_lbs = human_model.smpl_model.lbs_weights
        self.set_full_poses(human_model.smpl_meshes, human_model.smpl_model)
        self.smpl_model = human_model.smpl_model

        n = len(human_model.meshes)
        # self.smpl_vshaped = [None for _ in range(n)]
        self.scan_vshaped = [None for _ in range(n)]
        self.smpl_joints = [None for _ in range(n)]
        self.scan_joints = [None for _ in range(n)]

        for k in range(n):
            scan_lbs = human_model.new_lbs[k].to(self.device)  # about to minimize

            # initially same and will change accordingly.
            self.smpl_joints[k] = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, self.smpl_vshaped[k])
            if self.scan_joints[k] is None:
                self.scan_joints[k] = torch.tensor(self.smpl_joints[k], dtype=torch.float32).to(self.device)

            # change it to the canonical mesh in the future.
            ref_vertices = deform_vertices(self.smpl_vshaped[k], self.smpl_joints[k],
                                           self.smpl_model, self.smpl_lbs, self.full_pose[k], inverse=False)

            centroid_smpl, scale_smpl = self.get_normalize_params(to_trimesh(ref_vertices, self.smpl_model.faces))
            scan_vertices = human_model.meshes[k].vertices
            smpl_vertices = human_model.smpl_meshes[k].vertices
            centroid_scan, scale_scan = self.get_normalize_params(to_trimesh(smpl_vertices, self.smpl_model.faces))

            scan_vertices = (scan_vertices - centroid_scan) * scale_scan
            scan_vertices = scan_vertices / scale_smpl + centroid_smpl
            self.scan_vshaped[k] = deform_vertices(torch.tensor(scan_vertices, dtype=torch.float32).unsqueeze(0).to(self.device),
                                                   self.scan_joints[k],
                                                   self.smpl_model,
                                                   scan_lbs,
                                                   self.full_pose[k], inverse=True)

            # need to fetch color information w.r.t. each vertex.
            affine_params[k]['smpl_centroid_t'] = centroid_smpl
            affine_params[k]['smpl_scale_t'] = scale_smpl
            affine_params[k]['scan_centroid_t'] = centroid_scan
            affine_params[k]['scan_scale_t'] = scale_scan

            human_model.smpl_meshes[k].vertices = ref_vertices
            human_model.meshes[k].vertices = scan_vertices
            human_model.v_posed[k] = self.scan_vshaped[k].double()
        self.verbose = False
        if self.verbose:
            print('t_posed humans')
            for k in range(n):
                a = to_trimesh(self.scan_vshaped[k], human_model.meshes[k].faces)
                b = to_trimesh(self.smpl_vshaped[k], human_model.smpl_model.faces)
                show_meshes([a, b])

                a = to_trimesh(human_model.meshes[k].vertices, human_model.meshes[k].faces)
                b = to_trimesh(human_model.smpl_meshes[k].vertices, human_model.smpl_model.faces)
                show_meshes([a, b])

        return human_model, affine_params


# multi-view fusion.
class CanonicalFusion(AnimaterAbstract):
    def set_experimentation(self):
        opt_params = []
        self.smpl_canonical = {}
        n = len(self.smpl_params)
        for key in self.smpl_params[0].keys():
            if torch.is_tensor(self.smpl_params[0][key]):
                self.smpl_canonical[key] = torch.zeros_like(self.smpl_params[0][key])

        for i in range(n):
            for key in self.smpl_params[i].keys():
                if torch.is_tensor(self.smpl_params[i][key]):
                    self.smpl_params[i][key].requires_grad = False
                else:
                    continue

                if key == 'global_orient':
                    self.smpl_params[i]['global_orient'].requires_grad = True
                    opt_params.append(self.smpl_params[i]['global_orient'])
                elif self.smpl_params[i][key] is not None:
                    self.smpl_canonical[key] += self.smpl_params[i][key].detach()

        for key in self.smpl_canonical.keys():
            self.smpl_canonical[key] /= n

        self.smpl_canonical['body_pose'].requires_grad = True
        self.smpl_canonical['betas'].requires_grad = True
        opt_params.append(self.smpl_canonical['body_pose'])
        opt_params.append(self.smpl_canonical['betas'])

        self.optimizer = torch.optim.Adam(opt_params, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    def single_image_exception(self, human_model):
        self.smpl_canonical = human_model.smpl_params[0]

        return human_model

    def __call__(self, human_model):
        if len(human_model.meshes) == 1:
            return self.single_image_exception(human_model)

        self.human_model = human_model
        self.smpl_model = human_model.smpl_model
        self.smpl_meshes = human_model.smpl_meshes
        self.pred_meshes = human_model.meshes
        self.smpl_params = human_model.smpl_params
        self.v_label = human_model.v_label
        self.v_label2 = human_model.v_label_new
        self.smpl_centroid = human_model.smpl_centroid
        self.smpl_scale = human_model.smpl_scale
        self.centroid = human_model.centroid
        self.scale = human_model.scale

        self.set_experimentation()
        result_meshes = []
        result_params = []
        print('Merging multiple predictions in the canonical space :)')
        with tqdm(range(self.num_iters)) as pbar:
            for i in pbar:
                loss = 0
                for k in range(len(self.pred_meshes)):
                    smpl_params = self.smpl_params[k]
                    smpl_mesh = self.get_smpl_model(self.smpl_canonical, smpl_params['global_orient'])
                    # smpl_mesh = self.get_smpl_model(smpl_params)
                    src = smpl_mesh.vertices
                    target = torch.Tensor(self.pred_meshes[k].vertices).cuda().unsqueeze(0).float()
                    target.requires_grad = True

                    # a = to_trimesh(src, self.smpl_model.faces)
                    # b = to_trimesh(target, self.pred_meshes[k].faces)
                    # show_meshes([a, b])

                    loss += self.chamfer_loss(src, target)

                    for key in ['leftForeArm', 'rightForeArm', 'leftArm', 'rightArm',
                                'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']:
                        loss += self.chamfer_loss(src[:, self.v_label[key], :], target[:, self.v_label2[k][key], :])

                    pbar.set_description('iteration:{0}, loss:{1:.5f}'.format(k, loss.data))
                    if i == self.num_iters - 1:
                        result_meshes.append(smpl_mesh)
                        result_params.append(smpl_params)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()

        human_model.smpl_meshes = result_meshes
        human_model.smpl_params = result_params

        return human_model


class AnimaterBasic(AnimaterAbstract):
    def set_experimentation(self):
        opt_params = []
        for i in range(len(self.smpl_params)):
            for key in self.smpl_params[i].keys():
                if torch.is_tensor(self.smpl_params[i][key]):
                    self.smpl_params[i][key].requires_grad = False
            self.smpl_params[i]['body_pose'].requires_grad = True
            self.smpl_params[i]['betas'].requires_grad = True
            self.smpl_params[i]['global_orient'].requires_grad = True
            opt_params.append(self.smpl_params[i]['global_orient'])
            opt_params.append(self.smpl_params[i]['body_pose'])
            opt_params.append(self.smpl_params[i]['betas'])

        self.optimizer = torch.optim.Adam(opt_params, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    def align_models(self, human_model):
        smpl_lbs = human_model.smpl_model.lbs_weights
        scan_lbs = human_model.new_lbs[0].to(self.device)  # about to minimize

        self.set_full_poses(self.smpl_meshes, self.smpl_model)

        # initially same and will change accordingly.
        self.smpl_joints = smplx.lbs.vertices2joints(human_model.smpl_model.J_regressor, self.smpl_vshaped[0])
        if self.scan_joints is None:
            self.scan_joints = torch.tensor(self.smpl_joints)

        # change it to the canonical mesh in the future.
        ref_vertices = deform_vertices(self.smpl_vshaped[0], self.smpl_joints,
                                       human_model.smpl_model, smpl_lbs, self.full_pose[0], inverse=False)
        centroid_smpl, scale_smpl = self.get_normalize_params(to_trimesh(ref_vertices, self.smpl_model.faces))
        scan_vertices = torch.tensor(self.human_model.meshes[0].vertices, dtype=torch.float32).to(
            self.device).unsqueeze(0)
        smpl_vertices = torch.tensor(self.human_model.smpl_meshes[0].vertices, dtype=torch.float32).to(
            self.device)
        centroid_scan, scale_scan = self.get_normalize_params(to_trimesh(smpl_vertices, human_model.smpl_model.faces))

        centroid_smpl = torch.tensor(centroid_smpl, dtype=torch.float32).to(self.device)
        centroid_scan = torch.tensor(centroid_scan, dtype=torch.float32).to(self.device)
        scale_smpl = torch.tensor(scale_smpl, dtype=torch.float32).to(self.device)
        scale_scan = torch.tensor(scale_scan, dtype=torch.float32).to(self.device)

        ref_vertices = (ref_vertices - centroid_smpl) * scale_smpl
        ref_vertices = ref_vertices / scale_scan + centroid_scan

        human_model.smpl_params[0]['vertices'] = ref_vertices
        self.verbose = True
        if self.verbose:
            print('t_posed humans')
            a = to_trimesh(scan_vertices, human_model.meshes[0].faces)
            b = to_trimesh(ref_vertices, human_model.smpl_model.faces)
            c = to_trimesh(smpl_vertices, human_model.smpl_model.faces)
            show_meshes([a, b, c], offset=[-1, 0.0, 1])
            # show_meshes([b, c])

        return human_model

    # optimize for the loss function
    def __call__(self, human_model):
        self.human_model = human_model
        self.smpl_model = human_model.smpl_model
        self.smpl_meshes = human_model.smpl_meshes
        self.pred_meshes = human_model.meshes
        self.smpl_params = human_model.smpl_params
        self.v_label = human_model.v_label
        self.v_label2 = human_model.v_label_new
        self.smpl_centroid = human_model.smpl_centroid
        self.smpl_scale = human_model.smpl_scale
        self.centroid = human_model.centroid
        self.scale = human_model.scale

        self.set_experimentation()
        n = len(self.pred_meshes)
        result_meshes = [None for _ in range(n)]
        with tqdm(range(self.num_iters)) as pbar:
            for i in pbar:
                loss = 0.0
                for k in range(n):
                    smpl_mesh = self.get_smpl_model(self.smpl_params[k], global_orient=self.smpl_params[k]['global_orient'])
                    if i == self.num_iters - 1:
                        result_meshes[k] = smpl_mesh
                    src = smpl_mesh.vertices
                    target = torch.Tensor(self.pred_meshes[k].vertices).cuda().unsqueeze(0).float()
                    target.requires_grad = True

                    # loss += self.chamfer_loss(src, target)
                    loss_c, _ = chamfer_distance(src, target)
                    loss += loss_c
                    for key in ['leftForeArm', 'rightForeArm', 'leftArm', 'rightArm',
                                'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']:
                        loss_b, _ = chamfer_distance(src, target)
                        # loss += self.chamfer_loss(src[:, self.v_label[key], :], target[:, self.v_label2[k][key], :])
                        loss += loss_b

                # set consistency loss among different inputs.
                if n > 1:
                    if self.is_multiview:
                        for key in self.smpl_params[0].keys():
                            if torch.is_tensor(self.smpl_params[0][key]) and key is not 'global_orient':
                                for j in range(1, n):
                                    loss += self.l1_loss(self.smpl_params[j-1][key], self.smpl_params[j][key])
                                loss += self.l1_loss(self.smpl_params[0][key], self.smpl_params[-1][key])
                    elif self.is_seq:
                        # constraining the shape since it is the only feature that does not change over time.
                        key = 'betas'
                        for j in range(1, n):
                            loss += self.l1_loss(self.smpl_params[j - 1][key], self.smpl_params[j][key])
                        loss += self.l1_loss(self.smpl_params[0][key], self.smpl_params[-1][key])

                # pbar.set_description('iteration:{0}, loss:{1:.5f}'.format(i, loss.data))
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()

        human_model.smpl_meshes = result_meshes
        human_model.smpl_params = self.smpl_params
        human_model.meshes = self.pred_meshes
        # for k in range(n):
        #     smpl_mesh = self.get_smpl_model(self.smpl_params[k])
        #     a = to_trimesh(smpl_mesh.vertices, self.smpl_model.faces)
        #     b = to_trimesh(self.pred_meshes[k].vertices, self.pred_meshes[k].faces)
        #     show_meshes([a, b])
        return human_model


class AnimaterLBS(AnimaterAbstract):
    def deform(self, mv):
        global_orient = torch.tensor(mv[:3]).unsqueeze(0)
        body_pose = torch.tensor(mv[3:-6]).unsqueeze(0)
        # self.full_pose = mv.view(1, -1)
        self.full_pose[0, :3] = global_orient.reshape(-1, 3)
        self.full_pose[0, 3:66] = body_pose.reshape(-1, 63)
        # self.full_pose = torch.tensor(mv).unsqueeze(0)
        reposed = deform_vertices(self.scan_vshaped, self.scan_joints,
                                  self.smpl_model, self.scan_lbs, self.full_pose, inverse=False)

        return to_trimesh(reposed, self.human_model.meshes[0].faces, vertex_colors=self.human_model.meshes[0].visual.vertex_colors)

    def set_full_pose(self, smpl):
        self.full_pose = torch.zeros(1, 165, dtype=torch.float32).to(self.device)
        # smpl = self.smpl_params[0]
        if 'global_orient' in smpl and smpl['global_orient'] is not None:
            self.full_pose[0, :3] = smpl['global_orient'].reshape(-1, 3)
        if 'body_pose' in smpl and smpl['body_pose'] is not None:
            self.full_pose[0, 3:66] = smpl['body_pose'].reshape(-1, 63)
        if 'jaw_pose' in smpl and smpl['jaw_pose'] is not None:
            self.full_pose[0, 66:69] = smpl['jaw_pose'].reshape(-1, 3)
        if 'left_eye' in smpl and smpl['left_eye'] is not None:
            self.full_pose[0, 69:72] = smpl['left_eye'].reshape(-1, 3)
        if 'right_eye' in smpl and smpl['right_eye'] is not None:
            self.full_pose[0, 72:75] = smpl['right_eye'].reshape(-1, 3)
        if 'left_hand_pose' in smpl and smpl['left_hand_pose'] is not None:
            self.full_pose[0, 75:120] = smpl['left_hand_pose'].reshape(-1, 45)
        if 'right_hand_pose' in smpl and smpl['right_hand_pose'] is not None:
            self.full_pose[0, 120:165] = smpl['right_hand_pose'].reshape(-1, 45)

    def set_experimentation(self):
        opt_params = []

        self.smpl_lbs = self.smpl_model.lbs_weights
        self.scan_lbs = self.human_model.new_lbs[0].to(self.device)  # about to minimize
        # self.full_pose = self.human_model.full_pose[0]
        # full poses for reference, canonical, and target.
        self.set_full_pose(self.smpl_params[0])
        self.smpl_vshaped = self.smpl_model.v_template + \
                   smplx.lbs.blend_shapes(self.smpl_params[0]['betas'], self.smpl_model.shapedirs)

        # initially same and will change accordingly.
        self.smpl_joints = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, self.smpl_vshaped)
        if self.scan_joints is None:
            self.scan_joints = torch.tensor(self.smpl_joints)

        # change it to the canonical mesh in the future.
        ref_vertices = deform_vertices(self.smpl_vshaped, self.smpl_joints,
                                             self.smpl_model, self.smpl_lbs, self.full_pose, inverse=False)
        centroid_smpl, scale_smpl = self.get_normalize_params(to_trimesh(ref_vertices, self.smpl_model.faces))
        self.scan_vertices = torch.tensor(self.human_model.meshes[0].vertices, dtype=torch.float32).to(self.device).unsqueeze(0)
        self.smpl_vertices = torch.tensor(self.human_model.smpl_meshes[0].vertices, dtype=torch.float32).to(self.device)
        centroid_scan, scale_scan = self.get_normalize_params(to_trimesh(self.smpl_vertices, self.smpl_model.faces))

        centroid_smpl = torch.tensor(centroid_smpl, dtype=torch.float32).to(self.device)
        centroid_scan = torch.tensor(centroid_scan, dtype=torch.float32).to(self.device)
        scale_smpl = torch.tensor(scale_smpl, dtype=torch.float32).to(self.device)
        scale_scan = torch.tensor(scale_scan, dtype=torch.float32).to(self.device)
        self.scan_vertices = (self.scan_vertices - centroid_scan) * scale_scan
        self.scan_vertices = self.scan_vertices / scale_smpl + centroid_smpl
        self.smpl_vertices = (self.smpl_vertices - centroid_scan) * scale_scan
        self.smpl_vertices = self.smpl_vertices / scale_smpl + centroid_smpl

        self.scan_vshaped = deform_vertices(self.scan_vertices, self.scan_joints,
                                            self.smpl_model, self.scan_lbs, self.full_pose, inverse=True)

        self.verbose = False
        if self.verbose:
            print('t_posed humans')
            a = to_trimesh(self.scan_vshaped, self.human_model.meshes[0].faces)
            b = to_trimesh(self.smpl_vshaped, self.smpl_model.faces)
            show_meshes([a, b], offset=[-1.0, 1.0])

        self.smpl_joints = self.smpl_joints.detach()
        self.scan_joints = self.scan_joints.detach()
        self.scan_vshaped = self.scan_vshaped.detach()
        self.smpl_vshaped = self.smpl_vshaped.detach()

        self.optimize_lbs = False
        self.optimize_jts = True

        for i in range(len(self.smpl_params)):
            for key in self.smpl_params[i].keys():
                if torch.is_tensor(self.smpl_params[i][key]):
                    self.smpl_params[i][key].requires_grad = False
            if self.optimize_jts:
                self.scan_joints.requires_grad = True
                opt_params.append(self.scan_joints)
            else:
                self.scan_joints.requires_grad = False
            if self.optimize_lbs:
                self.scan_lbs.requires_grad = True
                opt_params.append(self.scan_lbs)
            else:
                self.scan_lbs.requires_grad = False

        self.optimizer = torch.optim.Adam(opt_params, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.999 ** epoch)

    # optimize for the loss function
    def __call__(self, human_model, visualizer):
        self.human_model = human_model
        self.smpl_params = human_model.smpl_params
        self.smpl_model = human_model.smpl_model
        self.target_pose = torch.zeros(1, 165, dtype=torch.float32).to(self.device)
        self.set_experimentation()

        optimized_jts = []
        optimized_lbs = []
        # original_lbs = self.scan_lbs.detach()
        with tqdm(range(self.num_iters)) as pbar:
            for i in pbar:
                loss = 0
                for k in range(len(self.smpl_params)):
                    # initialize different pose at each iteration.
                    motion = visualizer.get_random_motion()
                    self.target_pose[0, 3:66] = torch.tensor(motion[3:-6]).unsqueeze(0)

                    # normalize lbs to be summed into 1 for each vertex.
                    scan_lbs = F.normalize(self.scan_lbs, dim=1, p=1)

                    # deform to the target pose.
                    reposed_scan = deform_vertices(self.scan_vshaped, self.scan_joints,
                                                   self.smpl_model, scan_lbs, self.target_pose, inverse=False)
                    reposed_smpl = deform_vertices(self.smpl_vshaped, self.smpl_joints,
                                                   self.smpl_model, self.smpl_lbs, self.target_pose, inverse=False)

                    # deform to the original pose.
                    refpose_scan = deform_vertices(self.scan_vshaped, self.scan_joints,
                                                  self.smpl_model, scan_lbs, self.full_pose, inverse=False)
                    refpose_smpl = deform_vertices(self.smpl_vshaped, self.smpl_joints,
                                                  self.smpl_model, self.smpl_lbs, self.full_pose, inverse=False)

                    # loss += self.l1_loss(refpose_scan, self.scan_vertices)
                    # loss += self.chamfer_loss(reposed_scan, reposed_smpl)
                    # loss += self.chamfer_loss(refpose_scan, refpose_smpl)
                    loss_s, _ = chamfer_distance(reposed_scan, reposed_smpl)
                    loss_ca, _ = chamfer_distance(refpose_scan, refpose_smpl)
                    loss += loss_s
                    loss += loss_ca

                    if self.verbose is True and i % 100 == 0:
                        a = to_trimesh(reposed_scan, self.human_model.meshes[0].faces)
                        b = to_trimesh(reposed_smpl, self.smpl_model.faces)
                        show_meshes([a, b], offset=[-0.5, 0.5])
                    pbar.set_description('iteration:{0}, loss:{1:.5f}'.format(k, loss.data))
                    if i == self.num_iters - 1:
                        if self.optimize_lbs:
                            optimized_lbs.append(self.scan_lbs)
                        if self.optimize_jts:
                            optimized_jts.append(self.scan_joints)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()

        # human_model.smpl_meshes = result
        if self.optimize_jts:
            human_model.new_joints = optimized_jts
        if self.optimize_lbs:
            human_model.new_lbs = optimized_lbs
        return human_model

