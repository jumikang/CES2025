from __future__ import annotations
import torch.utils.data
import smplx
import json
import os.path
from pysdf import SDF
from utils.geometry import *
from utils.loader_utils import *
# from chamferdist import ChamferDistance
from sklearn.neighbors import KDTree
from utils.visualizer import *
from utils.gradients import get_grad3


class HumanRigger(nn.Module):
    def __init__(self):
        super(HumanRigger, self).__init__()

        # instance variables.
        self.human_model = []
        self.num_images = 0
        self.new_lbs, self.v_posed = [], []
        self.new_semantic = []
        self.colorized_semantic = []
        self.colorized_smpl = []

    def auto_rig(self, smpl_mesh, scan_mesh, smpl_model):
        # assume there is no scale difference.
        x = smpl_mesh
        y = scan_mesh

        ref_vertices = x.vertices.detach().cpu().numpy()
        kdtree = KDTree(ref_vertices.squeeze(), leaf_size=30, metric='euclidean')

        # for debug (alignment checking)
        do_debug = False
        if do_debug:
            a = to_trimesh(ref_vertices, smpl_model.faces)
            b = to_trimesh(y.vertices, y.faces)
            show_meshes([a, b])

        # dic = self.human_model.v_label
        # smpl_semantic = np.zeros((10475, 1))
        # for i, key in enumerate(dic.keys()):
        #     smpl_semantic[dic[key], 0] = i
        #
        # smpl_semantic = torch.tensor(smpl_semantic).cuda().int()
        smpl_lbs = smpl_model.lbs_weights.cuda()

        interpolate = False
        if interpolate:
            dist, idx = kdtree.query(y.vertices, k=3, return_distance=True)
            n = dist.shape[0]

            d_sum = [sum(i) for i in dist]
            d_val = [d_sum[k] - dist[k, :] / d_sum[k] for k in range(n)]
            d_sum = [sum(d) for d in d_val]
            d_val = np.array(d_val)
            d_val = [d_val[k, :] / d_sum[k] for k in range(n)]

            new_lbs = torch.zeros(n, smpl_lbs.shape[1]).cuda()

            for k in range(n):
                new_lbs[k, :] = smpl_lbs[idx[k, 0], :] * d_val[k][0] + \
                                smpl_lbs[idx[k, 1], :] * d_val[k][1] + \
                                smpl_lbs[idx[k, 2], :] * d_val[k][2]
        else:
            kd_idx = kdtree.query(y.vertices, k=1, return_distance=False)
            new_lbs = smpl_lbs[kd_idx.squeeze(), :]

        # v_shaped = ?
        return new_lbs
        # colormap = torch.Tensor(np.random.rand(55, 3)).float().cuda()
        # self.new_semantic[frame_idx] = smpl_semantic[kd_idx.squeeze(), 0]
        # self.colorized_semantic[frame_idx] = colormap[self.new_semantic[frame_idx].long(), :]
        # self.colorized_lbs[frame_idx] = torch.matmul(self.new_lbs[frame_idx], colormap)
        # self.colorized_smpl_lbs[frame_idx] = torch.matmul(smpl_lbs, colormap)
        # self.colorized_smpl_semantic[frame_idx] = colormap[smpl_semantic.long(), :]
        # self.wrist_labels[frame_idx] = {'leftWrist': [], 'rightWrist': []}
        # self.v_label_new[frame_idx] = dict()
        #
        # s_map = dict()
        # for i, key in enumerate(dic.keys()):
        #     s_map[i] = key
        #     self.v_label_new[frame_idx][key] = []
        #
        # for k in range(len(kd_idx)):
        #     key = self.new_semantic[frame_idx][k]
        #     self.v_label_new[frame_idx][s_map[np.int(key)]].append(k)
        #
        # for i, key in enumerate(dic.keys()):
        #     if key == 'leftWrist':
        #         for k, val in enumerate(smpl_semantic[kd_idx.squeeze(), 0]):
        #             if i == val:
        #                 self.wrist_labels[frame_idx]['leftWrist'].append(k)
        #     elif key == 'rightWrist':
        #         for k, val in enumerate(smpl_semantic[kd_idx.squeeze(), 0]):
        #             if i == val:
        #                 self.wrist_labels[frame_idx]['rightWrist'].append(k)

    def forward(self, human_model, interpolate=False, verbose=True):
        start_time = time.time()
        self.num_images = len(human_model.meshes)
        self.human_model = human_model
        self.new_lbs = [None for _ in range(self.num_images)]
        self.v_posed = [None for _ in range(self.num_images)]
        self.human_model.smpl_colors = [None for _ in range(self.num_images)]
        self.new_semantic = [None for _ in range(self.num_images)]
        self.colorized_semantic = [None for _ in range(self.num_images)]
        self.colorized_lbs = [None for _ in range(self.num_images)]
        self.colorized_smpl_semantic = [None for _ in range(self.num_images)]
        self.colorized_smpl_lbs = [None for _ in range(self.num_images)]
        self.wrist_labels = [None for _ in range(self.num_images)]
        self.v_label_new = [None for _ in range(self.num_images)]

        for frame_idx in range(self.num_images):
            self.auto_rig(frame_idx, interpolate=interpolate)

        self.human_model.new_lbs = self.new_lbs

        for frame_idx in range(self.num_images):
            self.v_posed[frame_idx] = self.human_model.deform(frame_idx, inverse=True)

        # update human model.
        self.human_model.v_posed = self.v_posed
        self.human_model.custom_color_lbs = self.colorized_lbs
        self.human_model.custom_color_semantic = self.colorized_semantic
        self.human_model.custom_color_smpl_lbs = self.colorized_smpl_lbs
        self.human_model.custom_color_smpl_semantic = self.colorized_smpl_semantic
        self.human_model.wrist_labels = self.wrist_labels
        self.human_model.v_label_new = self.v_label_new

        if verbose:
            print("> It took {:.2f}s seconds for rigging {} images :-)".format(time.time() - start_time, self.num_images))
        return self.human_model


# class CanonicalFusion(nn.Module):
#     def __init__(self, num_iters=100):
#         super(CanonicalFusion, self).__init__()
#
#         self.num_iters = num_iters
#         self.c_loss = ChamferDistance()
#         self.l1_loss = nn.L1Loss()
#
#     def forward(self, human_model):
#         # smpl vs warped_model
#         # warped_model -> canonical -> resposed model
#         # reposed model - smpl model ...
#         # 2D joints ...
#         # update lbs and joint positions not 3D geometry.
#         # test them all.
#
#         return None
#
# class Visualizer():
#     def __init__(self):
#         super(Visualizer, self).__init__()
#
#     def show_templates(self):
#         return None
#
#     def show_flow(self):
#         return None
#
#     def show_canonical_frames(self):
#         return None
#
#
# class VoxelFlow(nn.Module):
#     def __init__(self, res=256, v_min=-1.0, v_max=1.0):
#         super(VoxelFlow, self).__init__()
#         """
#             res: resolution of the distance volume (res x res x res)
#             v_min, v_max: boundary of the volume in the real scale
#         """
#         self.res = res
#         self.interval = 2.0 / res
#         self.sdf_scale_factor = 100.0
#         self.num_epoch = num_epoch
#
#         self.w = nn.Parameter(torch.zeros(1, 1, self.res, self.res, self.res).cuda(), requires_grad=True)
#
#         # set grid and corresponding points.
#         self.x_ind, self.y_ind, self.z_ind = torch.meshgrid(torch.linspace(v_min, v_max, self.res),
#                                                             torch.linspace(v_min, v_max, self.res),
#                                                             torch.linspace(v_min, v_max, self.res), indexing='ij')
#         self.grid = torch.stack((self.z_ind, self.y_ind, self.x_ind), dim=0).cuda().unsqueeze(0)
#         self.grid = self.grid.permute(0, 2, 3, 4, 1).float()
#         self.pt = np.concatenate((np.asarray(self.x_ind).reshape(-1, 1),
#                                   np.asarray(self.y_ind).reshape(-1, 1),
#                                   np.asarray(self.z_ind).reshape(-1, 1)), axis=1)
#         self.pt = self.pt.astype(float)
#
#         self.l1_loss = nn.L1Loss()
#         self.optimizer = torch.optim.AdamW(self.parameters(), 0.1)
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#
#     @staticmethod
#     def get_smooth_loss(x, num_stencil=7):
#         if num_stencil == 7:
#             kernel = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
#                       [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
#             kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda() / 6.0
#         else:
#             kernel = [[[2, 3, 2], [3, 6, 3], [2, 3, 2]], [[3, 6, 3], [6, -88, 6], [3, 6, 3]],
#                       [[2, 3, 2], [3, 6, 3], [2, 3, 2]]]
#             kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda() / 26.0
#         grad_xx = F.conv3d(x[:, :, :, :, 0].unsqueeze(1), kernel, stride=1, padding=1)
#         grad_yy = F.conv3d(x[:, :, :, :, 1].unsqueeze(1), kernel, stride=1, padding=1)
#         grad_zz = F.conv3d(x[:, :, :, :, 2].unsqueeze(1), kernel, stride=1, padding=1)
#         return grad_xx.mean() + grad_yy.mean() + grad_zz.mean()
#
#     def get_sdf(self, x):
#         sdf_func = SDF(x.vertices, x.faces)
#         sdf = sdf_func(self.pt)
#         sdf = sdf.reshape((self.res, self.res, self.res)) * -1.0
#         out = np.expand_dims(sdf, axis=[0, 1])
#
#         return torch.Tensor(out).float().cuda() * self.sdf_scale_factor, sdf
#
#     def warp_vertices(self, vertices, flow, res, forward=True):
#         # (1) to improve the quality, change this to float() and for loop to barycentric interpolation.
#         src_px = torch.tensor(vertices).int()
#         mapped = torch.zeros_like(src_px).float()
#         for k in range(src_px.shape[0]):
#             if src_px[k, 0] < res and src_px[k, 1] < res and src_px[k, 2] < res:
#                 mapped[k, 0] = flow[0, src_px[k, 0], src_px[k, 1], src_px[k, 2], 2]
#                 mapped[k, 1] = flow[0, src_px[k, 0], src_px[k, 1], src_px[k, 2], 1]
#                 mapped[k, 2] = flow[0, src_px[k, 0], src_px[k, 1], src_px[k, 2], 0]
#
#         sign = -1 if forward is True else 1
#         return src_px.float() + sign * mapped
#
#     def get_flow(self, v1, v2, alpha=0.5):
#         """
#             res: resolution of the distance volume (res x res x res)
#             v_min, v_max: boundary of the volume in the real scale
#         """
#         d_t = v1 - v2
#         g_x1, g_y1, g_z1 = get_grad3(v1)
#         g_x2, g_y2, g_z2 = get_grad3(v2)
#         g_x = g_x1 * alpha + g_x2 * (1 - alpha)
#         g_y = g_y1 * alpha + g_y2 * (1 - alpha)
#         g_z = g_z1 * alpha + g_z2 * (1 - alpha)
#
#         f_x = -g_x * d_t * self.w
#         f_y = -g_y * d_t * self.w
#         f_z = -g_z * d_t * self.w
#
#         flow = torch.cat((f_x, f_y, f_z), dim=1).permute(0, 2, 3, 4, 1)
#         grid_flow_forward = torch.clamp(self.grid + flow * self.interval, min=-1.0, max=1.0)   # mesh grid conversion
#         grid_flow_backward = torch.clamp(self.grid - flow * self.interval, min=-1.0, max=1.0)
#         dst_warped = torch.nn.functional.grid_sample(v1, grid_flow_forward, align_corners=True)
#         src_warped = torch.nn.functional.grid_sample(v2, grid_flow_backward, align_corners=True)
#         return src_warped, dst_warped, flow
#
#     def forward(self, x, y):
#         # x.vertices = (x.vertices + 1) / self.interval
#         # y.vertices = (y.vertices + 1) / self.interval
#         src, src_np = self.get_sdf(x)
#         dst, dst_np = self.get_sdf(y)
#         src, dst = src.cuda(), dst.cuda()
#
#         self.train()
#         with tqdm.tqdm(range(self.num_epoch)) as pbar:
#             for k in pbar:
#                 sdf_src, sdf_dst, uvd = self.get_flow(src, dst)
#                 loss = self.l1_loss(sdf_dst, dst)
#                 loss += self.l1_loss(sdf_src, src)
#                 loss += 0.2 * torch.mean(torch.abs(uvd))
#                 loss += 0.2 * VoxelFlow.get_smooth_loss(uvd, num_stencil=7)
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 self.scheduler.step()
#                 pbar.set_description('iteration:{0}, loss:{1:.5f}'.format(k, loss.data))
#
#         self.eval()
#         src_warped, dst_warped, flow = self.get_flow(src, dst)
#         return flow, src_warped, dst_warped