from __future__ import annotations
import torch.utils.data
import trimesh.remesh
import numpy as np
# from pysdf import SDF
# from torchmcubes import marching_cubes, grid_interp
from sklearn.neighbors import KDTree
from tqdm import tqdm
from torch import nn
import os
# from utils.loader_utils import *
from lib.utils.visualizer import o3d2trimesh
# from pytorch3d.structures import Meshes, Pointclouds
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.loss import (
#     chamfer_distance,
#     point_mesh_edge_distance,
#     point_mesh_face_distance,
#     mesh_edge_loss,
#     mesh_laplacian_smoothing,
#     mesh_normal_consistency,
# )
import open3d as o3d
from src.lib.utils.visualizer import trimesh2o3d
from src.smpl_optimizer.smpl_utils import init_semantic_labels, get_near_and_far_points



def mesh_smoothness_custom(meshes, vertex):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    l2_loss = nn.L1Loss()
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]
    v0_ori = vertex[faces_packed[:, 0]]
    v1_ori = vertex[faces_packed[:, 1]]
    v2_ori = vertex[faces_packed[:, 2]]
    e0 = v0 - v0_ori
    e1 = v1 - v1_ori
    e2 = v2 - v2_ori

    loss = l2_loss(e0, e1) + l2_loss(e1, e2) + l2_loss(e2, e0)
    return loss.sum()


def mesh_edge_loss_custom(meshes):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    l2_loss = nn.MSELoss()
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    v0 = verts_packed[faces_packed[:, 0]]
    v1 = verts_packed[faces_packed[:, 1]]
    v2 = verts_packed[faces_packed[:, 2]]

    e0 = (v0 - v1).norm(dim=1, p=2)
    e1 = (v1 - v2).norm(dim=1, p=2)
    e2 = (v2 - v0).norm(dim=1, p=2)
    # avg_len = torch.mean(e0 + e1 + e2)/3
    loss = l2_loss(e0, e1) + l2_loss(e1, e2) + l2_loss(e2, e0)
    # + l2_loss(e0, avg_len) + l2_loss(e1, avg_len) + l2_loss(e2, avg_len)
    return loss.sum()


class ARAPOptimizer(nn.Module):
    def __init__(self, cam_params, path2semantic, device='cuda:0'):
        super(ARAPOptimizer, self).__init__()
        # set device
        self.device = device

        # set camera parameters
        self.K = np.eye(3)
        self.K[0, 0], self.K[1, 1] = cam_params['fx'], cam_params['fy']
        self.K[0, 2], self.K[1, 2] = cam_params['px'], cam_params['py']
        self.R = np.eye(3)
        self.R[1, 1] *= -1.0
        self.R[2, 2] *= -1.0
        self.t = np.array(cam_params['cam_center']) / 100.0
        self.K = torch.FloatTensor(self.K[None, :, :]).to(self.device)
        self.R = torch.FloatTensor(self.R[None, :, :]).to(self.device)
        self.t = torch.FloatTensor(self.t[None, :]).to(self.device)
        self.width = cam_params['width']
        self.height = cam_params['height']

        # v_label(json file) must exist
        self.v_label = init_semantic_labels(path2semantic)

    def set_body_parts(self, target='left_arm'):
        if target == 'left_arm':
            static_label = self.v_label['body_idx']
            pass
        elif target == 'right_arm':
            pass
        elif target == 'left_leg':
            pass
        elif target == 'right_leg':
            pass

    def forward(self, scan_mesh, smpl_mesh, smpl_params, path2pose, smpl_optimizer,keypoints=None):
        keypoints = smpl_optimizer.set_openpose2smpl_keypoints(path2pose=path2pose, keypoints=keypoints)
        scan_near, scan_far = smpl_optimizer.get_3d_keypoints_near_far(scan_mesh, keypoints[0])

        smpl_joints = smpl_params['joints_2d']
        smpl_near, smpl_far = smpl_optimizer.get_3d_keypoints_near_far(smpl_mesh, smpl_joints.reshape(144, 2))

        mesh_o3d = trimesh2o3d(smpl_mesh)
        vertices = np.asarray(mesh_o3d.vertices)

        # exclude target's indices
        # static_ids = [idx for idx in np.where(vertices[:, 1] > 0.0)[0]]
        static_ids = [idx for idx in np.where(vertices[:, 1] < 0.0)[0]]
        static_ids += self.v_label['head']

        ids_dict = {'left_elbow': 18, 'right_elbow': 19,
                    'left_wrist': 20, 'right_wrist': 21,
                    'left_knee': 4, 'right_knee': 5,
                    'left_ankle': 7, 'right_ankle': 8,
                    'left_collar': 13, 'right_collar': 14,
                    'left_shoulder': 16, 'right_shoulder': 17,
                    'left_foot': 10, 'right_foot': 11,
                    'left_big_toe': 60, 'left_small_toe': 61, 'left_heel': 62,
                    'right_big_toe': 63, 'right_small_toe': 64, 'right_heel': 65}

        static_pos = []
        for id in static_ids:
            static_pos.append(vertices[id])
        # handle_ids = [9120]  # nose
        # handle_pos = [vertices[9120] + np.array((0, 0, 0.3))]
        # move limbs
        kdtree = KDTree(smpl_mesh.vertices, leaf_size=30, metric='euclidean')
        idx_near = kdtree.query(smpl_near, k=1, return_distance=False)
        idx_far = kdtree.query(smpl_far, k=1, return_distance=False)

        handle_ids, handle_pos = [], []
        def get_handle(handle_ids, handle_pos, target='left_elbow'):
            handle_ids += [idx_near[ids_dict[target]]]
            handle_pos += [vertices[idx_near[ids_dict[target]]].squeeze() +
                           scan_near[ids_dict[target]] - smpl_near[ids_dict[target]]]
            handle_ids += [idx_far[ids_dict[target]]]
            handle_pos += [vertices[idx_far[ids_dict[target]]].squeeze() +
                           scan_far[ids_dict[target]] - smpl_far[ids_dict[target]]]
            return handle_ids, handle_pos

        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='left_shoulder')
        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='right_shoulder')
        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='left_elbow')
        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='left_wrist')
        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='right_elbow')
        handle_ids, handle_pos = get_handle(handle_ids, handle_pos, target='right_wrist')

        # keep hands and feet static.
        constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
        constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

        # with o3d.utility.VerbosityContextManager as cm:
        mesh_prime = mesh_o3d.deform_as_rigid_as_possible(constraint_ids,
                                                          constraint_pos,
                                                          max_iter=50)
        return o3d2trimesh(mesh_prime)


class Mesh2PointOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(Mesh2PointOptimizer, self).__init__()
        self.device = device

    def forward(self, src_vts, src_faces, trg_vts, trg_color, return_verts=False, exclude_idx=None):
        src_mesh = Meshes(verts=[src_vts],
                          faces=[torch.Tensor(src_faces)]).to(self.device)

        trg_pcd = Pointclouds(points=[trg_vts],
                              features=[trg_color]).to(self.device)

        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        mask = torch.ones_like(deform_verts).to(self.device)
        if exclude_idx is not None and len(exclude_idx) > 0:
            mask[exclude_idx, :] = 0.0
        optimizer = torch.optim.SGD([deform_verts], lr=0.01, momentum=0.99)

        # Number of optimization steps
        iter = 500
        # Weight for the chamfer loss
        w_chamfer = 0.8
        # Weight for mesh edge loss
        w_edge = 1.0
        # Weight for mesh normal consistency
        w_normal = 0.1
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.05

        new_src_mesh = src_mesh.offset_verts(deform_verts)
        # new_src_pcd = src_pcd.offset_verts(deform_verts)

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for i in pbar:
                # Initialize optimizer
                optimizer.zero_grad()

                # We sample 10k points from the surface of each mesh
                sample_src = sample_points_from_meshes(new_src_mesh, trg_vts.shape[0], return_normals=False)
                loss_chamfer, _ = chamfer_distance(sample_src, trg_pcd)

                # We compare the two sets of pointclouds by computing (a) the chamfer loss
                loss_dist1 = point_mesh_edge_distance(new_src_mesh, trg_pcd)
                loss_dist2 = point_mesh_face_distance(new_src_mesh, trg_pcd)

                # and (b) the edge length of the predicted mesh
                loss_edge = mesh_edge_loss(new_src_mesh)
                loss_edge2 = mesh_edge_loss_custom(new_src_mesh)

                # mesh normal consistency
                loss_normal = mesh_normal_consistency(new_src_mesh)

                # mesh laplacian smoothing
                loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

                # Weighted sum of the losses
                loss = loss_chamfer + loss_dist1 + loss_dist2 \
                       + loss_edge * w_edge \
                       + loss_normal * w_normal \
                       + loss_laplacian * w_laplacian \

                # Print the losses
                pbar.set_description('total_loss = {0:.6f}'.format(loss))

                # Optimization step
                loss.backward()
                optimizer.step()

                # Deform the mesh
                new_src_mesh = src_mesh.offset_verts(deform_verts * mask)

        if return_verts:
            return new_src_mesh[0].verts_packed().detach().cpu().numpy()
        else:
            vis_mesh = trimesh.Trimesh(new_src_mesh[0].verts_packed().detach().cpu().numpy(),
                                       new_src_mesh[0].faces_packed().detach().cpu().numpy(), process=False)
            return vis_mesh


class SMPL2MeshOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(SMPL2MeshOptimizer, self).__init__()
        self.device = device

    def forward(self, src_mesh_trimesh, trg_mesh_trimesh, lr=0.1, iter=500, scale_factor=1.0, mode='init'):
        src_mesh_trimesh = src_mesh_trimesh.copy()
        trg_mesh_trimesh = trg_mesh_trimesh.copy()
        src_mesh_trimesh.vertices = src_mesh_trimesh.vertices * scale_factor
        trg_mesh_trimesh.vertices = trg_mesh_trimesh.vertices * scale_factor
        src_mesh = Meshes(verts=[torch.Tensor(src_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(src_mesh_trimesh.faces)]).to(self.device)
        trg_mesh = Meshes(verts=[torch.Tensor(trg_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(trg_mesh_trimesh.faces)]).to(self.device)
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.99)

        # weights
        if mode == 'init':
            w = {'chamfer': 1.0, 'face': 0.1, 'normal': 0.1, 'laplacian': 0.1,
                 'symmetry': 0.1, 'consistency': 0.1, 'penetration': 0.1, 'smooth': 0.1}
        else:
            w = {'chamfer': 1.0, 'face': 0.1, 'normal': 0.1, 'laplacian': 0.1,
                 'symmetry': 0.1, 'consistency': 0.1, 'penetration': 0.0, 'smooth': 0.1}

        # pre-defined variables (constants)
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        initial_normals = src_mesh.verts_normals_packed().detach().clone()
        initial_vertices = src_mesh.verts_packed().detach().clone()
        x_flip_mask = torch.ones_like(initial_normals).to(self.device)
        x_flip_mask[:, 0] *= -1.0
        l2_loss = nn.MSELoss()

        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                loss = 0.0
                # (a) chamfer loss (basic loss) - sampling-based or using entire vertices
                # sample_src = sample_points_from_meshes(new_src_mesh, 300000)
                # sample_trg = sample_points_from_meshes(trg_mesh, 300000)
                # loss_chamfer_dist, _ = chamfer_distance(sample_src.verts, sample_trg)
                loss_chamfer_dist, _ = chamfer_distance(new_src_mesh.verts_packed().unsqueeze(0),
                                                        trg_mesh.verts_packed().unsqueeze(0))
                loss += loss_chamfer_dist * w['chamfer']

                # (b) constraining the size and shape of faces and edges
                if w['face'] > 0.0:
                    loss_face = mesh_edge_loss_custom(new_src_mesh)
                    loss += loss_face * w['face']

                # (c) mesh normal consistency
                if w['normal'] > 0.0:
                    loss_normal = mesh_normal_consistency(new_src_mesh)
                    loss += loss_normal * w['normal']

                # (d) mesh smoothness ( |d_x| ~ |d_x_n| )
                if w['smooth'] > 0.0:
                    loss_smoothness = mesh_smoothness_custom(new_src_mesh, initial_vertices)
                    loss += loss_smoothness * w['smooth']

                # (e) mesh laplacian smoothing
                if w['laplacian'] > 0.0:
                    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, 'cot')
                    loss += loss_laplacian * w['laplacian']

                # (f) keep the normal consistent with initial (smpl) normals
                if w['consistency'] > 0.0:
                    loss_normal_consistency = l2_loss(new_src_mesh.verts_normals_packed(), initial_normals)
                    loss += loss_normal_consistency * w['consistency']

                # (g) keep the symmetry of canonical mesh
                if w['symmetry'] > 0.0:
                    loss_flip, _ = chamfer_distance(new_src_mesh.verts_packed().unsqueeze(0),
                                                    new_src_mesh.verts_packed().unsqueeze(0)*x_flip_mask.unsqueeze(0))
                    loss += loss_flip * w['symmetry']

                # (h) avoid penetration (encourage deformation outward skinned body)
                if w['penetration'] > 0.0:
                    cur_dir = (new_src_mesh.verts_packed() - initial_vertices) * initial_normals
                    loss_penetration = torch.mean(torch.abs(torch.clamp(torch.sum(cur_dir, dim=1), max=0.0)))
                    loss += loss_penetration * w['penetration']

                pbar.set_description('total_loss = {0:.6f}'.format(loss))
                loss.backward()
                optimizer.step()

                # update the current mesh
                new_src_mesh = src_mesh.offset_verts(deform_verts)
                initial_normals = src_mesh.verts_normals_packed().detach().clone()
                initial_vertices = src_mesh.verts_packed().detach().clone()

        # generating the output mesh.
        vertices = new_src_mesh[0].verts_packed().detach().cpu().numpy()
        faces = new_src_mesh[0].faces_packed().detach().cpu().numpy()
        tmp_mesh = trimesh.Trimesh(vertices, faces)
        tmp_mesh = trimesh.smoothing.filter_laplacian(tmp_mesh)
        # tmp_mesh = self.remesh(tmp_mesh)  # to generate mesh again
        # tmp_mesh = self.postprocess_mesh(tmp_mesh, num_faces=1000)
        kdtree = KDTree(trg_mesh_trimesh.vertices, leaf_size=30, metric='euclidean')
        idx = kdtree.query(tmp_mesh.vertices, k=1, return_distance=False)
        vis_mesh = trimesh.Trimesh(tmp_mesh.vertices/scale_factor, tmp_mesh.faces,
                                   vertex_colors=trg_mesh_trimesh.visual.vertex_colors[idx[:, 0], :])
        return vis_mesh

    @staticmethod
    def remesh(scan_mesh, scale_factor=1.0):
        def _get_grid_coord_(v_min, v_max, res):
            x_ind, y_ind, z_ind = torch.meshgrid(torch.linspace(v_min[0], v_max[0], res[0]),
                                                 torch.linspace(v_min[1], v_max[1], res[1]),
                                                 torch.linspace(v_min[2], v_max[2], res[2]), indexing='ij')
            grid = torch.stack((x_ind, y_ind, z_ind), dim=0)
            grid = grid.float()
            pt = np.concatenate((np.asarray(x_ind).reshape(-1, 1),
                                 np.asarray(y_ind).reshape(-1, 1),
                                 np.asarray(z_ind).reshape(-1, 1)), axis=1)
            pt = pt.astype(float)
            return pt, grid.permute(0, 3, 2, 1).contiguous()

        def get_mesh(volume, grid_coord, scale_factor=1.0):
            # mesh generation.
            if isinstance(volume, np.ndarray):
                volume = torch.Tensor(volume)
            vertices, faces = marching_cubes(volume, 0.0)
            new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
            new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
            new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
            return new_mesh

        if scale_factor is not None:
            scan_mesh.vertices *= scale_factor

        v_margin = 3
        v_min = np.floor(np.min((scan_mesh.bounds[0], scan_mesh.bounds[0]), axis=0)).astype(int)
        v_max = np.ceil(np.max((scan_mesh.bounds[1], scan_mesh.bounds[1]), axis=0)).astype(int)
        v_min -= v_margin
        v_max += v_margin
        res = (v_max - v_min) * 2

        query_pts, grid_coord = _get_grid_coord_(v_min, v_max, res=res)
        sdf_gt = SDF(scan_mesh.vertices, scan_mesh.faces)
        volume_gt = sdf_gt(query_pts)
        mesh_remehed = get_mesh(volume_gt.reshape(res), grid_coord)

        kdtree = KDTree(scan_mesh.vertices, leaf_size=30, metric='euclidean')
        kd_idx = kdtree.query(mesh_remehed.vertices, k=1, return_distance=False)
        vertex_colors = scan_mesh.visual.vertex_colors[kd_idx.squeeze(), :]

        if scale_factor is not None:
            mesh_remehed.vertices = mesh_remehed.vertices / scale_factor

        remeshed_gt = trimesh.Trimesh(mesh_remehed.vertices,
                                      mesh_remehed.faces,
                                      vertex_colors=vertex_colors)
        return remeshed_gt

    @staticmethod
    def postprocess_mesh(mesh, num_faces=None):
        """Post processing mesh by removing small isolated pieces.

        Args:
            mesh (trimesh.Trimesh): input mesh to be processed
            num_faces (int, optional): min face num threshold. Defaults to 4096.
        """
        total_num_faces = len(mesh.faces)
        if num_faces is None:
            num_faces = total_num_faces // 100
        cc = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=3)
        mask = np.zeros(total_num_faces, dtype=bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh


class Mesh2MeshOptimizer(nn.Module):
    """Implementation of single-stage mesh deformation."""
    def __init__(self, device='cuda:0'):
        super(Mesh2MeshOptimizer, self).__init__()
        self.device = device

    def forward(self, src_mesh_trimesh, trg_mesh_trimesh, return_verts=False, lr=0.1, iter=500):
        src_mesh = Meshes(verts=[torch.Tensor(src_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(src_mesh_trimesh.faces)]).to(self.device)
        trg_mesh = Meshes(verts=[torch.Tensor(trg_mesh_trimesh.vertices)],
                          faces=[torch.Tensor(trg_mesh_trimesh.faces)]).to(self.device)
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.99)

        # Number of optimization steps
        w = {'chamfer': 0.8, 'edge': 0.5, 'normal': 0.5, 'laplacian': 0.1,
             'face': 0.1, 'consistency': 0.5, 'penetration': 0.5, 'smooth': 0.5}

        new_src_mesh = src_mesh.offset_verts(deform_verts)
        with tqdm(range(iter), position=0, ncols=100) as pbar:
            for i in pbar:
                # Initialize optimizer
                optimizer.zero_grad()

                # We sample 10k points from the surface of each mesh
                sample_src, sample_src_normal = sample_points_from_meshes(new_src_mesh, 100000, return_normals=True)
                sample_trg, sample_trg_normal = sample_points_from_meshes(trg_mesh, 100000, return_normals=True)

                loss_chamfer_dist, loss_chamfer_normal = chamfer_distance(sample_src,
                                                                          sample_trg,
                                                                          x_normals=sample_src_normal,
                                                                          y_normals=sample_trg_normal)
                loss = loss_chamfer_dist * w['chamfer'] + loss_chamfer_normal * 0.2

                # and (b) the edge length of the predicted mesh
                if w['face'] > 0.0:
                    loss_face = mesh_edge_loss_custom(new_src_mesh)
                    loss += loss_face * w['face']
                if w['edge'] > 0.0:
                    loss_edge = mesh_edge_loss(new_src_mesh)
                    loss += loss_edge * w['edge']

                # # mesh normal consistency
                if w['normal'] > 0.0:
                    loss_normal = mesh_normal_consistency(new_src_mesh)
                    loss += loss_normal * w['normal']
                #
                # # mesh laplacian smoothing
                if w['laplacian'] > 0.0:
                    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="cot")
                    loss += loss_laplacian * w['laplacian']

                # loss_symmetry = sample_src - sample_src
                # loss_normal2 = torch.mean(torch.abs(src_mesh.verts_normals_packed() - vertex_normal))

                # Print the losses
                pbar.set_description('total_loss = {0:.6f}'.format(loss))

                # Optimization step
                loss.backward(retain_graph=True)
                optimizer.step()

                # Deform the mesh
                new_src_mesh = src_mesh.offset_verts(deform_verts)

        new_mesh = trimesh.Trimesh(new_src_mesh[0].verts_packed().detach().cpu().numpy(),
                                   new_src_mesh[0].faces_packed().detach().cpu().numpy(),
                                   visual=src_mesh_trimesh.visual, process=False)
        new_mesh = trimesh.smoothing.filter_laplacian(new_mesh, lamb=0.3)

        kdtree = KDTree(trg_mesh_trimesh.vertices, leaf_size=30, metric='euclidean')
        idx = kdtree.query(new_mesh.vertices, k=1, return_distance=False)
        vis_mesh = trimesh.Trimesh(new_mesh.vertices, new_mesh.faces,
                                   vertex_colors=trg_mesh_trimesh.visual.vertex_colors[idx[:, 0], :], process=False)

        return vis_mesh
