import numpy as np
import smplx
import torch
import math
import torch.nn.functional as F


def deform_vertices(vertices, smpl_model, lbs, full_pose, inverse=False, return_vshape=False, device='cuda:0'):
    v_shaped = smpl_model.v_template + \
               smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    # do not use smpl_model.joints -> it fails (don't know why)
    joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats, joints[:, :55, :], smpl_model.parents,
                                             inverse=inverse, dtype=torch.float32)

    weights = lbs.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(weights, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)
    homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32).to(device)
    v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    if return_vshape:
        return verts, v_shaped
    else:
        return verts


# Code
def deform_to_star_pose(vertices, smpl_model, lbs, full_pose,
                        inverse=True, return_vshape=True, device='cuda:0'):
    full_pose = full_pose.clone().detach() * 0
    full_pose[0, 5] = -math.radians(20)
    full_pose[0, 8] = math.radians(20)

    v_shaped = smpl_model.v_template + \
               smplx.lbs.blend_shapes(smpl_model.betas, smpl_model.shapedirs)
    # do not use smpl_model.joints -> it fails (don't know why)
    joints = smplx.lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view([1, -1, 3, 3])
    joints_warped, A = batch_rigid_transform(rot_mats, joints[:, :55, :], smpl_model.parents,
                                             inverse=inverse, dtype=torch.float32)

    weights = lbs.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
    num_joints = smpl_model.J_regressor.shape[0]
    T = torch.matmul(weights, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)

    verts = None
    if vertices is not None:
        homogen_coord = torch.ones([1, vertices.shape[1], 1], dtype=torch.float32).to(device)
        v_posed_homo = torch.cat([vertices, homogen_coord], dim=2)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]

    if return_vshape:
        weights_v = smpl_model.lbs_weights.unsqueeze(dim=0).expand([1, -1, -1]).to(device)
        T_v = torch.matmul(weights_v, A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)

        homogen_coord_v = torch.ones([1, v_shaped.shape[1], 1], dtype=torch.float32).to(device)
        v_posed_homo_v = torch.cat([v_shaped, homogen_coord_v], dim=2)

        v_homo_v = torch.matmul(T_v, torch.unsqueeze(v_posed_homo_v, dim=-1))
        v_shaped = v_homo_v[:, :, :3, 0]

        return verts, v_shaped
    return verts


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    transform_chain_inverse = [torch.zeros_like(transform_chain[0])]*55
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    if inverse is True:
        posed_joints = torch.unsqueeze(posed_joints, dim=-1)
        rel_joints = posed_joints.clone()
        rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
        # rot_inv = torch.transpose(rot_mats.view(-1, 3, 3), dim0=1, dim1=2)
        transforms_mat_inv = transform_mat(
            rot_mats.view(-1, 3, 3),
            torch.zeros_like(rel_joints.view(-1, 3, 1))).view(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat_inv[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat_inv[:, i])
            transform_chain.append(curr_res)

        for i in range(len(transform_chain)):
            transform_chain[i] = torch.inverse(transform_chain[i])
            transform_chain[i][:, :3, 3] = joints[:, i, :, :].view(-1, 3)

        transforms = torch.stack(transform_chain, dim=1)
        joints_homogen = F.pad(posed_joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
        return posed_joints, rel_transforms
    else:
        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms

def batch_rigid_transform2(rot_mats, joints, parents,
                          inverse=True,
                          dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    if inverse is True:
        posed_joints = torch.unsqueeze(posed_joints, dim=-1)
        rel_joints = posed_joints.clone()
        rel_joints[:, 1:] -= posed_joints[:, parents[1:]]
        # rot_inv = torch.transpose(rot_mats.view(-1, 3, 3), dim0=1, dim1=2)
        transforms_mat_inv = transform_mat(
            rot_mats.view(-1, 3, 3),
            torch.zeros_like(rel_joints.view(-1, 3, 1))).view(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat_inv[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat_inv[:, i])
            transform_chain.append(curr_res)

        for i in range(len(transform_chain)):
            # transform_chain[i] = torch.inverse(transform_chain[i])
            transform_chain[i][:, 3, :3] = joints[:, i, :, :].view(-1, 3)

        transforms = torch.stack(transform_chain, dim=1)
        transforms_t = torch.transpose(transforms, 3, 2)
        joints_homogen = F.pad(posed_joints, [0, 0, 0, 1])

        rel_transforms = transforms_t - F.pad(
            torch.matmul(transforms_t, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
        return posed_joints, rel_transforms
    else:
        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms