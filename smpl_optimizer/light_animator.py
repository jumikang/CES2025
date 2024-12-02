# from __future__ import annotations
import os

import cv2
import torch
import trimesh
import glob
import yaml

from pysdf import SDF
import secrets
import numpy as np
from torch import nn
from sklearn.neighbors import KDTree
from smpl_optimizer.animater_utils import (deform_vertices, deform_to_star_pose)

from torchmcubes import marching_cubes, grid_interp
from utils.depth2volume import volume_filter, postprocess_mesh
from fbx_utils.fbx_wrapper import SMPLX_JOINT_NAMES, set_pose_from_rodrigues

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues


class LightHumanAnimator(nn.Module):
    def __init__(self,
                 smpl_config=None,
                 smpl_root=None,
                 smpl_model=None,
                 rig_method='nearest',
                 device='cuda:0'):
        super(LightHumanAnimator, self).__init__()
        self.replace_hands = True
        if smpl_model is None:
            with open(smpl_config) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                smpl_config = config['DEFAULT']

            self.smpl_conf = smpl_config
            self.model_path = smpl_root
            self.regressor = os.path.join(smpl_root, smpl_config['regressor'])
            self.smpl_model = self.init_smpl_model()
            self.smpl_model = self.smpl_model.to(device)
        else:
            self.smpl_model = smpl_model

        self.rig_method = rig_method
        self.rigged_model = {
            "lbs": None,
            "vts": None,
            "faces": None,
            "texture": None,
            "canonical": None
        }
        self.target_motion = []
        self.motion_list = []
        self.device = device

    def init_smpl_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        import smplx
        return smplx.create(self.model_path,
                            model_type=self.smpl_conf['model_type'],
                            gender=self.smpl_conf['gender'],
                            num_betas=self.smpl_conf['num_betas'],
                            ext=self.smpl_conf['ext'],
                            use_face_contour=self.smpl_conf['use_face_contour'],
                            flat_hand_mean=self.smpl_conf['use_flat_hand'],
                            use_pca=self.smpl_conf['use_pca'],
                            num_pca_comps=self.smpl_conf['num_pca_comp']
                            )

    def _subdivide_smpl_(self, vertices, motion=None):
        if motion is None:
            augmented_data = np.hstack((vertices, self.smpl_model.lbs_weights.detach().cpu().numpy()))
        else:
            augmented_data = np.hstack((vertices,
                                        motion,
                                        self.smpl_model.lbs_weights.detach().cpu().numpy()
                                        ))

        vertices, faces = trimesh.remesh.subdivide(augmented_data, self.smpl_model.faces)

        # vertices, faces = trimesh.remesh.subdivide(vertices, faces)
        smpl_vertices = vertices[:, :3]
        if motion is not None:
            flow = vertices[:, 3:6]
            smpl_lbs = vertices[:, 6:]
            return smpl_vertices, smpl_lbs, faces, flow
        else:
            smpl_lbs = vertices[:, 3:]
            return smpl_vertices, smpl_lbs, faces

    def set_motion_list(self, path2motion="./resource/t2m-gpt-motions"):
        dir_list = glob.glob(os.path.join(path2motion, '*'))
        self.motion_list = [os.path.join(d_name, 't2m-gpt-motion.pkl') for d_name in dir_list]

    def fetch_motion(self, path2motion=None):
        if path2motion is not None and os.path.exists(path2motion):
            motion_data = np.load(path2motion, allow_pickle=True)
        else:
            random_motion = secrets.choice(self.motion_list)
            print('current motion:' + random_motion)
            motion_data = np.load(random_motion, allow_pickle=True)

        motions = []
        for k in range(motion_data['body_pose'].shape[0]):
            cur_dict = dict()
            cur_dict['global_orient'] = torch.Tensor(motion_data['global_orient'][k, 0:3].reshape(1, -1)).float()
            cur_dict['body_pose'] = torch.Tensor(motion_data['body_pose'][k, 0:63].reshape(1, -1)).float()

            cur_dict['body_pose'][0, 33:36] = 0.0  # fix neck (unstable)
            cur_dict['body_pose'][0, 42:45] = 0.0  # fix head (unstable)

            # cannot bend into the body.
            cur_dict['body_pose'][0, 47] *= 0.9
            cur_dict['body_pose'][0, 50] *= 0.9

            cur_dict['cam'] = torch.Tensor(motion_data['cam'][k, :, 0:3].reshape(1, -1)).float()
            cur_dict['cam'] /= 5
            motions.append(cur_dict)
        self.target_motion = motions

    def bake_fbx(self, canon_lbs, canon_mesh, canon_joints, motion_data, filename='out.fbx'):
        ## TODO
        # 1. Align foot on the ground
        # 2. Add texture

        import bpy
        from mathutils import Vector, Euler, Quaternion
        def convert_smplx_to_blender(vertices):
            # Swap Y and Z, and negate the new Y
            converted_vertices = vertices[..., [0, 2, 1]]
            converted_vertices[..., 1] = -converted_vertices[..., 1]
            return converted_vertices

        # save rigged mesh with generated motion
        # only works with open3d renderer (version issue)
        pose_data = motion_data["pose"].reshape([-1, 55, 3])
        trans_data = motion_data["transl"].reshape([-1, 3])
        # motion_data[0, 0, :] = 0.0 # fix pelvis

        num_frames, num_joints, _ = pose_data.shape

        # Remove default cube
        if 'Cube' in bpy.data.objects:
            bpy.data.objects['Cube'].select_set(True)
            bpy.ops.object.delete()

        mesh = bpy.data.meshes.new('ExplicitMesh')
        mesh.from_pydata(convert_smplx_to_blender(canon_mesh.vertices), [], canon_mesh.faces)
        # uv_layer = mesh.uv_layers.new(name='UVMap')
        # uv_layer.data.foreach_set("uv", [uv for pair in canon_mesh.uv for uv in pair])
        color_layer = mesh.vertex_colors.new(name='Col')
        for loop in mesh.loops:
            color_layer.data[loop.index].color = canon_mesh.visual.vertex_colors[loop.vertex_index]
        mesh.update()

        obj = bpy.data.objects.new('MeshObject', mesh)
        bpy.context.collection.objects.link(obj)

        # Create a material that uses vertex colors
        mat = bpy.data.materials.new(name="VertexColorMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        vertex_color_node = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
        vertex_color_node.layer_name = "Col"
        mat.node_tree.links.new(bsdf.inputs['Base Color'], vertex_color_node.outputs['Color'])
        # Assign the material to the object
        if len(obj.data.materials):
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Set the object as active and select it
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create vertex groups for each bone
        for i in range(num_joints):
            obj.vertex_groups.new(name=f'{SMPLX_JOINT_NAMES[i]}')

        # Assign vertices to vertex groups with weights
        for v_idx, weights in enumerate(canon_lbs):
            for b_idx, weight in enumerate(weights):
                obj.vertex_groups[b_idx].add([v_idx], weight.item(), 'REPLACE')

        # Create armature
        armature_data = bpy.data.armatures.new('Armature')
        armature_obj = bpy.data.objects.new('ArmatureObject', armature_data)
        bpy.context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj

        # Add bones to armature in EDIT mode
        bpy.ops.object.mode_set(mode='EDIT')
        canon_joints = canon_joints.detach().cpu().numpy().squeeze(0)
        canon_joints = convert_smplx_to_blender(canon_joints)
        bones = []
        for i in range(num_joints):
            bone = armature_data.edit_bones.new(f'{SMPLX_JOINT_NAMES[i]}')
            bone.head = Vector(canon_joints[i])
            bone.tail = Vector(canon_joints[i] + [0, 0, 1.0])  # Small offset for tail
            if self.smpl_model.parents[i] != -1:
                bone.parent = bones[self.smpl_model.parents[i]]
            bones.append(bone)

        obj.parent = armature_obj
        obj.modifiers.new(type='ARMATURE', name='Armature').object = armature_obj

        # Set pose for each frame in POSE mod
        bpy.ops.object.mode_set(mode='POSE')
        armature_obj.pose.bones['pelvis'].rotation_mode = 'QUATERNION'
        armature_obj.pose.bones['pelvis'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), np.radians(-90))
        armature_obj.pose.bones['pelvis'].keyframe_insert('rotation_quaternion',
                                                           frame=bpy.data.scenes[0].frame_current)
        armature_obj.pose.bones['pelvis'].keyframe_insert(data_path="location", frame=bpy.data.scenes[0].frame_current)

        for frame in range(num_frames):
            bpy.context.scene.frame_set(frame)
            for i in range(num_joints):
                bone = armature_obj.pose.bones[f'{SMPLX_JOINT_NAMES[i]}']
                # Set rotation angles (example: 45 degrees around X-axis)
                if f'{SMPLX_JOINT_NAMES[i]}' == 'pelvis':
                    bone.location = Vector(trans_data[frame])
                    bone.keyframe_insert(data_path="location", frame=frame)

                set_pose_from_rodrigues(armature_obj,
                                        f'{SMPLX_JOINT_NAMES[i]}',
                                        pose_data[frame, i],
                                        frame=frame)

        # add texture map to the mesh
        # bpy.context.view_layer.objects.active = obj
        # bpy.ops.object.mode_set(mode='OBJECT')
        #
        # mat = bpy.data.materials.new(name="TextureMaterial")
        # mat.use_nodes = True
        # bsdf = mat.node_tree.nodes["Principled BSDF"]
        # tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        # tex_image.image = bpy.data.images.load(texture_path)
        # mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
        # obj.data.materials.append(mat)

        # Select the armature and mesh objects
        bpy.context.view_layer.objects.active = armature_obj
        armature_obj.select_set(True)

        # Export to FBX
        bpy.ops.export_scene.fbx(
            filepath=filename,
            use_selection=True,
            bake_anim=True,
            add_leaf_bones=False,
            bake_space_transform=False,
            apply_scale_options='FBX_SCALE_ALL',
            path_mode='COPY',
            embed_textures=False,
            use_mesh_modifiers=True,
            mesh_smooth_type='FACE',
            bake_anim_use_all_bones=True,
            bake_anim_force_startend_keying=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False
        )

        print("FBX export completed.")

    def animate_next(self, canon_lbs,
                     canon_vts=None,
                     canon_mesh=None,
                     return_mesh=False):
        if len(self.target_motion) == 0:
            print('no more motion data')
            return None

        motion = self.target_motion.pop(0)
        for key in motion.keys():
            if torch.is_tensor(motion[key]):
                motion[key] = motion[key].to(self.device)
        smpl_output = self.smpl_model(global_orient=motion['global_orient'],
                                      body_pose=motion['body_pose'],
                                      return_full_pose=True)

        if canon_vts is None:
            canon_vts = torch.FloatTensor(canon_mesh.vertices[None, :, :]).to(self.device)

        custom_vts = deform_to_star_pose(canon_vts,
                                         self.smpl_model,
                                         canon_lbs,
                                         smpl_output.full_pose,
                                         inverse=False,
                                         return_vshape=False,
                                         device=self.device)
        custom_vts = deform_vertices(custom_vts,
                                     self.smpl_model,
                                     canon_lbs,
                                     smpl_output.full_pose,
                                     inverse=False,
                                     return_vshape=False,
                                     device=self.device)

        deformed_vts = custom_vts + motion['cam'].unsqueeze(0) * 0.2  # move around sma

        if return_mesh:
            # mesh should be the input.
            deformed_mesh = trimesh.Trimesh(deformed_vts.squeeze(0).detach().cpu().numpy(),
                                            canon_mesh.faces,
                                            visual=canon_mesh.visual,
                                            process=False)
            return deformed_mesh
        else:
            return deformed_vts

    def rig_linear(self, smpl_mesh, scan_mesh):
        ref_vertices, ref_lbs, ref_faces = \
            self._subdivide_smpl_(smpl_mesh.vertices)

        kdtree = KDTree(ref_vertices, leaf_size=30)
        dist, kd_idx = kdtree.query(scan_mesh.vertices, k=10, return_distance=True)
        weight = 1 - np.clip(dist, a_min=0, a_max=10) / 10
        weight = weight / np.sum(weight, axis=1, keepdims=True)
        custom_lbs = np.zeros((kd_idx.shape[0], 55))
        for k in range(kd_idx.shape[1]):
            custom_lbs += ref_lbs[kd_idx[:, k], :] * weight[:, k:k+1]
        return torch.FloatTensor(custom_lbs)

    def rig_nearest(self, smpl_mesh, scan_mesh):
        ref_vertices, ref_lbs, ref_faces = \
            self._subdivide_smpl_(smpl_mesh.vertices)

        kdtree = KDTree(ref_vertices, leaf_size=30)
        kd_idx = kdtree.query(scan_mesh.vertices, k=1, return_distance=False)
        custom_lbs = ref_lbs[kd_idx.squeeze(), :]
        return torch.FloatTensor(custom_lbs)

    def rig_arap(self, smpl_mesh, scan_mesh, vertex_flow):
        ref_vertices, ref_lbs, ref_faces, ref_flow = \
            self._subdivide_smpl_(smpl_mesh.vertices, motion=vertex_flow)

        # compare at the warped position.
        kdtree = KDTree(ref_vertices + ref_flow, leaf_size=30)
        kd_idx = kdtree.query(scan_mesh.vertices, k=1, return_distance=False)
        custom_lbs = ref_lbs[kd_idx.squeeze(), :]
        motion_compensation = ref_flow[kd_idx.squeeze(), :]
        return torch.FloatTensor(custom_lbs), motion_compensation

    @staticmethod
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

    @staticmethod
    def get_mesh(volume, grid_coord, scale_factor=1.0):
        # mesh generation.
        if isinstance(volume, np.ndarray):
            volume = torch.Tensor(volume)
        vertices, faces = marching_cubes(volume, 0.0)
        new_vertices = torch.Tensor(vertices.detach().cpu().numpy()[:, ::-1].copy())
        new_vertices = grid_interp(grid_coord.contiguous(), new_vertices)
        new_mesh = trimesh.Trimesh(new_vertices / scale_factor, faces)
        return new_mesh

    @staticmethod
    def get_face_length(vts, faces):
        # check faces.
        areas = []
        for k in range(faces.shape[0]):
            x, y, z = faces[k]
            a = sum((vts[x, :] - vts[y, :]) ** 2) ** 2
            b = sum((vts[y, :] - vts[z, :]) ** 2) ** 2
            c = sum((vts[x, :] - vts[z, :]) ** 2) ** 2
            s = a + b + c
            areas.append(s)
        return areas

    def rig(self, smpl_mesh, scan_mesh, smpl_param, colorize_normal=False,
            motion=None, return_mesh=False, v_label=None):
        if motion is not None:
            custom_lbs, motion_compensation = self.rig_arap(smpl_mesh, scan_mesh, motion)
            scan_mesh = trimesh.Trimesh(vertices=scan_mesh.vertices - motion_compensation,
                                        faces=scan_mesh.faces,
                                        visual=scan_mesh.visual, process=False)
        else:
            custom_lbs = self.rig_nearest(smpl_mesh, scan_mesh)

        smpl_param['scale'] = smpl_param['scale'].to(self.device)
        smpl_param['transl'] = smpl_param['transl'].to(self.device)
        smpl_param['full_pose'] = smpl_param['full_pose'].to(self.device)
        normalized_vertices = torch.FloatTensor(scan_mesh.vertices[None, :, :]).to(self.device)
        normalized_vertices = normalized_vertices/smpl_param['scale'] - smpl_param['transl']
        custom_vts, v_shaped, joints = deform_vertices(normalized_vertices,
                                               self.smpl_model,
                                               custom_lbs,
                                               smpl_param['full_pose'],
                                               inverse=True,
                                               return_vshape=True,
                                               device=self.device)

        # custom_vts, v_shaped = deform_to_star_pose(custom_vts,
        #                                            self.smpl_model,
        #                                            custom_lbs,
        #                                            smpl_param['full_pose'],
        #                                            inverse=True,
        #                                            return_vshape=True,
        #                                            device=self.device)
        # replace hands.
        if self.replace_hands:
            v_margin = 10
            scale = 100  # to set signed distance volume in the millimeter space
            smpl_mesh_scaled = trimesh.Trimesh(v_shaped[0].detach().cpu().numpy() * scale,
                                               self.smpl_model.faces, process=False)
            v_min = np.floor(np.min((smpl_mesh_scaled.bounds[0], smpl_mesh_scaled.bounds[0]), axis=0)).astype(int)
            v_max = np.ceil(np.max((smpl_mesh_scaled.bounds[1], smpl_mesh_scaled.bounds[1]), axis=0)).astype(int)
            v_min -= v_margin
            v_max += v_margin
            res = (v_max - v_min) * 2

            scan_mesh_scaled = trimesh.Trimesh(custom_vts[0].detach().cpu().numpy() * scale,
                                               scan_mesh.faces,
                                               visual=scan_mesh.visual,
                                               process=False)
            confidence = self.get_face_length(scan_mesh_scaled.vertices,
                                              scan_mesh_scaled.faces)

            conf_bool = np.asarray([conf < 30.0 for conf in confidence])
            scan_mesh_scaled.update_faces(conf_bool)
            scan_mesh_scaled.remove_degenerate_faces()

            query_pts, grid_coord = self._get_grid_coord_(v_min, v_max, res=res)
            sdf_scan = SDF(scan_mesh_scaled.vertices, scan_mesh.faces)
            sdf_smpl = SDF(smpl_mesh_scaled.vertices, smpl_mesh_scaled.faces)
            volume_smpl = sdf_smpl(query_pts)
            target_volume = sdf_scan(query_pts)

            volume_smpl = volume_smpl.reshape((res[0], res[1], res[2]))
            target_volume = target_volume.reshape((res[0], res[1], res[2]))

            avg_left = np.mean(smpl_mesh_scaled.vertices[v_label['left_wrist_idx'], :], axis=0)
            avg_right = np.mean(smpl_mesh_scaled.vertices[v_label['right_wrist_idx'], :], axis=0)

            delta = np.abs(grid_coord[0, 0, 0, 1] - grid_coord[0, 0, 0, 0])
            left_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_left[0]) <= delta)
            right_idx = np.where(np.abs(grid_coord[0, 0, 0, :] - avg_right[0]) <= delta)

            offset_left = left_idx[0][-1]  # right-most value
            offset_right = right_idx[0][0]  # left-most value
            
            med_idx = target_volume.shape[1] // 2

            target_volume = volume_filter(torch.FloatTensor(target_volume).unsqueeze(0), iter=1)
            target_volume = target_volume.numpy()
            target_volume[offset_left:, med_idx:, :] = volume_smpl[offset_left:, med_idx:, :]
            target_volume[:offset_right, med_idx:, :] = volume_smpl[:offset_right, med_idx:, :]

            # linearly blending two volumes near the wrist
            b_range = 10
            for k in range(b_range):
                alpha = (1 - k / (b_range - 1)) * 0.05
                target_volume[offset_left - k, med_idx:, :] = \
                    target_volume[offset_left - k, med_idx:, :] * (1 - alpha) + \
                    volume_smpl[offset_left - k, med_idx:, :] * alpha
                target_volume[offset_right + k, med_idx:, :] = \
                    target_volume[offset_right + k, med_idx:, :] * (1 - alpha) \
                    + volume_smpl[offset_right + k, med_idx:, :] * alpha

            mesh_merged = self.get_mesh(target_volume, grid_coord, scale_factor=scale)
            mesh_merged = postprocess_mesh(mesh_merged)

            # update lbs
            smpl_mesh = trimesh.Trimesh(v_shaped[0].detach().cpu().numpy(),
                                        self.smpl_model.faces, process=False)
            # custom_lbs = self.rig_nearest(smpl_mesh, mesh_merged)
            custom_lbs = self.rig_linear(smpl_mesh, mesh_merged)

            # coloring in the posed space.
            # posed_vts = deform_to_star_pose(torch.FloatTensor(mesh_merged.vertices[None, :, :]).to(self.device),
            #                                 self.smpl_model,
            #                                 custom_lbs,
            #                                 smpl_param['full_pose'],
            #                                 inverse=False,
            #                                 return_vshape=False,
            #                                 device=self.device)
            posed_vts = deform_vertices(torch.FloatTensor(mesh_merged.vertices[None, :, :]).to(self.device),
                                        self.smpl_model,
                                        custom_lbs,
                                        smpl_param['full_pose'],
                                        inverse=False,
                                        return_vshape=False,
                                        device=self.device)

            posed_scan = trimesh.Trimesh(normalized_vertices.squeeze(0).detach().cpu().numpy(),
                                         scan_mesh.faces,
                                         visual=scan_mesh.visual,
                                         process=False)

            posed_vts = posed_vts.squeeze(0).detach().cpu().numpy()
            kdtree_color = KDTree(posed_scan.vertices, leaf_size=30, metric='euclidean')
            kd_idx = kdtree_color.query(posed_vts, k=1, return_distance=False)

            if colorize_normal:
                tmp_mesh = trimesh.Trimesh(posed_vts, mesh_merged.faces, process=False)
                new_color = tmp_mesh.vertex_normals / 2.0 + 0.5
                new_color = (new_color * 255.0).astype(np.uint8)
            else:
                new_color = posed_scan.visual.vertex_colors[kd_idx.squeeze(), :]
            canon_mesh = trimesh.Trimesh(mesh_merged.vertices, mesh_merged.faces,
                                         vertex_colors=new_color, process=False)
            custom_vts = torch.FloatTensor(mesh_merged.vertices[None, :, :]).cuda()
            return custom_vts, custom_lbs, canon_mesh, smpl_mesh, joints

        # canonical model with corresponding LBS weights
        if return_mesh:
            canon_mesh = trimesh.Trimesh(custom_vts.squeeze(0).detach().cpu().numpy(),
                                         scan_mesh.faces,
                                         visual=scan_mesh.visual,
                                         process=False)
            return custom_vts, custom_lbs, canon_mesh, scan_mesh, joints
        else:
            return custom_vts, custom_lbs

    def animate(self, canon_mesh, canon_lbs, vis=False):
        while self.target_motion:
            self.animate_next(canon_mesh, canon_lbs, return_mesh=True)

    def save2video(self, images, video_path='video.mp4'):
        # video_path = 'video.mp4'
        frame_rate = 15  # Adjust the frame rate as needed

        # Write images to video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate,
                                       (images[0].shape[1], images[0].shape[0]))
        for img in images:
            video_writer.write((img * 255).astype(np.uint8))

        video_writer.release()
    def animate_pytorch3d(self, canon_vts, canon_lbs, canon_mesh, renderer, render_normal=True):
        renderer.set_mesh(canon_mesh, render_normal=render_normal)
        output = []
        dist, azim, elev = 2.5, 0.0, 6.0
        azim_delta = 360.0 / len(self.target_motion)

        count = 0
        while self.target_motion:
            # if count == 30:
            #     renderer.set_mesh(canon_mesh, render_normal=True)
            # elif count == 70:
            #     renderer.set_mesh(canon_mesh, render_normal=False)
            # count += 1
            # renderer.set_mesh(canon_mesh, render_normal=True)
            # deformed_vts = self.animate_next(canon_vts=canon_vts, canon_lbs=canon_lbs)
            # rendered_rgb = renderer.render(deformed_vts, reset_cam=True,
            #                            dist=dist, azim=azim, elev=elev, vis=False)
            # renderer.set_mesh(canon_mesh, render_normal=False)
            deformed_vts = self.animate_next(canon_vts=canon_vts, canon_lbs=canon_lbs)
            rendered = renderer.render(deformed_vts, reset_cam=True,
                                       dist=dist, azim=azim, elev=elev, vis=False)

            azim += azim_delta
            # elev += 0.2
            # rendered = np.concatenate((rendered_rgb, rendered_normal), axis=1)
            output.append(rendered[:, :, ::-1])
            cv2.imshow('results', rendered[:, :, ::-1])
            cv2.waitKey(1)

        return output

    def animate_open3d(self, canon_vts, canon_lbs, canon_mesh, renderer, method='pytorch3d'):
        while self.target_motion:
            deformed_vts = self.animate_next(canon_vts=canon_vts, canon_lbs=canon_lbs)
            renderer.render(deformed_vts[0].detach().cpu().numpy(),
                            canon_mesh.faces,
                            canon_mesh.visual.vertex_colors,
                            method='mesh')

    def save_motion(self, filename=None):
        motion_all = []
        trans_all = []
        for motion in self.target_motion:
            for key in motion.keys():
                if torch.is_tensor(motion[key]):
                    motion[key] = motion[key].to(self.device)
            smpl_output = self.smpl_model(global_orient=motion['global_orient'],
                                          body_pose=motion['body_pose'],
                                          return_full_pose=True)
            motion_all.append(smpl_output.full_pose.detach().cpu().numpy().tolist())
            trans_all.append(motion['cam'].detach().cpu().numpy().tolist())
        if filename is not None:
            import json
            motion = {"pose": np.asarray(motion_all).squeeze(1).tolist(),
                      "transl": np.asarray(trans_all).squeeze(1).tolist()}
            with open(filename, "w") as f:
                json.dump(motion, f, indent=4)
        else:
            motion = {"pose": np.asarray(motion_all).squeeze(1),
                      "transl": np.asarray(trans_all).squeeze(1)}
            return motion

    def forward(self):
        pass
