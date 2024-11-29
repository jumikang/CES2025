import bpy
import numpy as np
from mathutils import Vector, Quaternion
from aiohttp.web_routedef import static
from sympy.physics.units import energy


SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle',
    'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye', 'right_eye',
    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1',
    'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3',
    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1',
    'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2',
    'right_thumb3'
]

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None, frame=1):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()
    pbone = armature.pose.bones[bone_name]
    pbone.rotation_mode = 'QUATERNION'
    quat = Quaternion(axis, angle_rad)
    if rodrigues_reference is None:
        pbone.rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        pbone.rotation_quaternion = quat_result
    pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if bone_name == 'pelvis':
        pbone.keyframe_insert('location', frame=frame)

def convert_smplx_to_blender(vertices):
    converted_vertices = vertices[..., [0, 2, 1]]
    converted_vertices[..., 1] = -converted_vertices[..., 1]
    return converted_vertices

class FBXExporter:
    def __init__(self, smpl_model):
        """
        Initialize the FBX exporter.
        - smpl_model: an instance of SMPL-X model. It is used to get parent indices.
        """
        self.smpl_model = smpl_model

    def bake_fbx(self, canon_lbs, canon_mesh, canon_joints, motion_data, filename='out.fbx'):
        """
        Bake a motion sequence to an FBX file.
        Four inputs must be given.
        - canon_lbs: LBS weights for each vertex inherent from SMPL-X model.
        - canon_mesh: canonical mesh with vertex colors or texture map.
        - canon_joints: the positions of joints in the resting pose.
        - motion_data: a dictionary containing 'pose' and 'transl'.
        - filename: the name of the output FBX file.
        """
        pose_data = motion_data["pose"].reshape([-1, 55, 3])
        trans_data = motion_data["transl"].reshape([-1, 3])
        num_frames, num_joints, _ = pose_data.shape

        # blender initially has a cube object
        self._check_default_cube_()

        # set geometry, color, and material
        mesh, obj = self._create_mesh_object_(canon_mesh)
        self._create_vertex_colors_(mesh, canon_mesh)
        self._create_material_(obj)

        # set lights
        # self.add_point_light(location=(0, 0, 10), energy=1000)
        # self.add_point_light(location=(0, 10, 10), energy=500)
        # self.add_sun_light(location=(0, 0, 10), energy=5)

        # set animation
        self._create_vertex_groups_(obj, canon_lbs, num_joints)
        armature_obj = self._create_armature_(obj, canon_joints, num_joints)
        self._set_pose_(armature_obj, pose_data, trans_data, num_frames, num_joints)

        # export fbx file
        self._export_(filename, armature_obj)

    @staticmethod
    def add_sun_light(location=(0, 0, 10), energy=500):
        bpy.ops.object.light_add(type='SUN', location=location)
        light = bpy.context.object
        light.data.energy = energy

    @staticmethod
    def add_point_light(location=(0, 0, 10), energy=1000):
        bpy.ops.object.light_add(type='POINT', location=location)
        light = bpy.context.object
        light.data.energy = energy

    @staticmethod
    def _check_default_cube_():
        if 'Cube' in bpy.data.objects:
            bpy.data.objects['Cube'].select_set(True)
            bpy.ops.object.delete()

    @staticmethod
    def _create_mesh_object_(canon_mesh):
        mesh = bpy.data.meshes.new('ExplicitMesh')
        mesh.from_pydata(convert_smplx_to_blender(canon_mesh.vertices), canon_mesh.edges, canon_mesh.faces)
        mesh.update()

        obj = bpy.data.objects.new('MeshObject', mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return mesh, obj

    @staticmethod
    def _create_vertex_colors_(mesh, canon_mesh):
        color_layer = mesh.vertex_colors.new(name='Col')
        for loop in mesh.loops:
            color_layer.data[loop.index].color = canon_mesh.visual.vertex_colors[loop.vertex_index] / 255.0

    @staticmethod
    def _create_material_(obj):
        # mat = bpy.data.materials.new(name="VertexColorMaterial")
        mat = bpy.data.materials.new(name="CottonMaterial")

        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Roughness'].default_value = 0.8  # Increase roughness for a matte look
        bsdf.inputs['Specular IOR Level'].default_value = 0.1
        vertex_color_node = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
        vertex_color_node.layer_name = "Col"
        mat.node_tree.links.new(bsdf.inputs['Base Color'], vertex_color_node.outputs['Color'])
        if len(obj.data.materials):
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    @staticmethod
    def _create_vertex_groups_(obj, canon_lbs, num_joints):
        for i in range(num_joints):
            obj.vertex_groups.new(name=f'{SMPLX_JOINT_NAMES[i]}')
        for v_idx, weights in enumerate(canon_lbs):
            for b_idx, weight in enumerate(weights):
                obj.vertex_groups[b_idx].add([v_idx], weight.item(), 'REPLACE')

    def _create_armature_(self, obj, canon_joints, num_joints):
        armature_data = bpy.data.armatures.new('Armature')
        armature_obj = bpy.data.objects.new('ArmatureObject', armature_data)
        bpy.context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        canon_joints = canon_joints.detach().cpu().numpy().squeeze(0)
        canon_joints = convert_smplx_to_blender(canon_joints)
        bones = []
        for i in range(num_joints):
            bone = armature_data.edit_bones.new(f'{SMPLX_JOINT_NAMES[i]}')
            bone.head = Vector(canon_joints[i])
            bone.tail = Vector(canon_joints[i] + [0, 0, 1.0])
            if self.smpl_model.parents[i] != -1:
                bone.parent = bones[self.smpl_model.parents[i]]
            bones.append(bone)

        obj.parent = armature_obj
        obj.modifiers.new(type='ARMATURE', name='Armature').object = armature_obj
        return armature_obj

    @staticmethod
    def _set_pose_(armature_obj, pose_data, trans_data, num_frames, num_joints):
        bpy.ops.object.mode_set(mode='POSE')
        armature_obj.pose.bones['pelvis'].rotation_mode = 'QUATERNION'
        armature_obj.pose.bones['pelvis'].rotation_quaternion = Quaternion((1.0, 0.0, 0.0), np.radians(-90))
        armature_obj.pose.bones['pelvis'].keyframe_insert('rotation_quaternion', frame=bpy.data.scenes[0].frame_current)
        armature_obj.pose.bones['pelvis'].keyframe_insert(data_path="location", frame=bpy.data.scenes[0].frame_current)
        for frame in range(num_frames):
            bpy.context.scene.frame_set(frame)
            for i in range(num_joints):
                bone = armature_obj.pose.bones[f'{SMPLX_JOINT_NAMES[i]}']
                if f'{SMPLX_JOINT_NAMES[i]}' == 'pelvis':
                    bone.location = Vector(trans_data[frame])
                    bone.keyframe_insert(data_path="location", frame=frame)
                set_pose_from_rodrigues(armature_obj, f'{SMPLX_JOINT_NAMES[i]}', pose_data[frame, i], frame=frame)

    @staticmethod
    def _export_(filename, armature_obj):
        bpy.context.view_layer.objects.active = armature_obj
        armature_obj.select_set(True)
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