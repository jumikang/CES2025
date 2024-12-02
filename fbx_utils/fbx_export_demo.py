# def bake_fbx(self, canon_lbs, canon_mesh, filename='out.fbx'):
    #     import bpy
    #     vts_seq = []
    #     for _ in range(20):
    #         deformed_mesh = self.animate_next(canon_mesh=canon_mesh, canon_lbs=canon_lbs, return_mesh=True)
    #         vts_seq.append(deformed_mesh.vertices)
    #     faces = canon_mesh.faces
    #     num_frames = len(vts_seq)
    #
    #     mesh = bpy.data.meshes.new('ExplicitMesh')
    #     mesh.from_pydata(vts_seq[0], [], faces)
    #     mesh.update()
    #
    #     obj = bpy.data.objects.new('MeshObject', mesh)
    #     bpy.context.collection.objects.link(obj)
    #
    #     # Create Shape Keys
    #     obj.shape_key_add(name='Basis', from_mix=False)
    #
    #     for frame, verts in enumerate(vts_seq):
    #         shape_key = obj.shape_key_add(name=f'Key_{frame:03d}', from_mix=False)
    #         for i, vert in enumerate(verts):
    #             shape_key.data[i].co = vert
    #
    #     # Set animation frames
    #     bpy.context.scene.frame_start = 0
    #     bpy.context.scene.frame_end = num_frames - 1
    #
    #     for frame in range(num_frames):
    #         bpy.context.scene.frame_set(frame)
    #         for key_block in obj.data.shape_keys.key_blocks:
    #             if key_block.name == f'Key_{frame:03d}':
    #                 key_block.value = 1.0
    #                 key_block.keyframe_insert(data_path="value", frame=frame)
    #             else:
    #                 key_block.value = 0.0
    #                 key_block.keyframe_insert(data_path="value", frame=frame)
    #
    #     # Select object
    #     bpy.context.view_layer.objects.active = obj
    #     obj.select_set(True)
    #
    #     # Export to FBX
    #     bpy.ops.export_scene.fbx(
    #         filepath=filename,
    #         use_selection=True,
    #         bake_anim=True,
    #         add_leaf_bones=False,
    #         bake_space_transform=False,
    #         apply_scale_options='FBX_SCALE_ALL',
    #         path_mode='COPY',
    #         embed_textures=False,
    #         use_mesh_modifiers=True,
    #         mesh_smooth_type='FACE',
    #         bake_anim_use_all_bones=True,
    #         bake_anim_force_startend_keying=True,
    #         bake_anim_use_nla_strips=False,
    #         bake_anim_use_all_actions=False
    #     )
    #
    #     print("FBX export completed.")