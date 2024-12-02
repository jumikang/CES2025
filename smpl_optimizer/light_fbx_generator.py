import bpy
import numpy as np
import smplx
import torch
import pickle

# https://docs.blender.org/api/current/bpy.ops.export_scene.html

# 모든 오브젝트 삭제
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# SMPL 모델 로드
model_path = 'V:/sanghun/workspace/2K2K_main/_render_sanghun/smpl_related/models'  # SMPL 모델 파일 경로
model = smplx.create(model_path, model_type='smpl', gender='neutral', ext='pkl')

pkl_file_path = 'Y:/3D_RECON/SMPL/t2m-gpt-motions/t2m-gpt-motions/0026/t2m-gpt-motion.pkl'
save_path = 'Y:/3D_RECON/SMPL/t2m-gpt-motions/0026.fbx'
with open(pkl_file_path, 'rb') as pkl_file:
    data = pickle.load(pkl_file)

# 시퀀스 설정
num_frames = data['body_pose'].shape[0]
sequence = []

# 모든 프레임에 동일한 betas 사용
betas = torch.tensor(data['betas'][0:1], dtype=torch.float32)

for i in range(num_frames):
    body_pose = torch.tensor(data['body_pose'][i:i + 1], dtype=torch.float32)
    global_orient = torch.tensor(data['global_orient'][i:i + 1], dtype=torch.float32)

    # SMPL 모델 출력 생성
    output = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
    verts = output.vertices.detach().cpu().numpy().squeeze()

    sequence.append(verts)

# SMPL 모델의 faces를 Python 리스트로 변환
faces = model.faces.tolist()

# Blender에 메쉬 생성
mesh = bpy.data.meshes.new('SMPL_Mesh')
mesh.from_pydata(sequence[0], [], faces)
mesh.update()

obj = bpy.data.objects.new('SMPL_Object', mesh)
bpy.context.collection.objects.link(obj)

# Shape Key 생성
obj.shape_key_add(name='Basis', from_mix=False)

for frame, verts in enumerate(sequence):
    shape_key = obj.shape_key_add(name=f'Key_{frame:03d}', from_mix=False)
    for i, vert in enumerate(verts):
        shape_key.data[i].co = vert

# 애니메이션 설정
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = num_frames - 1

for frame in range(num_frames):
    bpy.context.scene.frame_set(frame)
    for key_block in obj.data.shape_keys.key_blocks:
        if key_block.name == f'Key_{frame:03d}':
            key_block.value = 1.0
            key_block.keyframe_insert(data_path="value", frame=frame)
        else:
            key_block.value = 0.0
            key_block.keyframe_insert(data_path="value", frame=frame)

# 오브젝트 선택
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# FBX로 내보내기
bpy.ops.export_scene.fbx(
    filepath=save_path,
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

# FBX 파일 확인
print("FBX export completed.")
