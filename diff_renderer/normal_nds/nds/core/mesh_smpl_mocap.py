import cv2
import os
import json
import smplx
import torch
import itertools
import trimesh
import numpy as np
import collections
from sklearn.neighbors import KDTree
from PIL import Image
from diff_renderer.normal_nds.nds.core.mesh_textured import TexturedMesh
from human_animator.animater_utils import batch_rigid_transform


class SMPLMesh(TexturedMesh):
    def __init__(self,
                 vertices=None,
                 indices=None,
                 uv_vts=None,
                 v_posed=None,
                 lbs_weights=None,
                 tex=None,
                 disp=None,
                 smpl_config=None,
                 device='cuda:0'):
        super(TexturedMesh).__init__()
        self.device = device
        self.smpl_config = smpl_config

        # variables that are calculated internally
        self.lbs_weights, self.joints, self.A = None, None, None
        self.uv_mapper = None
        self.seam = None
        self.face_normals, self.vertex_normals = None, None
        self._edges = None
        self._connected_faces = None
        self._laplacian = None
        self.shape_uv = None

        # initialize a posed mesh
        self.vertices = self.to_torch(vertices, device)
        self.indices = self.to_torch(indices, device)
        if self.indices is not None:
            self.indices = self.indices.type(torch.int64)
        self.uv_vts = self.to_torch(uv_vts, device)

        if tex is not None:
            self.tex = self.to_torch(tex, device)
        else:
            path2uv_texture = os.path.join(self.smpl_config["smpl_root"],
                                        self.smpl_config["model_type"] + '_uv',
                                        self.smpl_config["model_type"] + '_uv_with_eyes5.png')
            tex_eye = cv2.imread(path2uv_texture, cv2.IMREAD_ANYCOLOR)
            tex_eye = np.flip(tex_eye, axis=0)
            self.tex = torch.FloatTensor(tex_eye.copy()).to(device) / 255.0 * 0.9
            self.tex_eye = self.tex.clone()
            self.tex_eye.requires_grad_(False)
            self.tex_eye_mask = torch.FloatTensor(np.sum(tex_eye, axis=2, keepdims=True) > 0).to(device)
            path2uv_face = os.path.join(self.smpl_config["smpl_root"],
                                        self.smpl_config["model_type"] + '_uv',
                                        self.smpl_config["model_type"] + '_face_mask.png')
            tex_face = cv2.imread(path2uv_face, cv2.IMREAD_ANYCOLOR)
            tex_face = np.flip(tex_face, axis=0)
            tex_face = cv2.cvtColor(tex_face, cv2.COLOR_RGB2GRAY)
            tex_np = np.zeros_like(tex_face)
            tex_np[tex_face > 0.5] = 1.0
            self.tex_face_mask = torch.FloatTensor(tex_np).to(self.device)
            path2uv_hand = path2uv_face.replace('_face_', '_hand_')
            tex_hand = cv2.imread(path2uv_hand, cv2.IMREAD_ANYCOLOR)
            tex_hand = np.flip(tex_hand, axis=0)
            tex_hand = cv2.cvtColor(tex_hand, cv2.COLOR_RGB2GRAY)
            tex_np = np.zeros_like(tex_hand)
            tex_np[tex_hand > 0.5] = 1.0
            self.tex_hand_mask = torch.FloatTensor(tex_np).to(device)

        if v_posed is not None:
            self.v_posed = self.to_torch(v_posed, device)

        if lbs_weights is not None:
            self.lbs_weights = self.to_torch(lbs_weights, device)

        # displacement map (for optimization purpose, used internally)
        if disp is not None:
            self.disp = self.to_torch(disp, device)
        else:
            self.disp = torch.zeros((1024, 1024, 3),
                                    dtype=torch.float32,
                                    requires_grad=True).to(device)

        if self.indices is not None:
            self.compute_normals()

        if self.smpl_config is not None:
            self.smpl_model = self.init_smpl_model()

        self.idx2body = None
        self.target_parts = {"eyeballs": 2e-4, "leftHand": 5e-2, "rightHand": 5e-2,
                             "leftForeArm": 5e-2, "rightForeArm": 5e-2} # "leftArm": 5e-2, "rightArm": 5e-2,
        self.semantic = {}
        path2semantic = os.path.join(self.smpl_config['smpl_root'], self.smpl_config['segmentation'])
        with open(path2semantic) as f:
            self.v_label = json.load(f)  # for original smplx
            for key in self.v_label.keys():
                self.v_label[key] = np.asarray(self.v_label[key])

    def get_canonical_mesh(self, disp_vectors):
        vertices = self.v_posed + disp_vectors
        vertices = vertices.cpu().detach().numpy()
        uv = self.uv_vts.cpu().detach().numpy()
        faces = self.indices.cpu().detach().numpy().astype(np.int32)

        texture = self.tex.cpu().detach().numpy()
        texture = np.uint8(texture * 255)
        texture_image = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        texture_image = np.rot90(texture_image, k=1)
        texture_image = np.flip(texture_image, axis=1)
        texture_image = np.rot90(texture_image, k=3)
        tex_pil = Image.fromarray(texture_image)

        visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_pil)
        mesh = trimesh.Trimesh(vertices, faces, visual=visual, process=False)
        return mesh

    def init_smpl_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        if self.smpl_config['model_type'] == 'smpl':
            return smplx.create(self.smpl_config['smpl_root'],
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                ext=self.smpl_config['ext']
                                ).to(self.device)
        elif self.smpl_config['model_type'] == 'smplh':
            return smplx.create(self.smpl_config['smpl_root'],
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                num_betas=self.smpl_config['num_betas'],
                                ext=self.smpl_config['ext'],
                                flat_hand_mean=self.smpl_config['use_flat_hand'],
                                use_pca=self.smpl_config['use_pca'],
                                num_pca_comps=self.smpl_config['num_pca_comp']
                                ).to(self.device)
        elif self.smpl_config['model_type'] == 'smplx':
            return smplx.create(self.smpl_config['smpl_root'],
                                model_type=self.smpl_config['model_type'],
                                gender=self.smpl_config['gender'],
                                num_betas=self.smpl_config['num_betas'],
                                ext=self.smpl_config['ext'],
                                use_face_contour=self.smpl_config['use_face_contour'],
                                flat_hand_mean=self.smpl_config['use_flat_hand'],
                                use_pca=self.smpl_config['use_pca'],
                                num_pca_comps=self.smpl_config['num_pca_comp']
                                ).to(self.device)
        else:
            assert f"{self.smpl_config['model_type']} is not supported model type."

    def forward_smpl(self, smpl_params, return_mesh=False):
        if self.smpl_config['model_type'] == 'smpl':
            if smpl_params['body_pose'].shape[1] == 63:
                smpl_params['body_pose'] = (
                    torch.concatenate((smpl_params['body_pose'],
                                       torch.zeros_like(smpl_params['body_pose'][:, :6])), dim=1))

            smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                          body_pose=smpl_params['body_pose'],
                                          betas=smpl_params['betas'],
                                          global_orient=smpl_params['global_orient'],
                                          return_full_pose=True,
                                          return_verts=True)
        elif self.smpl_config['model_type'] == 'smplx':
            smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                          expression=smpl_params['expression'],
                                          body_pose=smpl_params['body_pose'],
                                          betas=smpl_params['betas'],
                                          global_orient=smpl_params['global_orient'],
                                          jaw_pose=smpl_params['jaw_pose'],
                                          left_hand_pose=smpl_params['left_hand_pose'],
                                          right_hand_pose=smpl_params['right_hand_pose'],
                                          return_full_pose=True,
                                          return_verts=True)
        else:
            assert f"{self.smpl_config['model_type']} is not supported model type."

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            smpl_output.joints = smpl_output.joints * smpl_params['scale']
            smpl_output.vertices = smpl_output.vertices * smpl_params['scale']

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy() * 100,
                                        faces=self.smpl_model.faces, process=False)
            return smpl_output, smpl_mesh
        else:
            return smpl_output

    def detach(self):
        mesh = SMPLMesh(vertices=self.vertices.detach(),
                        indices=self.indices.detach(),
                        uv_vts=self.uv_vts.detach() if self.uv_vts is not None else None,
                        v_posed=self.v_posed.detach() if self.v_posed is not None else None,
                        lbs_weights=self.lbs_weights.detach() if self.lbs_weights is not None else None,
                        tex=self.tex.detach() if self.tex is not None else None,
                        disp=self.disp.detach() if self.disp is not None else None,
                        smpl_config=self.smpl_config,
                        device=self.device)

        mesh.A = self.A.detach() if self.A is not None else None
        mesh.joints = self.joints.detach() if self.joints is not None else None

        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        mesh.seam = self.seam
        mesh.tex_eye = self.tex_eye
        mesh.tex_eye_mask = self.tex_eye_mask
        mesh.tex_hand_mask = self.tex_hand_mask
        mesh.tex_face_mask = self.tex_face_mask
        mesh.shape_uv = self.shape_uv
        mesh.idx2body = self.idx2body
        return mesh

    def update_seam(self, vertices):
        if torch.is_tensor(vertices):
            new_vts = (vertices.detach().cpu().numpy() * 1000000 // 1).astype(np.int32).tolist()
        else:
            new_vts = (vertices.copy() * 1000000 // 1).astype(np.int32).tolist()

        uv_vts = [f"{uv[0]:09}{uv[1]:09}{uv[2]:09}" for uv in new_vts]
        tmp_dict = collections.defaultdict(list)
        for i, uv in enumerate(uv_vts):
            tmp_dict[uv].append(i)

        seam = []
        for key in tmp_dict.keys():
            if len(tmp_dict[key]) == 2:
                seam.append(tmp_dict[key])
            elif len(tmp_dict[key]) == 3:
                seam.append(tmp_dict[key][0:2])
                seam.append(tmp_dict[key][1:3])
            elif len(tmp_dict[key]) == 4:
                seam.append(tmp_dict[key][0:2])
                seam.append(tmp_dict[key][1:3])
                seam.append(tmp_dict[key][2:4])
        self.seam = np.asarray(seam)

    def init_semantic_labels(self, vertices):
        def index_mapper(key):
            index = []
            for i in self.v_label[key]:
                index.append(self.uv_mapper['smplx_uv'].index(i))
            return index

        self.semantic = {}
        for key in self.target_parts.keys():
            kd_tree = KDTree(vertices[index_mapper(key), :], leaf_size=30, metric='euclidean')
            self.semantic[key] = kd_tree
        self.set_semantic_labels(vertices)

    def set_semantic_labels(self, vertices):
        def get_index(key, threshold=1e-3):
            dist, _ = self.semantic[key].query(vertices, k=1, return_distance=True)
            tmp_list = (dist.squeeze(1) < threshold).tolist()
            index = []
            for i, v in enumerate(tmp_list):
                if v is True:
                    index.append(i)
            return index

        self.idx2body = {}
        for key in self.target_parts.keys():
            self.idx2body[key] = get_index(key, threshold=self.target_parts[key])

        if 'leftHand' in self.idx2body.keys() and 'rightHand' in self.idx2body.keys():
            no_hands_index = []
            for k in range(vertices.shape[0]):
                if k not in self.idx2body['leftHand'] and k not in self.idx2body['rightHand']:
                    no_hands_index.append(k)
            self.idx2body['noHands'] = no_hands_index

    def set_canonical_smpl_in_uv(self, smpl_params):
        # load and set SMPL mesh with uv coordinates.
        path2uv_table = os.path.join(self.smpl_config["smpl_root"], 'smpl_uv_table.json')
        if os.path.exists(path2uv_table):
            with open(path2uv_table, 'r') as f:
                self.uv_mapper = json.load(f)

        path2uv_mesh = os.path.join(self.smpl_config["smpl_root"],
                                    self.smpl_config["model_type"] + '_uv', self.smpl_config["model_type"] + '_uv.obj')
        if os.path.exists(path2uv_mesh) and self.uv_vts is None:
            smpl_uv_mesh = trimesh.load_mesh(path2uv_mesh)
            uv_vts = np.round(smpl_uv_mesh.visual.uv, 7)
            self.uv_vts = self.to_torch(uv_vts, self.device)
            self.indices = self.to_torch(smpl_uv_mesh.faces, self.device).type(torch.int64)
            self.shape_uv = np.asarray(smpl_uv_mesh.vertices)
            self.update_seam(smpl_uv_mesh.vertices)
            self.init_semantic_labels(smpl_uv_mesh.vertices)
        else:
            assert "smpl_uv.obj file is necessary"
        for key in smpl_params.keys():
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].to(self.device)

        smpl_output = self.forward_smpl(smpl_params, return_mesh=False)
        v_shaped = self.smpl_model.v_template + \
                   smplx.lbs.blend_shapes(smpl_output.betas, self.smpl_model.shapedirs)
        self.joints = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, v_shaped)
        self.lbs_weights = self.smpl_model.lbs_weights.to(self.device).detach()

        # do not use smpl_model.joints -> it fails (don't know why)
        batch_size = 1
        rot_mats = smplx.lbs.batch_rodrigues(smpl_output.full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        # ident = torch.eye(3, dtype=torch.float32, device=self.device)
        # pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # # (N x P) x (P, V * 3) -> N x V x 3
        # pose_offsets = torch.matmul(
        #     pose_feature, self.smpl_model.posedirs).view(batch_size, -1, 3)

        if self.smpl_config["model_type"] == 'smplx':
            joints = self.joints[:, :55, :]
        else:
            joints = self.joints
        _, A = batch_rigid_transform(rot_mats,
                                     joints,
                                     self.smpl_model.parents,
                                     inverse=False,
                                     dtype=torch.float32)
        # pose offset is not necessary for real-world reconstruction.
        # v_posed = pose_offsets + v_shaped
        v_posed = v_shaped

        self.lbs_weights = self.lbs_weights[self.uv_mapper[self.smpl_config['model_type']+'_uv'], :]
        self.v_posed = v_posed.squeeze(0)[self.uv_mapper[self.smpl_config['model_type']+'_uv'], :]
        self.A = A

    # you can apply this only once.
    def subdivide(self):
        self.detach()
        self.disp_vector = self.uv_sampling(mode='disp')

        vertices, faces = trimesh.remesh.subdivide(
            vertices=np.hstack((self.vertices.detach().cpu().numpy(),
                                self.v_posed.detach().cpu().numpy(),
                                self.uv_vts.detach().cpu().numpy(),
                                self.disp_vector.detach().cpu().numpy(),
                                self.shape_uv,
                                self.lbs_weights.detach().cpu().numpy())),
            faces=self.indices.detach().cpu().numpy())

        # set subdivided smplx mesh with uv coordinates.
        self.vertices = self.to_torch(vertices[:, 0:3], self.device)
        self.v_posed = self.to_torch(vertices[:, 3:6], self.device)
        self.uv_vts = self.to_torch(vertices[:, 6:8], self.device)
        self.disp_vector = self.to_torch(vertices[:, 8:11], self.device)
        self.shape_uv = vertices[:, 11:14]  # unsampled shape uv
        self.lbs_weights = self.to_torch(vertices[:, 14:], self.device)
        self.indices = self.to_torch(faces, self.device).type(torch.int64)
        self.update_seam(self.shape_uv)
        self.set_semantic_labels(self.shape_uv)
        self.compute_normals()

    def forward_skinning(self, smpl_params=None, v_posed=None, update_smpl=False):
        if v_posed is None:
            v_posed = self.v_posed[None, :, :]

        if update_smpl:
            smpl_output = self.forward_smpl(smpl_params)
            rot_mats = smplx.lbs.batch_rodrigues(smpl_output.full_pose.view(-1, 3)).view([1, -1, 3, 3])
            _, self.A = batch_rigid_transform(rot_mats,
                                              self.joints[:, :55, :],
                                              self.smpl_model.parents,
                                              inverse=False,
                                              dtype=torch.float32)

        weights = self.lbs_weights.expand([1, -1, -1])
        num_joints = self.smpl_model.J_regressor.shape[0]
        T = torch.matmul(weights, self.A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)
        homogen_coord = torch.ones([1, v_posed.shape[1], 1], dtype=torch.float32).to(self.device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        # update vertices
        v_deformed = v_homo[:, :, :3, 0]
        if smpl_params is not None:
            v_deformed = (v_deformed + smpl_params['transl']) * smpl_params['scale']
        return v_deformed[0, :, :] * 100.0

    def fetch_from_eazymoap(self):
        pass

    def forward_skinning_eazymocap(self):
        pass
