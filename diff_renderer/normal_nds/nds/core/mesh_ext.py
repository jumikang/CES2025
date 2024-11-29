import cv2
import torch
import trimesh
import numpy as np
import collections
from PIL import Image
from ..utils.geometry import find_connected_faces

def to_torch(data, device, dtype=torch.float32):
    if torch.is_tensor(data):
        torch_tensor = data.to(device, dtype=dtype)
    else:
        torch_tensor = torch.tensor(data.copy(), dtype=dtype, device=device)
    return torch_tensor

class TexturedMesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        uv_vts (tensor) : UV coordinate for vertices (Vx2)
        uv_faces (tensor) : indices of uv_vts w.r.t. faces (Fx3)
        tex (tensor) : texture map (HxWx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, uv_vts, seam,
                 tex=None, disp=None, normal=None, device='cuda:0'):
        self.device = device
        self.vertices = to_torch(vertices, device)
        self.indices = to_torch(indices, device).type(torch.int64)
        self.uv_vts = to_torch(uv_vts, device)
        self.seam = seam

        if tex is not None:
            self.tex = to_torch(tex, device) / 255.0
        else:
            # initialize a texture map
            self.tex = torch.ones((1024, 1024, 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device) * 0.2

        # displacement map (for optimization purpose)
        if disp is None:
            self.disp = torch.zeros((1024, 1024, 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device)
        else:
            self.disp = to_torch(disp, device)

        # normal map (for optimization purpose)
        if normal is None:
            self.normal = torch.zeros((1024, 1024, 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device)
        else:
            self.normal = to_torch(normal, device)

        # additional information that are calculated internally
        self.face_normals = None
        self.vertex_normals = None
        self._edges = None
        self._connected_faces = None
        self._laplacian = None

        if self.indices is not None:
            self.compute_normals()

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

    def upsample(self):
        # e0, e1 = self.edges.unbind(1)
        # average_edge_length = torch.linalg.norm(self.vertices[e0] - self.vertices[e1], dim=-1).mean()
        # cutoff = average_edge_length.detach().cpu().numpy() * 2.0
        #
        mesh = self.to_trimesh(with_texture=False)
        # mesh = mesh.subdivide_to_size(cutoff, max_iter=10)
        mesh = mesh.subdivide()

        self.vertices = torch.FloatTensor(mesh.vertices).to(self.device)
        self.indices = torch.Tensor(mesh.faces).type(torch.int64).to(self.device)
        self.uv_vts = torch.FloatTensor(mesh.visual.uv).to(self.device)
        self.compute_normals()

    def uv_sampling(self, mode='color'):
        '''
        :param displacement: [B, C, H, W] image features
        :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
        :return: [B, C, N] image features at the uv coordinates
        '''
        # uv is in [0, 1] so we need to convert to be in [-1, 1]
        uv = self.uv_vts.unsqueeze(0).unsqueeze(2) * 2 - 1
        if mode == 'color':
            tex = self.tex.permute(2, 0, 1).unsqueeze(0)
        elif mode == 'disp':
            tex = self.disp.permute(2, 0, 1).unsqueeze(0)
        elif mode == 'normal':
            tex = self.normal.permute(2, 0, 1).unsqueeze(0)
        samples = torch.nn.functional.grid_sample(tex, uv, align_corners=True)  # [B, C, N, 1]
        return samples.squeeze().transpose(1, 0).contiguous()

    def to_trimesh(self, with_texture=False):
        vertices = self.vertices.cpu().detach().numpy()
        uv = self.uv_vts.cpu().detach().numpy()
        faces = self.indices.cpu().detach().numpy().astype(np.int32)
        tex_pil = None
        if with_texture:
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

    def to(self, device):
        mesh = TexturedMesh(self.vertices.to(device),
                            self.indices.to(device),
                            self.uv_vts.to(device),
                            self.tex.to(device),
                            device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        return mesh

    def detach(self):
        mesh = TexturedMesh(self.vertices.detach(),
                            self.indices.detach(),
                            self.uv_vts.detach(),
                            self.tex.detach(),
                            device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)

        mesh_new = TexturedMesh(self.vertices,
                                self.indices,
                                self.uv_vts,
                                self.tex,
                                device=self.device)

        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    def with_colors(self, colors):
        """ Create a mesh with the same connectivity but with different texture map

        Args:
            colors (tensor): New color values (HxWx3)
        """

        # assert len(colors) == len(self.colors)
        mesh_new = TexturedMesh(self.vertices,
                                self.indices,
                                self.uv_vts,
                                colors,
                                device=self.device)

        # mesh_new = TexturedMesh(self.vertices, self.indices, colors, self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    @property
    def edges(self):
        # if self._edges is None:
        from diff_renderer.normal_nds.nds.utils.geometry import find_edges
        self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        # if self._connected_faces is None:
        from diff_renderer.normal_nds.nds.utils.geometry import find_connected_faces
        self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        # if self._laplacian is None:
        from diff_renderer.normal_nds.nds.utils.geometry import compute_laplacian_uniform
        self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    def compute_connectivity(self):
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices.int()][:, 0, :]
        b = self.vertices[self.indices.int()][:, 1, :]
        c = self.vertices[self.indices.int()][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0].int(), self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1].int(), self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2].int(), self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)

    # @profile
    def connected_faces_with_mask(self, mask):
        masked_vertex_indices = mask.nonzero(as_tuple=True)[0]
        num_faces = self.indices.shape[0]
        is_vertex_masked = torch.isin(self.indices.reshape(-1), masked_vertex_indices).reshape([num_faces, -1])
        self.faces_with_mask = self.indices[is_vertex_masked.prod(axis=1).nonzero(as_tuple=True)[0]]
        connected_faces_with_mask_ = find_connected_faces(self.faces_with_mask)
        return connected_faces_with_mask_

    def compute_normals_with_mask(self):
        # Compute the face normals
        vertices = self.vertices[self.faces_with_mask]
        a = vertices[:, 0, :]
        b = vertices[:, 1, :]
        c = vertices[:, 2, :]
        return torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1)