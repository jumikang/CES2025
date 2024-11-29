import cv2
import torch
import trimesh
import numpy as np
from PIL import Image
from diff_renderer.normal_nds.nds.utils.geometry import find_connected_faces
from diff_renderer.normal_nds.nds.core.mesh import Mesh
# from mesh import Mesh


class TexturedMesh(Mesh):
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        uv_vts (tensor) : UV coordinate for vertices (Vx2)
        uv_faces (tensor) : indices of uv_vts w.r.t. faces (Fx3)
        tex (tensor) : texture map (HxWx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, uv_vts, tex=None, device='cuda:0'):
        super(Mesh).__init__()

        self.device = device
        self.vertices = self.to_torch(vertices, device)
        self.indices = self.to_torch(indices, device).type(torch.int64)
        self.uv_vts = self.to_torch(uv_vts, device)

        if tex is not None:
            self.tex = self.to_torch(tex, device) / 255.0
        else:
            self.tex = torch.ones((1024, 1024, 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device) * 0.2

        # displacement map (for optimization purpose, used internally)
        self.disp = torch.zeros((1024, 1024, 3),
                              dtype=torch.float32,
                              requires_grad=True).to(device)

        # additional information that are calculated internally
        self.face_normals = None
        self.vertex_normals = None
        self._edges = None
        self._connected_faces = None
        self._laplacian = None

        if self.indices is not None:
            self.compute_normals()

    @staticmethod
    def to_torch(data, device, dtype=torch.float32):
        if data is None:
            return None
        elif torch.is_tensor(data):
            torch_tensor = data.to(device, dtype=dtype)
        else:
            torch_tensor = torch.tensor(data.copy(), dtype=dtype, device=device)
        return torch_tensor

    # sampling vertex colors and displacement vectors in the uv space.
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
        else:
            tex = self.disp.permute(2, 0, 1).unsqueeze(0)
        samples = torch.nn.functional.grid_sample(tex, uv, align_corners=True)  # [B, C, N, 1]
        return samples.squeeze().transpose(1, 0).contiguous()

    def subdivide(self):
        # adaptive subdivision (generate irregular meshes, not recommend for training)
        # e0, e1 = self.edges.unbind(1)
        # average_edge_length = torch.linalg.norm(self.vertices[e0] - self.vertices[e1], dim=-1).mean()
        # cutoff = average_edge_length.detach().cpu().numpy() * 2.0
        # mesh = mesh.subdivide_to_size(cutoff, max_iter=10)
        mesh = self.to_trimesh(with_texture=False)
        mesh = mesh.subdivide()

        self.vertices = torch.FloatTensor(mesh.vertices).to(self.device)
        self.indices = torch.Tensor(mesh.faces).type(torch.int64).to(self.device)
        self.uv_vts = torch.FloatTensor(mesh.visual.uv).to(self.device)
        self.compute_normals()

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



