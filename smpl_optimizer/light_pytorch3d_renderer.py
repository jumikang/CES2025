import os
import trimesh
import torch
import numpy as np
import pytorch3d
import cv2
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from torch import nn


class LightPytorch3DRenderer(nn.Module):
    def __init__(self, cam_params=None, res=768, device='cuda:0'):
        super(LightPytorch3DRenderer, self).__init__()
        if cam_params is None:
            cam_params = {'width': res, 'height': res, 'fx': 532.37, 'fy': 532.37, 'px': res/2, 'py': res/2}

        self.res = res
        self.device = device
        R, T = look_at_view_transform(3.0, 5.0, 0)
        # T[0, 1] = -0.5  # height from the ground (in meter)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        self.cameras = cameras
        raster_settings = RasterizationSettings(
            image_size=(768, 1024),
            blur_radius=0.0,
            faces_per_pixel=1
            # bin_size=64  # 64
        )
        self.lights = PointLights(device=device, location=[[1.0, 3.0, 3.0]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=self.lights
            )
        )
        self.mesh = None

    def trimesh2pytorch3d(self, mesh, render_normal=False):
        vertices = torch.FloatTensor(mesh.vertices[None, :, :]).to(self.device)
        faces = torch.FloatTensor(mesh.faces[None, :, :]).to(self.device)
        if render_normal:
            textures = torch.FloatTensor(mesh.vertex_normals[None, :, :3] / 2.0 + 0.5).to(self.device)
        else:
            textures = torch.FloatTensor(mesh.visual.vertex_colors[None, :, :3] / 255.0).to(self.device)
        textures = TexturesVertex(verts_features=textures)
        mesh_pytorch3d = Meshes(verts=vertices, faces=faces, textures=textures)
        return mesh_pytorch3d

    def set_mesh(self, mesh, render_normal=False):
        if isinstance(mesh, pytorch3d.structures.Meshes):
            self.mesh = mesh
        else:
            self.mesh = self.trimesh2pytorch3d(mesh, render_normal=render_normal)

    def render_ploty(self, mesh):
        if not isinstance(mesh, pytorch3d.structures.Meshes):
            mesh = self.trimesh2pytorch3d(mesh)

        fig = plot_scene({
            "canonical mesh": {
                "cow_mesh": mesh
            }
        })
        fig.show()

    def set_camera(self, dist=3.0, elev=5.0, azim=0):
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.renderer.rasterizer.cameras = cameras

    # mesh, point cloud
    def render(self, vertices, vis=False, reset_cam=False, dist=3.0, elev=5.0, azim=0.0):
        # Input:
        # > vertices: torch matrix (B x N x 3)
        self.mesh = self.mesh.update_padded(vertices)
        if reset_cam:
            self.set_camera(dist, elev, azim)

        images = self.renderer(self.mesh)
        # self.renderer.
        img_np = images[0, ..., :3].detach().cpu().numpy()
        if vis:  # turn off this when it is run on the server.
            cv2.imshow("rendered", img_np[:, :, ::-1])
            cv2.waitKey(1)
        return img_np
