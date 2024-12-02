import os
import trimesh
import numpy as np
import open3d as o3d
from torch import nn


class LightOpen3DRenderer(nn.Module):
    def __init__(self, cam_params=None, res=1024):
        super(LightOpen3DRenderer, self).__init__()
        if cam_params is None:
            cam_params = {'width': res, 'height': res, 'fx': 532.37, 'fy': 532.37, 'px': res/2, 'py': res/2}

        self.res = res
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.res, height=self.res)
        self.vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().light_on = True  # slower

        self.ctr = self.vis.get_view_control()
        self.cam_params = self.ctr.convert_to_pinhole_camera_parameters()
        self.cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(cam_params['width'], cam_params['height'],
                                                                      cam_params['fy'], cam_params['fy'],
                                                                      cam_params['px'], cam_params['py'])
        self.cam_params.extrinsic = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 3.0], [0.0, 0, 0, 1]])
        self.mesh_o3d = None
        self.set_volume()

    def set_volume(self, d=5.0):
        # d is an offset value, assuming the object is centered
        mesh_o3d = o3d.geometry.PointCloud()
        vertices = np.asarray([[d, d, d], [d, d, -d], [d, -d, d], [-d, d, d],
                               [-d, -d, -d], [-d, -d, d], [-d, d, -d], [d, -d, -d],])

        mesh_o3d.points = o3d.utility.Vector3dVector(vertices)
        self.vis.add_geometry(mesh_o3d)

    def set_mesh(self):
        if self.mesh_o3d is not None:
            self.vis.remove_geometry(self.mesh_o3d)
            self.mesh_o3d = None

    # mesh, point cloud
    def render(self, vertices, faces, vertex_colors, method='pcd'):
        # Input:
        # > mesh (list): a list of meshes
        # > method : 'pcd' or 'mesh' in
        # vertices, faces, vertex_colors = input
        if self.mesh_o3d is None:
            # mesh
            if method == 'pcd':
                self.mesh_o3d = o3d.geometry.PointCloud()
                self.mesh_o3d.colors = o3d.cuda.pybind.utility.Vector3dVector(vertex_colors[:, 0:3] / 255)
                self.mesh_o3d.points = o3d.cuda.pybind.utility.Vector3dVector(vertices)
            else:
                self.mesh_o3d = o3d.geometry.TriangleMesh()
                self.mesh_o3d.triangles = o3d.cuda.pybind.utility.Vector3iVector(faces)
                self.mesh_o3d.vertex_colors = o3d.cuda.pybind.utility.Vector3dVector(vertex_colors[:, 0:3] / 255)
                self.mesh_o3d.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
            self.vis.add_geometry(self.mesh_o3d)
        else:
            if method == 'pcd':
                self.mesh_o3d.points = o3d.cuda.pybind.utility.Vector3dVector(vertices)
            else:
                self.mesh_o3d.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)

        self.vis.update_geometry(self.mesh_o3d)
        self.ctr.set_zoom(.2)
        self.vis.poll_events()
        self.vis.update_renderer()
