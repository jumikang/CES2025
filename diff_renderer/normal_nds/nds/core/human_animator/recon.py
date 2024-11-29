import random
import os
import re
import glob
import numpy as np
import cv2
import collections
import trimesh
import models
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.eval.evaluator_sample import *
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from utils.loader_utils import *
from utils.core import depth2volume

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# cudnn.benchmark = True
# cudnn.fastest = True


class HumanRecon(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='/home/keti/Workspace/DATASET/results',
                 ckpt_path='./checkpoints/CVPRW',
                 half_input=False,
                 half_output=False,
                 center_crop=True,
                 res=512,
                 voxel_size=512,
                 learning_rate=1e-3,
                 start_epoch=1,
                 model_name='DeepHumanNet_CVPRW',
                 eval_metrics=None,
                 device=torch.device('cuda')):
        super(HumanRecon, self).__init__()

        self.result_path = result_path
        self.half_input = half_input
        self.half_output = half_output
        self.center_crop = center_crop
        self.res = res
        self.voxel_size = voxel_size
        self.eval_metrics = eval_metrics
        self.device = torch.device(device)

        # load pre-trained model
        self.model = getattr(models, model_name)(half_input=self.half_input,
                                                 half_output=self.half_output,
                                                 split_last=True)
        self.model.to(self.device)
        self.model.eval()
        optimizer_G = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.load_checkpoint([ckpt_path],
                              self.model, optimizer_G, start_epoch,
                              is_evaluate=False, device=device)

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        self.RGB_MG = [10.0, 10.0, 10.0]

        # os.makedirs(self.result_path, exist_ok=True)

    def load_checkpoint(self, model_paths, model,
                        optimizer, start_epoch,
                        is_evaluate=False, device=None):

        for model_path in model_paths:
            items = glob.glob(os.path.join(model_path, '*.pth.tar'))
            items.sort()

            if len(items) > 0:
                if is_evaluate is True:
                    model_path = os.path.join(model_path, 'model_best.pth.tar')
                else:
                    if len(items) == 1:
                        model_path = items[0]
                    else:
                        model_path = items[len(items) - 1]

                print(("=> loading checkpoint '{}'".format(model_path)))
                checkpoint = torch.load(model_path, map_location=device)
                start_epoch = checkpoint['epoch'] + 1

                if hasattr(model, 'module'):
                    model_state_dict = checkpoint['model_state_dict']
                else:
                    model_state_dict = collections.OrderedDict(
                        {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

                model.load_state_dict(model_state_dict, strict=False)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print('=> generator optimizer has been loaded')
                except:
                    print('=> optimizer(g) not loaded (trying to train a new network?)')

                print(("=> loaded checkpoint (resumed epoch is {})".format(start_epoch)))
                return model, optimizer, start_epoch

        print(("=> no checkpoint found at '{}'".format(model_path)))
        return model, optimizer, start_epoch

    def evaluate(self, input_var, model, device=None):
        model.eval()
        evaluator = HumanEvaluator()

        with torch.no_grad():
            pred_var = model(input_var.unsqueeze(0))

            pred_color = pred_var[0]['pred_color']
            pred_color = torch.chunk(pred_color, chunks=(pred_color.shape[1] // 3), dim=1)

            img_front_np = evaluator.tensor2np_color(pred_color[0])
            img_back_np = evaluator.tensor2np_color(pred_color[1])

            pred_depth = pred_var[1]['pred_depth']
            pred_depth = torch.chunk(pred_depth, chunks=(pred_depth.shape[1]), dim=1)

            # depth_front_np = evaluator.tensor2np_depth(pred_depth[0], dir='front')
            # depth_back_np = evaluator.tensor2np_depth(pred_depth[1], dir='back')

            pred_front_depth = pred_depth[0]
            pred_back_depth = pred_depth[1]

            pred_front_depth[pred_front_depth < 0] = 0
            pred_back_depth[pred_back_depth < 0] = 0

            pred_volume = depth2occ_2view_torch(
                pred_front_depth, pred_back_depth, device=self.device,
                binarize=False, voxel_size=self.voxel_size)

            src_volume = pred_volume.squeeze(0).detach().cpu().numpy()
            pred_mesh, src_model_color = colorize_model(src_volume, img_back_np, img_front_np)
            # pred_mesh.vertices -= pred_mesh.bounding_box.centroid
            # pred_mesh.vertices *= 2 / np.max(pred_mesh.bounding_box.extents)

            # print(pred_mesh.bounding_box.centroid)
            # print(2 / np.max(pred_mesh.bounding_box.extents))
            pred_mesh = self.postprocess_mesh(pred_mesh)
            # print(pred_mesh.bounding_box.centroid)
            # print(2 / np.max(pred_mesh.bounding_box.extents))

        return pred_mesh, img_front_np, img_back_np

    def postprocess_mesh(self, mesh, num_faces=None):
        """Post processing mesh by removing small isolated pieces.

        Args:
            mesh (trimesh.Trimesh): input mesh to be processed
            num_faces (int, optional): min face num threshold. Defaults to 4096.
        """
        total_num_faces = len(mesh.faces)
        if num_faces is None:
            num_faces = total_num_faces // 100
        cc = trimesh.graph.connected_components(
            mesh.face_adjacency, min_len=3)
        mask = np.zeros(total_num_faces, dtype=np.bool)
        cc = np.concatenate([
            c for c in cc if len(c) > num_faces
        ], axis=0)
        mask[cc] = True
        mesh.update_faces(mask)

        return mesh

    def forward(self, images):
        pred_meshes = []
        pred_images = []
        pred_images_back = []
        for i in range(len(images)):
            if not images[i].shape[0] == 512:
                image = cv2.resize(images[i], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                image = images[i]

            if self.center_crop is True:
                width = image.shape[1]
                offset = int(width / 4)
                image = image[:, offset:width - offset, :]

            image = torch.Tensor(image).permute(2, 0, 1).float()
            image = image + torch.Tensor(self.RGB_MG).view(3, 1, 1)
            image = image / torch.Tensor(self.RGB_MAX).view(3, 1, 1)
            image = (image - torch.Tensor(self.RGB_MEAN).view(3, 1, 1)) \
                    / torch.Tensor(self.RGB_STD).view(3, 1, 1)

            if self.device is not None:
                if image is not None:
                    input_var = image.to(self.device)
            if input_var is not None:
                input_var = torch.autograd.Variable(input_var)

            pred_mesh, pred_image_front, pred_image_back = self.evaluate(input_var, self.model, device=self.device)
            pred_meshes.append(pred_mesh)
            pred_images.append(pred_image_front)
            pred_images_back.append(pred_image_back)

        return pred_meshes, pred_images, pred_images_back


class Renderer(nn.Module):
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 result_path='/home/keti/Workspace/DATASET/results',
                 fov=45,
                 res=512,
                 angle=None,
                 axis='x',
                 device=torch.device('cuda')):
        super(Renderer, self).__init__()

        self.result_path = result_path
        self.res = res
        self.fov = fov
        self.angle = angle
        self.axis = axis
        self.device = device

        self.RGB_MEAN = [0.485, 0.456, 0.406]
        self.RGB_STD = [0.229, 0.224, 0.225]
        self.RGB_MAX = [255.0, 255.0, 255.0]
        os.makedirs(self.result_path, exist_ok=True)

    def get_pers_imgs(self, mesh, scene, res, fov):
        scene.camera.resolution = [res, res]
        scene.camera.fov = fov * (scene.camera.resolution /
                                  scene.camera.resolution.max())
        # scene.camera_transform[0:3, 3] = 0.0
        # scene.camera_transform[2, 3] = 1.0
        pers_origins, pers_vectors, pers_pixels = scene.camera_rays()
        pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(
            pers_origins, pers_vectors, multiple_hits=True)
        pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                               pers_vectors[pers_index_ray])
        pers_colors = mesh.visual.face_colors[pers_index_tri]

        pers_pixel_ray = pers_pixels[pers_index_ray]
        pers_depth_far = np.zeros(scene.camera.resolution, dtype=np.float32)
        pers_color_far = np.zeros((res, res, 3), dtype=np.float32)

        pers_depth_near = np.ones(scene.camera.resolution, dtype=np.float32) * res
        pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

        denom = np.tan(np.radians(fov) / 2.0) * 5
        # pers_depth_int = (pers_depth - 3.5)*(res/denom) + res / 2
        pers_depth_int = (pers_depth - np.mean(pers_depth)) * (res / denom) + res / 2

        for k in range(pers_pixel_ray.shape[0]):
            u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
            if pers_depth_int[k] > pers_depth_far[v, u]:
                pers_color_far[v, u, ::-1] = pers_colors[k, 0:3] / 255.0
                pers_depth_far[v, u] = pers_depth_int[k]
            if pers_depth_int[k] < pers_depth_near[v, u]:
                pers_depth_near[v, u] = pers_depth_int[k]
                pers_color_near[v, u, ::-1] = pers_colors[k, 0:3] / 255.0

        pers_depth_near = pers_depth_near * (pers_depth_near != res)
        pers_color_near = np.flip(pers_color_near, 0)
        pers_depth_near = np.flip(pers_depth_near, 0)
        pers_color_far = np.flip(pers_color_far, 0)
        pers_depth_far = np.flip(pers_depth_far, 0)

        return pers_color_near, pers_depth_near, pers_color_far, pers_depth_far

    def rotate_mesh(self, mesh, angle, axis='x'):
        vertices = mesh.vertices
        vertices_re = (np.zeros_like(vertices))
        if axis == 'y':  # pitch
            rotation_axis = np.array([1, 0, 0])
        elif axis == 'x':  # yaw
            rotation_axis = np.array([0, 1, 0])
        elif axis == 'z':  # roll
            rotation_axis = np.array([0, 0, 1])
        else:  # default is x (yaw)
            rotation_axis = np.array([0, 1, 0])

        for i in range(vertices.shape[0]):
            vec = vertices[i, :]
            rotation_degrees = angle
            rotation_radians = np.radians(rotation_degrees)

            rotation_vector = rotation_radians * rotation_axis
            rotation = R.from_rotvec(rotation_vector)
            rotated_vec = rotation.apply(vec)
            vertices_re[i, :] = rotated_vec
        rot_mesh = trimesh.Trimesh(vertices=vertices_re, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)
        return rot_mesh

    def forward(self, mesh):
        mesh_list = []
        image_list = []

        mesh.vertices -= mesh.bounding_box.centroid
        mesh.vertices *= 2 / np.max(mesh.bounding_box.extents)

        scene = mesh.scene()
        scene.camera.resolution = [self.res, self.res]
        pers_color_front, pers_depth_front, pers_color_back, pers_depth_back = \
            self.get_pers_imgs(mesh, scene, self.res, self.fov)
        image_list.append(pers_color_front)

        return mesh_list, image_list


def recon_wrapper(path2image='/home/somewhere/in/your/pc',
                  path2mesh=None,
                  path2checkpoints=None):
    files = []
    images = []
    eval_metrics = "color_visualize_mesh2view_depth_visualize_normal_visualize"
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if path2checkpoints is None:
        path2checkpoints = './checkpoints'
    human_recon = HumanRecon(eval_metrics=eval_metrics, device=device, ckpt_path=path2checkpoints)
    ext = ['PNG', 'png', 'JPG', 'jpg', 'gif']
    [files.extend(sorted(glob.glob(path2image + '/*.' + e))) for e in ext]
    files = sorted(files)

    for f in files:
        images.append(cv2.imread(f, 1))

    # for i in range(len(images)):
    #     images[i] = cv2.resize(images[i], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

    # recon only.
    mesh_list, image_list, image_list_back = human_recon(images)

    # ext = ['ply', 'obj']
    # mesh_files = []
    # [mesh_files.extend(sorted(glob.glob(path2mesh + '/*.' + e))) for e in ext]
    # # generate from ground truth mesh.
    # renderer = Renderer(result_path=path2results, device=device)
    # image_list, mesh_list = renderer(mesh_list)
    #
    # if len(mesh_files) > 0:
    #     for f in mesh_files:
    #         mesh = trimesh.load_mesh(f)
    #         mesh_list, image_list = renderer(mesh)

    # for i in range(len(images)):
    #     images[i] = images[i] / 255.0

    return mesh_list, image_list, image_list_back


def generate_list_files_ori(data_root='/home/somewhere/in/your/pc',
                        path2results=None,
                        fIdx=None,
                        # angle=None,
                        data_type=None):
    path = data_root + '/%s' % data_type + '/data%d/' % fIdx
    files = []
    images = []
    eval_metrics = "color_visualize_mesh2view_depth_visualize_normal_visualize"
    human_recon = HumanRecon(result_path=path2results, eval_metrics=eval_metrics)
    renderer = Renderer(result_path=path2results)
    if data_type == 'image':
        ext = ['png', 'jpg', 'gif']
        [files.extend(sorted(glob.glob(path + '*.' + e))) for e in ext]
        images_angle = []

        for f in files:
            split_path = os.path.split(f)
            images_angle.append(int(split_path[1][:-4]))
            # images.append(cv2.cvtColor(cv2.imread(f, 1), cv2.COLOR_BGR2RGB))
            images.append(cv2.imread(f, 1))
        mesh_list, image_list = human_recon(images, fIdx, images_angle)
        if len(images_angle) == 1:
            mesh_list, image_list = renderer(mesh_list[0], fIdx)

    elif data_type == 'mesh':
        ext = ['ply', 'obj']
        [files.extend(sorted(glob.glob(path + '*.' + e))) for e in ext]

        for f in files:
            mesh = trimesh.load_mesh(f)
            mesh_list, image_list = renderer(mesh, fIdx)

    return mesh_list, image_list


# generate training related files when this function is called as main.
if __name__ == '__main__':
    data_path = '/home/keti/Workspace/IOYS_POSE_T'

    # split_list (os.path.join (data_path, 'list'), filename)
    filename = 'train_all'
    # split_list(os.path.join(data_path, 'list'), filename)
    # generate_list_files(data_root=data_path, filename=filename)