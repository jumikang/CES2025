import os
import json
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES=True

def create_dataset(opt, validation=False):
    """
    Create HumanScanDataset from config.yaml
    :param params: train and validation parameters
    :param validation: create different datasets for different configurations
    :return: dataloader instance
    """
    # opt = params['validation'] if validation else params['train']
    if validation:
        data_list = os.path.join(opt.data.root_dir, opt.data.validation.data_list)
    else:
        data_list = os.path.join(opt.data.root_dir, opt.data.train.data_list)
    dataset = HumanScanDataset(opt, data_list=data_list, validation=validation)
    if not validation:
        num_workers = opt.data.train.num_workers
        batch_size = opt.data.train.batch_size
    else:
        num_workers = opt.data.validation.num_workers
        batch_size = opt.data.validation.batch_size
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False if validation else True,
        drop_last=True,
        pin_memory=False,
        persistent_workers=True if num_workers == 0 else False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

class HumanScanDataset(Dataset):
    def __init__(self,
                 opt,
                 data_list=None,
                 validation=False) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(opt['DATA']['data_root'])
        self.return_disp = opt['DATA']['return_disp']
        self.return_uv = opt['DATA']['return_uv']
        self.validation = validation

        if data_list is not None:
            with open(data_list) as f:
                self.paths = json.load(f)
            print('============= length of dataset %d =============' % len(self.paths))
        self.res = opt['DATA']['res']
        self.pred_canon = opt['DATA']['pred_canon']
        self.dr_loss = opt['DATA']['dr_loss']
        self.RGB_MAX = np.array([255.0, 255.0, 255.0])
        self.RGB_MEAN = np.array([0.485, 0.456, 0.406])  # vgg mean
        self.RGB_STD = np.array([0.229, 0.224, 0.225])  # vgg std

        self.tform = transforms.Compose(
            [
                transforms.Resize(self.res),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # (0,1)->(-1,1)
            ]
        )

    def load_image(self, path, wa=False):
        # return: numpy array
        image = np.array(Image.open(path))

        if not image.shape[1] == self.res:
            image = cv2.resize(image, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        if wa:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_normal(self, path, wa=False):
        normal = np.array(Image.open(path))
        if not normal.shape[1] == self.res:
            normal = cv2.resize(normal, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        if wa:
            normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)
        else:
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return normal

    def load_dr(self, im, cam_nums=2):
        dr_img_path = (im.replace('DIFFUSE', 'ALBEDO').replace('_front.', '.'))
        dr_normal_path = dr_img_path.replace('COLOR', 'NORMAL')
        dr_depth_path = dr_img_path.replace('COLOR', 'DEPTH')

        dr_img_paths = []
        dr_normal_paths = []
        dr_depth_paths = []
        h = torch.zeros(cam_nums)
        v = torch.zeros(cam_nums)
        for i in range(0, cam_nums):
            h_degree = random.randrange(0, 360, 2)
            v_degree = random.randrange(-10, 11, 10)
            h[i] = h_degree
            v[i] = v_degree

            if v_degree<0:
                v_degree += 360

            file_name = dr_img_path.split('/')[-1]
            rot_file_name = '%03d_%03d_000.png' % (v_degree, h_degree)
            dr_img = dr_img_path.replace(file_name, rot_file_name)
            dr_normal = dr_normal_path.replace(file_name, rot_file_name)
            dr_depth = dr_depth_path.replace(file_name, rot_file_name)
            dr_img_paths.append(dr_img)
            dr_normal_paths.append(dr_normal)
            dr_depth_paths.append(dr_depth)

        return dr_img_paths, dr_normal_paths, dr_depth_paths, h, v

    def load_in_the_wild(self, img, dense_uv, wa=False):
        if not img.shape[1] == self.res:
            img = cv2.resize(img, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        # if wa:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_tensor = torch.FloatTensor(img).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))

        if not dense_uv.shape[1] == self.res:
            dense_uv = cv2.resize(dense_uv, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        # if wa:
        #     dense_uv = cv2.cvtColor(dense_uv, cv2.COLOR_BGRA2RGB)
        # else:
        dense_uv = cv2.cvtColor(dense_uv, cv2.COLOR_BGR2RGB)

        uv_tensor = torch.FloatTensor(dense_uv).permute(2, 0, 1)
        uv_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_uv_tensor =\
            ((uv_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))

        cond_posed = scaled_image_tensor
        cond_dense = scaled_uv_tensor
        return {"image_cond":cond_posed[None, ...],
                "dense_cond": cond_dense[None, ...]}

    def process_im(self, im, wa=False):
        # return: torch tensor
        image = self.load_image(im, wa=wa)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        # image_tensor = torch.flipud(torch.FloatTensor(image)).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))
        return scaled_image_tensor

    def process_dr_im(self, im, wa=False):
        # return: torch tensor
        image = self.load_image(im, wa=wa)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))
        return scaled_image_tensor

    def process_dr_normal(self, normal, wa=False):
        # return: torch tensor
        normal = self.load_normal(normal, wa=wa)
        normal_tensor = torch.FloatTensor(normal).permute(2, 0, 1)
        normal_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        normal_tensor = normal_tensor * 2 - 1
        return normal_tensor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = {}

        # input data load
        path2img_posed = os.path.join(self.root_dir, self.paths[index])
        path2img_posed = path2img_posed.replace('/./', '/')
        path2img_dense = path2img_posed.replace('DIFFUSE', 'DENSE_UV')

        cond_posed = self.process_im(path2img_posed, wa=True)
        cond_dense = self.process_im(path2img_dense, wa=False)
        data["image_cond"] = cond_posed
        data["uv_cond"] = cond_dense
        if self.validation:
            data["pred_uv_path"] = path2img_posed.replace('DIFFUSE', 'PRED_UV')
            data["pred_disp_path"] = (path2img_posed.replace('COLOR/DIFFUSE', 'DISP')
                                      .replace('png', 'pkl'))

        # target data load
        f_name = path2img_posed.split('/')[-1]
        data_name = path2img_posed.split('/')[-2]

        label_img = (path2img_posed.replace('CES/COLOR/DIFFUSE/', 'UV_CANON/')
                     .replace(f_name, 'material_0.png'))
        mesh_dic = np.load(label_img.replace('material_0.png', 'meshes_info.pkl'), allow_pickle=True)
        mesh_param_path = (label_img.replace('UV_CANON', 'SMPLX_UV')
                           .replace('material_0.png', '%s.json' % data_name))
        if self.return_uv:
            target_im = self.process_im(label_img, wa=False)
            data["uv_target"] = target_im

        if self.return_disp:
            if self.pred_canon:
                data["disp_target"] = torch.FloatTensor(mesh_dic['canon_disp']).transpose(1, 0)
            else:
                data["disp_target"] = torch.FloatTensor(mesh_dic['pose_disp']).transpose(1, 0)

        if self.dr_loss:
            data["smpl_param_path"] = mesh_param_path
            dr_img_paths, dr_normal_paths, dr_depth_paths, angle_h, angle_v \
                = self.load_dr(path2img_posed)
            dr_imgs = []
            dr_normals = []
            for i in range(len(dr_img_paths)):
                dr_imgs.append(self.process_dr_im(dr_img_paths[i], wa=True))
                dr_normals.append(self.process_dr_normal(dr_normal_paths[i], wa=True))
            data["dr_imgs_target"] = dr_imgs
            data["dr_normals_target"] = dr_normals
            data["angle_h"] = torch.FloatTensor(angle_h)
            data["angle_v"]= torch.FloatTensor(angle_v)

            if self.pred_canon:
                data["disp_target"] = torch.FloatTensor(mesh_dic['canon_disp']).transpose(1, 0)
                data["smpl_vertices"] = torch.FloatTensor(mesh_dic['canon_verts'])
                data["smpl_faces"] = torch.Tensor(mesh_dic['faces'])
                data["smpl_uv_vts"] = torch.FloatTensor(mesh_dic['uv_vts'])
                data["lbs_weights"] = torch.FloatTensor(mesh_dic['lbs_weights'])
                data["A"] = torch.FloatTensor(mesh_dic['A'])
            else:
                data["disp_target"] = torch.FloatTensor(mesh_dic['pose_disp']).transpose(1, 0)
                data["smpl_vertices"] = mesh_dic['posed_verts']
                data["smpl_faces"] = mesh_dic['faces']
                data["smpl_uv_vts"] = mesh_dic['uv_vts']

        return data