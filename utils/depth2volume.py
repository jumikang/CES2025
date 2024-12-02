import cv2
import math
import time
import torch
import torch.nn.functional as F
import torch.nn
import trimesh
import numpy as np
from PIL import Image
from skimage import measure
# from torchmcubes import grid_interp, marching_cubes
# pip install git+https://github.com/tatsy/torchmcubes.git
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def make_rotation_matrix(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def postprocess_mesh( mesh, num_faces=None):
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
    mask = np.zeros(total_num_faces, dtype=bool)
    cc = np.concatenate([
        c for c in cc if len(c) > num_faces
    ], axis=0)
    mask[cc] = True
    mesh.update_faces(mask)

    return mesh


def volume_filter(volume, iter=1):
    filters = torch.ones(1, 1, 3, 3, 3) / 27  # average filter
    for _ in range(iter):
        volume = F.conv3d(volume, filters, padding=1)
    return volume.squeeze()


def gen_volume_coordinate(xy, z_min=120, z_max=320, voxel_size=512):
    grid = torch.ones((3, xy.shape[1], xy.shape[2], voxel_size))
    z_range = z_max - z_min
    slope = z_range / voxel_size
    ones = torch.ones_like(xy[0:1, :, :])
    for k in range(voxel_size):
        z = z_min + slope * k
        grid[:, :, :, k] = torch.cat((xy * z, ones * z), dim=0)

    return grid


def grid_interpolation(mesh,
                       res=512,
                       fx=724.077343935,
                       fy=724.0773439357,
                       px=256.0,
                       py=256.0,
                       z_min=200.0,
                       z_max=400.0,
                       voxel_size=512.0):

    x = np.reshape((np.linspace(0, res, res) - int(px)) / fx,
                   [1, 1, 1, -1])
    y = np.reshape((np.linspace(0, res, res) - int(py)) / fy,
                   [1, 1, -1, 1])
    x = np.tile(x, [1, 1, res, 1])
    y = np.tile(y, [1, 1, 1, res])
    xy = torch.Tensor(np.concatenate((y, x), axis=1))
    coord = gen_volume_coordinate(xy=xy[0],
                                  z_min=z_min,
                                  z_max=z_max,
                                  voxel_size=voxel_size)

    new_vertices = grid_interp(coord, torch.Tensor(mesh.vertices))
    R = make_rotation_matrix(0, math.radians(0), math.radians(-90))
    vertices = np.matmul(np.asarray(new_vertices), R.transpose(1, 0))
    vertices[:, 2] *= (-1)  # look front (from back)
    # vertices[:, 1] += 0  # self.camera_height  # 60.0 # 308.0/512.0/self.focal*220.0
    # vertices[:, 2] += 300.0

    pred_mesh = trimesh.Trimesh(vertices=vertices,
                                faces=mesh.faces,
                                visual=mesh.visual)
    return pred_mesh


def volume2meshinfo(sdf, visualize=True, level=0.0):
    from torchmcubes import marching_cubes
    vertices, faces = marching_cubes(torch.Tensor(sdf), level)
    normals = None

    return vertices.numpy(), faces.numpy(), normals


def depth2occ_2view_torch_wcolor(color_front, color_back, depth_front, depth_back, voxel_size=256, slope=0.01, binarize=True, device=None):

    if depth_front.shape[1] == 1:
        depth_front = depth_front.squeeze(1)
    if depth_back.shape[1] == 1:
        depth_back = depth_back.squeeze(1)

    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size))
    occ_grid_color = torch.ones((color_front.shape[0], color_front.shape[1], color_front.shape[2], voxel_size))

    if device is not None:
        occ_grid = torch.autograd.Variable(occ_grid)
        occ_grid = occ_grid.to(device)
        occ_grid_color = torch.autograd.Variable(occ_grid_color)
        occ_grid_color = occ_grid_color.to(device)

    cost_front = depth_front * slope * voxel_size
    cost_back = (1 - depth_back * voxel_size) * slope

    for k in range(0, voxel_size):
        occ_grid[:, :, :, k] = torch.max(cost_front - slope * k, cost_back + slope * k)

    if occ_grid.shape[2] < occ_grid.shape[3]:
        offset = int(occ_grid.shape[2] / 2)
        occ_grid = torch.nn.functional.pad(occ_grid, (0, 0, offset, offset), "constant", 1)

    if binarize:
        occ_grid[occ_grid > 0] = 1
        occ_grid[occ_grid <= 0] = 0

    return occ_grid


def depth2sdf(depth_front,
              depth_back,
              z_min=120,
              z_max=320,
              voxel_size=256,
              binarize=False,
              device=None):

    if depth_front.shape[1] == 1:
        depth_front = depth_front.squeeze(1)
    if depth_back.shape[1] == 1:
        depth_back = depth_back.squeeze(1)
    z_range = z_max - z_min
    slope = z_range / voxel_size
    occ_grid = torch.ones((depth_front.shape[0], depth_front.shape[1], depth_front.shape[2], voxel_size))

    if device is not None:
        occ_grid = torch.autograd.Variable(occ_grid)
        occ_grid = occ_grid.to(device)

    cost_front = depth_front - z_min
    cost_back = z_min - depth_back

    for k in range(0, voxel_size):
        occ_grid[:, :, :, k] = torch.max(cost_front - slope * k, cost_back + slope * k)

    if binarize:
        occ_grid[occ_grid > 0] = 1.0
        occ_grid[occ_grid <= 0] = -1.0
    else:
        occ_grid[occ_grid > 5] = 5.0
        occ_grid[occ_grid < -5] = -5.0
        occ_grid /= 5.0
    return occ_grid


def colorize_model2(pred_mesh, img_front, img_back=None, flip_image=False):
    vertex_num = pred_mesh.vertices.shape[0]
    vertices = pred_mesh.vertices

    pred_normals = trimesh.geometry.weighted_vertex_normals(vertex_num, pred_mesh.faces,
                                                            pred_mesh.face_normals,
                                                            pred_mesh.face_angles,
                                                            use_loop=False)

    model_colors = np.zeros_like(pred_normals)
    if flip_image:
        # img_front = np.flip(img_front, 1)
        img_front = np.rot90(img_front, k=1)
        img_front = np.flip(img_front, axis=0)

        if img_back is not None:
            img_back = np.rot90(img_back, k=1)
            img_back = np.flip(img_back, axis=0)

    for k in range(vertex_num):
        u, v = vertices[k, 2], vertices[k, 1]
        u_d = u - np.floor(u)

        if img_front.shape[0] == 256:
            u = min(u.astype(int), 254)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 254)
        else:
            u = min(u.astype(int), 510)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 510)

        if pred_normals[k, 0] < 0.0:
            model_colors[k, 0] = (img_front[v, u, 2] * v_d + img_front[v + 1, u, 2] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 2] * v_d + img_front[v + 1, u + 1, 2] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 1] = (img_front[v, u, 1] * v_d + img_front[v + 1, u, 1] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 1] * v_d + img_front[v + 1, u + 1, 1] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 2] = (img_front[v, u, 0] * v_d + img_front[v + 1, u, 0] * (
                    1 - v_d)) * u_d + \
                                 (img_front[v, u + 1, 0] * v_d + img_front[v + 1, u + 1, 0] * (
                                         1 - v_d)) * (1 - u_d)
        elif img_back is not None:
            model_colors[k, 0] = (img_back[v, u, 2] * v_d + img_back[v + 1, u, 2] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 2] * v_d + img_back[v + 1, u + 1, 2] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 1] = (img_back[v, u, 1] * v_d + img_back[v + 1, u, 1] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 1] * v_d + img_back[v + 1, u + 1, 1] * (
                                         1 - v_d)) * (1 - u_d)
            model_colors[k, 2] = (img_back[v, u, 0] * v_d + img_back[v + 1, u, 0] * (
                    1 - v_d)) * u_d + \
                                 (img_back[v, u + 1, 0] * v_d + img_back[v + 1, u + 1, 0] * (
                                         1 - v_d)) * (1 - u_d)

    color_mesh = trimesh.Trimesh(vertices=vertices,
                                 vertex_colors=model_colors,
                                 faces=pred_mesh.faces,
                                 process=False,
                                 maintain_order=True)
    return color_mesh


def colorize_model(pred_mesh, img_front, img_back, mask=None, subdivide=False, texture_map=False):
    if subdivide:
        pred_mesh = pred_mesh.subdivide()

    vertices = pred_mesh.vertices
    faces = pred_mesh.faces
    vertex_num = vertices.shape[0]

    pred_normals = trimesh.geometry.weighted_vertex_normals(vertex_num, faces,
                                                            pred_mesh.face_normals,
                                                            pred_mesh.face_angles,
                                                            use_loop=False)

    model_colors = np.zeros_like(pred_normals)
    if texture_map:
        img_front = np.flip(img_front, axis=0)
        img_back = np.flip(img_back, axis=0)
    else:
        img_front = np.rot90(img_front, k=1)
        img_front = np.flip(img_front, axis=0)
        img_back = np.rot90(img_back, k=1)
        img_back = np.flip(img_back, axis=0)
        mask = np.rot90(mask, k=1)
        mask = np.flip(mask, axis=0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iter_erode = 3
    iter_dilate = 4
    img_front_eroded = cv2.erode(img_front, kernel, iterations=iter_erode)
    img_back_eroded = cv2.erode(img_back, kernel, iterations=iter_erode)
    if mask is not None:
        mask = cv2.resize(mask, (img_front.shape[0], img_front.shape[1]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.sum(img_front_eroded)
        mask[mask > 0] = 255.0
    mask_eroded = cv2.erode(mask, kernel, iterations=iter_erode)

    img_front_dialated = cv2.dilate(img_front_eroded, kernel, iterations=iter_dilate)
    img_back_dialated = cv2.dilate(img_back_eroded, kernel, iterations=iter_dilate)
    mask_dialated = cv2.dilate(mask_eroded, kernel, iterations=iter_dilate)
    dist = cv2.distanceTransform(1 - mask_eroded.astype(np.uint8), cv2.DIST_L2, 5)

    min_val = 0
    max_val = 7
    bw = np.clip(dist, a_min=min_val, a_max=max_val) / (max_val*2)  # 0.5~1.0
    bw[dist <= min_val] = 0.0
    bw[dist > max_val] = 0.5

    for i in range(img_front.shape[1]):
        for j in range(img_front.shape[0]):
            if mask_eroded[j, i] == 0 and mask_dialated[j, i] > 0:
                img_front[j, i, :] = img_front_dialated[j, i, :]
                img_back[j, i, :] = img_back_dialated[j, i, :]
    img_front = (img_front * 255)
    img_back = (img_back * 255)

    # resize and crop
    def resize_and_crop(image, d=1):
        w, h = image.shape[1]+d*2, image.shape[0]+d*2
        image = cv2.resize(image, (h, w))
        image = image[d:w-1, d:h-1, :]
        return image
    img_front = resize_and_crop(img_front, d=1)
    img_back = resize_and_crop(img_back, d=1)
    vts_uv = np.zeros_like(vertices[:, 0:2])

    for k in range(vertex_num):
        # scikit-learn marching cubes (original)
        # u, v = vertices[k, 0], vertices[k, 1]
        # torchmcubes marching cubes (reversed coordinate and colors)
        u, v = vertices[k, 2], vertices[k, 1]
        u *= 2  # res==1024
        v *= 2  # res==1024
        u_d = u - np.floor(u)
        if img_front.shape[0] == 1024:
            u = min(u.astype(int), 1022)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 1022)
        elif img_front.shape[0] == 512:
            u = min(u.astype(int), 510)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 510)
        elif img_front.shape[0] == 2048:
            u = min(u.astype(int), 2046)
            v_d = v - np.floor(v)
            v = min(v.astype(int), 2046)

        rgb_f = np.zeros(3)
        rgb_b = np.zeros(3)

        # if pred_normals[k, 0] < 0.0:
        vts_uv[k, 0] = (v + v_d) / 1024 # 512#2048#512
        vts_uv[k, 1] = (u + u_d) / 2048 + 0.5 #1024 + 0.5#4096 + 0.5#1024 + 0.5

        rgb_f[0] = (img_front[v, u, 2] * v_d + img_front[v + 1, u, 2] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 2] * v_d + img_front[v + 1, u + 1, 2] * (1 - v_d)) * (1 - u_d)
        rgb_f[1] = (img_front[v, u, 1] * v_d + img_front[v + 1, u, 1] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 1] * v_d + img_front[v + 1, u + 1, 1] * (1 - v_d)) * (1 - u_d)
        rgb_f[2] = (img_front[v, u, 0] * v_d + img_front[v + 1, u, 0] * (1 - v_d)) * u_d + \
                             (img_front[v, u + 1, 0] * v_d + img_front[v + 1, u + 1, 0] * (1 - v_d)) * (1 - u_d)

        if pred_normals[k, 0] == 0.0:
            pred_normals[k, :] = 0
        vts_uv[k, 0] = (v + v_d) / 1024 #512#2048#512
        vts_uv[k, 1] = (u + u_d) / 2048 #1024#4096#1024

        rgb_b[0] = (img_back[v, u, 2] * v_d + img_back[v + 1, u, 2] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 2] * v_d + img_back[v + 1, u + 1, 2] * (1 - v_d)) * (1 - u_d)
        rgb_b[1] = (img_back[v, u, 1] * v_d + img_back[v + 1, u, 1] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 1] * v_d + img_back[v + 1, u + 1, 1] * (1 - v_d)) * (1 - u_d)
        rgb_b[2] = (img_back[v, u, 0] * v_d + img_back[v + 1, u, 0] * (1 - v_d)) * u_d + \
                             (img_back[v, u + 1, 0] * v_d + img_back[v + 1, u + 1, 0] * (1 - v_d)) * (1 - u_d)

        if pred_normals[k, 0] < 0.0:
            if bw[v, u] > 0:
                model_colors[k, :] = rgb_f*(1 - bw[v, u]) + rgb_b*bw[v, u]
            else:
                model_colors[k, :] = rgb_f
        else:
            if bw[v, u] > 0:
                model_colors[k, :] = rgb_b*(1 - bw[v, u]) + rgb_f*bw[v, u]
            else:
                model_colors[k, :] = rgb_b

    normals = pred_normals
    if texture_map:
        texture_map = np.concatenate([img_front.astype(np.uint8), img_back.astype(np.uint8)], axis=0)
        texture_map = Image.fromarray(texture_map[:, :, ::-1])
        texture_visual = trimesh.visual.TextureVisuals(uv=vts_uv, image=texture_map)
        color_mesh = trimesh.Trimesh(vertices=vertices,
                                     vertex_colors=model_colors,
                                     vertex_normals=normals,
                                     faces=faces,
                                     visual=texture_visual,
                                     process=True,
                                     maintain_order=False)
    else:
        color_mesh = trimesh.Trimesh(vertices=vertices,
                                     vertex_colors=model_colors,
                                     vertex_normals=normals,
                                     faces=faces,
                                     process=True,
                                     maintain_order=False)

    return color_mesh


if __name__ == '__main__':

    # dataset_path = 'I:/smplx_dataset/VOXEL/SMPLX_50/0_voxel.npy'
    # data = np.load(dataset_path)
    end = time.time ()
    # sdf_volume = depth2volume_float(data, voxel_size=256, z_level=256)
    # print ('%0.2f sec.\n' % (time.time() - end))
    # volume2mesh (np.squeeze (sdf_volume))

    # # depth_front = cv2.imread ('I:/smplx_dataset/DEPTH/SMPLX_50/0_front.png', cv2.IMREAD_GRAYSCALE)
    # # depth_back = cv2.imread ('I:/smplx_dataset/DEPTH_PRED/SMPLX_50/0_back.png', cv2.IMREAD_GRAYSCALE)
    depth_front = cv2.imread('E:/iois_dataset/EVAL/_order1_keticvl-work10_d2d/input/10.png', cv2.IMREAD_GRAYSCALE)
    depth_back = cv2.imread('E:/iois_dataset/EVAL/_order1_keticvl-work10_d2d/pred/10.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow ('hi', depth_front)
    # cv2.waitKey (0)
    #
    depth_front = depth_front.astype(np.float)
    depth_back = depth_back.astype(np.float)

    end = time.time()
    for k in range(30):
        sdf_volume = depth2occ_double(depth_front, depth_back, voxel_size=256)
    print('%0.2f sec.\n' % (time.time() - end))
    volume2colormesh (np.squeeze(sdf_volume), level=0.5)

    # # print(np.max(depth_front))
    # # print(np.max(depth_back))
    # # depth_front[depth_front == 0] = 255
    # # print (np.min (depth_front))
    # #
    # input_stacked = np.stack ((depth_front, depth_back), axis=2)
    # sdf_volume = depth2volume_float(input_stacked, voxel_size=256, z_level=256)
    # volume2mesh (np.squeeze (sdf_volume))
    # cv2.imshow('hi', depth_front)
    # cv2.waitKey(0)

    # depth_front = cv2.imread ('F:/NIA(2020)/201012/201012_Inho1_results/Depth/temp.png', cv2.IMREAD_GRAYSCALE)
    # depth_map = depth_front.astype (np.float)
    # # detph_map = cv2.resize(depth_map, (256, 256))
    # sdf_volume = depth2volume_single (depth_map, voxel_size=256, z_level=256)
    # volume2mesh (np.squeeze (sdf_volume))

    # depth_front = cv2.imread ('I:/smplx_dataset/DEPTH/SMPLX_50/0_front.png', cv2.IMREAD_GRAYSCALE)
    # sdf_volume = depth2volume_lstm (depth_front, voxel_size=256, z_level=128)
    # volume2mesh (np.squeeze (sdf_volume))
    # print('hi')
    # end = time.time()
    # out = F.interpolate (torch.Tensor(np.expand_dims(sdf_volume, axis=0)), (256, 256), mode='bilinear',
    #                      align_corners=True)
    # new_sdf = out.numpy()
    # print ('%0.2f sec.\n' % (time.time() - end))



