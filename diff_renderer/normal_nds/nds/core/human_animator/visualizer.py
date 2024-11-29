import glob
import pickle
import random
import numpy as np
import torch
import cv2
import tqdm
import io
from PIL import Image
from utils.visualizer import to_trimesh


class Visualizer():
    def __init__(self, path2motion=None, seq=0):
        self.motion_datalist = glob.glob(path2motion)
        self.motion = None

        self.cur = 0
        self.seq = seq
        self.set_motion_data()
        self.n = 480

    def set_motion_data(self):
        with open(self.motion_datalist[self.seq], 'rb') as f:
            self.motion = pickle.load(f)

    def record_video(self):
        pass

    def _set_motion(self):
        mv = self.motion['smpl_poses'][self.cur]
        self.global_orient = torch.tensor(mv[:3]).unsqueeze(0)
        self.body_pose = torch.tensor(mv[3:-6]).unsqueeze(0)
        self.cur += 10

    def get_random_motion(self):
        idx = random.randrange(0, self.n-1)
        mv = self.motion['smpl_poses'][idx]
        # print(idx)
        return mv

    def get_cur_frame(self, avatars, idx=None, vis=False):
        if idx is not None:
            self.cur = idx
        self._set_motion()
        k = 0
        vertices = avatars.deform(k, global_pose=self.global_orient, body_pose=self.body_pose)
        reposed = to_trimesh(vertices, avatars.meshes[k].faces,
                             vertex_colors=avatars.meshes[k].visual.vertex_colors)

        if vis == True:
            reposed.show()
        reposed.export('test.obj')
        return reposed

    def save_as_video(self, avatars, idx_motion, idx_obj):
        memo = ''
        video_filename = './output_seq{0}_obj{1}'.format(idx_motion, idx_obj) + memo + '.avi'
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(video_filename, fourcc, 10, (width, height))

        if not video.isOpened():
            print('Failed to open a video file')
            video.release()
            return None

        with tqdm.tqdm(enumerate(self.motion['smpl_poses'])) as pbar:
            for k, mv in pbar:
                warped_mesh = self.get_cur_frame(avatars, idx=k, vis=False)

                scene = warped_mesh.scene()
                scene.set_camera(distance=3.0)

                image_data = scene.save_image(resolution=(width, height), visible=True)

                image = Image.open(io.BytesIO(image_data))
                image = np.asarray(image).astype('uint8')
                rgb_image = image[:, :, ::-1]
                rgb_image = rgb_image[:, :, 1:]
                video.write(rgb_image)
                cv2.waitKey(10)

                # save 300 frames only.
                # if k == 20:
                #     break
            video.release()
            cv2.destroyAllWindows()

            pbar.set_description('[{0}/{1}]'.format(k, len(self.motion['smpl_poses'])))