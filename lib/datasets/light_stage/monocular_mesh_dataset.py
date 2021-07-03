import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import snapshot_data_utils as snapshot_dutils
from . import monocular_dataset


class Dataset(monocular_dataset.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__(data_root, human, ann_file, split)

        self.data_root = data_root
        self.split = split

        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = snapshot_dutils.get_camera(camera_path)
        self.begin_ith_frame = cfg.begin_ith_frame
        self.num_train_frame = cfg.num_train_frame

        self.ims = np.arange(self.num_train_frame)
        self.num_cams = 1

        params_path = ann_file
        self.params = np.load(params_path, allow_pickle=True).item()

        self.nrays = cfg.N_rand

    def prepare_inside_pts(self, pts, msk, K, R, T):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)
        RT = np.concatenate([R, T], axis=1)
        pts2d = base_utils.project(pts3d, K, RT)

        H, W = msk.shape
        pts2d = np.round(pts2d).astype(np.int32)
        pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
        inside = msk[pts2d[:, 1], pts2d[:, 0]]
        inside = inside.reshape(*sh[:-1])

        return inside

    def __getitem__(self, index):
        latent_index = index
        index = index + self.begin_ith_frame
        frame_index = index

        img_path = os.path.join(self.data_root, 'image',
                                '{}.jpg'.format(index))
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk_path = os.path.join(self.data_root, 'mask', '{}.png'.format(index))
        msk = imageio.imread(msk_path)

        K = self.cam['K']
        D = self.cam['D']
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = self.cam['R']
        T = self.cam['T'][:, None]

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            index)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        img[msk == 0] = 0
        K = K.copy()
        K[:2] = K[:2] * cfg.ratio

        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, msk, K, R, T)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'pts': pts,
            'inside': inside
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_train_frame
