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


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = snapshot_dutils.get_camera(camera_path)
        self.num_train_frame = cfg.num_train_frame

        params_path = ann_file
        self.params = np.load(params_path, allow_pickle=True).item()

        self.nrays = cfg.N_rand

    def prepare_input(self, i):
        # read xyz, normal, color from the npy file
        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] -= 0.1
        max_xyz[1] += 0.1
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = self.params['pose'][i][:3]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = self.params['trans'][i].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] -= 0.1
        max_xyz[1] += 0.1
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return coord, out_sh, can_bounds, bounds, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'image',
                                '{}.jpg'.format(index))
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk_path = os.path.join(self.data_root, 'mask', '{}.png'.format(index))
        msk = imageio.imread(msk_path)

        frame_index = index
        latent_index = index

        K = self.cam['K']
        D = self.cam['D']
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = self.cam['R']
        T = self.cam['T'][:, None]
        RT = np.concatenate([R, T], axis=1).astype(np.float32)

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            frame_index)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K = K.copy().astype(np.float32)
        K[:2] = K[:2] * cfg.ratio

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'msk': msk
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': 0
        }
        ret.update(meta)

        Rh0 = self.params['pose'][index][:3]
        R0 = cv2.Rodrigues(Rh0)[0].astype(np.float32)
        Th0 = self.params['trans'][index].astype(np.float32)
        meta = {'R0_snap': R0, 'Th0_snap': Th0, 'K': K, 'RT': RT}
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_train_frame
