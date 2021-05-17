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
from lib.utils import render_utils
from lib.utils import snapshot_data_utils as snapshot_dutils


class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = snapshot_dutils.get_camera(camera_path)
        self.K = self.cam['K']
        self.RT = np.concatenate([self.cam['R'], self.cam['T'][:, None]],
                                 axis=1)
        self.ni = 60

        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask',
                                self.ims[index])[:-4] + '.png'
        msk = (imageio.imread(msk_path) > 0).astype(np.uint8)
        border = 3
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)
        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the npy file
        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        min_xyz = np.min(xyz, axis=0)
        min_xyz[1] -= 0.06
        max_xyz = np.max(xyz, axis=0)
        max_xyz[1] += 0.03
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # move the point cloud to the canonical frame, which eliminates the influence of translation
        cxyz = xyz.astype(np.float32) - min_xyz
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

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

        return feature, coord, out_sh, bounds

    def __getitem__(self, index):
        i = index + 15
        feature, coord, out_sh, bounds = self.prepare_input(i)

        # reduce the image resolution by ratio
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        K = self.K

        index = 0
        ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
            self.RT, K, bounds)

        ret = {
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        meta = {'bounds': bounds, 'i': i, 'index': index}
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni
