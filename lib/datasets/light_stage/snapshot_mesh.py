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
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = snapshot_dutils.get_camera(camera_path)
        self.begin_i = cfg.begin_i
        self.ni = cfg.ni

        params_path = os.path.join(data_root, 'params.npy')
        self.params = np.load(params_path, allow_pickle=True).item()

        self.nrays = cfg.N_rand

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        ply_path = os.path.join(self.data_root, 'smpl', '{}.ply'.format(i))
        xyz, nxyz = if_nerf_dutils.get_smpl_data(ply_path)

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

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] -= 0.1
        max_xyz[1] += 0.1
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
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

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans

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
        index = index + self.begin_i
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

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans = self.prepare_input(
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
            'feature': feature,
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
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': index + 1,
            'index': 0
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni
