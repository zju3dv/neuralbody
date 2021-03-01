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


class Dataset(data.Dataset):
    def __init__(self, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        ann_file = os.path.join(data_root, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        i = 0
        i = i + cfg.begin_i
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + cfg.ni]
        ])

        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(
            np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(
            np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(
            np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.ni = cfg.ni

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp.astype(np.uint8)
        # border = 3
        # kernel = np.ones((border, border), np.uint8)
        # msk = cv2.dilate(msk.copy(), kernel)
        return msk

    def prepare_input(self, i):
        i = i + cfg.begin_i

        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
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

    def get_mask(self, i, nv):
        im = self.ims[i, nv]

        msk_path = os.path.join(self.data_root, 'mask_cihp', im)[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp.astype(np.uint8)
        msk = cv2.resize(msk, (1000, 1002), interpolation=cv2.INTER_NEAREST)

        msk = cv2.undistort(msk, self.Ks[nv], self.Ds[nv])

        border = 7
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)

        return msk

    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        for nv in range(self.ims.shape[1]):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[nv], self.Ts[nv]], axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[nv], RT)

            msk = self.get_mask(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def __getitem__(self, index):
        i = index

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans = self.prepare_input(
            i)

        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, i)

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
            'i': i
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni
