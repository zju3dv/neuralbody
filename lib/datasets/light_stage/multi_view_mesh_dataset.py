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
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        i = 0
        i = i + cfg.begin_ith_frame
        ni = cfg.num_train_frame
        if cfg.num_render_frame > 0:
            ni = cfg.num_render_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + ni]
        ])
        self.num_cams = 1

        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(
            np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(
            np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(
            np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.ni = ni

    def prepare_input(self, i):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1

        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if self.human in ['CoreView_362']:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
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

    def get_mask(self, i, nv):
        im = self.ims[i, nv]

        msk_path = os.path.join(self.data_root, 'mask_cihp', im)[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk = (msk_cihp != 0).astype(np.uint8)

        msk = cv2.undistort(msk, self.Ks[nv], self.Ds[nv])

        border = 5
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
        latent_index = index
        frame_index = index + cfg.begin_ith_frame

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            frame_index)

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
            'coord': coord,
            'out_sh': out_sh,
            'pts': pts,
            'inside': inside
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'wbounds': can_bounds,
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni
