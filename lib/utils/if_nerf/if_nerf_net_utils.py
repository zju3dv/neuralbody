import torch
import numpy as np
import os
from lib.config import cfg
import trimesh


def update_loss_img(output, batch):
    mse = torch.mean((output['rgb_map'] - batch['rgb'])**2, dim=2)[0]
    mse = mse.detach().cpu().numpy().astype(np.float32)

    # load the loss img
    img_path = batch['meta']['img_path'][0]
    paths = img_path.split('/')
    paths[-1] = os.path.basename(img_path).replace('.jpg', '.npy')
    loss_img_path = os.path.join(paths[0], 'loss', *paths[1:])
    if os.path.exists(loss_img_path):
        loss_img = np.load(loss_img_path)
    else:
        os.system("mkdir -p '{}'".format(os.path.dirname(loss_img_path)))
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        loss_img = mse.mean() * np.ones([H, W]).astype(np.float32)

    coord = batch['img_coord'][0]
    coord = coord.detach().cpu().numpy()
    loss_img[coord[:, 0], coord[:, 1]] = mse
    np.save(loss_img_path, loss_img)


def init_smpl(smpl):
    data_root = 'data/light_stage'
    smpl_dir = os.path.join(data_root, cfg.smpl, cfg.human)
    for i in range(cfg.ni):
        smpl_path = os.path.join(smpl_dir, '{}.ply'.format(i + 1))
        ply = trimesh.load(smpl_path)
        xyz = np.array(ply.vertices).ravel()
        smpl.weight.data[i] = torch.FloatTensor(xyz)
    return smpl


def pts_to_can_pts(pts, batch):
    """transform pts from the world coordinate to the smpl coordinate"""
    Th = batch['Th']
    pts = pts - Th
    R = batch['R']
    pts = torch.matmul(pts, batch['R'])
    return pts


def pts_to_coords(pts, min_xyz):
    pts = pts.clone().detach()
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]
    min_dhw = min_xyz[:, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
    return dhw
