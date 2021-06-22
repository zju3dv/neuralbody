import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from . import if_clight_renderer_mmsk


class Renderer(if_clight_renderer_mmsk.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def prepare_inside_pts(self, wpts, batch):
        if 'R0_snap' not in batch:
            __import__('ipdb').set_trace()
            return raw

        # transform points from the world space to the smpl space
        Th = batch['Th']
        can_pts = wpts - Th[:, None, None]
        R = batch['R']
        can_pts = torch.matmul(can_pts, R)

        R0 = batch['R0_snap']
        Th0 = batch['Th0_snap']

        # transform pts from smpl coordinate to the world coordinate
        sh = can_pts.shape
        can_pts = can_pts.view(sh[0], -1, sh[3])
        pts = torch.matmul(can_pts, R0.transpose(2, 1)) + Th0[:, None]

        # project pts to image space
        R = batch['RT'][..., :3]
        T = batch['RT'][..., 3]
        pts = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
        pts = torch.matmul(pts, batch['K'].transpose(2, 1))
        pts2d = pts[..., :2] / pts[..., 2:]

        # ensure that pts2d is inside the image
        pts2d = pts2d.round().long()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
        pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)

        # remove the points outside the mask
        pts2d = pts2d[0]
        msk = batch['msk'][0]
        inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool()

        return inside
