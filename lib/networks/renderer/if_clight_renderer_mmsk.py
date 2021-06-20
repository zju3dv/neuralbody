import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from . import if_clight_renderer


class Renderer(if_clight_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def prepare_inside_pts(self, pts, batch):
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        insides = []
        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]
            T = batch['RT'][:, nv, :3, 3]
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)

            # remove the points outside the mask
            pts2d = pts2d[0]
            msk = batch['msks'][0, nv]
            inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool()
            insides.append(inside)

        inside = insides[0]
        for i in range(1, len(insides)):
            inside = inside * insides[i]

        return inside

    def get_density_color(self, wpts, viewdir, inside, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        wpts = wpts[inside][None]
        viewdir = viewdir[inside][None]
        full_raw = torch.zeros([n_batch, n_pixel * n_sample, 4]).to(wpts)
        if inside.sum() == 0:
            return full_raw

        raw = raw_decoder(wpts, viewdir)
        full_raw[inside] = raw[0]

        return full_raw

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume,
                        sp_input, batch):
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        inside = self.prepare_inside_pts(wpts, batch)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color(
            x_point, viewdir_val, feature_volume, sp_input)

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, inside, raw_decoder)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw = wpts_raw.reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret
