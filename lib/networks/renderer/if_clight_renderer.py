import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder


class Renderer:
    def __init__(self, net):
        self.net = net

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']

        return sp_input

    def get_density_color(self, wpts, viewdir, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume,
                        sp_input, batch):
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color(
            x_point, viewdir_val, feature_volume, sp_input)

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, raw_decoder)

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

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               feature_volume, sp_input, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
