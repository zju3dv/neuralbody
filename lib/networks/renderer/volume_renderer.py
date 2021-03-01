import torch
from lib.config import cfg
from .nerf_net_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def render_rays(self, ray_batch, net_c=None, pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          network_fn: function. Model for predicting RGB and density at each point
            in space.
          network_query_fn: function used for passing queries to network_fn.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          network_fine: "fine" network with same spec as network_fn.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:,
                                                      3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        if not cfg.lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, cfg.N_samples])

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
            ..., :, None]  # [N_rays, N_samples, 3]

        if net_c is None:
            raw = self.net(pts, viewdirs)
        else:
            raw = self.net(pts, viewdirs, net_c)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, cfg.raw_noise_std, cfg.white_bkgd)

        if cfg.N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid,
                                   weights[..., 1:-1],
                                   cfg.N_importance,
                                   det=(cfg.perturb == 0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[
                ..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # raw = run_network(pts, fn=run_fn)
            if net_c is None:
                raw = self.net(pts, viewdirs, model='fine')
            else:
                raw = self.net(pts, viewdirs, net_c, model='fine')

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, z_vals, rays_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map,
            'disp_map': disp_map,
            'acc_map': acc_map,
            'depth_map': depth_map
        }
        ret['raw'] = raw
        if cfg.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1,
                                     unbiased=False)  # [N_rays]

        for k in ret:
            DEBUG = False
            if (torch.isnan(ret[k]).any()
                    or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def batchify_rays(self, rays_flat, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render(self, batch):
        rays_o = batch['ray_o']
        rays_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        sh = rays_o.shape
        rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
        near, far = near.transpose(0, 1), far.transpose(0, 1)
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], dim=-1)
        ret = self.batchify_rays(rays, cfg.chunk)
        ret = {k: v.view(*sh[:-1], -1) for k, v in ret.items()}
        ret['depth_map'] = ret['depth_map'].view(*sh[:-1])
        return ret
