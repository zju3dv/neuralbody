import torch
from lib.config import cfg
from .nerf_net_utils import *
import numpy as np
import mcubes
import trimesh


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
        pts = ray_batch
        if net_c is None:
            alpha = self.net(pts)
        else:
            alpha = self.net(pts, net_c)

        if cfg.N_importance > 0:
            alpha_0 = alpha
            if net_c is None:
                alpha = self.net(pts, model='fine')
            else:
                alpha = self.net(pts, net_c, model='fine')

        ret = {
            'alpha': alpha
        }
        if cfg.N_importance > 0:
            ret['alpha0'] = alpha_0

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
        pts = batch['pts']
        sh = pts.shape

        inside = batch['inside'][0].bool()
        pts = pts[0][inside][None]

        pts = pts.view(sh[0], -1, 1, 3)

        ret = self.batchify_rays(pts, cfg.chunk)

        alpha = ret['alpha']
        alpha = alpha[0, :, 0, 0].detach().cpu().numpy()
        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
