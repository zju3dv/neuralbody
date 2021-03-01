import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh


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

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    # def batchify_rays(self, rays_flat, chunk=1024 * 32, net_c=None):
    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      chunk=1024 * 32,
                      net_c=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []
        for i in range(0, grid_coords.shape[1], chunk):
            # ret = self.render_rays(rays_flat[i:i + chunk], net_c)
            ret = self.net(sp_input, grid_coords[:, i:i + chunk])
            # for k in ret:
            #     if k not in all_ret:
            #         all_ret[k] = []
            #     all_ret[k].append(ret[k])
            all_ret.append(ret)
        # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        all_ret = torch.cat(all_ret, 1)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        inside = batch['inside'][0].bool()
        pts = pts[0][inside][None]

        pts = pts.view(sh[0], -1, 1, 3)
        pts = self.pts_to_can_pts(pts, batch)

        sp_input = self.prepare_sp_input(batch)

        grid_coords = self.get_grid_coords(pts, sp_input, batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)

        if grid_coords.size(1) < 1024 * 32:
            alpha = self.net(sp_input, grid_coords)
        else:
            alpha = self.batchify_rays(sp_input, grid_coords, 1024 * 32, None)

        alpha = alpha[0, :, 0].detach().cpu().numpy()
        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
