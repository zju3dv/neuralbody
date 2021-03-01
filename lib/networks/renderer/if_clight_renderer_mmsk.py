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
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

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

    # def batchify_rays(self, rays_flat, chunk=1024 * 32, net_c=None):
    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      chunk=1024 * 32,
                      net_c=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []
        for i in range(0, grid_coords.shape[1], chunk):
            # ret = self.render_rays(rays_flat[i:i + chunk], net_c)
            ret = self.net(sp_input, grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk], light_pts[:, i:i + chunk])
            # for k in ret:
            #     if k not in all_ret:
            #         all_ret[k] = []
            #     all_ret[k].append(ret[k])
            all_ret.append(ret)
        # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        all_ret = torch.cat(all_ret, 1)
        return all_ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        inside = self.prepare_inside_pts(pts, batch)

        light_pts = pts.clone()
        pts = self.pts_to_can_pts(pts, batch)

        ray_d0 = batch['ray_d']
        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)
        viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()

        # reshape to [batch_size, n, 3]
        pts = pts.view(sh[0], -1, 3)[inside][None]
        viewdir = viewdir.view(sh[0], -1, 3)[inside][None]
        light_pts = light_pts.view(sh[0], -1, 3)[inside][None]

        # light intensity varies with 3D location
        light_pts = embedder.xyz_embedder(light_pts)
        viewdir = embedder.view_embedder(viewdir)

        sp_input = self.prepare_sp_input(batch)
        grid_coords = self.get_grid_coords(pts, sp_input, batch)

        if ray_o.size(1) <= 2048:
            raw = self.net(sp_input, grid_coords, viewdir, light_pts)
        else:
            raw = self.batchify_rays(sp_input, grid_coords, viewdir, light_pts,
                                     1024 * 32, None)

        # construct full raw
        full_raw = torch.zeros([sh[0], sh[1] * z_vals.size(2), 4]).to(raw)
        full_raw[inside] = raw[0]

        # reshape to [num_rays, num_samples along ray, 4]
        raw = full_raw.reshape(-1, z_vals.size(2), 4)
        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)
        rgb_map = rgb_map.view(*sh[:-1], -1)
        acc_map = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])

        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        return ret
