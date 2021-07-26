import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from . import if_clight_renderer


class Renderer(if_clight_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def batchify_rays(self, wpts, alpha_decoder, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_batch, n_point = wpts.shape[:2]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = alpha_decoder(wpts[:, i:i + chunk])
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 1)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        inside = batch['inside'][0].bool()
        pts = pts[0][inside][None]

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)
        alpha_decoder = lambda x: self.net.calculate_density(
            x, feature_volume, sp_input)

        alpha = self.batchify_rays(pts, alpha_decoder, 2048 * 64)

        alpha = alpha[0, :, 0].detach().cpu().numpy()
        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)

        # vertices = (vertices - 10) * 0.005
        # vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
