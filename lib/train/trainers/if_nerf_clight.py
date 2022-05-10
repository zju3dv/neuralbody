import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
from lib.train import make_optimizer
import lpips
import cv2
import numpy as np


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.lpips = lpips.LPIPS(net='vgg')
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']

        rgb_map = ret['rgb_map'][mask] # (G * 32 * 32, 3)
        rgb_gt = batch['rgb'][mask] # (G * 32 * 32, 3)

        img_mse = self.img2mse(rgb_map, rgb_gt)

        ########################################## LPIPS PREP ##########################################

        # Normalise to [-1, 1]
        rgb_map = (rgb_map[..., [2, 1, 0]] * 2) - 1
        rgb_gt = (rgb_gt[..., [2, 1, 0]] * 2) - 1

        # The tensor needs to be of size (G, 3, H, W) for LPIPS
        lpips_map = rgb_map.view(2, 32, 32, 3).permute(0, 3, 1, 2)
        lpips_gt = rgb_gt.view(2, 32, 32, 3).permute(0, 3, 1, 2)

        # compute lpips
        img_lpips = self.lpips.forward(lpips_map, lpips_gt)

        ########################################## LPIPS PREP ##########################################

        scalar_stats.update({'img_loss': img_mse})
        loss += (0.2 * img_mse + img_lpips)

        if 'rgb0' in ret:
            print("yess")
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
