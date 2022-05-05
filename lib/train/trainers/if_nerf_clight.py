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

        # TODO : Should we change here the loss, i.e add a value lpips.
        # The function uses mse and smooth l1
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.lpips = lpips.LPIPS(net='vgg')
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])

        ########################################## LPIPS PREP ##########################################

        rgb_pred = ret['rgb_map'][0]
        rgb_gt = batch['rgb'][0]

        mask_at_box = batch['mask_at_box'][0]
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = torch.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = torch.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt

        #img_loss = np.mean((rgb_pred - rgb_gt)**2)

        rgb_pred = img_pred
        rgb_gt = img_gt
        
        # crop the object region
        mask_at_box_np = batch['mask_at_box'][0].detach().cpu().numpy()
        x, y, w, h = cv2.boundingRect(mask_at_box_np.astype(torch.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        
        img_pred = (img_pred[..., [2, 1, 0]] * 2) - 1
        img_gt = (img_gt[..., [2, 1, 0]] * 2) - 1

        # Transform (H, W, 3) in (1, 3, H, w)
        img_pred = img_pred.permute(2, 0, 1)[None, :, :, :]
        img_gt = img_gt.permute(2, 0, 1)[None, :, :, :]

        # compute lpips
        img_lpips = self.lpips.forward(img_pred, img_gt)

        ########################################## LPIPS PREP ##########################################

        scalar_stats.update({'img_loss': img_loss})
        loss += (0.2 * img_loss + img_lpips)

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
