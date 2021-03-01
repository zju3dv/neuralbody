import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os


class Visualizer:
    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        frame_root = 'data/perform/{}/{}'.format(cfg.exp_name,
                                                 0)
        os.system('mkdir -p {}'.format(frame_root))
        i = batch['index'].item()
        cv2.imwrite(os.path.join(frame_root, '%d.jpg' % i), img_pred * 255)
