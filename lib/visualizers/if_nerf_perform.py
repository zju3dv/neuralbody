import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        data_dir = 'data/perform/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir),
                      'yellow'))

    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        if cfg.white_bkgd:
            img_pred = img_pred + 1
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        frame_root = 'data/perform/{}/{}'.format(cfg.exp_name, 0)
        os.system('mkdir -p {}'.format(frame_root))
        frame_index = batch['frame_index'].item()
        view_index = batch['view_index'].item()
        cv2.imwrite(
            os.path.join(
                frame_root,
                'frame{:04d}_view{:04d}.png'.format(frame_index, view_index)),
            img_pred * 255)
