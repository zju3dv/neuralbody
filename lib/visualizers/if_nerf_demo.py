import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import cv2
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        data_dir = 'data/render/{}'.format(cfg.exp_name)
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

        depth_pred = np.zeros((H, W))
        depth_pred[mask_at_box] = output['depth_map'][0].detach().cpu().numpy()

        img_root = 'data/render/{}/frame_{:04d}'.format(
            cfg.exp_name, batch['frame_index'].item())
        os.system('mkdir -p {}'.format(img_root))
        index = batch['view_index'].item()

        # plt.imshow(depth_pred)
        # depth_dir = os.path.join(img_root, 'depth')
        # os.system('mkdir -p {}'.format(depth_dir))
        # plt.savefig(os.path.join(depth_dir, '{}.jpg'.format(index)))
        # plt.close()

        # mask_pred = np.zeros((H, W, 3))
        # mask_pred[acc_pred > 0.5] = 255

        # acc_dir = os.path.join(img_root, 'mask')
        # os.system('mkdir -p {}'.format(acc_dir))
        # mask = cv2.resize(mask_pred, (H * 8, W * 8), interpolation=cv2.INTER_NEAREST)
        # mask_path = os.path.join(acc_dir, 'img_{:04d}.jpg'.format(index))
        # cv2.imwrite(mask_path, mask)

        cv2.imwrite(os.path.join(img_root, '{:04d}.png'.format(index)),
                    img_pred * 255)
