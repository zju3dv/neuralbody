import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2


class Visualizer:
    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        result_dir = os.path.join('data/result/if-nerf', cfg.exp_name)

        if cfg.human in [302, 313, 315]:
            i = batch['i'].item() + 1
        else:
            i = batch['i'].item()
        i = i + cfg.begin_i
        cam_ind = batch['cam_ind'].item()
        frame_dir = os.path.join(result_dir, 'frame_{}'.format(i))
        pred_img_path = os.path.join(frame_dir,
                                     'pred_{}.jpg'.format(cam_ind + 1))

        os.system('mkdir -p {}'.format(os.path.dirname(pred_img_path)))
        img_pred = (img_pred * 255)[..., [2, 1, 0]]
        cv2.imwrite(pred_img_path, img_pred)

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()
