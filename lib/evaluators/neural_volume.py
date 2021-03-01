import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
import os
import cv2
import imageio


class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]
        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def evaluate(self, batch):
        if cfg.human in [302, 313, 315]:
            i = batch['i'].item() + 1
        else:
            i = batch['i'].item()
        i = i + cfg.begin_i
        cam_ind = batch['cam_ind'].item()

        # obtain the image path
        result_dir = 'data/result/neural_volumes/{}_nv'.format(cfg.human)
        frame_dir = os.path.join(result_dir, 'frame_{}'.format(i))
        gt_img_path = os.path.join(frame_dir, 'gt_{}.jpg'.format(cam_ind + 1))
        pred_img_path = os.path.join(frame_dir,
                                     'pred_{}.jpg'.format(cam_ind + 1))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        # gt_img_path = gt_img_path.replace('neural_volumes', 'gt')
        # os.system('mkdir -p {}'.format(os.path.dirname(gt_img_path)))
        # img_gt = img_gt[..., [2, 1, 0]] * 255
        # cv2.imwrite(gt_img_path, img_gt)

        img_pred = imageio.imread(pred_img_path).astype(np.float32) / 255.
        img_pred[mask_at_box != 1] = 0
        rgb_pred = img_pred[mask_at_box]

        # import matplotlib.pyplot as plt
        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_gt)
        # ax2.imshow(img_pred)
        # plt.show()
        # return

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

    def summarize(self):
        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(result_path, self.mse)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        self.mse = []
        self.psnr = []
        self.ssim = []
