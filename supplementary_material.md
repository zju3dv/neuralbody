# Supplementary Material

## Training and test data

We provide a [website](https://zju3dv.github.io/zju_mocap/) for visualization. 

The multi-view videos are captured by 23 cameras. We train our model on the "0, 6, 12, 18" cameras and test it on the remaining cameras.

The following table shows the detailed frame numbers for training and test of each video. Since the video length of each subject is different, we choose the appropriate number of frames for training and test. 

**Note that since rendering is very slow, we test our model every 30 frames. For example, although the frame range of video 313 is "0-59", we only test our model on the 0-th and 30-th frames.**

| Video   |  313  |  315  |  377  |  386  |  387  |  390  |  392  |  393  |  394  | 
| :-----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Number of frames  | 1470  | 2185  | 617   |  646  | 654   | 1171  | 556   | 658   | 859   |
| Frame Range (Training) | 0-59    |  0-399  |  0-299  |  0-299  |  0-299  |  700-999  |  0-299  |  0-299  |  0-299  |
| Frame Range (Unseen human poses)  | 60-1060    |  400-1400  |  300-617  |  300-646  |  300-654  |  0-700  |  300-556  |  300-658  |  300-859  |

## Evaluation metrics

**We save our rendering results on novel views of training frames and unseen human poses at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/Ea3VOUy204VAiVJ-V-OGd9YBxdhbtfpS-U6icD_rDq0mUQ?e=cAcylK).**

As described in the paper, we evaluate our model in terms of the PSNR and SSIM metrics.

A straightforward way for evaluation is calculating the metrics on the whole image. Since we already know the 3D bounding box of the target human, we can project the 3D box to obtain a `bound_mask` and make the colors of pixels outside the mask as zero, as shown in the following figure.

![fig](https://zju3dv.github.io/neuralbody/images/bound_mask.png)

As a result, the PSNR and SSIM metrics appear very high performances, as shown in the following table.

<table style="text-align: center">
   <tr>
      <td></td>
      <td colspan="2">Training frames</td>
      <td colspan="2">Unseen human poses</td>
   </tr>
   <tr>
      <td></td>
      <td>PSNR</td>
      <td>SSIM</td>
      <td>PSNR</td>
      <td>SSIM</td>
   </tr>
   <tr>
      <td>313</td>
      <td>35.21 </td>
      <td>0.985 </td>
      <td>29.02 </td>
      <td>0.964 </td>
   </tr>
   <tr>
      <td>315</td>
      <td>33.07 </td>
      <td>0.988 </td>
      <td>25.70 </td>
      <td>0.957 </td>
   </tr>
   <tr>
      <td>392</td>
      <td>35.76 </td>
      <td>0.984 </td>
      <td>31.53 </td>
      <td>0.971 </td>
   </tr>
   <tr>
      <td>393</td>
      <td>33.24 </td>
      <td>0.979 </td>
      <td>28.40 </td>
      <td>0.960 </td>
   </tr>
   <tr>
      <td>394</td>
      <td>34.31 </td>
      <td>0.980 </td>
      <td>29.61 </td>
      <td>0.961 </td>
   </tr>
   <tr>
      <td>377</td>
      <td>33.86 </td>
      <td>0.985 </td>
      <td>30.60 </td>
      <td>0.977 </td>
   </tr>
   <tr>
      <td>386</td>
      <td>36.07 </td>
      <td>0.984 </td>
      <td>33.05 </td>
      <td>0.974 </td>
   </tr>
   <tr>
      <td>390</td>
      <td>34.48 </td>
      <td>0.980 </td>
      <td>30.25 </td>
      <td>0.964 </td>
   </tr>
   <tr>
      <td>387</td>
      <td>31.39 </td>
      <td>0.975 </td>
      <td>27.68 </td>
      <td>0.961 </td>
   </tr>
   <tr>
      <td></td>
      <td>34.15 </td>
      <td>0.982 </td>
      <td>29.54 </td>
      <td>0.966 </td>
   </tr>
</table>

To overcome this problem, a solution is only calculating the metrics on pixels inside the `bound_mask`. Since the SSIM metric requires the input to have the image format, we first compute the 2D box that bounds the `bound_mask` and then crop the corresponding image region. 

```python
def ssim_metric(rgb_pred, rgb_gt, batch):
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
```


The following table lists corresponding results.

<table style="text-align: center">
   <tr>
      <td></td>
      <td colspan="2">Training frames</td>
      <td colspan="2">Unseen human poses</td>
   </tr>
   <tr>
      <td></td>
      <td>PSNR</td>
      <td>SSIM</td>
      <td>PSNR</td>
      <td>SSIM</td>
   </tr>
   <tr>
      <td>313</td>
      <td>30.56 </td>
      <td>0.971 </td>
      <td>23.95 </td>
      <td>0.905 </td>
   </tr>
   <tr>
      <td>315</td>
      <td>27.24 </td>
      <td>0.962 </td>
      <td>19.56 </td>
      <td>0.852 </td>
   </tr>
   <tr>
      <td>392</td>
      <td>29.44 </td>
      <td>0.946 </td>
      <td>25.76 </td>
      <td>0.909 </td>
   </tr>
   <tr>
      <td>394</td>
      <td>28.44 </td>
      <td>0.940 </td>
      <td>23.80 </td>
      <td>0.878 </td>
   </tr>
   <tr>
      <td>393</td>
      <td>27.58 </td>
      <td>0.939 </td>
      <td>23.25 </td>
      <td>0.893 </td>
   </tr>
   <tr>
      <td>377</td>
      <td>27.64 </td>
      <td>0.951 </td>
      <td>23.91 </td>
      <td>0.909 </td>
   </tr>
   <tr>
      <td>386</td>
      <td>28.60 </td>
      <td>0.931 </td>
      <td>25.68 </td>
      <td>0.881 </td>
   </tr>
   <tr>
      <td>387</td>
      <td>25.79 </td>
      <td>0.928 </td>
      <td>21.60 </td>
      <td>0.870 </td>
   </tr>
   <tr>
      <td>390</td>
      <td>27.59 </td>
      <td>0.926 </td>
      <td>23.90 </td>
      <td>0.870 </td>
   </tr>
   <tr>
      <td></td>
      <td>28.10 </td>
      <td>0.944 </td>
      <td>23.49 </td>
      <td>0.885 </td>
   </tr>
</table>

## Results of other methods on ZJU-MoCap

We save rendering results of other methods on novel views of training frames and unseen human poses at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EQaPRQww70NDqEXeSG-fOeAB5JXFSWiWDW223h5nmkHvwQ?e=mdofbl), including Neural Volumes, Multi-view Neural Human Rendering, and Deferred Neural Human Rendering. **Note that we only generate novel views of training frames for Neural Volumes.**

The following table lists quantitative results of Neural Volumes.

<table style="text-align: center">
    <tr>
        <td></td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>313</td>
        <td>20.09 </td>
        <td>0.831 </td>
    </tr>
    <tr>
        <td>315</td>
        <td>18.57 </td>
        <td>0.824 </td>
    </tr>
    <tr>
        <td>392</td>
        <td>22.88 </td>
        <td>0.726 </td>
    </tr>
    <tr>
        <td>394</td>
        <td>22.08 </td>
        <td>0.843 </td>
    </tr>
    <tr>
        <td>393</td>
        <td>21.29 </td>
        <td>0.842 </td>
    </tr>
    <tr>
        <td>377</td>
        <td>21.15 </td>
        <td>0.842 </td>
    </tr>
    <tr>
        <td>386</td>
        <td>23.21 </td>
        <td>0.820 </td>
    </tr>
    <tr>
        <td>387</td>
        <td>20.74 </td>
        <td>0.838 </td>
    </tr>
    <tr>
        <td>390</td>
        <td>22.49 </td>
        <td>0.825 </td>
    </tr>
    <tr>
        <td></td>
        <td>21.39 </td>
        <td>0.821 </td>
    </tr>
</table>

The following table lists quantitative results of Multi-view Neural Human Rendering.

<table style="text-align: center">
    <tr>
        <td></td>
        <td colspan="2">Training frames</td>
        <td colspan="2">Unseen human poses</td>
    </tr>
    <tr>
        <td></td>
        <td>PSNR</td>
        <td>SSIM</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>313</td>
        <td>26.68 </td>
        <td>0.935 </td>
        <td>23.05 </td>
        <td>0.893 </td>
    </tr>
    <tr>
        <td>315</td>
        <td>19.81 </td>
        <td>0.874 </td>
        <td>18.88 </td>
        <td>0.844 </td>
    </tr>
    <tr>
        <td>392</td>
        <td>24.73 </td>
        <td>0.902 </td>
        <td>23.66 </td>
        <td>0.893 </td>
    </tr>
    <tr>
        <td>394</td>
        <td>25.01 </td>
        <td>0.906 </td>
        <td>22.87 </td>
        <td>0.874 </td>
    </tr>
    <tr>
        <td>393</td>
        <td>23.47 </td>
        <td>0.894 </td>
        <td>22.27 </td>
        <td>0.885 </td>
    </tr>
    <tr>
        <td>377</td>
        <td>23.79 </td>
        <td>0.918 </td>
        <td>21.94 </td>
        <td>0.885 </td>
    </tr>
    <tr>
        <td>386</td>
        <td>25.02 </td>
        <td>0.879 </td>
        <td>23.70 </td>
        <td>0.853 </td>
    </tr>
    <tr>
        <td>387</td>
        <td>22.65 </td>
        <td>0.858 </td>
        <td>20.97 </td>
        <td>0.866 </td>
    </tr>
    <tr>
        <td>390</td>
        <td>23.72 </td>
        <td>0.873 </td>
        <td>22.65 </td>
        <td>0.858 </td>
    </tr>
    <tr>
        <td></td>
        <td>23.87 </td>
        <td>0.893 </td>
        <td>22.22 </td>
        <td>0.872 </td>
    </tr>
</table>

The following table lists quantitative results of Deferred Neural Human Rendering.

<table style="text-align: center">
    <tr>
        <td></td>
        <td colspan="2">Training frames</td>
        <td colspan="2">Unseen human poses</td>
    </tr>
    <tr>
        <td></td>
        <td>PSNR</td>
        <td>SSIM</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>313</td>
        <td>25.78 </td>
        <td>0.929 </td>
        <td>22.56 </td>
        <td>0.889 </td>
    </tr>
    <tr>
        <td>315</td>
        <td>19.44 </td>
        <td>0.869 </td>
        <td>18.38 </td>
        <td>0.841 </td>
    </tr>
    <tr>
        <td>392</td>
        <td>24.96 </td>
        <td>0.905 </td>
        <td>24.08 </td>
        <td>0.900 </td>
    </tr>
    <tr>
        <td>394</td>
        <td>24.84 </td>
        <td>0.903 </td>
        <td>22.67 </td>
        <td>0.871 </td>
    </tr>
    <tr>
        <td>393</td>
        <td>23.50 </td>
        <td>0.896 </td>
        <td>22.45 </td>
        <td>0.888 </td>
    </tr>
    <tr>
        <td>377</td>
        <td>23.74 </td>
        <td>0.917 </td>
        <td>22.07 </td>
        <td>0.886 </td>
    </tr>
    <tr>
        <td>386</td>
        <td>24.93 </td>
        <td>0.877 </td>
        <td>23.70 </td>
        <td>0.851 </td>
    </tr>
    <tr>
        <td>387</td>
        <td>22.44 </td>
        <td>0.888 </td>
        <td>20.64 </td>
        <td>0.862 </td>
    </tr>
    <tr>
        <td>390</td>
        <td>24.33 </td>
        <td>0.881 </td>
        <td>22.90 </td>
        <td>0.864 </td>
    </tr>
    <tr>
        <td></td>
        <td>23.77 </td>
        <td>0.896 </td>
        <td>22.16 </td>
        <td>0.872 </td>
    </tr>
</table>
