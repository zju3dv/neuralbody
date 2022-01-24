**News**

* `05/17/2021` To make the comparison on ZJU-MoCap easier, we save quantitative and qualitative results of other methods at [here](https://github.com/zju3dv/neuralbody/blob/master/supplementary_material.md#results-of-other-methods-on-zju-mocap), including Neural Volumes, Multi-view Neural Human Rendering, and Deferred Neural Human Rendering.
* `05/13/2021` To make the following works easier compare with our model, we save our rendering results of ZJU-MoCap at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/Ea3VOUy204VAiVJ-V-OGd9YBxdhbtfpS-U6icD_rDq0mUQ?e=cAcylK) and write a [document](supplementary_material.md) that describes the training and test protocols.
* `05/12/2021` The code supports the test and visualization on unseen human poses.
* `05/12/2021` We update the ZJU-MoCap dataset with better fitted SMPL using [EasyMocap](https://github.com/zju3dv/EasyMocap). We also release a [website](https://zju3dv.github.io/zju_mocap/) for visualization. Please see [here](https://github.com/zju3dv/neuralbody#potential-problems-of-provided-smpl-parameters) for the usage of provided smpl parameters.

# Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans
### [Project Page](https://zju3dv.github.io/neuralbody) | [Video](https://www.youtube.com/watch?v=BPCAMeBCE-8) | [Paper](https://arxiv.org/pdf/2012.15838.pdf) | [Data](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset)

![monocular](https://zju3dv.github.io/neuralbody/images/monocular.gif)

> [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://arxiv.org/pdf/2012.15838.pdf)  
> Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> CVPR 2021

Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

### Installation using docker

Please see [docker/README.md](docker/README.md).

Thanks to [Zhaoyi Wan](https://github.com/wanzysky) for providing the docker implementation.

## Run the code on the custom dataset

Please see [CUSTOM](tools/custom).

## Run the code on People-Snapshot

Please see [INSTALL.md](INSTALL.md) to download the dataset.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Enn43YWDHwBEg-XBqnetFYcBLr3cItZ0qUFU-oKUpDHKXw?e=FObjE9).

### Process People-Snapshot

We already provide some processed data. If you want to process more videos of People-Snapshot, you could use [tools/process_snapshot.py](tools/process_snapshot.py).

You can also visualize smpl parameters of People-Snapshot with [tools/vis_snapshot.py](tools/vis_snapshot.py).

### Visualization on People-Snapshot

Take the visualization on `female-3-casual` as an example. The command lines for visualization are recorded in [visualize.sh](visualize.sh).

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/female3c/latest.pth`.
2. Visualization:
    * Visualize novel views of single frame
    ```
    python run.py --type visualize --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c vis_novel_view True num_render_views 144
    ```

    ![monocular](https://zju3dv.github.io/neuralbody/images/monocular_render.gif)

    * Visualize views of dynamic humans with fixed camera
    ```
    python run.py --type visualize --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c vis_novel_pose True
    ```

    ![monocular](https://zju3dv.github.io/neuralbody/images/monocular_perform.gif)

    * Visualize mesh
    ```
    # generate meshes
    python run.py --type visualize --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c vis_mesh True train.num_workers 0
    # visualize a specific mesh
    python tools/render_mesh.py --exp_name female3c --dataset people_snapshot --mesh_ind 226
    ```

    ![monocular](https://zju3dv.github.io/neuralbody/images/monocular_mesh.gif)

3. The results of visualization are located at `$ROOT/data/render/female3c` and `$ROOT/data/perform/female3c`.

### Training on People-Snapshot

Take the training on `female-3-casual` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:
    ```
    # training
    python train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c resume False
    # distributed training
    python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c resume False gpus "0, 1, 2, 3" distributed True
    ```
2. Train with white background:
    ```
    # training
    python train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c resume False white_bkgd True
    ```
3. Tensorboard:
    ```
    tensorboard --logdir data/record/if_nerf
    ```

## Run the code on ZJU-MoCap

Please see [INSTALL.md](INSTALL.md) to download the dataset.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Enn43YWDHwBEg-XBqnetFYcBLr3cItZ0qUFU-oKUpDHKXw?e=FObjE9).

### Potential problems of provided smpl parameters

1. The newly fitted parameters locate in `new_params`. Currently, the released pretrained models are trained on previously fitted parameters, which locate in `params`.
2. The smpl parameters of ZJU-MoCap have different definition from the one of MPI's smplx.
    * If you want to extract vertices from the provided smpl parameters, please use `zju_smpl/extract_vertices.py`.
    * The reason that we use the current definition is described at [here](https://github.com/zju3dv/EasyMocap/blob/master/doc/02_output.md#attention-for-smplsmpl-x-users).

It is okay to train Neural Body with smpl parameters fitted by smplx.

### Test on ZJU-MoCap

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `sequence 313` as an example.

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/xyzc_313/latest.pth`.
2. Test on training human poses:
    ```
    python run.py --type evaluate --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313
    ```
3. Test on unseen human poses:
    ```
    python run.py --type evaluate --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 test_novel_pose True
    ```

### Visualization on ZJU-MoCap

Take the visualization on `sequence 313` as an example. The command lines for visualization are recorded in [visualize.sh](visualize.sh).

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/xyzc_313/latest.pth`.
2. Visualization:
    * Visualize novel views of single frame
    ```
    python run.py --type visualize --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 vis_novel_view True
    ```
    ![zju_mocap](https://zju3dv.github.io/neuralbody/images/zju_mocap_render_313.gif)

    * Visualize novel views of single frame by rotating the SMPL model
    ```
    python run.py --type visualize --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 vis_novel_view True num_render_views 100
    ```
    ![zju_mocap](https://zju3dv.github.io/neuralbody/images/rotate_smpl.gif)

    * Visualize views of dynamic humans with fixed camera
    ```
    python run.py --type visualize --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 vis_novel_pose True num_render_frame 1000 num_render_views 1
    ```
    ![zju_mocap](https://zju3dv.github.io/neuralbody/images/zju_mocap_perform_fixed_313.gif) 

    * Visualize views of dynamic humans with rotated camera
    ```
    python run.py --type visualize --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 vis_novel_pose True num_render_frame 1000
    ```
    ![zju_mocap](https://zju3dv.github.io/neuralbody/images/zju_mocap_perform_313.gif)

    * Visualize mesh
    ```
    # generate meshes
    python run.py --type visualize --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 vis_mesh True train.num_workers 0
    # visualize a specific mesh
    python tools/render_mesh.py --exp_name xyzc_313 --dataset zju_mocap --mesh_ind 0
    ```
    ![zju_mocap](https://zju3dv.github.io/neuralbody/images/zju_mocap_mesh.gif)

4. The results of visualization are located at `$ROOT/data/render/xyzc_313` and `$ROOT/data/perform/xyzc_313`.

### Training on ZJU-MoCap

Take the training on `sequence 313` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:
    ```
    # training
    python train_net.py --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 resume False
    # distributed training
    python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 resume False gpus "0, 1, 2, 3" distributed True
    ```
2. Train with white background:
    ```
    # training
    python train_net.py --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 resume False white_bkgd True
    ```
3. Tensorboard:
    ```
    tensorboard --logdir data/record/if_nerf
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2021neural,
  title={Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans},
  author={Peng, Sida and Zhang, Yuanqing and Xu, Yinghao and Wang, Qianqian and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2021}
}
```
