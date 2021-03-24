# Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans
### [Project Page](https://zju3dv.github.io/neuralbody) | [Video](https://www.youtube.com/watch?v=BPCAMeBCE-8) | [Paper](https://arxiv.org/pdf/2012.15838.pdf) | [Data](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Eo9zn4x_xcZKmYHZNjzel7gBdWf_d4m-pISHhPWB-GZBYw?e=Hf4mz7)

![monocular](https://zju3dv.github.io/neuralbody/images/monocular.gif)

> [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://arxiv.org/pdf/2012.15838.pdf)  
> Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> CVPR 2021

Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md).

## Run the code on People-Snapshot

Please see [INSTALL.md](INSTALL.md) to download the dataset.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Enn43YWDHwBEg-XBqnetFYcBLr3cItZ0qUFU-oKUpDHKXw?e=FObjE9).

### Visualization on People-Snapshot

Take the visualization on `female-3-casual` as an example. The command lines for visualization are recorded in [visualize.sh](visualize.sh).

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/female3c/latest.pth`.
2. Visualization:
    ```
    # visualize novel views of single frame
    python run.py --type visualize --cfg_file configs/snapshot_f3c_demo.yaml exp_name female3c
    # visualize views of dynamic humans
    python run.py --type visualize --cfg_file configs/snapshot_f3c_perform.yaml exp_name female3c
    # visualize mesh
    python run.py --type visualize --cfg_file configs/snapshot_f3c_mesh.yaml exp_name female3c train.num_workers 0
    ```
    3. The results of visualization are located at `$ROOT/data/render/female3c` and `$ROOT/data/perform/female3c`.

### Training on People-Snapshot

Take the training on `female-3-casual` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:
    ```
    # training
    python train_net.py --cfg_file configs/snapshot_f3c.yaml exp_name female3c resume False
    # distributed training
    python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/snapshot_f3c.yaml exp_name female3c resume False gpus "0, 1, 2, 3" distributed True
    ```
2. Tensorboard:
    ```
    tensorboard --logdir data/record/if_nerf
    ```

## Run the code on ZJU-Mocap

Please see [INSTALL.md](INSTALL.md) to download the dataset.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Enn43YWDHwBEg-XBqnetFYcBLr3cItZ0qUFU-oKUpDHKXw?e=FObjE9).

### Test on ZJU-Mocap

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `sequence 313` as an example.

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/xyzc_313/latest.pth`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313
    ```

### Visualization on ZJU-Mocap

Take the visualization on `sequence 313` as an example. The command lines for visualization are recorded in [visualize.sh](visualize.sh).

1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/xyzc_313/latest.pth`.
2. Visualization:
    ```
    # visualize novel views of single frame
    python run.py --type visualize --cfg_file configs/xyzc_demo_313.yaml exp_name xyzc_313
    # visualize novel views of dynamic humans
    python run.py --type visualize --cfg_file configs/xyzc_perform_313.yaml exp_name xyzc_313
    # visualize mesh
    python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_313.yaml exp_name xyzc_313 train.num_workers 0
    ```
3. The results of visualization are located at `$ROOT/data/render/xyzc_313` and `$ROOT/data/perform/xyzc_313`.

### Training on ZJU-Mocap

Take the training on `sequence 313` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:
    ```
    # training
    python train_net.py --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313 resume False
    # distributed training
    python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313 resume False gpus "0, 1, 2, 3" distributed True
    ```
2. Tensorboard:
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
