# People-Snapshot dataset

# training
# python train_net.py --cfg_file configs/snapshot_f3c.yaml exp_name female3c resume False
# python train_net.py --cfg_file configs/snapshot_f4c.yaml exp_name female4c resume False
# python train_net.py --cfg_file configs/snapshot_f6p.yaml exp_name female6p resume False
# python train_net.py --cfg_file configs/snapshot_f7p.yaml exp_name female7p resume False
# python train_net.py --cfg_file configs/snapshot_f8p.yaml exp_name female8p resume False
# python train_net.py --cfg_file configs/snapshot_m2c.yaml exp_name male2c resume False
# python train_net.py --cfg_file configs/snapshot_m2o.yaml exp_name male2o resume False
# python train_net.py --cfg_file configs/snapshot_m3c.yaml exp_name male3c resume False
# python train_net.py --cfg_file configs/snapshot_m5o.yaml exp_name male5o resume False

# ZJU-Mocap dataset

# training
# python train_net.py --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313 resume False
# python train_net.py --cfg_file configs/latent_xyzc_315.yaml exp_name xyzc_315 resume False
# python train_net.py --cfg_file configs/latent_xyzc_392.yaml exp_name xyzc_392 resume False
# python train_net.py --cfg_file configs/latent_xyzc_393.yaml exp_name xyzc_393 resume False
# python train_net.py --cfg_file configs/latent_xyzc_394.yaml exp_name xyzc_394 resume False
# python train_net.py --cfg_file configs/latent_xyzc_377.yaml exp_name xyzc_377 resume False
# python train_net.py --cfg_file configs/latent_xyzc_386.yaml exp_name xyzc_386 resume False
# python train_net.py --cfg_file configs/latent_xyzc_390.yaml exp_name xyzc_390 resume False
# python train_net.py --cfg_file configs/latent_xyzc_387.yaml exp_name xyzc_387 resume False

# distributed training
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_315.yaml exp_name xyzc_315 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_392.yaml exp_name xyzc_392 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_393.yaml exp_name xyzc_393 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_394.yaml exp_name xyzc_394 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_377.yaml exp_name xyzc_377 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_386.yaml exp_name xyzc_386 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_390.yaml exp_name xyzc_390 resume False gpus "0, 1, 2, 3" distributed True
# python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/latent_xyzc_387.yaml exp_name xyzc_387 resume False gpus "0, 1, 2, 3" distributed True
