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
