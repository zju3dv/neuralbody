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

# visualize novel views of single frame
# python run.py --type visualize --cfg_file configs/xyzc_demo_313.yaml exp_name xyzc_313_v1
# python run.py --type visualize --cfg_file configs/xyzc_demo_315.yaml exp_name xyzc_315
# python run.py --type visualize --cfg_file configs/xyzc_demo_392.yaml exp_name xyzc_392
# python run.py --type visualize --cfg_file configs/xyzc_demo_393.yaml exp_name xyzc_393
# python run.py --type visualize --cfg_file configs/xyzc_demo_394.yaml exp_name xyzc_394
# python run.py --type visualize --cfg_file configs/xyzc_demo_377.yaml exp_name xyzc_377
# python run.py --type visualize --cfg_file configs/xyzc_demo_386.yaml exp_name xyzc_386
# python run.py --type visualize --cfg_file configs/xyzc_demo_390.yaml exp_name xyzc_390
# python run.py --type visualize --cfg_file configs/xyzc_demo_387.yaml exp_name xyzc_387

# visualize novel views of dynamic humans
# python run.py --type visualize --cfg_file configs/xyzc_perform_313.yaml exp_name xyzc_313_v1
# python run.py --type visualize --cfg_file configs/xyzc_perform_315.yaml exp_name xyzc_315
# python run.py --type visualize --cfg_file configs/xyzc_perform_392.yaml exp_name xyzc_392
# python run.py --type visualize --cfg_file configs/xyzc_perform_393.yaml exp_name xyzc_393
# python run.py --type visualize --cfg_file configs/xyzc_perform_394.yaml exp_name xyzc_394
# python run.py --type visualize --cfg_file configs/xyzc_perform_377.yaml exp_name xyzc_377
# python run.py --type visualize --cfg_file configs/xyzc_perform_386.yaml exp_name xyzc_386
# python run.py --type visualize --cfg_file configs/xyzc_perform_390.yaml exp_name xyzc_390
# python run.py --type visualize --cfg_file configs/xyzc_perform_387.yaml exp_name xyzc_387

# visualize mesh
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_313.yaml exp_name xyzc_313_v1 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_315.yaml exp_name xyzc_315 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_392.yaml exp_name xyzc_392 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_393.yaml exp_name xyzc_393 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_394.yaml exp_name xyzc_394 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_377.yaml exp_name xyzc_377 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_386.yaml exp_name xyzc_386 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_390.yaml exp_name xyzc_390 train.num_workers 0
# python run.py --type visualize --cfg_file configs/latent_xyzc_mesh_387.yaml exp_name xyzc_387 train.num_workers 0

# visualize test views
# python run.py --type visualize --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313_v1 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_315.yaml exp_name xyzc_315 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_392.yaml exp_name xyzc_392 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_393.yaml exp_name xyzc_393 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_394.yaml exp_name xyzc_394 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_377.yaml exp_name xyzc_377 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_386.yaml exp_name xyzc_386 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_390.yaml exp_name xyzc_390 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'
# python run.py --type visualize --cfg_file configs/latent_xyzc_387.yaml exp_name xyzc_387 test_dataset_path 'lib/datasets/light_stage/can_smpl_test.py'  visualizer_path 'lib/visualizers/if_nerf_test.py' renderer_path 'lib/networks/renderer/if_clight_renderer_mmsk.py'

# visualize test views for NeRF
# python run.py --type visualize --cfg_file configs/nerf_313.yaml exp_name nerf_313 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_315.yaml exp_name nerf_315 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_392.yaml exp_name nerf_392 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_393.yaml exp_name nerf_393 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_394.yaml exp_name nerf_394 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_377.yaml exp_name nerf_377 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_386.yaml exp_name nerf_386 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_390.yaml exp_name nerf_390 visualizer_path 'lib/visualizers/if_nerf_test.py'
# python run.py --type visualize --cfg_file configs/nerf_387.yaml exp_name nerf_387 visualizer_path 'lib/visualizers/if_nerf_test.py'

# evaluation
# python run.py --type evaluate --cfg_file configs/latent_xyzc_313.yaml exp_name xyzc_313_v1
# python run.py --type evaluate --cfg_file configs/latent_xyzc_315.yaml exp_name xyzc_315
# python run.py --type evaluate --cfg_file configs/latent_xyzc_392.yaml exp_name xyzc_392
# python run.py --type evaluate --cfg_file configs/latent_xyzc_393.yaml exp_name xyzc_393
# python run.py --type evaluate --cfg_file configs/latent_xyzc_394.yaml exp_name xyzc_394
# python run.py --type evaluate --cfg_file configs/latent_xyzc_377.yaml exp_name xyzc_377
# python run.py --type evaluate --cfg_file configs/latent_xyzc_386.yaml exp_name xyzc_386
# python run.py --type evaluate --cfg_file configs/latent_xyzc_390.yaml exp_name xyzc_390
# python run.py --type evaluate --cfg_file configs/latent_xyzc_387.yaml exp_name xyzc_387
