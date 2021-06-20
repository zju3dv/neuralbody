from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = os.path.join(cfg.result_dir, 'mesh')
        print(colored('the results are saved at {}'.format(result_dir), 'yellow'))

    def visualize_voxel(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]
        cube[cube < cfg.mesh_th] = 0
        cube[cube > cfg.mesh_th] = 1

        sh = cube.shape
        square_cube = np.zeros((max(sh), ) * 3)
        square_cube[:sh[0], :sh[1], :sh[2]] = cube
        voxel_grid = voxels.VoxelGrid(square_cube)
        mesh = voxel_grid.to_mesh()
        mesh.show()

    def visualize(self, output, batch):
        mesh = output['mesh']
        # mesh.show()

        result_dir = os.path.join(cfg.result_dir, 'mesh')
        os.system('mkdir -p {}'.format(result_dir))
        i = batch['frame_index'].item()
        result_path = os.path.join(result_dir, '{:04d}.ply'.format(i))
        mesh.export(result_path)
