from lib.utils.if_nerf import voxels
import numpy as np
from lib.config import cfg
import os


class Visualizer:
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
        i = batch['i'][0].item()
        result_path = os.path.join(result_dir, '{}.ply'.format(i))
        mesh.export(result_path)
