import numpy as np
from scipy.spatial import cKDTree as KDTree
import os
import tqdm
from lib.utils import data_utils
import glob
from lib.utils.if_nerf.voxels import VoxelGrid
from lib.config import cfg


def get_scaled_model(model):
    min_xyz = np.min(model, axis=0)
    max_xyz = np.max(model, axis=0)
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    center = (min_xyz + max_xyz) / 2
    scale = np.max(max_xyz - min_xyz)
    model = (model - center) / scale
    return model, bounds


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape), ))
    Y = Y.reshape((np.prod(Y.shape), ))
    Z = Z.reshape((np.prod(Z.shape), ))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


def voxelized_pointcloud(model, kdtree, res):
    occupancies = np.zeros(res ** 3, dtype=np.int8)
    _, idx = kdtree.query(model)
    occupancies[idx] = 1
    compressed_occupancies = np.packbits(occupancies)
    return compressed_occupancies


def ply_to_occupancy():
    data_root = 'data/light_stage'
    point_cloud_dir = os.path.join(data_root, 'point_cloud')
    voxel_dir = os.path.join(data_root, 'voxel')
    os.system('mkdir -p {}'.format(voxel_dir))

    bb_min = -0.5
    bb_max = 0.5
    res = 256
    grid_points = create_grid_points_from_bounds(bb_min, bb_max, res)
    kdtree = KDTree(grid_points)

    humans = os.listdir(point_cloud_dir)
    for human in humans:
        current_pc_dir = os.path.join(point_cloud_dir, human)
        current_voxel_dir = os.path.join(voxel_dir, human)
        os.system('mkdir -p {}'.format(current_voxel_dir))
        paths = sorted(os.listdir(current_pc_dir))
        for path in tqdm.tqdm(paths):
            model = data_utils.load_ply(os.path.join(current_pc_dir, path))
            model, bounds = get_scaled_model(model)
            compressed_occupancies = voxelized_pointcloud(model, kdtree, res)
            i = int(path.split('.')[0])
            np.savez(os.path.join(current_voxel_dir, '{}.npz'.format(i)),
                     compressed_occupancies=compressed_occupancies,
                     bounds=bounds)


def create_voxel_off():
    data_root = 'data/light_stage/voxel/CoreView_313'
    voxel_paths = glob.glob(os.path.join(data_root, '*.npz'))
    res = 256
    for voxel_path in voxel_paths:
        voxel_data = np.load(voxel_path)
        occupancy = np.unpackbits(voxel_data['compressed_occupancies'])
        occupancy = occupancy.reshape(res, res, res).astype(np.float32)
        i = int(os.path.basename(voxel_path).split('.')[0])
        VoxelGrid(occupancy).to_mesh().export(f'/home/pengsida/{i}.off')
        __import__('ipdb').set_trace()
