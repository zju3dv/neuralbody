"""
Prepare blend weights of grid points
"""

import os
import json
import numpy as np
import cv2
import sys
sys.path.append('/mnt/data/home/pengsida/Codes/SMPL_CPP/build/python')
import pysmplceres
import open3d as o3d
import pyskeleton
from psbody.mesh import Mesh
import pickle

# initialize a smpl model
pysmplceres.loadSMPL('/mnt/data/home/pengsida/Codes/SMPL_CPP/model/smpl/',
                     'smpl')


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def process_shapedirs(shapedirs, vert_ids, bary_coords):
    arr = []
    for i in range(3):
        t = barycentric_interpolation(shapedirs[:, i, :][vert_ids],
                                      bary_coords)
        arr.append(t[:, np.newaxis, :])
    arr = np.concatenate(arr, axis=1)
    return arr


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(rot_mats, joints, parents):
    """
    rot_mats: 24 x 3 x 3
    joints: 24 x 3
    parents: 24
    """
    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    transformed_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - transformed_joints

    return transforms


def get_transform_params(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = np.array(smpl['v_template'])

    # add shape blend shapes
    shapedirs = np.array(smpl['shapedirs'])
    betas = params['shapes']
    v_shaped = v_template + np.sum(shapedirs * betas[None], axis=2)

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # 24 x 3 x 3
    rot_mats = batch_rodrigues(poses)
    # 23 x 3 x 3
    pose_feature = rot_mats[1:].reshape(23, 3, 3) - np.eye(3)[None]
    pose_feature = pose_feature.reshape(1, 1, 207)
    posedirs = np.array(smpl['posedirs'])
    # v_posed = v_shaped + np.sum(posedirs * pose_feature, axis=2)
    v_posed = v_shaped

    # obtain the joints
    joints = smpl['J_regressor'].dot(v_shaped)

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation(rot_mats, joints, parents)

    # apply global transformation
    R = cv2.Rodrigues(params['Rh'][0])[0]
    Th = params['Th']

    return A, R, Th


def get_colored_pc(pts, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    colors = np.zeros_like(pts)
    colors += rgb
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def get_grid_points(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    vsize = 0.025
    voxel_size = [vsize, vsize, vsize]
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts


def get_canpts(param_path):
    params = np.load(param_path, allow_pickle=True).item()
    vertices = pysmplceres.getVertices(params)[0]
    faces = pysmplceres.getFaces()
    mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(
        '/mnt/data/home/pengsida/Codes/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl'
    )
    # obtain the transformation parameters for linear blend skinning
    A, R, Th = get_transform_params(smpl, params)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))
    bweights = barycentric_interpolation(smpl['weights'][vert_ids],
                                         bary_coords)

    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pts - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    can_pts = np.sum(R_inv * can_pts[:, None], axis=2)

    can_pts = can_pts.reshape(*sh).astype(np.float32)

    return can_pts


def prepare_tpose():
    data_root = '/home/pengsida/Datasets/light_stage'
    human = 'CoreView_315'
    param_dir = os.path.join(data_root, human, 'params')
    canpts_dir = os.path.join(data_root, human, 'canpts')
    os.system('mkdir -p {}'.format(canpts_dir))

    for i in range(len(os.listdir(param_dir))):
        i = i + 1
        param_path = os.path.join(param_dir, '{}.npy'.format(i))
        canpts = get_canpts(param_path)
        canpts_path = os.path.join(canpts_dir, '{}.npy'.format(i))
        np.save(canpts_path, canpts)


prepare_tpose()
