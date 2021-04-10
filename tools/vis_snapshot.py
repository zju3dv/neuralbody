import pickle
import os
import h5py
import numpy as np
import open3d as o3d
from snapshot_smpl.renderer import Renderer
import cv2
import tqdm


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_KRTD(camera):
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    return K, R, T, D


def get_o3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def get_smpl(base_smpl, betas, poses, trans):
    base_smpl.betas = betas
    base_smpl.pose = poses
    base_smpl.trans = trans
    vertices = np.array(base_smpl)

    faces = base_smpl.f
    mesh = get_o3d_mesh(vertices, faces)

    return mesh


def render_smpl(vertices, img, K, R, T):
    rendered_img = renderer.render_multiview(vertices, K[None], R[None],
                                             T[None, None], [img])[0]
    return rendered_img


data_root = 'data/people_snapshot'
video = 'female-3-casual'

# if you do not have these smpl models, you could download them from https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/Eb_JIyA74O1Cnfhvn1ddrG4BC9TMK31022TykVxGdRenUQ?e=JU8pPt
model_paths = [
    'basicModel_f_lbs_10_207_0_v1.0.0.pkl',
    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
]

camera_path = os.path.join(data_root, video, 'camera.pkl')
camera = read_pickle(camera_path)
K, R, T, D = get_KRTD(camera)

mask_path = os.path.join(data_root, video, 'masks.hdf5')
masks = h5py.File(mask_path)['masks']

smpl_path = os.path.join(data_root, video, 'reconstructed_poses.hdf5')
smpl = h5py.File(smpl_path)
betas = smpl['betas']
pose = smpl['pose']
trans = smpl['trans']

if 'female' in video:
    model_path = model_paths[0]
else:
    model_path = model_paths[1]
model_data = read_pickle(model_path)
faces = model_data['f']
renderer = Renderer(height=1080, width=1080, faces=faces)

img_dir = os.path.join(data_root, video, 'image')
vertices_dir = os.path.join(data_root, video, 'vertices')

num_img = len(os.listdir(img_dir))
for i in tqdm.tqdm(range(num_img)):
    img = cv2.imread(os.path.join(img_dir, '{}.jpg'.format(i)))
    img = cv2.undistort(img, K, D)
    vertices = np.load(os.path.join(vertices_dir, '{}.npy'.format(i)))
    rendered_img = render_smpl(vertices, img, K, R, T)
    cv2.imshow('main', rendered_img)
    cv2.waitKey(50) & 0xFF
