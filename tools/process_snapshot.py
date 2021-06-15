import pickle
import os
import h5py
import sys
import numpy as np
import open3d as o3d
from snapshot_smpl.smpl import Smpl
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

    return vertices, mesh


def render_smpl(mesh, img, K, R, T):
    vertices = np.array(mesh.vertices)
    rendered_img = renderer.render_multiview(vertices, K[None], R[None],
                                             T[None, None], [img])[0]
    return rendered_img


def extract_image(data_path):
    data_root = os.path.dirname(data_path)
    img_dir = os.path.join(data_root, 'image')
    os.system('mkdir -p {}'.format(img_dir))

    if len(os.listdir(img_dir)) >= 200:
        return

    cap = cv2.VideoCapture(data_path)

    ret, frame = cap.read()
    i = 0

    while ret:
        cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(i)), frame)
        ret, frame = cap.read()
        i = i + 1

    cap.release()


def extract_mask(masks, mask_dir):
    if len(os.listdir(mask_dir)) >= len(masks):
        return

    for i in tqdm.tqdm(range(len(masks))):
        mask = masks[i].astype(np.uint8)

        # erode the mask
        border = 4
        kernel = np.ones((border, border), np.uint8)
        mask = cv2.erode(mask.copy(), kernel)

        cv2.imwrite(os.path.join(mask_dir, '{}.png'.format(i)), mask)


data_root = 'data/people_snapshot'
videos = ['female-3-casual']

model_paths = [
    'basicModel_f_lbs_10_207_0_v1.0.0.pkl',
    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
]

for video in videos:
    camera_path = os.path.join(data_root, video, 'camera.pkl')
    camera = read_pickle(camera_path)
    K, R, T, D = get_KRTD(camera)

    # process video
    video_path = os.path.join(data_root, video, video + '.mp4')
    extract_image(video_path)

    # process mask
    mask_path = os.path.join(data_root, video, 'masks.hdf5')
    masks = h5py.File(mask_path)['masks']
    mask_dir = os.path.join(data_root, video, 'mask')
    os.system('mkdir -p {}'.format(mask_dir))
    extract_mask(masks, mask_dir)

    smpl_path = os.path.join(data_root, video, 'reconstructed_poses.hdf5')
    smpl = h5py.File(smpl_path)
    betas = smpl['betas']
    pose = smpl['pose']
    trans = smpl['trans']

    pose = pose[len(pose) - len(masks):]
    trans = trans[len(trans) - len(masks):]

    # process smpl parameters
    params = {'beta': np.array(betas), 'pose': pose, 'trans': trans}
    params_path = os.path.join(data_root, video, 'params.npy')
    np.save(params_path, params)

    if 'female' in video:
        model_path = model_paths[0]
    else:
        model_path = model_paths[1]
    model_data = read_pickle(model_path)

    img_dir = os.path.join(data_root, video, 'image')
    vertices_dir = os.path.join(data_root, video, 'vertices')
    os.system('mkdir -p {}'.format(vertices_dir))

    num_img = len(os.listdir(img_dir))
    for i in tqdm.tqdm(range(num_img)):
        base_smpl = Smpl(model_data)
        vertices, mesh = get_smpl(base_smpl, betas, pose[i], trans[i])
        np.save(os.path.join(vertices_dir, '{}.npy'.format(i)), vertices)
