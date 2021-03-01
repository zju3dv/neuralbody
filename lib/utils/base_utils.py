import pickle
import os
import numpy as np


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def write_K_pose_inf(K, poses, img_root):
    K = K.copy()
    K[:2] = K[:2] * 8
    K_inf = os.path.join(img_root, 'Intrinsic.inf')
    os.system('mkdir -p {}'.format(os.path.dirname(K_inf)))
    with open(K_inf, 'w') as f:
        for i in range(len(poses)):
            f.write('%d\n'%i)
            f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(K.reshape(9).tolist()))
            f.write('\n')

    pose_inf = os.path.join(img_root, 'CamPose.inf')
    with open(pose_inf, 'w') as f:
        for pose in poses:
            pose = np.linalg.inv(pose)
            A = pose[0:3,:]
            tmp = np.concatenate([A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))
