import numpy as np
import json
import os
import cv2

from lib.config import cfg

from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts-c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def load_cam(ann_file):
    if ann_file.endswith('.json'):
        annots = json.load(open(ann_file, 'r'))
        cams = annots['cams']['20190823']
    else:
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']

    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        K[i][:2] = K[i][:2] * cfg.ratio

        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i]) / 1000.
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT


def get_center_rayd(K, RT):
    H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
    RT = np.array(RT)
    ray_o, ray_d = if_nerf_dutils.get_rays(H, W, K,
                            RT[:3, :3], RT[:3, 3])
    return ray_d[H // 2, W // 2]


def gen_path(RT, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.array(RT)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate([RT[:, :, 1:2], RT[:, :, 0:1],
                         -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, cfg.num_render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate([mat[:, 1:2], mat[:, 0:1],
                              -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c


def read_voxel(frame, args):
    voxel_path = os.path.join(args['data_root'], 'voxel', args['human'],
                              '{}.npz'.format(frame))
    voxel_data = np.load(voxel_path)
    occupancy = np.unpackbits(voxel_data['compressed_occupancies'])
    occupancy = occupancy.reshape(cfg.res, cfg.res,
                                  cfg.res).astype(np.float32)
    bounds = voxel_data['bounds'].astype(np.float32)
    return occupancy, bounds


def image_rays(RT, K, bounds):
    H = cfg.H * cfg.ratio
    W = cfg.W * cfg.ratio
    ray_o, ray_d = if_nerf_dutils.get_rays(H, W, K,
                            RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = if_nerf_dutils.get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    return ray_o, ray_d, near, far, center, scale, mask_at_box


def get_image_rays0(RT0, RT, K, bounds):
    """
    Use RT to get the mask_at_box and fill this region with rays emitted from view RT0
    """
    H = cfg.H * cfg.ratio
    ray_o, ray_d = if_nerf_dutils.get_rays(H, H, K,
                            RT[:3, :3], RT[:3, 3])

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = if_nerf_dutils.get_near_far(bounds, ray_o, ray_d)

    ray_o, ray_d = if_nerf_dutils.get_rays(H, H, K,
                            RT0[:3, :3], RT0[:3, 3])
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d[mask_at_box]

    return ray_d


def save_img(img, frame_root, index, mask_at_box):
    H = int(cfg.H * cfg.ratio)
    rgb_pred = img['rgb_map'][0].detach().cpu().numpy()
    mask_at_box = mask_at_box.reshape(H, H)

    img_pred = np.zeros((H, H, 3))
    img_pred[mask_at_box] = rgb_pred
    img_pred[:, :, [0, 1, 2]] = img_pred[:, :, [2, 1, 0]]

    print("saved frame %d" % index)
    cv2.imwrite(os.path.join(frame_root, '%d.jpg' % index), img_pred * 255)
