import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.patches as patches
from sklearn.manifold import TSNE
import open3d as o3d

kintree = {
    'kintree': [[1, 0], [2, 1], [3, 2], [4, 3], [5, 1], [6, 5], [7, 6], [8, 1],
                [9, 8], [10, 9], [11, 10], [12, 8], [13, 12], [14, 13],
                [15, 0], [16, 0], [17, 15], [18, 16], [19, 14], [20, 19],
                [21, 14], [22, 11], [23, 22], [24, 11]],
    'color': [
        'k', 'r', 'r', 'r', 'b', 'b', 'b', 'k', 'r', 'r', 'r', 'b', 'b', 'b',
        'y', 'y', 'y', 'y', 'b', 'b', 'b', 'r', 'r', 'r'
    ]
}


def plotSkel3D(pts,
               config=kintree,
               ax=None,
               phi=0,
               theta=0,
               max_range=1,
               linewidth=4,
               color=None):
    multi = False
    if torch.is_tensor(pts):
        if len(pts.shape) == 3:
            print(">>> Visualize multiperson ...")
            multi = True
            if pts.shape[1] != 3:
                pts = pts.transpose(1, 2)
        elif len(pts.shape) == 2:
            if pts.shape[0] != 3:
                pts = pts.transpose(0, 1)
        else:
            raise RuntimeError('The dimension of the points is wrong!')
        pts = pts.detach().cpu().numpy()
    else:
        if pts.shape[0] != 3:
            pts = pts.T
    # pts : bn, 3, NumOfPoints or (3, N)
    if ax is None:
        print('>>> create figure ...')
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111, projection='3d')
    for idx, (i, j) in enumerate(config['kintree']):
        if multi:
            for b in range(pts.shape[0]):
                ax.plot([pts[b][0][i], pts[b][0][j]],
                        [pts[b][1][i], pts[b][1][j]],
                        [pts[b][2][i], pts[b][2][j]],
                        lw=linewidth,
                        color=config['color'][idx] if color is None else color,
                        alpha=1)
        else:
            ax.plot([pts[0][i], pts[0][j]], [pts[1][i], pts[1][j]],
                    [pts[2][i], pts[2][j]],
                    lw=linewidth,
                    color=config['color'][idx],
                    alpha=1)
    if multi:
        for b in range(pts.shape[0]):
            ax.scatter(pts[b][0], pts[b][1], pts[b][2], color='r', alpha=1)
    else:
        ax.scatter(pts[0], pts[1], pts[2], color='r', alpha=1, s=0.5)
    ax.view_init(phi, theta)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-0.05, 2)

    # ax.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.zlabel('z')
    return ax


def plotSkel2D(pts,
               config=kintree,
               ax=None,
               linewidth=2,
               alpha=1,
               max_range=1,
               imgshape=None,
               thres=0.1):
    if len(pts.shape) == 2:
        pts = pts[None, :, :]  #(nP, nJ, 2/3)
    elif len(pts.shape) == 3:
        pass
    else:
        raise RuntimeError('The dimension of the points is wrong!')
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if pts.shape[2] == 3 or pts.shape[2] == 2:
        pts = pts.transpose((0, 2, 1))
    # pts : bn, 2/3, NumOfPoints or (2/3, N)
    if ax is None:
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111)
    if 'color' in config.keys():
        colors = config['color']
    else:
        colors = ['b' for _ in range(len(config['kintree']))]

    def inrange(imgshape, pts):
        if pts[0] < 5 or \
           pts[0] > imgshape[1] - 5 or \
           pts[1] < 5 or \
           pts[1] > imgshape[0] - 5:
            return False
        else:
            return True

    for nP in range(pts.shape[0]):
        for idx, (i, j) in enumerate(config['kintree']):
            if pts.shape[1] == 3:  # with confidence
                if np.min(pts[nP][2][[i, j]]) < thres:
                    continue
                lw = linewidth * 2 * np.min(pts[nP][2][[i, j]])
            else:
                lw = linewidth
            if imgshape is not None:
                if inrange(imgshape, pts[nP, :, i]) and \
                    inrange(imgshape, pts[nP, :, j]):
                    pass
                else:
                    continue
            ax.plot([pts[nP][0][i], pts[nP][0][j]],
                    [pts[nP][1][i], pts[nP][1][j]],
                    lw=lw,
                    color=colors[idx],
                    alpha=1)
        # if pts.shape[1] > 2:
        # ax.scatter(pts[nP][0], pts[nP][1], s=10*(pts[nP][2]-thres), c='r')
    if False:
        ax.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
    else:
        ax.axis('off')
    return ax


def draw_skeleton(img, kpts2d):
    cv_img = img.copy()
    for kp in kpts2d:
        if kp.shape[-1] == 2 or (kp.shape[-1] == 3 and kp[-1] > 0):
            cv_img = cv2.circle(cv_img, tuple(kp[:2].astype(int)), 10,
                                (255, 0, 0))
    return cv_img


def vis_frame(data_root, im_data, camera):
    from external.SMPL_CPP.build.python import pysmplceres
    from .smpl_renderer import Renderer

    imgs = [
        cv2.imread(os.path.join(data_root, im_path))
        for im_path in im_data['ims']
    ]
    imgs = [cv2.resize(img, (1024, 1024)) for img in imgs]

    Ks = np.array(camera['K'])
    Rs = np.array(camera['R'])
    Ts = np.array(camera['T']).transpose(0, 2, 1) / 1000

    faces = np.loadtxt('data/smpl/faces.txt').astype(np.int32)
    render = Renderer(height=1024, width=1024, faces=faces)
    vertices = pysmplceres.getVertices(im_data['smpl_result'])

    imgsrender = render.render_multiview(vertices[0], Ks, Rs, Ts, imgs)
    for img in imgsrender:
        plt.imshow(img[..., ::-1])
        plt.show()


def vis_skeleton_frame(data_root, im_data, camera):
    from external.SMPL_CPP.build.python import pysmplceres
    from .smpl_renderer import Renderer

    imgs = [
        cv2.imread(os.path.join(data_root, im_path))
        for im_path in im_data['ims']
    ]
    imgs = [cv2.resize(img, (1024, 1024)) for img in imgs]
    kpts2d = np.array(im_data['kpts2d'])

    for img, kpts in zip(imgs, kpts2d):
        _, ax = plt.subplots(1, 1)
        ax.imshow(img[..., ::-1])
        plotSkel2D(kpts, ax=ax)
        plt.show()


def vis_bbox(img, corners_2d, coord):
    _, ax = plt.subplots(1)
    ax.imshow(img)
    ax.add_patch(
        patches.Polygon(xy=corners_2d[[0, 1, 3, 2, 0, 4, 6, 2]],
                        fill=False,
                        linewidth=1,
                        edgecolor='g'))
    ax.add_patch(
        patches.Polygon(xy=corners_2d[[5, 4, 6, 7, 5, 1, 3, 7]],
                        fill=False,
                        linewidth=1,
                        edgecolor='g'))
    ax.plot(coord[:, 1], coord[:, 0], '.')
    plt.show()


def tsne_colors(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1,
                verbose=1,
                perplexity=40,
                n_iter=300,
                random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.Spectral(tsne_results)[:, :3]
    return colors


def get_colored_pc(pts, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    colors = np.zeros_like(pts)
    colors += rgb
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc
