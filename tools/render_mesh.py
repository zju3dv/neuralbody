# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
import sys
import os

from render.camera import Camera
from render.color_render import ColorRender
import trimesh

import cv2
import os
import argparse
from termcolor import colored

width = 512
height = 512


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


parser = argparse.ArgumentParser()
parser.add_argument('-ww', '--width', type=int, default=512)
parser.add_argument('-hh', '--height', type=int, default=512)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--mesh_ind', type=int, default=0)

args = parser.parse_args()

renderer = ColorRender(width=args.width, height=args.height)
cam = Camera(width=1.0, height=args.height / args.width)
cam.ortho_ratio = 1.2
cam.near = -100
cam.far = 10

data_root = 'data/result/if_nerf/{}/mesh'.format(
    args.exp_name)
obj_path = os.path.join(data_root, '{:04d}.ply'.format(args.mesh_ind))

mesh_render_dir = os.path.join(data_root, 'mesh{}_render'.format(args.mesh_ind))

os.system('mkdir -p {}'.format(mesh_render_dir))
obj_files = [obj_path]

if args.dataset == 'zju_mocap':
    R = make_rotate(0, math.radians(0), 0)  # zju-mocap
else:
    R = make_rotate(0, math.radians(90), math.radians(90))  # people-snapshot

print(colored('the results are saved at {}'.format(mesh_render_dir), 'yellow'))

for i, obj_path in enumerate(obj_files):

    print(obj_path)
    obj_file = obj_path.split('/')[-1]
    file_name = obj_file[:-4]

    if not os.path.exists(obj_path):
        continue
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces

    rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    vertices = np.dot(vertices, rot.T)
    mesh.vertices = vertices

    vertices = np.matmul(vertices, R.T)
    bbox_max = vertices.max(0)
    bbox_min = vertices.min(0)

    # notice that original scale is discarded to render with the same size
    vertices -= 0.5 * (bbox_max + bbox_min)[None, :]
    vertices /= bbox_max[1] - bbox_min[1]

    normals = compute_normal(vertices, faces)

    renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)

    self_rot = make_rotate(i, math.radians(-90), 0)
    vertices = np.matmul(vertices, self_rot.T)
    cnt = 0
    for j in range(0, 361, 4):
        cam.center = np.array([0, 0, 0])
        cam.eye = np.array([
            2.0 * math.sin(math.radians(0)), 0, 2.0 * math.cos(math.radians(0))
        ]) + cam.center

        self_rot = make_rotate(i, math.radians(-4), 0)
        vertices = np.matmul(vertices, self_rot.T)
        normals = compute_normal(vertices, faces)

        renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)
        renderer.set_camera(cam)
        renderer.display()

        img = renderer.get_color(0)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = img[..., :3]

        cv2.imwrite(os.path.join(mesh_render_dir, '%d.jpg' % cnt), 255 * img)
        cnt += 1
