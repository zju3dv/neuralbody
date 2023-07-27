import argparse
import json
import os
import os.path as osp
import sys
import glob
import numpy as np
import tqdm
import cv2


def visualize_o3d_pts(pts):
    import open3d as o3d
    pts = pts.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='my_313', type=str)
parser.add_argument('--type', default='annots', type=str)
args = parser.parse_args()

def get_cams():
    intri = cv2.FileStorage('intri.yml', cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage('extri.yml', cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in num_cams:
        cams['K'].append(intri.getNode('K_{}'.format(i)).mat())
        cams['D'].append(
            intri.getNode('dist_{}'.format(i)).mat().T)
        cams['R'].append(extri.getNode('Rot_{}'.format(i)).mat())
        cams['T'].append(extri.getNode('T_{}'.format(i)).mat() * 1000)
    return cams


def get_img_paths():
    all_ims = []
    for i in num_cams:
        data_root = 'images/{}'.format(i)
        ims = glob.glob(os.path.join(data_root, '*.jpg'))
        ims = sorted(ims)
        ims = np.array(ims)
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def gen_params_vertices(filename, param_in, param_out, vert_out):
    param_in_full = osp.join(param_in, filename)
    root = int(osp.splitext(filename)[0])

    params = json.load(open(param_in_full))['annots'][0]
    poses = np.array(params['poses'])
    Rh = np.array(params['Rh'])
    Th = np.array(params['Th'])
    shapes = np.array(params['shapes'])

    # the params of neural body
    params = {'poses': poses, 'Rh': Rh, 'Th': Th, 'shapes': shapes}
    # np.save('params_0.npy', params)
    np.save(osp.join(param_out, "{}.npy".format(root)), params)

    ori_poses = np.zeros((1, bodymodel.NUM_POSES_FULL))
    ori_poses[..., 3:] = poses
    vertices = bodymodel(poses=ori_poses, shapes=shapes, Rh=Rh, Th=Th)[0].detach().cpu().numpy()
    np.save(osp.join(vert_out, "{}.npy".format(root)), vertices)


if args.type == 'annots':
    os.chdir(args.input_dir)
    num_cams = os.listdir('images')
    num_cams = sorted(num_cams)

    cams = get_cams()
    img_paths = get_img_paths()

    annot = {}
    annot['cams'] = cams

    ims = []
    for img_path in img_paths:
        data = {}
        data['ims'] = img_path.tolist()
        ims.append(data)
    annot['ims'] = ims

    np.save('annots.npy', annot)
else:
    param_in = os.path.join(args.input_dir, 'output-smpl-3d/smpl/')
    param_out = osp.join(args.input_dir, 'params')
    vert_out = osp.join(args.input_dir, 'vertices')
    cfg_path = 'cfg_model.yml'

    os.system(f"mkdir -p {vert_out}")

    from easymocap.config.baseconfig import load_object, Config
    from easymocap.bodymodel.smplx import SMPLHModel, SMPLModel
    # load smpl model (maybe copy to gpu)
    cfg_model = Config.load(cfg_path)
    # cfg_model.args.model_path = cfg_model.args.model_path.replace('neutral', args.gender)
    cfg_model.module = cfg_model.module.replace('SMPLHModelEmbedding', 'SMPLHModel')
    # cfg_model.args.device = 'cpu'
    bodymodel: SMPLModel = load_object(cfg_model.module, cfg_model.args)

    for filename in tqdm.tqdm(sorted(os.listdir(param_in))):
        gen_params_vertices(filename, param_in, param_out, vert_out)
        
# generate annots.npy
# python easymocap_to_neuralbody.py --input_dir {data_dir} --type annots
# generate params and vertices
# python easymocap_to_neuralbody.py --input_dir {data_dir} --type vertices
