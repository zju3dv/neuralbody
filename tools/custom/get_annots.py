import cv2
import numpy as np
import glob
import os
import json


def get_cams():
    intri = cv2.FileStorage('intri.yml', cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage('extri.yml', cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in range(23):
        cams['K'].append(intri.getNode('K_Camera_B{}'.format(i + 1)).mat())
        cams['D'].append(
            intri.getNode('dist_Camera_B{}'.format(i + 1)).mat().T)
        cams['R'].append(extri.getNode('Rot_Camera_B{}'.format(i + 1)).mat())
        cams['T'].append(extri.getNode('T_Camera_B{}'.format(i + 1)).mat() * 1000)
    return cams


def get_img_paths():
    all_ims = []
    for i in range(23):
        i = i + 1
        data_root = 'Camera_B{}'.format(i)
        ims = glob.glob(os.path.join(data_root, '*.jpg'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


cams = get_cams()
img_paths = get_img_paths()

annot = {}
annot['cams'] = cams

ims = []
for img_path, kpt in zip(img_paths, kpts2d):
    data = {}
    data['ims'] = img_path.tolist()
    ims.append(data)
annot['ims'] = ims

np.save('annots.npy', annot)
np.save('annots_python2.npy', annot, fix_imports=True)
