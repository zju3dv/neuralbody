import os
import sys

import numpy as np
import torch
sys.path.append("../")
from smplmodel.body_model import SMPLlayer

smpl_dir = 'data/zju_mocap/CoreView_313/params'
verts_dir = 'data/zju_mocap/CoreView_313/vertices'

# Previously, EasyMocap estimated SMPL parameters without pose blend shapes.
# The newlly fitted SMPL parameters consider pose blend shapes.
new_params = False
if 'new' in os.path.basename(smpl_dir):
    new_params = True

smpl_path = os.path.join(smpl_dir, "1.npy")
verts_path = os.path.join(verts_dir, "1.npy")

## load precomputed vertices
verts_load = np.load(verts_path)

## create smpl model
model_folder = 'data/zju_mocap/smplx'
device = torch.device('cpu')
body_model = SMPLlayer(os.path.join(model_folder, 'smpl'),
                       gender='neutral',
                       device=device,
                       regressor_path=os.path.join(model_folder,
                                                   'J_regressor_body25.npy'))
body_model.to(device)

## load SMPL zju
params = np.load(smpl_path, allow_pickle=True).item()

vertices = body_model(return_verts=True,
                      return_tensor=False,
                      new_params=new_params,
                      **params)
