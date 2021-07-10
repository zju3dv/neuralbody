import os
import sys
import json

import numpy as np
import torch
sys.path.append("../")
from smplmodel.body_model import SMPLlayer

easymocap_params = json.load(open('zju_smpl/example.json'))[0]
poses = np.array(easymocap_params['poses'])
Rh = np.array(easymocap_params['Rh'])
Th = np.array(easymocap_params['Th'])
shapes = np.array(easymocap_params['shapes'])

# the params of neural body
params = {'poses': poses, 'Rh': Rh, 'Th': Th, 'shapes': shapes}
# np.save('params_0.npy', params)

# The newlly fitted SMPL parameters consider pose blend shapes.
new_params = True

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
vertices = body_model(return_verts=True,
                      return_tensor=False,
                      new_params=new_params,
                      **params)
# np.save('vertices_0.npy', vertices)
