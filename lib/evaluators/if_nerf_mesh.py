import numpy as np
from lib.config import cfg
import os


class Evaluator:
    def evaluate(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]

        pts = batch['pts'][0].detach().cpu().numpy()
        pts = pts[cube > cfg.mesh_th]

        i = batch['i'].item()
        result_dir = os.path.join(cfg.result_dir, 'pts')
        os.system('mkdir -p {}'.format(result_dir))
        result_path = os.path.join(result_dir, '{}.npy'.format(i))
        np.save(result_path, pts)

    def summarize(self):
        return {}
