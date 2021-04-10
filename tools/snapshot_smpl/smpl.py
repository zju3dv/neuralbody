import chumpy as ch
import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp
from chumpy.ch import Ch
from .vendor.smpl.posemapper import posemap, Rodrigues
from .vendor.smpl.serialization import backwards_compatibility_replacements


VERT_NOSE = 331
VERT_EAR_L = 3485
VERT_EAR_R = 6880
VERT_EYE_L = 2802
VERT_EYE_R = 6262


class Smpl(Ch):
    """
    Class to store SMPL object with slightly improved code and access to more matrices
    """
    terms = 'model',
    dterms = 'trans', 'betas', 'pose', 'v_personal'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        if not hasattr(self, 'trans'):
            self.trans = ch.zeros(3)

        if not hasattr(self, 'betas'):
            self.betas = ch.zeros(10)

        if not hasattr(self, 'pose'):
            self.pose = ch.zeros(72)

        if 'model' in which:
            if not isinstance(self.model, dict):
                dd = pkl.load(open(self.model))
            else:
                dd = self.model

            backwards_compatibility_replacements(dd)

            for s in ['posedirs', 'shapedirs']:
                if (s in dd) and not hasattr(dd[s], 'dterms'):
                    dd[s] = ch.array(dd[s])

            self.f = dd['f']
            self.v_template = dd['v_template']
            if not hasattr(self, 'v_personal'):
                self.v_personal = ch.zeros_like(self.v_template)
            self.shapedirs = dd['shapedirs']
            self.J_regressor = dd['J_regressor']
            if 'J_regressor_prior' in dd:
                self.J_regressor_prior = dd['J_regressor_prior']
            if sp.issparse(self.J_regressor):
                self.J_regressor = self.J_regressor.toarray()
            self.bs_type = dd['bs_type']
            self.weights = dd['weights']
            if 'vert_sym_idxs' in dd:
                self.vert_sym_idxs = dd['vert_sym_idxs']
            if 'weights_prior' in dd:
                self.weights_prior = dd['weights_prior']
            self.kintree_table = dd['kintree_table']
            self.posedirs = dd['posedirs']

            self._set_up()

    def _set_up(self):
        self.v_shaped = self.shapedirs.dot(self.betas) + self.v_template
        self.v_shaped_personal = self.v_shaped + self.v_personal
        self.J = ch.sum(self.J_regressor.T.reshape(-1, 1, 24) * self.v_shaped.reshape(-1, 3, 1), axis=0).T
        self.v_posevariation = self.posedirs.dot(posemap(self.bs_type)(self.pose))
        self.v_poseshaped = self.v_shaped_personal + self.v_posevariation

        self.A, A_global = self._global_rigid_transformation()
        self.Jtr = ch.vstack([g[:3, 3] for g in A_global])
        self.J_transformed = self.Jtr + self.trans.reshape((1, 3))

        self.V = self.A.dot(self.weights.T)

        rest_shape_h = ch.hstack((self.v_poseshaped, ch.ones((self.v_poseshaped.shape[0], 1))))
        self.v_posed = ch.sum(self.V.T * rest_shape_h.reshape(-1, 4, 1), axis=1)[:, :3]
        self.v = self.v_posed + self.trans

    def _global_rigid_transformation(self):
        results = {}
        pose = self.pose.reshape((-1, 3))
        parent = {i: self.kintree_table[0, i] for i in range(1, self.kintree_table.shape[1])}

        with_zeros = lambda x: ch.vstack((x, ch.array([[0.0, 0.0, 0.0, 1.0]])))
        pack = lambda x: ch.hstack([ch.zeros((4, 3)), x.reshape((4, 1))])

        results[0] = with_zeros(ch.hstack((Rodrigues(pose[0, :]), self.J[0, :].reshape((3, 1)))))

        for i in range(1, self.kintree_table.shape[1]):
            results[i] = results[parent[i]].dot(with_zeros(ch.hstack((
                Rodrigues(pose[i, :]),      # rotation around bone endpoint
                (self.J[i, :] - self.J[parent[i], :]).reshape((3, 1))     # bone
            ))))

        results = [results[i] for i in sorted(results.keys())]
        results_global = results

        # subtract rotated J position
        results2 = [results[i] - (pack(
            results[i].dot(ch.concatenate((self.J[i, :], [0]))))
        ) for i in range(len(results))]
        result = ch.dstack(results2)

        return result, results_global

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.trans and wrt is not self.betas and wrt is not self.pose and wrt is not self.v_personal:
            return None

        return self.v.dr_wrt(wrt)


def copy_smpl(smpl, model):
    new = Smpl(model, betas=smpl.betas)
    new.pose[:] = smpl.pose.r
    new.trans[:] = smpl.trans.r

    return new


def joints_coco(smpl):
    J = smpl.J_transformed
    nose = smpl[VERT_NOSE]
    ear_l = smpl[VERT_EAR_L]
    ear_r = smpl[VERT_EAR_R]
    eye_l = smpl[VERT_EYE_L]
    eye_r = smpl[VERT_EYE_R]

    shoulders_m = ch.sum(J[[14, 13]], axis=0) / 2.
    neck = J[12] - 0.55 * (J[12] - shoulders_m)

    return ch.vstack((
        nose,
        neck,
        2.1 * (J[14] - shoulders_m) + neck,
        J[[19, 21]],
        2.1 * (J[13] - shoulders_m) + neck,
        J[[18, 20]],
        J[2] + 0.38 * (J[2] - J[1]),
        J[[5, 8]],
        J[1] + 0.38 * (J[1] - J[2]),
        J[[4, 7]],
        eye_r,
        eye_l,
        ear_r,
        ear_l,
    ))


def model_params_in_camera_coords(trans, pose, J0, camera_t, camera_rt):
    root = Rodrigues(np.matmul(Rodrigues(camera_rt).r, Rodrigues(pose[:3]).r)).r.reshape(-1)
    pose[:3] = root

    trans = (Rodrigues(camera_rt).dot(J0 + trans) - J0 + camera_t).r

    return trans, pose


if __name__ == '__main__':
    smpl = Smpl(model='../vendor/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    smpl.pose[:] = np.random.randn(72) * .2
    smpl.pose[0] = np.pi
    # smpl.v_personal[:] = np.random.randn(*smpl.shape) / 500.

    # render test
    from opendr.renderer import ColoredRenderer
    from opendr.camera import ProjectPoints
    from opendr.lighting import LambertianPointLight

    rn = ColoredRenderer()

    # Assign attributes to renderer
    w, h = (640, 480)

    rn.camera = ProjectPoints(v=smpl, rt=np.zeros(3), t=np.array([0, 0, 3.]), f=np.array([w, w]),
                              c=np.array([w, h]) / 2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=smpl, f=smpl.f, bgcolor=np.zeros(3))

    # Construct point light source
    rn.vc = LambertianPointLight(
        f=smpl.f,
        v=rn.v,
        num_verts=len(smpl),
        light_pos=np.array([-1000, -1000, -2000]),
        vc=np.ones_like(smpl) * .9,
        light_color=np.array([1., 1., 1.]))

    # Show it using OpenCV
    import cv2

    cv2.imshow('render_SMPL', rn.r)
    print ('..Print any key while on the display window')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
