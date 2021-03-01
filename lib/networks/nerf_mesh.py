import torch.nn as nn
import torch
from lib.config import cfg
from .embedder import get_embedder
import torch.nn.functional as F


class Nerf(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_views=3,
                 skips=[4],
                 use_viewdirs=False):
        """
        """
        super(Nerf, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [
            nn.Linear(W, W) if i not in
            self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)
        ])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts = x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        return alpha

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1]))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.embed_fn, input_ch = get_embedder(cfg.xyz_res)
        self.embeddirs_fn, input_ch_views = get_embedder(cfg.view_res)

        skips = [4]
        self.model = Nerf(D=cfg.netdepth,
                          W=cfg.netwidth,
                          input_ch=input_ch,
                          skips=skips,
                          input_ch_views=input_ch_views,
                          use_viewdirs=cfg.use_viewdirs)

        # self.model_fine = Nerf(D=cfg.netdepth_fine,
        #                        W=cfg.netwidth_fine,
        #                        input_ch=input_ch,
        #                        skips=skips,
        #                        input_ch_views=input_ch_views,
        #                        use_viewdirs=cfg.use_viewdirs)

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        def ret(inputs):
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret

    def forward(self, inputs, model=''):
        """Prepares inputs and applies network 'fn'.
        """
        if model == 'fine':
            fn = self.model_fine
        else:
            fn = self.model

        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)
        outputs_flat = self.batchify(fn, cfg.netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs
