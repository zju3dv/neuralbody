import os
import imp


def make_network(cfg):
    # module = lib.networks.latent_xyzc
    module = cfg.network_module
    # path = lib/networks/latent_xyzc.py
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network
