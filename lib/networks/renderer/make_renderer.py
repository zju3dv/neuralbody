import os
import imp


def make_renderer(cfg, network):
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer = imp.load_source(module, path).Renderer(network)
    return renderer
