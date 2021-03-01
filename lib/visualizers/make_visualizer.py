import os
import imp


def make_visualizer(cfg):
    module = cfg.visualizer_module
    path = cfg.visualizer_path
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer
