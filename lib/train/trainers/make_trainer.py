from .trainer import Trainer
import imp


def _wrapper_factory(cfg, network):
    module = cfg.trainer_module
    path = cfg.trainer_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return Trainer(network)
