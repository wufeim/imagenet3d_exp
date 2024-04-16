import collections
from omegaconf import OmegaConf

from .general import get_abs_path


def load_config(config_path):
    config_path = get_abs_path(config_path)
    cfg = OmegaConf.load(config_path)
    includes = cfg.get('includes', [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError(
            "Includes must be a list, {} provided".format(type(includes))
        )

    include_cfg = OmegaConf.create()
    for include in includes:
        _cfg = load_config(include)
        include_cfg = OmegaConf.merge(include_cfg, _cfg)

    cfg.pop('includes', None)
    cfg = OmegaConf.merge(include_cfg, cfg)
    return cfg


def save_config(cfg, config_path):
    OmegaConf.save(config=cfg, f=config_path)
