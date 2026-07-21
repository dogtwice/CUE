import argparse

from .constants import *
from .config import Config
from . import defaults, experiments
def parse_config_name_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_name", required=True)
    return parser.parse_known_args()[0].config_name

def set_cfg_value(cfg, key_list, value):
    sub_cfg = cfg
    for key in key_list[:-1]:
        sub_cfg = getattr(sub_cfg, key)
    setattr(sub_cfg, key_list[-1], value)

def update_cfg(cfg):
    sep = "__"
    parser = argparse.ArgumentParser()
    cfg_dict = hparams_dict(cfg, sep=sep)

    parser.add_argument("-c", "--config", dest="config_name")
    parser.add_argument("--note", type=str, default='', help='备注信息（用于标注本次训练）')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    for key, value in cfg_dict.items():
        if isinstance(value, bool):
            parser.add_argument("--" + key, dest=key, default=value, type=str2bool)
        else:
            value_type = type(value) if isinstance(value, (int, float)) else None
            parser.add_argument("--" + key, dest=key, default=value, type=value_type)

    args = parser.parse_args()

    for key in cfg_dict.keys():
        key_list = key.split(sep)
        value = getattr(args, key)
        set_cfg_value(cfg, key_list, value)
    
    return args.note


def get_config_by_name(name):
    try:
        cfg = getattr(experiments, name)
    except Exception as err:
        raise RuntimeError(f"Config not found: {name}") from err
    cfg.network_config.loss_config.n_clusters = cfg.network_config.cm_config.n_clusters
    return cfg


def get_experiment_config():
    name = parse_config_name_arg()
    cfg = get_config_by_name(name)
    note = update_cfg(cfg)
    return name, cfg, note


def _insert_hparams(cfg_dict, hp_dict, key_prefix, skip_keys, sep="/"):
    hparam_types = (str, int, float, bool)
    for key, value in cfg_dict.items():
        if key in skip_keys:
            continue
        _key = f"{key_prefix}{sep}{key}" if key_prefix else key
        if isinstance(value, hparam_types) or value is None:
            hp_dict[_key] = value
        elif isinstance(value, dict):
            _insert_hparams(value, hp_dict, _key, skip_keys, sep=sep)


def hparams_dict(cfg, sep="/"):
    skip_keys = []
    hp_dict = {}
    _insert_hparams(cfg.dict(), hp_dict, "", skip_keys, sep=sep)
    return hp_dict
