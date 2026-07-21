from .. import config
from .cue import CUE


MODEL_CONSTRUCTORS = {
    "CUE": CUE,
}


def build_model(model_cfg):
    if model_cfg.class_name not in MODEL_CONSTRUCTORS:
        raise ValueError(f"Invalid model type: {model_cfg.class_name}")
    model = MODEL_CONSTRUCTORS[model_cfg.class_name](model_cfg).to(config.DEVICE, non_blocking=True)
    return model
