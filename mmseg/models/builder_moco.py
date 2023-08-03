from torch import nn

from mmseg.utils import build_from_cfg
from .registry import (BACKBONES, MODELS, NECKS, HEADS, MEMORIES, LOSSES)


def build(cfg, registry, default_args=None):
    """Build a module. 这部分代码从openselfsup 修改而来，为了初始化heads和necks

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Default: None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone_moco(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck_moco(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_memory(cfg):
    """Build memory."""
    return build(cfg, MEMORIES)


def build_head_moco(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss_moco(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_model_moco(cfg):
    """Build model."""
    return build(cfg, MODELS)