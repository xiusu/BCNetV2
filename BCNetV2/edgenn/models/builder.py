from ..utils import Registry, build_from_cfg

BackboneReg = Registry('backbone')
LossReg = Registry('loss')


def build_backbone(cfg):
    return build_from_cfg(cfg, BackboneReg)


def build_loss(cfg):
    return build_from_cfg(cfg, LossReg)
