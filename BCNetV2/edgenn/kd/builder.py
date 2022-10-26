from ..utils import Registry, build_from_cfg

TeacherReg = Registry('teacher')
LossReg = Registry('loss')


def build_teacher(cfg):
    return build_from_cfg(cfg, TeacherReg)


def build_loss(cfg):
    return build_from_cfg(cfg, LossReg)
