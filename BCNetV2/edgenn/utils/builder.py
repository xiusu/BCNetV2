from .registry import Registry, build_from_cfg

ModelReg = Registry('model')
EvaluatorReg = Registry('evaluator')
AlgorithmReg = Registry('algorithm')
TrainerReg = Registry('trainer')


def build_model(cfg):
    return build_from_cfg(cfg, ModelReg)


def build_evaluator(cfg):
    return build_from_cfg(cfg, EvaluatorReg)


def build_algorithm(cfg):
    return build_from_cfg(cfg, AlgorithmReg)


def build_trainer(cfg):
    return build_from_cfg(cfg, TrainerReg)
