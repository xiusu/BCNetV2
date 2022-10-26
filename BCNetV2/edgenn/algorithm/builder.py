from ..utils import Registry, build_from_cfg

SamplerReg = Registry('sampler')
SearcherReg = Registry('searcher')


def build_sampler(cfg):
    return build_from_cfg(cfg, SamplerReg)


def build_searcher(cfg):
    return build_from_cfg(cfg, SearcherReg)
