from .random_sampler import RandomSampler
from .evolution.evolution_sampler import EvolutionSampler
from .greedy.greedy_sample import Greedy_sampler
from .evolution.sense_evolution_sampler import SenseEvolutionSampler
from .sense_sampler import SenseSampler


#def build_sampler(cfg, model, tester, net_cfg, **kwargs):
def build_sampler(cfg, model, tester,net_cfg, **kwargs):
    sampler_type = cfg.get('type', 'evolution')
    kwargs2 = cfg.get('kwargs', {})
    kwargs = dict(**kwargs2, **kwargs)
    if sampler_type == 'evolution':
        return EvolutionSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    elif sampler_type == 'greedy_sample':
        return Greedy_sampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    elif sampler_type == 'random':
        return RandomSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    elif sampler_type == 'sense':
        return SenseSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    elif sampler_type == 'sense_evolution':
        return SenseEvolutionSampler(model=model, tester=tester, net_cfg=net_cfg, **kwargs)
    else:
        raise NotImplementedError(f'Sampler type {sampler_type} not implemented.')
