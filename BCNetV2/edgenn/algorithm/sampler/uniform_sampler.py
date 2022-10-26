import sys
import random
import numpy as np
import torch
from .base_sampler import BaseSampler
from ..builder import SamplerReg
from ..utils import get_flops


@SamplerReg.register_module('uniform')
class UniformSampler(BaseSampler):
    '''Uniform Sampling.
    '''
    def __init__(self, flops_min=0, flops_max=sys.maxsize, max_times=50, **kwargs):
        super(UniformSampler, self).__init__(**kwargs)
        self.flops_min = flops_min
        self.flops_max = flops_max
        self.max_times = max_times

    def gen_subnet(self, model, choice_modules):
        subnet = [0] * len(choice_modules)
        if self.rank == self.root:
            if self.flops_min != 0 or self.flops_max != sys.maxsize:
                for i in range(self.max_times):
                    subnet = [np.random.randint(len(mod)) for mod in choice_modules]
                    for choice, sub in zip(choice_modules, subnet):
                        choice.set_sub(sub)
                    cur_flops = get_flops(model)
                    if self.flops_min <= cur_flops and cur_flops <= self.flops_max:
                        #print(f"find subnet {subnet} {cur_flops} in times {i}")
                        break
            else:
                subnet = [np.random.randint(len(mod)) for mod in choice_modules]
        subnet = self._broadcast_subnet(subnet)
        return subnet
