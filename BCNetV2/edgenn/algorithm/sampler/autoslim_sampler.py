import sys
import random
import numpy as np
import torch
from .base_sampler import BaseSampler
from ..builder import SamplerReg
from ..utils import get_flops

@SamplerReg.register_module('autoslim')
class AutoSlimSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(AutoSlimSampler, self).__init__(**kwargs)

    def gen_subnet(self, choice_list):
        if max([x[0] for x in choice_list]) > 1:
            subnet = np.random.randint([x[0] for x in choice_list], [x[1] + 1 - x[0] for x in choice_list])
        else:
            subnet = np.random.randint([x[0] for x in choice_list], [x[1] + 1 for x in choice_list])
        subnet = self._broadcast_subnet(subnet)
        return subnet
