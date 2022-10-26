import sys
import random
import numpy as np
import torch
from .autoslim_sampler import AutoSlimSampler
from ..builder import SamplerReg
from ..utils import get_flops

@SamplerReg.register_module('bcnet')
class BCNetSampler(AutoSlimSampler):
    def __init__(self, complementary_sampler=True, **kwargs):
        super(BCNetSampler, self).__init__( **kwargs)
        self.complementary_sampler = complementary_sampler

        self._complementary_switch = False
        self._subnet_l_comp = None
        self._subnet_r_comp = None

    def gen_subnet(self, choice_list):
        # TODO
        if not self._complementary_switch:
            subnet = super(BCNetSampler, self).gen_subnet(choice_list)  # [c1, c2, ...]
            max_channels = [x[1] for x in choice_list]
            min_channels = [x[0] for x in choice_list]
            if max(min_channels) > 1:
                subnet_l = [[0, c] for c in subnet]
                subnet_r = [[max_c - c, max_c] for max_c, min_c, c in zip(max_channels, min_channels, subnet)]
                # print(f'subnet_l: {subnet_l}, subnet_r: {subnet_r}')
                if self.complementary_sampler:
                    self._subnet_l_comp =  [[min(c + 1, max_c - 1), max_c] for max_c, min_c, c in zip(max_channels, min_channels, subnet)]
                    self._subnet_r_comp = [[0, max(1, max_c - c - 1)] for max_c, min_c, c in zip(max_channels, min_channels, subnet)]
                    self._complementary_switch = not self._complementary_switch

            else:
                subnet_l = [[0, c] for c in subnet]
                subnet_r = [[max_c - c, max_c] for max_c, c in zip(max_channels, subnet)]
                if self.complementary_sampler:
                    self._subnet_l_comp =  [[min(c + 1, max_c - 1), max_c] for max_c, c in zip(max_channels, subnet)]
                    self._subnet_r_comp = [[0, max(1, max_c - c - 1)] for max_c, c in zip(max_channels, subnet)]
                    self._complementary_switch = not self._complementary_switch
            return subnet_l, subnet_r
        else:
            assert self._subnet_l_comp is not None
            assert self._subnet_r_comp is not None
            self._complementary_switch = not self._complementary_switch
            return self._subnet_l_comp, self._subnet_r_comp
        

