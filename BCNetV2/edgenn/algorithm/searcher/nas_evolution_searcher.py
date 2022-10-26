import os
import sys
import time
import glob
import pickle
import torch
import logging
import argparse
import torch
import random
import numpy as np

from abc import abstractmethod
import torch.distributed as dist
from ._evolution_searcher import EvolutionSearcher, choice
from ..builder import SearcherReg
from ..utils.get_flops import get_flops

logger = logging.getLogger()


@SearcherReg.register_module('evolution')
class NASEvolutionSearcher(EvolutionSearcher):
    def __init__(self, **kwargs):
        super(NASEvolutionSearcher, self).__init__(**kwargs)
    
    def gen_subnet(self, model, choice_modules):
        return tuple([choice(len(m)) for m in choice_modules])

    def is_legal(self, candidate, model, choice_modules):
        if candidate in self.score_map:
            return False
        else:
            self.score_map[candidate] = 0.
        if self.flops_limit and self.flops_limit != -1:
            assert len(choice_modules) == len(candidate)
            for idx, mod in zip(candidate, choice_modules):
                mod.set_sub(idx)
            flops = get_flops(model, input_shape=self.input_shape)
            if flops > self.flops_limit:
                return False
            logger.info(f'subnet: {candidate} FLOPs: {flops}')
        return True


