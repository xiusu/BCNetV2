import sys
import random
import numpy as np
import torch
from .base_sampler import BaseSampler
from ..builder import SamplerReg
from ..utils.get_flops import get_spos_flops
from ..utils.mct import MCTree 

import logging
logger = logging.getLogger()


@SamplerReg.register_module('mct')
class MCTSampler(BaseSampler):
    def __init__(self, flops_min=0, flops_max=sys.maxsize, max_times=100, warmup_iters=0, mct_start_iter=0,
                 beta=0.9, gamma=0.9, C1=1.0, C2=1.0, tau=1, **kwargs):
        super(MCTSampler, self).__init__(**kwargs)
        self.flops_min = flops_min
        self.flops_max = flops_max
        self.max_times = max_times
        self.warmup_iters = warmup_iters
        self.mct_start_iter = mct_start_iter
        self.cur_iter = 0

        self.tree = None    # initialize it later

        self.beta = beta
        self.gamma = gamma
        self.C1 = C1
        self.C2 = C2
        self.tau = tau
        self.tilde_L = 0.    # moving average of losses

    def gen_subnet(self, model, choice_modules):
        if self.tree is None:
            # initialize Monte-Carlo tree
            self.tree = MCTree([len(mod) for mod in choice_modules], self.gamma, self.C1, self.C2, self.tau)
        if self.cur_iter % 100 == 0:
            logger.info(f'MCTSampler: [{self.cur_iter:>6d}] current tree size {self.tree.size}')
        if self.cur_iter == self.mct_start_iter:
            logger.info(f'MCTSampler: start stage 3: sampling using MCTree')

        subnet = [0] * len(choice_modules)
        if self.rank == self.root:
            if self.cur_iter < self.mct_start_iter:
                # uniform sampling
                subnet = [np.random.randint(len(mod)) for mod in choice_modules]
            else:
                # MCTS
                if self.flops_min != 0 or self.flops_max != sys.maxsize:
                    for i in range(self.max_times):
                        subnet = self.tree.sample()

                        for choice, sub in zip(choice_modules, subnet):
                            choice.set_sub(sub)
                        cur_flops = get_spos_flops(model, subnet)
                        if self.flops_min <= cur_flops and cur_flops <= self.flops_max:
                            #print(f"find subnet {subnet} {cur_flops} in times {i}")
                            break
                else:
                    subnet = self.tree.sample()

        subnet = self._broadcast_subnet(subnet)
        self.cur_iter += 1
        return subnet

    def update_reward(self, subnet, loss):
        if self.cur_iter >= self.warmup_iters:
            if self.cur_iter == self.warmup_iters:
                logger.info(f'MCTSampler: start stage 2: building MCTree')
            self.tilde_L = self.beta * self.tilde_L + (1 - self.beta) * loss
            cur_Q = self.tilde_L / loss
            self.tree.update(subnet, cur_Q)  

