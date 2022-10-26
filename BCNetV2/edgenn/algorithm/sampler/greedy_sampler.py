from ..utils.candidate import Candidate
from ..utils import get_flops
from .base_sampler import BaseSampler
import os
import random
import torch
import math
import numpy as np
import logging

logger = logging.getLogger()

from ..builder import SamplerReg


@SamplerReg.register_module('greedy')
class GreedySampler(BaseSampler):
    '''GreedySampler with multi-path and candidate pool.
    Every generated path will be evaluated by a small val set.
    And paths will be pushed into a list ranked by val score.
    The searcher will generate m paths and evaluate them, then select top-k for training. 

    If FLOPs constraint is setted, only paths statisfied constraint will be pushed into pool.

    Attributes:
        flops_constraint: FLOPs constraint
    '''
    def __init__(self, **kwargs):
        super(GreedySampler, self).__init__()
        # flops
        self.flops_constraint = kwargs.get('flops_constraint', -1)
        self.input_shape = kwargs.get('input_shape', [3, 224, 224])
        # candidate pool
        self.pool_size = kwargs.get('pool_size', 1000)

        # multi-path
        multi_path = kwargs.get('multi_path', {})
        self.sample_num = multi_path.get('sample_num', 1)
        self.topk = multi_path.get('topk', 1)

        # p strategy
        p_stg = kwargs.get('p_strategy', {})
        self.p_stg = p_stg.get('type', 'linear')
        self.start_iter = p_stg.get('start_iter', 0)
        self.max_iter = p_stg.get('max_iter', 1)
        self.init_p = p_stg.get('init_p', 0.)
        self.max_p = p_stg.get('max_p', 1.)
        self.cur_p = 0.

        # adaptive stopping
        self.alpha = kwargs.get('alpha', 0.08)
        self.t = kwargs.get('t', 10000)
        self.early_stop = False
        self.prev_cand = []

        # init candidate
        self.cand = Candidate(self.pool_size)

        # subnet topk queue
        self.subnet_topk = []
        self.iter = 0   # optimized iteration count

    def _gen_subnet_uniform(self, choice_modules):
        subnet = [np.random.randint(len(mod)) for mod in choice_modules]
        subnet = self._broadcast_subnet(subnet)
        return subnet
    
    def _gen_subnet_from_pool(self):
        subnet = random.choice(self.cand)[0]
        subnet = self._broadcast_subnet(subnet)
        return subnet
    
    def _eval_subnet(self, choice_modules, subnet, eval_fn):
        '''This function evaluate a subnet using eval_func.
        Then add (subnet, score) into candidate pool. 
        Returns:
            evaluation score
        '''
        # set subnet
        for choice, sub in zip(choice_modules, subnet):
            choice.set_sub(sub)
        score, flops = eval_fn()
        #logger.info(f'evaluated subnet {subnet}, score: {score}, flops: {flops}')
        if self.flops_constraint == -1:
            self.cand._add((subnet, score), iter_num=self.iter)
        else:
            if flops <= self.flops_constraint:
                self.cand._add((subnet, score), flops=flops, iter_num=self.iter)
        return score

    def gen_subnet(self, choice_modules, eval_fn):
        if len(self.subnet_topk) == 0:
            # if subnet queue is empty, generate subnets
            # cal p
            if self.iter > self.max_iter:
                p = self.max_p
            elif self.iter < self.start_iter:
                p = 0   # warmup - do not sample from pool
            elif self.p_stg == 'linear':
                p = (self.max_p - self.init_p) / (self.max_iter - self.start_iter) * (self.iter - self.start_iter)
            elif self.p_stg == 'consine':
                p = (1 - self.init_p) * 0.5 * (1 + math.cos(
                    math.pi * (self.iter - self.start_iter) / (self.max_iter - self.start_iter)
                )) + self.init_p
            self.cur_p = p  # for verbose

            eval_subnets = []
            uniform_num = 0 
            for _ in range(self.sample_num):
                if random.random() > p or len(self.cand) == 0:
                    # uniform sample
                    eval_subnets.append(self._gen_subnet_uniform(choice_modules))
                    uniform_num += 1
                else:
                    # sample from pool
                    eval_subnets.append(self._gen_subnet_from_pool())
                if self.iter < self.start_iter:
                    # warmup stage: disable multi-path sampling
                    self.iter += 1
                    return eval_subnets[0]

            # eval subnets
            eval_results = []
            
            for subnet in eval_subnets:
                eval_results.append([subnet, self._eval_subnet(choice_modules, subnet, eval_fn)])
            eval_results.sort(key=lambda a: a[1], reverse=True)
            self.subnet_topk.extend([x[0] for x in eval_results[:self.topk]]) 
            # logger.info(f'GreedySampler: [{self.iter:>6d}] prob {self.cur_p:.3f} uniform_num {uniform_num}/{self.sample_num} top1_score {eval_results[0][1]:.3f}')
        
        if self.iter % self.t == 0: 
            self.prev_cand = [item[0] for item in self.cand]
        if self.iter % self.t == self.t-1:
            intersect_cand_num = 0
            for item in self.cand:
                if item[0] in self.prev_cand: intersect_cand_num += 1
            pi = 1.0 - intersect_cand_num / len(self.cand)
            logger.info(f'GreedySampler Early Stop: pi {pi:.3f} intersect {intersect_cand_num} cand_size {len(self.cand)}')
            if pi <= self.alpha:
                self.early_stop = True
        self.iter += 1
        return self.subnet_topk.pop(0)    

