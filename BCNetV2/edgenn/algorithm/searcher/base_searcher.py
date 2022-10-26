import logging
from abc import abstractmethod
import sys
from torch import distributed as dist
import torch
import torch.nn as nn
import time
import os
from ...utils import BaseClass
from ..utils.get_flops import get_flops

logger = logging.getLogger()

class BaseSearcher(BaseClass):
    def __init__(self,
                 search_num=1,
                 flops_limit=-1,
                 input_shape=[3, 224, 224], 
                 **kwargs):
        super(BaseSearcher, self).__init__()
        self.search_num = search_num
        self.flops_limit = flops_limit
        self.input_shape = input_shape
        self.rank = dist.get_rank()

        self.epoch = 0
        self.rank = dist.get_rank()

    def eval_subnet(self, candidate, model, choice_modules, evaluator, train_loader, val_loader):
        assert len(choice_modules) == len(candidate)
        for idx, mod in zip(candidate, choice_modules):
            mod.set_sub(idx)
        
        score = evaluator.eval(model, train_loader, val_loader)
        
        return score

    @abstractmethod
    def gen_subnet(self, model, choice_modules):
        pass

    def _broadcast_subnet(self, subnet):
        subnet = torch.tensor(subnet, dtype=torch.int32, device='cuda')
        dist.broadcast(subnet, src=0)
        return subnet.tolist()

    def search(self, model, choice_modules, evaluator, train_loader, val_loader):
        score_map = {}
        while len(score_map) < self.search_num:
            subnet = self.gen_subnet(model, choice_modules)
            subnet = self._broadcast_subnet(subnet)
            subnet = tuple(subnet)
            if subnet in score_map:
                continue
            assert len(subnet) == len(choice_modules)
            for idx, m in zip(subnet, choice_modules):
                m.set_sub(idx)
            # check flops
            if self.flops_limit is not None and self.flops_limit != -1:
                flops = get_flops(model, input_shape=self.input_shape)
                if flops > self.flops_limit:
                    continue

            # calc score
            top1, top5 = self.eval_subnet(subnet, model, choice_modules, evaluator, train_loader, val_loader)
            score_map[subnet] = top1
            logger.info(f'Eval[{len(score_map)}/{self.search_num}] {subnet} score {top1}')

        # get topk
        top_k = list(score_map.keys())
        top_k.sort(key=lambda x: score_map[x], reverse=True)
        top_k = top_k[:10]

        for i, top in enumerate(top_k):
            logger.info(f'top {i} choice {top} score {score_map[top]}')


