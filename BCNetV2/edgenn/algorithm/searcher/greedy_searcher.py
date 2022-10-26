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

from .base_searcher import BaseSearcher
import random
from ..builder import SearcherReg
from ..utils.get_flops import get_flops

logger = logging.getLogger()

@SearcherReg.register_module('greedy')
class GreedySearcher(BaseSearcher):
    '''GreedySearcher in AutoSlim.
    '''
    def __init__(self,
                 target_flops,
                 reset_batch_size=None,
                 channel_bins=12,
                 min_channel_bins=1,
                 **kwargs):
        super(GreedySearcher, self).__init__(**kwargs)
        self.target_flops = target_flops
        self.reset_batch_size = reset_batch_size
        self.channel_bins = channel_bins
        self.min_channel_bins = min_channel_bins
    
    def search(self, model, evaluator, train_loader, val_loader):
        if self.reset_batch_size is not None:
            # TODO: DANGEROUS! May not compatible for all frameworks
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset, batch_size=self.reset_batch_size, shuffle=False,
                num_workers=4, pin_memory=True, sampler=val_loader.sampler)

        target_flops = sorted(self.target_flops, reverse=True)
        result_choices = []
        result_flops = []

        choices = model.module.get_channel_choices(self.channel_bins, self.min_channel_bins)
        max_subnet = [x[1] for x in choices]
        min_subnet = [x[0] for x in choices]

        subnet = max_subnet
        model.module.set_channel_choices(subnet, self.channel_bins, self.min_channel_bins)
        flops = get_flops(model)
        for target in target_flops:
            if flops <= target:
                result_choices.append(subnet)
                result_flops.append(flops)
                logger.info(f'Find model {subnet} flops {flops} <= {target}')
                continue
            while flops > target:
                # search which layer needs to shrink
                best_score = None
                best_subnet = None
                for i in range(len(choices)):
                    new_subnet = subnet.copy()
                    new_subnet[i] -= 1
                    if min_subnet[i] <= new_subnet[i] <= max_subnet[i]:
                        # subnet is valid
                        model.module.set_channel_choices(new_subnet, self.channel_bins, self.min_channel_bins)
                        score, _ = evaluator.eval(model, train_loader, val_loader)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_subnet = new_subnet
                if best_subnet is None:
                    raise RuntimeError('Cannot find any valid model, check your configurations.')
                subnet = best_subnet
                model.module.set_channel_choices(subnet, self.channel_bins, self.min_channel_bins)
                flops = get_flops(model)
                logger.info(f'Greedy find model: {best_subnet} score: {best_score} FLOPs: {flops}')

            result_choices.append(subnet)
            result_flops.append(flops)
            logger.info(f'Find model {subnet} flops {flops} <= {target}')

        logger.info('Search models done.')
        for choice, flops in zip(result_choices, result_flops):
            logger.info(f'Find model {choice} FLOPs {flops}')

        return result_choices, result_flops


