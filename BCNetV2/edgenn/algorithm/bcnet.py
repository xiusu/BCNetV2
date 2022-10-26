import logging
import re
from numpy.core.records import record
import torch
import torch.nn as nn
from ..models import Choice
from .builder import build_sampler, build_searcher
from heapq import *
from .base import BaseAlgorithm
from ..utils import AlgorithmReg
import torch.distributed as dist
logger = logging.getLogger()

class LossRecord(object):
    def __init__(self, loss, subnet_l, subnet_r):
        super().__init__()
        self.loss = loss
        self.subnet_l = subnet_l
        self.subnet_r = subnet_r
    def __lt__(self, other):
        return self.loss > other.loss

@AlgorithmReg.register_module('bcnet')
class BCNetAlgorithm(BaseAlgorithm):

    def __init__(self, sampler, searcher, channel_bins=8, min_channel_bins=1, loss_rec_num=100, use_complementary=False):
        super(BCNetAlgorithm, self).__init__()
        self.sampler = build_sampler(sampler)
        if 'channel_bins' not in searcher:
            searcher['channel_bins'] = channel_bins
        if 'min_channel_bins' not in searcher:
            searcher['min_channel_bins'] = min_channel_bins
        self.searcher = build_searcher(searcher)
        
        self.channel_bins = channel_bins
        self.min_channel_bins = min_channel_bins
        self.subnet_left = None
        self.subnet_right = None
        
        self.loss_rec = []
        self.loss_rec_num = loss_rec_num

    def sample(self, model, left=True):
        # print(f'subnet_left: {self.subnet_left}, subnet_right: {self.subnet_right}')
        if left:
            choices = model.module.get_channel_choices(self.channel_bins, self.min_channel_bins)
            self.subnet_left, self.subnet_right = self.sampler.gen_subnet(choices)
            model.module.set_channel_choices(self.subnet_left, self.channel_bins, self.min_channel_bins)
        else:
            model.module.set_channel_choices(self.subnet_right, self.channel_bins, self.min_channel_bins)

    def search(self, model, evaluator, train_loader, val_loader, **kwargs):
        choices = model.module.get_channel_choices(self.channel_bins, self.min_channel_bins)
        loss_rec_list = [(record.loss, (record.subnet_l, record.subnet_r)) for record in self.loss_rec]
        self.searcher.search(model, choices, evaluator, train_loader, val_loader, loss_rec=loss_rec_list, **kwargs)

    def record_loss(self, loss):
        loss_tensor = torch.tensor([loss], dtype=torch.float32, device='cuda')
        dist.all_reduce(loss_tensor)
        avg_loss = loss_tensor[0] / dist.get_world_size()
        rec = LossRecord(avg_loss.item(), self.subnet_left, self.subnet_right)
        if len(self.loss_rec) < self.loss_rec_num:
            heappush(self.loss_rec, rec)
        else:
            heappushpop(self.loss_rec, rec)