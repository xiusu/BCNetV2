import logging
import torch
import torch.nn as nn
from ..models import Choice
from .builder import build_sampler, build_searcher

from .base import BaseAlgorithm
from ..utils import AlgorithmReg

logger = logging.getLogger()

@AlgorithmReg.register_module('autoslim')
class AutoSlimAlgorithm(BaseAlgorithm):

    def __init__(self, sampler, searcher, channel_bins=8, min_channel_bins=1):
        super(AutoSlimAlgorithm, self).__init__()
        self.sampler = build_sampler(sampler)
        if 'channel_bins' not in searcher:
            searcher['channel_bins'] = channel_bins
        if 'min_channel_bins' not in searcher:
            searcher['min_channel_bins'] = min_channel_bins
        self.searcher = build_searcher(searcher)
        self.channel_bins = channel_bins
        self.min_channel_bins = min_channel_bins

    def sample(self, model):
        choices = model.module.get_channel_choices(self.channel_bins, self.min_channel_bins)
        subnet = self.sampler.gen_subnet(choices)
        model.module.set_channel_choices(subnet, self.channel_bins, self.min_channel_bins)


    def search(self, model, evaluator, train_loader, val_loader):
        #model, choice_modules, evaluator, train_loader, val_loader, BCNet_sampler
        self.searcher.search(model, choice_modules = None, evaluator = evaluator, train_loader = train_loader, val_loader = val_loader)

