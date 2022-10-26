import logging
import torch
import torch.nn as nn
from .base import BaseAlgorithm
from ..models import Choice
from .builder import build_sampler, build_searcher

from ..utils import AlgorithmReg

logger = logging.getLogger()


@AlgorithmReg.register_module('spos')
class SPOSAlgorithm(BaseAlgorithm):

    def __init__(self, sampler, searcher):
        super(SPOSAlgorithm, self).__init__()
        self.sampler = build_sampler(sampler)
        self.searcher = build_searcher(searcher)

    def get_choice_modules(self, model):
        choice_modules = []
        for mod in model.modules():
            if isinstance(mod, Choice):
                choice_modules.append(mod)
        return choice_modules
        
    def sample(self, model):
        choice_modules = self.get_choice_modules(model)
        subnet = self.sampler.gen_subnet(model, choice_modules)
        for choice, sub in zip(choice_modules, subnet):
            choice.set_sub(sub)

    def search(self, model, evaluator, train_loader, val_loader):
        choice_modules = self.get_choice_modules(model)
        self.searcher.search(model, choice_modules, evaluator, train_loader, val_loader)
