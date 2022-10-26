import logging
import torch
import torch.nn as nn
from .spos import SPOSAlgorithm
from ..models import Choice
from .builder import build_sampler, build_searcher

from ..utils import AlgorithmReg

logger = logging.getLogger()


@AlgorithmReg.register_module('mctnas')
class MCTNASAlgorithm(SPOSAlgorithm):

    def __init__(self, sampler, searcher):
        super(MCTNASAlgorithm, self).__init__(sampler, searcher)
        self.last_subnet = None
       
    def sample(self, model):
        choice_modules = self.get_choice_modules(model)
        subnet = self.sampler.gen_subnet(model, choice_modules)
        for choice, sub in zip(choice_modules, subnet):
            choice.set_sub(sub)
        self.last_subnet = subnet

    def search(self, model, evaluator, train_evaluator, train_loader, val_loader):
        if 'mctsearcher' in self.searcher.__class__.__name__.lower():
            self.searcher.set_mct(self.sampler.tree)
            self.searcher.set_tilde_L(self.sampler.tilde_L)
            self.searcher.set_train_evaluator(train_evaluator)
        choice_modules = self.get_choice_modules(model)
        self.searcher.search(model, choice_modules, evaluator, train_loader, val_loader)

    def update_reward(self, loss):
        self.sampler.update_reward(self.last_subnet, loss)

