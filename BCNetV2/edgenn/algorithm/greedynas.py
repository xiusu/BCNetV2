import logging
import torch
import torch.nn as nn
from ..models import Choice
from .builder import build_sampler, build_searcher
from .spos import SPOSAlgorithm 
from ..utils import AlgorithmReg

logger = logging.getLogger()

@AlgorithmReg.register_module('greedynas')
class GreedyNASAlgorithm(SPOSAlgorithm):

    def __init__(self, sampler, searcher):
        super(GreedyNASAlgorithm, self).__init__(sampler, searcher)

    def sample(self, model, train_evaluator, val_loader):
        eval_fn = lambda : train_evaluator.eval(model, val_loader)
        choice_modules = self.get_choice_modules(model)
        subnet = self.sampler.gen_subnet(choice_modules, eval_fn)
        for choice, sub in zip(choice_modules, subnet):
            choice.set_sub(sub)

    def search(self, model, evaluator, train_loader, val_loader, **kwargs):
        choice_modules = self.get_choice_modules(model)

        # get population from candidate pool
        population = [tuple(x[0]) for x in self.sampler.cand[:self.searcher.population_num]]

        self.searcher.search(model, choice_modules, evaluator, train_loader, val_loader, population=population, **kwargs)
