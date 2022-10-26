from ..utils import TrainerReg, build_algorithm, build_evaluator
from .base import BaseTrainer


@TrainerReg.register_module('spos')
class SPOSTrainer(BaseTrainer):

    def __init__(self, stage, algorithm=None, evaluator=None, train_evaluator=None):
        super(SPOSTrainer, self).__init__()
        self._ignored_keys = ['val_loader']
        self.stage = stage
        if algorithm is not None:
            self.alg = build_algorithm(algorithm)
        if evaluator is not None:
            self.evaluator = build_evaluator(evaluator)
        if train_evaluator is not None:
            self.train_evaluator = build_evaluator(train_evaluator)
            self.val_loader = None

    def forward(self, model, inputs, targets):
        if 'supernet' in self.stage:
            if hasattr(self, 'train_evaluator'):
                assert self.val_loader is not None, ''
                self.alg.sample(model, self.train_evaluator, self.val_loader)
            else:
                self.alg.sample(model)
        output, loss = model(inputs, targets)
        return output, loss
    
    def early_stop(self):
        if 'greedynas' not in self.alg.__class__.__name__.lower():
            return False
        if 'supernet' not in self.stage:
            return False
        return self.alg.sampler.early_stop

    def search(self, model, train_loader, val_loader, **kwargs):
        if 'search' in self.stage:
            self.alg.search(model, self.evaluator, train_loader, val_loader, **kwargs)

    def set_val_loader(self, val_loader):
        '''some algorithms need to use additional validation data'''
        self.val_loader = val_loader

