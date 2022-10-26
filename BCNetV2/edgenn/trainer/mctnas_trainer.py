from ..utils import TrainerReg, build_algorithm, build_evaluator
from .base import BaseTrainer


@TrainerReg.register_module('mctnas')
class MCTNASTrainer(BaseTrainer):

    def __init__(self, stage, algorithm=None, evaluator=None, train_evaluator=None):
        super(MCTNASTrainer, self).__init__()
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
            self.alg.sample(model)
        output, loss = model(inputs, targets)
        self.alg.update_reward(loss.item())
        return output, loss

    def search(self, model, train_loader, val_loader):
        if 'search' in self.stage:
            self.alg.search(model, self.evaluator, self.train_evaluator, train_loader, val_loader)

    def set_val_loader(self, val_loader):
        '''some algorithms need to use additional validation data'''
        self.val_loader = val_loader

