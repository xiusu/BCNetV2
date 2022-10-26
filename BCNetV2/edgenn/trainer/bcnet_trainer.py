import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from ..utils import TrainerReg, build_algorithm, build_evaluator

@TrainerReg.register_module('bcnet')
class BCNetTrainer(BaseTrainer):

    def __init__(self, stage, algorithm=None, evaluator=None):
        super(BCNetTrainer, self).__init__()
        self.stage = stage
        if algorithm is not None:
            self.alg = build_algorithm(algorithm)
        if evaluator is not None:
            self.evaluator = build_evaluator(evaluator)

    def forward(self, model, inputs, targets):
        if 'supernet' in self.stage:
           self.alg.sample(model, left=True)
           output, loss1 = model(inputs, targets)
           (loss1 * 0.5).backward()
           self.alg.sample(model, left=False)
           output, loss2 = model(inputs, targets)
           (loss2 * 0.5).backward()
        #    print(f'left loss: {loss1}, right loss: {loss2}')
           loss = (loss1.item() + loss2.item()) / 2 * torch.ones([1], device=loss1.device, requires_grad=True)  # fake loss for verbose
           self.alg.record_loss(loss.item())
        else:
            output, loss = model(inputs, targets)
        return output, loss

    def search(self, model, train_loader, val_loader, **kwargs):
        if 'search' in self.stage:
            self.alg.search(model, self.evaluator, train_loader, val_loader, **kwargs)


