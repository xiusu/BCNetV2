import logging
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import BaseEvaluator
from ..algorithm.utils.get_flops import get_flops
from ..utils import EvaluatorReg

logger = logging.getLogger()


@EvaluatorReg.register_module('train_evaluator')
class TrainEvaluator(BaseEvaluator):
    def __init__(self, update_data_freq=-1, metric='loss'):
        super(TrainEvaluator, self).__init__()
        self._ignored_keys = ['_val_loader', '_val_iterator', '_images', '_targets']
        self.cur_iter = 0
        self.update_data_freq = update_data_freq
        self.metric = metric
        assert self.metric in ['loss', 'ACC'], f'Evaluation metric {metric} not implemented.'

        self._val_loader = None
        self._val_iterator = None
        self._images = None
        self._targets = None
        #if metric == 'CrossEntropyLoss':
        #    self.metric = nn.CrossEntropyLoss()
        #elif metric == 'ACC':
        #    self.metric = self._accuracy
        #else:
        #    raise NotImplementedError(f'Evaluation metric {metric} not implemented.')

    def _accuracy(self, output, target, topk=1):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            batch_size = target.size(0)
    
            _, pred = output.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            correct_k = correct[:topk].reshape(-1).float().sum(0)
            res = correct_k * 100. / batch_size
            return res

    def eval(self, model, val_loader):
        if self._val_loader != val_loader:
           self._images = None
           self._targets = None
           self._val_loader = val_loader
           self._val_iterator = iter(self._val_loader)

        if self._images is None or \
           (self.update_data_freq != -1 and self.cur_iter % self.update_data_freq == 0):
            # generate new batch of data
            try:
                images, target = next(self._val_iterator)
            except StopIteration:
                self._val_iterator = iter(self._val_loader)
                images, target = next(self._val_iterator)
            self._images, self._target = images.cuda(), target.cuda()

        with torch.no_grad():
            # compute output
            output, loss = model(self._images, self._target)
            if self.metric == 'loss':
                score = -loss.item()  # lower is better
            elif self.metric == 'ACC':
                score = self._accuracy(output, target)

            score = torch.tensor([score], dtype=torch.float32, device='cuda')
            dist.all_reduce(score)
            score = score.item() / dist.get_world_size()

        flops = get_flops(model)  # cost 0.02~0.03 s
        self.cur_iter += 1
        return score, flops

