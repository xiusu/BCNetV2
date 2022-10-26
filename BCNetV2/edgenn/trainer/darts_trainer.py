import torch
import torch.nn.functional as F

from ..utils import TrainerReg, build_algorithm, build_evaluator
from .base import BaseTrainer


@TrainerReg.register_module('darts')
class DARTSTrainer(BaseTrainer):

    def __init__(self, stage, warmup_epoch=0, early_stop=None, algorithm=None):
        super(DARTSTrainer, self).__init__()
        self._ignored_keys = ['val_loader', 'val_iter']
        self.stage = stage
        self.warmup_epoch = warmup_epoch
        self.epoch = None

        if algorithm is not None:
            self.alg = build_algorithm(algorithm)
        self.val_loader = None
        self.val_iter = None
        self.optimizer = None

        self.skip_in_norm_cell = None
        self.arch_rank_stable_epoch = None
        if early_stop is not None:
            if 'skip_in_norm_cell' in early_stop:
                self.skip_in_norm_cell = early_stop.skip_in_norm_cell
            if 'arch_rank_stable_epoch' in early_stop:
                self.arch_rank_stable_epoch = early_stop.arch_rank_stable_epoch
                self.early_stop_same_rank_cnt = 0
                self.early_stop_rank_list = [[],[]]

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_val_loader(self, val_loader):
        self.val_loader = val_loader
        self.val_iter = iter(self.val_loader)

    def next_val(self):
        try:
            inputs, target = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, target = next(self.val_iter)
        return inputs, target

    def forward(self, model, inputs, target):
        if 'supernet' in self.stage:
            if self.epoch is None or self.epoch >= self.warmup_epoch:
                inputs_val, target_val = self.next_val()
                self.alg.step(model, self.optimizer, inputs, target, inputs_val, target_val)
        output, loss = model(inputs, target)
        return output, loss

    def search(self, model, *args, **kwargs):
        if 'search' in self.stage:
            print(model.module.search())

    def early_stop(self, model):
        if 'supernet' in self.stage:
            if self.skip_in_norm_cell is not None:
                skip_cnt = 0
                this_arch = model.module.search()
                for op in this_arch.normal:
                    if op[0] == 'skip_connect':
                        skip_cnt += 1
                if skip_cnt >= self.skip_in_norm_cell:
                    return True
            if self.arch_rank_stable_epoch is not None:
                cur_rank_list = [self._get_learnable_arch_rank(model.module.arch_parameters()[0]),
                                 self._get_learnable_arch_rank(model.module.arch_parameters()[1])]
                if cur_rank_list == self.early_stop_rank_list:
                    self.early_stop_same_rank_cnt += 1
                else:
                    self.early_stop_rank_list = cur_rank_list
                    self.early_stop_same_rank_cnt = 1

                if self.early_stop_same_rank_cnt >= self.arch_rank_stable_epoch:
                    return True
            return False
        else:
            return False
            
    def _get_learnable_arch_rank(self, params):
        rank_list = []
        alpha = params[:,4:].detach().clone().cpu()

        for w in alpha:
            #_, rank = torch.sort(w, stable=True)
            _, rank = torch.sort(w)
            rank_list.append(rank.tolist())
        return rank_list
