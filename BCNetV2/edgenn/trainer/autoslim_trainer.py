import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from ..utils import TrainerReg, build_algorithm, build_evaluator

@TrainerReg.register_module('autoslim')
class AutoSlimTrainer(BaseTrainer):

    def __init__(self, stage, n=4, inplace_distillation=True, T=3, algorithm=None, evaluator=None):
        super(AutoSlimTrainer, self).__init__()
        self.stage = stage
        if algorithm is not None:
            self.alg = build_algorithm(algorithm)
        if evaluator is not None:
            self.evaluator = build_evaluator(evaluator)
        self.inplace_distillation = inplace_distillation
        self.n = n
        if inplace_distillation is True:
            self.kd_criterion = nn.KLDivLoss(reduction='batchmean')
            self.T = T

    def forward(self, model, inputs, targets):
        if 'supernet' in self.stage:
            if self.inplace_distillation is True:
                choices = model.module.get_channel_choices(self.alg.channel_bins, self.alg.min_channel_bins)
                max_subnet = [x[1] for x in choices]
                min_subnet = [x[0] for x in choices]
                model.module.set_channel_choices(max_subnet, self.alg.channel_bins, self.alg.min_channel_bins)
                output, loss = model(inputs, targets)
                loss.backward()
                loss = loss.item() * torch.ones([1], device=loss.device, requires_grad=True)  # fake loss for verbose

                teacher_logits = output.detach()
                model.module.set_channel_choices(min_subnet, self.alg.channel_bins, self.alg.min_channel_bins)
                student_logits, _ = model(inputs, targets)
                kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
                kd_loss.backward()
                for _ in range(self.n - 2):
                    self.alg.sample(model)
                    student_logits, _ = model(inputs, targets)
                    kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
                    kd_loss.backward()
            else:
                self.alg.sample(model)
                output, loss = model(inputs, targets)
        else:
            output, loss = model(inputs, targets)
        return output, loss

    def search(self, model, train_loader, val_loader):
        if 'search' in self.stage:
            self.alg.search(model, self.evaluator, train_loader, val_loader)
            # self.alg.search(model, train_loader, val_loader)

    def _compute_kd_loss(self, teacher_logits, student_logits):
        p = F.log_softmax(student_logits / self.T, dim=1)
        q = F.softmax(teacher_logits / self.T, dim=1)
        loss = self.kd_criterion(p, q)
        return loss

