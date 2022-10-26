import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LossReg


# knowledge distillation
@LossReg.register_module()
class KDLoss(nn.Module):
    def __init__(self, T=3, alpha=0.5, eps=1e-10):
        super(KDLoss, self).__init__()
        self.eps = eps
        self.T = T
        self.alpha = alpha
        self.teacher_logits = None
        self.student_logits = None
        self.student_loss_module = None
        self.student_loss = None
        self.teacher_loss = None
        self.kl_loss_module = nn.KLDivLoss(reduction='batchmean')
        self.kl_loss = None

    def register_teacher_logits(self, loss):
        def get_logits_hook(module, input):
            self.teacher_logits = input[0].clone() 
        loss.register_forward_pre_hook(get_logits_hook)
        def get_loss_hook(module, input, output):
            self.teacher_loss = output.clone() 
        loss.register_forward_hook(get_loss_hook)

    def register_student_logits(self, loss):
        def get_logits_hook(module, input):
            self.student_logits = input[0].clone() 
        loss.register_forward_pre_hook(get_logits_hook)
        self.student_loss_module = loss

    def forward(self, *args, **kwargs):
        self.student_loss = self.student_loss_module(*args, **kwargs)

        if self.training:
            p = F.log_softmax(self.student_logits / self.T, dim=1)
            q = F.softmax(self.teacher_logits / self.T, dim=1)
            self.kl_loss = self.kl_loss_module(p, q)
            loss = self.kl_loss * self.alpha + self.student_loss * (1. - self.alpha)
            return loss
        else:
            return self.student_loss


## Selective knowledge distillation
#class SelectiveKDLoss(nn.Module):
#    def __init__(self, T=3, alpha=1.0, eps=1e-10, topk=12000):
#        super(SelectiveKDLoss, self).__init__()
#        self.eps = eps
#        self.T = T
#        self.alpha = alpha
#        self.topk = topk
#
#    def forward(self, pred_logits, gt_logits, labels):
#        p = F.softmax(pred_logits / self.T, dim=1)
#        q = F.softmax(gt_logits / self.T, dim=1)
#        valid_n_all = pred_logits.shape[0]
#        # select the topk probabilities for distillation
#        t_index2 = torch.topk(q, self.topk)[1]
#        t_index1 = np.tile(range(labels.size(0)),
#                           self.topk).reshape(self.topk, -1).T
#        p = p[t_index1, t_index2]
#        q = q[t_index1, t_index2]
#        loss_kl = torch.mean(-torch.dot(q.view(-1), (torch.log(
#            (p + self.eps) / (q + self.eps))).view(-1)))
#        return loss_kl / valid_n_all
