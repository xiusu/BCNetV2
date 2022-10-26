import torch
import torch.nn as nn
from collections import OrderedDict
from .builder import build_teacher, build_loss

from ..utils import KDReg

@KDReg.register_module()
class KD():

    def __init__(self, teacher, student_loss, loss):
        super(KD, self).__init__()
        self.teacher = build_teacher(teacher).cuda()
        self.student_loss = student_loss
        self.kdloss = build_loss(loss).cuda()
        self.kdloss.register_teacher_logits(self.teacher.loss)

    def register_student_logits(self, student_loss_module):
        self.kdloss.register_student_logits(student_loss_module)

    def forward_teacher(self, *args, **kwargs):
        self.teacher.eval()
        with torch.no_grad():
            self.teacher(*args, **kwargs)

    def get_loss_value(self):
        return self.kdloss.teacher_loss.detach(), self.kdloss.student_loss.detach(), self.kdloss.kl_loss.detach()
