import torch
import torch.nn as nn

from ..utils import ModelReg
from .builder import build_backbone, build_loss

@ModelReg.register_module('dartssearchmodel')
class DARTSSearchModel(nn.Module):
    def __init__(self, backbone, loss, architect=None):
        super(DARTSSearchModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.loss = build_loss(loss)
        if architect is not None:
            self.fix_architect(architect)

    def arch_parameters(self):
        return self.backbone.arch_parameters()

    def forward(self, images, target):
        output = self.backbone(images)
        loss = self.loss(output, target)
        return output, loss

    def search(self):
        return self.backbone.genotype()
