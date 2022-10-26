import torch
import torch.nn as nn

from ..utils import ModelReg
from .builder import build_backbone, build_loss


@ModelReg.register_module('sposmodel')
class SPOSModel(nn.Module):
    def __init__(self, backbone, loss):
        super(SPOSModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.loss = build_loss(loss)

    def forward(self, images, target):
        output = self.backbone(images)
        loss = self.loss(output, target)
        return output, loss
