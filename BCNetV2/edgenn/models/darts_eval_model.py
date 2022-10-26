import torch
import torch.nn as nn

from ..utils import ModelReg
from .builder import build_backbone, build_loss

from .choice import Choice

@ModelReg.register_module('dartsevalmodel')
class DARTSEvalModel(nn.Module):
    def __init__(self, backbone, loss, drop_path_prob, auxiliary_weight):
        super(DARTSEvalModel, self).__init__()
        self.backbone = build_backbone(backbone)
        self.loss = build_loss(loss)
        self.drop_path_prob = drop_path_prob
        self.auxiliary_weight = auxiliary_weight

    def update_drop_path_prob(self, epoch, max_epochs):
        self.backbone.drop_path_prob = self.drop_path_prob * epoch / max_epochs

    def forward(self, images, target):
        output, output_aux = self.backbone(images)
        loss = self.loss(output, target)
        if output_aux is not None:
            loss_aux = self.loss(output_aux, target)
            loss += self.auxiliary_weight * loss_aux
        return output, loss
