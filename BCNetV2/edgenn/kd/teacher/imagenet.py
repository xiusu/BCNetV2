import torch
import torch.nn as nn
import torchvision.models as models
from ..builder import TeacherReg

@TeacherReg.register_module()
class ImageNetTeacher(nn.Module):

    def __init__(self, arch, pretrain=None):
        super(ImageNetTeacher, self).__init__()
        self.model = models.__dict__[arch]().cuda()
        self.loss = nn.CrossEntropyLoss().cuda()
        self.load_pretrain(pretrain)

    def load_pretrain(self, pretrain):
        checkpoint = torch.load(pretrain, map_location="cpu")
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def forward(self, input, target):
        logits = self.model(input)
        loss = self.loss(logits, target)
        return loss
