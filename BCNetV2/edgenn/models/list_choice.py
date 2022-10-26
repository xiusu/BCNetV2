import torch
import torch.nn as nn
from .choice import Choice


class ListChoice(Choice):
    def __init__(self, module_list):
        super(ListChoice, self).__init__()
        self.module_list = nn.ModuleList(module_list)

    def __iter__(self):
        for layer in self.module_list:
            yield layer

    def __len__(self):
        return len(self.module_list)

    def forward(self, x):
        if self.sub is not None:
            x = self.module_list[self.sub](x)
        else: 
            x = self.module_list[0](x)
        return x
