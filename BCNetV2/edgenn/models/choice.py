import torch
import torch.nn as nn


class Choice(nn.Module):
    def __init__(self):
        super(Choice, self).__init__()
        self.sub = None
        self.fixed = False

    def set_sub(self, sub):
        assert self.fixed == False
        self.sub = sub

    def fix_sub(self, sub):
        self.set_sub(sub)
        self.fixed = True
