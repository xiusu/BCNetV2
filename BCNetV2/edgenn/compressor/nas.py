from .base import BaseCompressor

from ..utils import CompressorReg
from ..utils import build_nas#, build_pruning, build_kd, build_quant

@CompressorReg.register_module()
class NASCompressor(BaseCompressor):

    def __init__(self, nas):
        self.nas = build_nas(nas)
        self.model = None
        self.loss = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

    def convert_model(self, model):
        self.model = self.nas.convert_model(model)
        return self.model

    def convert_loss(self, loss):
        self.loss = loss
        return self.loss

    def before_run(self):
        pass

    def after_run(self, val_func):
        self.nas.search(val_func, self.model)

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass

    def before_train_epoch(self):
        self.before_epoch()

    def before_val_epoch(self):
        self.before_epoch()

    def after_train_epoch(self):
        self.after_epoch()

    def after_val_epoch(self):
        self.after_epoch()

    def before_train_iter(self, *args, **kwargs):
        self.before_iter()
        self.nas.sample(self.model)

    def before_val_iter(self):
        self.before_iter()

    def after_train_iter(self):
        self.after_iter()

    def after_val_iter(self):
        self.after_iter()

    def before_backward(self):
        pass

    def before_optimizer(self):
        pass
