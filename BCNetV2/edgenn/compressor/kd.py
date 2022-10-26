from .base import BaseCompressor

from ..utils import CompressorReg
from ..utils import build_kd

@CompressorReg.register_module()
class KDCompressor(BaseCompressor):

    def __init__(self, kd):
        self.kd = build_kd(kd)
        self.model = None
        self.loss = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

    def convert_model(self, model):
        self.model = model
        return self.model

    def convert_loss(self, loss):
        self.kd.register_student_logits(loss)
        self.loss = self.kd.kdloss
        return self.loss

    def before_run(self):
        pass

    def after_run(self):
        pass

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        loss = self.kd.get_loss_value()
        print(f'teacher loss {loss[0]}, student loss {loss[1]}, kl loss {loss[2]}')

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
        self.kd.forward_teacher(*args, **kwargs)

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
