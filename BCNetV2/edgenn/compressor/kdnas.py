from .base import BaseCompressor

from ..utils import CompressorReg
from ..utils import build_nas, build_kd
from ..utils import get_recur_module, set_recur_module

@CompressorReg.register_module()
class KDNASCompressor(BaseCompressor):

    def __init__(self, kd, nas):
        self.kd = build_kd(kd)
        self.nas = build_nas(nas)
        self.model = None
        self.loss = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

    def convert_modelwithloss(self, model):
        self.model = self.nas.convert_model(model)
        loss_mod = get_recur_module(self.model, self.kd.student_loss)
        self.kd.register_student_logits(loss_mod)
        self.loss = self.kd.kdloss
        set_recur_module(self.model, self.kd.student_loss, self.loss)
        return self.model

    def convert_model(self, model):
        self.model = self.nas.convert_model(model)
        return self.model

    def convert_loss(self, loss):
        self.kd.register_student_logits(loss)
        self.loss = self.kd.kdloss
        return self.loss

    def after_run(self, val_func):
        self.nas.search(val_func, self.model)


    def before_train_iter(self, *args, **kwargs):
        self.before_iter()
        self.kd.forward_teacher(*args, **kwargs)
        self.nas.sample(self.model)

    def after_train_iter(self):
        self.after_iter()
        loss = self.kd.get_loss_value()
        print(f'teacher loss {loss[0]:.3f}, student loss {loss[1]:.3f}, kl loss {loss[2]:.3f}')

    def before_val_iter(self):
        self.before_iter()

    def after_val_iter(self):
        self.after_iter()

    def before_backward(self):
        pass

    def before_optimizer(self):
        pass
