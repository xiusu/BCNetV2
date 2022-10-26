class BaseCompressor():

    def convert_model(self, model):
        return model

    def convert_loss(self, loss):
        return loss

    def before_run(self):
        pass

    def after_run(self):
        pass

    def before_epoch(self):
        pass

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

    def before_train_iter(self):
        self.before_iter()

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
