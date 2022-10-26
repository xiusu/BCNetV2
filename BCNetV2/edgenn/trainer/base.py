from abc import abstractmethod
from ..utils import BaseClass


class BaseTrainer(BaseClass):
    def __init__(self):
        super(BaseTrainer, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def search(self):
        pass

