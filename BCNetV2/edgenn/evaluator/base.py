from abc import abstractmethod
from ..utils import BaseClass


class BaseEvaluator(BaseClass):
    def __init__(self):
        super(BaseEvaluator, self).__init__()

    @abstractmethod
    def eval(self):
        pass

