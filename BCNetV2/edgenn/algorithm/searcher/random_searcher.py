from .base_searcher import BaseSearcher
import random
from ..builder import SearcherReg

@SearcherReg.register_module('random')
class RandomSearcher(BaseSearcher):
    def __init__(self, **kwargs):
        super(RandomSearcher, self).__init__(**kwargs)

    
    def gen_subnet(self, model, choice):
        subnet = []
        for layer in choice:
            subnet.append(random.randint(0, len(layer) - 1))
        return subnet
    
