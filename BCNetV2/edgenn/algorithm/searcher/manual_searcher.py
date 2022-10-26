from .base_searcher import BaseSearcher
import random
from ..builder import SearcherReg
import logging

logger = logging.getLogger()

@SearcherReg.register_module('manual')
class ManualSearcher(BaseSearcher):
    def __init__(self, subnet_path, **kwargs):
        super(ManualSearcher, self).__init__(**kwargs)
        subnets = open(subnet_path, 'r').read().split('\n')
        self.subnets = [eval(x) for x in subnets if x != '']
        self.search_num = len(self.subnets)
        logger.info(f'[ManualSearcher] number of subnets: {self.search_num}')
        self.subnet_idx = 0

    
    def gen_subnet(self, model, choice_modules):
        subnet = self.subnets[self.subnet_idx]
        self.subnet_idx += 1
        assert len(subnet) == len(choice_modules)
        return subnet
    
