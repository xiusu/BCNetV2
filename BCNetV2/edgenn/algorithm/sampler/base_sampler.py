import torch
import sys
from torch import distributed as dist
from abc import abstractmethod
from ...utils import BaseClass


class BaseSampler(BaseClass):
    '''Base searcher for NAS searching stage.
    '''
    def __init__(self, root=0):
        super(BaseSampler, self).__init__()
        self.root = 0
        self.rank = dist.get_rank()

    @abstractmethod
    def gen_subnet(self, choice_modules):
        '''Generate subnet list.
        Must be implemented in sub-class.
        And do not forget to broadcast_subnet in this method.
        Args:
            choice_modules: a list of choice modules for generating subnet
        Returns:
            Generated subnet list.
        '''
        pass

    def _broadcast_subnet(self, subnet=None):
        '''Broadcast subnet list to all distributed process.
        Args:
            subnet: subnet list
        Returns:
            broadcasted subnet list
        '''
        subnet = torch.tensor(subnet, dtype=torch.int32, device='cuda')
        dist.broadcast(subnet, src=self.root)
        subnet = subnet.tolist()
        return subnet

