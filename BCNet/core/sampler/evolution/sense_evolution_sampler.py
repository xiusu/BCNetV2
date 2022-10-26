from core.sampler.evolution.evolution_sampler import EvolutionSampler
from core.searcher.sense_searcher import Candidate
import pickle
import numpy as np
from core.utils.flops import count_flops
import torch


class SenseEvolutionSampler(EvolutionSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.candidate = None
        self.cand_idx = 0
        assert 'cand_path' in kwargs  # '../generalNAS_exp/imagenet_mbnet/arch/iter_150000_arch.info'
        self.candidate_path = kwargs.pop('cand_path')
        self.flops_constrant = kwargs.get('flops_constrant', 500e6)
        self.load_candidate(self.candidate_path)

        # get population
        self.initial_pop = []
        idx = 0
        while len(self.initial_pop) < self.pop_size and idx < len(self.candidate):
            subnet = self.candidate[idx][0]
            flops = count_flops(self.model.net, subnet)
            if self.rank == 0:
                print('subnet: {}, FLOPs: {}'.format(subnet, flops))
            if flops <= self.flops_constrant:
                self.initial_pop.append(subnet)
            idx += 1
        self.initial_pop = np.array(self.initial_pop, dtype=np.int)

        #self.cand_idx = len(self.candidate) - 1 - 100

    def load_candidate(self, pickle_path):
        self.candidate = pickle.load(open(pickle_path, 'rb'))['cands']

    def sample(self):
        super().sample(self.initial_pop)

