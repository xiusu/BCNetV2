from core.sampler.base_sampler import BaseSampler
from core.searcher.sense_searcher import Candidate
import pickle


class SenseSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.candidate = None
        self.cand_idx = 0
        assert 'cand_path' in kwargs  # '../generalNAS_exp/imagenet_mbnet/arch/iter_150000_arch.info'
        self.candidate_path = kwargs.pop('cand_path')
        self.load_candidate(self.candidate_path)  # TODO: remove hard code
        #self.cand_idx = len(self.candidate) - 1 - 100

    def load_candidate(self, pickle_path):
        self.candidate = pickle.load(open(pickle_path, 'rb'))['cands']

    def generate_subnet(self):
        subnet = self.candidate[self.cand_idx][0]
        #self.cand_idx += len(self.candidate) // 100
        self.cand_idx += 1
        if self.cand_idx > len(self.candidate) - 1:
            self.cand_idx = len(self.candidate) - 1
        return subnet
