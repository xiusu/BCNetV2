from .base_searcher import BaseSearcher
import logging
import random
from ..builder import SearcherReg
from ..utils.get_flops import get_spos_flops

logger = logging.getLogger()

@SearcherReg.register_module('mct')
class MCTSearcher(BaseSearcher):
    def __init__(self, n_thrd=10, flops_min=0, tau=1., **kwargs):
        super(MCTSearcher, self).__init__(**kwargs)
        self.n_thrd = n_thrd
        self.flops_min = flops_min
        self.tau = tau
        self.tree = None
        self.tilde_L = None
        self.train_evaluator = None
    
    def set_mct(self, tree):
        self.tree = tree
        self.tree.tau = self.tau

    def set_tilde_L(self, tilde_L):
        self.tilde_L = tilde_L

    def set_train_evaluator(self, train_evaluator):
        self.train_evaluator = train_evaluator

    def _eval_hierarchical_subnet(self, candidate, model, choice_modules, evaluator, val_loader, flops=False):
        assert len(choice_modules) == len(candidate)
        candidate = self._broadcast_subnet(candidate)
        for idx, mod in zip(candidate, choice_modules):
            mod.set_sub(idx)
        if flops:
            return get_spos_flops(model, candidate, input_shape=self.input_shape)
        score = evaluator.eval(model, val_loader)
        return score

    def gen_subnet(self, model, choice_modules, val_loader):
        if self.tree is None:
            raise RuntimeError(f'MC tree must be set before search.')
        eval_fn = lambda subnet, flops=False: self._eval_hierarchical_subnet(subnet, model, choice_modules, self.train_evaluator, val_loader, flops)
        return self.tree.hierarchical_sample(self.n_thrd, eval_fn, (self.flops_min, self.flops_limit), self.tilde_L)
    
    def search(self, model, choice_modules, evaluator, train_loader, val_loader):
        score_map = {}
        while len(score_map) < self.search_num:
            subnet = self.gen_subnet(model, choice_modules, val_loader)
            subnet = self._broadcast_subnet(subnet)
            subnet = tuple(subnet)
            if subnet in score_map:
                continue
            # check flops
            assert len(subnet) == len(choice_modules)
            for idx, m in zip(subnet, choice_modules):
                m.set_sub(idx)
            flops = get_spos_flops(model, subnet, input_shape=self.input_shape)
            if self.flops_min > flops or (self.flops_limit != -1 and flops > self.flops_limit):
                # remove the leaf nodes
                self.tree.remove_leaf_node(subnet)
                continue

            # calc score
            top1, top5 = self.eval_subnet(subnet, model, choice_modules, evaluator, train_loader, val_loader)
            self.tree.remove_leaf_node(subnet)  # remove evaluated subnet
            score_map[subnet] = top1
            logger.info(f'[{len(score_map)}/{self.search_num}] subnet: {subnet}, FLOPs: {flops}, score: {top1}')

        # get topk
        top_k = list(score_map.keys())
        top_k.sort(key=lambda x: score_map[x], reverse=True)
        top_k = top_k[:10]

        for i, top in enumerate(top_k):
            logger.info(f'top {i} choice {top} score {score_map[top]}')

