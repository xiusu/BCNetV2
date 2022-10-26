import os
import sys
import time
import glob
import pickle
import torch
import logging
import argparse
import torch
import random
import numpy as np

from abc import abstractmethod
import torch.distributed as dist
import torch.nn.functional as Func
from torch.optim import optimizer
from ._evolution_searcher import EvolutionSearcher
from ..builder import SearcherReg
from ..utils.get_flops import get_flops

logger = logging.getLogger()


@SearcherReg.register_module('bcnet')
class BCNetSearcher(EvolutionSearcher):
    def __init__(self, channel_bins, min_channel_bins, prior_init=True, **kwargs):
        super(BCNetSearcher, self).__init__(**kwargs)
        self.channel_bins = channel_bins
        self.min_channel_bins = min_channel_bins
        self.prior_init = prior_init
        self.prob = None
    

    def gen_subnet(self, model, choice_modules):
        # TODO
        L = len(choice_modules)
        subnet = []
        # if self.rank == 0:
        if max([x[0] for x in choice_modules]) > 1:
            
            for i in range(L):
                subnet.append(np.random.choice(a=np.arange(choice_modules[i][0], choice_modules[i][1]+1), 
                                                p=self.prob[i]))
            subnet = tuple(subnet)
        # else:
        #     if max([x[0] for x in choice_modules]) > 1:
        #         for i in range(L):
        #             subnet.append(0)
        #         subnet = tuple(subnet)


        # subnet = torch.tensor(subnet, dtype=torch.int32, device='cuda')
        # dist.broadcast(subnet, src=0)
        # subnet = subnet.tolist()
        # subnet = tuple(subnet)
            # subnet = np.random.randint([x[0] for x in choice_modules], [x[1] + 1 - x[0] for x in choice_modules])
        # else:
        #     subnet = []
        #     for i in range(L):
        #         subnet.append(np.random.choice(a=np.arange(choice_modules[i][0], choice_modules[i][1]+1), 
        #                                         p=self.prob[i]))
        #     subnet = tuple(subnet)
            # subnet = np.random.randint([x[0] for x in choice_modules], [x[1] + 1 for x in choice_modules]) 
        # subnet = tuple(subnet.tolist())
        print(f'subnet: {subnet}, rank: {self.rank}')
        return subnet

    def is_legal(self, candidate, model, choice_modules):
        if candidate in self.score_map:
            return False
        else:
            self.score_map[candidate] = 0.
        if isinstance(candidate, tuple):
            candidate = list(candidate)
        subnet = [[0, c] for c in candidate]
        if self.flops_limit:
            assert len(choice_modules) == len(candidate)
            model.module.set_channel_choices(subnet, self.channel_bins, self.min_channel_bins)
            flops = get_flops(model, input_shape=self.input_shape)
            logger.info(f'subnet: {candidate} FLOPs: {flops}')
            if flops > self.flops_limit or flops < 0.8 * self.flops_limit:
                return False
        return True

    def eval_subnet(self, candidate, model, choice_modules, evaluator, train_loader, val_loader):
        assert len(choice_modules) == len(candidate)
        if isinstance(candidate, tuple):
            candidate = list(candidate)
        # TODO
        subnet_l = [[0, c] for c in candidate]
        max_channels = [x[1] for x in choice_modules]
        min_channels = [x[0] for x in choice_modules]
        if max(min_channels) > 1:
            subnet_r = [[max_c - c, max_c] for max_c, min_c, c in zip(max_channels, min_channels, candidate)]
        
        else:
            subnet_r = [[max_c - c, max_c] for max_c, c in zip(max_channels, candidate)]
            
        model.module.set_channel_choices(subnet_l, self.channel_bins, self.min_channel_bins)
        
        # if self.flops_limit:
        #     flops = get_flops(model, self.input_shape)
        #     if flops > self.flops_limit: 
        #         return 1.0 / flops

        score1, _ = evaluator.eval(model, train_loader, val_loader)
        # print(f'subnet_l: {subnet_l}, subnet_r: {subnet_r}')
        # print(f'max_channels: {max_channels}, min_channels: {min_channels}')
        # print(f'candidate: {candidate}')
        model.module.set_channel_choices(subnet_r, self.channel_bins, self.min_channel_bins)
        score2, _ = evaluator.eval(model, train_loader, val_loader)
        score = (score1 + score2) / 2

        return score

    def _init_population(self, model, choice_modules, loss_rec):
        logger.info(f'Initializing Prior Population with {len(loss_rec)} Loss Records...')
        L = len(choice_modules) # num of layers
        num_width = self.channel_bins - self.min_channel_bins + 1 # all possible num of width value
        E = torch.zeros((L, num_width), device='cuda')
        layer_width_cnt = torch.zeros_like(E)
        # subnet from sampler is [low, high)
        # subnet from searcher is width
        for loss, (subnet_l, subnet_r) in loss_rec:
            assert L == len(subnet_l)
            assert len(subnet_l) == len(subnet_r)
            for i in range(L):
                width = subnet_l[i][1] - subnet_l[i][0]
                E[i, width-1] += loss
                layer_width_cnt[i, width-1] += 1
        # for numerical stability, give width never chosen super large error
        E[layer_width_cnt == 0] = 1e-4
        E /= (layer_width_cnt + 1e-5)

        # flops calculation, layer flops is proportional to i/o channel width 
        F = torch.zeros((L, num_width, num_width), device='cuda')
        flop_list = model.module.get_layer_flops(input_shape=self.input_shape)
        temp = torch.zeros((num_width, num_width), device='cuda')
        for c_i in range(1, num_width+1):
            for c_j in range(1, num_width+1):
                temp[c_i-1, c_j-1] = c_i * c_j
        temp /= self.channel_bins ** 2
        for i in range(L):
            F[i, :, :] = flop_list[i] * temp     
            
        # output sample possibility is softmax of P
        P = torch.autograd.Variable(torch.randn_like(E), requires_grad=True)
        optim = torch.optim.SGD([P], lr=0.01)
        for _ in range(2000):
            optim.zero_grad()
            prob = Func.softmax(P, dim=1)
            prob_shift = Func.pad(prob, (0, 0, 0, 1), value=1.0/num_width)[1:, :]
            z = (prob * E).sum(dim=1)
            
            F_e = (F * prob.view(L, num_width, 1) * prob_shift.view(L, 1, num_width)).sum()
            loss = z.mean() + 1000 * (1.0 - F_e / (self.flops_limit*0.4))**2
            # print(f'z.mean: {z.mean()}, flops: {(1.0 - F_e / self.flops_limit)**2}')
            loss.backward()
            optim.step()
            if _ % 100 == 0: 
                logger.info(f'Initialize Prior Population: Epoch {_} Loss {loss.item()}, z.mean(): {z.mean()}, flops: {(1.0 - F_e / (self.flops_limit*.90))**2}')

        P.detach_()
        prob = Func.softmax(P, dim=1).cpu().numpy()
        self.prob = prob
        to_ret = []
        logger.info(f'Initialize Prior Population Done: P {prob}')
        for _ in range(self.population_num):
            while 1:
                subnet = []
                for i in range(L):
                    subnet.append(np.random.choice(a=np.arange(choice_modules[i][0], choice_modules[i][1]+1), 
                                                    p=prob[i]))
                subnet = tuple(subnet)

                if self.is_legal(subnet, model, choice_modules): 
                    to_ret.append(subnet)
                    model.module.set_channel_choices([[0, c] for c in subnet], self.channel_bins, self.min_channel_bins)
                    flops = get_flops(model, self.input_shape)
                    logger.info(f'Initialize Prior Population Subnet flops:{flops/1e6}M')
                    break
        return to_ret

    def search(self, *args, **kwargs):
        population = []
        if self.prior_init and len(self.population) == 0:
            loss_rec = kwargs.pop('loss_rec', [])
            if self.rank == 0:
                population = self._init_population(args[0], args[1], loss_rec)

            population = torch.tensor(population, dtype=torch.int32, device='cuda')
            dist.broadcast(population, src=0)
            population = population.tolist()

            # self.prob = torch.tensor(self.prob, dtype=torch.float, device='cuda')
            # dist.broadcast(self.prob, src=0)
            # self.prob = self.prob.tolist()
            
        kwargs['population'] = population
        return super(BCNetSearcher, self).search(*args, **kwargs)

