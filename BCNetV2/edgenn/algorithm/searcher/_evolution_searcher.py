from builtins import print
import os
import sys
import time
import glob
import torch
import logging
import argparse
import torch
import random
import numpy as np

from abc import abstractmethod
from torch import distributed as dist
from .base_searcher import BaseSearcher

logger = logging.getLogger()

def choice(x):
    if isinstance(x, (list, tuple)):
        return x[np.random.randint(len(x))]
    elif isinstance(x, int):
        return np.random.randint(x)
    else:
        raise ValueError(f'illegal input {type(x)}')

class EvolutionSearcher(BaseSearcher):
    def __init__(self,
                 max_epoch=20,
                 population_num=50,
                 top_k_num=10,
                 mutation_num=25,
                 mutation_prob=0.1,
                 crossover_num=25,
                 **kwargs):
        super(EvolutionSearcher, self).__init__(**kwargs)
        self.max_epoch = max_epoch
        self.population_num = population_num
        self.top_k_num = top_k_num
        self.mutation_num = mutation_num
        self.mutation_prob = mutation_prob
        self.crossover_num = crossover_num

        self.epoch = 0
        self.population = []
        self.score_map = {}
        self.top_k = []
    
    @abstractmethod
    def is_legal(self, candidate, model, choice_modules):
        pass

    @abstractmethod
    def gen_subnet(self, model, choice_modules):
        pass

    def search(self, model, choice_modules, evaluator, train_loader, val_loader, **kwargs):
        population = kwargs['population']
        ckpt_manager = kwargs.get('ckpt_manager', None)
        top_k = []
        if len(self.population) != 0:
            # resume
            population = self.population
            top_k = self.top_k
            # print(top_k)
        else:
            self.score_map = {}

        start_time = time.time()

        while self.epoch < self.max_epoch:
            logger.info(f'== search epoch {self.epoch}')
            epoch_time = time.time()
            # add random candidate
            while len(population) < self.population_num:
                candidate = self.gen_subnet(model, choice_modules)
                if not self.is_legal(candidate, model, choice_modules):
                    continue
                population.append(candidate)
                
            print(f'1111, rank: {self.rank}')
            # broadcast population
            for i, candidate in enumerate(population):
                candidate = torch.tensor(candidate, dtype=torch.int32, device='cuda')
                dist.broadcast(candidate, src=0)
                population[i] = tuple(candidate.tolist())

            print(f'2222, rank: {self.rank}')
            # calc score for all candidates
            for candidate in population:
                self.score_map[candidate] = self.eval_subnet(candidate, model, choice_modules, evaluator, train_loader, val_loader)
                num_evaluated = sum([1 for score in self.score_map.values() if score != 0.])
                logger.info(f'eval[{num_evaluated}/{self.population_num*self.max_epoch}] {candidate} get score {self.score_map[candidate]}')

            # get topk
            top_k += population
            top_k.sort(key=lambda x: self.score_map[x], reverse=True)
            top_k = top_k[:self.top_k_num]
            self.top_k = top_k

            if self.rank == 0:
                for i, top in enumerate(top_k):
                    logger.info(f'top {i} choice {top} score {self.score_map[top]}')

            # get mutation
            mutation = []
            max_iters = self.mutation_num * 10
            while len(mutation) < self.mutation_num and max_iters > 0:
                max_iters -= 1
                candidate = list(choice(top_k))
                for i in range(len(candidate)):
                    if np.random.random_sample() < self.mutation_prob:
                        candidate[i] = self.gen_subnet(model, choice_modules)[i]
                candidate = tuple(candidate)
                if not self.is_legal(candidate, model, choice_modules):
                    continue
                mutation.append(candidate)
                
            # get crossover
            crossover = []
            max_iters = self.crossover_num * 10
            while len(crossover) < self.crossover_num and max_iters > 0:
                max_iters -= 1
                c1 = choice(top_k)
                c2 = choice(top_k)
                candidate = tuple(choice([i, j]) for i, j in zip(c1, c2))
                if not self.is_legal(candidate, model, choice_modules):
                    continue
                crossover.append(candidate)

            # update population
            population = mutation + crossover
            self.population = population
            logger.info(f'epoch {self.epoch} time {(time.time() - epoch_time) / 3600:.2f} hours')
            if ckpt_manager is not None:
                metric = {'score': self.score_map[top_k[0]]}
                ckpt_manager.update(self.epoch, metric, score_key='score', searching=True)
            self.epoch += 1
            
        logger.info(f'total searching time {(time.time() - start_time) / 3600:.2f} hours')

        
        for i, top in enumerate(top_k):
            logger.info(f'top {i} choice {top} score {self.score_map[top]}')
