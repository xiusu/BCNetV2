from .base_searcher import BaseSearcher
import random
from ..builder import SearcherReg
import logging
import json
import os

logger = logging.getLogger()

@SearcherReg.register_module('manual_bcnet')
class ManualSearcher_bcnet(BaseSearcher):
    def __init__(self, **kwargs):
        super(ManualSearcher_bcnet, self).__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        with open(self.subnet_path, 'r') as json_file:
            subnets = json.load(json_file)
        self.subnets = [eval(k) for k, v in subnets.items()]
        self.subnets.sort()
        self.search_num = len(self.subnets)
        logger.info(f'[ManualSearcher] number of subnets: {self.search_num}')
        self.subnet_idx = 0
        self.save_results = {'Result':[]}
        self.check_init()

    
    def gen_subnet(self):
        subnet = self.subnets[self.subnet_idx]
        self.subnet_idx += 1
        # assert len(subnet) == len(choice_modules)
        return subnet

    def num_2_list(self, num):
        temp = str(num)
        result_list = [eval(i) for i in temp]
        return result_list
    

    def eval_subnet(self, candidate, model, choice_modules, evaluator, train_loader, val_loader, AutoSlim = False):
        # assert len(choice_modules) == len(candidate)
        # if isinstance(candidate, tuple):
        #     candidate = list(candidate)
        candidate = self.num_2_list(candidate)
        subnet_l = [[0, c] for c in candidate]

        model.module.set_channel_choices(subnet_l, self.channel_bins, self.min_channel_bins)
        score1, _ = evaluator.eval(model, train_loader, val_loader)

        if AutoSlim:
            return score1

        max_channels = [x[1] for x in choice_modules]
        min_channels = [x[0] for x in choice_modules]
        if max(min_channels) > 1:
            subnet_r = [[max_c - c, max_c] for max_c, min_c, c in zip(max_channels, min_channels, candidate)]
        
        else:
            subnet_r = [[max_c - c, max_c] for max_c, c in zip(max_channels, candidate)]
            


        model.module.set_channel_choices(subnet_r, self.channel_bins, self.min_channel_bins)
        score2, _ = evaluator.eval(model, train_loader, val_loader)
        score = (score1 + score2) / 2

        return score
    
    def check_init(self):
        if os.path.exists(self.record_path):
            with open(self.record_path) as file_obj:
                self.save_results = json.load(file_obj)
                self.subnet_idx = len(self.save_results['Result'])
            logger.info(f'Resume number of subnets: {self.subnet_idx}, from path: {self.record_path}')

    def record_results(self):
        if self.rank == 0:
            with open(self.record_path,'w') as file_obj:
                json.dump(self.save_results,file_obj)
        logger.info(f'record to path: {self.save_results}, num: {self.subnet_idx}')

    def search(self, model,choice_modules, evaluator, train_loader, val_loader, **kwargs):
        while(self.subnet_idx < self.search_num-1):
            subnet = self.gen_subnet()
            score = self.eval_subnet(subnet, model, choice_modules, evaluator, train_loader, val_loader, self.AutoSlim)
            logger.info(f'Num: {self.subnet_idx}, subnet: {subnet}, score: {score}')
            self.save_results['Result'].append({'Num':self.subnet_idx,'subnet':subnet,'score':score})
            if self.subnet_idx % 20 == 0:
                self.record_results()
        self.record_results()
        raise RuntimeError(f'Finished, all results: {self.save_results}')