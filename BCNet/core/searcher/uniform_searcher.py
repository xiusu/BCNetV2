if __name__ == '__main__':
    from  base_searcher import BaseSearcher
else:
    from .base_searcher import BaseSearcher
from core.dataset.build_dataloader import build_dataloader
import core.dataset.build_dataloader as BD

import torch
import cellular.pape.distributed as dist
from core.search_space.ops import channel_mults
import random
import time
from pthflops import count_ops


class UniformSearcher(BaseSearcher):
    def __init__(self, **kwargs):
        super(UniformSearcher, self).__init__()
        self.rank = dist.get_rank()
        self.flops_constrant = kwargs.pop('flops_constrant', 400e6)

    def generate_subnet(self, model):
        if self.rank == 0:
            subnet = []
            for block in model.net:
                subnet.append(random.randint(0, len(block) - 1))
            for idx, block in enumerate(model.net):
                if getattr(block[0], 'channel_search', False):
                    subnet.append(random.randint(0, len(channel_mults) - 1))
                else:
                    subnet.append(channel_mults.index(1.0))
            subnet = torch.IntTensor(subnet)
        else:
            subnet = torch.zeros([len(model.net) * 2], dtype=torch.int32)
        dist.broadcast(subnet, 0)
        subnet = subnet.tolist()
        return subnet

        # rubbish
        model.subnet = subnet  # hard code      
        flops = count_ops(model, torch.zeros([1, 3, 224, 224]).cuda(), ignore_layers=['BatchNorm2d', 'DynamicBatchNorm2d', 'ReLU'], verbose=False, print_readable=False)[0]
        if flops < self.flops_constrant:
            print('nani2subnet: {}, FLOPs: {}'.format(subnet, flops))
            return subnet
        else:
            return self.generate_subnet(model)


class UniformMultiPathSearcher(BaseSearcher):
    def __init__(self, **kwargs):
        super(UniformMultiPathSearcher, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._init_eval_dataset(kwargs['eval_dataset'])
        multi_path = kwargs.get('multi_path', {})
        self.sample_num = multi_path.get('sample_num', 10)
        self.top_k = multi_path.get('top_k', 8)
        self.strip = multi_path.get('strip', 0)
        self.num_not_eval = self.strip
        self.path_pool = []
        self.iter_num = 0
        self.images, self.labels = None, None

    def _init_eval_dataset(self, cfg_data):
        self.data_loader = build_dataloader(cfg_data, is_test=True)
        self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))

    def _eval_path(self, subnet, model):
        if self.iter_num % 100 == 0 or self.images is None:
            images, labels = next(self.iter_loader)
            self.images, self.labels = images, labels
        else:
            images, labels = self.images, self.labels
        if images is None:
            self.data_loader.sampler.epoch += 1
            self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))
            images, labels = next(self.iter_loader)
        input_all = {}
        input_all['images'] = images
        input_all['labels'] = labels
        with torch.no_grad():
            output = model(input_all, subnet)
        loss = output['loss']
        reduced_loss = loss.data.clone() / self.world_size
        dist.all_reduce(reduced_loss)
        score = 1./ reduced_loss
        return score.item()

    def _generate_subnet(self, model):
        if self.rank == 0:
            subnet = []
            for block in model.net:
                subnet.append(random.randint(0, len(block) - 1))
            for idx, block in enumerate(model.net):
                if getattr(block[0], 'channel_search', False):
                    subnet.append(random.randint(0, len(channel_mults) - 1))
                else:
                    subnet.append(channel_mults.index(1.0))
            subnet = torch.IntTensor(subnet)
        else:
            subnet = torch.zeros([len(model.net) * 2], dtype=torch.int32)
        dist.broadcast(subnet, 0)
        subnet = subnet.tolist()
        return subnet

    def generate_subnet(self, model):
        if len(self.path_pool) == 0:
            if self.num_not_eval != self.strip:
                self.num_not_eval += 1
                return self._generate_subnet(model)
            else:
                self.num_not_eval = 0
            # generate paths
            subnets = []
            for _ in range(self.sample_num):
                subnet = self._generate_subnet(model)
                #t = time.time()
                subnets.append((subnet, self._eval_path(subnet, model)))
                #print('eval_time: ', time.time() - t)
            subnets.sort(key=lambda x: x[1], reverse=True)
            self.path_pool.extend([x[0] for x in subnets[:self.top_k]])
            self.iter_num += 1
        return self.path_pool.pop(0)
