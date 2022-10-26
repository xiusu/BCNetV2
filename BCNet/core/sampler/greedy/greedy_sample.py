import numpy as np
from core.sampler.sampler_model import Sampler_extension, Sampler_1Dto2D
import cellular.pape.distributed as dist
from core.sampler.greedy.base_greedy import Base_Greedy
from core.utils.flops import count_sample_flops, Model_channels
import torch
import time
import copy

class Greedy_sampler(Base_Greedy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample(self, sampling=None):

        self.greedy_eval()
        print("Evolution_Sampler finished")

        sample_flist = self.subnet_top1

        Prob_list = []
        for i in range(len(sample_flist)):
            Prob_list.append(self.P_train[sample_flist[i]])

        self.subnet_channels_top1 = [int(i * j) for i, j in zip(Prob_list, self.Model_Channels)]
        self.subnet_channels_top1 = Sampler_extension(self.skip_list, self.subnet_channels_top1)

        #flops = count_sample_flops(self.model.net, self.subnet_channels_top1, input_shape=self.input_shape)

        #assert  self.check_flops == flops, "flops neq to check flops, flops is {}, check flops is {}".format(flops, self.check_flops)
        if self.rank == 0:
            print("Best subnet is {}".format(self.subnet_channels_top1))