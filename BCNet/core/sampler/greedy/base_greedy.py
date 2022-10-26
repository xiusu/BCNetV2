from abc import abstractmethod
from core.model.net import Net
from tools.eval.base_tester import BaseTester
import cellular.pape.distributed as dist
from core.utils.arch_util import _decode_arch
from core.utils.flops import count_sample_flops, Model_channels
from core.sampler.sampler_model import Sampler_extension, Sampler_1Dto2D, Sample_channels, Channel_list
import torch
import time
import copy
from core.utils.flops import count_flops

class Base_Greedy:
    def __init__(self, model: Net, tester: BaseTester, **kwargs):
        self.model = model
        self.tester = tester
        self.rank = dist.get_rank()
        #self.model_channels = Model_channels(self.model.net)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.flops_constrant = self.Flops

    def greedy_eval(self):
        """
        Do eval for a list of subnet.
        :return: a list of score
        """
        if getattr(self,'Var_len'):
            #n_var = sum(self.bin_number_list)
            n_var = self.Var_len
        else:
            raise RuntimeError('n_var did not defined in evolution_sampler.py')

        subnet_list = [len(self.P_train) - 1] * n_var
        #assert len(self.model.net) == len(subnet_list) or len(self.model.net) * 2 == len(subnet_list)
        assert self.Var_len == len(subnet_list), "subnet_list length wrong. n_var len is {}, subnet_list is {}".format(self.Var_len, subnet_list)
        assert len(self.P_train) <= 10, "code only support not more than 100D skip_list, skip list is {}".format(self.P_train)

        #flops_constrant = getattr(self, 'flops_constraint', 330e6)
        # check flops first
        #flops = count_flops(self.model.net, subnet_list)

        Iteration = 0

        #prune topk value each iteration
        topk = getattr(self, 'topk', 10)
        while True:
            Topk = []
            for i in range(n_var):
                temp_list = copy.deepcopy(subnet_list)
                if temp_list[i] > 0:
                    temp_list[i] = temp_list[i] - 1
                    #temp_list = Sampler_extension(self.skip_list, temp_list)
                    subnet_list_L, subnet_list_R = Channel_list(self.P_train, self.skip_list, self.Model_Channels, temp_list)
                    flops = count_sample_flops(self.model.net, subnet_list_L, input_shape=self.input_shape)
                    score = self.Evaluate(subnet_list_L, temp_list)

                    if hasattr(self, 'evtwo_side') and self.evtwo_side:
                        score_R = self.Evaluate(subnet_list_R, temp_list)
                        score = ( score + score_R )/2

                    if self.rank == 0:
                        print('Iteration: {} == subnet_list: {}, FLOPs: {}, score: {}'.format(Iteration, temp_list, flops, score))
                    Topk.append(score)
                else:
                    Topk.append(0.1)
                    continue

            Index = []
            for i in range(topk):
                number = max(Topk)
                index = Topk.index(number)
                Topk[index] = 0
                Index.append(index)

            while(len(Index)>0):
                subnet_list[Index.pop(0)] -= 1

            if self.rank == 0:
                subnet = torch.IntTensor(subnet_list)
            else:
                subnet = torch.zeros([self.Var_len], dtype=torch.int32)
            dist.broadcast(subnet, 0)
            subnet_list = subnet.tolist()

            subnet_list_L, _ = Channel_list(self.P_train, self.skip_list, self.Model_Channels, subnet_list)
            flops = count_sample_flops(self.model.net, subnet_list_L, input_shape=self.input_shape)

            if self.rank == 0:
                print('Iteration over: {} == subnet_list: {}, FLOPs: {}'.format(Iteration, subnet_list, flops))

            Iteration += 1
            if flops <= self.flops_constrant:
                self.check_flops = flops
                self.subnet_top1 = subnet_list
                if self.rank == 0:
                    print('Greedy search over == Best_subnet_list: {}, FLOPs: {}'.format(subnet_list_L, flops))
                return

    def Evaluate(self, subnet_list, Channel_dropout):
        # recal bn
        if hasattr(self, 'train_eval') and self.train_eval:
            self.model.reset_bn()
            self.model.train()
            time.sleep(2)  # process may be stuck in dataloader
            subnet_temp = copy.deepcopy(subnet_list)
            score_L = self.tester.test(Channel_dropout=subnet_temp)
        else:
            self.model.reset_bn()
            self.model.train()
            # need to forward twice to recal bn - (2 x 50k / gpu_num) imgs
            #self.tester.dataloader = None
            time.sleep(2)  # process may be stuck in dataloader
            for i in range(self.cal_time):
                subnet_temp = copy.deepcopy(subnet_list)
                score = self.tester.test(Channel_dropout = subnet_temp)
                time.sleep(2)  # process may be stuck in dataloader
            if self.rank == 0:
                print('==training subnet: {}, score: {}'.format(str(Channel_dropout), score))

            self.model.eval()
            subnet_temp = copy.deepcopy(subnet_list)
            score_L = self.tester.test(Channel_dropout=subnet_temp)
        return score_L


