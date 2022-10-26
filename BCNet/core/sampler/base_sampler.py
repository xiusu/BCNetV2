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


class BaseSampler:
    def __init__(self, model: Net, tester: BaseTester, **kwargs):
        self.model = model
        self.tester = tester
        self.rank = dist.get_rank()
        #self.model_channels = Model_channels(self.model.net)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.flops_constrant = self.Flops
    def forward_subnet(self, input):
        """
        run one step
        :return:
        """
        pass

    def eval_subnet(self, subnet_list):
        """
        Do eval for a list of subnet.
        :return: a list of score
        """
        #assert len(self.model.net) == len(subnet_list) or len(self.model.net) * 2 == len(subnet_list)
        assert self.Var_len == len(subnet_list), "subnet_list length wrong. n_var len is {}, subnet_list is {}".format(self.Var_len, subnet_list)
        assert len(self.P_train) <= 100, "code only support not more than 100D skip_list, skip list is {}".format(self.P_train)

        #flops_constrant = getattr(self, 'flops_constraint', 330e6)
        # check flops first         
        #flops = count_flops(self.model.net, subnet_list)

        subnet_list = copy.deepcopy(subnet_list)
        if isinstance(subnet_list,list):
            Channel_dropout = subnet_list
        else:
            Channel_dropout = subnet_list.tolist()
        #Channel_dropout = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

        subnet_list_L, subnet_list_R = Channel_list(self.P_train, self.skip_list, self.Model_Channels, Channel_dropout)
        #subnet_list = self.bin_list(Channel_dropout, self.bin_number_list, self.bin_size_list)

        flops = count_sample_flops(self.model.net, subnet_list_L, input_shape = self.input_shape)

        if self.rank == 0:
            print('==subnet: {}, FLOPs: {}'.format(str(Channel_dropout), flops))
        if flops > self.flops_constrant:
            return 5 - (flops - self.flops_constrant) / self.flops_constrant

        ev_two_side = getattr(self, 'evtwo_side', True)
        if ev_two_side:
            score_L = self.Evaluate(subnet_list_L, Channel_dropout)
            if self.rank == 0:
                print('==testing subnet_L: {}, score: {}'.format(str(Channel_dropout), score_L))
            score_R = self.Evaluate(subnet_list_R, Channel_dropout)
            if self.rank == 0:
                print('==testing subnet_R: {}, score: {}'.format(str(Channel_dropout), score_R))
            score = (score_L + score_R) / 2
        else:
            score = self.Evaluate(subnet_list_L, Channel_dropout)
            if self.rank == 0:
                print('==testing subnet_L: {}, score: {}'.format(str(Channel_dropout), score))
        return score

    @abstractmethod
    def generate_subnet(self):
        """
        generate one subnet
        :return: block indexes for each choice block
        """

    def sample(self):
        subnet_eval_dict = {}
        subnet_eval_list = []
        while len(subnet_eval_dict) < self.search_num:
            # first, generate a subnet
            if self.rank == 0:
                subnet = self.generate_subnet()
                subnet = torch.IntTensor(subnet)
            else:
                subnet = torch.zeros([len(self.generate_subnet())], dtype=torch.int32)
            dist.broadcast(subnet, 0)
            subnet = subnet.tolist()
            subnet_t = tuple(subnet)
            if subnet_eval_dict.get(subnet_t) is not None:
                # already searched
                continue
            # set subnet
            score = self.eval_subnet(subnet)
            if score == 0:  # flops not suitable, continue to next subnet
                continue
            if self.rank == 0:
                print('==testing subnet: {}, score: {}'.format(str(subnet), score))
            subnet_eval_dict[subnet_t] = score
            subnet_eval_list.append((subnet_t, score))
        sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i:i[1], reverse=True)
        sorted_subnet_key = [x[0] for x in sorted_subnet]
        subnet_topk = sorted_subnet_key[:self.sample_num]
        if self.rank == 0:
            print('== search result ==')
            print(sorted_subnet)
            print('== best subnet ==')
            print(subnet_topk)
            print('== subnet eval list ==')
            print([x[1] for x in subnet_eval_list])
        return None
        return _decode_arch(subnet_topk)  # TODO: fix bug
    '''
    def Channel_list(self, subnet_list):
        Prob_list = []
        for i in range(len(subnet_list)):
            Prob_list.append(self.P_train[subnet_list[i]])

        assert len(Prob_list) == len(self.Model_Channels), "Prob_list len must eq to Model_Channels, Prob_list is {}, Model_Channels is {}， Pro is {}, M_C is {}".format(len(Prob_list), len(self.Model_Channels), Prob_list, Model_Channels)

        L_Channels, R_Channels = Sample_channels(Prob_list, self.Model_Channels)

        L_Channels = Sampler_extension(self.skip_list, L_Channels)
        R_Channels = Sampler_extension(self.skip_list, R_Channels)

        return L_Channels, R_Channels
    '''
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


    def bin_list(self,subnet_list, bin_number_list, bin_size_list):

        # list 1D to 2D
        bin_number_list = copy.deepcopy(bin_number_list)
        bin_size_list = copy.deepcopy(bin_size_list)
        Channel = []
        L1 = 0
        assert len(subnet_list) == sum(bin_number_list), "subnet_list number must eq to bin_number_list, subnet_list is {}, bin_number_list is {}".format(subnet_list, bin_number_list)
        #print("subnet_list is {}".format(subnet_list))
        #print("bin_number_list is {}".format(bin_number_list))
        while (len(bin_number_list) > 0):
            L2 = L1 + bin_number_list.pop(0)
            Channel.append(subnet_list[L1:L2])
            L1 = L2

        #test structure is break?
        for k in Channel:
            if sum(k) == 0:
                return False

        #print("Channel is {}".format(Channel))
        Final_channels = []
        while(len(bin_size_list) > 0):
            gama = bin_size_list.pop(0)
            List = Channel.pop(0)
            List_temp2 = []

            for i in List:
                List_temp = [i for _ in range(gama)]
                List_temp2.append(List_temp)
            List_temp2 = sum(List_temp2, [])
            Final_channels.append(List_temp2)

        Total_list = Sampler_extension(self.skip_list, Final_channels)
        return Total_list

