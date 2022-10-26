import importlib
from core.model.net import Net
from core.searcher import build_searcher
from core.sampler import build_sampler, rebuild_model
from core.utils.lr_scheduler import IterLRScheduler
from core.utils.optimizer import build_optimizer
from core.dataset.build_dataloader import build_dataloader
from tools.eval.build_tester import build_tester
from tools.trainer import build_trainer
from cellular.parallel import DistributedModel as DM
import cellular.pape.distributed as dist
import time
from os.path import join, exists
import os
from core.utils.measure import measure_model
import numpy as np
import torch.nn as nn
from core.utils.flops import count_flops
from core.utils.flops import count_latency
import copy
from core.sampler.sampler_model import Init_sample, Neq_Random, Sampler_deduction, Sampler_extension
from core.utils.flops import Model_channels
import math
import torch
import random


# V2 in cifar10 296473088   params: 2236682
# V2 in imgnet width multiplier == 1.0 Flops 300774272      params 3504872
# V2 in imgnet with width multiplier == 1.5  Flops 652116288
# VGG19 FLOPs: 398136320   params: 20035018
# VGG16 FLOPs: 313201664    params: 14724042
# VGG *1.5 =703372800, VGG *1 = 313M in Cifar10
# resnet50 FLOPs: 4089184256 params:25557032
# resnet50 1.5x = 9110618112 Params = 55919176
class NASAgent:
    def __init__(self, config):

        self.model_config = config.pop('model')

        if 'width_multiplier' in self.model_config:
            width_multiplier = self.model_config.pop('width_multiplier')
            self.model_config['channels'] = [int(x * width_multiplier) for x in self.model_config['channels']]

        self.cfg_retrain = config.pop('retrain')
        self.cfg_sample = config.pop('sample')
        self.final_train = config.pop('final_train')
        self.cfg_test = config.pop('test')

    def run(self):
        # retrain config
        cfg_retrain_stg = self.cfg_retrain.pop('strategy')
        cfg_retrain_data = self.cfg_retrain.pop('data')

        self.Input_size = [cfg_retrain_data['final_channel'], cfg_retrain_data['final_height'],
                           cfg_retrain_data['final_width']]

        # sample config
        cfg_sample_sampler = self.cfg_sample.pop('sampler')
        cfg_sample_data = self.cfg_sample.pop('data')
        cfg_sample_stg = self.cfg_sample.pop('strategy')

        # final_train config
        cfg_fintrain_stg = self.final_train.pop('strategy')
        cfg_fintrain_data = self.final_train.pop('data')

        # test config
        cfg_test_stg = self.cfg_test.pop('strategy')
        cfg_test_data = self.cfg_test.pop('data')

        self.rank, self.world_size, self.local_rank = dist.init()

        self.sample_epoch = self.cfg_sample.get('epoch', 1)

        cfg_subnet = copy.deepcopy(self.model_config)

        # config for retrain
        self.mksave_path(cfg_retrain_stg)
        self._build_model(cfg_subnet)


        #self.skip_list = cfg_retrain_stg['bin_config']['skip_list']
        '''
        self.epoch_flops()
        save_channel = copy.deepcopy(self.model_config['channels'])
        skip_list = cfg_retrain_stg['bin_config']['skip_list']
        ori_list = Sampler_deduction(skip_list ,save_channel)
        save_list = []
        if self.rank == 0:
            for i in range(1000):
                save_channel = copy.deepcopy(ori_list)
                Len = len(save_channel)
                new_channel = []
                for i in range(Len):
                    temp = save_channel.pop(0)
                    temp = temp * random.uniform(0.4, 1.0)
                    new_channel.append(round(temp))

                new_channel = Sampler_extension(skip_list, new_channel)

                cfg_subnet = copy.deepcopy(self.model_config)
                cfg_subnet['channels'] = new_channel
                temp_channels = copy.deepcopy(new_channel)
                model = Net(cfg_subnet)
                FLOPs = count_flops(model.net, input_shape=self.Input_size)

                if i % 100 == 0:
                    print("FLOPs_constraint is {}".format(self.Flops[0]))
                    print("FLOPs ratio is {}".format(FLOPs/self.Flops[0]))
                    raise RuntimeError("stop")

                if FLOPs < self.Flops[0] and FLOPs > 0.85* self.Flops[0]:
                    save_list.append(temp_channels)

        print("list is {}".format(save_list))
        raise RuntimeError("stop")
        '''

        # print('Retraining subnet FLOPs: {}'.format(count_flops(self.model.net, input_shape=self.Input_size)))
        # assert 1==2

        # retrain and sampler
        if self.cfg_sample['flag']:
            self.retrain_sample(cfg_subnet, cfg_retrain_data, cfg_retrain_stg, cfg_sample_stg, cfg_sample_data,
                                cfg_sample_sampler)
        else:
            self.sampler = Sample_temp(subnet_channels_top1=copy.deepcopy(self.model_config['channels']))

        # final_train config
        cfg_fintrain_stg = cfg_fintrain_stg[self.final_train['train_mode']]
        if cfg_fintrain_stg['use_basic_path']:
            cfg_fintrain_stg['save_path'] = cfg_retrain_stg['basic_path'] + '/final_train'

        if self.final_train['flag']:
            if self.final_train['train_mode'] == 'retrain':
                # check top1,maybe pop before
                if self.rank == 0:
                    print("rebuild_channels is {}".format(self.sampler.subnet_channels_top1))
                Channels = copy.deepcopy(self.sampler.subnet_channels_top1)
                self._build_model(cfg_subnet, rebuild_channels=Channels)

            self._build_retrainer(cfg_fintrain_data, cfg_fintrain_stg)
            self.retrain(finaltrain=True)

        # final tester
        if self.cfg_test['flag']:
            if cfg_test_stg['use_basic_path']:
                cfg_test_stg['save_path'] = cfg_fintrain_stg['save_path'] + '/checkpoint'
            Channels = copy.deepcopy(self.sampler.subnet_channels_top1)
            self._build_model(cfg_subnet, rebuild_channels=Channels)

            measure_model(self.model)
            self._build_tester(cfg_test_data, cfg_test_stg)
            self.test_top1 = {}
            self.test_top5 = {}
            self.test(cfg_test_stg, test_top1=self.test_top1, test_top5=self.test_top5)

            if self.rank == 0:
                temp_best = 0
                for k, v in self.test_top1.items():
                    if v > temp_best:
                        temp_name = k
                        temp_best = v
                print("Best pt is {}, Best top1 is {}, top5 is {}".format(temp_name,
                                                                          self.test_top1['{}'.format(temp_name)],
                                                                          self.test_top5['{}'.format(temp_name)]))

        # print pruning_results
        if self.cfg_sample['flag']:
            self.pruning_results()

    def _build_model(self, cfg_net, rebuild=0, **kwargs):
        model = Net(cfg_net, **kwargs)
        self.model_ema = copy.deepcopy(model)
        model = DM(model)
        #DM_model_copy = copy.deepcopy(model)
        model.cuda()

        if rebuild == 1:
            self.rebuild_model = model
        else:
            self.model = model

    def _build_searcher(self, cfg_searcher, cfg_data_search, cfg_stg_search):

        self.search_dataloader = build_dataloader(cfg_data_search)

        opt = build_optimizer(self.model, cfg_stg_search['optimizer'])
        lr_scheduler = IterLRScheduler(opt, **cfg_stg_search['lr_scheduler'])

        for searcher_type, start_iter in zip(cfg_searcher['type'], cfg_searcher['start_iter']):
            searcher = build_searcher(searcher_type, **cfg_searcher.get(searcher_type, {}))
            self.model.add_searcher(searcher, start_iter)

        self.search_trainer = build_trainer(cfg_stg_search, self.search_dataloader, self.model,
                                            opt, lr_scheduler, time.strftime("%Y%m%d_%H%M%S", time.localtime()),
                                            input_shape=self.Input_size)
        # time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        if cfg_stg_search.get('resume', False):
            self.search_trainer.load(join(cfg_stg_search['save_path'], 'checkpoint', cfg_stg_search['load_name']))

    def _build_sampler(self, cfg_sampler, cfg_data_sample, cfg_stg_sample, cfg_net, skip_list):
        self.tester = build_tester(cfg_stg_sample, cfg_data_sample, self.model)

        Total_channels = Model_channels(self.model.net)
        Total_channels = Sampler_deduction(self.trainer.skip_list, Total_channels)

        self.Model_Channels = Total_channels
        self.Model_Length = len(self.Model_Channels)

        # self.sampler = build_sampler(cfg_sampler, self.model, self.tester, net_cfg = cfg_net, Flops = self.Flops.pop(0), bin_size_list = Model_Length,
        #                             bin_number_list = self.trainer.bin_number, skip_list= skip_list, input_shape = self.Input_size)


        self.sampler = build_sampler(cfg_sampler, self.model, self.tester, net_cfg=cfg_net, Flops=self.Flops[0],
                                     Var_len=self.Model_Length, Model_Channels=self.Model_Channels,
                                     P_train=self.trainer.P_train, skip_list=skip_list, input_shape=self.Input_size)

    def _build_retrainer(self, cfg_data_retrain, cfg_stg_retrain):
        self.search_dataloader = build_dataloader(cfg_data_retrain)

        opt = build_optimizer(self.model, cfg_stg_retrain['optimizer'])
        lr_scheduler = IterLRScheduler(opt, **cfg_stg_retrain['lr_scheduler'])

        self.trainer = build_trainer(cfg_stg_retrain, self.search_dataloader, self.model, opt, lr_scheduler, '',
                                     input_shape=self.Input_size, model_ema = self.model_ema)
        if cfg_stg_retrain.get('resume', False):
            self.trainer.load(join(cfg_stg_retrain['save_path'], 'checkpoint', cfg_stg_retrain['load_name']))

    def _build_tester(self, cfg_data_test, cfg_stg_test):
        self.tester = build_tester(cfg_stg_test, cfg_data_test, self.model)

    def search(self):
        self.search_trainer.train()
        if hasattr(self.model.searcher, 'get_best_arch'):
            self.subnet_candidates = self.model.searcher.get_best_arch()
        self.model.remove_searcher()

    def sample(self, cfg_sampler=None):

        Prb_list = self.Init_Sample(cfg_sampler)

        if cfg_sampler is not None:
            self.sampler.sample(sampling=Prb_list)
        else:
            self.sampler.sample()

    def retrain_sample(self, cfg_subnet, cfg_retrain_data, cfg_retrain_stg, cfg_sample_stg, cfg_sample_data,
                       cfg_sample_sampler):
        # sample epoch

        # save training results
        self.Pruning_Channels = []
        self.Pruning_Channel_num = []

        # each flops of every epoch
        self.epoch_flops()
        self.epoch_Sample_p(cfg_sample_sampler)

        if self.sample_epoch >= 2 or self.final_train['train_mode'] == 'finetune':
            self.reAlign_top1 = {}
            self.reAlign_top5 = {}

        for epoch in range(self.sample_epoch):
            # retrain
            cfg_retrain_stg['save_path'] = cfg_retrain_stg['basic_path'] + '/epoch{}'.format(epoch)
            # if self.cfg_retrain['flag']:
            if epoch > 0:
                self.cfg_stg_faster(cfg_retrain_stg)
            self._build_retrainer(cfg_retrain_data, cfg_retrain_stg)
            self.retrain(eph=epoch, train_flag=self.cfg_retrain['flag'])

            # sample
            self.sample_load_name(cfg_retrain_stg, cfg_sample_stg)
            self._build_sampler(cfg_sample_sampler, cfg_sample_data, cfg_sample_stg, cfg_subnet,
                                skip_list=cfg_retrain_stg['bin_config']['skip_list'])
            self.sample(cfg_sample_sampler)
            if self.rank == 0:
                print("rebuild_channels is {}".format(self.sampler.subnet_channels_top1))
            if self.sample_epoch >= 2 or self.final_train['train_mode'] == 'finetune':
                self.rebuild(cfg_subnet)

                if self.rank == 0:
                    Save_path = cfg_retrain_stg['basic_path'] + '/epoch{}/reAlign'.format(epoch)
                    path = join(Save_path, 'epoch_{}_ckpt.pth.tar'.format(epoch))
                    torch.save({'step': cfg_retrain_stg['max_iter'], 'state_dict': self.model.net.state_dict()}, path)
                assert 1 == 2
                self.Retrain_Align(cfg_retrain_data, cfg_retrain_stg, cfg_sample_sampler)

            # record pruning results
            self.pruning_recode()

    def rebuild(self, cfg_net):
        raise RuntimeError("code don't support rebuild func")
        # self.model()
        # self.sampler.subnet_topk()  #! change to subnet_top1()
        # jiashe self.sampler.subnet_topk() have channged to 2D, with the first dimension channel
        rebuild_channels = copy.deepcopy(self.sampler.subnet_channels_top1)
        print("rebuild_channels is {}".format(rebuild_channels))
        self._build_model(cfg_net, rebuild=1, rebuild_channels=rebuild_channels)
        rebuild_channels = copy.deepcopy(self.sampler.subnet_top1)
        print("rebuild_channels is {}".format(rebuild_channels))
        self.model = rebuild_model(self.model, self.rebuild_model, rebuild_channels)

    def test(self, cfg_test_stg, **kwargs):
        for ckpt_iter in range(cfg_test_stg['start'], cfg_test_stg['end'] + 1, cfg_test_stg['strip']):
            model_name = f'iter_{ckpt_iter}_ckpt.pth.tar'
            ema_name = f'ema_iter_{ckpt_iter}_ckpt.pth.tar'
            model_folder = cfg_test_stg.get('load_path', cfg_test_stg['save_path'])
            while not exists(join(model_folder, model_name)):
                if self.rank == 0:
                    print('{} not exists, waiting for training,'
                          ' current time: {}'.format(join(model_folder, model_name),
                                                     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                time.sleep(300)

            self.model.eval()
            self.tester.set_model_path(model_name=model_name, model_folder=model_folder)
            self.tester.test(**kwargs)

            if 'test_ema' in cfg_test_stg and cfg_test_stg['test_ema']:
                self.model.eval()
                self.tester.set_model_path(model_name=ema_name, model_folder=model_folder)
                self.tester.test(**kwargs)


    def retrain(self, finaltrain=False, eph=0, train_flag=True):
        # count ops
        if self.rank == 0:
            print('Retraining subnet FLOPs:')
            print(count_flops(self.model.net, input_shape=self.Input_size))
            print("Retraining Params")
            print(sum([m.numel() for m in self.model.net.parameters()]))
            print('Retraining subnet latency:')
            print(count_latency(self.model.net, input_shape=self.Input_size))
        # this place change to
        self.trainer.train(finaltrain, eph, train_flag)

    def BinaryRandomProbabilitySampling(self, size, Prob=0.5):
        low = Prob - 0.5
        high = Prob + 0.5
        val = np.random.uniform(low=low, high=high, size=size)
        return (val > 0.5).astype(np.int)

    def Init_Sample(self, cfg_sampler):

        if self.rank == 0:
            P_list = Init_sample(self.model.net, self.trainer.Record_Efficient, self.trainer.P_train, self.Flops[0],
                                 self.trainer.skip_list, input_shape=self.Input_size)
            if cfg_sampler['kwargs']['Bet_pop_size'] > 0:
                B_Prob_list = Neq_Random(P_list, P_train=self.trainer.P_train, skip_list=self.trainer.skip_list,
                                         Model_Channels=self.Model_Channels, input_shape=self.Input_size,
                                         model=self.model.net, FLOPs_Constrain=self.Flops[0],
                                         Num=cfg_sampler['kwargs']['Bet_pop_size'])
            else:
                B_Prob_list = []

            Sample_Prob = cfg_sampler['sample_P'].pop(0)

            if cfg_sampler['kwargs']['Bet_pop_size'] < cfg_sampler['kwargs']['pop_size']:
                R_Prob_list = self.BinaryRandomProbabilitySampling(
                    size=(
                        cfg_sampler['kwargs']['pop_size'] - cfg_sampler['kwargs']['Bet_pop_size'], len(P_list)),
                    Prob=Sample_Prob)

                R_Prob_list = list(R_Prob_list)
                for i in range(len(R_Prob_list)):
                    B_Prob_list.append(R_Prob_list[i])

            finished = torch.Tensor([1])
            dist.broadcast(finished, 0)
        else:
            while (1):
                finished = torch.Tensor([0])
                dist.broadcast(finished, 0)
                if finished[0] == 1:
                    break

        if self.rank == 0:
            B_Prob_list_temp = torch.IntTensor(B_Prob_list)
        else:
            B_Prob_list_temp = torch.zeros(cfg_sampler['kwargs']['pop_size'], self.Model_Length, dtype=torch.int32)

        dist.broadcast(B_Prob_list_temp, 0)
        B_Prob_list = B_Prob_list_temp.tolist()

        Prob_list = np.array(B_Prob_list)
        if self.rank == 0:
            print("Prob_list is {}".format(Prob_list))
        assert Prob_list.shape[
                   1] == self.Model_Length, "Prob_list channel length must eq to Model_Length, len is {}, Model_length is {}, Prob_list is {}".format(
            Prob_list.shape[1], self.Model_Length, Prob_list.shape)
        # may_be I just need the Index of the Prob_list, and make several index for Init, and than for sample, we just need sample Index, and cal Sample_channels inside the Sample model
        # L_Channels, R_Channels = Sample_channels(Prob_list, Model_Channels = Model_channels(self.model.net))

        return Prob_list

    def statistic_flops(self):
        sampler = build_sampler({'type': 'random'}, self.model, None, None)
        logger = open('./statistic_flops.txt', 'a')
        for _ in range(12500):  # x8
            subnet = sampler.generate_subnet()
            flops = count_flops(self.model.net, subnet, input_shape=self.Input_size)
            print('{},{}'.format(subnet, flops))
            logger.write('{},{}\n'.format(subnet, flops))
            logger.flush()

    def sample_load_name(self, cfg_retrain_stg, cfg_sample_stg):
        if cfg_sample_stg['use_basic_path']:
            cfg_sample_stg['save_path'] = cfg_retrain_stg['save_path']
            cfg_sample_stg['load_name'] = 'iter_{}_ckpt.pth.tar'.format(cfg_retrain_stg['max_iter'])

    def epoch_flops(self):
        Flops = []
        flops = count_flops(self.model.net, input_shape=self.Input_size)
        b_constraint = (1 - self.cfg_sample['flops_constrant']) / (1 - math.pow(0.5, self.sample_epoch))
        for i in range(self.sample_epoch):
            Percent = 1 - b_constraint * (1 - math.pow(0.5, i + 1))
            Flops.append(flops * Percent)
        self.Flops = Flops

    def epoch_Sample_p(self, cfg_sampler):
        Sample_P = []
        flops = count_flops(self.model.net, input_shape=self.Input_size)
        for i in range(self.sample_epoch):
            P = pow(self.Flops[i] / flops, 0.5)
            Sample_P.append(P - cfg_sampler['Deduce_P'])
        cfg_sampler['sample_P'] = Sample_P

    def mksave_path(self, cfg_retrain_stg):
        for epoch in range(self.sample_epoch):
            save_path = cfg_retrain_stg['basic_path'] + '/epoch{}'.format(epoch)
            if self.rank == 0:
                if not exists(join(save_path, 'checkpoint')):
                    os.makedirs(join(save_path, 'checkpoint'))
                if not exists(join(save_path, 'events')):
                    os.makedirs(join(save_path, 'events'))
                if not exists(join(save_path, 'log')):
                    os.makedirs(join(save_path, 'log'))
                if not exists(join(save_path, 'reAlign')):
                    os.makedirs(join(save_path, 'reAlign'))

        save_path = cfg_retrain_stg['basic_path'] + '/final_train'
        if self.rank == 0:
            if not exists(join(save_path, 'checkpoint')):
                os.makedirs(join(save_path, 'checkpoint'))
            if not exists(join(save_path, 'events')):
                os.makedirs(join(save_path, 'events'))
            if not exists(join(save_path, 'log')):
                os.makedirs(join(save_path, 'log'))

    def cfg_stg_faster(self, cfg_retrain_stg):
        cfg_retrain_stg['max_iter'] = cfg_retrain_stg.get('max_iter_keeptrain', cfg_retrain_stg['max_iter'])
        if 'decay_step_keeptrain' in cfg_retrain_stg['lr_scheduler'] and cfg_retrain_stg['lr_scheduler'][
            'decay_step_keeptrain'] is not None:
            cfg_retrain_stg['lr_scheduler']['decay_step'] = cfg_retrain_stg['lr_scheduler']['decay_step_keeptrain']
        else:
            cfg_retrain_stg['lr_scheduler']['decay_step'] = cfg_retrain_stg['max_iter']

    def pruning_recode(self):
        self.Pruning_Channels.append(self.sampler.subnet_top1)
        self.Pruning_Channel_num.append(self.sampler.subnet_channels_top1)

    def pruning_results(self):
        if self.rank == 0:
            Length = len(self.Pruning_Channel_num)
            for i in range(Length):
                print("Sampling epoch of {}".format(i))
                print("Sampling Channel_num is {}".format(self.Pruning_Channel_num[i]))
                print("  ")
                print("  ")

            print('Retraining subnet FLOPs: {}'.format(count_flops(self.model.net, input_shape=self.Input_size)))

    def Retrain_Align(self, cfg_retrain_data, cfg_retrain_stg, cfg_sample_sampler):

        if 'keeptrain_load_test' in cfg_retrain_stg and cfg_retrain_stg['keeptrain_load_test'] is not False:
            def map_func(storage, location):
                return storage.cuda()

            assert exists(cfg_retrain_stg['keeptrain_load_test']), "{} not exist.".format(
                cfg_retrain_stg['keeptrain_load_test'])
            ckpt = torch.load(cfg_retrain_stg['keeptrain_load_test'], map_location=map_func)
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
            ckpt_keys = set(ckpt['state_dict'].keys())
            own_keys = set(self.model.net.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                if self.rank == 0:
                    self.logger.info(f'**missing key while loading search_space**: {k}')
                    raise RuntimeError(f'**missing key while loading search_space**: {k}')

        if cfg_retrain_stg['keeptrain_mode'] == 'train_net':
            self.make_Alignp(cfg_retrain_stg)
            self._build_retrainer(cfg_retrain_data, cfg_retrain_stg['keeptrain_stg'])
            self.retrain(finaltrain=True)


        elif cfg_retrain_stg['keeptrain_mode'] == 'train_BN':

            for ms in self.model.net:
                for m in ms[0].modules():
                    if not isinstance(m, nn.BatchNorm2d):
                        for param in m.parameters():
                            param.require_grads = False

            self.make_Alignp(cfg_retrain_stg)
            self._build_retrainer(cfg_retrain_data, cfg_retrain_stg['keeptrain_stg'])
            self.retrain(finaltrain=True)

            for ms in self.model.net:
                for m in ms[0].modules():
                    for param in m.parameters():
                        param.require_grads = True



        elif cfg_retrain_stg['keeptrain_mode'] == 'eval_BN':
            self.model.reset_bn()
            self.model.train()
            time.sleep(2)  # process may be stuck in dataloader

            for i in range(cfg_sample_sampler['kwargs']['cal_time']):
                # default sampler tester
                score = self.tester.test()
                time.sleep(2)  # process may be stuck in dataloader
            if self.rank == 0:
                print('==Cal_BN , score: {}'.format(score))


        elif cfg_retrain_stg['keeptrain_mode'] != None:
            raise RuntimeError("cfg_retrain_stg['keeptrain_mode'] not in the support list")

        self.model.eval()
        self.tester.test(test_top1=self.reAlign_top1, test_top5=self.reAlign_top5)

        if self.rank == 0:
            temp_best = 0
            for k, v in self.reAlign_top1.items():
                if v > temp_best:
                    temp_name = k
                    temp_best = v
            print("Best pt is {}, Best top1 is {}, top5 is {}".format(temp_name,
                                                                      self.reAlign_top1['{}'.format(temp_name)],
                                                                      self.reAlign_top5['{}'.format(temp_name)]))
        return

    def make_Alignp(self, cfg_retrain_stg):
        assert 'keeptrain_stg' in cfg_retrain_stg, "Wrong!, keeptrain_stg not in retrain_stg"
        cfg_retrain_stg['keeptrain_stg']['task_type'] = cfg_retrain_stg['task_type']
        cfg_retrain_stg['keeptrain_stg']['snapshot_freq'] = cfg_retrain_stg['snapshot_freq']
        cfg_retrain_stg['keeptrain_stg']['print_freq'] = cfg_retrain_stg['print_freq']
        if cfg_retrain_stg['keeptrain_stg']['use_basic_path']:
            cfg_retrain_stg['keeptrain_stg']['save_path'] = cfg_retrain_stg['save_path'] + '/reAlign'


class Sample_temp():
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])


if __name__ == '__main__':
    import yaml

    config = yaml.load(open('../../config/config.yaml', 'r'))
    print(config)
    cfg_data = config['data']
    training_param = cfg_data.pop('training_param')
    agent = NASAgent(config)
    print(agent.model)
