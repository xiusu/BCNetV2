import time
from os.path import join, exists

import torch
import cellular.pape.distributed as dist
import core.dataset.build_dataloader as BD
from core.searcher.sense_searcher import SenseSearcher, SenseMultiPathSearcher
from tools.trainer.base_trainer import BaseTrainer
from core.utils.flops import Channel_flops, Model_channels
from core.sampler.sampler_model import Sampler_extension, Sampler_deduction
import numpy as np
import random
import math
import copy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImagenetTrainer(BaseTrainer):
    ''' Imagenet Trainer
    requires attrs:
        - in Base Trainer
        (train) search_space, optimizer, lr_scheduler, dataloader, cur_iter
        (log) logger, tb_logger
        (save) print_freq, snapshot_freq, save_path

        - in Customized Trainer
        (dist) rank, world_size
        (train) max_iter
        (time) data_time, forw_time, batch_time
        (loss&acc) <task_name>_disp_loss, <task_name>_disp_acc
        (task) task_key, task_training_shapes
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'Random_layer'):
            self.Random_layer = False

        if not hasattr(self, 'Efficient_topk'):
            self.Efficient_topk = False

        if not hasattr(self, 'Last_record'):
            self.Last_record = self.max_iter   #dafault setting

        if not hasattr(self, 'Equal_train'):
            self.Equal_train = False   #dafault setting


        print("Last Record is {}".format(self.Last_record))

        '''
        if self.rank == 0:
            if hasattr(self, 'Last_record'):
                print("Last_record is {}".format(self.Last_record))
            else:
                print("using default setting for Last_record")
        '''

        # check customized trainer has all required attrs
        self.required_atts = ('rank', 'world_size', 'max_iter',
                              'data_time', 'forw_time', 'batch_time')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTrainer must has attr: {att}')
        self.task_key = 'classification'
        self.logger.info("task key: %s" % (self.task_key))
        if not hasattr(self, 'disp_acc_top1'):
            raise RuntimeError(f'ImagenetTrainer must has attr: disp_acc_top1')
        if not hasattr(self, 'disp_acc_top5'):
            raise RuntimeError(f'ImagenetTrainer must has attr: disp_acc_top5')
        self.logger.info(f'[rank{self.rank}]ImagenetTrainer build done.')
        if self.rank == 0:
            self.logger.info(self.model)

    def train(self, finaltrain = False, eph = 0, train_flag = True):
        self.model.train()
        if self.rank == 0:
            self.logger.info('Start training...')
            self.logger.info(f'Loading classification data')
        loader_iter = iter(BD.DataPrefetcher(self.dataloader))
        input_all = {}
        end_time = time.time()

        if finaltrain == False:
            Total_channels = Model_channels(self.model.net)
            Total_channels = Sampler_deduction(self.skip_list, Total_channels)

            assert isinstance(self.P_train, list), "self.P_train is not list, it is {}".format(self.P_train)

            self.Record_Efficient = []
            if self.Efficient_topk != False:
                for i in range(len(Total_channels)):
                    temp = []
                    for j in range(len(self.P_train)):
                        temp.append([])
                    self.Record_Efficient.append(temp)
            else:
                for i in range(len(Total_channels)):
                    self.Record_Efficient.append([0.1 for _ in range(len(self.P_train))])

            if not train_flag:
                return

        while self.cur_iter <= self.max_iter:
            self.lr_scheduler.step(self.cur_iter)
            tmp_time = time.time()
            images, target = next(loader_iter)
            if images is None:
                epoch = int(self.cur_iter / len(self.dataloader))
                self.logger.info('classification epoch-{} done at iter-{}'.format(epoch, self.cur_iter))
                self.dataloader.sampler.set_epoch(int(self.cur_iter / len(self.dataloader)))
                loader_iter = iter(BD.DataPrefetcher(self.dataloader))
                images, target = loader_iter.next()
            if self.mixed_training:
                images = images.half()
            input_all['images'] = images
            input_all['labels'] = target

            self.data_time.update(time.time() - tmp_time)
            tmp_time = time.time()

            if finaltrain == False:
                if self.Equal_train==False:
                    if self.cur_iter % 2 == 1:
                        Channel_dropout_Left, Channel_dropout_Right = self.Channel_wise_dropout(Total_channels)
                        output = self.model(input_all, c_iter=self.cur_iter, Channel_dropout = Channel_dropout_Left)
                        output_Right = self.model(input_all, c_iter=self.cur_iter, Channel_dropout=Channel_dropout_Right)

                        if isinstance(output['loss'], tuple):
                            raise RuntimeError("it shouldn't be tuple")
                        output['loss'] = (output['loss'] + output_Right['loss']) / 2    ##data add or add directly
                        output['accuracy'] = [(i+j)/2 for i,j in zip(output['accuracy'],output_Right['accuracy'])]

                    elif self.cur_iter % 2 == 0:
                        output = self.model(input_all, c_iter=self.cur_iter)
                    else:
                        raise RuntimeError(f'self.cur_iter not Odd or even')
                elif self.Equal_train==True:
                    if self.cur_iter % 3 == 1:
                        Channel_dropout_Left, Channel_dropout_Right = self.Channel_wise_dropout(Total_channels)
                        #Channel_next_Left = copy.deepcopy(Channel_dropout_Left)
                        #Channel_next_Right = copy.deepcopy(Channel_dropout_Right)
                        Channel_next_Left, Channel_next_Right = self.List_reverse(Channel_dropout_Left, Channel_dropout_Right)


                        output = self.model(input_all, c_iter=self.cur_iter, Channel_dropout=Channel_dropout_Left)
                        output_Right = self.model(input_all, c_iter=self.cur_iter,
                                                  Channel_dropout=Channel_dropout_Right)

                        if isinstance(output['loss'], tuple):
                            raise RuntimeError("it shouldn't be tuple")
                        output['loss'] = (output['loss'] + output_Right['loss']) / 2  ##data add or add directly
                        output['accuracy'] = [(i + j) / 2 for i, j in zip(output['accuracy'], output_Right['accuracy'])]

                    elif self.cur_iter % 3 == 2:
                        Channel_dropout_Left = Channel_next_Left
                        Channel_dropout_Right = Channel_next_Right
                        Channel_next_next_Left = copy.deepcopy(Channel_next_Left)
                        Channel_next_netxt_Right = copy.deepcopy(Channel_next_Right)
                        self.Luck_Index = self.Luck_Index_next

                        output = self.model(input_all, c_iter=self.cur_iter, Channel_dropout=Channel_dropout_Left)
                        output_Right = self.model(input_all, c_iter=self.cur_iter,
                                                  Channel_dropout=Channel_dropout_Right)

                        if isinstance(output['loss'], tuple):
                            raise RuntimeError("it shouldn't be tuple")
                        output['loss'] = (output['loss'] + output_Right['loss']) / 2  ##data add or add directly
                        output['accuracy'] = [(i + j) / 2 for i, j in zip(output['accuracy'], output_Right['accuracy'])]

                        '''
                        elif self.cur_iter % 3 == 3:
                            Channel_dropout_Left = Channel_next_next_Left
                            Channel_dropout_Right = Channel_next_netxt_Right
                            self.Luck_Index = self.Luck_Index_next
    
                            output = self.model(input_all, c_iter=self.cur_iter, Channel_dropout=Channel_dropout_Left)
                            output_Right = self.model(input_all, c_iter=self.cur_iter,
                                                      Channel_dropout=Channel_dropout_Right)
    
                            if isinstance(output['loss'], tuple):
                                raise RuntimeError("it shouldn't be tuple")
                            output['loss'] = (output['loss'] + output_Right['loss']) / 2  ##data add or add directly
                            output['accuracy'] = [(i + j) / 2 for i, j in zip(output['accuracy'], output_Right['accuracy'])]
                        '''
                    elif self.cur_iter % 3 == 0:
                        output = self.model(input_all, c_iter=self.cur_iter)
                    else:
                        raise RuntimeError(f'self.cur_iter not moad 3')

            elif finaltrain:
                output = self.model(input_all, c_iter=self.cur_iter)

            self.forw_time.update(time.time() - tmp_time)

            loss = output['loss']
            reduced_loss = loss.data.clone() / self.world_size
            dist.all_reduce(reduced_loss)
            self.disp_loss.update(reduced_loss.item())

            if finaltrain == False:
                if self.Equal_train == False:
                    if self.cur_iter % 2 != 0:
                    # disp_loss is a norm number or a tensor, copy.deepcopy or just clone?
                        self.Recode_Index_EMA(self.disp_loss.avg)
                elif self.Equal_train == True:
                    if self.cur_iter % 3 != 0:
                    # disp_loss is a norm number or a tensor, copy.deepcopy or just clone?
                        self.Recode_Index_EMA(self.disp_loss.avg)


            if self.task_has_accuracy:
                prec1, prec5 = output['accuracy']
                reduced_prec1 = prec1.clone() / self.world_size
                dist.all_reduce(reduced_prec1)
                reduced_prec5 = prec5.clone() / self.world_size
                dist.all_reduce(reduced_prec5)
                self.disp_acc_top1.update(reduced_prec1.item())
                self.disp_acc_top5.update(reduced_prec5.item())

            self.optimizer.zero_grad()

            if self.mixed_training:
                loss *= self.optimizer.loss_scale
            loss.backward()
            self.model.average_gradients()
            self.optimizer.step()
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()

            if hasattr(self, 'Model_ema'):
                if self.Model_ema.device == 'cpu':
                    if self.rank == 0:
                        self.Model_ema.update(self.model)
                else:
                    self.Model_ema.update(self.model)

            # vis loss
            if self.cur_iter % self.print_freq == 0 and self.rank == 0:
                self.tb_logger.add_scalar('lr', self.lr_scheduler.get_lr()[0], self.cur_iter)
                self.logger.info('Iter: [{0}/{1}] '
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                                 'Loss {loss.avg:.4f} | '
                                 'Prec@1 {top1.avg:.3f} | '
                                 'Prec@5 {top5.avg:.3f} | '
                                 'Total {batch_time.all:.2f}hrs | '
                                 'LR {lr:.6f} ETA {eta:.2f} hrs'.format(
                    self.cur_iter, self.max_iter,
                    batch_time=self.batch_time,
                    data_time=self.data_time,
                    loss=self.disp_loss,
                    top1=self.disp_acc_top1,
                    top5=self.disp_acc_top5,
                    lr=self.lr_scheduler.get_lr()[0],
                    eta=self.batch_time.avg * (self.max_iter - self.cur_iter) / 3600))

                self.tb_logger.add_scalar('batch_time', self.batch_time.val, self.cur_iter)
                self.logger.info('\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=self.disp_loss))
                self.tb_logger.add_scalar('loss', getattr(self, 'disp_loss').val, self.cur_iter)
                self.logger.info('\tPrec@1 {acc1.val:.4f} ({acc1.avg:.4f})'.format(acc1=self.disp_acc_top1))
                self.tb_logger.add_scalar('acc1_train', getattr(self, 'disp_acc_top1').val, self.cur_iter)
                self.logger.info('\tPrec@5 {acc5.val:.4f} ({acc5.avg:.4f})'.format(acc5=self.disp_acc_top5))
                self.tb_logger.add_scalar('acc5_train', self.disp_acc_top5.val, self.cur_iter)
                if output['c_searcher'] is not None:
                    self.logger.info('\tcurrent searcher: {}'.format(output['c_searcher'].__class__))
                if output['c_searcher'] is not None and output['c_searcher'].__class__ in [SenseSearcher, SenseMultiPathSearcher]:
                    searcher = output['c_searcher']
                    self.logger.info('\tcurrent best arch: (net:{}/cand:{}/in out:{} p: {:.4f})'.format(
                        searcher.cands.sample_from_net, searcher.cands.sample_from_cand, searcher.cands.in_out, searcher.p_cur))
                    c_best_p = searcher.get_topk_path(1)
                    if c_best_p is not None:
                        c_best_p = c_best_p[0]
                        self.logger.info('\t arch: {}, score: {:.6f}'.format(c_best_p[0][:len(c_best_p[0])//2], c_best_p[1]))
                        self.logger.info('\t channel: {}'.format(c_best_p[0][len(c_best_p[0])//2:]))
                    self.logger.info('\tcurrent number of candidates {}'.format(len(searcher.cands)))
                self.logger.info('--------------------')

            # save search_space
            if self.cur_iter % self.snapshot_freq == 0 or (self.cur_iter >= self.Last_record and self.cur_iter % 200 == 0):  #742000
                time.sleep(1)
                if self.rank == 0:
                    self.save()
                    if self.cur_iter >= self.Last_record and self.cur_iter % 200 == 0 and hasattr(self, 'Model_ema'):
                        self.save_ema()
            self.cur_iter += 1
  
            # early stopping
            if output.get('c_searcher') is not None and getattr(output['c_searcher'], 'early_stop', False):
                break

        if self.Efficient_topk != False:
            self.Topk_mean()


        # finish training
        self.logger.info('Finish training {} iterations.'.format(self.cur_iter))

    def save(self):
        ''' save search_space '''
        path = join(self.save_path, 'iter_{}_ckpt.pth.tar'.format(self.cur_iter))
        torch.save({'step': self.cur_iter, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
        self.logger.info('[rank{}]Saved search_space to {}.'.format(self.rank, path))

    def save_ema(self):
        ''' save search_space '''
        path = join(self.save_path, 'ema_iter_{}_ckpt.pth.tar'.format(self.cur_iter))
        torch.save({'step': self.cur_iter, 'state_dict': self.Model_ema.ema.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
        self.logger.info('[rank{}]Saved search_space to {}.'.format(self.rank, path))

    def load(self, ckpt_path):
        ''' load search_space and optimizer '''

        def map_func(storage, location):
            return storage.cuda()

        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        ckpt = torch.load(ckpt_path, map_location=map_func)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        ckpt_keys = set(ckpt['state_dict'].keys())
        own_keys = set(self.model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            if self.rank == 0:
                self.logger.info(f'**missing key while loading search_space**: {k}')
                raise RuntimeError(f'**missing key while loading search_space**: {k}')

        # load optimizer
        self.cur_iter = ckpt['step'] + 1
        epoch = int(self.cur_iter / len(self.dataloader))
        self.dataloader.sampler.set_epoch(epoch)
        if self.rank == 0:
            self.logger.info('load [resume] search_space done, '
                             f'current iter is {self.cur_iter}')

    def check_eq(self, bin_size_list, Total_channels):
        # only support div by 3 and 5, can expend

        assert len(bin_size_list) == len(Total_channels), "Total_channels and bin_size_list must have the same length, bin_size_list is {}, Total_channels is {}".format(bin_size_list, Total_channels)
        for i in range(len(bin_size_list)):
            if Total_channels[i] % bin_size_list[i] != 0:
                if (Total_channels[i] % 9 == 0) and (bin_size_list[i] % 8 == 0):
                    bin_size_list[i] = int(bin_size_list[i] * 9 / 8)
                    if Total_channels[i] % bin_size_list[i] == 0:
                        continue

                if (Total_channels[i] % 5 == 0) and (bin_size_list[i] % 4 == 0):
                    bin_size_list[i] = int(bin_size_list[i] * 5 / 4)
                    if Total_channels[i] % bin_size_list[i] == 0:
                        continue

                assert Total_channels[i] % bin_size_list[i] == 0, "Total_channels and bin_size_list must have the same length, Loc is{}, bin_size_list is {}, Total_channels is {}".format(
                    i, bin_size_list, Total_channels)

    def FLOPs_aware(self, FLOPs_list):
        min_list = min(FLOPs_list)
        max_list = max(FLOPs_list)
        FLOPs_aware_list = [ math.ceil(math.log(1 / i * max_list, 2)) for i in FLOPs_list]
        FLOPs_aware_list = [ math.pow(2, i) for i in FLOPs_aware_list]
        FLOPs_aware_list = [ max(i, self.min_bin) for i in FLOPs_aware_list]
        bin_size_list = [min(i, self.max_bin) for i in FLOPs_aware_list]

        return bin_size_list

    def Channel_wise_dropout(self, Total_channels):
    # luck index only need rank 0 for sampling
        Total_list_Left = []
        Total_list_Right = []
        Total_channels_copy = copy.deepcopy(Total_channels)

        self.Luck_Index = []
        while len(Total_channels_copy) > 0:
            if isinstance(self.P_train, list):
                Index = [i for i in range(len(self.P_train))]
                Luck_Index = random.sample(Index, 1)[0]
                Total_Percent = self.P_train[Luck_Index]
                self.Luck_Index.append(Luck_Index)
            else:
                raise RuntimeError("self.P_train is not a training list")

            T_channels = Total_channels_copy.pop(0)

            Save_oup = int(math.ceil(T_channels * Total_Percent))
            temp_list_left = [0] * T_channels

            for i in range(Save_oup):
                temp_list_left[i] = 1

            if self.rank == 0:
                Channel_list_left = torch.IntTensor(temp_list_left)
            else:
                Channel_list_left = torch.zeros([len(temp_list_left)], dtype=torch.int32)
            dist.broadcast(Channel_list_left, 0)
            Channel_list_left = Channel_list_left.tolist()

            Channel_list_right = copy.deepcopy(Channel_list_left)
            Channel_list_right.reverse()

            Total_list_Left.append(Channel_list_left)
            Total_list_Right.append(Channel_list_right)

        Total_list_Left = Sampler_extension(self.skip_list, Total_list_Left)
        Total_list_Right = Sampler_extension(self.skip_list, Total_list_Right)


        if self.Equal_train == True:
            # luck index only need rank 0 for sampling
            Len = len(self.P_train) - 1
            self.Luck_Index_next = [Len - i for i in self.Luck_Index]

        return Total_list_Left, Total_list_Right

    def Recode_Index_EMA(self, Loss):
        assert len(self.Luck_Index)>0, "self.Luck_Index len is 0, self.Luck_Index is {}".format(self.Luck_Index)
        if self.Efficient_topk != False:
            for i in range(len(self.Luck_Index)):
                if len(self.Record_Efficient[i][self.Luck_Index[i]]) < self.Efficient_topk:
                    self.Record_Efficient[i][self.Luck_Index[i]].append(Loss)
                else:
                    self.Record_Efficient[i][self.Luck_Index[i]].append(Loss)
                    self.Record_Efficient[i][self.Luck_Index[i]].remove(max(self.Record_Efficient[i][self.Luck_Index[i]]))

        else:
            for i in range(len(self.Luck_Index)):
                self.Record_Efficient[i][self.Luck_Index[i]] = self.EMA * self.Record_Efficient[i][self.Luck_Index[i]] + (1 - self.EMA) * Loss

    def Topk_mean(self):
        for i in range(len(self.Record_Efficient)):
            for j in range(len(self.P_train)):
                self.Record_Efficient[i][j] = sum(self.Record_Efficient[i][j]) / len(self.Record_Efficient[i][j])
        if self.rank == 0:
            print("Efficient_topk is {}, Record_Efficient is {}".format(self.Efficient_topk, self.Record_Efficient))

    def List_reverse(self, Channel_dropout_Left, Channel_dropout_Right):
        def list_R(a):
            List = []
            for i in a:
                temp = []
                for j in i:
                    if j == 0:
                        temp.append(1)
                    elif j == 1:
                        temp.append(0)
                    else:
                        raise RuntimeError("wrong j {}, list is {}".format(j, a))
                List.append(temp)
            return List

        Channel_next_Left = copy.deepcopy(Channel_dropout_Right)
        Channel_next_Right = copy.deepcopy(Channel_dropout_Left)


        Channel_next_Left = list_R(Channel_next_Left)

        Channel_next_Right = list_R(Channel_next_Right)

        return Channel_next_Left, Channel_next_Right







