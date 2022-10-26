import os
import time
from os.path import join, exists

import torch
import cellular.pape.distributed as dist
from core.utils.misc import AverageMeter
import core.dataset.build_dataloader as BD
from tools.eval.base_tester import BaseTester
import copy

class ImagenetTester(BaseTester):
    ''' Multi-Source Tester: test multi dataset one by one
    requires attrs:
    - in Base Tester
    (load) model_folder, model_name
    (config) cfg_data[with all datasets neet to be tested], cfg_stg[build dataloader]

    - in Customized Tester
    (dist) rank, world_size
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.required_atts = ('rank', 'world_size')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTester must has attr: {att}')
        self.dataloader = None

    def test(self, subnet=None, **kwargs):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        ''' test setted model in self '''
        if not self.model_loaded:
            self.load()
            self.model_loaded = True
        if self.dataloader is None:
            self.dataloader = self.gen_dataloader()
        dataloader = iter(BD.DataPrefetcher(self.dataloader))
        input, target = next(dataloader)
        i = 0
        end_time = time.time()
        if 'Channel_dropout' in kwargs:
            Channel_temp = copy.deepcopy(kwargs['Channel_dropout'])
        '''
        if 'Sample_test_num' in kwargs and kwargs['Sample_test_num'] != None:
            assert 1==2
            Sample_num = 0
        '''
        with torch.no_grad():
            while input is not None:
                '''
                if 'Sample_test_num' in kwargs and kwargs['Sample_test_num'] != None:
                    Sample_num += 1
                    if (Sample_num >= kwargs['Sample_test_num']):
                        break
                '''
                input_all = {}
                input_all['images'] = input
                input_all['labels'] = target
                if 'Channel_dropout' in kwargs:
                    kwargs['Channel_dropout'] = copy.deepcopy(Channel_temp)
                output = self.model(input_all, subnet, **kwargs)
                loss = output['loss']
                reduced_loss = loss.data.clone() / self.world_size
                dist.all_reduce(reduced_loss)
                prec1, prec5 = output['accuracy']
                reduced_prec1 = prec1.clone() / self.world_size
                dist.all_reduce(reduced_prec1)
                reduced_prec5 = prec5.clone() / self.world_size
                dist.all_reduce(reduced_prec5)

                losses.update(reduced_loss.item())
                top1.update(reduced_prec1.item())
                top5.update(reduced_prec5.item())
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                if self.rank == 0 and i % 10 == 0 and False:
                    print('Test: [{0}/{1}] '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'
                          'Prec@1 {acc1.val:.4f} ({acc1.avg:.4f})'
                          'Prec@5 {acc5.val:.4f} ({acc5.avg:.4f})'.
                          format(i, len(dataloader.loader),
                                 batch_time=batch_time,
                                 loss=losses,
                                 acc1=top1,
                                 acc5=top5))
                    print('--------------------')
                input, target = next(dataloader)
                i += 1

            if self.rank == 0:
                print(' * Prec@1 {acc1.avg:.3f} Prec@5 {acc5.avg:.3f}'.format(acc1=top1, acc5=top5))
                self.save_eval_result(top1.avg, top5.avg)
            if 'test_top1' in kwargs:
                kwargs['test_top1']['{}'.format(self.model_name)] = top1.avg
                kwargs['test_top5']['{}'.format(self.model_name)] = top5.avg
            return top1.avg

    def gen_dataloader(self):
        return self.dataloader_fun(self.cfg_data)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= self.world_size
        return rt

    def load(self):
        if self.model_loaded or self.model_name is None:
            return
        # load state_dict
        ckpt_path = join(self.model_folder, self.model_name)
        print ("load ckpt path {}".format(ckpt_path))
        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        if self.rank == 0:
            print(f'==[rank{self.rank}]==loading checkpoint from {ckpt_path}')

        def map_func(storage, location):
            return storage.cuda()

        ckpt = torch.load(ckpt_path, map_location=map_func)
        from collections import OrderedDict
        fixed_ckpt = OrderedDict()
        for k in ckpt['state_dict']:
            if 'head' in k:
                k1 = k.replace('classification_head', 'head')
                fixed_ckpt[k1] = ckpt['state_dict'][k]
                continue
            fixed_ckpt[k] = ckpt['state_dict'][k]
        ckpt['state_dict'] = fixed_ckpt
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        ckpt_keys = set(ckpt['state_dict'].keys())
        own_keys = set(self.model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        #for k in missing_keys:
        #    print(f'==[rank{self.rank}]==**missing key**:{k}')
        if self.rank == 0:
            print(f'==[rank{self.rank}]==load model done.')


    def eval_init(self):
        self.eval_t = 0.0
        self.eval_f = 0.0

    def save_eval_result(self, acc1, acc5):

        if not exists(join(self.model_folder, f'test_result')):
            os.makedirs(join(self.model_folder, f'test_result'))
        fid = open(join(self.model_folder, f'test_result.txt'), 'a+')
        result_line = f'ckpt: {self.model_name}\tacc1: {acc1:.4f}\tacc5: {acc5:.4f}'
        print(f'==[rank{self.rank}]=={result_line}')
        fid.write(f'{result_line}\n')
        fid.close()
