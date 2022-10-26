import logging
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import BaseEvaluator
from ..utils import EvaluatorReg

logger = logging.getLogger()


@EvaluatorReg.register_module('imagenet')
class ImageNet(BaseEvaluator):
    def __init__(self, recal_bn_iters=0, bn_training_mode=False, print_freq=20):
        super(ImageNet, self).__init__()
        self.recal_bn_iters = recal_bn_iters
        self.bn_training_mode = bn_training_mode
        self.print_freq = print_freq

    def recal_bn(self, model, train_loader):
        if self.recal_bn_iters > 0:
            # recal bn
            logger.info(f'recalculating bn stats {self.recal_bn_iters} iters')
            for mod in model.modules():
                if isinstance(mod, nn.BatchNorm2d) or issubclass(mod.__class__, nn.BatchNorm2d):
                    mod.reset_running_stats()
                    # for small recal_bn_iters like 20, must set mod.momentum = None
                    # for big recal_bn_iters like 300, mod.momentum can be 0.1
                    mod.momentum = None

            model.train()
            with torch.no_grad():
                cnt = 0
                while cnt < self.recal_bn_iters:
                    for i, (images, target) in enumerate(train_loader):
                        images = images.cuda()
                        target = target.cuda()
                        output, loss = model(images, target)
                        cnt += 1
                        if i % self.print_freq == 0 or cnt == self.recal_bn_iters:
                            logger.info(f'recal bn iter {i} loss {loss:.3f}')
                        if cnt >= self.recal_bn_iters:
                            break
                    time.sleep(2)

        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d) or issubclass(mod.__class__, nn.BatchNorm2d):
                if mod.track_running_stats:
                    dist.all_reduce(mod.running_mean)
                    dist.all_reduce(mod.running_var)
                    mod.running_mean /= dist.get_world_size()
                    mod.running_var /= dist.get_world_size()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
    
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k.item())
            res.append(batch_size)
            return res

    def eval(self, model, train_loader, val_loader):
        if self.bn_training_mode:
            # switch to evaluate mode
            model.eval()
            if self.bn_training_mode:
                for mod in model.modules():
                    if isinstance(mod, nn.BatchNorm2d) or issubclass(mod.__class__, nn.BatchNorm2d):
                        mod.reset_running_stats()
                        mod.training = True
        else:
            self.recal_bn(model, train_loader)
            model.eval()

        logger.info('evaluating model')
        with torch.no_grad():
            full_acc1, full_acc5, full_batch_size = 0, 0, 0
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                output, loss = model(images, target)

                # measure accuracy and record loss
                acc1, acc5, batch_size = self.accuracy(output, target, topk=(1, 5))
                loss = loss.item()
                top1 = acc1 * 100. / batch_size
                top5 = acc5 * 100. / batch_size
                full_acc1 += acc1
                full_acc5 += acc5
                full_batch_size += batch_size

                if i % self.print_freq == 0:
                    logger.info(f'iter {i} loss {loss:.3f} top1 {top1:.3f} top5 {top5:.3f}')
                #if i == 20:
                #    break
            stats = torch.tensor([full_acc1, full_acc5, full_batch_size], dtype=torch.int32, device='cuda')
            dist.all_reduce(stats)
            full_acc1 = stats[0].item()
            full_acc5 = stats[1].item()
            full_batch_size = stats[2].item()

            # TODO: this should also be done with the ProgressMeter
            full_top1 = full_acc1 * 100. / full_batch_size
            full_top5 = full_acc5 * 100. / full_batch_size
            logger.info(f' * Acc@1 {full_top1:.3f} Acc@5 {full_top5:.3f}')
        return full_top1, full_top5
