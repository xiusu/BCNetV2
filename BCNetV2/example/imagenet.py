import argparse
import math
import os
import random
import shutil
import time
import yaml
import warnings
import io
import re
import logging
import socket
from PIL import Image
import numpy as np
from addict import Dict


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import mc
import edgenn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path of edgenn config (default: none)')


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class McDataset(Dataset):
    r"""
    Dataset using memcached to read data.

    Arguments
        * root (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``.
        * transform (callable, optional): A function that transforms the given PIL image
          and returns a transformed image.
    """
    def __init__(self, root, meta_file, transform=None):
        self.root = root
        self.transform = transform
        with open(meta_file) as f:
            meta_list = f.readlines()
        self.num = len(meta_list)
        self.metas = []
        for line in meta_list:
            path, cls = line.strip().split()
            self.metas.append((path, int(cls)))
        self.initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        filename = self.root + '/' + self.metas[index][0]
        cls = self.metas[index][1]

        # memcached
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_buf)
        with Image.open(buff) as img:
            img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger_all = logging.getLogger('all')

def main():
    args = parser.parse_args()
    cfgs = Dict(yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader))

    if 'seed' in cfgs and cfgs.seed is not None:
        random.seed(cfgs.seed)
        torch.manual_seed(cfgs.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # setup environment
    cfgs.rank = int(os.environ['SLURM_PROCID'])
    cfgs.world_size = int(os.environ['SLURM_NTASKS'])
    cfgs.local_rank = int(os.environ['SLURM_LOCALID'])
 
    node_list = str(os.environ['SLURM_NODELIST'])
    node_parts = re.findall('[0-9]+', node_list)
    cfgs.master_addr = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
    job_id = int(os.environ["SLURM_JOBID"])
    cfgs.master_port = f'2{job_id%10000:04}'
 
    os.environ['MASTER_ADDR'] = str(cfgs.master_addr)
    os.environ['MASTER_PORT'] = str(cfgs.master_port)
    os.environ['WORLD_SIZE'] = str(cfgs.world_size)
    os.environ['RANK'] = str(cfgs.rank)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(cfgs.local_rank) 
 
    if cfgs.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)
 
    logger_all.info(f"JOB {job_id}, {cfgs.rank} of {cfgs.world_size} in {socket.gethostname()}")
    dist.barrier()
 
    if cfgs.rank == 0:
        cfgs.tb_writer = SummaryWriter(cfgs.tensorboard.root)

    # create dataloader
    if cfgs.data.type.lower() in ['cifar', 'cifar10']:
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        if cfgs.data.cutout:
            train_transform.transforms.append(Cutout(cfgs.data.cutout_length))

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = datasets.CIFAR10(root=cfgs.data.root, train=True, download=False, transform=train_transform)
        if 'train_portion' in cfgs.data and cfgs.data.train_portion > 0.:
            train_num = int(len(train_dataset) * cfgs.data.train_portion)
            val_num = len(train_dataset) - train_num
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_num, val_num])
            val_dataset.transform=val_transform
        else:
            val_dataset = datasets.CIFAR10(root=cfgs.data.root, train=False, download=False, transform=val_transform)
    elif cfgs.data.type.lower() == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = McDataset(
            cfgs.data.traindir, cfgs.data.trainlabel,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                #transforms.RandomResizedCrop(224, scale=(0.25, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                #Lighting(0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = McDataset(
            cfgs.data.valdir, cfgs.data.vallabel,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfgs.batch_size, shuffle=(train_sampler is None),
        num_workers=cfgs.workers, pin_memory=True, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfgs.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=val_sampler)

    # create model
    model = edgenn.build_model(cfgs.model)
    model.cuda()
    model = DDP(model, device_ids=[cfgs.local_rank],
                find_unused_parameters=True)
    logger.info(f"=> model\n{model}")
 
    #from edgenn.algorithm.utils import get_flops
    #logger.info(f"=> model flops: {get_flops(model)}")

    if 'weight_decay_adjust' in cfgs and cfgs.weight_decay_adjust is True:
        normal_params = []
        no_wd_params = []
        for param in model.parameters():
            ps = param.shape
            # only normal conv and fc have weight decay
            # depthwise conv, bias, bn param don't have weight decay
            if (len(ps) == 4 and ps[1] != 1) or len(ps) == 2:
                normal_params.append(param)
            else:
                no_wd_params.append(param)
        optimizer = torch.optim.SGD([{'params': normal_params, 'weight_decay': cfgs.weight_decay},
                                     {'params': no_wd_params, 'weight_decay': 0}],
                                    cfgs.lr_scheduler.lr,
                                    momentum=cfgs.momentum,
                                    nesterov=cfgs.nesterov)

    else:
        optimizer = torch.optim.SGD(model.parameters(), cfgs.lr_scheduler.lr,
                                    momentum=cfgs.momentum,
                                    weight_decay=cfgs.weight_decay,
                                    nesterov=cfgs.nesterov)
    logger.info(f"=> optimizer\n{optimizer}")

    trainer = edgenn.build_trainer(cfgs.trainer)
    if 'darts' in trainer.__class__.__name__.lower() or hasattr(trainer, 'train_evaluator'):
        trainer.set_val_loader(val_loader)

    best_acc1 = 0
    if cfgs.resume:
        if os.path.isfile(cfgs.resume):
            logger.info(f"=> loading checkpoint '{cfgs.resume}'")
            checkpoint = torch.load(cfgs.resume, map_location="cpu")
            cfgs.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.load_state_dict(checkpoint['trainer'])
            logger.info(f"=> resume from epoch {checkpoint['epoch']}")
        else:
            logger.info(f"=> no checkpoint found at '{cfgs.resume}'")

    if cfgs.evaluate:
        validate(val_loader, model, cfgs)
        return

    for epoch in range(cfgs.start_epoch, cfgs.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, cfgs.epochs, cfgs.lr_scheduler)

        if 'dartseval' in model.module.__class__.__name__.lower():
            model.module.update_drop_path_prob(epoch, cfgs.epochs)
            logger.info(f"drop path prob is {model.module.backbone.drop_path_prob}")
        if 'darts' in trainer.__class__.__name__.lower():
            trainer.epoch = epoch

        # train for one epoch
        train(train_loader, model, optimizer, trainer, epoch, cfgs)

        if epoch != cfgs.epochs - 1 and 'val_freq' in cfgs and epoch % cfgs.val_freq != 0:
            continue

        # evaluate on validation set
        if 'autoslim' in trainer.__class__.__name__.lower():
            choice_list = model.module.backbone.get_choice()
            max_subnet = [choices[-1] for choices in choice_list]
            model.module.backbone.set_choice(max_subnet)
        acc1 = validate(val_loader, model, cfgs)
        if cfgs.rank == 0:
            cfgs.tb_writer.add_scalar('Test/top1', acc1, epoch)
            cfgs.tb_writer.flush()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if cfgs.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfgs.model,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'optimizer': optimizer.state_dict(),
                'trainer': trainer.state_dict()
            }, is_best, cfgs.save)
        if 'darts' in trainer.__class__.__name__.lower():
            if trainer.early_stop(model):
                break
    if 'darts' in trainer.__class__.__name__.lower():
        trainer.search(model)
    else:
        trainer.search(model, train_loader, val_loader)
    if cfgs.rank == 0:
        cfgs.tb_writer.close()


def train(train_loader, model, optimizer, trainer, epoch, cfgs):
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.3f')
    top5 = AverageMeter('Acc@5', ':.3f')
    memory = AverageMeter('Mem(MB)', ':.0f', 1)
    lr = AverageMeter('lr', ':.5f', 1)
    lr.update(optimizer.param_groups[0]['lr'])
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, memory, lr],
        prefix=f"Epoch: [{epoch+1}/{cfgs.epochs}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # compute output
        if 'darts' in trainer.__class__.__name__.lower():
            output, loss = trainer.forward(model, optimizer, images, target)
        else:
            output, loss = trainer.forward(model, images, target)

        # measure accuracy and record loss
        acc1, acc5, batch_size = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1 * 100. / batch_size)
        top5.update(acc5 * 100. / batch_size)
        memory.update(torch.cuda.max_memory_allocated()/1024/1024)

        # compute gradient and do SGD step
        if 'autoslim' not in trainer.__class__.__name__.lower():
            loss.backward()
        if 'grad_clip' in cfgs:
            nn.utils.clip_grad_norm_(model.parameters(), cfgs.grad_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfgs.print_freq == 0:
            progress.display(i)
            if cfgs.rank == 0:
                cfgs.tb_writer.add_scalar('Train/loss', loss.item(), epoch * len(train_loader) + i)
                cfgs.tb_writer.add_scalar('Train/top1', acc1 * 100. / batch_size, epoch * len(train_loader) + i)
                cfgs.tb_writer.flush()

        #if i == 20:
        #    break


def validate(val_loader, model, cfgs):
    batch_time = AverageMeter('Time', ':.3f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.3f', 1)
    top5 = AverageMeter('Acc@5', ':.3f', 1)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        full_acc1, full_acc5, full_batch_size = 0, 0, 0
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output, loss = model(images, target)

            # measure accuracy and record loss
            acc1, acc5, batch_size = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item())
            top1.update(acc1 * 100. / batch_size)
            top5.update(acc5 * 100. / batch_size)
            full_acc1 += acc1
            full_acc5 += acc5
            full_batch_size += batch_size

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfgs.print_freq == 0:
                progress.display(i)
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
    return full_top1


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def save_checkpoint(state, is_best, path):
    filename = os.path.join(path, f"epoch_{state['epoch']}_acc1_{state['acc1']}.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(path, f"best_model.pth")
        shutil.copyfile(filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value
       When length = 0 , save all history data """

    def __init__(self, name, fmt=':f', length=100):
        assert length >= 0
        self.name = name
        self.fmt = fmt
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        elif self.length == 0:
            self.count = 0
            self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        elif self.length < 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        if self.length == 1:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, max_epochs, cfg):
    if epoch < cfg.warmup_epochs:
        lr = cfg.lr * (epoch + 1) / (cfg.warmup_epochs + 1)
    else:
        if cfg.type == 'cosine':
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * (epoch - cfg.warmup_epochs) / (max_epochs - cfg.warmup_epochs)))
        elif cfg.type == 'linear':
            lr = cfg.lr * (1 - (epoch - cfg.warmup_epochs) / (max_epochs - cfg.warmup_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    


def accuracy(output, target, topk=(1,)):
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


if __name__ == '__main__':
    main()
