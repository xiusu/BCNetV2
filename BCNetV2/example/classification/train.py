import os
import shutil
import torch
import torch.nn as nn
import yaml
import subprocess
import logging
import time
import random
import importlib
import numpy as np
from addict import Dict
import torchvision.datasets as datasets

from lib.utils.args import parse_args

from lib.models.nas_model import gen_nas_model
from lib.models.darts_model import gen_darts_model
from lib.models import resnet
from lib.models.loss import CrossEntropyLabelSmooth
from lib.dataset.dataset import McDataset
from lib.dataset.dataloader import fast_collate, DataPrefetcher
from lib.dataset import transform
from lib.utils.scheduler import build_scheduler
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.optim import build_optimizer
from lib.utils.measure import get_params, get_flops

'''dist'''
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = f'experiments/{args.model}'
    if args.model in ['nas_model', 'darts_model']:
        args.exp_dir += '-' + args.model_config.split('/')[-1].split('.')[0]
    if args.edgenn_config != '':
        import edgenn
        args.exp_dir = args.exp_dir[:args.exp_dir.find('/')+1] + 'edgenn/' + args.exp_dir[args.exp_dir.find('/')+1:]
        args.exp_dir += '-' + args.edgenn_config.split('/')[-1].split('.')[0]
        with open(args.edgenn_config, 'r') as f:
            edgenn_cfgs = Dict(yaml.safe_load(f))
    if args.experiment != '':
        args.exp_dir += '-' + args.experiment
    args.exp_dir += '-' + time.strftime("%Y%m%d-%H%M%S", time.localtime())

    '''distributed'''
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.slurm:
        args.distributed = True
    if not args.distributed:
        # task with single GPU also needs to use distributed module
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        args.local_rank = 0
        args.distributed = True

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        if args.slurm:
            proc_id = int(os.environ['SLURM_PROCID'])
            ntasks = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f'scontrol show hostname {node_list} | head -n1')
            os.environ['MASTER_ADDR'] = addr
            os.environ['WORLD_SIZE'] = str(ntasks)
            args.local_rank = proc_id % num_gpus
            os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
            os.environ['RANK'] = str(proc_id)
            logging.info('Using slurm with master node: {}, rank: {}, world size: {}'.format(addr, proc_id, ntasks))

        os.environ['MASTER_PORT'] = args.dist_port
        args.device = 'cuda:%d' % args.local_rank
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if not args.slurm:
            torch.cuda.set_device(args.rank)
        logger.info('Training in distributed model with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on 1 GPU.')

    if args.rank == 0:
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.exp_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
        # save args
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        if args.edgenn_config != '':
            shutil.copy(args.edgenn_config, os.path.join(args.exp_dir, 'edgenn_config.yaml'))
        logger.info(f'Experiment directory: {args.exp_dir}')

    else:
        logger.setLevel(logging.ERROR)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''create dataloader'''
    if args.dataset == 'imagenet':
        args.data_path = 'data/imagenet' if args.data_path == '' else args.data_path
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)
    elif args.dataset == 'cifar10':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
    elif args.dataset == 'cifar100':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 100
        args.input_shape = (3, 32, 32)

    if args.dataset == 'imagenet':
        train_transforms_l, train_transforms_r = transform.build_train_transforms(args.aa, args.color_jitter, args.reprob, args.remode)
        train_dataset = McDataset(os.path.join(args.data_path, 'train'), os.path.join(args.data_path, 'meta/train.txt'), transform=train_transforms_l)
    elif args.dataset == 'cifar10':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10()    # TODO: add cutout
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transforms_l)
    elif args.dataset == 'cifar100':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10()    # TODO: add cutout
        train_dataset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transforms_l)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
        pin_memory=False, sampler=train_sampler, collate_fn=fast_collate)
    train_loader = DataPrefetcher(train_loader, train_transforms_r)

    if args.dataset == 'imagenet':
        val_transforms_l, val_transforms_r = transform.build_val_transforms()
        val_dataset = McDataset(os.path.join(args.data_path, 'val'), os.path.join(args.data_path, 'meta/val.txt'), transform=val_transforms_l)
    elif args.dataset == 'cifar10':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10()
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transforms_l)
    elif args.dataset == 'cifar100':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10()
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=val_transforms_l)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch_size * args.val_batch_size_multiplier), 
        shuffle=False, num_workers=args.workers, pin_memory=False, 
        sampler=val_sampler, collate_fn=fast_collate)
    val_loader = DataPrefetcher(val_loader, val_transforms_r)

    '''build model'''
    if args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=args.smoothing).cuda()
    if args.edgenn_config == '':
        # normal model
        if args.model.lower() == 'nas_model':
            model = gen_nas_model(yaml.safe_load(open(args.model_config, 'r')), drop_rate=args.drop, 
                                  drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)
        elif args.model.lower() == 'darts_model':
            model = gen_darts_model(yaml.safe_load(open(args.model_config, 'r')), args.dataset, drop_rate=args.drop, 
                                    drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)
        elif args.model.lower() == 'nas_pruning_model':
            from edgenn.models import EdgeNNModel
            model_config = yaml.safe_load(open(args.model_config, 'r'))
            channel_settings = model_config.pop('channel_settings')
            model = gen_nas_model(model_config, drop_rate=args.drop, drop_path_rate=args.drop_path_rate, auxiliary_head=args.auxiliary)
            edgenn_model = EdgeNNModel(model, loss_fn=None, pruning=True, input_shape=args.input_shape)
            logger.info(edgenn_model.graph)
            edgenn_model.fold_dynamic_nn(channel_settings['choices'], channel_settings['bins'], channel_settings['min_bins'])
            logger.info(model)
        elif args.model.lower().startswith('resnet'):
            model = getattr(resnet, args.model.lower())(num_classes=args.num_classes)
        else:
            raise NotImplementedError(f'Model {args.model} not implemented.')
        logger.info(
            f'Model {args.model} created, params: {get_params(model)}, FLOPs: {get_flops(model, input_shape=args.input_shape)}')
    else:
        # edgenn model
        if args.model.lower() in ['nas_model', 'nas_pruning_model']:
            model = gen_nas_model(yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader), drop_rate=args.drop, drop_path_rate=args.drop_path_rate)
            model = edgenn.models.EdgeNNModel(model, loss_fn, pruning=(args.model=='nas_pruning_model'))
        elif args.model.lower().startswith('resnet'):
            model = getattr(resnet, args.model.lower())(num_classes=args.num_classes)
            model = edgenn.models.EdgeNNModel(model, loss_fn, pruning=True)
        elif args.model == 'edgenn':
            model = edgenn.build_model(edgenn_cfgs.model)
        else:
            raise NotImplementedError(f'Model {args.model} not implemented.')

    model.cuda()

    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=(args.edgenn_config!=''))
    logger.info('Created DDP model')

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''create optimizer'''
    optimizer = build_optimizer(args.opt, model, args.lr, eps=args.opt_eps, momentum=args.momentum, 
                                weight_decay=args.weight_decay, filter_bias_and_bn=not args.opt_no_filter, nesterov=not args.sgd_no_nesterov)
    # scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(args.sched, optimizer, warmup_steps, args.warmup_lr, decay_steps, args.decay_rate, 
                                total_steps, steps_per_epoch=steps_per_epoch, decay_by_epoch=args.decay_by_epoch, min_lr=args.min_lr)

    '''create edgenn trainer'''
    if args.edgenn_config != '':
        trainer = edgenn.build_trainer(edgenn_cfgs.trainer)
        if 'darts' in trainer.__class__.__name__.lower() or hasattr(trainer, 'train_evaluator'):
            val_sampler_ = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
            val_loader_ = torch.utils.data.DataLoader(
                val_dataset, batch_size=int(args.batch_size * args.val_batch_size_multiplier), 
                shuffle=False, num_workers=args.workers, pin_memory=False, 
                sampler=val_sampler_, collate_fn=fast_collate)
            val_loader_ = DataPrefetcher(val_loader_, val_transforms_r)
            trainer.set_val_loader(val_loader_)
        if 'darts' in trainer.__class__.__name__.lower():
            trainer.set_optimizer(optimizer)
    else:
        trainer = None

    '''resume'''
    ckpt_manager = CheckpointManager(model, optimizer, ema_model=model_ema, save_dir=args.exp_dir, rank=args.rank, additions={'edgenn_trainer': trainer})
    
    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        scheduler.step(start_epoch * len(train_loader))
        logger.info(f'Resume ckpt {args.resume} done, start training from epoch {start_epoch}')
    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None

    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = args.drop_path_rate * epoch / args.epochs

        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader, 
            optimizer, loss_fn, scheduler, auxiliary_buffer, trainer)

        # validate
        test_metrics = validate(args, epoch, model, val_loader, loss_fn)
        if model_ema is not None:
            test_metrics = validate(args, epoch, model_ema.module, val_loader, loss_fn, log_suffix='(EMA)')

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics)
        logger.info('\n'.join(['Checkpoints:'] + ['        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts]))

    '''edgenn search'''
    if trainer is not None:
        trainer.search(model, train_loader, val_loader)


def train_epoch(args, epoch, model, model_ema, loader, optimizer, loss_fn, scheduler, auxiliary_buffer=None, edgenn_trainer=None):
    loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    for batch_idx, (input, target) in enumerate(loader):
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        optimizer.zero_grad()
        if edgenn_trainer is None:
            output = model(input)
            loss = loss_fn(output, target)
        else:
            output, loss = edgenn_trainer.forward(model, input, target)

        if auxiliary_buffer is not None:
            loss_aux = loss_fn(auxiliary_buffer.output, target)
            loss += loss_aux * auxiliary_buffer.loss_weight

        loss.backward()
        if args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.6f} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s'
                            .format(epoch, batch_idx, len(loader), 
                                    loss=loss_m, lr=optimizer.param_groups[0]['lr'], 
                                    batch_time=batch_time_m, data_time=data_time_m))
        scheduler.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()
        
    return {'train_loss': loss_m.avg}


def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            if 'edgenn' in model.module.__class__.__name__.lower():
                output, loss = model(input, target)
            else:
                output = model(input)
                loss = loss_fn(output, target)
        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1*100, n=input.size(0))
        top5_m.update(top5*100, n=input.size(0))
        
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'
                            .format(log_suffix, epoch, batch_idx, len(loader), 
                                    loss=loss_m, top1=top1_m, top5=top5_m, 
                                    batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()




