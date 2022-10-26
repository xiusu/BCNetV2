import cellular.pape.distributed as dist
from tensorboardX import SummaryWriter
from core.utils.misc import AverageMeter
from core.utils.logger import create_logger
from tools.trainer.imagenet.trainer import ImagenetTrainer
from tools.trainer.imagenet.model_ema import ModelEma


def build_trainer(cfg_stg, dataloader, model, optimizer, lr_scheduler, now, **kwargs):
    ''' Build trainer and return '''
    # choose trainer function
    kwargs = dict(**kwargs)
    kwargs['rank'] = dist.get_rank()
    kwargs['world_size'] = dist.get_world_size()
    kwargs['max_iter'] = cfg_stg['max_iter']
    kwargs['quantization'] = cfg_stg.get('quantization', None)
    kwargs['data_time'] = AverageMeter(20)
    kwargs['forw_time'] = AverageMeter(20)
    kwargs['batch_time'] = AverageMeter(20)
    kwargs['mixed_training'] = cfg_stg.get('mixed_training', False)

    #kwargs['min_bin'] = cfg_stg.get('min_bin', 1) #default = 1
    #kwargs['max_bin'] = cfg_stg.get('max_bin', 32) #default = 1
    #kwargs['P_train'] = cfg_stg.get('P_train', 0.5)  # default = 1

    #kwargs['task_keys'] = search_space.search_space.task_names
    if cfg_stg['task_type'] in ['imagenet']:
        trainer = ImagenetTrainer
        kwargs['disp_loss'] = AverageMeter()
        kwargs['disp_acc_top1'] = AverageMeter()
        kwargs['disp_acc_top5'] = AverageMeter()
        kwargs['task_has_accuracy'] = True #search_space.head.task_has_accuracy
    else:
        raise RuntimeError('task_type {} invalid, must be in imagenet'.format(cfg_stg['task_type']))

    if 'bin_search' in cfg_stg and cfg_stg['bin_search'] == True:
        for k in cfg_stg['bin_config']:
            kwargs[k] = cfg_stg['bin_config'][k]



    if 'ema_flag' in cfg_stg and cfg_stg['ema_flag']:
        Model_Ema = ModelEma( kwargs['model_ema'], decay=cfg_stg['ema']['model_ema_decay'],
            device='cpu' if cfg_stg['ema']['model_ema_force_cpu'] else '',
            resume='')
        kwargs['Model_ema'] = Model_Ema


    if 'Last_record' in cfg_stg:
        kwargs['Last_record'] = cfg_stg['Last_record']


    if now != '':
        now = '_' + now

    # build logger
    if cfg_stg['task_type'] in ['verify']:
        logger = create_logger('global_logger',
                               '{}/log/log_task{}_train{}.txt'.format(cfg_stg['save_path'], now, model.task_id))
    # TRACKING_TIP
    elif cfg_stg['task_type'] in ['attribute', 'gaze', 'imagenet', 'tracking', 'smoking']:
        logger = create_logger('global_logger',
                               '{}/log/'.format(cfg_stg['save_path']) + '/log_train{}.txt'.format(now))
    else:
        raise RuntimeError('task_type musk be in verify/attribute/gaze/imagenet/tracking')
    tb_logger = SummaryWriter('{}/events'.format(cfg_stg['save_path']))

    # build trainer

    final_trainer = trainer(dataloader, model, optimizer, lr_scheduler,
                            cfg_stg.get('print_freq', 20), cfg_stg['save_path'] + '/checkpoint',
                            cfg_stg.get('snapshot_freq', 5000), logger, tb_logger,
                            **kwargs)
    return final_trainer


if __name__ == '__main__':
    pass
