if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import datasets
    import samplers
else:
    import core.dataset.datasets as datasets
    import core.dataset.samplers as samplers
import time
import torch
import cellular.pape.distributed as dist
from torch.utils.data import DataLoader, Dataset
from .augmentations.augmentation import augmentation_cv



class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_items = next(self.loader)
        except StopIteration:
            self.next_items = [None for _ in self.next_items]
            return self.next_items
        except:
            raise RuntimeError('load data error')

        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_items)):
                if not isinstance(self.next_items[i][0], str):
                    self.next_items[i] = self.next_items[i].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_items = self.next_items
        self.preload()
        return next_items

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def build_dataset(cfg_data, transforms, preprocessor):
    ''' cfg_data is a dict, contains one or more datasets of one task '''
    if 'batch_size' not in cfg_data:
        # this place need to be changed for imagenet and cifar10
        cfg_data = cfg_data['imagenet']
    dataset_fun = datasets.NormalDataset
    # print(f'building dataset using {dataset_fun}')
    final_dataset = dataset_fun(cfg_data, transforms, preprocessor)
    return final_dataset


def build_sampler(dataset, is_test=False):
    sampler = samplers.DistributedSampler if not is_test else samplers.DistributedTestSampler
    final_sampler = sampler(dataset, dist.get_world_size(), dist.get_rank())
    return final_sampler


def build_dataloader(cfg_data, is_test=False):
    ''' Build dataloader for train and test
    For multi-source task, return a dict.
    For other task and test, return a data loader.
    '''
    transform_param = cfg_data.get('augmentation')


    rank = dist.get_rank()
    preprocessor = transform_param.get('preprocessor', 'cv')
    transforms = augmentation_cv(transform_param)

    dataset = build_dataset(cfg_data, transforms, preprocessor)
    sampler = build_sampler(dataset, is_test)

    if dataset is None:
        dataloader = None
    else:
        #cfg_data = list(cfg_data.values())[0]
        if 'batch_size' not in cfg_data:
            batch_size = cfg_data['imagenet']['batch_size']
        else:
            batch_size = cfg_data['batch_size']
        str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print(f'==[rank{rank}]==creating [ssst] dataloader for imagenet,'
        #       f' [{str_time}]')
        dl = DataLoader(dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=cfg_data.get('workers', 0),
                        sampler=sampler,
                        pin_memory=False
                        )
        dataloader = dl

    return dataloader
