if __name__ == '__main__':
    from  base_searcher import BaseSearcher
else:
    from .base_searcher import BaseSearcher
from core.dataset.build_dataloader import build_dataloader
from core.dataset.datasets.base_mc_dataset import BaseMcDataset, img_loader
import torch
import cellular.pape.distributed as dist
from core.search_space.ops import channel_mults
from core.dataset.augmentations.augmentation import augmentation_cv
from torch.utils.data import DataLoader
from collections import defaultdict
import core.dataset.build_dataloader as BD
import core.dataset.samplers as samplers
import json, os
import random
import mc
import numpy as np
from core.utils.arch_util import _decode_arch
import pickle
from core.utils.flops import count_flops


class UniformDataset(BaseMcDataset):
    def __init__(self, cfg, transform=None, preprocessor='cv'):
        super().__init__(preprocessor)
        self.prefix = cfg['prefix']
        self.transform = transform
        self.cfg = cfg
        self.parse_json_()

    def parse_json_(self):
        # print('loading json file: {}'.format(self.cfg['json_path']))
        jdata = json.load(open(self.cfg['json_path'], 'r'))

        # print('building dataset from %s: %d images' % (self.prefix, self.num))
        self.data_dict = defaultdict(list)
        self.img_labels = []
        self.img_fs = []
        for idx, data in enumerate(jdata[list(jdata.keys())[0]]):
            label = data['annos']['imagenet']['classification']
            img_f = os.path.join(self.prefix, data['img_info']['filename'])
            self.data_dict[label].append(idx)
            self.img_labels.append(label)
            self.img_fs.append(img_f)

        self.batch_info = []
        for i in range(self.n_batch):
            batch_ids = []
            for k in self.data_dict:
                idx = random.randint(0, len(self.data_dict[k]) - 1)
                batch_ids.append(idx)
            self.batch_info.append(batch_ids)
        self.num = len(self.batch_info)

    def __getitem__(self, idx):
        # memcached
        self._init_memcached()
        batch_imgs = []
        clss = []
        for f_idx in self.batch_info[idx]:
            filename = self.img_fs[f_idx]
            cls = self.img_labels[f_idx]
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            img = img_loader(value_str, self.preprocessor)
            # transform
            if self.transform is not None:
                img = self.transform(**{'image': img})
                img = img['image']
            batch_imgs.append(img)
            clss.append(cls)
        return np.asarray(batch_imgs), np.asarray(clss)


class Candidate(list):

    def __init__(self, n, log_path=None, rank=-1, not_changed_topk=100):
        super().__init__()
        self.n = n
        self.total = 0
        self._idx = []
        self.sample_from_net = 0
        self.sample_from_cand = 0
        self.FLAG = False
        self.in_out = 0
        self.logger = None
        self.rank = rank
        self.iter = 0
        self.not_changed_topk = not_changed_topk
        if log_path is not None and self.rank == 0:
            self.logger = open(log_path, 'a')
            self.logger.write(f'iter,score,ori_idx,new_idx,total,subnet,flops\n')

    def _add(self, c, flops=0):
        assert isinstance(c, tuple)
        inserted = False

        # smooth score
        ori_idx = -1
        new_idx = -1
        if c[0] in self._idx:
            t_idx = self._idx.index(c[0])
            ori_idx = t_idx
            c = (c[0], (self[t_idx][1] + c[1]) / 2.)
            del self[t_idx]
            del self._idx[t_idx]
            self.total -= 1

        # insert c
        for i in range(self.total):
            _c_score = self[i][1]
            if c[1] > _c_score:
                self.insert(i, c)
                self._idx.insert(i, c[0])
                self.total += 1
                inserted = True
                new_idx = i
                break

        if not inserted and self.total < self.n:
            # insert to tail
            self.append(c)
            self._idx.append(c[0])
            self.total += 1
            new_idx = self.total - 1

        if inserted and self.FLAG and self.total >= self.n:
            self.in_out += 1

        if self.logger is not None:
            self.logger.write(f'{self.iter},{c[1]},{ori_idx},{new_idx},{self.total},{c[0]},{flops}\n')

        if self.total > self.n:
            self.total -= 1
            if self.logger is not None:
                self.logger.write(f'{self.iter},{self[-1][1]},{self.total - 1},{-1},{self.total},{self._idx[-1]},{0}\n')
            del self[-1]
            del self._idx[-1]
  
        if self.logger is not None:
            self.logger.flush()
        self.iter += 1

        # for early stopping
        if (ori_idx == -1 or ori_idx > self.not_changed_topk - 1) and new_idx < self.not_changed_topk and new_idx != -1:
            return True
        return False


class SenseSearcher(BaseSearcher):

    def __init__(self, **kwargs):
        super(SenseSearcher, self).__init__()
        self.rank = dist.get_rank()
        # verbose
        if self.rank == 0:
            print('==Sense Searcher config:') 
            print(kwargs)

        self.world_size = dist.get_world_size()
        self.init_iter = 0
        self.batch_eval_data_idx = 0
        self.old_batch_eval_data_idx = -1
        self.p_init = kwargs.get('p_init', 0.)
        self.topk = kwargs.get('topk', 5)
        self.p_max = kwargs.get('p_max', 1.)
        self.p_stg = kwargs.get('p_stg', 'linear')
        self.snapshot_freq = kwargs.get('snapshot_freq', 2000)
        self.arch_save_path = os.path.join(kwargs['arch_save_path'], 'arch')
        self.start_iter = kwargs.get('start_iter', 0)
        self.flops_constrant = kwargs.get('flops_constrant', 400e6)
        self.enable_early_stop = kwargs.get('enable_early_stop', False)
        self.early_stop = False
        self.early_stop_num = kwargs.get('early_stop_num', 500)
        self.not_changed_topk = kwargs.get('not_changed_topk', 100)
        self.not_changed = 0

        if not os.path.exists(self.arch_save_path) and self.rank == 0:
            os.makedirs(self.arch_save_path)
        self.cands = Candidate(kwargs.get('n_cand', 50000),
                               log_path=os.path.join(self.arch_save_path, 'arch.log'), rank=self.rank)

        if kwargs.get('resume_arch', False):
            self.load_arch(kwargs.get('load_name'))
        else:
            self.init_iter = kwargs.get('cur_iter', 0)
        self.p_cur = self.p_init
        self.cur_iter = self.init_iter
        if self.p_stg in ['linear', 'cosine']:
            assert 'max_iter' in kwargs
            self.max_iter = kwargs['max_iter']
            if self.p_stg == 'linear':
                self.scale = (self.p_max - self.p_init) / self.max_iter
                if self.cur_iter >= self.start_iter:
                    self.p_cur += self.scale * (self.cur_iter - self.start_iter)
                else:
                    self.p_cur = 0
            if self.p_stg == 'cosine':
                assert 'cos_step' in kwargs
                self.cos_step = kwargs['cos_step']
        elif self.p_stg == 'step':
            assert 'p_steps' in kwargs and 'p_mults' in kwargs
            self.p_steps = kwargs['p_steps']
            self.p_mults = kwargs['p_mults']
            for idx, step in enumerate(self.p_steps):
                if self.cur_iter < step:
                    break
                self.p_init += self.p_mults[idx]
            assert len(self.p_steps) == len(self.p_mults)
        else:
            raise ValueError(f'Invalid p stg {self.p_stg}, use linear/step/cosine instead')
        self.max_path = kwargs.get('max_path', 1)
        self.n_batch = kwargs.get('n_batch', 10000)

        self.aug = None
        self.eval_dataset = self._init_eval_dataset(kwargs['eval_dataset'])

        self.images, self.labels = None, None  # val data buffer

    def get_best_arch(self):
        best_arch = self.cands._idx[:self.topk]
        if self.rank == 0:
            print('====best arch===')
            for arch in best_arch:
                print(arch)
        return _decode_arch(best_arch)

    def load_arch(self, load_name):
        l_f = os.path.join(self.arch_save_path, load_name)
        arch = pickle.load(open(l_f, 'rb'))
        self.cands = arch['cands']
        self.init_iter = int(load_name.split('iter_')[-1].split('_arch')[0])

    def save_arch(self):
        logger = self.cands.logger
        self.cands.logger = None
        f = os.path.join(self.arch_save_path,  'iter_{}_arch.info'.format(self.cur_iter))
        pickle.dump({'cands': self.cands}, open(f, 'wb'))
        self.cands.logger = logger

    def _init_eval_dataset(self, cfg_data):

        # _dataset = UniformDataset(cfg_data['imagenet'], augmentation_cv(cfg_data['augmentation']))
        # _sampler = samplers.DistributedSampler(_dataset, dist.get_world_size(), dist.get_rank())
        # self.data_loader = DataLoader(_dataset,
        #                 shuffle=False,
        #                 batch_size=1,
        #                 num_workers=cfg_data.get('workers', 0),
        #                 sampler=_sampler,
        #                 pin_memory=False
        #                 )
        self.data_loader = build_dataloader(cfg_data, is_test=True)
        self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))
        # if self.rank == 0:
        #     img_fs = []
        #     img_labels = []
        #     self.npy_save_path = data_cfg['npy_save_path']
        #     if os.path.exists(self.npy_save_path) and not data_cfg['overwrite']:
        #         return
        #     if not os.path.exists(self.npy_save_path):
        #         os.makedirs(self.npy_save_path)
        #     data_dict = defaultdict(list)
        #     aug_param = data_cfg.get('augmentation', None)
        #     if aug_param is not None:
        #         t_aug = augmentation_cv(aug_param)
        #     else:
        #         t_aug = None
        #     meta_info = data_cfg['imagenet']
        #     json_path = meta_info['json_path']
        #     prefix = meta_info['prefix']
        #     json_data = json.load(open(json_path))
        #     for idx, data in enumerate(json_data[list(json_data.keys())[0]]):
        #         label = data['annos']['imagenet']['classification']
        #         img_f = os.path.join(prefix, data['img_info']['filename'])
        #         data_dict[label].append(idx)
        #         img_labels.append(label)
        #         img_fs.append(img_f)
        #     self.batch_info = []
        #     for i in range(self.n_batch):
        #         d_f = os.path.join(self.npy_save_path, 'batch_{}_data.npy'.format(i + 1))
        #         batch_ids = []
        #         b_img_datas = []
        #         b_img_labels = []
        #         for k in data_dict:
        #             idx = random.randint(0, len(data_dict[k]) - 1)
        #             img = cv2.imread(img_fs[idx])
        #             if t_aug is not None:
        #                 img = t_aug(**{'image': img})['image'].detach().numpy()
        #             b_img_datas.append(img)
        #             b_img_labels.append(img_labels[idx])
        #             batch_ids.append(idx)
        #         b_img_datas = np.asarray(b_img_datas)
        #         b_img_labels = np.asarray(b_img_labels)
        #         np.savez(d_f, images=b_img_datas, labels=b_img_labels)
        #         self.batch_info.append(batch_ids)

    def _eval_path(self, subnet, model, batch_idx):
        # load_f = os.path.join(self.npy_save_path, 'batch_{}_data.npy'.format(batch_idx))
        # data = np.load(load_f)
        # images = torch.FloatTensor(data['images'])
        # labels = torch.LongTensor(data['labels'])
        if self.cur_iter % 100 == 0 or self.images is None:
            images, labels = next(self.iter_loader)
            self.images, self.labels = images, labels
        else:
            images, labels = self.images, self.labels
        if images is None:
            self.data_loader.sampler.epoch += 1
            self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))
            images, labels = next(self.iter_loader)
        input_all = {}
        input_all['images'] = images
        input_all['labels'] = labels
        with torch.no_grad():
            output = model(input_all, subnet)
        loss = output['loss']
        reduced_loss = loss.data.clone() / self.world_size
        dist.all_reduce(reduced_loss)
        score = 1./ reduced_loss
        return score.item()

    def get_topk_path(self, k):
        if len(self.cands) < k:
            return None
        return self.cands[:k]

    def _update_prob(self):
        self.cur_iter += 1
        if self.p_stg == 'step':
            try:
                pos = self.p_steps.index(self.cur_iter)
                self.p_cur += self.p_mults[pos]
            except ValueError:
                pass
        elif self.p_stg == 'linear':
            if self.cur_iter >= self.start_iter:
                self.p_cur += self.scale
        elif self.p_stg == 'cosine':
            if self.cur_iter >= self.start_iter:
                c_step = min(self.cur_iter - self.start_iter, self.cos_step)
                t = (1 - self.p_init) * 0.5 * (1 + np.cos(
                    np.pi * c_step / self.cos_step))
                self.p_cur = 1 - t

    def generate_subnet(self, model):

        batch_path = []
        # if self.rank == 0:
        #     batch_eval_data_idx = random.randint(1, self.n_batch)
        #     batch_eval_data_idx = torch.IntTensor([batch_eval_data_idx])
        # else:
        #     batch_eval_data_idx = torch.zeros([1], dtype=torch.int32)
        # dist.broadcast(batch_eval_data_idx, 0)
        # batch_eval_data_idx = batch_eval_data_idx.item()
        self._update_prob()
        if self.cur_iter % self.snapshot_freq == 0 and self.rank == 0:
            self.save_arch()
        self.batch_eval_data_idx = (self.cur_iter // 100) % self.n_batch
        while len(batch_path) < self.max_path:
            if self.rank == 0:
                if random.random() > self.p_cur or len(self.cands) == 0:
                    self.cands.FLAG = True
                    self.cands.sample_from_net += 1
                    subnet = []
                    for block in model.net:
                        subnet.append(random.randint(0, len(block)-1))
                    for idx, block in enumerate(model.net):
                        if getattr(block[0], 'channel_search', False):
                            subnet.append(random.randint(0, len(channel_mults) - 1))
                        else:
                            subnet.append(channel_mults.index(1.0))
                    subnet = torch.IntTensor(subnet)
                else:
                    self.cands.FLAG = False
                    self.cands.sample_from_cand += 1
                    rand_idx = random.randint(0, len(self.cands) - 1)
                    subnet = torch.IntTensor(self.cands[rand_idx][0])
            else:
                subnet = torch.zeros([len(model.net) * 2], dtype=torch.int32)
            dist.broadcast(subnet, 0)
            subnet = subnet.tolist()
            batch_path.append((subnet, self._eval_path(subnet, model, self.batch_eval_data_idx)))
            ''' rubbish
            model.subnet = subnet  # hard code      
            flops = count_ops(model, torch.zeros([1, 3, 224, 224]).cuda(), ignore_layers=['BatchNorm2d', 'DynamicBatchNorm2d', 'ReLU'], verbose=False, print_readable=False)[0]
            if flops < self.flops_constrant:
                print('nanisubnet: {}, FLOPs: {}'.format(subnet, flops))
                batch_path.append((subnet, self._eval_path(subnet, model, self.batch_eval_data_idx)))
            '''
        chosen_path = sorted(batch_path, key=lambda a: a[1])[-1]
        flops = count_flops(model.net, chosen_path[0])
        if flops < self.flops_constrant:
            res = self.cands._add(chosen_path)
            if res:  # is True
                self.not_changed = 0
            else:
                self.not_changed += 1
            if self.not_changed >= self.early_stop_num and self.enable_early_stop:
                self.early_stop = True
        return chosen_path[0]


class SenseMultiPathSearcher(BaseSearcher):

    def __init__(self, **kwargs):
        super(SenseMultiPathSearcher, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # multi-path
        multi_path = kwargs.get('multi_path', {})
        self.sample_num = multi_path.get('sample_num', 10)
        self.top_k = multi_path.get('top_k', 8)
        self.path_pool = []

        self.init_iter = 0
        self.batch_eval_data_idx = 0
        self.old_batch_eval_data_idx = -1
        self.p_init = kwargs.get('p_init', 0.)
        self.topk = kwargs.get('topk', 5)
        self.p_max = kwargs.get('p_max', 1.)
        self.p_stg = kwargs.get('p_stg', 'linear')
        self.snapshot_freq = kwargs.get('snapshot_freq', 2000)
        self.arch_save_path = os.path.join(kwargs['arch_save_path'], 'arch')
        self.start_iter = kwargs.get('start_iter', 0)
        self.flops_constrant = kwargs.get('flops_constrant', 400e6)

        if not os.path.exists(self.arch_save_path) and self.rank == 0:
            os.makedirs(self.arch_save_path)
        self.cands = Candidate(kwargs.get('n_cand', 50000),
                               log_path=os.path.join(self.arch_save_path, 'arch.log'), rank=self.rank)

        if kwargs.get('resume_arch', False):
            self.load_arch(kwargs.get('load_name'))
        else:
            self.init_iter = kwargs.get('cur_iter', 0)
        self.p_cur = self.p_init
        self.cur_iter = self.init_iter
        if self.p_stg in ['linear', 'cosine']:
            assert 'max_iter' in kwargs
            self.max_iter = kwargs['max_iter']
            if self.p_stg == 'linear':
                self.scale = (self.p_max - self.p_init) / self.max_iter
                if self.cur_iter >= self.start_iter:
                    self.p_cur += self.scale * (self.cur_iter - self.start_iter)
                else:
                    self.p_cur = 0
            if self.p_stg == 'cosine':
                assert 'cos_step' in kwargs
                self.cos_step = kwargs['cos_step']
        elif self.p_stg == 'step':
            assert 'p_steps' in kwargs and 'p_mults' in kwargs
            self.p_steps = kwargs['p_steps']
            self.p_mults = kwargs['p_mults']
            for idx, step in enumerate(self.p_steps):
                if self.cur_iter < step:
                    break
                self.p_init += self.p_mults[idx]
            assert len(self.p_steps) == len(self.p_mults)
        else:
            raise ValueError(f'Invalid p stg {self.p_stg}, use linear/step/cosine instead')
        self.max_path = kwargs.get('max_path', 1)
        self.n_batch = kwargs.get('n_batch', 10000)

        self.aug = None
        self.eval_dataset = self._init_eval_dataset(kwargs['eval_dataset'])

        self.images, self.labels = None, None  # val data buffer

    def get_best_arch(self):
        best_arch = self.cands._idx[:self.topk]
        if self.rank == 0:
            print('====best arch===')
            for arch in best_arch:
                print(arch)
        return _decode_arch(best_arch)

    def load_arch(self, load_name):
        l_f = os.path.join(self.arch_save_path, load_name)
        arch = pickle.load(open(l_f, 'rb'))
        self.cands = arch['cands']
        self.init_iter = int(load_name.split('iter_')[-1].split('_arch')[0])

    def save_arch(self):
        logger = self.cands.logger
        self.cands.logger = None
        f = os.path.join(self.arch_save_path,  'iter_{}_arch.info'.format(self.cur_iter))
        pickle.dump({'cands': self.cands}, open(f, 'wb'))
        self.cands.logger = logger

    def _init_eval_dataset(self, cfg_data):

        # _dataset = UniformDataset(cfg_data['imagenet'], augmentation_cv(cfg_data['augmentation']))
        # _sampler = samplers.DistributedSampler(_dataset, dist.get_world_size(), dist.get_rank())
        # self.data_loader = DataLoader(_dataset,
        #                 shuffle=False,
        #                 batch_size=1,
        #                 num_workers=cfg_data.get('workers', 0),
        #                 sampler=_sampler,
        #                 pin_memory=False
        #                 )
        self.data_loader = build_dataloader(cfg_data, is_test=True)
        self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))
        # if self.rank == 0:
        #     img_fs = []
        #     img_labels = []
        #     self.npy_save_path = data_cfg['npy_save_path']
        #     if os.path.exists(self.npy_save_path) and not data_cfg['overwrite']:
        #         return
        #     if not os.path.exists(self.npy_save_path):
        #         os.makedirs(self.npy_save_path)
        #     data_dict = defaultdict(list)
        #     aug_param = data_cfg.get('augmentation', None)
        #     if aug_param is not None:
        #         t_aug = augmentation_cv(aug_param)
        #     else:
        #         t_aug = None
        #     meta_info = data_cfg['imagenet']
        #     json_path = meta_info['json_path']
        #     prefix = meta_info['prefix']
        #     json_data = json.load(open(json_path))
        #     for idx, data in enumerate(json_data[list(json_data.keys())[0]]):
        #         label = data['annos']['imagenet']['classification']
        #         img_f = os.path.join(prefix, data['img_info']['filename'])
        #         data_dict[label].append(idx)
        #         img_labels.append(label)
        #         img_fs.append(img_f)
        #     self.batch_info = []
        #     for i in range(self.n_batch):
        #         d_f = os.path.join(self.npy_save_path, 'batch_{}_data.npy'.format(i + 1))
        #         batch_ids = []
        #         b_img_datas = []
        #         b_img_labels = []
        #         for k in data_dict:
        #             idx = random.randint(0, len(data_dict[k]) - 1)
        #             img = cv2.imread(img_fs[idx])
        #             if t_aug is not None:
        #                 img = t_aug(**{'image': img})['image'].detach().numpy()
        #             b_img_datas.append(img)
        #             b_img_labels.append(img_labels[idx])
        #             batch_ids.append(idx)
        #         b_img_datas = np.asarray(b_img_datas)
        #         b_img_labels = np.asarray(b_img_labels)
        #         np.savez(d_f, images=b_img_datas, labels=b_img_labels)
        #         self.batch_info.append(batch_ids)

    def _eval_path(self, subnet, model, batch_idx):
        # load_f = os.path.join(self.npy_save_path, 'batch_{}_data.npy'.format(batch_idx))
        # data = np.load(load_f)
        # images = torch.FloatTensor(data['images'])
        # labels = torch.LongTensor(data['labels'])
        if self.cur_iter % 100 == 0 or self.images is None:
            images, labels = next(self.iter_loader)
            self.images, self.labels = images, labels
        else:
            images, labels = self.images, self.labels
        if images is None:
            self.data_loader.sampler.epoch += 1
            self.iter_loader = iter(BD.DataPrefetcher(self.data_loader))
            images, labels = next(self.iter_loader)
        input_all = {}
        input_all['images'] = images
        input_all['labels'] = labels
        with torch.no_grad():
            output = model(input_all, subnet)
        loss = output['loss']
        reduced_loss = loss.data.clone() / self.world_size
        dist.all_reduce(reduced_loss)
        score = 1./ reduced_loss
        return score.item()

    def get_topk_path(self, k):
        if len(self.cands) < k:
            return None
        return self.cands[:k]

    def _update_prob(self):
        self.cur_iter += 1
        if self.p_stg == 'step':
            try:
                pos = self.p_steps.index(self.cur_iter)
                self.p_cur += self.p_mults[pos]
            except ValueError:
                pass
        elif self.p_stg == 'linear':
            if self.cur_iter >= self.start_iter:
                self.p_cur += self.scale
        elif self.p_stg == 'cosine':
            if self.cur_iter >= self.start_iter:
                c_step = min(self.cur_iter - self.start_iter, self.cos_step)
                t = (1 - self.p_init) * 0.5 * (1 + np.cos(
                    np.pi * c_step / self.cos_step))
                self.p_cur = 1 - t

    def _generate_subnet(self, model):

        batch_path = []

        self.batch_eval_data_idx = (self.cur_iter // 100) % self.n_batch
        while len(batch_path) < self.sample_num:
            if self.rank == 0:
                if random.random() > self.p_cur or len(self.cands) == 0:
                    self.cands.FLAG = True
                    self.cands.sample_from_net += 1
                    subnet = []
                    for block in model.net:
                        subnet.append(random.randint(0, len(block)-1))
                    for idx, block in enumerate(model.net):
                        if getattr(block[0], 'channel_search', False):
                            subnet.append(random.randint(0, len(channel_mults) - 1))
                        else:
                            subnet.append(channel_mults.index(1.0))
                    subnet = torch.IntTensor(subnet)
                else:
                    self.cands.FLAG = False
                    self.cands.sample_from_cand += 1
                    rand_idx = random.randint(0, len(self.cands) - 1)
                    subnet = torch.IntTensor(self.cands[rand_idx][0])
            else:
                subnet = torch.zeros([len(model.net) * 2], dtype=torch.int32)
            dist.broadcast(subnet, 0)
            subnet = subnet.tolist()
            batch_path.append((subnet, self._eval_path(subnet, model, self.batch_eval_data_idx)))
        batch_path = sorted(batch_path, key=lambda a: a[1], reverse=True)
        return batch_path

    def generate_subnet(self, model):
        self._update_prob()
        if self.cur_iter % self.snapshot_freq == 0 and self.rank == 0:
            self.save_arch()
        if len(self.path_pool) == 0:
            subnets = self._generate_subnet(model)
            self.path_pool.extend([x[0] for x in subnets[:self.top_k]])
            for subnet in subnets[:self.top_k]:
                self.cands.iter += 1
                flops = count_flops(model.net, subnet[0])
                if flops < self.flops_constrant:
                    self.cands._add(subnet, flops=flops)
                    self.cands.iter -= 1
        return self.path_pool.pop(0)
