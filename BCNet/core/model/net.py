
from core.utils.misc import get_cls_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.search_space import init_model
from core.search_space.ops import FC

# this place name from where?
if __name__ == '__main__':
    from basemodel import BaseModel
else:
    from .basemodel import BaseModel


class Net(BaseModel):
    def __init__(self, cfg_net, **kwargs):
        super(Net, self).__init__()
        self.loss_type = cfg_net['loss_type']
        self.net = init_model(cfg_net, **kwargs)
        self.subnet = None  # hard code

        assert self.loss_type in ['softmax', 's-softmax']
        self._init_params()

    def get_loss(self, logits, label):
        if self.loss_type == 'softmax':
            label = label.long()
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            loss = criterion(logits, label)
        elif self.loss_type == 's-softmax':
            label = label.long()
            predict = logits
            batch_size = predict.size(0)
            num_class = predict.size(1)
            label_smooth = torch.zeros((batch_size, num_class)).cuda()
            label_smooth.scatter_(1, label.unsqueeze(1), 1)
            ones_idx = label_smooth == 1
            zeros_idx = label_smooth == 0
            label_smooth[ones_idx] = 0.9
            label_smooth[zeros_idx] = 0.1 / (num_class - 1)
            loss = -torch.sum(F.log_softmax(predict, 1) * label_smooth.type_as(predict)) / batch_size
        return loss

    def forward(self, input, subnet=None, c_iter=None, **kwargs):

        if isinstance(input, dict):
            x = input['images']
        else:
            x = input

        if self.subnet is not None:
            subnet = self.subnet
            self.subnet = None

        # search
        c_searcher = None
        if self.searcher is not None and subnet is None:# and not self.searcher.searched:
            if c_iter is None:
                raise RuntimeError('param c_iter can not be None in search mode.')
            searcher_keys = list(self.searcher.keys())
            searcher_keys.sort(reverse=True)
            for s_iter in searcher_keys:
                if s_iter < c_iter:
                    c_searcher = self.searcher[s_iter]
                    break
            assert c_searcher is not None
            subnet = c_searcher.generate_subnet(self)

        # sample
        if subnet is not None:
            self.set_subnet(subnet)
            assert len(subnet) == len(self.net), "subnet is {}, net is {}".format(subnet,len(self.net))
            for idx, c_mult_idx, block in zip(subnet[:len(self.net)], subnet[len(self.net):], self.net):
                block[idx].c_mult_idx = c_mult_idx
                x = block[idx](x)
            logits = x
        # retrain && test
        else:
            logits = self.normal_step(x, **kwargs)
        if isinstance(input, dict) and 'labels' in input:
            accuracy = get_cls_accuracy(logits,
                                        input['labels'],
                                        topk=(1, 5))
            loss = self.get_loss(logits, input['labels'])
        elif not isinstance(input, dict):  # for count ops
            return logits
        else:
            accuracy = -1
            loss = -1


        output = {'output': logits, 'accuracy': accuracy, 'loss': loss, 'c_searcher': c_searcher}
        return output

    def normal_step(self, x, **kwargs):
        for block in self.net:
            total_op = len(block)
            assert total_op == 1
            #print(len(kwargs['Channel_dropout']))
            #print(block[0])
            if isinstance(block[0], (nn.AdaptiveAvgPool2d, FC, nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                x = block[0](x)
            else:
                x = block[0](x, **kwargs)
        return x

