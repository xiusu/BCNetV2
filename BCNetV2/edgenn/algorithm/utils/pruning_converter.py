import torch
import torch.nn as nn
from ...models.pruning_ops import DynamicConv2d, DynamicLinear, DynamicBatchNorm2d


def convert_conv(module: nn.Module, conv_func=DynamicConv2d, **kwargs):
    for key, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            setattr(module, key, conv_func(m.in_channels, m.out_channels, 
                                           m.kernel_size, m.stride, m.padding, 
                                           m.dilation, m.groups, m.bias is not None, 
                                           **kwargs))
        else:
            convert_conv(m, conv_func)


def convert_linear(module: nn.Module, linear_func=DynamicLinear, **kwargs):
    for key, m in module.named_children():
        if isinstance(m, nn.Linear):
            setattr(module, key, linear_func(m.in_features, m.out_features, m.bias is not None, **kwargs))
        else:
            convert_linear(m, linear_func)


def convert_bn(module: nn.Module, bn_func=DynamicBatchNorm2d, **kwargs):
    for key, m in module.named_children():
        if isinstance(m, nn.BatchNorm2d):
            setattr(module, key, bn_func(m.num_features, m.eps, m.momentum, 
                                         m.affine, m.track_running_stats, **kwargs))
        else:
            convert_bn(m, bn_func)

