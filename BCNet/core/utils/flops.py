import torch
import torch.nn as nn
import pickle
import yaml
import copy
from core.search_space.ops import InvertedResidual, FC, Conv2d, SqueezeExcite


def count_flops(model, subnet=None, input_shape=[3, 224, 224], FLOPs_list = False):
    if subnet is None:
        subnet = [0] * len(model)
    flops = []
    m_list = []
    skip = 0
    for ms, idx in zip(model, subnet):
        for m in ms[idx].modules():
            if isinstance(m, SqueezeExcite):
                skip = 2
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if skip == 0:
                    m_list.append(m)
                else:
                    flops.append(m.in_channels * m.out_channels)
                    skip -= 1
            elif isinstance(m, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                m_list.append(m)

    c, w, h = input_shape

    for m in m_list:
        if isinstance(m, nn.Conv2d):
            c = m.out_channels
            if m.kernel_size[0] != 1:
                w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
                h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]

            flops.append(m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            flops.append(m.in_features * m.out_features)
        elif isinstance(m, nn.MaxPool2d):
            w = (w + m.padding * 2 - m.kernel_size + 1) // m.stride
            h = (h + m.padding * 2 - m.kernel_size + 1) // m.stride
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            w = m.output_size
            h = m.output_size
    if FLOPs_list == False:
        return sum(flops)
    else:
        return flops

#single channel_flops
def Channel_flops(model, subnet=None, input_shape=[3, 224, 224]):
    if subnet is None:
        subnet = [0] * len(model)
    flops = []
    m_list = []
    skip = 0
    for ms, idx in zip(model, subnet):
        for m in ms[idx].modules():
            if isinstance(m, SqueezeExcite):
                skip = 2
                raise RuntimeError(f'flops.py FLOPs_aware skip must be 0')
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if skip == 0:
                    m_list.append(m)
                else:
                    flops.append(m.in_channels * m.out_channels)
                    skip -= 1
            elif isinstance(m, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                m_list.append(m)

    c, w, h = input_shape
    m = m_list.pop(0)
    w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
    h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
    FLOPs_bf = m.in_channels * 1 * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1]

    for m in m_list:
        if isinstance(m, nn.Conv2d):
            c = m.out_channels
            if m.kernel_size[0] != 1:
                w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
                h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
            FLOPs_af = 1 * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1]
            flops.append(FLOPs_af + FLOPs_bf)
            FLOPs_bf = m.in_channels * 1 * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1]

        elif isinstance(m, nn.Linear):
            FLOPs_af = 1 * m.out_features
            flops.append(FLOPs_bf + FLOPs_af)
            FLOPs_bf = m.in_features * 1
        elif isinstance(m, nn.MaxPool2d):
            w = (w + m.padding * 2 - m.kernel_size + 1) // m.stride
            h = (h + m.padding * 2 - m.kernel_size + 1) // m.stride
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            w = m.output_size
            h = m.output_size
    # the output of last layer can't be changed ,1000 for imagenet and 10 for cifar10
    #flops.append(FLOPs_bf)
    return flops


def Model_channels(model, subnet=None):
    if subnet is None:
        subnet = [0] * len(model)
    channel_list = []
    m_list = []
    skip = 0
    for ms, idx in zip(model, subnet):
        for m in ms[idx].modules():
            if isinstance(m, SqueezeExcite):
                skip = 2
                raise RuntimeError(f'flops.py Model_channels skip must be 0')
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if skip == 0:
                    m_list.append(m)
                else:
                    channel_list.append(m.in_channels * m.out_channels)
                    skip -= 1

    for m in m_list:
        if isinstance(m, nn.Conv2d):
            channel_list.append(m.out_channels)
        elif isinstance(m, nn.Linear):
            channel_list.append(m.out_features)
    # the last one is 1000 for imagenet or 10 for cifar10
    del channel_list[-1]
    return channel_list

def count_sample_flops(model, Channel_dropout, subnet=None, input_shape=[3, 224, 224]):
    if subnet is None:
        subnet = [0] * len(model)
    flops = 0
    temp = []
    Channel_dropout_copy = copy.deepcopy(Channel_dropout)

    c, w, h = input_shape
    for block in model:
        total_op = len(block)
        assert total_op == 1
        if not isinstance(block[0], (nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d)) and not isinstance(block[0], FC):
            Flops, w, h, Channel_dropout_copy, c = block[0].Count_flops_channel_dropout(w, h, Channel_dropout_copy, inchannel = c)
            flops += Flops
            temp.append(Flops)
        elif isinstance(block[0], (nn.MaxPool2d)):
            w = (w + block[0].padding * 2 - block[0].kernel_size + 1) // block[0].stride
            h = (h + block[0].padding * 2 - block[0].kernel_size + 1) // block[0].stride
        elif isinstance(block[0], nn.AdaptiveMaxPool2d):
            w = block[0].output_size
            h = block[0].output_size
    return flops







trim = yaml.load(open('mb_imagenet_timedict_v1/mobile_trim.yaml', 'r'))


def count_latency(model, subnet=None, input_shape=(3, 224, 224), dump_path=''):
    if subnet is None:
        subnet = [0] * len(model)
    flops = []
    m_list = []
    c = 0
    for ms, idx in zip(model, subnet):
        c += 1
        for m in ms[idx].modules():
            if isinstance(m, (InvertedResidual, FC, Conv2d)):
                m_list.append(m)
    
    latency = []    
    c, w, h = input_shape
    for m in m_list:
        if isinstance(m, Conv2d):
            if m.k == 1:
                latency.append(trim.get(f'Conv_1-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}', {'mean': 0})['mean'])
            else:
                latency.append(trim.get(f'Conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}', {'mean': 0})['mean'])
            c = m.oup
            w = w // m.stride
            h = h // m.stride
        elif isinstance(m, InvertedResidual):
            latency.append(trim.get(f'expanded_conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}-expand:{m.t}-kernel:{m.k}-stride:{m.stride}-idskip:{1 if m.use_shortcut else 0}', {'mean': 0})['mean'])
            if latency[-1] == 0:
                print(f'expanded_conv-input:{w}x{h}x{c}-output:{w//m.stride}x{h//m.stride}x{m.oup}-expand:{m.t}-kernel:{m.k}-stride:{m.stride}-idskip:{1 if m.use_shortcut else 0}')
            c = m.oup
            w = w // m.stride
            h = h // m.stride
        elif isinstance(m, FC):
            latency.append(trim.get(f'Logits-input:{w}x{h}x{c}-output:{m.oup}', {'mean': 0})['mean'])
    #print(latency)        
    return sum(latency)



if __name__ == '__main__':
    def conv(inp, oup, k, s, p, g=1):
        c = nn.Sequential(nn.Conv2d(inp, oup, k, stride=s, padding=p, groups=g),
                          nn.BatchNorm2d(oup),
                          nn.ReLU(inplace=True))
        return c

    s1 = nn.ModuleList([conv(3, 16, 3, 2, 1, 1), conv(3, 100, 1, 2, 0, 1)])
    s2 = nn.ModuleList([conv(16, 16, 3, 2, 1, g=16), conv(100, 100, 1, 2, 0, 1)])
    final = nn.ModuleList([nn.Linear(1000, 1000)])
    model = nn.ModuleList([s1, s2, final])
    count_flops(model, [0, 0, 0])
    count_latency(model, [0, 0, 0])
    
    



