import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import random
import cellular.pape.distributed as dist


OPS = OrderedDict()
# CAUTION: The assign order is Strict

'''
ops for single path one shot net, based on shufflenetv2
'''

OPS['Choice_3'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 3, 1], max_ks=3, channel_search=c_search)
OPS['Choice_5'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 5, 1], max_ks=5, channel_search=c_search)
OPS['Choice_7'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[1, 7, 1], max_ks=7, channel_search=c_search)
OPS['Choice_x'] = lambda inp, oup, t, stride, c_search: \
    SNetInvertedResidual(inp, oup, stride, conv_list=[3, 1, 3, 1, 3, 1], max_ks=3, channel_search=c_search)
'''
end ops for single path one shot net, based on shufflenetv2
'''


OPS['ir_3x3_nse'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search,
                                                                           activation=HSwish, use_se=False)
OPS['ir_5x5_nse'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search,
                                                                           activation=HSwish, use_se=False)
OPS['ir_7x7_nse'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search,
                                                                           activation=HSwish, use_se=False)
OPS['ir_3x3_se'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search,
                                                                           activation=HSwish, use_se=True)
OPS['ir_5x5_se'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search,
                                                                           activation=HSwish, use_se=True)
OPS['ir_7x7_se'] = lambda inp, oup, t, stride, c_search: InvertedResidual(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search,
                                                                           activation=HSwish, use_se=True)


OPS['ir_3x3'] = lambda inp, oup, t, stride, c_search, channels: InvertedResidual(inp=inp, channels=channels, t=t, stride=stride, k=3, channel_search=c_search)
OPS['ir_5x5'] = lambda inp, oup, t, stride, c_search, channels: InvertedResidual(inp=inp, channels=channels, t=t, stride=stride, k=5, channel_search=c_search)
OPS['ir_7x7'] = lambda inp, oup, t, stride, c_search, channels: InvertedResidual(inp=inp, channels=channels, t=t, stride=stride, k=7, channel_search=c_search)

OPS['nr_3x3'] = lambda inp, oup, t, stride, c_search, channels: NormalResidual(inp=inp, channels=channels, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['nr_5x5'] = lambda inp, oup, t, stride, c_search, channels: NormalResidual(inp=inp, channels=channels, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['nr_7x7'] = lambda inp, oup, t, stride, c_search, channels: NormalResidual(inp=inp, channels=channels, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)


OPS['nb_3x3'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['nb_5x5'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['nb_7x7'] = lambda inp, oup, t, stride, c_search: DualBlock(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['rec_3x3'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['rec_5x5'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['rec_7x7'] = lambda inp, oup, t, stride, c_search: RecBlock(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['ds_3x3'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=3, channel_search=c_search)
OPS['ds_5x5'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=5, channel_search=c_search)
OPS['ds_7x7'] = lambda inp, oup, t, stride, c_search: DepthwiseSeparableConv(inp=inp, oup=oup, t=t, stride=stride, k=7, channel_search=c_search)

OPS['lb_3x3'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=3, channel_search=c_search)
OPS['lb_5x5'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=5, channel_search=c_search)
OPS['lb_7x7'] = lambda inp, oup, t, stride, c_search: LinearBottleneck(inp=inp, oup=oup, stride=stride, k=7, channel_search=c_search)


OPS['id'] = lambda inp, oup, t, stride, c_search: Identity(inp=inp, oup=oup, t=t, stride=stride, k=1, channel_search=c_search)
OPS['conv2d'] = lambda inp, oup, t, stride, c_search: Conv2d(inp=inp, oup=oup, stride=stride, k=1, channel_search=c_search)
OPS['conv3x3'] = lambda inp, oup, t, stride, c_search: Conv2d(inp=inp, oup=oup, stride=stride, k=3, channel_search=c_search)
OPS['conv7x7'] = lambda inp, oup, t, stride, c_search: Conv2d(inp=inp, oup=oup, stride=stride, k=7, channel_search=False)
OPS['maxpool2x2'] = lambda inp, oup, t, stride, c_search: nn.MaxPool2d(kernel_size=2, stride=stride, padding=1)
OPS['maxpool3x3'] = lambda inp, oup, t, stride, c_search: nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
OPS['Adaptmaxpool'] = lambda inp, oup, t, stride, c_search: nn.AdaptiveMaxPool2d(1)
'''
ops for single path one shot net, based on shufflenetv2
'''
channel_mults = [1.0, 0.8, 0.6, 0.4, 0.2]


# custom conv2d for channel search
class DynamicConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', False)
        super().__init__(*args, **kwargs)
        self.out_c = self.out_channels

    def forward(self, input):
        out_c = self.out_c
        if self.channel_search and self.groups == 1 and out_c != self.out_channels:
            in_c = input.shape[1]
            #out_c = int(self.out_channels / max(channel_mults) * random.choice(channel_mults))
            w = self.weight[:out_c, :in_c].contiguous()
            b = self.bias[:out_c].contiguous() if self.bias is not None else None
            return F.conv2d(input, w, b, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.in_channels != input.shape[1]:
            # channel num mismatch, upper layer must be using channel search
            in_c = input.shape[1]
            if self.groups != 1:
                w = self.weight[:in_c, :in_c].contiguous()
            else:
                w = self.weight[:, :in_c].contiguous()
            return F.conv2d(input, w, self.bias, self.stride,
                            self.padding, self.dilation, in_c if self.groups!=1 else 1)
        else:
            return super().forward(input)


class DynamicLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', False)
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.channel_search:
            in_dim = input.shape[-1]
            w = self.weight[:, :in_dim].contiguous()
            return F.linear(input, w, self.bias)
        else:
            return super().forward(input)


class DynamicBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        self.channel_search = kwargs.pop('channel_search', True)
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.channel_search:
            in_c = input.shape[1]

            self._check_input_dim(input)
            # exponential_average_factor is self.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum
            w = self.weight[:in_c].contiguous()
            b = self.bias[:in_c].contiguous() if self.bias is not None else None
            r_mean = self.running_mean[:in_c].contiguous()
            r_var = self.running_var[:in_c].contiguous()
            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return super().forward(input)

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class SNetInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, conv_list, max_ks, channel_search):
        super(SNetInvertedResidual, self).__init__()
        self.stride = stride
        self.channel_search = channel_search
        self.c_mult_idx = None
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.stride == 1:
            branch2 = []
            for idx, conv_ks in enumerate(conv_list):  # last pw conv can not use channel search
                branch2.append(DynamicConv2d(oup_inc, oup_inc, conv_ks, padding=(conv_ks - 1) // 2,
                                         groups=oup_inc if conv_ks != 1 else 1, bias=False, channel_search=channel_search if idx != len(conv_list) - 1 else False))
                if idx == len(conv_list) - 1:
                    setattr(branch2[-1], 'last_pw', True)
                branch2.append(DynamicBatchNorm2d(oup_inc, channel_search=channel_search if idx != len(conv_list) - 1 else False))
                if conv_ks == 1:
                    branch2.append(nn.ReLU(inplace=True))
            self.branch2 = nn.Sequential(*branch2)
        else:
            self.branch1 = nn.Sequential(
                # dw
                DynamicConv2d(inp, inp, max_ks, stride=2, padding=(max_ks - 1) // 2,
                              groups=inp, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(inp, channel_search=channel_search),
                # pw-linear
                DynamicConv2d(inp, oup_inc, 1, 1, 0, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(oup_inc, channel_search=channel_search),
                nn.ReLU(inplace=True),
            )

            branch2 = []
            first_pw = True
            first_down = True
            channel_num = inp
            for conv_ks in conv_list:
                if first_pw and conv_ks == 1:
                    branch2.append(DynamicConv2d(channel_num, oup_inc, conv_ks, stride=2 if conv_ks != 1 else 1,
                                             padding=(conv_ks - 1) // 2,
                                             groups=inp if conv_ks != 1 else 1, bias=False))
                    channel_num = oup_inc
                    first_pw = False
                elif first_down and conv_ks != 1:
                    branch2.append(DynamicConv2d(channel_num, channel_num, conv_ks,
                                             stride=2,
                                             padding=(conv_ks - 1) // 2,
                                             groups=channel_num if conv_ks != 1 else 1, bias=False))
                    first_down = False
                else:
                    branch2.append(DynamicConv2d(channel_num, channel_num, conv_ks,
                                             stride=1,
                                             padding=(conv_ks - 1) // 2,
                                             groups=channel_num if conv_ks != 1 else 1, bias=False))
                branch2.append(DynamicBatchNorm2d(channel_num))
                if conv_ks == 1:
                    branch2.append(nn.ReLU(inplace=True))
            self.branch2 = nn.Sequential(*branch2)

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.channel_search:
            assert self.c_mult_idx is not None
            for module in self.modules():
                if isinstance(module, DynamicConv2d):
                    module.out_c = int(module.out_channels * channel_mults[self.c_mult_idx])
                    if module.out_c % 2 == 1:
                        module.out_c += 1
                if getattr(module, 'last_pw', False):
                    module.out_c = x.shape[1] // 2
        if 1 == self.stride:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            b2 = self.branch2(x2)
            out = self._concat(x1, b2)
        elif 2 == self.stride:
            b1, b2 = self.branch1(x), self.branch2(x)
            out = self._concat(b1, b2)

        return channel_shuffle(out, 2)


'''
end ops for single path one shot net, based on shufflenetv2
'''


class FC(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn, dp=0., act='nn.ReLU'):
        super(FC, self).__init__()
        self.oup = dim_out
        self.module = []
        self.module.append(DynamicLinear(dim_in, dim_out))
        if use_bn:
            self.module.append(nn.BatchNorm1d(dim_out))
        if act is not None:
            self.module.append(eval(act)(inplace=True))
        if dp != 0:
            self.module.append(nn.Dropout(dp))
        self.module = nn.Sequential(*self.module)

    def forward(self, x, **kwargs):
        if x.dim() != 2:
            x = x.flatten(1)
        return self.module(x)


class BasicOp(nn.Module):

    def __init__(self, oup, **kwargs):
        super(BasicOp, self).__init__()
        self.oup = oup
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def get_output_channles(self):
        return self.oup


class Conv2d(BasicOp):
    def __init__(self, inp, oup, stride, k, activation=nn.ReLU, **kwargs):
        super(Conv2d, self).__init__(oup, **kwargs)
        self.stride = stride
        self.k = k
        channel_search = kwargs.pop('channel_search', False)
        self.conv = nn.Sequential(
            DynamicConv2d(inp, oup, kernel_size=k, stride=stride, padding=k//2, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(oup, channel_search=channel_search),
            activation()
        )

    def Count_flops_channel_dropout(self, w, h, Channel_dropout, inchannel):
        flops = []
        inp = inchannel
        oup = sum(Channel_dropout.pop(0))
        m = self.conv[0]
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])
            #flops.append(m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])
        return sum(flops), w, h, Channel_dropout, oup


    def forward(self, x, **kwargs):
        x = self.conv(x)

        if 'Channel_dropout' in kwargs:
            size = x.size()
            Channel_dropout_tensor = kwargs['Channel_dropout'].pop(0)
            Channel_dropout_tensor = torch.FloatTensor(Channel_dropout_tensor).cuda()
            Channel_dropout_tensor = Channel_dropout_tensor.reshape(1, size[1], 1, 1)
            #Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
            x = x * Channel_dropout_tensor

        return x


class LinearBottleneck(BasicOp):
    def __init__(self, inp, oup, stride, k, activation=nn.ReLU, **kwargs):
        super(LinearBottleneck, self).__init__(oup, **kwargs)
        channel_search = kwargs.pop('channel_search', False)

        neck_dim = oup // 4
        self.conv1 = DynamicConv2d(inp, neck_dim, kernel_size=1, stride=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(neck_dim, channel_search=channel_search)
        self.act1 = activation()
        self.conv2 = DynamicConv2d(neck_dim, neck_dim, kernel_size=k, stride=stride, padding=k//2, bias=False, channel_search=channel_search)
        self.bn2 = DynamicBatchNorm2d(neck_dim, channel_search=channel_search)
        self.act2 = activation()
        self.conv3 = DynamicConv2d(neck_dim, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)

        return out


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SqueezeExcite, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act


class InvertedResidual(BasicOp):
    def __init__(self, inp, stride, t, k=3, activation=nn.ReLU, use_se=False, channels=None, **kwargs):
        # this place super why have channels and **kwargs without others
        super(InvertedResidual, self).__init__(channels[-1], **kwargs)
        print('ir inp: {}, channels: {}'.format(inp, channels))
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        channel_search = kwargs.pop('channel_search', False)
        oup = channels[-1]
        if t == 1:
            assert len(channels) == 2
            assert inp == channels[0]
            self.conv1 = nn.Sequential(
                # dw            
                DynamicConv2d(inp, inp, k, stride, padding=k//2, groups=inp, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(inp, channel_search=channel_search),
                activation(inplace=True))
            self.conv2 = nn.Sequential(
                # pw-linear
                DynamicConv2d(inp, oup, 1, 1, 0, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(oup, channel_search=channel_search)
            )
        else:
            assert len(channels) == 3
            hidden_dim = channels[0]
            assert channels[0] == channels[1]
            self.conv1 = nn.Sequential(
                # pw
                DynamicConv2d(inp, hidden_dim, 1, 1, 0, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(hidden_dim, channel_search=channel_search),
                activation(inplace=True))
            self.conv2 = nn.Sequential(
                # dw
                DynamicConv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(hidden_dim, channel_search=channel_search),
                activation(inplace=True))
            self.conv3 = nn.Sequential(
                # pw-linear
                DynamicConv2d(hidden_dim, oup, 1, 1, 0, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(oup, channel_search=channel_search),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x, **kwargs):

        if 'Channel_dropout' in kwargs:
            if self.t == 1:
                Channel_dropout1 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout1 = torch.FloatTensor(Channel_dropout1).cuda()
                Channel_dropout2 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout2 = torch.FloatTensor(Channel_dropout2).cuda()
                #if self.use_shortcut:

                y = self.conv1(x)
                size = y.size()
                Channel_dropout = Channel_dropout1.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                y = y * Channel_dropout

                y = self.conv2(y)
                size = y.size()
                Channel_dropout = Channel_dropout2.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                y = y * Channel_dropout
                if self.use_shortcut:
                    y = y + x
                y = y * Channel_dropout

            else:
                Channel_dropout1 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout1 = torch.FloatTensor(Channel_dropout1).cuda()

                Channel_dropout2 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout2 = torch.FloatTensor(Channel_dropout2).cuda()

                Channel_dropout3 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout3 = torch.FloatTensor(Channel_dropout3).cuda()

                y = self.conv1(x)
                size = y.size()
                Channel_dropout = Channel_dropout1.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                y = y * Channel_dropout

                y = self.conv2(y)
                size = y.size()
                Channel_dropout = Channel_dropout2.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                y = y * Channel_dropout

                y = self.conv3(y)
                size = y.size()
                Channel_dropout = Channel_dropout3.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                if self.use_shortcut:
                    y = y + x
                y = y * Channel_dropout
        else:
            y = self.conv1(x)
            y = self.conv2(y)
            if self.use_shortcut:
                if self.t == 1:
                    y = y + x
                else:
                    y = self.conv3(y) + x
            else:
                if not self.t == 1:
                    y = self.conv3(y)
        return y

    def Count_flops_channel_dropout(self, w, h, Channel_dropout, inchannel):

        flops = []
        inp = inchannel
        oup = sum(Channel_dropout.pop(0))
        m = self.conv1[0]
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])
            #flops.append(m.in_channels * m.out_channels * w * h // m.groups * m.kernel_size[0] * m.kernel_size[1])

        inp = oup
        oup = sum(Channel_dropout.pop(0))
        m = self.conv2[0]
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])

        if self.t != 1:
            inp = oup
            oup = sum(Channel_dropout.pop(0))
            if self.t != 1:
                m = self.conv3[0]
                w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
                h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
                if m.groups == m.in_channels:
                    flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
                else:
                    flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])

        return sum(flops), w, h, Channel_dropout, oup
        #        if self.use_shortcut:
        #            x = self.conv(x) + x
        #            return x
        #        else:
        #            x = self.conv(x)

        #    return self.conv(x) + x
        #return self.conv(x)



class NormalResidual(BasicOp):
    def __init__(self, inp, stride, t, k=3, activation=nn.ReLU, channels=None, **kwargs):
        super(NormalResidual, self).__init__(**kwargs)
        self.stride = stride
        assert stride in [1, 2]
        assert len(channels) == 3
        oup = channels[-1]
        self.residual_connection = inp == oup and stride == 1
        '''
        self.conv = nn.Sequential(
            # pw
            DynamicConv2d(inp, channels[0], 1, 1, 0, bias=False),
            DynamicBatchNorm2d(channels[0]),
            activation(inplace=True),
            # dw
            DynamicConv2d(channels[0], channels[1], k, stride, padding=k//2, groups=1, bias=False),
            DynamicBatchNorm2d(channels[1]),
            activation(inplace=True),
            # pw-linear
            DynamicConv2d(channels[1], oup, 1, 1, 0, bias=False),
            DynamicBatchNorm2d(oup),
        )
        '''
        #pw
        self.conv1 = nn.Sequential(
            DynamicConv2d(inp, channels[0], 1, 1, 0, bias=False),
            DynamicBatchNorm2d(channels[0]),
            activation(inplace=True))
        # dw
        self.conv2 = nn.Sequential(
            DynamicConv2d(channels[0], channels[1], k, stride, padding=k // 2, groups=1, bias=False),
            DynamicBatchNorm2d(channels[1]),
            activation(inplace=True))
        # pw-linear
        self.conv3 = nn.Sequential(
            DynamicConv2d(channels[1], oup, 1, 1, 0, bias=False),
            DynamicBatchNorm2d(oup))


        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                DynamicConv2d(inp, oup, 1, stride=stride, bias=False),
                DynamicBatchNorm2d(oup),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        if 'Channel_dropout' in kwargs:
            Channel_dropout1 = kwargs['Channel_dropout'].pop(0)
            Channel_dropout1 = torch.FloatTensor(Channel_dropout1).cuda()
            Channel_dropout2 = kwargs['Channel_dropout'].pop(0)
            Channel_dropout2 = torch.FloatTensor(Channel_dropout2).cuda()
            Channel_dropout3 = kwargs['Channel_dropout'].pop(0)
            Channel_dropout3 = torch.FloatTensor(Channel_dropout3).cuda()

            base = self.conv1(x)
            size = base.size()
            Channel_dropout = Channel_dropout1.reshape(1, size[1], 1, 1)
            Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
            base = base * Channel_dropout

            base = self.conv2(base)
            size = base.size()
            Channel_dropout = Channel_dropout2.reshape(1, size[1], 1, 1)
            Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
            base = base * Channel_dropout

            base = self.conv3(base)
            size = base.size()
            Channel_dropout = Channel_dropout3.reshape(1, size[1], 1, 1)
            Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
            base = base * Channel_dropout

            if self.residual_connection:
                base += x

            else:
                Channel_dropout4 = kwargs['Channel_dropout'].pop(0)
                Channel_dropout4 = torch.FloatTensor(Channel_dropout4).cuda()
                short_cut = self.shortcut(x)
                size = short_cut.size()
                Channel_dropout = Channel_dropout4.reshape(1, size[1], 1, 1)
                Channel_dropout = Channel_dropout.repeat(size[0], 1, size[2], size[3])
                short_cut = short_cut * Channel_dropout
                base += short_cut
            return self.post_relu(base)
        else:
            base = self.conv1(x)
            base = self.conv2(base)
            base = self.conv3(base)
            if self.residual_connection:
                base += x
            else:
                base += self.shortcut(x)
            return self.post_relu(base)
    def Count_flops_channel_dropout(self, w, h, Channel_dropout, inchannel):

        flops = []
        inp = inchannel
        oup = sum(Channel_dropout.pop(0))
        m = self.conv1[0]
        #assert oup == m.out_channels, "oup is {}, out_channels is {}".format(oup, m.out_channels)
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])


        inp = oup
        oup = sum(Channel_dropout.pop(0))
        m = self.conv2[0]
        #assert oup == m.out_channels, "oup is {}, out_channels is {}".format(oup, m.out_channels)
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])


        inp = oup
        oup = sum(Channel_dropout.pop(0))
        m = self.conv3[0]
        w = (w + m.padding[0] * 2 - m.kernel_size[0] + 1) // m.stride[0]
        h = (h + m.padding[1] * 2 - m.kernel_size[1] + 1) // m.stride[1]
        if m.groups == m.in_channels:
            flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
        else:
            flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])

        if not self.residual_connection:
            inp = inchannel
            oup = sum(Channel_dropout.pop(0))
            m = self.shortcut[0]
            if m.groups == m.in_channels:
                flops.append(inp * w * h * m.kernel_size[0] * m.kernel_size[1])
            else:
                flops.append(inp * oup * w * h * m.kernel_size[0] * m.kernel_size[1])
        return sum(flops), w, h, Channel_dropout, oup



class DepthwiseSeparableConv(BasicOp):
    def __init__(self, inp, oup, stride, k=3, activation=nn.ReLU, **kwargs):
        super(DepthwiseSeparableConv, self).__init__(oup, **kwargs)
        self.stride = stride
        assert stride in [1, 2]
        channel_search = kwargs.pop('channel_search', False)
        self.conv_dw = nn.Sequential(
            DynamicConv2d(inp, inp, k, stride, groups=inp, bias=False, padding=k//2, channel_search=channel_search),
            DynamicBatchNorm2d(inp, eps=1e-10, momentum=0.05, channel_search=channel_search),
            activation()
        )

        self.conv_pw = nn.Sequential(
            DynamicConv2d(inp, oup, 1, 1, bias=False, channel_search=channel_search),
            DynamicBatchNorm2d(oup, eps=1e-10, momentum=0.05, channel_search=channel_search),
        )

    def forward(self, x, drop_connect_rate=None):
        x = self.conv_dw(x)
        #if self.has_se:
        #    x = self.se(x)
        x = self.conv_pw(x)
        return x


class DualBlock(BasicOp):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, **kwargs):
        super(DualBlock, self).__init__(oup, **kwargs)
        padding = k // 2
        channel_search = kwargs.pop('channel_search', False)
        self.conv1 = DynamicConv2d(inp, inp * t, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation1 = activation()
        self.conv2_1 = DynamicConv2d(inp * t, inp * t, kernel_size=k, stride=1, padding=padding, bias=False, channel_search=channel_search)
        self.bn2_1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.conv2_2 = DynamicConv2d(inp * t, inp * t, kernel_size=k, stride=stride, padding=padding, bias=False, channel_search=channel_search)
        self.bn2_2 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation2 = activation()
        self.conv3 = DynamicConv2d(inp * t, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)
        self.activation3 = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation2(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation3(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class RecBlock(BasicOp):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, **kwargs):
        super(RecBlock, self).__init__(oup, **kwargs)
        padding = k // 2
        self.time = 0
        channel_search = kwargs.pop('channel_search', False)
        self.conv1 = DynamicConv2d(inp, inp * t, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation1 = activation()
        self.conv2_1 = DynamicConv2d(inp * t, inp * t, kernel_size=(1, k), stride=(1, stride),
                                 padding=(0, padding), bias=False, channel_search=channel_search)
        self.bn2_1 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation2 = activation()
        self.conv2_2 = DynamicConv2d(inp * t, inp * t, kernel_size=(k, 1), stride=(stride, 1),
                                 padding=(padding, 0), bias=False, channel_search=channel_search)
        self.bn2_2 = DynamicBatchNorm2d(inp * t, channel_search=channel_search)
        self.activation3 = activation()
        self.conv3 = DynamicConv2d(inp * t, oup, kernel_size=1, bias=False, channel_search=channel_search)
        self.bn3 = DynamicBatchNorm2d(oup, channel_search=channel_search)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation2(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation3(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class Identity(BasicOp):
    def __init__(self, inp, oup, stride, **kwargs):
        super(Identity, self).__init__(oup, **kwargs)
        channel_search = kwargs.pop('channel_search', False)
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                DynamicConv2d(inp, oup, kernel_size=1, stride=stride, bias=False, channel_search=channel_search),
                DynamicBatchNorm2d(oup, channel_search=channel_search),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x


if __name__ == '__main__':
    model = DynamicConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2, channel_search=True)
    print(model(torch.zeros(3, 128, 112, 112)).shape)

    linear = DynamicLinear(in_features=512, out_features=1000, channel_search=True)
    print(linear(torch.zeros(3, 256)).shape)

    bn = DynamicBatchNorm2d(num_features=512, channel_search=False)
    print(bn(torch.zeros(3, 512, 112, 112)).shape)

