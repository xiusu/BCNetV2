import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, inp, oup, stride, **kwargs):
        super(Identity, self).__init__(**kwargs)
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, use_shortcut=False, **kwargs):
        super(InvertedResidual, self).__init__(**kwargs)
        self.stride = stride
        self.expand_ratio = expand_ratio 
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw            
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )   
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )   
        self.use_shortcut = use_shortcut
 
    def forward(self, x): 
        if self.use_shortcut:
            return self.conv(x) + x 
        return self.conv(x)


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SlimmableConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.is_depthwise = (groups == in_channels)
        self.width = out_channels

    def forward(self, input):
        in_channels = input.size(1)
        out_channels = self.width
        self.groups = in_channels if self.is_depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        output = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return output


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SlimmableLinear, self).__init__(
            in_features, out_features, bias=bias)

    def forward(self, input):
        in_features = input.size(1)
        weight = self.weight[:, :in_features]
        return nn.functional.linear(input, weight, self.bias)


class SlimmableBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(SlimmableBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)

    def forward(self, input):
        num_features = input.size(1)
        if num_features == self.num_features:
            output = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps)
        else:
            if self.training:
                output = nn.functional.batch_norm(
                    input,
                    None,
                    None,
                    self.weight[:num_features],
                    self.bias[:num_features],
                    self.training,
                    self.momentum,
                    self.eps)
            else:
                output = nn.functional.batch_norm(
                    input,
                    self.running_mean[:num_features],
                    self.running_var[:num_features],
                    self.weight[:num_features],
                    self.bias[:num_features],
                    self.training,
                    self.momentum,
                    self.eps)
        return output


class SlimmableInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, use_shortcut=False, **kwargs):
        super(SlimmableInvertedResidual, self).__init__(**kwargs)
        self.stride = stride
        self.expand_ratio = expand_ratio 
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw            
                SlimmableConv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
                SlimmableBatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                SlimmableConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SlimmableBatchNorm2d(oup)
            )   
        else:
            self.conv = nn.Sequential(
                # pw
                SlimmableConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                SlimmableBatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                SlimmableConv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
                SlimmableBatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                SlimmableConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                SlimmableBatchNorm2d(oup),
            )   
        self.use_shortcut = use_shortcut
 
    def set_width(self, inner_width, out_width):
        if self.expand_ratio == 1:
            self.conv[0].width = inner_width
            self.conv[3].width = out_width
        else:
            self.conv[0].width = inner_width
            self.conv[3].width = inner_width
            self.conv[6].width = out_width

    def forward(self, x): 
        if self.use_shortcut:
            return self.conv(x) + x 
        return self.conv(x)
