import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLinear_multiLn(nn.Module)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.in_indices = None
        self.out_indices = None

        if isinstance(self.weight_num, int):
            Ln_list = []
            for i in range(self.weight_num):
                Ln_list.append(DynamicLinear(self.in_features, self.out_features, bias=self.bias))

            self.Ln_list = nn.Sequential(*Ln_list)
        else:
            raise RuntimeError(f'not multi Ln, self: {self}')

    def set_indices(self, in_indices, out_indices):
        self.in_indices = in_indices
        self.out_indices = out_indices

    def set_weights(self):
        pass
    
    def set_bias(self):
        pass

    def forward(self, input):
        if self.in_indices is not None:
            assert self.out_indices is None, 'current version does not support searching linear modules'
            w = self.set_weights()
            #w = self.weight[:, self.in_indices[0]:self.in_indices[1]+1].contiguous()
            b = self.set_bias() if self.bias is not None else None
            return F.linear(input, w, b)
        else:
            assert len(self.Ln_list) == 1, f'len(self.Ln_list): {len(self.Ln_list)}, self: {len(self.Ln_list)}'
            return self.Ln_list(input)
            #return super().forward(input)
    
    def build_nn_module(self):
        in_features = self.in_features
        out_features = self.out_features
        if self.in_indices is not None:
            in_features = self.in_indices[1] - self.in_indices[0] + 1
        else:
            in_features = self.in_features
        if self.out_indices is not None:
            out_features = self.out_indices[1] - self.out_indices[0] + 1
        else:
            out_features = self.out_features
        return torch.nn.Linear(in_features, out_features, self.bias is not None)


class DynamicLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_indices = None
        self.out_indices = None

    def set_indices(self, in_indices, out_indices):
        self.in_indices = in_indices
        self.out_indices = out_indices

    def forward(self, input):
        if self.in_indices is not None:
            assert self.out_indices is None, 'current version does not support searching linear modules'
            w = self.weight[:, self.in_indices[0]:self.in_indices[1]+1].contiguous()
            return F.linear(input, w, self.bias)
        else:
            return super().forward(input)
    
    def build_nn_module(self):
        in_features = self.in_features
        out_features = self.out_features
        if self.in_indices is not None:
            in_features = self.in_indices[1] - self.in_indices[0] + 1
        else:
            in_features = self.in_features
        if self.out_indices is not None:
            out_features = self.out_indices[1] - self.out_indices[0] + 1
        else:
            out_features = self.out_features
        return torch.nn.Linear(in_features, out_features, self.bias is not None)


class DynamicBatchNorm2d_multiBN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicBatchNorm2d_multiBN, self).__init__():

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.indices = None
        #self.weight_num = kwargs.pop('weight_num', False)

        if isinstance(self.weight_num, int):
            bn_list = []
            for i in range(self.weight_num):
                bn_list.append(DynamicBatchNorm2d(self.num_features))

            self.bn_list_ = nn.Sequential(*bn_list)
        else:
            raise RuntimeError(f'not multi bn, self: {self}')

    def set_indices(self, indices):
        self.indices = indices

    def set_weights(self):
        pass

    def set_bias(self):
        pass

    def set_running_mean(self):
        pass

    def set_running_var(self):
        pass

    def forward(self, input):
        if self.indices is None or self.indices[0] is None:
            assert len(self.bn_list_) == 1, f'self.indices: {self.indices}, len(self.bn_list_): {len(self.bn_list_)}'
            return self.bn_list_(input)
        elif self.indices is not None and not isinstance(self.indices[0], (tuple, list)):
            self.bn_list_[0]._check_input_dim(input)
            if self.bn_list_[0].momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.bn_list_[0].momentum

            if self.bn_list_[0].training and self.bn_list_[0].track_running_stats:
                if self.bn_list_[0].num_batches_tracked is not None:
                    self.bn_list_[0].num_batches_tracked += 1
                    if self.bn_list_[0].momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.bn_list_[0].num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.bn_list_[0].momentum

            # w = self.weight[self.indices[0]:self.indices[1]+1].contiguous()
            #b = self.bias[self.indices[0]:self.indices[1]+1].contiguous() if self.bias is not None else None
            #r_mean = self.running_mean[self.indices[0]:self.indices[1]+1].contiguous()
            #r_var = self.running_var[self.indices[0]:self.indices[1]+1].contiguous()
            w = self.set_weights()
            b = self.set_bias()
            r_mean = self.set_running_mean()
            r_var = self.running_var()

            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.bn_list_[0].training or not self.bn_list_[0].track_running_stats,
                exponential_average_factor, self.bn_list_[0].eps)

        else:    # several non-continuous groups of channels
            raise RuntimeError('stop, not support (tuple, list) for index[0] right now')
            self.bn_list_[0]._check_input_dim(input)
            # exponential_average_factor is self.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.bn_list_[0].momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.bn_list_[0].momentum

            if self.bn_list_[0].training and self.bn_list_[0].track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.bn_list_[0].num_batches_tracked is not None:
                    self.bn_list_[0].num_batches_tracked += 1
                    if self.bn_list_[0].momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.bn_list_[0].num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.bn_list_[0].momentum
            ws = []
            bs = []
            r_means = []
            r_vars = []
            for indices in self.indices:
                ws.append(self.weight[indices[0]:indices[1]+1])
                bs.append(self.bias[indices[0]:indices[1]+1])
                r_means.append(self.running_mean[indices[0]:indices[1]+1])
                r_vars.append(self.running_var[indices[0]:indices[1]+1])
            w = torch.cat(ws, dim=0).contiguous()
            b = torch.cat(bs, dim=0).contiguous()
            r_mean = torch.cat(r_means, dim=0).contiguous()
            r_var = torch.cat(r_vars, dim=0).contiguous()

            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

    def build_nn_module(self):
        if self.indices is not None:
            if not isinstance(self.indices[0], int):
                num_features = 0
                for indices in self.indices:
                    if indices is None:
                        continue
                    num_features += indices[1] - indices[0] + 1
                if num_features == 0:
                    num_features = self.num_features
            else:
                num_features = self.indices[1] - self.indices[0] + 1
        else:
            num_features = self.num_features
        
        return nn.BatchNorm2d(num_features, self.eps, self.momentum, self.weight is not None, self.track_running_stats)

            
class DynamicBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = None

    def set_indices(self, indices):
        self.indices = indices

    def forward(self, input):
        if self.indices is None or self.indices[0] is None:
            return super().forward(input)
        elif self.indices is not None and not isinstance(self.indices[0], (tuple, list)):
            self._check_input_dim(input)
            # exponential_average_factor is self.momentum set to
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
            w = self.weight[self.indices[0]:self.indices[1]+1].contiguous()
            b = self.bias[self.indices[0]:self.indices[1]+1].contiguous() if self.bias is not None else None
            r_mean = self.running_mean[self.indices[0]:self.indices[1]+1].contiguous()
            r_var = self.running_var[self.indices[0]:self.indices[1]+1].contiguous()
            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:    # several non-continuous groups of channels
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
            ws = []
            bs = []
            r_means = []
            r_vars = []
            for indices in self.indices:
                ws.append(self.weight[indices[0]:indices[1]+1])
                bs.append(self.bias[indices[0]:indices[1]+1])
                r_means.append(self.running_mean[indices[0]:indices[1]+1])
                r_vars.append(self.running_var[indices[0]:indices[1]+1])
            w = torch.cat(ws, dim=0).contiguous()
            b = torch.cat(bs, dim=0).contiguous()
            r_mean = torch.cat(r_means, dim=0).contiguous()
            r_var = torch.cat(r_vars, dim=0).contiguous()

            return F.batch_norm(
                input, r_mean, r_var, w, b,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

    def build_nn_module(self):
        if self.indices is not None:
            if not isinstance(self.indices[0], int):
                num_features = 0
                for indices in self.indices:
                    if indices is None:
                        continue
                    num_features += indices[1] - indices[0] + 1
                if num_features == 0:
                    num_features = self.num_features
            else:
                num_features = self.indices[1] - self.indices[0] + 1
        else:
            num_features = self.num_features
        
        return nn.BatchNorm2d(num_features, self.eps, self.momentum, self.weight is not None, self.track_running_stats)

class DynamicConv2d_multiconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(DynamicConv2d_multiconv2d, self).__init__()
        self.in_indices = None
        self.out_indices = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.weight_num = kwargs.pop('weight_num')
        if isinstance(self.weight_num, int):
            conv_list = []
            for i in range(self.weight_num):
                conv_list.append(DynamicConv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups, bias = bias))
            self.conv_list = nn.Sequential(*conv_list)
        else:
            raise RuntimeError("not support this type")

    def forward(self, x):
        # get channel settings 
        in_indices, out_indices = self.in_indices, self.out_indices
        if in_indices is None and out_indices is None:
            # normal conv
            assert len(self.conv_list) == 1, f'len(self.conv_list): {len(self.conv_list)}, self.in_indices: {self.in_indices}, self.out_indices: {self.out_indices}'
            return self.conv_list(x)
            #return super().forward(x)
        if out_indices is None:
            raise RuntimeError('stop, not support None type')
            # not searching module
            in_c = x.size(1)
            if in_c == self.in_channels:
                return super().forward(x)
            else:
                if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                    ws = []
                    for indices in in_indices:
                        ws.append(self.weight[:, indices[0]:indices[1]+1])
                    w = torch.cat(ws, dim=1).contiguous()
                else:
                    w = self.weight[:, in_indices[0]:in_indices[1]+1].contiguous()
                return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif in_indices is None or in_indices[0] is None:
            raise RuntimeError('stop, not support None type')
            w = self.weight[out_indices[0]:out_indices[1]+1].contiguous()
            b = self.bias[out_indices[0]:out_indices[1]+1] if self.bias is not None else None
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, 
                            int(self.groups * w.shape[0] / self.out_channels) if self.groups != 1 else 1)
        else:
            if self.groups == 1:
                if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels TODO: support group conv
                    raise RuntimeError('stop, we do not support tuple and list')
                    ws = []
                    for indices in in_indices:
                        ws.append(self.weight[out_indices[0]:out_indices[1]+1, indices[0]:indices[1]+1])
                    w = torch.cat(ws, dim=1).contiguous()
                else:
                    w = self.set_weights()
                    #w = self.weight[out_indices[0]:out_indices[1]+1, in_indices[0]:in_indices[1]+1].contiguous()
            else:
                w = self.set_weights()
                #w = self.weight[out_indices[0]:out_indices[1]+1]
            b = self.set_bias() if self.bias is not None else None
            #b = self.bias[out_indices[0]:out_indices[1]+1] if self.bias is not None else None
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, 
                            int(self.groups * w.shape[0] / self.out_channels) if self.groups != 1 else 1)

    def set_indices(self, in_indices, out_indices):
        self.in_indices = in_indices
        self.out_indices = out_indices

    def set_weights(self):
        pass

    def set_bias(self):
        pass

    def set_running_mean(self):
        pass

    def set_running_var(self):
        pass
    
    def flops(self, input_shape):
        c, w, h = input_shape
        w = (w + self.padding[0] * 2 - self.kernel_size[0]) // self.stride[0] + 1
        h = (h + self.padding[1] * 2 - self.kernel_size[1]) // self.stride[1] + 1
        in_indices, out_indices = self.in_indices, self.out_indices
        in_channels, out_channels, groups = self.in_channels, self.out_channels, self.groups  # normal conv
        if in_indices is not None and in_indices[0] is not None:
            if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                in_channels = 0
                for indices in in_indices:
                    in_channels += indices[1] - indices[0] + 1
            else:
                in_channels = in_indices[1] - in_indices[0] + 1
        if out_indices is not None:
            out_channels = out_indices[1] - out_indices[0] + 1
        if self.groups != 1:
            groups = int(self.groups * out_channels / self.out_channels)
        c = out_channels
        flops = in_channels * out_channels * w * h // groups * self.kernel_size[0] * self.kernel_size[1]
        return flops, (c, w, h)

    def build_nn_module(self):
        in_indices, out_indices = self.in_indices, self.out_indices
        in_channels, out_channels, groups = self.in_channels, self.out_channels, self.groups  # normal conv
        if in_indices is not None and in_indices[0] is not None:
            if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                in_channels = 0
                for indices in in_indices:
                    in_channels += indices[1] - indices[0] + 1
            else:
                in_channels = in_indices[1] - in_indices[0] + 1
        if out_indices is not None:
            out_channels = out_indices[1] - out_indices[0] + 1
        if self.groups != 1:
            groups = int(self.groups * out_channels / self.out_channels)
        return nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, 
                         self.dilation, groups, self.bias is not None)    


class DynamicConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(DynamicConv2d, self).__init__(*args, **kwargs)
        self.in_indices = None
        self.out_indices = None

    def forward(self, x):
        # get channel settings 
        in_indices, out_indices = self.in_indices, self.out_indices
        if in_indices is None and out_indices is None:
            # normal conv
            return super().forward(x)
        if out_indices is None:
            # not searching module
            in_c = x.size(1)
            if in_c == self.in_channels:
                return super().forward(x)
            else:
                if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                    ws = []
                    for indices in in_indices:
                        ws.append(self.weight[:, indices[0]:indices[1]+1])
                    w = torch.cat(ws, dim=1).contiguous()
                else:
                    w = self.weight[:, in_indices[0]:in_indices[1]+1].contiguous()
                return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif in_indices is None or in_indices[0] is None:
            w = self.weight[out_indices[0]:out_indices[1]+1].contiguous()
            b = self.bias[out_indices[0]:out_indices[1]+1] if self.bias is not None else None
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, 
                            int(self.groups * w.shape[0] / self.out_channels) if self.groups != 1 else 1)
        else:
            if self.groups == 1:
                if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels TODO: support group conv
                    ws = []
                    for indices in in_indices:
                        ws.append(self.weight[out_indices[0]:out_indices[1]+1, indices[0]:indices[1]+1])
                    w = torch.cat(ws, dim=1).contiguous()
                else:
                    w = self.weight[out_indices[0]:out_indices[1]+1, in_indices[0]:in_indices[1]+1].contiguous()
            else:
                w = self.weight[out_indices[0]:out_indices[1]+1]
            b = self.bias[out_indices[0]:out_indices[1]+1] if self.bias is not None else None
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, 
                            int(self.groups * w.shape[0] / self.out_channels) if self.groups != 1 else 1)

    def set_indices(self, in_indices, out_indices):
        self.in_indices = in_indices
        self.out_indices = out_indices
    
    def flops(self, input_shape):
        c, w, h = input_shape
        w = (w + self.padding[0] * 2 - self.kernel_size[0]) // self.stride[0] + 1
        h = (h + self.padding[1] * 2 - self.kernel_size[1]) // self.stride[1] + 1
        in_indices, out_indices = self.in_indices, self.out_indices
        in_channels, out_channels, groups = self.in_channels, self.out_channels, self.groups  # normal conv
        if in_indices is not None and in_indices[0] is not None:
            if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                in_channels = 0
                for indices in in_indices:
                    in_channels += indices[1] - indices[0] + 1
            else:
                in_channels = in_indices[1] - in_indices[0] + 1
        if out_indices is not None:
            out_channels = out_indices[1] - out_indices[0] + 1
        if self.groups != 1:
            groups = int(self.groups * out_channels / self.out_channels)
        c = out_channels
        flops = in_channels * out_channels * w * h // groups * self.kernel_size[0] * self.kernel_size[1]
        return flops, (c, w, h)

    def build_nn_module(self):
        in_indices, out_indices = self.in_indices, self.out_indices
        in_channels, out_channels, groups = self.in_channels, self.out_channels, self.groups  # normal conv
        if in_indices is not None and in_indices[0] is not None:
            if isinstance(in_indices[0], (tuple, list)):  # several non-continuous groups of channels
                in_channels = 0
                for indices in in_indices:
                    in_channels += indices[1] - indices[0] + 1
            else:
                in_channels = in_indices[1] - in_indices[0] + 1
        if out_indices is not None:
            out_channels = out_indices[1] - out_indices[0] + 1
        if self.groups != 1:
            groups = int(self.groups * out_channels / self.out_channels)
        return nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, self.padding, 
                         self.dilation, groups, self.bias is not None)        


