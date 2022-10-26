from edgenn.models.pruning_ops import DynamicConv2d
import torch


def get_flops(model, input_shape=(3, 224, 224)):
    if hasattr(model, 'module'):
        model = model.module
    training_flag = model.training
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        if self.groups != 1:
            groups = input_channels // (self.in_channels // self.groups)    # for DynamicConv2d
        else:
            groups = 1

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (input_channels // groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width
        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net, hook_handle):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_handle.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handle.append(net.register_forward_hook(linear_hook))
            return
        for c in childrens:
            foo(c, hook_handle)

    hook_handle = []
    foo(model, hook_handle)
    input = torch.rand(*input_shape).unsqueeze(0).cuda()
    target = torch.tensor([0]).long().cuda()
    model.eval()
    with torch.no_grad():
        out, loss = model(input, target)
    for handle in hook_handle:
        handle.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(training_flag)
    return total_flops


from ...models import Choice
import logging
from collections import OrderedDict
logger = logging.getLogger()

_basic_flops = 0
_searchable_flops = []

def get_spos_flops(model, test_subnet, input_shape=(3, 224, 224)):
    if len(_searchable_flops) == 0:
        logger.info('Measuring FLOPs...')
        choice_modules = []
        for mod in model.modules():
            if isinstance(mod, Choice):
                choice_modules.append(mod)
        choices = [len(m) for m in choice_modules]
        for length in choices:
            _searchable_flops.append([0] * length)
        # measure all choice blocks
        for i in range(max(choices)):
            subnet = [min(i, length - 1) for length in choices]
            for idx, m in zip(subnet, choice_modules):
                m.set_sub(idx)
            total_flops, basic_flops, choice_flops = _get_spos_flops(model, subnet, input_shape)
            global _basic_flops
            _basic_flops = basic_flops
            for idx, sub_idx in enumerate(subnet):
                _searchable_flops[idx][sub_idx] = choice_flops[idx]

    return _basic_flops + sum([_searchable_flops[idx][sub_idx] for idx, sub_idx in enumerate(test_subnet)])
    
def _get_spos_flops(model, subnet, input_shape=(3, 224, 224)):
    if hasattr(model, 'module'):
        model = model.module
    training_flag = model.training
    basic_flops = 0

    choice_flops = OrderedDict()
    def conv_hook(self, input, output, choice_id=None):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        if self.groups != 1:
            groups = input_channels // (self.in_channels // self.groups)    # for DynamicConv2d
        else:
            groups = 1

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (input_channels // groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        if choice_id is None:
            nonlocal basic_flops
            basic_flops += flops
        else:
            if choice_id not in choice_flops:
                choice_flops[choice_id] = 0
            choice_flops[choice_id] += flops

    def linear_hook(self, input, output, choice_id=None):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement()
        flops = batch_size * weight_ops
        if choice_id is None:
            nonlocal basic_flops
            basic_flops += flops
        else:
            if choice_id not in choice_flops:
                choice_flops[choice_id] = 0
            choice_flops[choice_id] += flops

    choice_num = 0
    def foo(net, hook_handle, choice_id=None):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_handle.append(net.register_forward_hook(lambda *args: conv_hook(*args, choice_id)))
            if isinstance(net, torch.nn.Linear):
                hook_handle.append(net.register_forward_hook(lambda *args: linear_hook(*args, choice_id)))
            return
        for c in childrens:
            if isinstance(c, Choice):
                nonlocal choice_num
                choice_id = choice_num
                choice_num += 1
            foo(c, hook_handle, choice_id)

    hook_handle = []
    foo(model, hook_handle)
    input = torch.rand(*input_shape).unsqueeze(0).cuda()
    target = torch.tensor([0]).long().cuda()
    model.eval()
    with torch.no_grad():
        out, loss = model(input, target)
    for handle in hook_handle:
        handle.remove()

    model.train(training_flag)

    choice_flops = [choice_flops[idx] if idx in choice_flops else 0 for idx in range(len(subnet))]
    total_flops = basic_flops + sum(choice_flops)
    return total_flops, basic_flops, choice_flops


