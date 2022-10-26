import re
import os
import torch
import math
import warnings
from collections import OrderedDict
from ...utils.base_class import BaseClass
from . import _pytorch_graph as pytorch_graph
from ...models.pruning_ops import DynamicConv2d, DynamicLinear, DynamicBatchNorm2d


def _check_torch_version(version, cmp_version='1.7.0'):
    if version.startswith('parrots'):
        return False
    major, minor, *_ = version.split('.')
    cmajor, cminor, *_ = cmp_version.split('.')
    if int(major) < int(cmajor):
        return False
    elif int(major) == int(cmajor):
        if int(minor) < int(cminor):
            return False
        else:
            return True
    else:
        return True


if _check_torch_version(torch.__version__, '1.7.0'):
    _SCOPE_VERSION = 1
else:
    _SCOPE_VERSION = 0


_PYTORCH_OP_MAP = {'aten::hardtanh': 'Activation', 'aten::relu': 'Activation', 'aten::_convolution': 'Conv', 
                   'aten::batch_norm': 'BatchNorm', 'aten::add': 'Add', 'aten::adaptive_avg_pool2d': 'AdaptiveAvgPool2d', 
                   'aten::view': 'View', 'aten::addmm': 'Linear', 'aten::mul': 'Mul', 'aten::div': 'Div'}
_PYTORCH_IGNORED_OP = ['IO Node', 'aten::t', 'aten::dropout']

_IGNORED_OPS = ['Activation', 'AdaptiveAvgPool2d', 'View']


def _get_module_by_scope(module, scope):
    attrs = scope.split('.')
    for key in attrs:
        module = getattr(module, key, None)
        if module is None:
            return None
    return module


def _get_channel_settings(module, scope, include_parents=True):
    if include_parents:
        attrs = scope.split('.')
        channel_bins, min_channel_bins = None, None
        ignore_pruning = False
        for key in attrs:
            module = getattr(module, key)
            if hasattr(module, 'channel_bins'):
                channel_bins = module.channel_bins
            if hasattr(module, 'min_channel_bins'):
                min_channel_bins = module.min_channel_bins
            if hasattr(module, 'ignore_pruning'):
                ignore_pruning = module.ignore_pruning
    else:
        module = _get_module_by_scope(module, scope)
        channel_bins = getattr(module, 'channel_bins', None)
        min_channel_bins = getattr(module, 'min_channel_bins', None)
        ignore_pruning = getattr(module, 'ignore_pruning', False)
    return ignore_pruning, channel_bins, min_channel_bins


def _get_pytorch_graph(model, input_shape=[3, 224, 224]):
    assert len(input_shape) in [3, 4]
    if len(input_shape) == 3:
        x = torch.randn([1] + list(input_shape)).to(next(model.parameters()).device)
    else:
        x = torch.randn(input_shape).to(next(model.parameters()).device)

    nodes = pytorch_graph.graph(model, (x), verbose=False)

    pattern = re.compile('\\[(.*?)]')
    output_tensor_names = []
    valid_nodes = []
    for node in nodes:
        if node['output_size'] is None or len(node['output_size']) == 0 or len(node['input']) == 0 or node['op'] in _PYTORCH_IGNORED_OP:
            continue
        output_tensor_names.append(node['node_name'])
        res = pattern.findall(node['node_name'])
        if len(res) != 0:
            if _SCOPE_VERSION == 0:
                res = res[1:]            
        if len(res) != 0:
            node['node_name'] = '.'.join(res)
        node['op'] = _PYTORCH_OP_MAP[node['op'][:-1] if node['op'].endswith('_') else node['op']]
        del node['attributes']
        inputs = set()
        for item in node['input']:
            if item not in output_tensor_names and 'input' not in item:
                # weights or some states, we only need to collect the output of module
                continue
            results = pattern.findall(item)
            if len(results) != 0:
                if _SCOPE_VERSION == 0:
                    results = results[1:]
            if len(results) != 0:
                res = '.'.join(results)
            else:
                # no module name of this tensor, it may be input or output tensor
                res = item
            if res != node['node_name']:
                inputs.add(res)
        node['input'] = list(inputs)
        if len(inputs) == 0:
            # weights or some states, we only need to collect the output of module
            continue
        valid_nodes.append(node)
    return valid_nodes


class EdgeNNGraph(BaseClass):
    def __init__(self, model, input_shape=[3, 224, 224], load_graph_path='', save_graph_path=''):
        super(EdgeNNGraph, self).__init__()
        self.nodes = OrderedDict()

        if load_graph_path == '' or not os.path.exists(load_graph_path):
            nodes = _get_pytorch_graph(model, input_shape)
            if save_graph_path != '':
                torch.save(nodes, save_graph_path)
                print(f'Save traced graph to {save_graph_path}')
        else:
            print(f'Load traced graph from {load_graph_path}')
            nodes = torch.load(load_graph_path)

        _node_name_map = {}
        def _get_root_node_name(name):
            while name in _node_name_map:
                name = _node_name_map[name]
            return name

        for node in nodes:
            op = node['op']
            module = _get_module_by_scope(model, node['node_name'])
            if op == 'Conv':
                ignore_pruning, channel_bins, min_channel_bins = _get_channel_settings(model, node['node_name'])
                prev_node = _get_root_node_name(node['input'][0])
                if module.groups != 1:
                    same_as = prev_node
                else:
                    same_as = None
                self.nodes[node['node_name']] = ConvNode(module, output_size=node['output_size'], 
                                                         prev_node=prev_node, same_as=same_as, ignore_pruning=ignore_pruning,
                                                         channel_bins=channel_bins, min_channel_bins=min_channel_bins)
            elif op == 'BatchNorm':
                assert len(node['input']) == 1, 'Current version dose not support multiple inputs of BN.'
                self.nodes[node['input'][0]].bn_module = module
                root_node_name = _get_root_node_name(node['input'][0])
                _node_name_map[node['node_name']] = root_node_name
            elif op == 'Linear':
                prev_node = _get_root_node_name(node['input'][0])
                if prev_node not in self.nodes:
                    last_node = list(self.nodes.keys())[-1]
                    prev_node = last_node
                    warnings.warn(f'Previous node "{prev_node}" of "{node["node_name"]}" not found, automatically set to last node "{last_node}"') 
                self.nodes[node['node_name']] = LinearNode(module, prev_node=prev_node, output_size=node['output_size'])
            elif op in ['Add', 'Mul', 'Div']:
                # element-wise operations
                if len(node['input']) == 2:
                    a, b = node['input']
                    a = _get_root_node_name(a)
                    b = _get_root_node_name(b)
                    node_names = list(self.nodes.keys())
                    a_idx, b_idx = node_names.index(a), node_names.index(b)
                    if a_idx > b_idx:
                        a, b = b, a
                    self.nodes[b].set_same_as(a)
                    _node_name_map[node['node_name']] = b
                elif len(node['input']) == 1:
                    # Tensor + scala
                    _node_name_map[node['node_name']] = _get_root_node_name(node['input'][0])
                else:
                    raise RuntimeError(f'[{op}] node {node["node_name"]}: number of inputs > 2 is not supported')
            elif op == 'Concat':
                # TODO
                pass
            elif op in _IGNORED_OPS:
                root_node_name = _get_root_node_name(node['input'][0])
                _node_name_map[node['node_name']] = root_node_name
            else:
                raise NotImplementedError('Unsupported op type: {}'.format(op))

    def __repr__(self):
        str_ = ''
        for key, node in self.nodes.items():
            str_ += '({}): {}\n'.format(key, node)
        return str_

    def get_channel_choices(self, bins, min_bins):
        choices = []
        for key, node in self.nodes.items():
            if isinstance(node.module, DynamicConv2d):
                if node.same_as is None and not node.ignore_pruning:
                    if node.channel_bins is not None:
                        max_bins_ = node.channel_bins
                    else:
                        max_bins_ = bins
                    if node.min_channel_bins is not None:
                        min_bins_ = node.min_channel_bins
                    else:
                        min_bins_ = min_bins
                    choices.append([min_bins_, max_bins_])
            elif isinstance(node.module, DynamicLinear):
                # TODO: support search of out_features in Linear modules
                pass
        return choices

    def get_layer_flops(self, input_shape=(3, 224, 224)):
        flops = []
        for key, node in self.nodes.items():
            if isinstance(node.module, DynamicConv2d):
                if node.same_as is None and not node.ignore_pruning:
                    conv = self.nodes[key].module
                    in_channels = conv.in_channels
                    out_channels = conv.out_channels
                    out_shape = node.output_size
                    kernel_size = conv.kernel_size
                    ops = in_channels * out_channels * kernel_size[0] * \
                                kernel_size[1] * out_shape[2] * out_shape[3] / conv.groups
                    flops.append(ops)
            elif isinstance(node.module, DynamicLinear):
                # TODO: support search of out_features in Linear modules
                pass
        return flops

    def set_channel_choices(self, choices, bins, min_bins):
        choices = choices.copy()
        for key, node in self.nodes.items():
            if isinstance(node.module, DynamicConv2d):

                if node.prev_node is not None and not node.prev_node.startswith('input'):
                    in_indices = self.nodes[node.prev_node].module.out_indices
                else:
                    in_indices = None

                if node.ignore_pruning:
                    out_indices = None
                elif node.same_as is not None:
                    out_indices = self.nodes[node.same_as].module.out_indices
                else:
                    if node.channel_bins is not None:
                        max_bins_ = node.channel_bins
                    else:
                        max_bins_ = bins
                    channels_per_bin = node.module.out_channels / max_bins_
                    choice = choices.pop(0)
                    if isinstance(choice, int):
                        max_indice = math.ceil(choice * channels_per_bin)
                        out_indices = [0, max_indice - 1]
                    elif isinstance(choice, list):
                        if isinstance(choice[0], list):
                            # TODO: CafeNet
                            raise NotImplementedError()
                        else:
                            min_indice = math.floor(choice[0] * channels_per_bin)
                            max_indice = math.ceil(choice[1] * channels_per_bin)
                            out_indices = [min_indice, max_indice - 1]

                node.module.set_indices(in_indices, out_indices)
                if node.bn_module is not None:
                    node.bn_module.set_indices(out_indices)

            elif isinstance(node.module, DynamicLinear):
                if node.prev_node is not None and not node.prev_node.startswith('input'):
                    in_indices = self.nodes[node.prev_node].module.out_indices
                else:
                    in_indices = None
                node.module.set_indices(in_indices, None)

    def fold_dynamic_nn(self, model, choices=None, bins=None, min_bins=None):
        '''
        convert dynamic nn to traditional nn modules in torch.nn
        '''
        if choices is not None:
            self.set_channel_choices(choices, bins, min_bins)

        def _convert_nn(module):
            for k, m in module.named_children():
                if isinstance(m, (DynamicConv2d, DynamicLinear, DynamicBatchNorm2d)):
                    setattr(module, k, m.build_nn_module())
                else:
                    _convert_nn(m)
        _convert_nn(model)

class ConvNode(BaseClass):
    def __init__(self, module, bn_module=None, prev_node=None, same_as=None, output_size=None, 
                 ignore_pruning=False, channel_bins=8, min_channel_bins=1):
        super(ConvNode, self).__init__()
        self._ignored_keys = ['module', 'bn_module']
        self.module = module
        self.bn_module = bn_module
        self.prev_node = prev_node
        self.same_as = same_as
        self.output_size = output_size
        self.ignore_pruning = ignore_pruning
        self.channel_bins = channel_bins
        self.min_channel_bins = min_channel_bins

    def set_same_as(self, node_name):
        self.same_as = node_name

    def __repr__(self):
        return f'ConvNode(bn_module={None if self.bn_module is None else True}, output_size={self.output_size}, '\
               f'ignore_pruning={self.ignore_pruning}, '\
               f'channel_bins={self.channel_bins}, min_channel_bins={self.min_channel_bins}, same_as={self.same_as})'


class LinearNode(BaseClass):
    def __init__(self, module, prev_node=None, same_as=None, output_size=None):
        super(LinearNode, self).__init__()
        self._ignored_keys = ['module']
        self.module = module
        self.prev_node = prev_node
        self.same_as = same_as
        self.output_size = output_size

    def set_same_as(self, node_name):
        self.same_as = node_name

    def __repr__(self):
        return f'LinearNode(output_size={self.output_size}, same_as={self.same_as})'


class ConcatNode(BaseClass):
    # TODO
    def __init__(self, prev_nodes, output_size=None):
        super(ConcatNode, self).__init__()
        self.prev_nodes = prev_nodes
        self.output_size = output_size

    
