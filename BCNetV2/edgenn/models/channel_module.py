import torch
import torch.nn as nn
from .ops import DynamicConv2d, DynamicBatchNorm2d, DynamicLinear
from autopath.utils.model_converter import convert_conv, convert_bn, convert_linear
from autopath.utils.measure import conv2d_flops
import re
import numpy as np
import math


class ChannelModule(nn.Module):
    def __init__(self, module, input_shape=[3, 96, 96], subnet=None, bin_size=None):
        super(ChannelModule, self).__init__()
        assert isinstance(module, nn.Module), 'module must be nn.Module or its subclass'
        self.module = module

        # convert module to channel pruning module
        convert_conv(self.module)
        convert_bn(self.module)
        convert_linear(self.module)

        # build graph
        module_status = self.module.training
        self.module.eval()
        print('==tracing computational graph')
        input = torch.zeros([1] + input_shape)
        trace, _ = torch.jit.get_trace_graph(self.module, input)
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        graph = trace.graph()

        conv_graph = Graph()
        for node in graph.nodes():
            kind = node.kind()
            if ignore_op.get(kind):
                ignore_op[kind](self.module, conv_graph, node)
            elif normal_op.get(kind):
                normal_op[kind](self.module, conv_graph, node)
            elif special_op.get(kind):
                pass  # deal it later
            else:
                raise Exception('not supported op type: {}'.format(kind))
        # deal with special op later
        conv_graph.is_first = True
        for node in graph.nodes():
            kind = node.kind()
            if ignore_op.get(kind):
                ignore_op[kind](self.module, conv_graph, node)
            elif normal_op.get(kind):
                normal_op[kind](self.module, conv_graph, node)
            elif special_op.get(kind):
                special_op[kind](self.module, conv_graph, node)
        # output of last node can not search
        *_, last_node = conv_graph.values()
        last_node.set_same_as(None)
        print(conv_graph)
        self.conv_graph = conv_graph

        if module_status:
            self.module.train()

        # get valid search number
        '''
        self.valid_search_number = 0
        for node_id in conv_graph:
            if conv_graph[node_id].same_as is None and not conv_graph[node_id].ignore:
                self.valid_search_number += 1
        self.valid_search_number -= 1   # last bridge conv
        '''

        self.valid_search_number = 0
        for idx, (key, node) in enumerate(conv_graph.items()):
            if node.same_as is not None or node.ignore:
                continue
            conv = node.module
            if idx == len(conv_graph) - 1:
                '''bridge module'''    
                break
            if conv.groups == 1:
                self.valid_search_number += 1
        print('number of search conv blocks: ', self.valid_search_number)

        print('FLOPs of original supernet: ', self.conv_graph.flops(input_shape))

        # convert the supernet into subnet
        if subnet is not None:
            assert len(subnet) == self.valid_search_number
            # first, set subnet
            self.set_subnet(self.decode_subnet(subnet, bin_size)[0])
            # then, convert dynamic op to nn.op
            _convert_subnet(self.module)
            # count flops
            print('FLOPs of subnet: ', self.conv_graph.flops(input_shape))


    def decode_subnet(self, subnet, bin_size):
        if isinstance(subnet, np.ndarray):
            subnet = subnet.tolist()
        assert len(subnet) == self.valid_search_number
        return self.conv_graph.decode_subnet(subnet, bin_size)

    def set_subnet(self, subnet):
        self.conv_graph.set_subnet(subnet)
        
    def __iter__(self):
        for layer in self.module:
            yield layer

    def __len__(self):
        return len(self.module)

    def reset_grad(self):
        pass

    def forward(self, x):
        return self.module(x)


from collections import OrderedDict
class Graph(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redir_in_dict = {}
        self.redir_out_dict = {}
        self.is_first = True
        self.first_id = None

    def __str__(self):
        print_str = ''
        for key in self:
            print_str += '{} {}\n'.format(key, self[key])
        return print_str

    def set_subnet(self, subnet):
        subnet = subnet.copy()
        for node_id in self:
            if self[node_id].concats is not None:
                in_indices_list = []
                for id in self[node_id].concats:
                    in_indices_list.append(self[id].module.out_indices)
                in_indices = in_indices_list
            else:
                if self[node_id].input_id is None:
                    in_indices = None
                else:
                    in_indices = self[self[node_id].input_id].module.out_indices
            if self[node_id].ignore:
                # ignore
                self[node_id].module.set_indices(in_indices, None)
            elif self[node_id].same_as is None:
                out_indices = subnet.pop(0)
                self[node_id].module.set_indices(in_indices, out_indices)
            else:
                same_module = self[self[node_id].same_as].module
                out_indices = same_module.out_indices
                self[node_id].module.set_indices(in_indices, out_indices)

            ''' bn '''
            if self[node_id].bn_module is not None:
                self[node_id].bn_module.set_indices(self[node_id].module.out_indices)
            if self[node_id].concat_bn is not None:
                self[node_id].concat_bn.set_indices(self[node_id].module.in_indices)
        assert len(subnet) == 0
    
    def decode_subnet(self, subnet, bin_size):
        ''' positive direction '''
        bins = subnet.copy()
        subnet1 = []
        for idx, (key, node) in enumerate(self.items()):
            if node.same_as is not None or node.ignore:
                continue
            conv = node.module
            if idx == len(self) - 1:
                '''for bridge module'''    
                subnet1.append(None)
                break
            if conv.groups == 1:
                bin_number = bins.pop(0)
                channels = math.ceil(bin_number / bin_size * conv.out_channels)
                subnet1.append([0, channels-1])
                if node.input_id is None:   # first node
                    conv.set_indices(None, [0, channels-1])
                else:
                    if node.concats is None:
                        prev_node = self[node.input_id]
                        conv.set_indices(prev_node.module.out_indices, [0, channels-1])
                    else:
                        in_indices_list = []
                        for node_id in node.concats:
                            prev_node = self[node_id]
                            in_indices_list.append(prev_node.module.out_indices)
                        conv.set_indices(in_indices_list, [0, channels-1])
                
            else:  # group convolution
                if len(subnet1) == 0:  # first conv is group conv
                    conv.set_indices(None, None)
                    subnet1.append(None)
                else:
                    prev_node = self[node.input_id]
                    in_channels_per_group = conv.in_channels // conv.groups
                    out_channels_per_group = conv.out_channels // conv.groups
                    start_group_idx = prev_node.module.out_indices[0] // in_channels_per_group
                    end_group_idx = (prev_node.module.out_indices[1] + 1) // in_channels_per_group
                    conv.set_indices(prev_node.module.out_indices, 
                                [start_group_idx*out_channels_per_group, end_group_idx*out_channels_per_group-1])
                    subnet1.append([start_group_idx*out_channels_per_group, end_group_idx*out_channels_per_group-1])

        ''' negative direction '''
        bins = subnet.copy()
        subnet2 = []
        
        for idx, (key, node) in enumerate(self.items()):
            if node.same_as is not None or node.ignore:
                continue
            conv = node.module
            if idx == len(self) - 1:
                '''for bridge module'''    
                subnet2.append(None)
                break
            if conv.groups == 1:
                bin_number = bins.pop(0)  # keep the same with positive direction
                channels = math.ceil(bin_number / bin_size * conv.out_channels)
                subnet2.append([conv.out_channels - channels, conv.out_channels-1])
                if node.input_id is None:   # first node
                    conv.set_indices(None, [conv.out_channels - channels, conv.out_channels-1])
                else:
                    if node.concats is None:
                        prev_node = self[node.input_id]
                        conv.set_indices(prev_node.module.out_indices, [conv.out_channels - channels, conv.out_channels-1])
                    else:
                        in_indices_list = []
                        for node_id in node.concats:
                            prev_node = self[node_id]
                            in_indices_list.append(prev_node.module.out_indices)
                        conv.set_indices(in_indices_list, [conv.out_channels - channels, conv.out_channels-1])

            else:  # group convolution
                if len(subnet2) == 0:  # first conv is group conv
                    conv.set_indices(None, None)
                    subnet2.append(None)
                else:
                    prev_node = self[node.input_id]
                    in_channels_per_group = conv.in_channels // conv.groups
                    out_channels_per_group = conv.out_channels // conv.groups
                    start_group_idx = prev_node.module.out_indices[0] // in_channels_per_group
                    end_group_idx = (prev_node.module.out_indices[1] + 1) // in_channels_per_group
                    conv.set_indices(prev_node.module.out_indices, 
                                [start_group_idx*out_channels_per_group, end_group_idx*out_channels_per_group-1])
                    subnet2.append([start_group_idx*out_channels_per_group, end_group_idx*out_channels_per_group-1])

        return subnet1, subnet2

    def flops(self, input_shape, subnet=None):
        if subnet is not None:
            self.set_subnet(subnet)
        total_flops = 0
        for node_id, node in self.items():
            if node.input_id is None:   # first node
                if isinstance(node.module, DynamicConv2d):
                    flops, shape = node.module.flops(input_shape)
                else:
                    flops, shape = conv2d_flops(node.module, input_shape)
                total_flops += flops
                node.output_shape = shape
            else:
                if node.concats is not None:
                    #in_shape = [-1] + self[node.concats[0]].output_shape[1:]
                    in_shape = self[node.concats[0]].output_shape
                else:
                    in_shape = self[node.input_id].output_shape
                if isinstance(node.module, DynamicConv2d):
                    flops, shape = node.module.flops(in_shape)
                else:
                    flops, shape = conv2d_flops(node.module, in_shape)
                total_flops += flops
                node.output_shape = shape
        return total_flops


class NormalNode(object):
    def __init__(self, id, scope, module, input_id, bn_module=None, concats=None, same_as=None, concat_bn=None, ignore=False):
        self.id = id
        self.input_id = input_id
        self.module = module
        self.bn_module = bn_module
        self.concats = concats
        self.same_as = same_as
        self.scope = scope
        self.concat_bn = concat_bn
        self.ignore = ignore
    
    def __str__(self):
        return 'NormalNode(id={}, scope={}, input_id={}, concats={}, same_as={}, ignore={}, module={}, bn_module={}, concat_bn={})'.\
            format(self.id, self.scope, self.input_id, self.concats, self.same_as, self.ignore, self.module, self.bn_module, self.concat_bn)

    def set_bn_module(self, bn_module):
        self.bn_module = bn_module

    def set_same_as(self, same_as):
        self.same_as = same_as

    def set_concats(self, concats):
        self.concats = concats

    def set_concat_bn(self, concat_bn):
        self.concat_bn = concat_bn


def _convert_subnet(model):
    for key, child in model.named_children():
        if isinstance(child, (DynamicLinear, DynamicConv2d, DynamicBatchNorm2d)):
            setattr(model, key, child.build_nn_module())
        else:
            _convert_subnet(child)

'''
def _convert_subnet(model, conv_graph):
    for idx, (key, node) in enumerate(conv_graph.items()):
        module = _get_module_by_scope(model, node.scope)
        # find parent module
        p = re.compile(r'\[(.*?)\]', re.S)
        attrs = re.findall(p, node.scope) 
        parent = model
        for key in attrs[:-1]:
            parent = getattr(parent, key)
        if isinstance(module, (DynamicLinear, DynamicConv2d, DynamicBatchNorm2d)):
            setattr(parent, attrs[-1], module.build_nn_module())
        else:
            pass
'''


def _get_real_id(redir_dict, input_id):
    while redir_dict.get(input_id) is not None:
        input_id = redir_dict[input_id]
    return input_id


def _get_module_by_scope(module, scope):
    p = re.compile(r'\[(.*?)\]', re.S)
    attrs = re.findall(p, scope) 
    for key in attrs:
        module = getattr(module, key)
        assert module is not None
    return module

def _bn(model, conv_graph, node):
    input_id = int(str(next(node.inputs())).split(' ')[0])
    if conv_graph.get(input_id) is not None:
        conv_graph[input_id].set_bn_module(_get_module_by_scope(model, node.scopeName()))
    else:
        # concat bn
        out_conv_id = _get_real_id(conv_graph.redir_out_dict, input_id)
        if conv_graph.get(out_conv_id) is not None:
            conv_graph[out_conv_id].set_concat_bn(_get_module_by_scope(model, node.scopeName()))
    _redir(model, conv_graph, node)

def _redir(model, conv_graph, node):
    id = int(str(node).split(':')[0].replace('%', ''))
    input_id = int(str(next(node.inputs())).split(' ')[0])
    conv_graph.redir_in_dict[id] = input_id
    if conv_graph.get(input_id) is None:
        conv_graph.redir_out_dict[input_id] = id

def _conv(model, conv_graph: Graph, node):
    id = int(str(node).split(':')[0].replace('%', ''))
    if not conv_graph.is_first:
        input_id = int(str(next(node.inputs())).split(' ')[0])
        input_id = _get_real_id(conv_graph.redir_in_dict, input_id)
        if input_id == conv_graph.first_id:
            input_id = None
    else:
        input_id = None
        conv_graph.is_first = False
        conv_graph.first_id = int(str(next(node.inputs())).split(' ')[0])

    scope = node.scopeName()
    module = _get_module_by_scope(model, scope)
    ignore = getattr(module, 'pruning_ignore', False)
    if input_id:
        conv_graph.redir_out_dict[input_id] = id
    if conv_graph.get(id) is not None:
        old = conv_graph[id]
        conv_graph[id] = NormalNode(id, scope, module, input_id, old.bn_module, old.concats, old.same_as, old.concat_bn, ignore=ignore)
    else:
        conv_graph[id] = NormalNode(id, scope, module, input_id, ignore=ignore)

def _add(model, conv_graph, node):
    inputs = node.inputs()
    id = int(str(node).split(':')[0].replace('%', ''))
    input_id1 = int(str(next(inputs)).split(' ')[0])
    input_id2 = int(str(next(inputs)).split(' ')[0])
    # get conv id
    input_id1 = _get_real_id(conv_graph.redir_in_dict, input_id1)
    input_id2 = _get_real_id(conv_graph.redir_in_dict, input_id2)
    input_id1, input_id2 = min(input_id1, input_id2), max(input_id1, input_id2)
    conv_graph.redir_in_dict[id] = input_id1  # same as conv id1
    conv_graph.redir_out_dict[input_id1] = id
    conv_graph[input_id2].set_same_as(input_id1)

def _concat(model, conv_graph, node):
    id = int(str(node).split(':')[0].replace('%', ''))
    concats = []
    for input in node.inputs():
        input_id = int(str(input).split(' ')[0])
        concats.append(_get_real_id(conv_graph.redir_in_dict, input_id))

    for output in node.outputs():
        output_id = int(str(output).split(' ')[0])
        conv_graph[_get_real_id(conv_graph.redir_out_dict, output_id)].set_concats(concats)
    # conv_graph.redir_out_dict[id] = output_id
    

normal_op = dict({
    'onnx::Conv': _conv
})

special_op = dict(
    {
        'onnx::Concat': _concat,
        'onnx::Add': _add
    }
)

ignore_op = dict({
    'onnx::Relu': _redir,
    'onnx::BatchNormalization': _bn
})


