import torch.nn as nn
from core.search_space.ops import OPS
from core.search_space.ops import FC
import copy

def init_model(cfg_net, **kwargs):
    model = nn.ModuleList()

    if 'rebuild_channels' in kwargs:
        channels = kwargs.pop('rebuild_channels')
    else:
        channels = cfg_net.pop('channels')
    assert channels is not None
    for _type in cfg_net:
        if _type == 'backbone':
            if 'final_pooling' in cfg_net[_type]:
                final_pooling = cfg_net[_type].pop('final_pooling')
            else:
                final_pooling = True
            stage_inp = 3  # hard code
            for stage in cfg_net[_type]:
                inp = stage_inp
                # 1. this place I can change to achieve channel search and certain block?
                n, stride, _, _, t, c_search, ops = cfg_net[_type][stage]
                # for suxiu
                if len(t) == 1:
                    t = t * len(ops)
                elif len(t) == 0:
                    t = [1] * len(ops)
                for i in range(n):
                    stride = stride if i == 0 else 1
                    module_ops = nn.ModuleList()
                    # this place ops only one ?
                    for _t, op in zip(t, ops):
                        if op in ['conv2d', 'conv3x3', 'conv7x7']:
                            oup = channels.pop(0)
                            module_ops.append(OPS[op](inp, oup, _t, stride, c_search))
                        elif op in ['maxpool3x3', 'maxpool2x2', 'Adaptmaxpool']:
                            module_ops.append(OPS[op](inp, oup, _t, stride, c_search)) 
                        elif 'ir' in op or 'nr' in op:
                            if _t == 1 and 'ir' in op:
                                op_channels = [channels.pop(0), channels.pop(0)]
                            else:
                                op_channels = [channels.pop(0), channels.pop(0), channels.pop(0)]
                            oup = op_channels[-1]
                            if 'nr' in op and (stride != 1 or inp != oup):
                                channels.pop(0)
                            module_ops.append(OPS[op](inp, oup, _t, stride, c_search, op_channels))
                    model.add_module(f'{_type}_{stage}_{i}', module_ops)
                    inp = oup
                stage_inp = oup
            assert len(channels) == 0
            if final_pooling:
                model.add_module(f'{_type}_final_pooling', nn.ModuleList([nn.AdaptiveAvgPool2d(1)]))
        elif _type != 'loss_type':
            for fc_cfg in cfg_net[_type]:
                #print("type is {}, fc_cfg is {}".format(_type, fc_cfg))
                cfg = cfg_net[_type][fc_cfg]
                dim_in = stage_inp
                dim_out = cfg['dim_out']
                use_bn = cfg.get('use_bn', False)
                act = cfg.get('act', None)
                dp = cfg.get('dp', 0)

                model.add_module('_'.join([_type, fc_cfg]),
                                 nn.ModuleList([FC(dim_in, dim_out, use_bn, dp, act)]))
    return model
