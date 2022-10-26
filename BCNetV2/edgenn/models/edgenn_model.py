import torch
import torch.nn as nn
from ..algorithm.utils.pruning_converter import convert_conv, convert_bn, convert_linear
from ..algorithm.utils.edgenn_graph import EdgeNNGraph

import logging
logger = logging.getLogger()

class EdgeNNModel(nn.Module):
    def __init__(self, model: nn.Module, loss_fn: nn.Module, pruning=False, input_shape=[3, 224, 224], load_graph_path='', save_graph_path=''):
        super(EdgeNNModel, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

        if pruning:
            # convert the ops in model to pruning ops
            convert_conv(self.model)
            convert_bn(self.model)
            convert_linear(self.model)
            logger.info('Converted model to pruning model.')

            self.graph = EdgeNNGraph(self.model, input_shape, load_graph_path, save_graph_path)
            logger.info('Trace computational graph done ==>')
            logger.info(self.graph)

    def forward(self, input, target):
        output = self.model(input)
        loss = self.loss_fn(output, target)
        return output, loss

    def get_layer_flops(self, input_shape):
        return self.graph.get_layer_flops(input_shape)

    def get_channel_choices(self, bins, min_bins):
        return self.graph.get_channel_choices(bins, min_bins)

    def set_channel_choices(self, choices, bins, min_bins):
        self.graph.set_channel_choices(choices, bins, min_bins)

    def fold_dynamic_nn(self, choices=None, bins=None, min_bins=None):
        self.graph.fold_dynamic_nn(self.model, choices, bins, min_bins)

