from typing import Union, Dict, Sequence

import torch.nn as nn
from golem.core.dag.graph_node import GraphNode

from nas.graph.node.nas_graph_node import NasNode
from nas.model.pytorch.layers.kan_convolutional.KANConv import KAN_Convolutional_Layer, KAN_CrossConvolution
from nas.model.pytorch.layers.kan_convolutional.KANLinear import KANLinear
from nas.model.pytorch.layers.torch_conv_kan.conv_kan import KANConv2DLayer
from nas.model.pytorch.layers.torch_conv_kan.fast_kan_conv import FastKANConv2DLayer

from .torch_conv_kan.fast_kan_conv import FastKANConv2DLayer
from .torch_conv_kan.conv_kan import KANConv2DLayer
from .torch_conv_kan.layers import FastKANLayer, KANLayer


def conv2d(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('out_shape')
    kernel_size = node.parameters.get('kernel_size')
    stride = node.parameters.get('stride', 1)
    # _padding = kernel_size // 2
    _padding = node.parameters.get('padding')
    is_transposed = node.parameters.get("is_transposed", False)
    return nn.Conv2d(input_dim, out_shape, kernel_size, stride, padding=_padding) if not is_transposed else \
        nn.ConvTranspose2d(input_dim, out_shape, kernel_size, stride, padding=_padding, output_padding=_padding)


def kan_conv2d(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('out_shape')
    # Check the out_shape % input_dim == 0:
    # print(input_dim, out_shape, out_shape % input_dim == 0)
    # if out_shape % input_dim != 0:
    #     print("FAIL")
    #     raise ValueError(f'out_shape must be divisible by input_dim')

    kernel_size = node.parameters.get('kernel_size')
    # stride = node.parameters.get('stride', 1)
    padding = node.parameters.get('padding')
    if isinstance(padding, Sequence):
        padding = padding[0]

    # padding = kernel_size // 2

    # Spline stuff:
    grid_size = node.parameters.get('grid_size')
    spline_order = node.parameters.get('spline_order')

    return KAN_CrossConvolution(
        in_channels=input_dim,
        out_channels=out_shape,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(padding, padding),

        grid_size=grid_size,
        spline_order=spline_order,

        transposed=node.parameters.get('is_transposed')
    )

    # return FastKANConv2DLayer(
    #     input_dim=input_dim,
    #     output_dim=out_shape,
    #     kernel_size=kernel_size,
    #     groups=1,
    #     padding=kernel_size - 1 if node.parameters.get('is_transposed') else 0,
    #     stride=1,
    #     dilation=1,
    #     grid_size=grid_size
    #     # spline_order=spline_order
    # )

    # return KANConv2DLayer(
    #     input_dim=input_dim,
    #     output_dim=out_shape,
    #     kernel_size=kernel_size,
    #     groups=1,
    #     padding=kernel_size - 1 if node.parameters.get('is_transposed') else 0,
    #     stride=1,
    #     dilation=1,
    #     grid_size=grid_size,
    #     spline_order=spline_order
    # )


def linear(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('out_shape')
    return nn.Linear(input_dim, out_shape)


def kan_linear(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    out_shape = node.parameters.get('out_shape')

    # Spline stuff:
    grid_size = node.parameters.get('grid_size')
    spline_order = node.parameters.get('spline_order')

    return KANLinear(in_features=input_dim, out_features=out_shape, grid_size=grid_size, spline_order=spline_order)

    # Spline order isn't applicable since rbfs are used instead of splines
    # return FastKANLayer(input_dim, out_shape, num_grids=grid_size)

    # return KANLayer(input_dim, out_shape, grid_size=grid_size, spline_order=spline_order)


def dropout(node: NasNode, **kwargs):
    dropout_prob = node.parameters.get('drop')
    return nn.Dropout(p=dropout_prob)


def batch_norm(node: NasNode, **inputs_dict):
    input_dim = inputs_dict.get('input_dim')
    eps = node.parameters.get('epsilon')
    momentum = node.parameters.get('momentum')
    return nn.BatchNorm2d(input_dim, eps, momentum)


def supplementary_pooling(node: NasNode, **inputs_dict):
    # Just kernel size:
    kernel_size = node.parameters.get('pooling_kernel_size')
    pool_layer = nn.MaxPool2d if node.parameters['pooling_mode'] == 'max' else nn.AvgPool2d
    return pool_layer(kernel_size)


def pooling(node: NasNode, **inputs_dict):
    kernel_size = node.parameters.get('pool_size')
    stride = node.parameters.get('pool_stride')
    padding = node.parameters.get('padding', 0)
    pool_layer = nn.MaxPool2d if node.parameters['mode'] == 'max' else nn.AvgPool2d
    return pool_layer(kernel_size, stride, padding=padding)


def ada_pool2d(node: NasNode, **inputs_dict):
    out_shape = node.parameters.get('out_shape')
    mode = node.parameters.get('mode')
    pool_layer = nn.AdaptiveMaxPool2d if mode == 'max' else nn.AdaptiveAvgPool2d
    return pool_layer(out_shape)


def flatten(*args, **kwargs):
    return nn.Flatten()


class TorchLayerFactory:
    @staticmethod
    def get_layer(node: Union[GraphNode, NasNode]) -> Dict:
        _layers = {'conv2d': conv2d,
                   'kan_conv2d': kan_conv2d,
                   'linear': linear,
                   'kan_linear': kan_linear,
                   'dropout': dropout,
                   'batch_norm': batch_norm,
                   'pooling2d': pooling,
                   'adaptive_pool2d': ada_pool2d,
                   'flatten': flatten}
        layer = {}
        layer_type = node.name
        layer_fun = _layers.get(layer_type)
        layer['weighted_layer'] = layer_fun
        if layer_fun is None:
            raise ValueError(f'Wrong layer type: {layer_type}')
        if 'momentum' in node.parameters:
            layer['normalization'] = _layers.get('batch_norm')
        if 'pooling_kernel_size' in node.parameters:
            layer['pooling'] = supplementary_pooling
        return layer

    @staticmethod
    def get_activation(activation_name: str):
        activations = {'relu': nn.ReLU,
                       'elu': nn.ELU,
                       'selu': nn.SELU,
                       'softmax': nn.Softmax,
                       'sigmoid': nn.Sigmoid,
                       'tanh': nn.Tanh,
                       'softplus': nn.Softplus,
                       'softsign': nn.Softsign,
                       'hard_sigmoid': nn.Hardsigmoid,
                       }
        activation = activations.get(activation_name)
        if activation is None:
            raise ValueError(f'Wrong activation function: {activation_name}')
        return activation
