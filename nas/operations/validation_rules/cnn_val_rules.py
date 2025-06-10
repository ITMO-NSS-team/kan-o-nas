"""
Checkers verifying that the model structure is maintained after evolutionary operators.

The operators are rerun until the model is valid.
"""

import torch
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.dag.verification_rules import ERROR_PREFIX
from typing import Callable, Optional, List, Union

from nas.graph.base_graph import NasGraph
from nas.model.model_interface import NeuralSearchModel
from nas.model.pytorch.base_model import NASTorchModel

import traceback


def model_has_several_roots(graph: NasGraph):
    if hasattr(graph.root_node, '__iter__'):
        raise ValueError(f'{ERROR_PREFIX} model must not has more than 1 root node.')


def model_has_several_starts(graph: NasGraph):
    starts = 0
    for node in graph.nodes:
        n = 0 if node.nodes_from else 1
        starts += n
        if starts > 1:
            raise ValueError(f'{ERROR_PREFIX} model must not has more than 1 start.')


def model_has_wrong_number_of_flatten_layers(graph: NasGraph):
    flatten_count = 0
    for node in graph.nodes:
        if node.content['name'] == 'flatten':
            flatten_count += 1
    if flatten_count != 1:
        raise ValueError(f'{ERROR_PREFIX} model has wrong number of flatten layers.')


def conv_net_check_structure(graph: NasGraph):
    prohibited_node_types = ['average_pool2d', 'max_pool2d', 'conv2d', 'kan_conv2d']
    for node in graph.nodes:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'flatten':
            return True
        elif node_name in prohibited_node_types:
            raise ValueError(f'{ERROR_PREFIX} node {node} can not be after flatten layer.')


def model_has_no_conv_layers(graph: NasGraph):
    was_flatten = False
    was_conv = False
    for node in graph.nodes:
        node_name = 'conv2d' if 'conv' in node.content['name'] else node.content['name']
        if node_name == 'conv2d':
            was_conv = True
        elif node_name == 'flatten':
            was_flatten = True
    if not was_conv and was_flatten:
        raise ValueError(f'{ERROR_PREFIX} model has no convolutional layers.')


def model_has_dim_mismatch(input_shape: list, output_shape: Union[int, List[int]]):
    def validate(graph: NasGraph):
        try:
            with torch.no_grad():
                m = NeuralSearchModel(NASTorchModel).compile_model(graph, input_shape, output_shape).model
                device = 'cuda' if torch.cuda.is_available() else "cpu"
                m.to(device)
                try:
                    batch_size = 1
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        res = m.forward(
                            torch.rand([batch_size, *input_shape[::-1]], device=m.parameters().__next__().device))
                    if isinstance(output_shape, int) and res.shape[1] != output_shape or (
                            not isinstance(output_shape, int) and list(res.shape)[1:] != output_shape[::-1]):
                        print("Output shape mismatch!")
                        # print(res.shape, output_shape, isinstance(output_shape, int))
                        raise ValueError(f'{ERROR_PREFIX} graph has incorrect output shape.')
                except IndexError:
                    traceback.print_exc()
                    raise ValueError(f'{ERROR_PREFIX} graph has dimension conflict.')
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f'{ERROR_PREFIX} graph has dimension conflict.')
        return True

    return validate


def skip_has_no_pools(graph: NasGraph):
    for n in graph.nodes:
        cond = len(n.nodes_from) > 1 and 'pool' in n.name
        if cond:
            raise ValueError(f'{ERROR_PREFIX} pooling in skip connection may lead to dimension conflict.')
    return True


def only_conv_layers(graph: NasGraph):
    for n in graph.nodes:
        if 'conv' not in n.content['name']:
            raise ValueError(f'{ERROR_PREFIX} only conv layers are allowed.')
    return True


def filter_size_changes_monotonically(increases: bool = True):
    def rule(graph: NasGraph):
        """
        Performs DFS starting from root (sink) node back to the source node while
        maintaining last filter size of the conv layers on the path
        thereby ensuring that filter size increases (non-strictly) monotonically.
        """

        def dfs(node: LinkedGraphNode, last_filter_size: int = None):
            if 'conv' in node.content['name']:
                if last_filter_size is not None and (
                        node.content['params']['out_shape'] > last_filter_size if increases else
                        node.content['params']['out_shape'] < last_filter_size):
                    # print("Filter size must increase monotonically.")
                    raise ValueError(f'{ERROR_PREFIX} filter size must change monotonically.')
                last_filter_size = node.content['params']['out_shape']
            for n in node.nodes_from:
                dfs(n, last_filter_size)

        dfs(graph.root_node)
        return True

    return rule


def output_node_has_channels(output_channels):
    def rule(graph: NasGraph):
        output_node = graph.root_node
        if output_node.content['params']['out_shape'] != output_channels:
            raise ValueError(f'{ERROR_PREFIX} output node must have {output_channels} channels.')

        return True

    return rule


def no_transposed_layers_before_conv(graph: NasGraph):
    """
    Performs DFS starting from root (sink) node and starts checking after conv layers
    thereby checking if there are transposed layers in the conv segment.
    """

    def dfs(node: LinkedGraphNode, encountered_conv: bool = False):
        if not node.parameters['is_transposed']:
            encountered_conv = True
        if encountered_conv and node.parameters['is_transposed']:
            print("Conv after transposed")
            raise ValueError(f'{ERROR_PREFIX} conv layer must not be after transposed layer.')
        for n in node.nodes_from:
            dfs(n, encountered_conv)

    dfs(graph.root_node)
    return True


def right_output_size(graph: NasGraph):
    """
    Performs DFS starting from root (sink) node and checks that the size is right along all paths.
    """

    def dfs(node: LinkedGraphNode, size_delta_until_finish: int = 0):
        size_delta_until_finish += (node.parameters["kernel_size"] - 1) * (
            1 if node.parameters['is_transposed'] else -1)
        if not node.nodes_from and size_delta_until_finish != 0:
            print("Bad size delta")
            raise ValueError(f'{ERROR_PREFIX} Side size delta must be 0 along the network')
        for n in node.nodes_from:
            dfs(n, size_delta_until_finish)

    dfs(graph.root_node)
    return True


def no_linear_layers_before_flatten(graph: NasGraph):
    """
    Performs DFS starting from root (sink) node and starts checking after Flatten layers
    thereby checking if there are linear layers in the conv (with 3 or 4 dims) segment.
    """

    def dfs(node: LinkedGraphNode, encountered_flatten: bool = False):
        if node.content['name'] == 'flatten':
            encountered_flatten = True
        if encountered_flatten and node.content['name'] == 'linear':
            print("Linear after flatten")
            raise ValueError(f'{ERROR_PREFIX} linear layer must not be after flatten layer.')
        for n in node.nodes_from:
            dfs(n, encountered_flatten)

    dfs(graph.root_node)
    return True


def has_too_much_parameters(max_allowed_parameters: int, parameter_count_complexity_metric: Callable[[NasGraph], int]):
    def rule(graph):
        if parameter_count_complexity_metric(graph) > max_allowed_parameters:
            raise ValueError(f'{ERROR_PREFIX} model has too much parameters.')
        return True

    return rule


def has_too_much_flops(max_allowed_flops: int, flops_complexity_metric: Callable[[NasGraph], int]):
    def rule(graph: NasGraph):
        if flops_complexity_metric(graph) > max_allowed_flops:
            raise ValueError(f'{ERROR_PREFIX} model has too much FLOPs.')
        return True

    return rule


def has_too_much_time(max_allowed_time: float, time_complexity_metric: Callable[[NasGraph], float]):
    def rule(graph: NasGraph):
        if time_complexity_metric(graph) > max_allowed_time:
            raise ValueError(f'{ERROR_PREFIX} model has too much time complexity.')
        return True

    return rule
