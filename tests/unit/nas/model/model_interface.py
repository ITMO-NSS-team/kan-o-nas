from nas.composer.requirements import ModelRequirements
from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.builder.cnn_builder import ConvGraphMaker


def test_builder():
    is_correct = False
    graph_builder = BaseGraphBuilder().set_builder(ConvGraphMaker(ModelRequirements(input_data_shape=[64, 64, 3])))
    try:
        graph_builder.build(10)
    except ValueError:
        is_correct = True
    assert is_correct
