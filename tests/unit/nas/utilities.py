from nas.composer.requirements import load_default_requirements
from nas.graph.base_graph import NasGraph
from nas.graph.builder.base_graph_builder import BaseGraphBuilder
from nas.graph.builder.cnn_builder import ConvGraphMaker
from nas.operations.validation_rules.cnn_val_rules import *
from golem.core.dag.verification_rules import *


def get_graph() -> NasGraph:
    requirements = load_default_requirements()
    builder = BaseGraphBuilder()
    cnn_builder = ConvGraphMaker(requirements=requirements.model_requirements, rules=[
        model_has_several_starts, model_has_no_conv_layers, model_has_wrong_number_of_flatten_layers,
        model_has_several_roots,
        has_no_cycle, has_no_self_cycled_nodes, skip_has_no_pools,

        filter_size_changes_monotonically(increases=True),
        no_linear_layers_before_flatten,

        model_has_dim_mismatch([*requirements.model_requirements.input_shape, requirements.model_requirements.channels_num], requirements.model_requirements.output_shape),
    ])
    builder.set_builder(cnn_builder)
    return builder.build(initial_population_size=1)[0]
