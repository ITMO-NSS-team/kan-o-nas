import datetime
import json
import os
import pathlib
import sys
import time
import traceback

import torch.utils.data
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from typing import NamedTuple, Type, Optional, Union

from cases.load_imagenet.focal_loss import FocalLoss
from nas.graph.node.nas_graph_node import NasNode

project_root_path = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root_path))
sys.path.append(str(project_root_path / "cases"))

import numpy as np
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer
from golem.core.adapter.adapter import DirectAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.advisor import DefaultChangeAdvisor
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from nas.utils.random_split_hack import random_split
from torchvision.datasets import FashionMNIST, MNIST
from cases.cached_datasets import EuroSAT, CachedFashionMNIST, CachedMNIST
from torchvision.transforms import transforms

import nas.composer.requirements as nas_requirements
from nas.composer.nn_composer import NNComposer
from nas.graph.builder.base_graph_builder import BaseGraphBuilder, GraphGenerator
from nas.graph.builder.cnn_builder import ConvGraphMaker
from nas.graph.node.node_factory import NNNodeFactory
from nas.model.constructor import ModelConstructor
from nas.model.pytorch.base_model import compute_total_graph_parameters, get_flops_from_graph, get_time_from_graph
from nas.operations.validation_rules.cnn_val_rules import *
from nas.optimizer.objective.nas_cnn_optimiser import NNGraphOptimiser
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import set_root, project_root
from nas.composer.requirements import _get_image_channels_num

from experiment_utils import *

set_root(project_root())


class FixedGraphGenerator(GraphGenerator):

    def __init__(self, graph: NasGraph):
        self.graph = graph

    def _add_node(self, *args, **kwargs):
        raise NotImplementedError()

    def build(self, *args, **kwargs) -> NasGraph:
        return self.graph


def generate_basic_kkan(starting_value, factor) -> NasGraph:
    conv_layer_type = LayersPoolEnum.kan_conv2d
    node_types = [
        conv_layer_type,
        LayersPoolEnum.pooling2d,

        conv_layer_type,
        LayersPoolEnum.pooling2d,

        # LayersPoolEnum.adaptive_pool2d,
        LayersPoolEnum.flatten
    ]

    graph = NasGraph()
    parent_node = None
    shape = starting_value
    for node_type in node_types:
        if node_type == conv_layer_type:
            shape *= factor

        param_variants = {
            'kan_conv2d': {
                'out_shape': shape,
                'kernel_size': 3,
                'activation': 'tanh',
                'stride': 1,
                'padding': 1,
                'grid_size': 5,
                'spline_order': 3,
                'output_node_grid_size': 10,
                'output_node_spline_order': 3
            },
            'pooling2d': {
                'pool_size': 2,
                'pool_stride': 2,
                'mode': "max"
            },
            'adaptive_pool2d': {'mode': 'max', 'out_shape': 1},
            'flatten': {}
        }

        node = NasNode(
            content={'name': node_type.value, 'params': param_variants[node_type.value]},
            nodes_from=[parent_node] if parent_node is not None else None
        )

        graph.add_node(node)
        parent_node = node

    return graph


def build_mnist_cls(save_path, dataset_cls, conv_is_kan=False, linear_is_kan=False, repetitions_for_final_choices=1,
                    history_path_instead_of_evolution=None):
    colored_datasets = [EuroSAT, CachedCIFAR10, "ImageNet1k"]

    visualize = False
    cv_folds = None
    num_classes = 1000 if dataset_cls == "ImageNet1k" else 10
    image_side_size = {EuroSAT: 64, CachedCIFAR10: 32, CachedMNIST: 28, CachedFashionMNIST: 28, "ImageNet1k": 224}[
        dataset_cls]
    batch_size = {(False, False): 128, (True, False): 128, (True, True): 128}[linear_is_kan, conv_is_kan]
    epochs = 100 if dataset_cls in colored_datasets else 20
    optimization_epochs = 30 if dataset_cls in colored_datasets else 10
    num_of_generations = 25
    initial_population_size = 8
    max_population_size = 8
    color_mode = 'color' if dataset_cls in colored_datasets else 'grayscale'
    dataloader_num_workers = 27 if dataset_cls == "ImageNet1k" else 0

    set_root(project_root())
    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.accuracy)

    input_channels = _get_image_channels_num(color_mode)

    dataset_train, dataset_test = RunConfiguration.construct_dataset(dataset_cls)

    if linear_is_kan:
        fc_layers_pool = [LayersPoolEnum.kan_linear, ]
        min_fc_layers = 1
        max_fc_layers = 2 if dataset_cls in colored_datasets else 1
    else:
        fc_layers_pool = [LayersPoolEnum.linear, ]
        min_fc_layers = 2
        max_fc_layers = 3

    if conv_is_kan:
        conv_layers_pool = [LayersPoolEnum.kan_conv2d, ]
        min_conv_layers = 2
        max_conv_layers = 7 if dataset_cls in colored_datasets else 3
    else:
        conv_layers_pool = [LayersPoolEnum.conv2d, ]

        min_conv_layers = 4
        max_conv_layers = 10 if dataset_cls in colored_datasets else 6

    mutations = [MutationTypesEnum.single_add, MutationTypesEnum.single_drop, MutationTypesEnum.single_edge,
                 MutationTypesEnum.single_change]

    fc_requirements = nas_requirements.BaseLayerRequirements(min_number_of_neurons=64,
                                                             max_number_of_neurons=1024)
    conv_requirements = nas_requirements.ConvRequirements(
        min_number_of_neurons=32, max_number_of_neurons=256,
        conv_strides=[1],
        pool_size=[2], pool_strides=[2],
        supplementary_pooling_prob=0.35
    )

    kan_linear_requirements = nas_requirements.KANLinearRequirements(min_number_of_neurons=32,
                                                                     max_number_of_neurons=512)
    kan_conv_requirements = nas_requirements.KANConvRequirements(
        min_number_of_neurons=16, max_number_of_neurons=256,
        pooling_prob=0.35
    )

    model_requirements = nas_requirements.ModelRequirements(
        input_shape=[image_side_size, image_side_size, input_channels],
        output_shape=num_classes,
        color_mode=color_mode,
        num_of_classes=num_classes,
        conv_requirements=conv_requirements,
        fc_requirements=fc_requirements,
        primary=conv_layers_pool,
        kan_conv_requirements=kan_conv_requirements,
        kan_linear_requirements=kan_linear_requirements,
        secondary=fc_layers_pool,
        epochs=epochs,
        batch_size=batch_size,
        min_nn_depth=min_fc_layers,
        # Fc layers including last, output layer
        max_nn_depth=max_fc_layers,
        min_num_of_conv_layers=min_conv_layers,
        max_num_of_conv_layers=max_conv_layers)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=model_requirements,
                                                           timeout=datetime.timedelta(hours=9.),
                                                           num_of_generations=num_of_generations,
                                                           early_stopping_iterations=None,
                                                           early_stopping_timeout=10000000000000000000000000000000000.,
                                                           # TODO: fix datatype bug in GOLEM
                                                           parallelization_mode='sequential',
                                                           n_jobs=1,
                                                           cv_folds=cv_folds,
                                                           min_arity=1,  # Number of parents which data flow comes from
                                                           max_arity=2  # For the shortcut case
                                                           )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    focal_loss = FocalLoss()

    def focal_loss_by_ohe_targets(logits, targets):
        target_indices = targets.argmax(dim=-1)
        return focal_loss(logits, target_indices)

    model_trainer = ModelConstructor(model_class=NASTorchModel, trainer=NeuralSearchModel,
                                     device=device,
                                     loss_function=focal_loss_by_ohe_targets, optimizer=AdamW,
                                     metrics=[lambda *args: -accuracy_score(*args)], preprocess_graph_for_imagenet=True)

    basic_graph_time = get_time_from_graph(
        generate_basic_kkan(input_channels, 4 if dataset_cls in colored_datasets else 5),
        [image_side_size, image_side_size, input_channels],
        num_classes, device, batch_size)
    print("Basic graph time: ", basic_graph_time)

    def parameter_count_complexity_metric(graph: NasGraph):
        return compute_total_graph_parameters(graph, [image_side_size, image_side_size, input_channels], num_classes)

    def flops_complexity_metric(graph: NasGraph):
        return get_flops_from_graph(graph, [image_side_size, image_side_size, input_channels], num_classes)

    def time_complexity_metric(graph: NasGraph):
        return get_time_from_graph(graph, [image_side_size, image_side_size, input_channels], num_classes, device,
                                   batch_size)

    validation_rules = [
        model_has_several_starts, model_has_no_conv_layers, model_has_wrong_number_of_flatten_layers,
        model_has_several_roots,
        has_no_cycle, has_no_self_cycled_nodes, skip_has_no_pools,

        filter_size_changes_monotonically(increases=True),
        no_linear_layers_before_flatten,

        model_has_dim_mismatch([image_side_size, image_side_size, input_channels], num_classes),

        has_too_much_parameters(1_000_000_000, parameter_count_complexity_metric),
        # has_too_much_flops(3_000_000, flops_complexity_metric)
        # has_too_much_time(basic_graph_time * 2.5, time_complexity_metric)
    ]

    optimizer_parameters = GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                 mutation_types=mutations,
                                                 crossover_types=[CrossoverTypesEnum.subtree],
                                                 pop_size=initial_population_size,
                                                 max_pop_size=max_population_size,
                                                 regularization_type=RegularizationTypesEnum.none,
                                                 multi_objective=True)

    adapter = DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode)
    graph_generation_parameters = GraphGenerationParams(
        adapter=adapter,
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    # builder = ResNetBuilder(model_requirements=requirements.model_requirements, model_type='resnet_18')

    builder = ConvGraphMaker(requirements=requirements.model_requirements, rules=validation_rules,
                             max_generation_attempts=5_000)

    # builder = FixedGraphGenerator(graph=generate_kkan_from_paper())

    if history_path_instead_of_evolution is None:
        graph_generation_function = BaseGraphBuilder()
        graph_generation_function.set_builder(builder)
        initial_pipelines = graph_generation_function.build(initial_population_size)

        builder = ComposerBuilder(task).with_composer(NNComposer).with_optimizer(NNGraphOptimiser). \
            with_requirements(requirements).with_metrics(
            [objective_function, parameter_count_complexity_metric]).with_optimizer_params(optimizer_parameters). \
            with_initial_pipelines(initial_pipelines). \
            with_graph_generation_param(graph_generation_parameters)

        composer = builder.build()
        composer.set_trainer(model_trainer)
        composer.set_dataloader_num_workers(dataloader_num_workers)
        # composer.set_dataset_builder(dataset_builder)

        # The actual composition #######
        _optimizer_result = composer.compose_pipeline(*random_split(dataset_train, [.8, .2]))
        history = composer.history
        if save_path:
            composer.save(path=save_path)
    else:
        history = OptHistory.load(history_path_instead_of_evolution)
        print(f"Loaded from history {history_path_instead_of_evolution}: {history}")
        pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
    final_choices = history.final_choices

    if visualize:
        history_visualizer = PipelineHistoryVisualizer(history)
        history_visualizer.fitness_line()
        # history_visualizer.fitness_box(best_fraction=0.5)
        # history_visualizer.operations_kde()
        # history_visualizer.operations_animated_bar(save_path='example_animation.gif', show_fitness=True)
        history_visualizer.fitness_line_interactive()

    # Train the final choices #######

    # train_data, val_data = train_test_data_setup(train_data, split_ratio=.7, shuffle_flag=False)
    # final_dataset_train, final_dataset_val = random_split(dataset_train, [.9, .1])
    final_dataset_train = dataset_train

    # small_test = random_split(dataset_test, [.01, .99])[0]
    # print("Small test size:", len(small_test))

    final_train_dataloader = DataLoader(final_dataset_train, batch_size=requirements.model_requirements.batch_size,
                                        num_workers=dataloader_num_workers,
                                        shuffle=True)
    # final_val_dataloader = DataLoader(final_dataset_val, batch_size=requirements.model_requirements.batch_size,
    #                                   num_workers=dataloader_num_workers,
    #                                   shuffle=True)
    final_test_dataloader = DataLoader(dataset_test, batch_size=requirements.model_requirements.batch_size,
                                       num_workers=dataloader_num_workers,
                                       shuffle=False)

    # small_test_dataloader = DataLoader(small_test, batch_size=requirements.model_requirements.batch_size, num_workers=dataloader_num_workers
    #                                    shuffle=True)

    final_results = {}
    for final_choice_i, final_choice in enumerate(final_choices):
        if final_choice_i != 1:
            continue
        optimized_network = adapter.restore(final_choice.graph)
        for repetition in range(repetitions_for_final_choices):
            trainer = model_trainer.build([image_side_size, image_side_size, input_channels], num_classes,
                                          optimized_network)
            print(type(trainer.model.output_layer))
            trainer.fit_model(final_train_dataloader, final_test_dataloader, epochs, timeout_seconds=60 * 60 * 24 * 1.5)
            predictions, targets = trainer.predict(final_test_dataloader)

            # trainer.fit_model(small_test_dataloader, small_test_dataloader, epochs, timeout_seconds=60 * 60 * 8)
            # predictions, targets = trainer.predict(small_test_dataloader)
            # print(predictions, targets)

            all_labels = list(range(num_classes))
            loss = log_loss(targets, predictions, labels=all_labels)
            roc = roc_auc_score(targets, predictions, multi_class='ovo', labels=all_labels)
            top_5_accuracy = top_k_accuracy_score(targets, predictions, k=5, labels=all_labels)

            major_predictions = np.argmax(predictions, axis=-1)
            f1 = f1_score(targets, major_predictions, average='weighted', labels=all_labels)
            accuracy = accuracy_score(targets, major_predictions)

            print(f"=== Trained final choice {final_choice_i + 1}/{len(final_choices)},"
                  f"repetition {repetition + 1}/{repetitions_for_final_choices}")

            print(f'Composed ROC AUC of {final_choice.uid} is {round(roc, 3)}')
            print(f'Composed LOG LOSS of {final_choice.uid} is {round(loss, 3)}')
            print(f'Composed F1 of {final_choice.uid} is {round(f1, 5)}')
            print(f'Composed accuracy of {final_choice.uid} is {round(accuracy, 5)}')
            print(f'Composed top-5 accuracy of {final_choice.uid} is {round(top_5_accuracy, 5)}')

            repetition_results = {
                'roc_auc': roc,
                'log_loss': loss,
                'f1': f1,
                'accuracy': accuracy,
                'top_5_accuracy': top_5_accuracy
            }
            if final_choice.uid in final_results:
                final_results[final_choice.uid].append(repetition_results)
            else:
                final_results[final_choice.uid] = [repetition_results]

            # Save json:
            with open(f'{save_path}/final_results.json', 'w') as f:
                json.dump(final_results, f, indent=4)


def posttrain_final_choices(history):
    """
    Posttrains the models, computes metrics and saves
    - jsons with metric values by key=individual.uid
    - pytorch state dicts
    """
    # TODO: move from build_cls
    pass


def run_with_configuration(cfg: RunConfiguration, dir_name: Union[str, os.PathLike]):
    print(f"Save path: {dir_name}")

    build_mnist_cls(dir_name, dataset_cls=cfg.dataset_cls, linear_is_kan=cfg.linear_is_kan, conv_is_kan=cfg.conv_is_kan,
                    repetitions_for_final_choices=cfg.repetitions_for_final_choices,
                    history_path_instead_of_evolution=cfg.history_path_instead_of_evolution
                    )


def supported_linear_conv_configurations():
    return [(False, False), (True, False), (True, True), ]


if __name__ == '__main__':
    input_run_prefix = "cifar-thorough"
    output_run_prefix = "imagenet-eval"
    basic_config = RunConfiguration(
        # dataset_cls="ImageNet1k",
        dataset_cls=CachedCIFAR10,
        linear_is_kan=True,
        conv_is_kan=True,
        repetitions_for_final_choices=1,
        history_path_instead_of_evolution=None
    )

    for (linear_is_kan, conv_is_kan) in [(False, False), ]:
        run_config = (basic_config
                      ._replace(linear_is_kan=linear_is_kan)
                      ._replace(conv_is_kan=conv_is_kan)
                      )
        # run_config = run_config._replace(
        #     history_path_instead_of_evolution=get_canonical_history_dir(run_config._replace(
        #         dataset_cls=CachedCIFAR10 if run_config.dataset_cls == "ImageNet1k" else run_config.dataset_cls),
        #         name_prefix=input_run_prefix,
        #         deduplicate=False) + "/history.json"
        # )

        print(f"Running with run configuration {run_config}")
        try:
            run_with_configuration(run_config,
                                   dir_name=get_canonical_history_dir(run_config, name_prefix=output_run_prefix,
                                                                      deduplicate=True)
                                   )
        except Exception:
            print(f"FAILED run with run configuration {run_config} with the following traceback:", file=sys.stderr)
            # sys.stdout.flush()
            traceback.print_exc()
            # sys.stderr.flush()
