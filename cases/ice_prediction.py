import datetime
import json
import os
import pathlib
import sys
import time
import traceback

import torch.utils.data
from PIL import Image
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from typing import NamedTuple, Type, Optional, Union

from matplotlib import pyplot as plt
from torch import nn

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
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
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
from nas.repository.layer_types_enum import LayersPoolEnum, ActivationTypesIdsEnum
from nas.utils.utils import set_root, project_root
from nas.composer.requirements import _get_image_channels_num

from golem.core.adapter import register_native
from golem.core.optimisers.genetic.operators.base_mutations import single_add_mutation, single_drop_mutation
from golem.core.optimisers.graph import OptGraph

from golem.core.optimisers.optimization_parameters import GraphRequirements

from nas.data.ts_dataset import create_test_train_sea_dataset

from pytorch_msssim import ssim

from experiment_utils import *

set_root(project_root())


def combined_mutation(m1, m2):
    """ Remember: should be done in both orders """

    @register_native
    def combined_mutation(graph: OptGraph,
                          requirements: GraphRequirements,
                          graph_gen_params: GraphGenerationParams,
                          parameters: AlgorithmParameters,
                          ) -> OptGraph:
        graph = m1(graph, requirements, graph_gen_params, parameters)
        return m2(graph, requirements, graph_gen_params, parameters)

    return combined_mutation


def save_numpy_image_with_pil(array, filename, cmap='Blues', vmin=0, vmax=1):
    """
    Saves a NumPy array as an image with the specified colormap using PIL.

    Parameters:
    - array (np.ndarray): 2D NumPy array representing the image.
    - filename (str): Path where the image will be saved.
    - cmap (str): Matplotlib colormap name to apply. Default is 'Blues'.
    - vmin (float): Minimum value for normalization. Default is 0.
    - vmax (float): Maximum value for normalization. Default is 1.
    """

    norm_array = (array - vmin) / (vmax - vmin)
    norm_array = np.clip(norm_array, 0, 1)

    cmap = plt.get_cmap(cmap)
    colored_array = cmap(norm_array)[:, :, :3]  # Exclude alpha channel

    colored_array = (colored_array * 255).astype(np.uint8)

    img = Image.fromarray(colored_array)
    img.save(filename)
    print(f"Image saved to {filename}")


def build_ice_forcaster(save_path, sea_name, conv_is_kan=False, repetitions_for_final_choices=1,
                        history_path_instead_of_evolution=None):
    visualize = False
    batch_size = 8
    epochs = 45
    optimization_epochs = 35
    num_of_generations = 4
    initial_population_size = 5
    max_population_size = 5

    filter_scale_steps = 10

    set_root(project_root())

    task = Task(TaskTypesEnum.classification)
    objective_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.accuracy)

    # input_channels = None

    ds_path = rf"C:\dev\aim\datasets\sea\{sea_name}"
    # ds_path = rf"./datasets/{sea_name}"
    # first_quartal_year = 2020
    first_quartal_year = None
    dataset_train, dataset_test = create_test_train_sea_dataset(ds_path, first_quartal_test_year=first_quartal_year)
    print(dataset_train.tensors[0].shape)
    _, input_channels, image_h, image_w = dataset_train.tensors[0].shape
    _, output_channels, _, _ = dataset_train.tensors[1].shape
    input_shape = [image_w, image_h, input_channels]
    output_shape = [image_w, image_h, output_channels]

    if conv_is_kan:
        conv_layers_pool = [LayersPoolEnum.kan_conv2d, ]
        min_conv_layers = 2
        max_conv_layers = 6
    else:
        conv_layers_pool = [LayersPoolEnum.conv2d, ]

        min_conv_layers = 2
        max_conv_layers = 8

    mutations = [combined_mutation(single_add_mutation, single_drop_mutation),
                 combined_mutation(single_add_mutation, single_add_mutation),
                 combined_mutation(single_drop_mutation, single_add_mutation),
                 combined_mutation(single_drop_mutation, single_drop_mutation),
                 MutationTypesEnum.single_edge,
                 MutationTypesEnum.single_change]

    conv_requirements = nas_requirements.ConvRequirements(
        min_number_of_neurons=output_channels, max_number_of_neurons=input_channels,
        conv_strides=[1],
        pool_size=[], pool_strides=[],
        supplementary_pooling_prob=0.,
        kernel_size=[3],

        _batch_norm_prob=0.,
        _dropout_prob=0.,
        _max_dropout_val=0.,
        activation_types=[ActivationTypesIdsEnum.relu],

        filter_log_scale_n_steps=filter_scale_steps,
    )

    kan_conv_requirements = nas_requirements.KANConvRequirements(
        min_number_of_neurons=output_channels, max_number_of_neurons=input_channels,
        pooling_prob=0.,
        kernel_size=[3, 5],

        filter_log_scale_n_steps=filter_scale_steps,
    )

    model_requirements = nas_requirements.ModelRequirements(input_shape=input_shape,
                                                            output_shape=output_shape,
                                                            color_mode=None,
                                                            num_of_classes=None,
                                                            conv_requirements=conv_requirements,
                                                            fc_requirements=None,
                                                            primary=conv_layers_pool,
                                                            kan_conv_requirements=kan_conv_requirements,
                                                            kan_linear_requirements=None,
                                                            secondary=[],
                                                            epochs=epochs,
                                                            batch_size=batch_size,
                                                            min_nn_depth=-100000000001,
                                                            max_nn_depth=-10000000000,
                                                            min_num_of_conv_layers=min_conv_layers,
                                                            max_num_of_conv_layers=max_conv_layers)

    requirements = nas_requirements.NNComposerRequirements(opt_epochs=optimization_epochs,
                                                           model_requirements=model_requirements,
                                                           timeout=datetime.timedelta(hours=10.),
                                                           num_of_generations=num_of_generations,
                                                           early_stopping_iterations=None,
                                                           early_stopping_timeout=10000000000000000000000000000000000.,
                                                           # TODO: fix datatype bug in GOLEM
                                                           parallelization_mode='sequential',
                                                           n_jobs=1,
                                                           cv_folds=None,
                                                           min_arity=1,  # Number of parents which data flow comes from
                                                           max_arity=2  # For the shortcut case
                                                           )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_trainer = ModelConstructor(model_class=NASTorchModel, trainer=NeuralSearchModel,
                                     device=device,
                                     loss_function=torch.nn.L1Loss(), optimizer=AdamW,
                                     metrics=None)

    def parameter_count_complexity_metric(graph: NasGraph):
        return compute_total_graph_parameters(graph, input_shape, output_shape)

    validation_rules = [
        only_conv_layers,

        model_has_several_starts, model_has_no_conv_layers, model_has_several_roots,
        output_node_has_channels(output_channels),
        has_no_self_cycled_nodes,
        has_no_cycle,

        filter_size_changes_monotonically(increases=False),
        right_output_size,
        no_transposed_layers_before_conv,

        model_has_dim_mismatch(input_shape, output_shape),
    ]

    optimizer_parameters = GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                                                 mutation_types=mutations,
                                                 crossover_types=[CrossoverTypesEnum.subtree],
                                                 pop_size=initial_population_size,
                                                 max_pop_size=max_population_size,
                                                 regularization_type=RegularizationTypesEnum.none,
                                                 multi_objective=True)

    graph_generation_parameters = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode),
        rules_for_constraint=validation_rules, node_factory=NNNodeFactory(requirements.model_requirements,
                                                                          DefaultChangeAdvisor()))

    # builder = ResNetBuilder(model_requirements=requirements.model_requirements, model_type='resnet_18')

    builder = ConvGraphMaker(requirements=requirements.model_requirements, rules=validation_rules,
                             max_generation_attempts=5000)

    # builder = FixedGraphGenerator(graph=generate_kkan_from_paper())

    graph_generation_function = BaseGraphBuilder()
    graph_generation_function.set_builder(builder)

    builder = ComposerBuilder(task).with_composer(NNComposer).with_optimizer(NNGraphOptimiser). \
        with_requirements(requirements).with_metrics(
        [objective_function, parameter_count_complexity_metric]).with_optimizer_params(optimizer_parameters). \
        with_initial_pipelines(graph_generation_function.build(initial_population_size)). \
        with_graph_generation_param(graph_generation_parameters)

    composer = builder.build()
    composer.set_trainer(model_trainer)
    # composer.set_dataset_builder(dataset_builder)

    # The actual composition #######
    if history_path_instead_of_evolution is None:
        _optimizer_result = composer.compose_pipeline(dataset_train, dataset_test)
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
    # final_dataset_train, final_dataset_val = random_split(dataset_train, [.7, .3])
    final_dataset_train = dataset_train

    final_train_dataloader = DataLoader(final_dataset_train, batch_size=requirements.model_requirements.batch_size,
                                        shuffle=True)
    # final_val_dataloader = DataLoader(final_dataset_val, batch_size=requirements.model_requirements.batch_size,
    #                                   shuffle=True)
    final_test_dataloader = DataLoader(dataset_test[0], batch_size=requirements.model_requirements.batch_size,
                                       shuffle=False)

    final_results = {}
    for final_choice_i, final_choice in enumerate(final_choices):
        optimized_network = composer.optimizer.graph_generation_params.adapter.restore(final_choice.graph)
        for repetition in range(repetitions_for_final_choices):
            trainer = model_trainer.build(input_shape, output_shape,
                                          optimized_network)
            trainer.fit_model(final_train_dataloader, final_test_dataloader, epochs=epochs,
                              # Test dataset for monitoring loss ↑↑↑
                              timeout_seconds=60 * 100)
            repetition_results = {}
            for quartal_i, quartal in enumerate(dataset_test):
                final_test_dataloader = DataLoader(quartal, batch_size=requirements.model_requirements.batch_size,
                                                   shuffle=False)

                predictions, targets = trainer.predict(final_test_dataloader)
                if first_quartal_year is not None:
                    predictions = predictions[:13]
                    targets = targets[:13]

                # loss = log_loss(targets, predictions)
                # roc = roc_auc_score(targets, predictions, multi_class='ovo')
                # major_predictions = np.argmax(predictions, axis=-1)
                # f1 = f1_score(targets, major_predictions, average='weighted')
                # accuracy = accuracy_score(targets, major_predictions)

                # Uniform across batch
                outputs = torch.from_numpy(predictions)
                Y = torch.from_numpy(targets)
                print(outputs.shape)
                print(Y.shape)
                l1_loss = nn.L1Loss()(outputs, Y).cpu().detach().item()
                ssim_value = ssim(outputs, Y, data_range=1, size_average=True).item()

                if first_quartal_year is not None:
                    print("Quartal_i:", quartal_i)
                    # print(len(quartal))
                    year = first_quartal_year + quartal_i // 4
                    quartal_index = (quartal_i % 4) + 1
                    quartal_name = f"Q{quartal_index}"

                    full_quartal_name = f"{year} {quartal_name}"
                else:
                    full_quartal_name = "all-quartals"

                print(f"=== Trained final choice {final_choice_i + 1}/{len(final_choices)},"
                      f"repetition {repetition + 1}/{repetitions_for_final_choices}, quartal {full_quartal_name} ===")

                # print(f'Composed ROC AUC of {final_choice.uid} is {round(roc, 3)}')
                # print(f'Composed LOG LOSS of {final_choice.uid} is {round(loss, 3)}')
                # print(f'Composed F1 of {final_choice.uid} is {round(f1, 3)}')
                # print(f'Composed accuracy of {final_choice.uid} is {round(accuracy, 3)}')
                print(f'Composed L1 loss of {final_choice.uid} is {round(l1_loss, 3)}')
                print(f'Composed SSIM of {final_choice.uid} is {round(ssim_value, 3)}')

                # Build images:
                def build_image_from_tensor(t: torch.Tensor, name_to_save: str):
                    arr = np.array(t)
                    save_numpy_image_with_pil(arr, f"{save_path}/{name_to_save}.png")

                for (nearness_name, nearness_index) in [("near", 0), ("far", -1)]:
                    for (forecast_term_name, forecast_term_index) in [("first", 0), ("last", -1)]:
                        for (target_or_actual_name, target_or_actual_index) in [("target", 0), ("actual", 1)]:
                            build_image_from_tensor(
                                [Y, outputs][target_or_actual_index][nearness_index][forecast_term_index],
                                f"{full_quartal_name}_model_{final_choice_i}_{nearness_name}_future_{forecast_term_name}_{target_or_actual_name}")

                repetition_results[f"{full_quartal_name}_l1" if first_quartal_year else "l1"] = l1_loss
                repetition_results[f"{full_quartal_name}_ssim" if first_quartal_year else "ssim"] = ssim_value

            if final_choice.uid in final_results:
                final_results[final_choice.uid].append(repetition_results)
            else:
                final_results[final_choice.uid] = [repetition_results]

            # Save json:
            with open(f'{save_path}/final_results.json', 'w') as f:
                json.dump(final_results, f, indent=4)


if __name__ == '__main__':
    is_kan = True
    for sea_name in [
        "laptev",
    ]:
        path = f'./_results/_ice_{"dense-kan" if is_kan else "cnn"}_{sea_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        print(f"Save path: {path}")
        build_ice_forcaster(path, sea_name="laptev", conv_is_kan=is_kan,
                            # history_path_instead_of_evolution="./_results/_ice_cnn_laptev_2025-01-29_20-28-07/history.json",
                            history_path_instead_of_evolution="./_results/_ice_dense-kan_laptev_2025-01-30_11-40-14/history.json",
                            repetitions_for_final_choices=5
                            )
