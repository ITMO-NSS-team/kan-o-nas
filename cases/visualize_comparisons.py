import json
import os
import random
from collections.abc import Sequence
from copy import deepcopy
from itertools import groupby
from pathlib import Path

import seaborn
import tabulate
from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt, ticker, rcParams
from typing import Tuple, List, Dict

from numpy import average
from seaborn import histplot
import seaborn as sns

from cases.experiment_utils import parse_run_name
from nas.graph.base_graph import NasGraph
from nas.graph.node.nas_graph_node import NasNode

from nas.model.pytorch.base_model import NASTorchModel
from nas.model.model_interface import NeuralSearchModel

dataset_results_dir = {
    "MNIST": {
        # Less iterations:
        # "less-iter-kan": r"C:\dev\aim\nas_kan_results\_results\smaller-kans-mnist",
        # "less-iter-cnn": r"C:\dev\aim\nas_kan_results\_results\better-cnn-mnist",

        # More iterations:
        # "kan": r"C:\dev\aim\nas_kan_results\_results\kan-much-epochs-mnist",
        # "cnn": r"C:\dev\aim\nas_kan_results\_results\cnn-much-epochs-mnist",
        # "cnn-kan": r"C:\dev\aim\nas_kan_results\_results\cnn-kan-much-epochs-mnist",
        # "cnn-kan-smaller": r"C:\dev\aim\nas_kan_results\_results\mnist-kan-cnn-smaller",
        # "dense-kan": r"C:\dev\aim\nas_kan_results\_results\dense-kan-mnist"

        # "cnn": r"C:\dev\aim\nas_kan_results\_results\ltr-cnn-mnist",
    },
    "FashionMNIST": {
        # Less iterations:
        # "less-iter-smaller-kan": r"C:\dev\aim\nas_kan_results\_results\smaller-kan-fashion",
        # "less-iter-kan": r"C:\dev\aim\nas_kan_results\_results\kan-fashion-mnist",
        # "less-iter-cnn": r"C:\dev\aim\nas_kan_results\_results\better-cnn-fashion",

        # More iterations:
        # "kan": r"C:\dev\aim\nas_kan_results\_results\kan-fashion-mnist-much-epochs",
        # "cnn": r"C:\dev\aim\nas_kan_results\_results\cnn-fashion-mnist-much-epochs",
        # "cnn-kan": r"C:\dev\aim\nas_kan_results\_results\kan-cnn-fashion-mnist-more-epochs",
        # "dense-kan": r"C:\dev\aim\nas_kan_results\_results\dense-kans-fashion",

        # LTR
        # "cnn": r"C:\dev\aim\nas_kan_results\_results\ltr-cnn-fashion-mnist",
    },
    "EuroSAT": {
        # "kan": r"C:\dev\aim\nas_kan_results\_results\eurosat-kan",
        "kan": [r"C:\dev\aim\nas_kan_results\_results\more-iters-kan-eurosat"],
        "cnn-kan": [r"C:\dev\aim\nas_kan_results\_results\cnn-kan-eurosat-many-epochs"],
        "cnn": [r"C:\dev\aim\nas_kan_results\_results\eurosat-cnn"],
    },
}


def dir_dict_with_grid(root_dir, modification, dataset_names: List[Tuple[str, str]], algorithm_names,
                       only_existing_dirs=False, special_prefix_for_configs=None):
    """
    :param special_prefix_for_configs: indexed by [(dataset, model)], is List[prefix | None]
    Empty list designates ignoring this pair. None in the list means the :param modification: prefix.

    :returns: d[dataset][model]: List[path]; Configurations with empty lists are not present in the dict
    """
    if special_prefix_for_configs is None:
        special_prefix_for_configs = {}

    res = {}
    for (ds_presentation_name, ds_path_name) in dataset_names:
        ds_res = {}
        for algo_name in algorithm_names:
            config_res = []
            special_prefixes_trial = special_prefix_for_configs.get((ds_path_name, algo_name))
            if special_prefixes_trial is not None:
                if special_prefixes_trial == []:
                    continue
                input_prefixes = special_prefixes_trial
            else:
                input_prefixes = [modification]

            for prefix in input_prefixes:
                if prefix is None:
                    prefix = modification
                result_dir = f"{root_dir}/{prefix}-{algo_name}-{ds_path_name}"
                if not only_existing_dirs or os.path.isdir(result_dir):
                    config_res.append(result_dir)

            ds_res[algo_name] = config_res
        res[ds_presentation_name] = ds_res
    return res


result_root_dir = "C:/dev/aim/nas_kan_results/_results"

"""
See :dir_dict_with_grid: for the format
"""
dataset_results_dir = dir_dict_with_grid(
    result_root_dir,
    "ltr",
    [
        ("CIFAR10", 'cifar10'),
        ("MNIST", "mnist"), ("FashionMNIST", "fashion-mnist"),
        ("EuroSAT", "eurosat"),
    ],
    [
        "cnn", "dense-kan", "cnn-kan",
        "kan", "fast-kan", "torch-kan"
    ],
    only_existing_dirs=False,
    special_prefix_for_configs={
        # ("mnist", "dense-kan"): ["ltr-re-eval"],
        ("fashion-mnist", "kan"): [None, "small"],
        ("fashion-mnist", "cnn"): [None, "small"],

        ("mnist", "cnn"): ["moderate"],
        ("fashion-mnist", "cnn"): [None, "moderate"],

        ("mnist", "cnn-kan"): ["ltr-re-eval"],
        # ("mnist", "cnn"): ["ltr-re-eval"],

        ("eurosat", "kan"): [
            "ltr-re-eval",
            "old-revision"
        ],
        ("eurosat", "cnn-kan"): [
            None,
            "old-revision"
        ],
        ("eurosat", "cnn"): [
            None,
            "old-revision"
        ],

        ("cifar10", "cnn"): [
            "cifar-bigger",
            "cifar-thorough",
            "cifar-giant",
        ],
        ("cifar10", "cnn-kan"): [
            "cifar-bigger",
            "cifar-bigger-kan-conv"
        ],
        ("cifar10", "kan"): [],
        ("cifar10", "dense-kan"): [
            # ARCHAIC:
            # "cifar-bigger",

            # USED:
            "cifar-bigger-removed-stats",
            "cifar-bigger-kan-conv",
        ],

        ("mnist", "fast-kan"): [],
        ("fashion-mnist", "fast-kan"): [],
        ("eurosat", "fast-kan"): [],

        ("mnist", "torch-kan"): [],
        ("fashion-mnist", "torch-kan"): [],
        ("eurosat", "torch-kan"): [],

        ("cifar10", "torch-kan"): [
            "torch-kan"
        ],
        ("cifar10", "fast-kan"): [
            # "cifar-thorough" # Unused
            "fast"
        ],
    }
)

graph_name_mapping = {
    "kan": "KAN",
    "dense-kan": "denseConvKAN",
    "cnn": "CNN",
    "cnn-kan": "CNN+KAN",
    "fast-kan": "FastKAN",
    "torch-kan": "TorchKAN"
}

dataset_results_dir = {"laptev": {"cnn": [
    r"C:\dev\aim\nas_kan_results\_results\_ice_cnn_laptev_2025-01-29_20-28-07",
    # r"C:\dev\aim\nas_kan_results\_results\_ice_cnn_laptev_2025-01-29_19-49-49"
], "dense-kan": [r"C:\dev\aim\nas_kan_results\_results\_ice_dense-kan_laptev_2025-01-30_11-40-14"],
    "fast-kan": [
        # r"C:\dev\aim\nas_kan_results\_results\_ice_dense-kan_laptev_2025-03-11_20-26-17",
        r"C:\dev\aim\nas_kan_results\_results\_ice_dense-kan_laptev_2025-03-12_15-05-27"
    ],
    # "torch-kan": [r"C:\dev\aim\nas_kan_results\_results\_ice_dense-kan_laptev_2025-03-11_23-48-45"]
}}
ignore_original_evals = {
    ("laptev", "denseConvKAN")
}


def transplant_path_dict(d, root_dir):
    res = {}
    for k, v in d.items():
        # res[os.path.join(root_dir, k)] = os.path.join(root_dir, v)
        res[k] = [os.path.join(root_dir, p) for p in v]
    return res


""" Doesn't exhibit new runs, new models. Is only treated as a container for additional stat evaluations """
extra_evals_folders = transplant_path_dict({
    "ltr-cnn-fashion-mnist": [
        "ci-ltr-cnn-fashion-mnist", "ci2-ltr-cnn-fashion-mnist"
    ],
    "ltr-dense-kan-fashion-mnist": [
        "ci-ltr-dense-kan-fashion-mnist", "ci2-ltr-dense-kan-fashion-mnist"
    ],
    "ltr-cnn-kan-fashion-mnist": ["ci-cnn-kan-fashion-mnist"],
    "ltr-kan-fashion-mnist": ["ci-kan-fashion-mnist"],

    "ltr-cnn-mnist": ["ci-ltr-cnn-mnist"],
    "ltr-cnn-kan-mnist": ["ci-ltr-cnn-kan-mnist"],
    "ltr-dense-kan-mnist": ["ltr-re-eval-dense-kan-mnist"],
    "ltr-kan-mnist": ["ci-ltr-kan-mnist"],

    "old-revision-kan-eurosat": ["old-stats-extra-kan-eurosat", "old-stats-kan-eurosat"],
    "old-revision-cnn-kan-eurosat": ["old-stats-cnn-kan-eurosat"],
    "old-revision-cnn-eurosat": ["old-stats-cnn-eurosat"],
    "cifar-bigger-removed-stats-dense-kan-cifar10": [
        "eval-dense-kan-cifar10_2025-03-01_16-10-20",  # Only 1st
        "eval-dense-kan-cifar10_2025-03-01_18-40-41"  # Others than 1 and 2
    ],
    "cifar-bigger-kan-conv-dense-kan-cifar10": [
        "eval-dense-kan-cifar10_2025-03-01_17-12-02",  # First one
        "eval-dense-kan-cifar10_2025-03-01_19-08-30"  # Remaining
    ],

    "_ice_cnn_laptev_2025-01-29_20-28-07": [
        "_ice_cnn_laptev_2025-01-30_21-32-17",
    ],
    "_ice_dense-kan_laptev_2025-01-30_11-40-14": [
        # "_ice_dense-kan_laptev_2025-01-30_22-12-40"
        "_ice_dense-kan_laptev_2025-03-18_21-29-51",
        "_ice_dense-kan_laptev_2025-03-19_12-12-35",
    ]
}, result_root_dir)


def deduplicate_by_lambda(items, key_func):
    seen = set()
    result = []
    for item in items:
        item_key = key_func(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


def get_individual_dump(history):
    res = []
    for gen in history.individuals:
        for ind in gen:
            res.append(ind)

    return res


def individuals_pool(history):
    return deduplicate_by_lambda(get_individual_dump(history), lambda x: x.uid)


def plot_opt_fitness_scatter(history_path):
    history = OptHistory.load(history_path)
    individuals = individuals_pool(history)

    plt.scatter([ind.fitness.values[0] for ind in individuals], [ind.fitness.values[1] for ind in individuals])


color_cycle_index = 0


def retrieve_next_color():
    global color_cycle_index
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    res = color_cycle[color_cycle_index]
    color_cycle_index += 1
    return res


def reset_next_color_index():
    global color_cycle_index
    color_cycle_index = 0


def visualise_pareto_front(front: Sequence[Individual | Tuple[float, float]],
                           objectives_numbers: Tuple[int, int] = (0, 1),
                           objectives_names: Sequence[str] = ('ROC-AUC', 'Complexity'),
                           file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                           folder: str = f'../../tmp/pareto',
                           case_name: str = None,
                           minmax_x: List[float] = None,
                           minmax_y: List[float] = None,
                           color: str = 'red',
                           label: str = None,
                           try_building_statintervals: bool = True,
                           plot_line_between_points: bool = False,
                           only_best_run: bool = False
                           ):
    front.sort(key=lambda t: t[1])
    pareto_obj_first, pareto_obj_second = [], []
    for ind in front:
        fitness_list = ind.fitness.values if isinstance(ind, Individual) else ind
        fit_first = fitness_list[objectives_numbers[0]]
        pareto_obj_first.append(abs(fit_first))
        fit_second = fitness_list[objectives_numbers[1]]
        pareto_obj_second.append(abs(fit_second))

    # Check if there are a lot of ys for any x
    n_has_duplicates = 0
    total_xs = 0
    tmp_pareto_obj_first = []
    tmp_pareto_obj_second = []
    real_x = -1e10
    for x, xys in groupby(list(zip(pareto_obj_first, pareto_obj_second)), lambda xy: xy[0]):
        # if x >= 800_000:  # Excessively large models hinder the study
        #     continue
        total_xs += 1
        if x - real_x < pareto_obj_first[-1] / 35:
            x = real_x
        else:
            real_x = x
        xys = list(xys)
        if len(xys) > 1:
            n_has_duplicates += 1

        if only_best_run:
            tmp_pareto_obj_first.append(x)
            tmp_pareto_obj_second.append(max([y for _, y in xys]))
        else:
            tmp_pareto_obj_first.extend([x] * len(xys))
            tmp_pareto_obj_second.extend([y for _, y in xys])
    pareto_obj_second = tmp_pareto_obj_second
    pareto_obj_first = tmp_pareto_obj_first

    print(f"{n_has_duplicates}/{total_xs} samples have duplicates")
    has_duplicates = n_has_duplicates / total_xs >= 0.5

    if color is None:
        color = retrieve_next_color()

    # if not try_building_statintervals or not has_duplicates:
    # Didn't want to or failed to build stat-intervals
    # deduplicate_params = deduplicate_by_lambda(pareto_obj_first, lambda p: p)
    # dists_adj = [deduplicate_params[i + 1] - deduplicate_params[i] for i in range(len(deduplicate_params) - 1)]
    # target_width_anchor = min([d for d in dists_adj if d > deduplicate_params[-1] / 10])
    # anchored_width = min(dists_adj)

    if has_duplicates and try_building_statintervals:
        sns.boxplot(x=pareto_obj_first, y=pareto_obj_second,
                    color=color,
                    native_scale=True,
                    # width=0.18 * target_width_anchor/anchored_width
                    fliersize=0
                    )
    # plt.scatter([], [], color=color, label=label, alpha=1)

    seaborn.lineplot(x=pareto_obj_first, y=pareto_obj_second, label=label, color=color)
    # label=None if plot_line_between_points else label,
    # plt.show()

    # plt.xscale('log')
    if plot_line_between_points:
        plt.plot(pareto_obj_first, pareto_obj_second, color=color, label=label)
    # else:
    #     seaborn.lineplot(x=pareto_obj_first, y=pareto_obj_second, label=label, color=color)

    # plt.title(f'Pareto frontier for {case_name}', fontsize=15)
    plt.tight_layout()

    plt.xlabel(objectives_names[0], fontsize=15)
    plt.ylabel(objectives_names[1], fontsize=15)
    # plt.xscale('log')

    if minmax_x is not None:
        plt.xlim(*minmax_x)
    if minmax_y is not None:
        plt.ylim(*minmax_y)
    # fig.set_figwidth(8)
    # fig.set_figheight(8)
    if save:
        if not os.path.isdir('../tmp'):
            os.mkdir('../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')

        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight', dpi=600)
    if show:
        plt.show()
        reset_next_color_index()

    # plt.cla()
    # plt.clf()
    # plt.close('all')


def plot_opt_pareto_front(history, case_name: str, label: str):
    individuals = individuals_pool(history)

    # fig, ax = plt.subplots()

    visualise_pareto_front(
        history.final_choices.data, show=False, objectives_names=("Parameters", "LogLoss",), label=label, color=None,
        case_name=case_name,
        objectives_numbers=(1, 0)
    )


def get_extra_final_results():
    # this_results_folder_name = os.path.basename(this_results_dir.rstrip("\\").rstrip("/"))
    # extra_final_result_files = extra_runs_folders.get(this_results_folder_name, [])
    res = {}
    for run_name, this_run_extra_evals_folders in extra_evals_folders.items():
        for f in this_run_extra_evals_folders:
            j = json.load(open(f + "/final_results.json"))
            for uid, graph_res in j.items():
                if uid not in res:
                    res[uid] = []
                if isinstance(graph_res, List):
                    res[uid].extend(graph_res)
                else:
                    res[uid].append(graph_res)
    return res


def plot_final_pareto_front(histories: List[OptHistory], final_results, case_name: str, label: str,
                            final_metric_name: str,
                            extra_results: Dict[str, List], remove_dominated: bool,
                            try_building_statintervals: bool,
                            plot_line_between_points: bool,
                            only_best_run: bool
                            ):
    """
    Called ones per trajectory to plot.
    The trajectory could be a scatter+line or stat-intervals if enough models have enough evaluation attempts.

    The trajectory could be obtained either from a single history or from the whole collection.
    The latter case is general, so the former one is considered as a partial case with list length = 1.
    """
    # individuals = individuals_pool(history)
    final_choices: List = sum([history.final_choices.data for history in histories], [])
    final_choices.sort(key=lambda ind: ind.fitness.values[1])

    front = []
    model_averages = []
    for i, ind in enumerate(final_choices):
        # print(ind.fitness, ind.uid)
        # print(final_results[ind.uid])
        # assert final_metric_name == "accuracy"

        #  ### Compute complexity metric:
        complexity_metric = ind.fitness.values[1]
        # side_size, colors, classes = get_dataset_dims(case_name)
        # try:
        #     m = convert_ind_to_model_for_dataset(ind, case_name)
        # except:
        #     continue
        # complexity_metric = count_flops_number(m, side_size, colors)
        print(complexity_metric)

        # final_metric = -ind.fitness.values[0]  # For in-optimizer score

        this_extra_final_results = extra_results.get(ind.uid, [])
        this_results = final_results.get(ind.uid, []) if (case_name, label) not in ignore_original_evals else []
        if not isinstance(this_results, List):
            this_results = [this_results]
        this_results.extend(this_extra_final_results)
        if not this_results:
            print("NO STATS", ind, ind.fitness.values)
            # continue
            this_results.append({final_metric_name: abs(ind.fitness.values[0])})
        else:
            print("With stats", ind, ind.fitness.values)

        print(len(this_results))
        # max_measurements = 10
        # if len(this_results) > max_measurements:
        #     this_results = this_results[:max_measurements]
        metric_sum = 0
        total_having_metric = 0
        for this_run in this_results:
            if final_metric_name in this_run:
                final_metric = this_run[final_metric_name]  # For after-learning
                total_having_metric += 1

                # if final_metric < 0.6:
                #     continue

                metric_sum += final_metric
        this_model_average = metric_sum / total_having_metric

        # Remove dominated by any of the previous models of the front
        if remove_dominated and len(model_averages) >= 1:
            if this_model_average > model_averages[-1]:
                continue
            # elif complexity_metric / front[-1][1] - 1 < 0.1:
            #     front = [(metric, size) for metric, size in front if size != front[-1][1]]
        model_averages.append(this_model_average)

        for this_run in this_results:
            if final_metric_name not in this_run:
                continue
            final_metric = this_run[final_metric_name]
            # if final_metric < 0.6:
            #     continue

            front.append(
                (
                    final_metric,
                    complexity_metric
                )
            )

    # fig, ax = plt.subplots()

    print(front)
    visualise_pareto_front(
        front, show=False, objectives_names=("Parameters", final_metric_name), label=label, color=None,
        case_name=case_name,
        objectives_numbers=(1, 0),
        file_name=f"{case_name}-pareto.png",
        try_building_statintervals=try_building_statintervals,
        plot_line_between_points=plot_line_between_points,
        only_best_run=only_best_run
    )


def plot_parameter_number_hist(histories: List, dataset_name, arch, relative=True):
    param_values = []
    fc_share = []
    param_values_for_shares = []
    for history in histories:
        for ind in individuals_pool(history):
            param_values.append(ind.fitness.values[1])
            try:
                m = convert_ind_to_model_for_dataset(ind, dataset_name)
                fc_share.append(count_fc_percantage(m))
                param_values_for_shares.append(count_parameters(m))
            except:
                print("COUNDN'T COUNT FC PERCENTAGE: INVALID MODEL")
                import traceback
                traceback.print_exc()

    print(fc_share)

    conv_share = [1 - s for s in fc_share]

    def plot_hist(share, what='fc'):
        histplot(share, log_scale=True, bins=15, stat="proportion" if relative else "count", kde=False)
        # plt.title(f"{what} share")
        # plt.tight_layout()
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.ylabel("Fraction of models", fontsize=16)
        plt.xlabel("Parameter share of convolutional part", fontsize=16)
        plt.savefig(f"../../tmp/{arch}-{dataset_name}-{what}_share-hist.png")
        plt.show()

    def plot_scatter(share, what='fc'):
        plt.scatter(param_values_for_shares, share)
        # plt.title(f"{what} share")
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"../../tmp/{arch}-{dataset_name}-{what}_share-scatter.png")
        plt.show()

    # plot_hist(fc_share, what='fc')
    # plot_scatter(fc_share, what='fc')

    plot_hist(conv_share, what='conv')
    # plot_scatter(conv_share, what='conv')

    histplot(param_values, log_scale=True, bins=4, stat="proportion" if relative else "count")


def format_parameter_count(n, decimals=1):
    """
    Format a number into a string with suffixes: k for thousands, M for millions, B for billions, etc.

    Args:
        n (int or float): The number to format.
        decimals (int): Number of decimal places.

    Returns:
        str: The formatted string.
    """
    if n is None:
        return "—"

    suffixes = ['', 'k', 'M', 'B', 'T', "P"]
    magnitude = 0
    abs_n = abs(n)

    while abs_n >= 1000 and magnitude < len(suffixes) - 1:
        abs_n /= 1000.0
        magnitude += 1

    formatted = f"{n / (1000 ** magnitude):.{decimals}f}{suffixes[magnitude]}"

    # Remove trailing zeros and decimal point if not needed
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')

    return formatted


def get_dataset_dims(dataset_name):
    side_size, colors, classes = {
        "MNIST": (28, 1, 10),
        "FashionMNIST": (28, 1, 10),
        "EuroSAT": (64, 3, 10),
        "CIFAR10": (32, 3, 10)
    }[dataset_name]
    return side_size, colors, classes


def convert_ind_to_model_for_dataset(ind: Individual, dataset_name):
    side_size, colors, classes = get_dataset_dims(dataset_name)
    return convert_ind_to_model(ind, side_size, colors, classes)


def convert_ind_to_model(ind: Individual, side_size, colors, classes=10):
    return convert_graph_to_model(DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode).restore(ind.graph),
                                  side_size, colors, classes)


def convert_graph_to_model(graph: NasGraph, side_size, colors, classes=10):
    input_shape = [side_size, side_size, colors]
    return NeuralSearchModel(NASTorchModel).compile_model(graph, input_shape, classes,
                                                          preprocess_graph_for_imagenet=False).model


def count_graph_flops(graph: NasGraph, side_size, colors):
    m = convert_graph_to_model(graph, side_size, colors)
    return count_flops(m, side_size, colors)


def count_flops(torch_model, side_size, colors):
    import torch
    from torchtnt.utils.flops import FlopTensorDispatchMode
    import copy
    input = torch.randn(1, colors, side_size, side_size).to(torch_model.parameters().__next__().device)

    with FlopTensorDispatchMode(torch_model) as ftdm:
        # count forward flops
        res = torch_model(input).mean()
        flops_forward = copy.deepcopy(ftdm.flop_counts)

        # reset count before counting backward flops
        # ftdm.reset()
        # res.backward()
        # flops_backward = copy.deepcopy(ftdm.flop_counts)
    return flops_forward


def count_flops_number(torch_model, side_size, colors):
    return sum(count_flops(torch_model, side_size, colors)[""].values())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_parameters(model):
    total_conv = 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        # if len(param.shape) == 4:  # Assume there are only conv and fc layers
        layer_name = name.split('.')[0]
        if layer_name == "output_layer":
            continue
        name_parts = layer_name.split('_')
        node_word, node_uid = name_parts[:2]
        assert node_word == "node"

        if 'conv' in name or 'features' in name or "conv" in model._graph.get_node_by_uid(node_uid).name:
            total_conv += param.numel()
    return total_conv


def count_fc_percantage(model):
    parameters = count_parameters(model)
    return (parameters - count_conv_parameters(model)) / parameters

def table_of_best_models_for_cls_dataset(include_flops=False, table_format="latex_raw"):
    """
    For each dataset, builds a separate LaTeX table with the best models of each type.
    The best model is determined by the highest accuracy of all its evaluation attempts.
    
    Use tabulate for the table building.
    
    table_format is either "github" or "latex_raw"
    """
    convkan_metrics = {
        "FashionMNIST": {"cnn": (157_030, 0.9014, 32_340_992), "kan": (94_875, 0.8969, 606_305),
                         "cnn-kan": (94_950, 0.8884, 113_680)},
        "MNIST": {"cnn": (157_030, 0.9912, 32_340_992), "kan": (94_875, 0.9890, 606_305),
                  "cnn-kan": (94_950, 0.9875, 113_680)},
        "CIFAR10": {"cnn": (8_453_642, 0.7608, None), "torch-kan": (75_870_343, 0.6462, None),
                    "fast-kan": (75_870_314, 0.5721, None)},
    }
    extra_results = get_extra_final_results()
    for dataset in dataset_results_dir:
        dataset_table = []
        for arch in dataset_results_dir[dataset]:
            convkan_metric = convkan_metrics.get(dataset, {}).get(arch, None)

            print(f"Dataset: {dataset}, Model: {arch}")
            base_eval_paths = dataset_results_dir[dataset][arch]
            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), base_eval_paths))
            final_results = merge_dicts(list(
                map(lambda path: json.load(open(path + "/final_results.json")), base_eval_paths)))

            def best_model_by(first_better, filter):
                best_model_accuracy = None
                best_model_params = None
                best_model_flops = None
                for history in histories:
                    for ind in history.final_choices:
                        this_results = final_results.get(ind.uid, [])
                        if not isinstance(this_results, List):
                            this_results = [this_results]
                        this_results.extend(extra_results.get(ind.uid, []))
                        if not this_results:
                            continue

                        ind_metric_values = [this_run["accuracy"] for this_run in this_results]
                        this_model_aggregated = max(ind_metric_values)  # average(ind_metric_values)
                        this_model_params = ind.fitness.values[1]
                        if filter is not None and not filter(this_model_params, this_model_aggregated):
                            continue
                        if best_model_params is None or first_better((this_model_params, this_model_aggregated),
                                                                     (best_model_params, best_model_accuracy)):
                            best_model_accuracy = this_model_aggregated
                            best_model_params = this_model_params

                            image_side_size = \
                                {"EuroSAT": 64, "CIFAR10": 32, "MNIST": 28, "FashionMNIST": 28, "ImageNet1k": 224}[
                                    dataset]
                            try:
                                best_model_flops = sum(count_graph_flops(
                                    DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode).restore(
                                        ind.graph),
                                    image_side_size,
                                    1 if dataset not in ["EuroSAT", "CIFAR10", "ImageNet1k"] else 3)[""].values())
                            except AttributeError:
                                best_model_flops = None
                return best_model_params, best_model_accuracy, best_model_flops

            best_accuracy_model = best_model_by(lambda new, old: new[1] > old[1], None)
            dataset_table.append(
                (graph_name_mapping[arch], best_accuracy_model[0], best_accuracy_model[1], best_accuracy_model[2]))
            if convkan_metric is not None:
                smallest_model_winning_convkan = best_model_by(lambda new, old: new[0] < old[0],
                                                               lambda p, acc: acc > convkan_metric[1])
                if smallest_model_winning_convkan[
                    1] is not None and smallest_model_winning_convkan != best_accuracy_model:
                    dataset_table.append(
                        (graph_name_mapping[arch], smallest_model_winning_convkan[0], smallest_model_winning_convkan[1],
                         smallest_model_winning_convkan[2]))

                # Add the ConvKAN metric to the table
                competitor = "ConvKAN" if dataset in ["MNIST", "FashionMNIST"] else "Dronkin"
                dataset_table.append(
                    (f"{graph_name_mapping[arch]} ({competitor})", convkan_metric[0], convkan_metric[1],
                     convkan_metric[2]))

        # Before, round the accuracy to 3 decimal places.
        dataset_table = [(arch, params, round(acc, 3), flops) for arch, params, acc, flops in dataset_table]
        best_metric = max([acc for _, _, acc, _ in dataset_table])
        # Highlight the best model's accuracy in bold
        dataset_table = [(arch, format_parameter_count(params), f"\\textbf{{{acc}}}" if acc == best_metric else acc,
                          format_parameter_count(flops)) for
                         arch, params, acc, flops in
                         dataset_table]
        print(dataset_table)

        if not include_flops:
            dataset_table = [(arch, params, acc) for arch, params, acc, _ in dataset_table]

        headers = ["Architecture", "\\#Parameters", "Accuracy"]
        if include_flops:
            headers.append("FLOPs")
        table = tabulate.tabulate(dataset_table, headers=headers,
                                  tablefmt=table_format)
        print(table)



def table_of_best_models_for_ts_dataset(include_flops=False):
    """
    For each dataset, builds a separate LaTeX table with the best models of each type.
    The best model is determined by the highest accuracy of all its evaluation attempts.
    
    Use tabulate for the table building.
    """
    convkan_metrics = {
        "FashionMNIST": {"cnn": (157_030, 0.9014, 32_340_992), "kan": (94_875, 0.8969, 606_305),
                         "cnn-kan": (94_950, 0.8884, 113_680)},
        "MNIST": {"cnn": (157_030, 0.9912, 32_340_992), "kan": (94_875, 0.9890, 606_305),
                  "cnn-kan": (94_950, 0.9875, 113_680)},
    }
    extra_results = get_extra_final_results()
    for dataset in dataset_results_dir:
        dataset_table = []
        for arch in dataset_results_dir[dataset]:
            convkan_metric = convkan_metrics.get(dataset, {}).get(arch, None)

            print(f"Dataset: {dataset}, Model: {arch}")
            base_eval_paths = dataset_results_dir[dataset][arch]
            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), base_eval_paths))
            final_results = merge_dicts(list(
                map(lambda path: json.load(open(path + "/final_results.json")), base_eval_paths)))

            for history in histories:
                for ind in history.final_choices:
                    this_results = final_results[ind.uid] if ((dataset, graph_name_mapping[arch])
                                                              not in ignore_original_evals) else []
                    if not isinstance(this_results, List):
                        this_results = [this_results]
                    this_results.extend(extra_results.get(ind.uid, []))

                    ind_metric_values = [this_run["l1"] for this_run in this_results]
                    this_model_aggregated = min(ind_metric_values)  # average(ind_metric_values)
                    ind_metric_values_ssim = [this_run["ssim"] for this_run in this_results if "ssim" in this_run]
                    this_model_aggregated_ssim = max(ind_metric_values_ssim)
                    this_model_params = ind.fitness.values[1]

                    dataset_table.append((graph_name_mapping[arch], this_model_params, this_model_aggregated,
                                          this_model_aggregated_ssim))
            # if convkan_metric is not None:
            #     smallest_model_winning_convkan = best_model_by(lambda new, old: new[0] < old[0],
            #                                                    lambda p, acc: acc > convkan_metric[1])
            #     if smallest_model_winning_convkan[
            #         1] is not None and smallest_model_winning_convkan != best_accuracy_model:
            #         dataset_table.append((arch, smallest_model_winning_convkan[0], smallest_model_winning_convkan[1],
            #                               smallest_model_winning_convkan[2]))
            # 
            #     # Add the ConvKAN metric to the table
            #     dataset_table.append((f"{arch} (ConvKAN)", convkan_metric[0], convkan_metric[1], convkan_metric[2]))

        # Before, round the accuracy to 3 decimal places.
        dataset_table = [(arch, params, round(l1, 3), round(ssim, 3)) for arch, params, l1, ssim in dataset_table]
        best_metric = min([l1 for _, _, l1, _ in dataset_table])
        best_ssim = max([ssim for _, _, _, ssim in dataset_table])
        # Highlight the best model's accuracy in bold
        dataset_table = [(arch, format_parameter_count(params),
                          f"\\textbf{{{l1}}}" if l1 == best_metric else l1,
                          f"\\textbf{{{ssim}}}" if ssim == best_ssim else ssim,
                          ) for
                         arch, params, l1, ssim in
                         dataset_table]
        print(dataset_table)

        if not include_flops:
            dataset_table = [(arch, params, l1, ssim) for arch, params, l1, ssim in dataset_table]

        headers = ["Architecture", "\\#Parameters", "L1 Loss", "SSIM"]
        if include_flops:
            headers.append("FLOPs")
        table = tabulate.tabulate(dataset_table, headers=headers,
                                  tablefmt="latex_raw")
        print(table)

def merge_dicts(l):
    result = {}
    for d in l:
        result.update(d)
    return result


def plot_all_pareto_fronts(remove_domniated: bool, merge_runs: bool,
                           try_building_statintervals: bool = True,
                           plot_line_between_points: bool = False,
                           only_best_run: bool = False,
                           metric_name: str = "accuracy"
                           ):
    for dataset in dataset_results_dir:
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_parameter_count(x, decimals=0)))
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=14)

        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            base_eval_paths = dataset_results_dir[dataset][model]
            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), base_eval_paths))
            final_results = merge_dicts(list(
                map(lambda path: json.load(open(path + "/final_results.json")), base_eval_paths)))

            # plot_opt_pareto_front(history, label=model, case_name=dataset)
            extra_results = get_extra_final_results()

            if merge_runs:
                history_groups = [histories]
                group_model_names = [model]
            else:
                def listify(l):
                    return [[el] for el in l]

                history_groups = listify(histories)
                group_model_names = [f"{graph_name_mapping[model]} ({parse_run_name(Path(path).name)[0]})" for path in
                                     base_eval_paths]
            for group_model_name, group_histories in zip(group_model_names, history_groups):
                print("RUN PREFIX:", group_model_name)
                plot_final_pareto_front(group_histories, final_results,
                                        label=graph_name_mapping.get(group_model_name, group_model_name),
                                        case_name=dataset,
                                        final_metric_name=metric_name, extra_results=extra_results,
                                        remove_dominated=remove_domniated,
                                        try_building_statintervals=try_building_statintervals,
                                        plot_line_between_points=plot_line_between_points,
                                        only_best_run=only_best_run
                                        )
        plt.legend()
        # plt.tight_layout()
        plt.show()
        reset_next_color_index()
        # exit()


def plot_parameter_numbers(merge_groups=False, relative=True):
    for dataset in dataset_results_dir:
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            if not merge_groups:
                for base_eval_path in dataset_results_dir[dataset][model]:
                    print(base_eval_path)
                    prefix, _, _ = parse_run_name(Path(base_eval_path).name)
                    print(f"Prefix: {prefix}")
                    history_path = base_eval_path + "/history.json"
                    history = OptHistory.load(history_path)

                    plot_parameter_number_hist([history], dataset_name=dataset, arch=model, relative=relative)
                    plt.title(f"Parameter number distribution for {dataset} {graph_name_mapping[model]} ({prefix})")
                    plt.show()
                    reset_next_color_index()
            else:
                histories = list(
                    map(lambda path: OptHistory.load(path + "/history.json"), dataset_results_dir[dataset][model]))
                plot_parameter_number_hist(histories, dataset_name=dataset, arch=model, relative=relative)
                plt.title(f"Parameter number distribution for {dataset} {graph_name_mapping[model]}")
                plt.savefig(f"../../tmp/pareto/{dataset.lower()}-{model}-hist.png")
                plt.show()
                reset_next_color_index()


def plot_fitness_line():
    for dataset in dataset_results_dir:
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")
            for history_path in dataset_results_dir[dataset][model]:
                print(history_path)
                history = OptHistory.load(history_path + "/history.json")

                history.show(PlotTypesEnum.fitness_line)


def scatter_of_models_in_histories_by_flops():
    for dataset in dataset_results_dir:
        results_by_arch = {}
        for model in dataset_results_dir[dataset]:
            print(f"Dataset: {dataset}, Model: {model}")

            histories = list(
                map(lambda path: OptHistory.load(path + "/history.json"), dataset_results_dir[dataset][model]))
            all_params = []
            all_flops = []
            all_accuracies = []
            for history in histories:
                for ind in individuals_pool(history):
                    side_size, colors, classes = get_dataset_dims(dataset)
                    try:
                        m = convert_ind_to_model_for_dataset(ind, dataset)
                    except:
                        continue
                    fitness = ind.fitness.values
                    accuracy = abs(fitness[0])
                    if accuracy < 0.5:
                        continue
                    # print(ind.metadata)
                    flops = count_flops_number(m, side_size, colors)
                    all_accuracies.append(accuracy)
                    all_params.append(ind.fitness.values[1])
                    all_flops.append(flops)
            results_by_arch[model] = (all_accuracies, all_params, all_flops)

        min_trials = min(len(r[0]) for r in results_by_arch.values())
        print(f"Min num trials: {min_trials}")
        for model in dataset_results_dir[dataset]:
            all_accuracies, all_params, all_flops = results_by_arch[model]
            random.seed(42)
            chosen_indices = sorted(random.sample(range(len(all_accuracies)), min_trials))

            def select(l):
                return [l[i] for i in chosen_indices]

            all_accuracies = select(all_accuracies)
            all_params = select(all_params)
            all_flops = select(all_flops)

            print(f"Total models: {len(all_flops)}")
            print(f"Total FLOPs per NAS: {format_parameter_count(sum(all_flops))}")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_parameter_count(x, decimals=0)))
            plt.scatter(all_flops, all_accuracies, label=graph_name_mapping[model])

            # plt.title(f"FLOPs vs accuracy for {dataset} {graph_name_mapping[model]}")

        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('legend', fontsize=14)
        plt.xlabel("FLOPs", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.tight_layout()

        # plt.title(f"FLOPs vs accuracy for {dataset}")
        plt.legend()
        plt.savefig(f"../../tmp/scatter/{dataset.lower()}-all-scatter.png")
        plt.show()
        reset_next_color_index()


def quartal_table():
    seas5_l1 = [
        0.046,
        0.083,
        0.208,
        0.044,
        0.090,
        0.229,
        0.048,
        0.074,
        0.131,
        0.046,
        0.077,
        0.142,
    ]

    seas5_ssim = [0.857, 0.753, 0.472, 0.852, 0.743, 0.447, 0.857, 0.771, 0.533, 0.855, 0.765, 0.531]
    cnn_manual_l1 = [0.035, 0.062, 0.182, 0.037, 0.068, 0.144, 0.037, 0.058, 0.125, 0.036, 0.059, 0.118]
    cnn_manual_ssim = [0.888, 0.770, 0.474, 0.882, 0.775, 0.483, 0.891, 0.793, 0.544, 0.891, 0.782, 0.533]

    prefix = "C:/dev/aim/nas_kan_results/_results/"
    result_paths = {
        # "CNN": "_ice_cnn_laptev_2025-03-14_15-57-55",
        "CNN": "_ice_cnn_laptev_2025-03-15_13-56-06",  # Fixed quartal splitting
        # "KAN": "_ice_dense-kan_laptev_2025-03-15_00-28-20"
        "KAN": "_ice_dense-kan_laptev_2025-03-15_00-28-20"
    }

    def unique(d):
        for k in d:
            return d[k]

    result_jsons = {
        p: unique(json.load(open(prefix + result_paths[p] + "/final_results.json"))) for p in result_paths
    }

    table_header = ["Period", "S5 L1", "S5 SSIM", "Borisova L1", "Borisova SSIM"]
    for arch in result_paths:
        table_header.append(f"{arch} L1")
        table_header.append(f"{arch} SSIM")
    table = []
    for index in range(len(seas5_ssim) - 1):
        q = (index % 3) + 1
        year = index // 3 + 2020

        period = f"{year} Q{q}" 
        row = [period, seas5_l1[index], seas5_ssim[index], cnn_manual_l1[index], cnn_manual_ssim[index]]
        for arch in result_paths:
            q_results_l1 = []
            q_results_ssim = []
            for evaluation in result_jsons[arch]:
                q_results_l1.append(evaluation[f"{year} Q{q}_l1"])
                q_results_ssim.append(evaluation[f"{year} Q{q}_ssim"])
            row.append(average(q_results_l1))
            row.append(average(q_results_ssim))
        table.append(row)
    print(table)
    return table, table_header


def format_table(table, headers):
    # 0: Period,
    # 1: SEAS5 L1, 2: SEAS5 SSIM,
    # 3: Manual CNN L1, 4: Manual CNN SSIM,
    # 5: CNN (NAS) L1, 6: CNN (NAS) SSIM.
    # L1 — the smallest value, SSIM — the highest.
    formatted_table = []

    for row in table:
        new_row = row.copy()
        # --- L1 columns (indexes 1, 3, 5) ---
        l1_indices = [1, 3, 5, 7]
        l1_values = [row[i] for i in l1_indices]
        best_l1 = min(l1_values)
        for i in l1_indices:
            # if row[i] == best_l1:
            #     new_row[i] = "\\textbf{{{:.3f}}}".format(row[i])
            # else:
                new_row[i] = "{:.3f}".format(row[i])
        # --- SSIM columns (indexes 2, 4, 6) ---
        ssim_indices = [2, 4, 6, 8]
        ssim_values = [row[i] for i in ssim_indices]
        best_ssim = max(ssim_values)
        for i in ssim_indices:
            # if row[i] == best_ssim:
            #     new_row[i] = "\\textbf{{{:.3f}}}".format(row[i])
            # else:
                new_row[i] = "{:.3f}".format(row[i])

        formatted_table.append(new_row)

    n = len(table)
    num_columns = len(table[0])
    avg_row = ["Average"]
    col_avgs = []
    for j in range(1, num_columns):
        col_vals = [row[j] for row in table]
        avg_val = sum(col_vals) / len(col_vals)
        col_avgs.append(avg_val)

    l1_avgs = [col_avgs[i] for i in [0, 2, 4, 6]]
    best_avg_l1 = min(l1_avgs)

    ssim_avgs = [col_avgs[i] for i in [1, 3, 5, 7]]
    best_avg_ssim = max(ssim_avgs)

    # The average row with formatting.
    for j in range(1, num_columns):
        val = col_avgs[j - 1]
        # Decide based on header keyword:
        if "L1" in headers[j]:
            if val == best_avg_l1:
                avg_row.append("\\textbf{{{:.3f}}}".format(val))
            else:
                avg_row.append("{:.3f}".format(val))
        elif "SSIM" in headers[j]:
            if val == best_avg_ssim:
                avg_row.append("\\textbf{{{:.3f}}}".format(val))
            else:
                avg_row.append("{:.3f}".format(val))
        else:
            avg_row.append("{:.3f}".format(val))
    formatted_table.append(avg_row)
    return formatted_table


def print_quartal_table():
    table, headers = quartal_table()
    formatted_table = format_table(table, headers)
    print(tabulate.tabulate(formatted_table, headers=headers, tablefmt="latex_raw"))

if __name__ == '__main__':
    # Boxplot
    # plot_all_pareto_fronts(remove_domniated=True, merge_runs=True, try_building_statintervals=True,
    #                        plot_line_between_points=False, only_best_run=False,
    #                        metric_name="l1"
    #                        )

    # Non-dominated stat-intervals
    # plot_all_pareto_fronts(remove_domniated=True, merge_runs=True, try_building_statintervals=True,
    #                        plot_line_between_points=True, only_best_run=False)

    # plot_parameter_numbers(merge_groups=True, relative=True)

    # table_of_best_models_for_dataset(include_flops=True, table_format="github")

    # plot_fitness_line()

    print_quartal_table()

    # scatter_of_models_in_histories_by_flops()
