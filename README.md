# KAN'o'NAS: Neural Architecture Search for fair architecture comparison

[![PyPI version](https://badge.fury.io/py/kan-o-nas.svg)](https://pypi.org/project/kan-o-nas/)

KAN'o'NAS is an architecture-agnostic, evolutionary full NAS framework for fair comparison of convolutional networks (CNN) and Kolmogorov–Arnold networks (KAN), including hybrids.
It represents networks as DAGs, jointly optimizes topology and per-node hyperparameters, and selects models on a Pareto frontier of quality vs. complexity.
The framework is based on [GOLEM](https://github.com/aimclub/GOLEM) (evolutionary graph optimization) and uses PyTorch as the training backend.

---

## Install

```bash
git clone https://github.com/ITMO-NSS-team/kan-o-nas.git
cd kan-o-nas
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Also, install the corresponding datasets: MNIST, FashionMNIST, EuroSAT or CIFAR-10 for image classification, 

Quick start

The repository provides two runnable examples. They demonstrate how to define a search space, run NAS, and post-train selected finalists.

### Image classification

```python cases/image_classification.py```


### Spatial time series forecasting

```python cases/ts_forecasting.py```


Both scripts save NAS artifacts and final metrics to the output directory chosen inside the script.

## Method Overview

- Representation. Candidate models are encoded as DAGs. Nodes are layers (Conv2D, KANConv2D, Linear, KANLinear, Pool, Flatten). Edges define data flow and allow short-cuts (1–2 inputs per node).
- Search. An evolutionary algorithm with subtree crossover and layer/edge mutations explores structures and per-node hyperparameters.
- Objectives. Multi-objective selection by task quality (e.g., accuracy or L1) and complexity (parameters by default; FLOPs or wall time are supported).
- Validation. Graph-level rules ensure acyclicity, shape consistency, feasible connectivity, and complexity bounds.
- Evaluation. Finalists are retrained to estimate mean performance; the Pareto set is reported.


## Configure the framework

1) Task definition
- Choose task and shapes: `Task`, `TaskTypesEnum`, `ModelRequirements(input_shape=..., output_shape=... or num_of_classes=...)`.

2) Search space
- Layer families: `LayersPoolEnum` (e.g., `conv2d`, `kan_conv2d`, `linear`, `kan_linear`).
- Model scaffold: `ModelRequirements` fields `primary`, `secondary`, `min_num_of_conv_layers`, `max_num_of_conv_layers`, `min_nn_depth`, `max_nn_depth`.
- Per-node ranges: `ConvRequirements`, `KANConvRequirements`, `BaseLayerRequirements`, `KANLinearRequirements`.
- Initial graphs: `ConvGraphMaker(requirements=..., rules=...)`, `BaseGraphBuilder().set_builder(...).build(pop_size)`.
- Graph types (when needed): `NasGraph`, `NasNode`.

3) Validation rules
- DAG soundness: `has_no_cycle`, `has_no_self_cycled_nodes`.
- Classification constraints: `model_has_no_conv_layers`, `model_has_several_starts`, `model_has_several_roots`, `model_has_wrong_number_of_flatten_layers`, `no_linear_layers_before_flatten`, `filter_size_changes_monotonically(increases=True)`.
- Forecasting constraints: `only_conv_layers`, `no_transposed_layers_before_conv`, `filter_size_changes_monotonically(increases=False)`, `right_output_size`, `output_node_has_channels(...)`.
- Shape/complexity checks: `model_has_dim_mismatch(...)`, `has_too_much_parameters(...)` (optionally `has_too_much_flops(...)`, `has_too_much_time(...)`).
- Attach via `GraphGenerationParams(..., rules_for_constraint=[...])`.

4) Objectives and metrics
- Quality: for classification use `MetricsRepository().metric_by_id(ClassificationMetricsEnum.accuracy)`; for forecasting train with `L1Loss` and report `L1` and `ssim`.
- Complexity: `compute_total_graph_parameters(...)`, optionally `get_flops_from_graph(...)`, `get_time_from_graph(...)`.
- Provide to composer: `.with_metrics([quality_metric, complexity_metric_fn])`.

5) Search parameters
- Genetic setup: `GPAlgorithmParameters(genetic_scheme_type=GeneticSchemeTypesEnum.steady_state, mutation_types=[MutationTypesEnum.*], crossover_types=[CrossoverTypesEnum.subtree], pop_size=..., max_pop_size=..., regularization_type=RegularizationTypesEnum.none, multi_objective=True)`.
- Custom operators (optional): define `combined_mutation` with `register_native`.
- Graph generation: `DirectAdapter(...)`, `NNNodeFactory(..., DefaultChangeAdvisor())`, `GraphGenerationParams(adapter=..., rules_for_constraint=..., node_factory=...)`.

6) Training setup
- Trainer: `ModelConstructor(model_class=NASTorchModel, trainer=NeuralSearchModel, device=..., loss_function=..., optimizer=AdamW, metrics=...)`.
    - Classification losses: `CrossEntropyLoss` or `FocalLoss`.
    - Forecasting loss: `L1Loss`.
- Composer pipeline:  
  `ComposerBuilder(task).with_composer(NNComposer).with_optimizer(NNGraphOptimiser).with_requirements(NNComposerRequirements(...)).with_metrics([...]).with_optimizer_params(GPAlgorithmParameters(...)).with_initial_pipelines(initial_pipelines).with_graph_generation_param(GraphGenerationParams(...))` → `composer = builder.build()` → `composer.set_trainer(model_trainer)` → `composer.compose_pipeline(train_data, valid_or_test_data)`.

7) Outputs
- Persist and reuse: `composer.save(path)`, access `composer.history.final_choices`, restore with `DirectAdapter.restore(...)`, reload runs via `OptHistory.load(path)`.
- Summaries: write metrics to JSON (e.g., `final_results.json`).

## Outputs

- NAS history for reuse or post-training only runs.
- Finalist graphs and trained weights if enabled.
- Metrics summary per finalist (e.g., accuracy for classification; L1 and SSIM for forecasting).
- Optional qualitative images for forecasting.

## Roadmap

- Richer KAN variants, kernel function libraries
- Larger and more diverse datasets (e.g. ImageNet).
- Experimentation with optimizer, including surrogate models and indirect encodings for search efficiency at the domain of large models.

## Citation

TBD

## License

The code is published under the MIT License.
