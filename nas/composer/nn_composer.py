from pathlib import Path
from typing import Optional, Union, List

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from torch.utils.data import Dataset

from nas.composer.requirements import NNComposerRequirements
from nas.model.model_interface import BaseModelInterface
from nas.optimizer.objective.nas_objective_evaluate import NASObjectiveEvaluate


class NNComposer(Composer):
    def __init__(self,
                 optimizer: EvoGraphOptimizer,
                 composer_requirements: NNComposerRequirements,
                 pipeline_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None,
                 dataloader_num_workers: int = 1,
                 **kwargs):
        super().__init__(optimizer=optimizer)
        self.best_models = ()
        self._dataset_builder = None
        self.trainer = None
        self.pipeline_cache = pipeline_cache
        self.preprocessing_cache = preprocessing_cache
        self.composer_requirements = composer_requirements
        self.dataloader_num_workers = dataloader_num_workers

    # @property
    # def dataset_builder(self):
    #     return self._dataset_builder
    # 
    # @dataset_builder.setter
    # def dataset_builder(self, val):
    #     self._dataset_builder = val
    # 
    # def set_dataset_builder(self, dataset_builder: ImageDatasetBuilder):
    #     self.dataset_builder = dataset_builder
    #     return self

    def set_trainer(self, trainer: BaseModelInterface):
        self.trainer = trainer
        return self

    def set_dataloader_num_workers(self, dataloader_num_workers: int):
        self.dataloader_num_workers = dataloader_num_workers
        return self

    def compose_pipeline(self, train_dataset: Dataset, validation_dataset: Dataset) -> Union[Pipeline, List[Pipeline]]:
        if self.history:
            self.history.clean_results()

        # Data preparation phase
        # data_producer = DataSourceSplitter(self.composer_requirements.cv_folds).build(data)
        assert self.composer_requirements.cv_folds is None
        objective_eval = NASObjectiveEvaluate(objective=self.optimizer.objective,
                                              # data_producer=data_producer,
                                              train_dataset=train_dataset,
                                              validation_dataset=validation_dataset,
                                              model_trainer_builder=self.trainer,
                                              pipeline_cache=self.pipeline_cache,
                                              preprocessing_cache=self.preprocessing_cache,
                                              dataloader_num_workers=self.dataloader_num_workers,
                                              requirements=self.composer_requirements,
                                              # nn_dataset_builder=self.dataset_builder
                                              )

        if self.composer_requirements.collect_intermediate_metric:
            self.optimizer.set_evaluation_callback(objective_eval.evaluate_intermediate_metrics)

        optimization_result = self.optimizer.optimise(objective_eval.evaluate)
        self._convert_best_model(optimization_result)
        return self.best_models if self.optimizer.objective.is_multi_objective else self.best_models[0]

    def _convert_best_model(self, optimization_result):
        adapter = self.optimizer.graph_generation_params.adapter
        self.best_models = [adapter.restore(g) for g in optimization_result]

    def save(self, path: str):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.best_models[0].save(path)
        if self.history:
            self.history.save(path / 'history.json')
