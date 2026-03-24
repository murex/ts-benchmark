"""Benchmark executor for scenario-generation models."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..benchmark.protocol import Protocol
from ..dataset.runtime import DatasetInstance
from ..metrics.definition import MetricConfig
from ..metrics.distributional import compute_distributional_metrics
from ..metrics.scoring import compute_sample_scoring_metrics
from ..model.contracts import RuntimeContext, ScenarioModel, ScenarioRequest, TrainingData
from ..results import BenchmarkResults, MetricResult, ModelResult, ScenarioOutput
from ..serialization import to_jsonable
from ..utils import JsonObject


class ScenarioBenchmark:
    """Fit models, generate scenarios, and compute benchmark metrics."""

    def __init__(
        self,
        models: Mapping[str, ScenarioModel],
        *,
        protocol: Protocol,
        metric_configs: list[MetricConfig],
        runtime: RuntimeContext | None = None,
        keep_scenarios: bool = False,
        freq: str = "B",
    ):
        if not models:
            raise ValueError("At least one model must be provided.")

        self.models = dict(models)
        self.protocol = protocol
        self.metric_configs = list(metric_configs)
        self.runtime = runtime or RuntimeContext()
        self.keep_scenarios = bool(keep_scenarios)
        self.freq = str(freq)

    def _runtime(self) -> RuntimeContext:
        return RuntimeContext(device=self.runtime.device, seed=self.runtime.seed)

    def _training_data(self, dataset: DatasetInstance) -> TrainingData:
        return TrainingData(
            returns=dataset.train_returns,
            protocol=self.protocol,
            asset_names=list(dataset.asset_names),
            freq=self.freq,
            runtime=self._runtime(),
            metadata=JsonObject(
                {
                    "dataset_name": dataset.name,
                    "dataset_source": dataset.source,
                    "runtime": to_jsonable(self._runtime()),
                }
            ),
        )

    def _sample_model_scenarios(
        self,
        model_name: str,
        dataset: DatasetInstance,
        model_index: int,
    ) -> np.ndarray:
        model = self.models[model_name]
        samples = []
        runtime = self._runtime()
        for context_index, context in enumerate(dataset.contexts):
            base_seed = 0 if self.runtime.seed is None else int(self.runtime.seed)
            sample_seed = base_seed + 10_000 * (model_index + 1) + context_index
            request = ScenarioRequest(
                context=context,
                horizon=self.protocol.horizon,
                n_scenarios=self.protocol.n_model_scenarios,
                protocol=self.protocol,
                mode=self.protocol.generation_mode,
                seed=sample_seed,
                asset_names=list(dataset.asset_names),
                freq=self.freq,
                runtime=runtime,
                metadata=JsonObject(
                    {
                        "context_index": context_index,
                        "model_name": model_name,
                        "dataset_name": dataset.name,
                        "dataset_source": dataset.source,
                        "runtime": to_jsonable(runtime),
                        "evaluation_timestamp": None
                        if dataset.evaluation_timestamps is None
                        else dataset.evaluation_timestamps[context_index],
                    }
                ),
            )
            model_paths = model.sample(request).samples
            samples.append(model_paths)
        return np.stack(samples, axis=0)

    def _selected_metric_names(self) -> set[str]:
        return {metric.name for metric in self.metric_configs}

    def _metric_results_for_model(
        self,
        *,
        model_name: str,
        model_samples: np.ndarray,
        realized_futures: np.ndarray,
        reference_scenarios: np.ndarray | None,
    ) -> list[MetricResult]:
        selected = self._selected_metric_names()
        metric_values = {
            name: value
            for name, value in compute_sample_scoring_metrics(model_samples, realized_futures).items()
            if name in selected
        }
        if reference_scenarios is not None:
            metric_values.update(
                {
                    name: value
                    for name, value in compute_distributional_metrics(model_samples, reference_scenarios).items()
                    if name in selected
                }
            )
        metric_by_name = {metric.name: metric for metric in self.metric_configs}
        return [
            MetricResult(
                model_name=model_name,
                metric_name=metric.name,
                value=float(metric_values[metric.name]),
                direction=metric.direction,
                category=metric.category,
                granularity=metric.granularity,
                aggregation=metric.aggregation,
            )
            for metric in self.metric_configs
            if metric.name in metric_values
        ]

    def _prepare_reference_scenarios(self, dataset: DatasetInstance) -> np.ndarray | None:
        if not dataset.has_reference_scenarios():
            return None
        return dataset.sample_reference_scenarios(
            n_scenarios=self.protocol.n_reference_scenarios,
            seed=self.runtime.seed,
        )

    def _evaluate_model(
        self,
        name: str,
        model: ScenarioModel,
        model_index: int,
        dataset: DatasetInstance,
        training_data: TrainingData,
        reference_scenarios: np.ndarray | None,
    ) -> ModelResult:
        model.fit(training_data)
        model_samples = self._sample_model_scenarios(name, dataset, model_index)
        scenario_output = None
        if self.keep_scenarios:
            scenario_output = ScenarioOutput(
                model_name=name,
                generated_scenarios=np.asarray(model_samples, dtype=float),
            )
        return ModelResult(
            model_name=name,
            metric_results=self._metric_results_for_model(
                model_name=name,
                model_samples=model_samples,
                realized_futures=dataset.realized_futures,
                reference_scenarios=reference_scenarios,
            ),
            scenario_output=scenario_output,
        )

    def _assemble_results(
        self,
        dataset: DatasetInstance,
        model_results: list[ModelResult],
        reference_scenarios: np.ndarray | None,
    ) -> BenchmarkResults:
        metadata = JsonObject(
            {
                "dataset_name": dataset.name,
                "dataset_source": dataset.source,
                "device": self.runtime.device,
                "has_reference_scenarios": bool(reference_scenarios is not None),
                **to_jsonable(self.protocol),
            }
        )
        return BenchmarkResults.from_model_results(
            model_results,
            metric_configs=self.metric_configs,
            reference_scenarios=(
                None
                if not self.keep_scenarios or reference_scenarios is None
                else np.asarray(reference_scenarios, dtype=float)
            ),
            metadata=metadata,
        )

    def run(self, dataset: DatasetInstance) -> BenchmarkResults:
        if dataset.protocol != self.protocol:
            raise ValueError(
                "ScenarioBenchmark received a dataset instance whose protocol does not "
                "match the benchmark protocol passed to the executor."
            )
        training_data = self._training_data(dataset)
        reference_scenarios = self._prepare_reference_scenarios(dataset)
        model_results = [
            self._evaluate_model(name, model, i, dataset, training_data, reference_scenarios)
            for i, (name, model) in enumerate(self.models.items())
        ]
        return self._assemble_results(dataset, model_results, reference_scenarios)
