"""Benchmark executor for scenario-generation models."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..benchmark.protocol import Protocol, protocol_metadata_payload
from ..dataset.runtime import DatasetInstance
from ..dataset.windows import rolling_history_context_future_triplets, rolling_series_windows
from ..metrics.definition import MetricConfig
from ..metrics.distributional import compute_distributional_metrics
from ..metrics.scoring import compute_sample_scoring_metrics
from ..model.contracts import (
    ForecastWindowCollection,
    RuntimeContext,
    ScenarioModel,
    ScenarioRequest,
    TrainPathCollection,
    TrainingData,
)
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

    def _forecast_window_collection(self, dataset: DatasetInstance) -> ForecastWindowCollection:
        histories_parts: list[list[np.ndarray]] = []
        contexts_parts: list[np.ndarray] = []
        targets_parts: list[np.ndarray] = []
        source_kind = "single_path"
        if dataset.train_paths is not None:
            source_kind = "path_dataset"
            for path in dataset.train_paths:
                histories, contexts, targets = rolling_history_context_future_triplets(
                    path,
                    context_length=self.protocol.context_length,
                    horizon=self.protocol.horizon,
                    stride=self.protocol.train_stride,
                )
                histories_parts.append(histories)
                contexts_parts.append(np.asarray(contexts, dtype=float))
                targets_parts.append(np.asarray(targets, dtype=float))
        else:
            histories, contexts, targets = rolling_history_context_future_triplets(
                dataset.train_returns,
                context_length=self.protocol.context_length,
                horizon=self.protocol.horizon,
                stride=self.protocol.train_stride,
            )
            histories_parts.append(histories)
            contexts_parts.append(np.asarray(contexts, dtype=float))
            targets_parts.append(np.asarray(targets, dtype=float))
        return ForecastWindowCollection(
            histories=[history for histories in histories_parts for history in histories],
            contexts=np.concatenate(contexts_parts, axis=0),
            targets=np.concatenate(targets_parts, axis=0),
            source_kind=source_kind,
            stride=self.protocol.train_stride,
        )

    def _unconditional_path_collection(self, dataset: DatasetInstance) -> TrainPathCollection:
        train_data_mode = self.protocol.unconditional_train_data_mode
        if train_data_mode == "path_dataset":
            if dataset.train_paths is None:
                raise ValueError(
                    "Dataset does not expose train_paths, but the benchmark protocol requested "
                    "unconditional_train_data_mode='path_dataset'."
                )
            return TrainPathCollection(
                paths=[np.asarray(path, dtype=float) for path in dataset.train_paths],
                source_kind="path_dataset",
            )
        if train_data_mode != "windowed_path":
            raise ValueError(
                "Unconditional benchmarks require protocol.unconditional_train_data_mode to be "
                "'windowed_path' or 'path_dataset'."
            )
        window_length = self.protocol.unconditional_train_window_length
        if window_length is None:
            raise ValueError(
                "Unconditional windowed-path benchmarks require protocol.unconditional_train_window_length."
            )
        windows = rolling_series_windows(
            dataset.train_returns,
            window_length=window_length,
            stride=self.protocol.train_stride,
        )
        return TrainPathCollection(
            paths=[np.asarray(window, dtype=float) for window in windows],
            source_kind="windowed_path",
            window_length=window_length,
            stride=self.protocol.train_stride,
        )

    def _training_data(self, dataset: DatasetInstance) -> TrainingData:
        forecast_windows = None
        path_collection = None
        if self.protocol.generation_mode == "forecast":
            forecast_windows = self._forecast_window_collection(dataset)
        else:
            path_collection = self._unconditional_path_collection(dataset)
        return TrainingData(
            returns=dataset.train_returns,
            protocol=self.protocol,
            asset_names=list(dataset.asset_names),
            freq=self.freq,
            forecast_windows=forecast_windows,
            path_collection=path_collection,
            runtime=self._runtime(),
            metadata=JsonObject(
                {
                    "dataset_name": dataset.name,
                    "dataset_source": dataset.source,
                    "runtime": to_jsonable(self._runtime()),
                    "forecast_window_collection": None
                    if forecast_windows is None
                    else {
                        "source_kind": forecast_windows.source_kind,
                        "n_windows": int(forecast_windows.contexts.shape[0]),
                        "context_length": int(forecast_windows.contexts.shape[1]),
                        "horizon": int(forecast_windows.targets.shape[1]),
                        "stride": forecast_windows.stride,
                    },
                    "path_collection": None
                    if path_collection is None
                    else {
                        "source_kind": path_collection.source_kind,
                        "n_paths": len(path_collection.paths),
                        "window_length": path_collection.window_length,
                        "stride": path_collection.stride,
                    },
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
                **protocol_metadata_payload(self.protocol),
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
