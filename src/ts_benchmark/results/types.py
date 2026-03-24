"""Typed benchmark result-domain objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..metrics import rank_metrics_table
from ..metrics.definition import MetricConfig
from ..model.catalog.plugins import ModelPluginManifest, PluginInfo
from ..model.definition import (
    ModelExecutionConfig,
    ModelReferenceConfig,
    PipelineConfig,
)
from ..run.definition import OutputConfig, RunConfig, TrackingConfig
from ..utils import JsonObject


@dataclass
class BenchmarkDiagnostics:
    distribution_summary: pd.DataFrame | None = None
    distribution_summary_by_asset: pd.DataFrame | None = None
    per_window_metrics: pd.DataFrame | None = None
    functional_smoke_summary: pd.DataFrame | None = None
    functional_smoke_checks: pd.DataFrame | None = None
    model_debug_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    model_name: str
    metric_name: str
    value: float
    direction: str
    category: str | None = None
    granularity: str = "per_window"
    aggregation: str = "mean"
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)


@dataclass(frozen=True)
class MetricRanking:
    model_name: str
    metric_name: str
    rank: float


@dataclass
class ScenarioOutput:
    model_name: str
    generated_scenarios: np.ndarray | None = None
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)


@dataclass
class ModelExecutionRecord:
    requested: ModelExecutionConfig | None = None
    assigned_device: str | None = None
    runtime_device: str | None = None
    resolved_device: str | None = None
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    @property
    def mode(self) -> str:
        if self.requested is None:
            return "inprocess"
        return str(self.requested.mode)


@dataclass
class ModelResult:
    model_name: str
    description: str | None = None
    reference: ModelReferenceConfig | None = None
    params: JsonObject = field(default_factory=JsonObject)
    execution: ModelExecutionRecord | None = None
    pipeline: PipelineConfig | None = None
    metadata: JsonObject = field(default_factory=JsonObject)
    plugin_info: PluginInfo | None = None
    runtime_manifest: ModelPluginManifest | None = None
    fitted_model_info: dict[str, Any] | None = None
    metric_results: list[MetricResult] = field(default_factory=list)
    metric_rankings: list[MetricRanking] = field(default_factory=list)
    average_rank: float | None = None
    scenario_output: ScenarioOutput | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.params, JsonObject):
            self.params = JsonObject(self.params)
        if self.execution is not None and not isinstance(self.execution, ModelExecutionRecord):
            self.execution = ModelExecutionRecord(**self.execution)
        if self.pipeline is not None and not isinstance(self.pipeline, PipelineConfig):
            self.pipeline = PipelineConfig(**self.pipeline)
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)
        if self.scenario_output is not None and not isinstance(self.scenario_output, ScenarioOutput):
            self.scenario_output = ScenarioOutput(**self.scenario_output)

    def metric_map(self) -> dict[str, float]:
        return {metric.metric_name: float(metric.value) for metric in self.metric_results}

    def ranking_map(self) -> dict[str, float]:
        return {ranking.metric_name: float(ranking.rank) for ranking in self.metric_rankings}


@dataclass(frozen=True)
class ModelDeviceAssignment:
    model_name: str
    device: str


@dataclass
class ResolvedRunExecution:
    scheduler: str = "auto"
    execution_mode: str = "sequential"
    requested_device: str | None = None
    resolved_devices: tuple[str, ...] = field(default_factory=tuple)
    assigned_devices: list[ModelDeviceAssignment] = field(default_factory=list)
    parallel_workers: int | None = None

    def assigned_device_map(self) -> dict[str, str]:
        return {assignment.model_name: assignment.device for assignment in self.assigned_devices}


@dataclass
class RunTrackingRecord:
    backend: str | None = None
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    artifact_uri: str | None = None


@dataclass
class RunRecord:
    name: str | None = None
    description: str | None = None
    seed: int = 7
    device: str | None = None
    scheduler: str = "auto"
    output: OutputConfig = field(default_factory=OutputConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    metadata: JsonObject = field(default_factory=JsonObject)
    status: str = "pending"
    requested_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    resolved_output_dir: Path | None = None
    resolved_execution: ResolvedRunExecution = field(default_factory=ResolvedRunExecution)
    tracking_result: RunTrackingRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)
        if not isinstance(self.output, OutputConfig):
            self.output = OutputConfig(**self.output)
        if not isinstance(self.tracking, TrackingConfig):
            self.tracking = TrackingConfig(**self.tracking)


def _rank_model_results(
    model_results: Iterable[ModelResult],
    metric_configs: Iterable[MetricConfig],
) -> list[ModelResult]:
    metric_list = list(metric_configs)
    ranked = [copy.deepcopy(model_result) for model_result in model_results]
    if not ranked or not metric_list:
        return ranked

    allowed_metric_names = {metric.name for metric in metric_list}
    rows: list[dict[str, float]] = []
    index: list[str] = []
    for model_result in ranked:
        model_result.metric_results = [
            metric_result
            for metric_result in model_result.metric_results
            if metric_result.metric_name in allowed_metric_names
        ]
        rows.append(
            {
                metric_result.metric_name: float(metric_result.value)
                for metric_result in model_result.metric_results
            }
        )
        index.append(model_result.model_name)

    metrics_table = pd.DataFrame(rows, index=index)
    metrics_table.index.name = "model"

    required_metric_names = [metric.name for metric in metric_list]
    if any(name not in metrics_table.columns for name in required_metric_names):
        return ranked

    complete_mask = metrics_table[required_metric_names].notna().all(axis=1)
    rankable_names = metrics_table.index[complete_mask].tolist()
    if not rankable_names:
        return ranked

    filtered, rank_table = rank_metrics_table(metrics_table.loc[rankable_names, required_metric_names], metric_list)
    by_name = {model_result.model_name: model_result for model_result in ranked}
    ordered: list[ModelResult] = []
    ranked_names: set[str] = set()
    for model_name in filtered.index.tolist():
        model_result = by_name[model_name]
        metric_by_name = {metric.metric_name: metric for metric in model_result.metric_results}
        model_result.metric_results = [
            replace(
                metric_by_name[metric.name],
                value=float(filtered.loc[model_name, metric.name]),
                direction=metric.direction,
                category=metric.category,
                granularity=metric.granularity,
                aggregation=metric.aggregation,
            )
            for metric in metric_list
        ]
        model_result.metric_rankings = [
            MetricRanking(
                model_name=model_name,
                metric_name=metric.name,
                rank=float(rank_table.loc[model_name, metric.name]),
            )
            for metric in metric_list
        ]
        model_result.average_rank = float(rank_table.loc[model_name, "average_rank"])
        ordered.append(model_result)
        ranked_names.add(str(model_name))

    for model_result in ranked:
        if model_result.model_name in ranked_names:
            continue
        model_result.metric_rankings = []
        model_result.average_rank = None
        ordered.append(model_result)
    return ordered


@dataclass
class BenchmarkResults:
    """Typed benchmark outputs with object-native per-model results."""

    run: RunRecord | None = None
    model_results: list[ModelResult] = field(default_factory=list)
    reference_scenarios: np.ndarray | None = None
    diagnostics: BenchmarkDiagnostics | None = None
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    @classmethod
    def from_model_results(
        cls,
        model_results: Iterable[ModelResult],
        *,
        metric_configs: Iterable[MetricConfig],
        reference_scenarios: np.ndarray | None = None,
        metadata: JsonObject | Mapping[str, object] | None = None,
        run: RunRecord | None = None,
        diagnostics: BenchmarkDiagnostics | None = None,
    ) -> "BenchmarkResults":
        metric_list = list(metric_configs)
        return cls(
            run=run,
            model_results=_rank_model_results(list(model_results), metric_list),
            reference_scenarios=None if reference_scenarios is None else np.asarray(reference_scenarios, dtype=float),
            diagnostics=diagnostics,
            metadata=JsonObject() if metadata is None else (metadata if isinstance(metadata, JsonObject) else JsonObject(metadata)),
        )

    def model_result_map(self) -> dict[str, ModelResult]:
        return {model_result.model_name: model_result for model_result in self.model_results}

    def metric_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for model_result in self.model_results:
            for metric_result in model_result.metric_results:
                if metric_result.metric_name not in seen:
                    seen.add(metric_result.metric_name)
                    names.append(metric_result.metric_name)
        return names

    def metrics_frame(self, *, include_metadata: bool = False) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        index: list[str] = []
        for model in self.model_results:
            row = model.metric_map()
            if model.average_rank is not None:
                row["average_rank"] = float(model.average_rank)
            rows.append(row)
            index.append(model.model_name)
        frame = pd.DataFrame(rows, index=index)
        frame.index.name = "model"
        if include_metadata and not frame.empty:
            insert_at = 0
            for key, value in self.metadata.items():
                frame.insert(insert_at, key, value)
                insert_at += 1
        return frame

    def rankings_frame(self, *, include_metadata: bool = False) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        index: list[str] = []
        for model in self.model_results:
            row = model.ranking_map()
            if model.average_rank is not None:
                row["average_rank"] = float(model.average_rank)
            rows.append(row)
            index.append(model.model_name)
        frame = pd.DataFrame(rows, index=index)
        frame.index.name = "model"
        if include_metadata and not frame.empty:
            insert_at = 0
            for key, value in self.metadata.items():
                frame.insert(insert_at, key, value)
                insert_at += 1
        return frame

    def scenario_outputs(self) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        for model in self.model_results:
            if model.scenario_output is not None and model.scenario_output.generated_scenarios is not None:
                outputs[model.model_name] = np.asarray(model.scenario_output.generated_scenarios, dtype=float)
        return outputs

    def all_metric_results(self) -> list[MetricResult]:
        out: list[MetricResult] = []
        for model in self.model_results:
            out.extend(model.metric_results)
        return out

    def with_metric_configs(self, metric_configs: Iterable[MetricConfig]) -> "BenchmarkResults":
        metric_list = list(metric_configs)
        allowed = {metric.name for metric in metric_list}
        filtered_model_results = []
        for model_result in self.model_results:
            filtered = copy.deepcopy(model_result)
            filtered.metric_results = [
                metric_result
                for metric_result in filtered.metric_results
                if metric_result.metric_name in allowed
            ]
            filtered.metric_rankings = []
            filtered.average_rank = None
            filtered_model_results.append(filtered)
        return BenchmarkResults.from_model_results(
            filtered_model_results,
            metric_configs=metric_list,
            reference_scenarios=self.reference_scenarios,
            metadata=self.metadata,
            run=self.run,
            diagnostics=self.diagnostics,
        )

    def save_metrics_csv(self, path: str) -> None:
        self.metrics_frame(include_metadata=True).to_csv(path)

    def save_ranks_csv(self, path: str) -> None:
        self.rankings_frame(include_metadata=True).to_csv(path)

    def without_scenarios(self) -> "BenchmarkResults":
        stripped_models = [replace(model, scenario_output=None) for model in self.model_results]
        return BenchmarkResults(
            run=self.run,
            model_results=stripped_models,
            reference_scenarios=None,
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )
