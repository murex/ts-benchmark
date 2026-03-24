"""Result enrichment and diagnostics helpers for benchmark runs."""

from __future__ import annotations

import copy
from typing import Any

from ..model.catalog.plugins import PluginInfo, extract_model_plugin_manifest
from ..results import (
    BenchmarkDiagnostics,
    BenchmarkResults,
    ModelExecutionRecord,
    ModelResult,
    build_distribution_summaries,
    build_functional_smoke_tables,
    build_per_window_metrics,
)
from ..benchmark.definition import BenchmarkConfig
from ..utils import JsonObject


def build_diagnostics(
    *,
    config: BenchmarkConfig,
    dataset,
    results: BenchmarkResults,
    model_debug_artifacts: dict[str, Any] | None,
    selected_metrics: list,
) -> BenchmarkDiagnostics | None:
    if not config.diagnostics.enabled():
        return None

    generated_scenarios = results.scenario_outputs()
    distribution_summary = None
    distribution_summary_by_asset = None
    per_window_metrics = None
    functional_smoke_summary = None
    functional_smoke_checks = None

    if config.diagnostics.save_distribution_summary:
        distribution_summary, distribution_summary_by_asset = build_distribution_summaries(
            dataset=dataset,
            generated_scenarios=generated_scenarios,
            reference_scenarios=results.reference_scenarios,
        )

    if config.diagnostics.save_per_window_metrics:
        per_window_metrics = build_per_window_metrics(
            dataset=dataset,
            generated_scenarios=generated_scenarios,
            reference_scenarios=results.reference_scenarios,
            include=[metric.name for metric in selected_metrics],
        )

    if config.diagnostics.functional_smoke.enabled:
        functional_smoke_summary, functional_smoke_checks = build_functional_smoke_tables(
            smoke_config=config.diagnostics.functional_smoke,
            dataset=dataset,
            results=results,
        )

    return BenchmarkDiagnostics(
        distribution_summary=distribution_summary,
        distribution_summary_by_asset=distribution_summary_by_asset,
        per_window_metrics=per_window_metrics,
        functional_smoke_summary=functional_smoke_summary,
        functional_smoke_checks=functional_smoke_checks,
        model_debug_artifacts=dict(model_debug_artifacts or {})
        if config.diagnostics.save_model_debug_artifacts
        else {},
    )


def strip_results_scenarios(
    results: BenchmarkResults,
    *,
    keep_scenarios: bool,
) -> BenchmarkResults:
    if keep_scenarios:
        return results
    return results.without_scenarios()


def merge_model_results(
    results: BenchmarkResults,
    base_model_results: dict[str, ModelResult],
) -> BenchmarkResults:
    merged: list[ModelResult] = []
    for model_result in results.model_results:
        base = copy.deepcopy(base_model_results.get(model_result.model_name, ModelResult(model_name=model_result.model_name)))
        base.metric_results = list(model_result.metric_results)
        base.metric_rankings = list(model_result.metric_rankings)
        base.average_rank = model_result.average_rank
        base.scenario_output = model_result.scenario_output
        merged.append(base)
    return BenchmarkResults(
        run=results.run,
        model_results=merged,
        reference_scenarios=results.reference_scenarios,
        diagnostics=results.diagnostics,
        metadata=results.metadata,
    )


def refresh_model_results(
    models: dict[str, object],
    results: BenchmarkResults,
) -> BenchmarkResults:
    refreshed: list[ModelResult] = []
    for model_result in results.model_results:
        name = model_result.model_name
        wrapped = models[name]
        base_model = getattr(wrapped, "model", wrapped)
        runtime_manifest = extract_model_plugin_manifest(base_model, default_name=name)
        fitted_info = base_model.model_info() if hasattr(base_model, "model_info") else None
        merged = copy.deepcopy(model_result)
        if fitted_info is not None:
            merged.fitted_model_info = fitted_info
        if runtime_manifest is not None:
            if merged.plugin_info is None:
                merged.plugin_info = PluginInfo(
                    name=name,
                    source="runtime",
                    target=name,
                    manifest=runtime_manifest,
                )
            else:
                merged.runtime_manifest = runtime_manifest
        if merged.execution is None:
            merged.execution = ModelExecutionRecord()
        if fitted_info is not None:
            runtime_device = fitted_info.get("runtime_device")
            resolved_device = fitted_info.get("resolved_device")
            if runtime_device is not None:
                merged.execution.runtime_device = str(runtime_device)
            if resolved_device is not None:
                merged.execution.resolved_device = str(resolved_device)
        refreshed.append(merged)
    return BenchmarkResults(
        run=results.run,
        model_results=refreshed,
        reference_scenarios=results.reference_scenarios,
        diagnostics=results.diagnostics,
        metadata=results.metadata,
    )
