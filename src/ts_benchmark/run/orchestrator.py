"""High-level orchestration from a benchmark JSON config."""

from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass
import logging
import multiprocessing
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

import numpy as np

from ..benchmark.definition import BenchmarkConfig
from ..benchmark.catalog import resolve_benchmark_reference
from ..benchmark.io import load_benchmark_config
from ..dataset.factory import build_dataset
from ..dataset.runtime import DatasetInstance
from ..metrics import select_metric_configs_for_run
from ..tracking import log_benchmark_run_to_mlflow
from ..model.contracts import RuntimeContext
from ..results import (
    BenchmarkResults,
    ModelDeviceAssignment,
    ModelExecutionRecord,
    ModelResult,
    ResolvedRunExecution,
    RunRecord,
)
from ..serialization import to_jsonable
from ..utils import JsonObject
from .evaluator import ScenarioBenchmark
from .execution import (
    build_run_record,
    format_devices_for_metadata,
    iso_now,
    resolve_execution_devices,
    should_parallelize_models,
)
from .model_runtime import (
    build_models,
    close_models,
    collect_model_debug_artifacts,
)
from .results_assembly import build_diagnostics, merge_model_results, refresh_model_results, strip_results_scenarios
from .storage import needs_runtime_scenarios, resolve_output_dir, save_outputs

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool


@dataclass
class BenchmarkRunArtifacts:
    config: BenchmarkConfig
    dataset: DatasetInstance
    run: RunRecord
    results: BenchmarkResults
    output_dir: Path | None


def _run_loaded_config_sequential(
    config: BenchmarkConfig,
    *,
    runtime_device: str | None,
) -> BenchmarkRunArtifacts:
    requested_at = iso_now()
    started_at = requested_at
    dataset = build_dataset(config.dataset, config.protocol, seed=config.run.seed, source_path=config.source_path)
    selected_metrics = select_metric_configs_for_run(
        config.metrics,
        has_reference_scenarios=dataset.has_reference_scenarios(),
        n_assets=int(dataset.train_returns.shape[1]),
        dataset_source=dataset.source,
    )
    models, base_model_results = build_models(config, runtime_device=runtime_device)

    try:
        benchmark = ScenarioBenchmark(
            models=models,
            protocol=config.protocol,
            metric_configs=selected_metrics,
            runtime=RuntimeContext(device=runtime_device, seed=config.run.seed),
            keep_scenarios=needs_runtime_scenarios(config),
            freq=config.dataset.freq,
        )
        raw_results = benchmark.run(dataset)
        raw_results = merge_model_results(raw_results, base_model_results)
        raw_results = refresh_model_results(models, raw_results)
        model_debug_artifacts = (
            collect_model_debug_artifacts(models) if config.diagnostics.save_model_debug_artifacts else {}
        )
        diagnostics = build_diagnostics(
            config=config,
            dataset=dataset,
            results=raw_results,
            model_debug_artifacts=model_debug_artifacts,
            selected_metrics=selected_metrics,
        )
        results = raw_results.with_metric_configs(selected_metrics)
        results.diagnostics = diagnostics
        results = strip_results_scenarios(
            results,
            keep_scenarios=(config.run.output.keep_scenarios or config.run.output.save_scenarios),
        )

        output_dir = resolve_output_dir(config)
        finished_at = iso_now()
        assigned_devices = {
            model.name: runtime_device or "cpu"
            for model in config.models
        }
        run_record = build_run_record(
            config=config,
            output_dir=output_dir,
            resolved_execution=ResolvedRunExecution(
                scheduler=config.run.scheduler,
                execution_mode="sequential",
                requested_device=config.run.device,
                resolved_devices=tuple(() if runtime_device is None else (runtime_device,)),
                assigned_devices=[
                    ModelDeviceAssignment(model_name=model_name, device=device)
                    for model_name, device in assigned_devices.items()
                ],
                parallel_workers=1,
            ),
            status="succeeded",
            requested_at=requested_at,
            started_at=started_at,
            finished_at=finished_at,
        )
        results.run = run_record
        if output_dir is not None:
            save_outputs(output_dir, config, dataset, run_record, results)

        return BenchmarkRunArtifacts(
            config=config,
            dataset=dataset,
            run=run_record,
            results=results,
            output_dir=output_dir,
        )
    finally:
        close_models(models)


def _run_single_model_worker(
    config: BenchmarkConfig,
    *,
    model_name: str,
    assigned_device: str,
) -> dict[str, object]:
    worker_config = copy.deepcopy(config)
    worker_config.models = [model for model in worker_config.models if model.name == model_name]
    worker_config.run.device = assigned_device
    worker_config.run.output.keep_scenarios = needs_runtime_scenarios(config)
    worker_config.metrics = copy.deepcopy(config.metrics)
    worker_config.run.output.output_dir = None
    worker_config.run.output.save_scenarios = False
    worker_config.diagnostics.save_distribution_summary = False
    worker_config.diagnostics.save_per_window_metrics = False
    worker_config.diagnostics.functional_smoke.enabled = False
    artifacts = _run_loaded_config_sequential(worker_config, runtime_device=assigned_device)
    model_result = next(result for result in artifacts.results.model_results if result.model_name == model_name)
    return {
        "model_name": model_name,
        "assigned_device": assigned_device,
        "model_result": model_result,
        "reference_scenarios": artifacts.results.reference_scenarios,
        "model_debug_artifacts": {}
        if artifacts.results.diagnostics is None
        else dict(artifacts.results.diagnostics.model_debug_artifacts),
    }


def _run_loaded_config_parallel(
    config: BenchmarkConfig,
    *,
    execution_devices: list[str],
) -> BenchmarkRunArtifacts:
    requested_at = iso_now()
    started_at = requested_at
    dataset = build_dataset(config.dataset, config.protocol, seed=config.run.seed, source_path=config.source_path)
    selected_metrics = select_metric_configs_for_run(
        config.metrics,
        has_reference_scenarios=dataset.has_reference_scenarios(),
        n_assets=int(dataset.train_returns.shape[1]),
        dataset_source=dataset.source,
    )
    model_names = [model.name for model in config.models]
    max_workers = min(len(execution_devices), len(model_names))
    assignments = {
        model_name: execution_devices[index % len(execution_devices)]
        for index, model_name in enumerate(model_names)
    }

    reference_scenarios: np.ndarray | None = None
    model_results_by_name: dict[str, ModelResult] = {}
    model_debug_artifacts: dict[str, object] = {}

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future_map = {
            executor.submit(
                _run_single_model_worker,
                config,
                model_name=model_name,
                assigned_device=assignments[model_name],
            ): model_name
            for model_name in model_names
        }
        model_errors: dict[str, str] = {}
        for future in concurrent.futures.as_completed(future_map):
            model_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                model_errors[model_name] = error_msg
                logger.warning("Model '%s' failed during parallel execution: %s", model_name, error_msg)
                continue
            model_result = result["model_result"]
            if reference_scenarios is None and result["reference_scenarios"] is not None:
                reference_scenarios = result["reference_scenarios"]
            if model_result.execution is None:
                model_result.execution = ModelExecutionRecord()
            model_result.execution.assigned_device = str(result["assigned_device"])
            model_results_by_name[model_name] = model_result
            if result.get("model_debug_artifacts"):
                model_debug_artifacts.update(result["model_debug_artifacts"])

    for model_name, error_msg in model_errors.items():
        model_results_by_name[model_name] = ModelResult(
            model_name=model_name,
            metadata=JsonObject({"error": error_msg}),
        )

    metadata = JsonObject(
        {
            "dataset_name": dataset.name,
            "dataset_source": dataset.source,
            "device": format_devices_for_metadata(execution_devices),
            "has_reference_scenarios": bool(dataset.has_reference_scenarios()),
            **to_jsonable(config.protocol),
            "execution_mode": "model_parallel",
        }
    )
    raw_results = BenchmarkResults.from_model_results(
        [model_results_by_name[name] for name in model_names],
        metric_configs=selected_metrics,
        reference_scenarios=reference_scenarios,
        metadata=metadata,
    )
    diagnostics = build_diagnostics(
        config=config,
        dataset=dataset,
        results=raw_results,
        model_debug_artifacts=model_debug_artifacts,
        selected_metrics=selected_metrics,
    )
    results = raw_results.with_metric_configs(selected_metrics)
    results.diagnostics = diagnostics
    results = strip_results_scenarios(
        results,
        keep_scenarios=(config.run.output.keep_scenarios or config.run.output.save_scenarios),
    )

    output_dir = resolve_output_dir(config)
    finished_at = iso_now()
    run_record = build_run_record(
        config=config,
        output_dir=output_dir,
        resolved_execution=ResolvedRunExecution(
            scheduler=config.run.scheduler,
            execution_mode="model_parallel",
            requested_device=config.run.device,
            resolved_devices=tuple(execution_devices),
            assigned_devices=[
                ModelDeviceAssignment(model_name=model_name, device=device)
                for model_name, device in assignments.items()
            ],
            parallel_workers=max_workers,
        ),
        status="partial" if model_errors else "succeeded",
        requested_at=requested_at,
        started_at=started_at,
        finished_at=finished_at,
    )
    results.run = run_record
    if output_dir is not None:
        save_outputs(output_dir, config, dataset, run_record, results)

    return BenchmarkRunArtifacts(
        config=config,
        dataset=dataset,
        run=run_record,
        results=results,
        output_dir=output_dir,
    )


def _run_loaded_config(
    config: BenchmarkConfig,
    *,
    allow_parallel: bool = True,
) -> BenchmarkRunArtifacts:
    execution_devices = resolve_execution_devices(config.run.device)
    if allow_parallel and should_parallelize_models(config, execution_devices):
        return _run_loaded_config_parallel(config, execution_devices=execution_devices)
    runtime_device = execution_devices[0] if execution_devices else config.run.device
    return _run_loaded_config_sequential(config, runtime_device=runtime_device)


def run_benchmark_from_config(
    config_or_path: BenchmarkConfig | str | Path | dict[str, object],
) -> BenchmarkRunArtifacts:
    if isinstance(config_or_path, BenchmarkConfig):
        config = config_or_path
    else:
        config = load_benchmark_config(resolve_benchmark_reference(config_or_path))
    artifacts = _run_loaded_config(config)
    tracking_result = log_benchmark_run_to_mlflow(
        config=artifacts.config,
        dataset=artifacts.dataset,
        run=artifacts.run,
        results=artifacts.results,
    )
    if tracking_result is not None:
        artifacts.run.tracking_result = tracking_result
        artifacts.results.run = artifacts.run
        if artifacts.output_dir is not None:
            save_outputs(
                artifacts.output_dir,
                artifacts.config,
                artifacts.dataset,
                artifacts.run,
                artifacts.results,
            )
    return artifacts


__all__ = [
    "BenchmarkRunArtifacts",
    "run_benchmark_from_config",
]
