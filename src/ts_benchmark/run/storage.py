"""Persistence and serialization helpers for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmark import benchmark_key_for_path
from ..benchmark import dump_benchmark_config
from ..benchmark.protocol import protocol_config_payload, protocol_metadata_payload
from ..benchmark.definition import BenchmarkConfig
from ..metrics import select_metric_configs_for_run
from ..paths import OUTPUT_DIR
from ..results import BenchmarkResults, RunRecord
from ..serialization import to_jsonable


def resolve_output_dir(config: BenchmarkConfig) -> Path | None:
    output_dir = config.run.output.output_dir
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute() and config.source_path is not None:
            path = config.source_path.parent / path
        return path.resolve()
    if config.source_path is None:
        return None
    if config.source_path.name != "benchmark.json":
        return None
    return (OUTPUT_DIR / benchmark_key_for_path(config.source_path)).resolve()


def needs_runtime_scenarios(config: BenchmarkConfig) -> bool:
    return bool(
        config.run.output.keep_scenarios
        or config.run.output.save_scenarios
        or config.diagnostics.save_distribution_summary
        or config.diagnostics.save_per_window_metrics
        or config.diagnostics.functional_smoke.enabled
    )


def dataset_summary(config: BenchmarkConfig, dataset, results: BenchmarkResults) -> dict[str, Any]:
    dataset_config = to_jsonable(config.dataset)
    metric_configs = select_metric_configs_for_run(
        config.metrics,
        has_reference_scenarios=dataset.has_reference_scenarios(),
        n_assets=int(dataset.train_returns.shape[1]),
        dataset_source=dataset.source,
    )
    return {
        "name": config.name,
        "description": config.description,
        "dataset": {
            **dataset_config,
            "resolved_name": dataset.name,
            "resolved_source": dataset.source,
            "runtime_metadata": to_jsonable(dataset.metadata),
        },
        "protocol": {
            **protocol_config_payload(config.protocol),
            **protocol_metadata_payload(config.protocol),
            "n_eval_windows": int(dataset.contexts.shape[0]),
            "n_assets": int(dataset.train_returns.shape[1]),
            "asset_names": list(dataset.asset_names),
        },
        "metrics": [to_jsonable(metric) for metric in metric_configs],
        "run": {} if results.run is None else to_jsonable(results.run),
        "runtime": to_jsonable(results.metadata),
        "models": [model.name for model in config.models],
    }


def save_outputs(
    output_dir: Path,
    config: BenchmarkConfig,
    dataset,
    run: RunRecord,
    results: BenchmarkResults,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.save_metrics_csv(str(output_dir / "metrics.csv"))
    results.save_ranks_csv(str(output_dir / "ranks.csv"))

    dumped_config = dump_benchmark_config(config)
    with (output_dir / "benchmark_config.json").open("w", encoding="utf-8") as f:
        json.dump(dumped_config, f, indent=2)
    with (output_dir / "effective_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(dumped_config, f, indent=2)

    with (output_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(run), f, indent=2)

    if config.run.output.save_model_info:
        with (output_dir / "model_results.json").open("w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(results.model_results),
                f,
                indent=2,
            )

    if config.run.output.save_summary:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(dataset_summary(config, dataset, results)), f, indent=2)

    scenario_outputs = results.scenario_outputs()
    if config.run.output.save_scenarios and scenario_outputs:
        arrays = {f"model__{name}": values for name, values in scenario_outputs.items()}
        if results.reference_scenarios is not None:
            arrays["reference_scenarios"] = np.asarray(results.reference_scenarios, dtype=float)
        np.savez_compressed(output_dir / "scenarios.npz", **arrays)

    diagnostics = results.diagnostics
    if diagnostics is not None:
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if diagnostics.distribution_summary is not None:
            diagnostics.distribution_summary.to_csv(diagnostics_dir / "distribution_summary.csv", index=False)
        if diagnostics.distribution_summary_by_asset is not None:
            diagnostics.distribution_summary_by_asset.to_csv(
                diagnostics_dir / "distribution_summary_by_asset.csv",
                index=False,
            )
        if diagnostics.per_window_metrics is not None:
            diagnostics.per_window_metrics.to_csv(diagnostics_dir / "per_window_metrics.csv", index=False)
        if diagnostics.functional_smoke_summary is not None:
            diagnostics.functional_smoke_summary.to_csv(
                diagnostics_dir / "functional_smoke_summary.csv",
                index=False,
            )
        if diagnostics.functional_smoke_checks is not None:
            diagnostics.functional_smoke_checks.to_csv(
                diagnostics_dir / "functional_smoke_checks.csv",
                index=False,
            )
        if diagnostics.model_debug_artifacts:
            debug_dir = diagnostics_dir / "model_debug_artifacts"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for model_name, payload in diagnostics.model_debug_artifacts.items():
                with (debug_dir / f"{model_name}.json").open("w", encoding="utf-8") as f:
                    json.dump(to_jsonable(payload), f, indent=2)
