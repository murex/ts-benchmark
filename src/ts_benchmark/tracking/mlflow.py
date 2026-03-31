"""Optional MLflow experiment tracking for benchmark runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..benchmark import dump_benchmark_config
from ..benchmark.definition import BenchmarkConfig
from ..benchmark.protocol import protocol_metadata_payload
from ..dataset.runtime import DatasetInstance
from ..metrics import select_metric_configs_for_run
from ..results import BenchmarkResults, RunRecord, RunTrackingRecord
from ..serialization import to_jsonable


def _import_mlflow():
    try:
        import mlflow
        from mlflow.entities import ViewType
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - exercised in runtime envs
        raise RuntimeError(
            "MLflow tracking is enabled, but the 'mlflow' package is not installed. "
            "Install it with 'pip install mlflow' or the benchmark tracking extra."
        ) from exc
    return mlflow, MlflowClient, ViewType


def mlflow_available() -> bool:
    try:
        _import_mlflow()
    except RuntimeError:
        return False
    return True


def _dataset_summary_payload(
    *,
    config: BenchmarkConfig,
    dataset: DatasetInstance,
    run: RunRecord,
    results: BenchmarkResults,
) -> dict[str, Any]:
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
            **protocol_metadata_payload(config.protocol),
            "n_eval_windows": int(dataset.contexts.shape[0]),
            "n_assets": int(dataset.train_returns.shape[1]),
            "asset_names": list(dataset.asset_names),
        },
        "metrics": [to_jsonable(metric) for metric in metric_configs],
        "run": to_jsonable(run),
        "runtime": to_jsonable(results.metadata),
        "models": [model.name for model in config.models],
    }


def _flatten_mapping(prefix: str, value: Any) -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_mapping(next_prefix, item))
        return flat
    if isinstance(value, (list, tuple)):
            flat[prefix] = json.dumps(to_jsonable(value), sort_keys=True)
            return flat
    jsonable = to_jsonable(value)
    flat[prefix] = json.dumps(jsonable) if isinstance(jsonable, (bool, type(None), dict, list)) else str(jsonable)
    return flat


def _tracking_params(config: BenchmarkConfig) -> dict[str, str]:
    params: dict[str, str] = {}
    config_dict = dump_benchmark_config(config)
    params["benchmark.version"] = str(config_dict["version"])
    params.update(_flatten_mapping("benchmark", config_dict.get("benchmark", {})))
    params.update(_flatten_mapping("run", config_dict.get("run", {})))
    for model in config.models:
        params[f"model.{model.name}.reference.kind"] = model.reference.kind
        params[f"model.{model.name}.reference.value"] = model.reference.value
        params[f"model.{model.name}.pipeline.name"] = model.pipeline.name
        params.update(_flatten_mapping(f"model.{model.name}.pipeline", to_jsonable(model.pipeline)))
        params[f"model.{model.name}.execution_mode"] = (
            "inprocess" if model.execution is None else model.execution.mode
        )
        params.update(_flatten_mapping(f"model.{model.name}.params", model.params))
    return params


def _tracking_tags(
    *,
    config: BenchmarkConfig,
    dataset: DatasetInstance,
    run: RunRecord,
    results: BenchmarkResults,
) -> dict[str, str]:
    tags = {
        "benchmark.framework": "ts-benchmark",
        "benchmark.version": str(config.version),
        "benchmark.name": config.name,
        "dataset.name": dataset.name,
        "dataset.source": dataset.source,
        "tracking.source_path": "" if config.source_path is None else str(config.source_path),
    }
    if config.description:
        tags["benchmark.description"] = config.description
    for key, value in results.metadata.items():
        tags[f"runtime.{key}"] = str(value)
    tags["run.status"] = str(run.status)
    tags.update(to_jsonable(config.run.tracking.mlflow.tags))
    return tags


def _stage_artifacts(
    *,
    root: Path,
    config: BenchmarkConfig,
    dataset: DatasetInstance,
    run: RunRecord,
    results: BenchmarkResults,
    include_model_info: bool,
    include_diagnostics: bool,
    include_scenarios: bool,
) -> None:

    results.save_metrics_csv(str(root / "metrics.csv"))
    results.save_ranks_csv(str(root / "ranks.csv"))
    (root / "benchmark_config.json").write_text(
        json.dumps(dump_benchmark_config(config), indent=2),
        encoding="utf-8",
    )
    (root / "run.json").write_text(
        json.dumps(to_jsonable(run), indent=2),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(_dataset_summary_payload(config=config, dataset=dataset, run=run, results=results), indent=2),
        encoding="utf-8",
    )

    if include_model_info:
        (root / "model_results.json").write_text(
            json.dumps(to_jsonable(results.model_results), indent=2),
            encoding="utf-8",
        )

    scenario_outputs = results.scenario_outputs()
    if include_scenarios and scenario_outputs:
        arrays = {f"model__{name}": values for name, values in scenario_outputs.items()}
        if results.reference_scenarios is not None:
            arrays["reference_scenarios"] = np.asarray(results.reference_scenarios, dtype=float)
        np.savez_compressed(root / "scenarios.npz", **arrays)

    diagnostics = results.diagnostics
    if include_diagnostics and diagnostics is not None:
        diagnostics_dir = root / "diagnostics"
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
                (debug_dir / f"{model_name}.json").write_text(
                    json.dumps(to_jsonable(payload), indent=2),
                    encoding="utf-8",
                )

def log_benchmark_run_to_mlflow(
    *,
    config: BenchmarkConfig,
    dataset: DatasetInstance,
    run: RunRecord,
    results: BenchmarkResults,
) -> RunTrackingRecord | None:
    tracking_cfg = config.run.tracking.mlflow
    if not tracking_cfg.enabled:
        return None

    mlflow, _, _ = _import_mlflow()
    if tracking_cfg.tracking_uri:
        mlflow.set_tracking_uri(tracking_cfg.tracking_uri)
    experiment = mlflow.set_experiment(tracking_cfg.experiment_name)
    run_name = (
        tracking_cfg.run_name
        or config.description
        or f"{dataset.name}:{','.join(model.name for model in config.models)}"
    )

    artifact_stage: tempfile.TemporaryDirectory[str] | None = None
    with mlflow.start_run(
        run_name=run_name,
        tags=_tracking_tags(config=config, dataset=dataset, run=run, results=results),
    ) as active_run:
        mlflow.log_params(_tracking_params(config))

        metrics_frame = results.metrics_frame().drop(columns=["average_rank"], errors="ignore").copy()
        for model_name, row in metrics_frame.iterrows():
            for metric_name, value in row.items():
                if pd.isna(value):
                    continue
                mlflow.log_metric(f"model.{model_name}.{metric_name}", float(value))

        mlflow.log_metric("summary.model_count", float(len(config.models)))
        mlflow.log_metric("summary.n_eval_windows", float(dataset.contexts.shape[0]))

        diagnostics = results.diagnostics
        if diagnostics is not None and diagnostics.functional_smoke_summary is not None:
            for _, row in diagnostics.functional_smoke_summary.iterrows():
                model_name = str(row["model"])
                mlflow.log_metric(
                    f"model.{model_name}.functional_smoke_passed",
                    1.0 if str(row["overall_status"]) == "pass" else 0.0,
                )
                mlflow.log_metric(
                    f"model.{model_name}.functional_smoke_failed_checks",
                    float(row["failed_checks"]),
                )

        if tracking_cfg.log_artifacts:
            artifact_stage = tempfile.TemporaryDirectory(prefix="tsbench-mlflow-")
            artifact_stage_path = Path(artifact_stage.name)
            _stage_artifacts(
                root=artifact_stage_path,
                config=config,
                dataset=dataset,
                run=run,
                results=results,
                include_model_info=tracking_cfg.log_model_info,
                include_diagnostics=tracking_cfg.log_diagnostics,
                include_scenarios=tracking_cfg.log_scenarios,
            )
            mlflow.log_artifacts(str(artifact_stage_path), artifact_path="benchmark_outputs")

        tracking_info = RunTrackingRecord(
            backend="mlflow",
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.name,
            run_id=active_run.info.run_id,
            run_name=run_name,
            artifact_uri=active_run.info.artifact_uri,
        )

    if artifact_stage is not None:
        artifact_stage.cleanup()
    return tracking_info


def list_mlflow_experiments(
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    _, MlflowClient, ViewType = _import_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    if hasattr(client, "search_experiments"):
        experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    else:  # pragma: no cover - older MLflow compatibility
        experiments = client.list_experiments()
    rows = [
        {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": getattr(experiment, "artifact_location", None),
            "lifecycle_stage": getattr(experiment, "lifecycle_stage", None),
        }
        for experiment in experiments
    ]
    return pd.DataFrame(rows)


def search_mlflow_runs(
    *,
    tracking_uri: str | None,
    experiment_ids: list[str],
    max_results: int = 100,
) -> pd.DataFrame:
    _, MlflowClient, ViewType = _import_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=int(max_results),
        order_by=["attributes.start_time DESC"],
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        }
        for key, value in run.data.metrics.items():
            row[f"metric.{key}"] = value
        for key, value in run.data.params.items():
            row[f"param.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def get_mlflow_run_payload(
    *,
    tracking_uri: str | None,
    run_id: str,
) -> dict[str, Any]:
    _, MlflowClient, _ = _import_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    return {
        "info": {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "lifecycle_stage": run.info.lifecycle_stage,
            "artifact_uri": run.info.artifact_uri,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        },
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def list_mlflow_artifacts(
    *,
    tracking_uri: str | None,
    run_id: str,
    artifact_path: str | None = None,
) -> pd.DataFrame:
    _, MlflowClient, _ = _import_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    entries = client.list_artifacts(run_id, path=artifact_path)
    rows = [
        {
            "path": entry.path,
            "is_dir": entry.is_dir,
            "file_size": entry.file_size,
        }
        for entry in entries
    ]
    return pd.DataFrame(rows)


def download_mlflow_artifact(
    *,
    tracking_uri: str | None,
    run_id: str,
    artifact_path: str,
) -> Path:
    mlflow, MlflowClient, _ = _import_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    if hasattr(client, "download_artifacts"):
        return Path(client.download_artifacts(run_id, artifact_path))
    return Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=tracking_uri,
        )
    )
