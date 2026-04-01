"""Load and validate benchmark configs from JSON."""

from __future__ import annotations

import json
import copy
from importlib import resources
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import jsonschema

from ..dataset.definition import (
    CsvDatasetProviderConfig,
    DatasetConfig,
    DatasetProviderConfig,
    ParquetDatasetProviderConfig,
    SyntheticDatasetProviderConfig,
    dataset_provider_from_mapping,
)
from ..metrics.definition import BUILTIN_METRIC_DEFINITIONS, resolve_metric_configs
from ..model.definition import (
    ModelConfig,
    ModelExecutionConfig,
    ModelReferenceConfig,
    PipelineConfig,
    model_params_to_builtin,
    pipeline_step_from_object,
    pipeline_step_payload,
)
from ..run.definition import (
    DiagnosticsConfig,
    FunctionalSmokeConfig,
    MlflowTrackingConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
)
from ..serialization import to_jsonable
from ..utils import JsonObject, StringMap
from .definition import BenchmarkConfig
from .protocol import Protocol, protocol_config_payload, protocol_from_mapping

BENCHMARK_OWNED_PROTOCOL_FIELDS = {
    "kind",
    "horizon",
    "n_model_scenarios",
    "n_reference_scenarios",
    "forecast",
    "unconditional_windowed",
    "unconditional_path_dataset",
    "train_size",
    "test_size",
    "context_length",
    "eval_stride",
    "train_stride",
    "n_train_paths",
    "n_realized_paths",
}


def _schema_dict() -> dict[str, Any]:
    package = "ts_benchmark.schemas"
    text = resources.files(package).joinpath("benchmark_config.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [] if not text else [text]
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    return value


def validate_benchmark_config(config_dict: dict[str, Any]) -> None:
    schema = _schema_dict()
    jsonschema.Draft202012Validator(schema).validate(config_dict)

    benchmark_block = _as_mapping(config_dict.get("benchmark"), field_name="benchmark")
    model_names = [item["name"] for item in benchmark_block.get("models", [])]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model names must be unique.")
    if not benchmark_block.get("models"):
        raise ValueError("At least one model is required.")

    dataset = _as_mapping(benchmark_block.get("dataset"), field_name="benchmark.dataset")
    provider = _as_mapping(dataset.get("provider"), field_name="benchmark.dataset.provider")
    provider_config = _as_mapping(provider.get("config"), field_name="benchmark.dataset.provider.config")
    schema_block = _as_mapping(dataset.get("schema"), field_name="benchmark.dataset.schema")
    source = provider.get("kind")
    if source == "synthetic":
        if not provider_config.get("generator"):
            raise ValueError("Synthetic datasets require dataset.provider.config.generator.")
    elif source in {"csv", "parquet"}:
        if not provider_config.get("path"):
            raise ValueError(f"{source} datasets require dataset.provider.config.path.")
        layout = str(schema_block.get("layout") or "wide")
        if layout not in {"wide", "long"}:
            raise ValueError(
                f"Tabular provider '{source}' supports dataset.schema.layout='wide' or 'long'."
            )
        if layout == "long":
            if not schema_block.get("time_column"):
                raise ValueError(f"{source} datasets with layout='long' require dataset.schema.time_column.")
            if not schema_block.get("series_id_columns"):
                raise ValueError(
                    f"{source} datasets with layout='long' require dataset.schema.series_id_columns."
                )
            if not provider_config.get("value_column"):
                raise ValueError(
                    f"{source} datasets with layout='long' require dataset.provider.config.value_column."
                )
    else:
        raise ValueError(f"Unsupported dataset.provider.kind '{source}'.")

    protocol_block = _as_mapping(benchmark_block.get("protocol"), field_name="benchmark.protocol")
    protocol = protocol_from_mapping(protocol_block)
    if protocol.kind == "unconditional_path_dataset" and source != "synthetic":
        raise ValueError(
            "protocol.kind='unconditional_path_dataset' is currently supported only for synthetic datasets."
        )

    metrics_block = benchmark_block.get("metrics")
    if metrics_block:
        has_reference = source == "synthetic"
        invalid_metrics: list[str] = []
        for metric_item in metrics_block:
            metric_name = str(metric_item.get("name", "") if isinstance(metric_item, Mapping) else getattr(metric_item, "name", "")).strip()
            if metric_name in BUILTIN_METRIC_DEFINITIONS:
                defn = BUILTIN_METRIC_DEFINITIONS[metric_name]
                reasons: list[str] = []
                if defn.requirements.synthetic_only and source != "synthetic":
                    reasons.append(f"requires synthetic dataset (got '{source}')")
                if defn.requirements.requires_reference_scenarios and not has_reference:
                    reasons.append("requires reference scenarios (only available with synthetic data)")
                if reasons:
                    invalid_metrics.append(f"{metric_name} ({'; '.join(reasons)})")
        if invalid_metrics:
            raise ValueError(
                "Metric(s) are not compatible with the configured dataset: "
                + ", ".join(invalid_metrics)
            )

    def _validate_execution_block(execution: Mapping[str, Any], *, field_prefix: str) -> None:
        mode = str(execution.get("mode", "inprocess"))
        if mode not in {"inprocess", "subprocess"}:
            raise ValueError(
                f"{field_prefix} has unsupported mode '{mode}'. Use 'inprocess' or 'subprocess'."
            )
        has_python = bool(execution.get("python"))
        has_venv = bool(execution.get("venv"))
        if has_python and has_venv:
            raise ValueError(
                f"{field_prefix} must define at most one of 'python' or 'venv'."
            )
        if mode == "inprocess" and (has_python or has_venv):
            raise ValueError(
                f"{field_prefix} sets mode='inprocess' but also specifies "
                "'python'/'venv'. Use mode='subprocess' instead."
            )

    run = _as_mapping(config_dict.get("run"), field_name="run")
    run_execution = _as_mapping(run.get("execution"), field_name="run.execution")
    run_model_execution = run_execution.get("model_execution")
    if run_model_execution is not None:
        _validate_execution_block(
            _as_mapping(run_model_execution, field_name="run.execution.model_execution"),
            field_prefix="run.execution.model_execution",
        )

    for model in benchmark_block.get("models", []):
        reference = _as_mapping(model.get("reference"), field_name=f"benchmark.models[{model['name']}].reference")
        reference_kind = str(reference.get("kind") or "").strip()
        reference_value = str(reference.get("value") or "").strip()
        if not reference_kind or not reference_value:
            raise ValueError(
                f"Model '{model['name']}' must define reference.kind and reference.value."
            )
        if reference_kind not in {"builtin", "plugin", "entrypoint"}:
            raise ValueError(
                f"Model '{model['name']}' has unsupported reference.kind '{reference_kind}'. "
                "Use 'builtin', 'plugin', or 'entrypoint'."
            )

        pipeline = _as_mapping(model.get("pipeline"), field_name=f"benchmark.models[{model['name']}].pipeline")
        pipeline_name = str(pipeline.get("name") or "").strip()
        if not pipeline_name:
            raise ValueError(f"Model '{model['name']}' must define pipeline.name.")
        steps = pipeline.get("steps")
        if steps is None or not isinstance(steps, list):
            raise ValueError(f"Model '{model['name']}' must define pipeline.steps as a list.")

        params_block = _as_mapping(model.get("params"), field_name=f"benchmark.models[{model['name']}].params")
        forbidden = sorted(BENCHMARK_OWNED_PROTOCOL_FIELDS & set(params_block))
        if forbidden:
            raise ValueError(
                f"Model '{model['name']}' duplicates benchmark-owned protocol fields in params: {forbidden}. "
                "Move them to the top-level protocol block instead."
            )

        execution = model.get("execution")
        if execution is not None:
            _validate_execution_block(
                _as_mapping(execution, field_name=f"benchmark.models[{model['name']}].execution"),
                field_prefix=f"Model '{model['name']}' execution",
            )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_config_source(config_or_path: str | Path | dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(config_or_path, (str, Path)):
        source_path = Path(config_or_path).resolve()
        return source_path, _read_json(source_path)
    return None, dict(config_or_path)


def _parse_dataset_config(dataset_block: Mapping[str, Any]) -> DatasetConfig:
    schema_block = _as_mapping(dataset_block.get("schema"), field_name="benchmark.dataset.schema")
    return DatasetConfig(
        name=dataset_block.get("name"),
        description=dataset_block.get("description"),
        provider=dataset_provider_from_mapping(
            _as_mapping(dataset_block["provider"], field_name="benchmark.dataset.provider")
        ),
        layout=str(schema_block.get("layout", "wide")),
        time_column=schema_block.get("time_column"),
        series_id_columns=_normalize_string_list(schema_block.get("series_id_columns")),
        target_columns=_normalize_string_list(schema_block.get("target_columns")),
        feature_columns=_normalize_string_list(schema_block.get("feature_columns")),
        static_feature_columns=_normalize_string_list(schema_block.get("static_feature_columns")),
        frequency=schema_block.get("frequency"),
        semantics=JsonObject(_as_mapping(dataset_block.get("semantics"), field_name="benchmark.dataset.semantics")),
        metadata=JsonObject(_as_mapping(dataset_block.get("metadata"), field_name="benchmark.dataset.metadata")),
    )


def _parse_execution_config(raw: Any, *, field_name: str) -> ModelExecutionConfig | None:
    if raw is None:
        return None
    exec_block = _as_mapping(raw, field_name=field_name)
    return ModelExecutionConfig(
        mode=str(exec_block.get("mode", "inprocess")),
        venv=exec_block.get("venv"),
        python=exec_block.get("python"),
        cwd=exec_block.get("cwd"),
        pythonpath=_normalize_string_list(exec_block.get("pythonpath")),
        env=StringMap(_as_mapping(exec_block.get("env"), field_name=f"{field_name}.env")),
    )


def _parse_model_config(item: Mapping[str, Any], *, default_execution: ModelExecutionConfig | None = None) -> ModelConfig:
    model = _as_mapping(item, field_name="benchmark.models[]")
    reference = _as_mapping(model["reference"], field_name="benchmark.models[].reference")
    params = _as_mapping(model.get("params"), field_name="benchmark.models[].params")
    pipeline = _as_mapping(model.get("pipeline"), field_name="benchmark.models[].pipeline")
    meta = _as_mapping(model.get("metadata"), field_name="benchmark.models[].metadata")
    execution = _parse_execution_config(model.get("execution"), field_name="benchmark.models[].execution")
    if execution is None and default_execution is not None and default_execution.mode == "subprocess":
        execution = copy.deepcopy(default_execution)

    return ModelConfig(
        name=str(model["name"]),
        description=model.get("description"),
        reference=ModelReferenceConfig(**reference),
        params=params,
        pipeline=PipelineConfig(
            name=str(pipeline.get("name", "raw")),
            steps=[
                pipeline_step_from_object(
                    {
                        "type": str(_as_mapping(step, field_name="benchmark.models[].pipeline.steps[]").get("type", "")),
                        "params": _as_mapping(
                            _as_mapping(step, field_name="benchmark.models[].pipeline.steps[]").get("params"),
                            field_name="benchmark.models[].pipeline.steps[].params",
                        ),
                    }
                )
                for step in list(pipeline.get("steps", []))
            ],
        ),
        metadata=JsonObject(meta),
        execution=execution,
    )


def _parse_run_config(run_block: Mapping[str, Any]) -> RunConfig:
    run = _as_mapping(run_block, field_name="run")
    execution = _as_mapping(run.get("execution"), field_name="run.execution")
    output = _as_mapping(run.get("output"), field_name="run.output")
    tracking = _as_mapping(run.get("tracking"), field_name="run.tracking")
    mlflow = _as_mapping(tracking.get("mlflow"), field_name="run.tracking.mlflow")
    return RunConfig(
        name=run.get("name"),
        description=run.get("description"),
        seed=int(run.get("seed", 7)),
        device=execution.get("device"),
        scheduler=str(execution.get("scheduler", "auto")),
        model_execution=_parse_execution_config(
            execution.get("model_execution"),
            field_name="run.execution.model_execution",
        )
        or ModelExecutionConfig(),
        tracking=TrackingConfig(
            mlflow=MlflowTrackingConfig(
                enabled=bool(mlflow.get("enabled", False)),
                tracking_uri=mlflow.get("tracking_uri"),
                experiment_name=str(mlflow.get("experiment_name", "ts-benchmark")),
                run_name=mlflow.get("run_name"),
                tags=StringMap(_as_mapping(mlflow.get("tags"), field_name="run.tracking.mlflow.tags")),
                log_artifacts=bool(mlflow.get("log_artifacts", True)),
                log_model_info=bool(mlflow.get("log_model_info", True)),
                log_diagnostics=bool(mlflow.get("log_diagnostics", True)),
                log_scenarios=bool(mlflow.get("log_scenarios", False)),
            ),
        ),
        output=OutputConfig(**output),
        metadata=JsonObject(_as_mapping(run.get("metadata"), field_name="run.metadata")),
    )


def _parse_diagnostics_config(diagnostics_block: Mapping[str, Any] | None) -> DiagnosticsConfig:
    block = _as_mapping(diagnostics_block, field_name="diagnostics")
    smoke = _as_mapping(block.get("functional_smoke"), field_name="diagnostics.functional_smoke")
    optional_smoke_threshold_keys = (
        "mean_abs_error_max",
        "std_ratio_min",
        "std_ratio_max",
        "crps_max",
        "energy_score_max",
        "cross_correlation_error_max",
    )
    if smoke.get("enabled") and smoke and all(smoke.get(key) is None for key in optional_smoke_threshold_keys):
        smoke = {
            key: value
            for key, value in smoke.items()
            if key not in optional_smoke_threshold_keys
        }
    return DiagnosticsConfig(**{**block, "functional_smoke": FunctionalSmokeConfig(**smoke)})


def _dump_dataset_config(dataset: DatasetConfig) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "description": dataset.description,
        "provider": {
            "kind": str(dataset.provider.kind),
            "config": dataset.provider.config_payload(),
        },
        "schema": {
            "layout": dataset.layout,
            "time_column": dataset.time_column,
            "series_id_columns": list(dataset.series_id_columns),
            "target_columns": list(dataset.target_columns),
            "feature_columns": list(dataset.feature_columns),
            "static_feature_columns": list(dataset.static_feature_columns),
            "frequency": dataset.frequency,
        },
        "semantics": to_jsonable(dataset.semantics),
        "metadata": to_jsonable(dataset.metadata),
    }


def _dump_model_execution_config(execution: ModelExecutionConfig | None) -> dict[str, Any] | None:
    if execution is None:
        return None
    return {
        "mode": execution.mode,
        "venv": execution.venv,
        "python": execution.python,
        "cwd": execution.cwd,
        "pythonpath": list(execution.pythonpath),
        "env": to_jsonable(execution.env),
    }


def _execution_configs_match(left: ModelExecutionConfig | None, right: ModelExecutionConfig | None) -> bool:
    return _dump_model_execution_config(left) == _dump_model_execution_config(right)


def _dump_model_config(model: ModelConfig, *, default_execution: ModelExecutionConfig | None) -> dict[str, Any]:
    payload = {
        "name": model.name,
        "description": model.description,
        "reference": to_jsonable(model.reference),
        "params": model_params_to_builtin(model.params),
        "pipeline": {
            "name": model.pipeline.name,
            "steps": [pipeline_step_payload(step) for step in model.pipeline.steps],
        },
        "metadata": to_jsonable(model.metadata),
    }
    if model.execution is not None and not _execution_configs_match(model.execution, default_execution):
        payload["execution"] = _dump_model_execution_config(model.execution)
    return payload


def dump_benchmark_config(config: BenchmarkConfig) -> dict[str, Any]:
    """Serialize a loaded benchmark config back into the external JSON schema shape."""

    return {
        "version": str(config.version),
        "benchmark": {
            "name": config.name,
            "description": config.description,
            "dataset": _dump_dataset_config(config.dataset),
            "protocol": protocol_config_payload(config.protocol),
            "metrics": to_jsonable(config.metrics),
            "models": [
                _dump_model_config(model, default_execution=config.run.model_execution)
                for model in config.models
            ],
        },
        "run": {
            "name": config.run.name,
            "description": config.run.description,
            "seed": int(config.run.seed),
            "execution": {
                "device": config.run.device,
                "scheduler": str(config.run.scheduler),
                "model_execution": _dump_model_execution_config(config.run.model_execution),
            },
            "tracking": to_jsonable(config.run.tracking),
            "output": to_jsonable(config.run.output),
            "metadata": to_jsonable(config.run.metadata),
        },
        "diagnostics": to_jsonable(config.diagnostics),
    }


def load_benchmark_config(config_or_path: str | Path | dict[str, Any]) -> BenchmarkConfig:
    source_path, config_dict = _read_config_source(config_or_path)
    validate_benchmark_config(config_dict)

    benchmark_block = _as_mapping(config_dict["benchmark"], field_name="benchmark")
    run_config = _parse_run_config(config_dict.get("run"))

    return BenchmarkConfig(
        version=str(config_dict["version"]),
        name=str(benchmark_block["name"]),
        description=benchmark_block.get("description"),
        dataset=_parse_dataset_config(benchmark_block["dataset"]),
        protocol=protocol_from_mapping(
            _as_mapping(benchmark_block["protocol"], field_name="benchmark.protocol")
        ),
        metrics=resolve_metric_configs(benchmark_block.get("metrics")),
        models=[
            _parse_model_config(item, default_execution=run_config.model_execution)
            for item in benchmark_block["models"]
        ],
        run=run_config,
        diagnostics=_parse_diagnostics_config(config_dict.get("diagnostics")),
        source_path=source_path,
    )
