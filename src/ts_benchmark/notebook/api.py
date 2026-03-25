"""Notebook-facing execution and result helpers.

This module is a thin Python API over the benchmark runtime and the
saved-result loading surface. It is intended for notebook users who want
to launch a benchmark, request only the artifacts they care about, and
inspect those artifacts through DataFrames, arrays, text reports, and
optional plots.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..benchmark import BenchmarkConfig, dump_benchmark_config, load_benchmark_config, resolve_benchmark_reference
from ..dataset.definition import DatasetConfig, DatasetProviderConfig
from ..dataset.factory import build_dataset
from ..dataset.providers.tabular import load_returns_frame
from ..model.definition import (
    ModelConfig,
    ModelExecutionConfig,
    ModelReferenceConfig,
    PipelineConfig,
    PipelineStepConfig,
)
from ..run import BenchmarkRunArtifacts, run_benchmark_from_config
from ..run.storage import dataset_summary
from ..serialization import to_jsonable
from ..utils import JsonObject, StringMap
from ..ui.services.datasets import save_dataset_definition as _save_dataset_definition
from ..ui.services.model_catalog import list_saved_catalog_models as _list_saved_catalog_models
from ..ui.services.model_catalog import save_catalog_model as _save_catalog_model
from ..ui.services.model_catalog import saved_model_paths as _saved_model_paths
from ..ui.services.runs import (
    load_run_artifacts as _load_saved_run_artifacts,
    materialize_benchmark_results as _materialize_benchmark_results,
    previous_results_dir_for_path as _previous_results_dir_for_path,
)

RESULTS_METADATA_COLUMNS = {
    "dataset_name",
    "dataset_source",
    "device",
    "has_reference_scenarios",
    "train_size",
    "test_size",
    "generation_mode",
    "context_length",
    "horizon",
    "eval_stride",
    "train_stride",
    "n_model_scenarios",
    "n_reference_scenarios",
    "execution_mode",
    "average_rank",
}

_ARTIFACT_GROUPS: dict[str, set[str]] = {
    "diagnostics": {"distribution", "per_window", "functional_smoke", "model_debug"},
    "all_diagnostics": {"distribution", "per_window", "functional_smoke", "model_debug"},
    "all": {"scenarios", "distribution", "per_window", "functional_smoke", "model_debug"},
}

_ARTIFACT_ALIASES = {
    "distribution_summary": "distribution",
    "distribution_summary_by_asset": "distribution",
    "per_window_metrics": "per_window",
    "debug_artifacts": "model_debug",
    "model_debug_artifacts": "model_debug",
}

_OFFICIAL_ADAPTER_EXTRA_BY_NAME = {
    "pytorchts_timegrad": "timegrad",
    "timegrad": "timegrad",
    "gluonts_deepvar": "gluonts-mx",
    "gluonts_gpvar": "gluonts-mx",
    "gluonts_mx": "gluonts-mx",
    "gluonts-mx": "gluonts-mx",
}


@dataclass(frozen=True)
class NotebookModelSpec:
    """Run-scoped model declaration used by the notebook wrapper.

    This is intentionally small and execution-oriented. It lets a notebook user
    inject a model into a benchmark run without editing the source benchmark JSON.
    The injected model is recorded in the effective config saved under the run
    directory, but the original config object or file remains unchanged.
    """

    name: str
    reference_kind: str
    reference_value: str
    params: Mapping[str, Any] = field(default_factory=dict)
    pipeline_name: str = "raw"
    pipeline_steps: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    execution: ModelExecutionConfig | Mapping[str, Any] | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotebookDatasetSpec:
    """Notebook-side dataset declaration for tabular CSV/Parquet sources."""

    source: str
    path: str
    name: str | None = None
    description: str | None = None
    layout: str = "wide"
    frequency: str | None = "B"
    time_column: str | None = None
    target_columns: Sequence[str] = field(default_factory=tuple)
    series_id_columns: Sequence[str] = field(default_factory=tuple)
    feature_columns: Sequence[str] = field(default_factory=tuple)
    static_feature_columns: Sequence[str] = field(default_factory=tuple)
    provider_params: Mapping[str, Any] = field(default_factory=dict)
    semantics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDatasetView:
    """Notebook-friendly tabular view over a benchmark dataset."""

    frame: pd.DataFrame
    info: dict[str, Any]


@dataclass(frozen=True)
class NotebookAdapterEnv:
    """Provisioned subprocess environment for an official adapter model."""

    adapter_name: str
    extra_name: str
    venv_dir: Path
    python_executable: Path
    benchmark_install_target: str
    adapter_install_target: str

    @property
    def execution(self) -> ModelExecutionConfig:
        return ModelExecutionConfig(mode="subprocess", venv=str(self.venv_dir))


def _coerce_execution_config(
    execution: ModelExecutionConfig | Mapping[str, Any] | None,
    *,
    default_mode: str = "inprocess",
) -> ModelExecutionConfig:
    if execution is None:
        return ModelExecutionConfig(mode=default_mode)
    if isinstance(execution, ModelExecutionConfig):
        return copy.deepcopy(execution)
    block = dict(execution)
    return ModelExecutionConfig(
        mode=str(block.get("mode", default_mode)),
        venv=block.get("venv"),
        python=block.get("python"),
        cwd=block.get("cwd"),
        pythonpath=[str(item) for item in list(block.get("pythonpath") or []) if str(item).strip()],
        env=StringMap({str(key): str(value) for key, value in dict(block.get("env") or {}).items()}),
    )


def _normalize_official_adapter_extra(adapter_name: str) -> str:
    normalized = str(adapter_name).strip().lower().replace(" ", "_")
    extra = _OFFICIAL_ADAPTER_EXTRA_BY_NAME.get(normalized)
    if extra is None:
        supported = ", ".join(sorted(_OFFICIAL_ADAPTER_EXTRA_BY_NAME))
        raise ValueError(
            f"Unsupported official adapter '{adapter_name}'. Supported adapter names: {supported}."
        )
    return extra


def _install_target_label(
    *,
    root: str | Path | None,
    package_name: str,
    extras: str | None = None,
) -> str:
    suffix = f"[{extras}]" if extras else ""
    if root is None:
        return f"{package_name}{suffix}"
    return f"{Path(root).expanduser().resolve()}{suffix}"


def _pip_install_command(
    *,
    python_executable: Path,
    root: str | Path | None,
    package_name: str,
    extras: str | None = None,
    editable: bool,
) -> list[str]:
    target = _install_target_label(root=root, package_name=package_name, extras=extras)
    command = [str(python_executable), "-m", "pip", "install"]
    if editable and root is not None:
        command.extend(["-e", target])
    else:
        command.append(target)
    return command


def _optional_dependencies_from_pyproject(root: str | Path, extra_name: str) -> list[str]:
    pyproject_path = Path(root).expanduser().resolve() / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    optional = project.get("optional-dependencies") or {}
    values = optional.get(extra_name) or []
    return [str(item) for item in values]


def _project_dependencies_from_pyproject(root: str | Path) -> list[str]:
    pyproject_path = Path(root).expanduser().resolve() / "pyproject.toml"
    if not pyproject_path.exists():
        return []
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    values = project.get("dependencies") or []
    return [str(item) for item in values]


def _worker_requirements_from_pyproject(root: str | Path) -> list[str]:
    """Return the minimal benchmark requirements needed by the subprocess worker."""

    requirements = _project_dependencies_from_pyproject(root)
    return [
        requirement
        for requirement in requirements
        if requirement.lower().startswith("jsonschema")
    ]


def _adapter_env_is_usable(venv_python: Path, adapter_name: str) -> bool:
    """Return True when the provisioned adapter env can start the worker and adapter backend."""

    normalized = str(adapter_name).strip().lower().replace(" ", "_")
    adapter_probe = ""
    if normalized in {"pytorchts_timegrad", "timegrad"}:
        adapter_probe = (
            "from ts_benchmark_official_adapters.timegrad "
            "import _ensure_gluonts_distribution_output_compat\n"
            "_ensure_gluonts_distribution_output_compat()\n"
            "from pts.model.time_grad import TimeGradEstimator\n"
        )
    elif normalized in {"gluonts_deepvar", "gluonts_gpvar", "gluonts_mx", "gluonts-mx"}:
        adapter_probe = "import gluonts\n"
    probe = (
        "import jsonschema\n"
        "import ts_benchmark.model.wrappers.worker\n"
        f"{adapter_probe}"
    )
    completed = subprocess.run(
        [str(venv_python), "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _normalize_notebook_dataset_spec(dataset: NotebookDatasetSpec | Mapping[str, Any]) -> NotebookDatasetSpec:
    spec = (
        NotebookDatasetSpec(
            source=str(dataset.get("source") or dataset.get("kind") or ""),
            path=str(dataset.get("path") or ""),
            name=None if dataset.get("name") is None else str(dataset.get("name")),
            description=None if dataset.get("description") is None else str(dataset.get("description")),
            layout=str(dataset.get("layout", "wide") or "wide"),
            frequency=None if dataset.get("frequency") is None else str(dataset.get("frequency")),
            time_column=None if dataset.get("time_column") is None else str(dataset.get("time_column")),
            target_columns=tuple(str(item) for item in list(dataset.get("target_columns") or [])),
            series_id_columns=tuple(str(item) for item in list(dataset.get("series_id_columns") or [])),
            feature_columns=tuple(str(item) for item in list(dataset.get("feature_columns") or [])),
            static_feature_columns=tuple(str(item) for item in list(dataset.get("static_feature_columns") or [])),
            provider_params=dict(dataset.get("provider_params") or dataset.get("params") or {}),
            semantics=dict(dataset.get("semantics") or {}),
            metadata=dict(dataset.get("metadata") or {}),
        )
        if isinstance(dataset, Mapping)
        else dataset
    )
    if not isinstance(spec, NotebookDatasetSpec):
        raise TypeError("with_dataset must be a NotebookDatasetSpec or mapping.")
    source = str(spec.source).strip().lower()
    if source not in {"csv", "parquet"}:
        raise ValueError(f"Notebook dataset source must be 'csv' or 'parquet', got '{spec.source}'.")
    path = str(spec.path).strip()
    if not path:
        raise ValueError("Notebook dataset specs must define a non-empty path.")
    return NotebookDatasetSpec(
        source=source,
        path=path,
        name=None if spec.name is None else str(spec.name),
        description=None if spec.description is None else str(spec.description),
        layout=str(spec.layout or "wide"),
        frequency=None if spec.frequency is None else str(spec.frequency),
        time_column=None if spec.time_column is None else str(spec.time_column),
        target_columns=tuple(str(item) for item in spec.target_columns),
        series_id_columns=tuple(str(item) for item in spec.series_id_columns),
        feature_columns=tuple(str(item) for item in spec.feature_columns),
        static_feature_columns=tuple(str(item) for item in spec.static_feature_columns),
        provider_params=dict(spec.provider_params),
        semantics=dict(spec.semantics),
        metadata=dict(spec.metadata),
    )


def _dataset_config_from_spec(spec: NotebookDatasetSpec) -> DatasetConfig:
    provider_params = dict(spec.provider_params)
    provider_params["path"] = str(spec.path)
    if spec.layout == "long" and spec.target_columns and "value_column" not in provider_params:
        provider_params["value_column"] = str(spec.target_columns[0])
    return DatasetConfig(
        name=spec.name,
        description=spec.description,
        provider=DatasetProviderConfig(
            kind=str(spec.source),
            config=JsonObject(provider_params),
        ),
        layout=str(spec.layout or "wide"),
        time_column=spec.time_column,
        series_id_columns=list(spec.series_id_columns),
        target_columns=list(spec.target_columns),
        feature_columns=list(spec.feature_columns),
        static_feature_columns=list(spec.static_feature_columns),
        frequency=spec.frequency,
        semantics=JsonObject(dict(spec.semantics)),
        metadata=JsonObject(dict(spec.metadata)),
    )


def _coerce_pipeline_config(
    *,
    pipeline_name: str = "raw",
    pipeline_steps: Sequence[Mapping[str, Any]] | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        name=str(pipeline_name or "raw"),
        steps=[
            PipelineStepConfig(
                type=str(dict(step).get("type", "")),
                params=JsonObject(dict(dict(step).get("params") or {})),
            )
            for step in list(pipeline_steps or [])
        ],
    )


def _mapping_to_notebook_model_spec(model: Mapping[str, Any]) -> NotebookModelSpec:
    reference = model.get("reference")
    entrypoint = model.get("entrypoint")
    if entrypoint is not None:
        reference_kind = "entrypoint"
        reference_value = str(entrypoint)
    else:
        if not isinstance(reference, Mapping):
            raise TypeError(
                "Notebook-injected model mappings must define either 'entrypoint' or "
                "'reference={\"kind\": ..., \"value\": ...}'."
            )
        reference_kind = str(reference.get("kind") or "").strip()
        reference_value = str(reference.get("value") or "").strip()
    return NotebookModelSpec(
        name=str(model.get("name") or ""),
        reference_kind=reference_kind,
        reference_value=reference_value,
        params=dict(model.get("params") or {}),
        pipeline_name=str(dict(model.get("pipeline") or {}).get("name", "raw")),
        pipeline_steps=list(dict(model.get("pipeline") or {}).get("steps") or []),
        execution=model.get("execution"),
        description=None if model.get("description") is None else str(model.get("description")),
        metadata=dict(model.get("metadata") or {}),
    )


def _normalize_notebook_model_spec(model: NotebookModelSpec | Mapping[str, Any]) -> NotebookModelSpec:
    spec = _mapping_to_notebook_model_spec(model) if isinstance(model, Mapping) else model
    if not isinstance(spec, NotebookModelSpec):
        raise TypeError("with_model/with_models entries must be NotebookModelSpec objects or mappings.")
    name = str(spec.name).strip()
    if not name:
        raise ValueError("Notebook-injected models must define a non-empty name.")
    reference_kind = str(spec.reference_kind).strip()
    reference_value = str(spec.reference_value).strip()
    if not reference_kind or not reference_value:
        raise ValueError(
            f"Notebook-injected model '{name}' must define a non-empty reference kind and value."
        )
    if reference_kind not in {"entrypoint", "plugin", "builtin"}:
        raise ValueError(
            f"Notebook-injected model '{name}' has unsupported reference kind '{reference_kind}'."
        )
    return NotebookModelSpec(
        name=name,
        reference_kind=reference_kind,
        reference_value=reference_value,
        params=dict(spec.params),
        pipeline_name=str(spec.pipeline_name or "raw"),
        pipeline_steps=list(spec.pipeline_steps),
        execution=_coerce_execution_config(spec.execution, default_mode="inprocess"),
        description=None if spec.description is None else str(spec.description),
        metadata=dict(spec.metadata),
    )


def _to_model_config(spec: NotebookModelSpec) -> ModelConfig:
    return ModelConfig(
        name=spec.name,
        description=spec.description,
        reference=ModelReferenceConfig(kind=spec.reference_kind, value=spec.reference_value),
        params=JsonObject(dict(spec.params)),
        pipeline=_coerce_pipeline_config(
            pipeline_name=spec.pipeline_name,
            pipeline_steps=spec.pipeline_steps,
        ),
        execution=_coerce_execution_config(spec.execution, default_mode="inprocess"),
        metadata=JsonObject(dict(spec.metadata)),
    )


def _catalog_model_payload_from_spec(spec: NotebookModelSpec) -> dict[str, Any]:
    metadata = dict(spec.metadata)
    if spec.pipeline_name != "raw" or list(spec.pipeline_steps):
        metadata.setdefault(
            "notebook_pipeline",
            {
                "name": str(spec.pipeline_name or "raw"),
                "steps": [dict(step) for step in list(spec.pipeline_steps or [])],
            },
        )
    execution = _coerce_execution_config(spec.execution, default_mode="inprocess")
    if execution.mode != "inprocess" or execution.venv or execution.python:
        metadata.setdefault(
            "notebook_execution",
            {
                "mode": execution.mode,
                "venv": execution.venv,
                "python": execution.python,
                "cwd": execution.cwd,
                "pythonpath": list(execution.pythonpath),
                "env": dict(execution.env),
            },
        )
    return {
        "name": str(spec.name),
        "description": "" if spec.description is None else str(spec.description),
        "reference": {
            "kind": str(spec.reference_kind),
            "value": str(spec.reference_value),
        },
        "params": dict(spec.params),
        "metadata": metadata,
    }


def _saved_dataset_payload_from_spec(spec: NotebookDatasetSpec) -> dict[str, Any]:
    provider_config = dict(spec.provider_params)
    provider_config["path"] = str(Path(spec.path).expanduser().resolve())
    return {
        "name": "" if spec.name is None else str(spec.name),
        "description": "" if spec.description is None else str(spec.description),
        "provider": {
            "kind": str(spec.source),
            "config": provider_config,
        },
        "schema": {
            "layout": str(spec.layout or "wide"),
            "time_column": spec.time_column,
            "series_id_columns": list(spec.series_id_columns),
            "target_columns": list(spec.target_columns),
            "feature_columns": list(spec.feature_columns),
            "static_feature_columns": list(spec.static_feature_columns),
            "frequency": spec.frequency,
        },
        "semantics": dict(spec.semantics),
        "metadata": dict(spec.metadata),
    }


def _notebook_models_to_add(
    *,
    with_model: NotebookModelSpec | Mapping[str, Any] | None,
    with_models: Sequence[NotebookModelSpec | Mapping[str, Any]] | None,
) -> list[ModelConfig]:
    requested: list[NotebookModelSpec | Mapping[str, Any]] = []
    if with_model is not None:
        requested.append(with_model)
    if with_models is not None:
        requested.extend(list(with_models))
    return [_to_model_config(_normalize_notebook_model_spec(item)) for item in requested]


def _append_notebook_models(config: BenchmarkConfig, models: Sequence[ModelConfig]) -> None:
    if not models:
        return
    existing = {model.name for model in config.models}
    for model in models:
        if model.name in existing:
            raise ValueError(
                f"Notebook-injected model '{model.name}' collides with an existing benchmark model name."
            )
        config.models.append(copy.deepcopy(model))
        existing.add(model.name)


def _strip_config_for_result_reuse(config_dict: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config_dict)
    benchmark = dict(payload.get("benchmark") or {})
    benchmark.pop("models", None)
    payload["benchmark"] = benchmark
    run = dict(payload.get("run") or {})
    run.pop("name", None)
    run.pop("description", None)
    run.pop("output", None)
    run.pop("tracking", None)
    payload["run"] = run
    payload.pop("diagnostics", None)
    return to_jsonable(payload)


def _configs_compatible_for_result_reuse(previous_config: dict[str, Any] | None, current_config: BenchmarkConfig) -> bool:
    if previous_config is None:
        return False
    return _strip_config_for_result_reuse(previous_config) == _strip_config_for_result_reuse(
        dump_benchmark_config(current_config)
    )


def _official_results_dir_for_config(config: BenchmarkConfig) -> Path | None:
    if config.source_path is None:
        return None
    return _previous_results_dir_for_path(Path(config.source_path))


def _result_model_names(payload: Mapping[str, Any]) -> set[str]:
    names = {
        str(item.get("model_name") or "").strip()
        for item in list(payload.get("model_results") or [])
        if str(item.get("model_name") or "").strip()
    }
    if names:
        return names
    summary = dict(payload.get("summary") or {})
    names.update(
        str(name).strip()
        for name in list(summary.get("models") or [])
        if str(name).strip()
    )
    metrics = payload.get("metrics")
    if isinstance(metrics, pd.DataFrame) and "model" in metrics.columns:
        names.update(
            str(name).strip()
            for name in metrics["model"].astype(str).tolist()
            if str(name).strip()
        )
    return names


def _resolve_notebook_output_path(config: BenchmarkConfig, output_dir: str | Path | None) -> Path | None:
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        if not path.is_absolute() and config.source_path is not None:
            path = Path(config.source_path).parent / path
        return path.resolve()
    configured = config.run.output.output_dir
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute() and config.source_path is not None:
        path = Path(config.source_path).parent / path
    return path.resolve()


def _temp_notebook_dir(prefix: str, *, parent: Path | None = None) -> Path:
    root = None if parent is None else str(parent)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root)).resolve()


def _execute_with_reused_results(
    *,
    effective: BenchmarkConfig,
    forced_model_names: set[str],
    previous_results_dir: Path,
    output_dir: Path | None,
) -> NotebookRun | None:
    previous_payload = _load_saved_run_artifacts(previous_results_dir)
    previous_config = previous_payload.get("config")
    if not isinstance(previous_config, dict):
        return None
    if not _configs_compatible_for_result_reuse(previous_config, effective):
        return None

    previous_model_names = _result_model_names(previous_payload)
    models_to_run = [
        copy.deepcopy(model)
        for model in effective.models
        if model.name in forced_model_names or model.name not in previous_model_names
    ]
    if not models_to_run:
        return NotebookRun.from_saved_run(previous_results_dir)
    if len(models_to_run) == len(effective.models):
        return None

    output_parent = output_dir.parent if output_dir is not None else None
    raw_run_dir = _temp_notebook_dir("notebook_run_", parent=output_parent)
    merged_dir = output_dir or _temp_notebook_dir("notebook_results_", parent=output_parent)

    execution_config = copy.deepcopy(effective)
    execution_config.models = models_to_run
    execution_config.run.output.output_dir = str(raw_run_dir)

    try:
        run_benchmark_from_config(execution_config)
        merged_dir = _materialize_benchmark_results(
            benchmark_path=Path(effective.source_path),
            benchmark_config=dump_benchmark_config(effective),
            source_run_dir=raw_run_dir,
            previous_results_dir=previous_results_dir,
            destination_dir=merged_dir,
        )
        return NotebookRun.from_saved_run(merged_dir)
    finally:
        if raw_run_dir.exists():
            shutil.rmtree(raw_run_dir, ignore_errors=True)


def entrypoint_model(
    name: str,
    entrypoint: str,
    *,
    params: Mapping[str, Any] | None = None,
    pipeline: str = "raw",
    steps: Sequence[Mapping[str, Any]] | None = None,
    execution: ModelExecutionConfig | Mapping[str, Any] | None = None,
    description: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **param_overrides: Any,
) -> NotebookModelSpec:
    """Create a run-scoped entrypoint model for notebook execution.

    By default the injected model is forced to run in-process in the current
    notebook environment, regardless of the benchmark's run-level model
    execution defaults.
    """

    merged_params = dict(params or {})
    merged_params.update(param_overrides)
    return NotebookModelSpec(
        name=str(name),
        reference_kind="entrypoint",
        reference_value=str(entrypoint),
        params=merged_params,
        pipeline_name=str(pipeline or "raw"),
        pipeline_steps=list(steps or []),
        execution=_coerce_execution_config(execution, default_mode="inprocess"),
        description=description,
        metadata=dict(metadata or {}),
    )


def save_model_to_catalog(
    model: NotebookModelSpec | Mapping[str, Any],
    *,
    model_dir: str | Path | None = None,
) -> Path:
    """Persist a notebook model into the Streamlit model catalog.

    If the same model reference is already cataloged, the existing catalog path
    is returned so notebook promotion remains idempotent.
    """

    spec = _normalize_notebook_model_spec(model)
    payload = _catalog_model_payload_from_spec(spec)
    target_dir = None if model_dir is None else Path(model_dir).expanduser().resolve()
    try:
        if target_dir is None:
            return _save_catalog_model(payload)
        return _save_catalog_model(payload, model_dir=target_dir)
    except ValueError as exc:
        reference = dict(payload.get("reference") or {})
        ref_kind = str(reference.get("kind") or "").strip()
        ref_value = str(reference.get("value") or "").strip()
        existing_entries = (
            _list_saved_catalog_models()
            if target_dir is None
            else _list_saved_catalog_models(model_dir=target_dir)
        )
        for entry in existing_entries:
            entry_ref = dict(entry.get("reference") or {})
            if str(entry_ref.get("kind") or "").strip() != ref_kind:
                continue
            if str(entry_ref.get("value") or "").strip() != ref_value:
                continue
            paths = _saved_model_paths() if target_dir is None else _saved_model_paths(model_dir=target_dir)
            for path in paths.values():
                try:
                    existing_payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(existing_payload.get("name") or "").strip() == str(entry.get("name") or "").strip():
                    return path.resolve()
        raise exc


def provision_adapter_venv(
    folder: str | Path,
    adapter_name: str,
    *,
    benchmark_root: str | Path | None = None,
    adapters_root: str | Path | None = None,
    python: str | Path | None = None,
    editable: bool = True,
    upgrade_pip: bool = True,
    reinstall: bool = False,
    validate: bool = False,
) -> NotebookAdapterEnv:
    """Create a subprocess venv for an official adapter and install its dependencies.

    This is intended for notebook users who want to keep heavyweight adapter
    dependencies out of the main notebook environment. The returned object
    exposes `execution`, which can be attached to a benchmark model config.
    Existing environments are reused by default; pass ``reinstall=True`` to
    refresh the environment in place. Pass ``validate=True`` to run an
    import-based health check before reusing an existing environment.
    """

    extra_name = _normalize_official_adapter_extra(adapter_name)
    venv_dir = Path(folder).expanduser().resolve()
    creator_python = (
        Path(python).expanduser().resolve()
        if python is not None
        else Path(sys.executable).resolve()
    )
    venv_python = venv_dir / "bin" / "python"
    marker_path = venv_dir / ".ts_benchmark_adapter_env.json"
    existed_before = venv_python.exists()
    reusable = False
    if existed_before and marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            reusable = (
                str(marker.get("adapter_name") or "") == str(adapter_name)
                and str(marker.get("extra_name") or "") == extra_name
                and str(marker.get("creator_python") or "") == str(creator_python)
            )
        except Exception:
            reusable = False

    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
        existed_before = False
        reusable = False

    if not reinstall and existed_before and not reusable and venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
        existed_before = False

    if not venv_python.exists():
        subprocess.run([str(creator_python), "-m", "venv", str(venv_dir)], check=True)
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Expected a Python executable at '{venv_python}' after creating the adapter venv."
        )

    if not reinstall and existed_before and reusable:
        if not validate:
            return NotebookAdapterEnv(
                adapter_name=str(adapter_name),
                extra_name=extra_name,
                venv_dir=venv_dir,
                python_executable=venv_python,
                benchmark_install_target=_install_target_label(
                    root=benchmark_root,
                    package_name="ts-benchmark",
                ),
                adapter_install_target=_install_target_label(
                    root=adapters_root,
                    package_name="ts-benchmark-official-adapters",
                    extras=extra_name,
                ),
            )
        if _adapter_env_is_usable(venv_python, adapter_name):
            return NotebookAdapterEnv(
                adapter_name=str(adapter_name),
                extra_name=extra_name,
                venv_dir=venv_dir,
                python_executable=venv_python,
                benchmark_install_target=_install_target_label(
                    root=benchmark_root,
                    package_name="ts-benchmark",
                ),
                adapter_install_target=_install_target_label(
                    root=adapters_root,
                    package_name="ts-benchmark-official-adapters",
                    extras=extra_name,
                ),
            )
        shutil.rmtree(venv_dir, ignore_errors=True)
        subprocess.run([str(creator_python), "-m", "venv", str(venv_dir)], check=True)
        if not venv_python.exists():
            raise FileNotFoundError(
                f"Expected a Python executable at '{venv_python}' after recreating the adapter venv."
            )

    if upgrade_pip:
        subprocess.run([str(venv_python), "-m", "pip", "install", "-U", "pip"], check=True)

    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            *(
                ["-e", str(Path(benchmark_root).expanduser().resolve())]
                if benchmark_root is not None and editable
                else [_install_target_label(root=benchmark_root, package_name="ts-benchmark")]
            ),
            "--no-deps",
        ],
        check=True,
    )
    if benchmark_root is not None and editable:
        benchmark_requirements = _worker_requirements_from_pyproject(benchmark_root)
        if benchmark_requirements:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", *benchmark_requirements],
                check=True,
            )
    if adapters_root is not None and editable:
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "-e",
                str(Path(adapters_root).expanduser().resolve()),
                "--no-deps",
            ],
            check=True,
        )
        extra_requirements = _optional_dependencies_from_pyproject(adapters_root, extra_name)
        base_requirements = [
            requirement
            for requirement in _project_dependencies_from_pyproject(adapters_root)
            if not requirement.lower().startswith("ts-benchmark")
        ]
        combined_requirements = [*base_requirements, *extra_requirements]
        if combined_requirements:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", *combined_requirements],
                check=True,
            )
    else:
        subprocess.run(
            _pip_install_command(
                python_executable=venv_python,
                root=adapters_root,
                package_name="ts-benchmark-official-adapters",
                extras=extra_name,
                editable=editable,
            ),
            check=True,
        )

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "adapter_name": str(adapter_name),
                "extra_name": extra_name,
                "creator_python": str(creator_python),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return NotebookAdapterEnv(
        adapter_name=str(adapter_name),
        extra_name=extra_name,
        venv_dir=venv_dir,
        python_executable=venv_python,
        benchmark_install_target=_install_target_label(
            root=benchmark_root,
            package_name="ts-benchmark",
        ),
        adapter_install_target=_install_target_label(
            root=adapters_root,
            package_name="ts-benchmark-official-adapters",
            extras=extra_name,
        ),
    )


def tabular_dataset(
    path: str | Path,
    *,
    source: str,
    name: str | None = None,
    description: str | None = None,
    layout: str = "wide",
    frequency: str | None = "B",
    time_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    series_id_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    static_feature_columns: Sequence[str] | None = None,
    semantics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provider_params: Mapping[str, Any] | None = None,
    **provider_param_overrides: Any,
) -> NotebookDatasetSpec:
    """Create a tabular notebook dataset spec for CSV/Parquet files."""

    merged_provider_params = dict(provider_params or {})
    merged_provider_params.update(provider_param_overrides)
    return NotebookDatasetSpec(
        source=str(source),
        path=str(Path(path).expanduser()),
        name=name,
        description=description,
        layout=str(layout or "wide"),
        frequency=frequency,
        time_column=time_column,
        target_columns=tuple(target_columns or ()),
        series_id_columns=tuple(series_id_columns or ()),
        feature_columns=tuple(feature_columns or ()),
        static_feature_columns=tuple(static_feature_columns or ()),
        provider_params=merged_provider_params,
        semantics=dict(semantics or {}),
        metadata=dict(metadata or {}),
    )


def csv_dataset(path: str | Path, **kwargs: Any) -> NotebookDatasetSpec:
    """Create a CSV-backed notebook dataset spec."""

    return tabular_dataset(path, source="csv", **kwargs)


def parquet_dataset(path: str | Path, **kwargs: Any) -> NotebookDatasetSpec:
    """Create a Parquet-backed notebook dataset spec."""

    return tabular_dataset(path, source="parquet", **kwargs)


def save_dataset_definition(
    dataset: NotebookDatasetSpec | Mapping[str, Any],
    *,
    dataset_dir: str | Path | None = None,
) -> Path:
    """Persist a notebook dataset spec into the saved-dataset catalog format."""

    spec = _normalize_notebook_dataset_spec(dataset)
    payload = _saved_dataset_payload_from_spec(spec)
    if dataset_dir is None:
        return _save_dataset_definition(payload)
    return _save_dataset_definition(payload, dataset_dir=Path(dataset_dir).expanduser().resolve())


def _normalize_artifact_name(name: object) -> str:
    text = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ARTIFACT_ALIASES.get(text, text)


def _normalize_artifact_requests(include: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    for item in include or []:
        artifact = _normalize_artifact_name(item)
        if not artifact:
            continue
        normalized.update(_ARTIFACT_GROUPS.get(artifact, {artifact}))
    return normalized


def _override_requested_artifacts(config: BenchmarkConfig, requested: set[str]) -> None:
    # Notebook workflows expect rich inspection surfaces by default.
    config.run.output.save_model_info = True
    config.run.output.save_summary = True

    if "scenarios" in requested:
        config.run.output.keep_scenarios = True
        config.run.output.save_scenarios = True
    if "distribution" in requested:
        config.diagnostics.save_distribution_summary = True
    if "per_window" in requested:
        config.diagnostics.save_per_window_metrics = True
    if "model_debug" in requested:
        config.diagnostics.save_model_debug_artifacts = True
    if "functional_smoke" in requested:
        config.diagnostics.functional_smoke.enabled = True


def _dataset_values_for_split(dataset: object, split: str) -> np.ndarray:
    key = str(split or "full").strip().lower()
    if key == "full":
        return np.asarray(getattr(dataset, "full_returns"), dtype=float)
    if key == "train":
        return np.asarray(getattr(dataset, "train_returns"), dtype=float)
    if key == "test":
        return np.asarray(getattr(dataset, "test_returns"), dtype=float)
    raise ValueError("split must be one of 'full', 'train', or 'test'.")


def _source_dataset_frame_view(spec: NotebookDatasetSpec) -> BenchmarkDatasetView:
    dataset_config = _dataset_config_from_spec(spec)
    provider_params = dataset_config.provider.config.to_builtin()
    if dataset_config.layout == "wide" and dataset_config.target_columns:
        provider_params["asset_columns"] = list(dataset_config.target_columns)
    if dataset_config.time_column:
        provider_params["date_column"] = dataset_config.time_column
    if dataset_config.layout == "long" and dataset_config.series_id_columns:
        provider_params["series_id_columns"] = list(dataset_config.series_id_columns)
    if dataset_config.semantics.get("target_kind") is not None and provider_params.get("value_type") is None:
        provider_params["value_type"] = dataset_config.semantics.get("target_kind")
    if dataset_config.semantics.get("return_kind") is not None and provider_params.get("return_kind") is None:
        provider_params["return_kind"] = dataset_config.semantics.get("return_kind")
    returns_frame, timestamps, loader_metadata = load_returns_frame(
        path=spec.path,
        source=spec.source,
        params=provider_params,
    )
    frame = returns_frame.copy()
    if timestamps is not None:
        frame.insert(0, "__timestamp__", pd.to_datetime(timestamps).reset_index(drop=True))
    info = {
        "name": spec.name or Path(spec.path).stem,
        "source": spec.source,
        "description": spec.description,
        "layout": dataset_config.layout,
        "freq": dataset_config.freq,
        "path": str(Path(spec.path).expanduser().resolve()),
        "n_rows": int(len(returns_frame)),
        "n_assets": int(returns_frame.shape[1]),
        "asset_names": list(returns_frame.columns),
        "provider": {
            "kind": spec.source,
            "config": to_jsonable(provider_params),
        },
        "semantics": to_jsonable(dataset_config.semantics),
        "metadata": {
            **to_jsonable(dataset_config.metadata),
            **to_jsonable(loader_metadata),
        },
    }
    frame.index.name = "row"
    return BenchmarkDatasetView(frame=frame, info=info)


def _dataset_info_payload(config: BenchmarkConfig, dataset: object, *, split: str) -> dict[str, Any]:
    provider = config.dataset.provider
    payload = {
        "name": str(getattr(dataset, "name")),
        "source": str(getattr(dataset, "source")),
        "split": str(split),
        "freq": str(getattr(dataset, "freq")),
        "asset_names": list(getattr(dataset, "asset_names")),
        "n_rows": int(_dataset_values_for_split(dataset, split).shape[0]),
        "n_assets": int(_dataset_values_for_split(dataset, split).shape[1]),
        "protocol": to_jsonable(config.protocol),
        "provider": {
            "kind": str(provider.kind),
            "config": to_jsonable(provider.config),
        },
        "metadata": to_jsonable(getattr(dataset, "metadata")),
    }
    if provider.kind == "synthetic":
        payload["synthetic"] = {
            "generator": provider.config.get("generator"),
            "params": to_jsonable(provider.config.get("params") or {}),
            "n_points_to_generate": int(config.protocol.train_size) + int(config.protocol.test_size),
        }
    return payload


def _dataset_frame_view(config: BenchmarkConfig, dataset: object, *, split: str = "full") -> BenchmarkDatasetView:
    values = _dataset_values_for_split(dataset, split)
    frame = pd.DataFrame(values, columns=list(getattr(dataset, "asset_names")))
    frame.index.name = "step"
    return BenchmarkDatasetView(
        frame=frame,
        info=_dataset_info_payload(config, dataset, split=split),
    )


def dataset_frame(
    config_or_path: BenchmarkConfig | str | Path | dict[str, Any] | NotebookDatasetSpec,
    *,
    split: str = "full",
) -> BenchmarkDatasetView:
    """Return a pandas view for either a benchmark dataset or a raw tabular dataset.

    - When given a benchmark config/path, this builds the benchmark dataset and
      returns the selected split (`full`, `train`, or `test`).
    - When given a ``NotebookDatasetSpec``, this loads the raw CSV/Parquet
      returns table after benchmark-compatible value conversion.

    For synthetic benchmark datasets, the returned info payload exposes the
    generator name, generator parameters, and the total number of points
    implied by the benchmark protocol (`train_size + test_size`).
    """

    if isinstance(config_or_path, NotebookDatasetSpec):
        return _source_dataset_frame_view(_normalize_notebook_dataset_spec(config_or_path))
    if isinstance(config_or_path, BenchmarkConfig):
        config = config_or_path
    else:
        config = load_benchmark_config(resolve_benchmark_reference(config_or_path))
    dataset = build_dataset(
        config.dataset,
        config.protocol,
        seed=config.run.seed,
        source_path=config.source_path,
    )
    return _dataset_frame_view(config, dataset, split=split)


def _normalize_results_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    out = frame.copy()
    if "model" in out.columns:
        out = out.set_index("model")
    out.index = out.index.map(str)
    out.index.name = "model"
    return out


def _copy_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    return None if frame is None else frame.copy(deep=True)


def _normalize_diagnostic_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    out = frame.copy(deep=True)
    if "evaluation_timestamp" in out.columns:
        out["evaluation_timestamp"] = out["evaluation_timestamp"].astype(object).where(
            out["evaluation_timestamp"].notna(),
            None,
        )
    return out


def _copy_array_map(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {str(key): np.array(value, dtype=float, copy=True) for key, value in values.items()}


def _copy_diagnostics_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            payload[key] = _normalize_diagnostic_frame(value)
        else:
            payload[key] = copy.deepcopy(value)
    return payload


def _diagnostics_from_live_run(artifacts: BenchmarkRunArtifacts) -> dict[str, Any]:
    diagnostics = artifacts.results.diagnostics
    if diagnostics is None:
        return {}
    return {
        "distribution_summary": _normalize_diagnostic_frame(diagnostics.distribution_summary),
        "distribution_summary_by_asset": _normalize_diagnostic_frame(diagnostics.distribution_summary_by_asset),
        "per_window_metrics": _normalize_diagnostic_frame(diagnostics.per_window_metrics),
        "functional_smoke_summary": _normalize_diagnostic_frame(diagnostics.functional_smoke_summary),
        "functional_smoke_checks": _normalize_diagnostic_frame(diagnostics.functional_smoke_checks),
        "model_debug_artifacts": copy.deepcopy(diagnostics.model_debug_artifacts),
    }


def _metric_names(summary: dict[str, Any], metrics: pd.DataFrame | None) -> list[str]:
    names = [
        str(metric.get("name") or "").strip()
        for metric in list(summary.get("metrics") or [])
        if str(metric.get("name") or "").strip()
    ]
    if names:
        return names
    if metrics is None:
        return []
    return [
        str(column)
        for column in metrics.columns
        if str(column) not in RESULTS_METADATA_COLUMNS
    ]


def _model_config_entry(config: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    benchmark = dict(config.get("benchmark") or {})
    for model in benchmark.get("models") or []:
        if str(model.get("name") or "").strip() == model_name:
            return dict(model)
    return None


def _metrics_row_for_model(metrics: pd.DataFrame | None, model_name: str) -> dict[str, object]:
    if metrics is None or model_name not in metrics.index:
        return {}
    row = metrics.loc[model_name]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.replace({np.nan: None}).to_dict()


def _filter_frame_by_model(frame: pd.DataFrame | None, model_name: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    if "model" not in frame.columns:
        return frame.copy(deep=True)
    subset = frame[frame["model"].astype(str) == model_name].copy()
    if subset.empty:
        return None
    return subset.reset_index(drop=True)


def _diagnostics_for_model(diagnostics: dict[str, Any], model_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in [
        "distribution_summary",
        "distribution_summary_by_asset",
        "per_window_metrics",
        "functional_smoke_summary",
        "functional_smoke_checks",
    ]:
        filtered = _filter_frame_by_model(diagnostics.get(key), model_name)
        if filtered is not None and not filtered.empty:
            payload[key] = filtered
    debug_artifacts = dict(diagnostics.get("model_debug_artifacts") or {})
    if model_name in debug_artifacts:
        payload["model_debug_artifacts"] = copy.deepcopy(debug_artifacts[model_name])
    return payload


def _scenario_band_dataframe(samples: np.ndarray, realized: np.ndarray, asset_index: int) -> pd.DataFrame:
    asset_samples = np.asarray(samples[:, :, asset_index], dtype=float)
    q05, q50, q95 = np.quantile(asset_samples, [0.05, 0.50, 0.95], axis=0)
    frame = pd.DataFrame(
        {
            "realized": realized[:, asset_index],
            "p05": q05,
            "median": q50,
            "p95": q95,
        }
    )
    frame.index.name = "step"
    return frame


def _model_overview_frame(
    *,
    summary: dict[str, Any],
    metrics: pd.DataFrame | None,
    model_results: list[dict[str, Any]],
) -> pd.DataFrame | None:
    frame = None if metrics is None else metrics.reset_index().copy()
    metric_names = _metric_names(summary, metrics)

    extras: list[dict[str, Any]] = []
    for item in model_results:
        name = str(item.get("model_name") or "").strip()
        if not name:
            continue
        reference = dict(item.get("reference") or {})
        pipeline = item.get("pipeline")
        plugin_info = dict(item.get("plugin_info") or {})
        manifest = dict(plugin_info.get("manifest") or {})
        metadata = dict(item.get("metadata") or {})
        error = str(metadata.get("error") or "").strip()
        extras.append(
            {
                "model": name,
                "display_name": manifest.get("display_name") or "",
                "reference": f"{reference.get('kind')}:{reference.get('value')}" if reference else "",
                "pipeline": pipeline.get("name") if isinstance(pipeline, dict) else (pipeline or ""),
                "status": "Failed" if error else "Completed",
                "error": error,
            }
        )

    if frame is None or frame.empty:
        if not extras:
            return None
        frame = pd.DataFrame(extras)
    elif extras:
        frame = pd.DataFrame(extras).merge(frame, on="model", how="outer")

    preferred = ["model"]
    if "display_name" in frame.columns and any(
        str(value).strip() and str(value).strip() != str(model).strip()
        for value, model in zip(frame["display_name"], frame["model"])
    ):
        preferred.append("display_name")
    if "status" in frame.columns and frame["status"].fillna("").astype(str).str.strip().any():
        preferred.append("status")
    if "reference" in frame.columns and frame["reference"].fillna("").astype(str).str.strip().any():
        preferred.append("reference")
    if "pipeline" in frame.columns and frame["pipeline"].fillna("").astype(str).str.strip().any():
        preferred.append("pipeline")
    preferred.extend([name for name in metric_names if name in frame.columns])
    if "average_rank" in frame.columns:
        preferred.append("average_rank")
    if "error" in frame.columns and frame["error"].fillna("").astype(str).str.strip().any():
        preferred.append("error")
    overview = frame[preferred].copy()
    if "average_rank" in overview.columns:
        overview = overview.sort_values("average_rank", ascending=True, na_position="last")
    return overview.reset_index(drop=True)


def _format_json_block(value: object) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=True)


def _format_dataframe_block(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "Unavailable"
    cleaned = frame.replace({np.nan: None})
    return cleaned.to_string(index=False)


def _format_array_block(array: object) -> str:
    if array is None:
        return "Unavailable"
    values = np.asarray(array)
    return (
        f"shape: {values.shape}\n"
        + np.array2string(
            values,
            precision=6,
            threshold=max(int(values.size), 1),
            max_line_width=160,
        )
    )


def _extract_model_log_text(diagnostics: dict[str, Any], model_name: str) -> str:
    debug_artifacts = dict(diagnostics.get("model_debug_artifacts") or {})
    model_debug = dict(debug_artifacts.get(model_name) or {})
    wrapped = dict(model_debug.get("wrapped_debug_artifacts") or {})
    training_log = wrapped.get("training_log")
    if training_log is None and "training_log" in model_debug:
        training_log = model_debug.get("training_log")
    if training_log is None:
        return "No model training log was saved under wrapped_debug_artifacts.training_log."
    if isinstance(training_log, str):
        return training_log
    return _format_json_block(training_log)


def _section(title: str, body: str) -> str:
    return f"{title}\n{'=' * len(title)}\n{body.strip()}\n"


def _build_model_debug_report(
    *,
    model_name: str,
    benchmark_name: str,
    config: dict[str, Any],
    metrics: pd.DataFrame | None,
    model_results: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    dataset: object | None,
    generated_scenarios: dict[str, np.ndarray],
    reference_scenarios: np.ndarray | None,
) -> str:
    model_result = next(
        (dict(item) for item in model_results if str(item.get("model_name") or "").strip() == model_name),
        {},
    )
    model_config = _model_config_entry(config, model_name) or {}
    model_diagnostics = _diagnostics_for_model(diagnostics, model_name)
    generated = generated_scenarios.get(model_name)
    training_payload = {
        "train_returns": None if dataset is None else np.asarray(getattr(dataset, "train_returns")),
        "contexts": None if dataset is None else np.asarray(getattr(dataset, "contexts")),
        "realized_futures": None if dataset is None else np.asarray(getattr(dataset, "realized_futures")),
        "reference_scenarios": reference_scenarios,
    }
    metric_payload = {
        "aggregated_metrics": _metrics_row_for_model(metrics, model_name),
        "metric_results": model_result.get("metric_results") or [],
        "metric_rankings": model_result.get("metric_rankings") or [],
        "average_rank": model_result.get("average_rank"),
    }
    sections = [
        _section(
            "Benchmark",
            _format_json_block(
                {
                    "name": benchmark_name,
                    "protocol": dict(config.get("benchmark", {}).get("protocol") or {}),
                }
            ),
        ),
        _section(
            "Model Hyperparameters",
            _format_json_block(
                {
                    "name": model_name,
                    "reference": model_config.get("reference") or model_result.get("reference") or {},
                    "params": model_result.get("params") or model_config.get("params") or {},
                    "pipeline": model_result.get("pipeline") or model_config.get("pipeline") or {},
                    "execution": model_result.get("execution") or {},
                }
            ),
        ),
        _section(
            "Diagnostics",
            _format_json_block(
                {
                    key: (
                        json.loads(value.replace({np.nan: None}).to_json(orient="records"))
                        if isinstance(value, pd.DataFrame)
                        else value
                    )
                    for key, value in model_diagnostics.items()
                }
            ),
        ),
        _section("Metrics", _format_json_block(metric_payload)),
        _section(
            "Training Scenarios",
            "\n\n".join(
                [
                    f"{label}\n{_format_array_block(value)}"
                    for label, value in training_payload.items()
                ]
            ),
        ),
        _section("Generated Scenarios", _format_array_block(generated)),
        _section("Model Logs", _extract_model_log_text(diagnostics, model_name)),
    ]
    return "\n".join(sections)


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - exercised only when matplotlib is missing
        raise ImportError(
            "Notebook plotting requires matplotlib. Install the optional notebook extras "
            "or add matplotlib to your environment."
        ) from exc
    return plt


def show(value: object, *, title: str | None = None) -> object:
    """Display a value in IPython when available and return it unchanged."""

    try:  # pragma: no cover - display behavior is notebook-specific
        from IPython.display import Markdown, display
    except Exception:
        return value
    if title:
        display(Markdown(f"### {title}"))
    display(value)
    return value


@dataclass
class NotebookRun:
    """Notebook-native view of a benchmark run."""

    config_payload: dict[str, Any]
    run_payload: dict[str, Any] | None
    summary_payload: dict[str, Any]
    metrics_payload: pd.DataFrame | None
    ranks_payload: pd.DataFrame | None
    model_results_payload: list[dict[str, Any]] = field(default_factory=list)
    diagnostics_payload: dict[str, Any] = field(default_factory=dict)
    generated_scenarios_payload: dict[str, np.ndarray] = field(default_factory=dict)
    reference_scenarios_payload: np.ndarray | None = None
    dataset_obj: object | None = None
    dataset_error_text: str | None = None
    output_dir: Path | None = None

    @classmethod
    def from_benchmark_run(cls, artifacts: BenchmarkRunArtifacts) -> "NotebookRun":
        output_dir = None if artifacts.output_dir is None else Path(artifacts.output_dir).resolve()
        return cls(
            config_payload=dump_benchmark_config(artifacts.config),
            run_payload=to_jsonable(artifacts.run),
            summary_payload=to_jsonable(dataset_summary(artifacts.config, artifacts.dataset, artifacts.results)),
            metrics_payload=_normalize_results_frame(
                artifacts.results.metrics_frame(include_metadata=True).reset_index()
            ),
            ranks_payload=_normalize_results_frame(
                artifacts.results.rankings_frame(include_metadata=True).reset_index()
            ),
            model_results_payload=to_jsonable(artifacts.results.model_results),
            diagnostics_payload=_diagnostics_from_live_run(artifacts),
            generated_scenarios_payload=_copy_array_map(artifacts.results.scenario_outputs()),
            reference_scenarios_payload=(
                None
                if artifacts.results.reference_scenarios is None
                else np.array(artifacts.results.reference_scenarios, dtype=float, copy=True)
            ),
            dataset_obj=artifacts.dataset,
            dataset_error_text=None,
            output_dir=output_dir,
        )

    @classmethod
    def from_saved_run(cls, run_dir: str | Path) -> "NotebookRun":
        payload = _load_saved_run_artifacts(Path(run_dir))
        return cls(
            config_payload=dict(payload.get("config") or {}),
            run_payload=None if payload.get("run") is None else dict(payload.get("run") or {}),
            summary_payload=dict(payload.get("summary") or {}),
            metrics_payload=_normalize_results_frame(payload.get("metrics")),
            ranks_payload=_normalize_results_frame(payload.get("ranks")),
            model_results_payload=copy.deepcopy(payload.get("model_results") or []),
            diagnostics_payload=_copy_diagnostics_payload(dict(payload.get("diagnostics") or {})),
            generated_scenarios_payload=_copy_array_map(dict(payload.get("generated_scenarios") or {})),
            reference_scenarios_payload=(
                None
                if payload.get("reference_scenarios") is None
                else np.array(payload.get("reference_scenarios"), dtype=float, copy=True)
            ),
            dataset_obj=payload.get("dataset"),
            dataset_error_text=payload.get("dataset_error"),
            output_dir=Path(payload["run_dir"]).resolve(),
        )

    def config(self) -> dict[str, Any]:
        return copy.deepcopy(self.config_payload)

    def run_record(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.run_payload)

    def summary(self) -> dict[str, Any]:
        return copy.deepcopy(self.summary_payload)

    def dataset(self) -> object | None:
        return self.dataset_obj

    def dataset_error(self) -> str | None:
        return None if self.dataset_error_text is None else str(self.dataset_error_text)

    def dataset_frame(self, *, split: str = "full") -> BenchmarkDatasetView:
        if self.dataset_obj is None:
            if self.dataset_error_text:
                raise RuntimeError(
                    f"Dataset reconstruction failed for this run, so a pandas dataset view is unavailable: "
                    f"{self.dataset_error_text}"
                )
            raise RuntimeError("Dataset view requires a dataset object, but no dataset is available.")
        config = load_benchmark_config(self.config_payload)
        return _dataset_frame_view(config, self.dataset_obj, split=split)

    def metric_names(self) -> list[str]:
        return _metric_names(self.summary_payload, self.metrics_payload)

    def model_names(self) -> list[str]:
        names: set[str] = set()
        if self.metrics_payload is not None:
            names.update(str(name) for name in self.metrics_payload.index)
        names.update(str(item.get("model_name") or "") for item in self.model_results_payload if str(item.get("model_name") or "").strip())
        names.update(str(name) for name in self.generated_scenarios_payload)
        return sorted(name for name in names if name.strip())

    def metrics(self, *, include_metadata: bool = False) -> pd.DataFrame:
        if self.metrics_payload is None:
            return pd.DataFrame()
        frame = self.metrics_payload.copy(deep=True)
        if include_metadata:
            return frame
        selected = [name for name in self.metric_names() if name in frame.columns]
        if "average_rank" in frame.columns:
            selected.append("average_rank")
        return frame.loc[:, selected]

    def ranks(self, *, include_metadata: bool = False) -> pd.DataFrame:
        if self.ranks_payload is None:
            return pd.DataFrame()
        frame = self.ranks_payload.copy(deep=True)
        if include_metadata:
            return frame
        selected = [name for name in self.metric_names() if name in frame.columns]
        if "average_rank" in frame.columns:
            selected.append("average_rank")
        return frame.loc[:, selected]

    def model_results(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.model_results_payload)

    def model_result(self, model_name: str) -> dict[str, Any]:
        target = str(model_name)
        for item in self.model_results_payload:
            if str(item.get("model_name") or "") == target:
                return copy.deepcopy(item)
        raise KeyError(f"Unknown model '{model_name}'. Available: {self.model_names()}")

    def model_overview(self) -> pd.DataFrame:
        frame = _model_overview_frame(
            summary=self.summary_payload,
            metrics=self.metrics_payload,
            model_results=self.model_results_payload,
        )
        return pd.DataFrame() if frame is None else frame

    def diagnostics(self, model_name: str | None = None) -> dict[str, Any]:
        diagnostics = _copy_diagnostics_payload(self.diagnostics_payload)
        if model_name is None:
            return diagnostics
        return _diagnostics_for_model(diagnostics, str(model_name))

    def _require_diagnostic_frame(self, key: str, include_name: str) -> pd.DataFrame:
        frame = self.diagnostics_payload.get(key)
        if isinstance(frame, pd.DataFrame):
            return frame.copy(deep=True)
        raise RuntimeError(
            f"{key} is unavailable for this run. Request include=['{include_name}'] when launching "
            "the notebook benchmark wrapper."
        )

    def distribution_summary(
        self,
        *,
        model_name: str | None = None,
        series_type: str | None = None,
    ) -> pd.DataFrame:
        frame = self._require_diagnostic_frame("distribution_summary", "distribution")
        if model_name is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == str(model_name)]
        if series_type is not None and "series_type" in frame.columns:
            frame = frame[frame["series_type"].astype(str) == str(series_type)]
        return frame.reset_index(drop=True)

    def distribution_summary_by_asset(
        self,
        *,
        model_name: str | None = None,
        series_type: str | None = None,
        asset: str | None = None,
    ) -> pd.DataFrame:
        frame = self._require_diagnostic_frame("distribution_summary_by_asset", "distribution")
        if model_name is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == str(model_name)]
        if series_type is not None and "series_type" in frame.columns:
            frame = frame[frame["series_type"].astype(str) == str(series_type)]
        if asset is not None and "asset" in frame.columns:
            frame = frame[frame["asset"].astype(str) == str(asset)]
        return frame.reset_index(drop=True)

    def per_window_metrics(self, model_name: str | None = None) -> pd.DataFrame:
        frame = self._require_diagnostic_frame("per_window_metrics", "per_window")
        if model_name is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == str(model_name)]
        return frame.reset_index(drop=True)

    def functional_smoke_summary(self, model_name: str | None = None) -> pd.DataFrame:
        frame = self._require_diagnostic_frame("functional_smoke_summary", "functional_smoke")
        if model_name is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == str(model_name)]
        return frame.reset_index(drop=True)

    def functional_smoke_checks(self, model_name: str | None = None) -> pd.DataFrame:
        frame = self._require_diagnostic_frame("functional_smoke_checks", "functional_smoke")
        if model_name is not None and "model" in frame.columns:
            frame = frame[frame["model"].astype(str) == str(model_name)]
        return frame.reset_index(drop=True)

    def model_debug_artifacts(self, model_name: str | None = None) -> dict[str, Any]:
        payload = dict(self.diagnostics_payload.get("model_debug_artifacts") or {})
        if not payload:
            raise RuntimeError(
                "model_debug_artifacts are unavailable for this run. Request include=['model_debug'] "
                "when launching the notebook benchmark wrapper."
            )
        if model_name is None:
            return copy.deepcopy(payload)
        target = str(model_name)
        if target not in payload:
            raise KeyError(f"No debug artifacts were saved for model '{model_name}'.")
        return copy.deepcopy(payload[target])

    def scenarios(self, model_name: str) -> np.ndarray:
        if not self.generated_scenarios_payload:
            raise RuntimeError(
                "Generated scenarios are unavailable for this run. Request include=['scenarios'] "
                "when launching the notebook benchmark wrapper."
            )
        target = str(model_name)
        if target not in self.generated_scenarios_payload:
            raise KeyError(f"No generated scenarios were saved for model '{model_name}'. Available: {sorted(self.generated_scenarios_payload)}")
        return np.array(self.generated_scenarios_payload[target], dtype=float, copy=True)

    def reference_scenarios(self) -> np.ndarray | None:
        if self.reference_scenarios_payload is None:
            return None
        return np.array(self.reference_scenarios_payload, dtype=float, copy=True)

    def _asset_index(self, asset: int | str) -> tuple[int, str]:
        if self.dataset_obj is None:
            if self.dataset_error_text:
                raise RuntimeError(
                    f"Dataset reconstruction failed for this run, so asset-aware views are unavailable: "
                    f"{self.dataset_error_text}"
                )
            raise RuntimeError("Asset-aware views require a dataset object, but no dataset is available.")
        asset_names = list(getattr(self.dataset_obj, "asset_names"))
        if isinstance(asset, int):
            if asset < 0 or asset >= len(asset_names):
                raise IndexError(f"Asset index {asset} is out of range for assets {asset_names}.")
            return int(asset), asset_names[int(asset)]
        if str(asset) not in asset_names:
            raise KeyError(f"Unknown asset '{asset}'. Available: {asset_names}")
        return asset_names.index(str(asset)), str(asset)

    def scenario_band(
        self,
        model_name: str,
        *,
        evaluation_window: int = 0,
        asset: int | str = 0,
    ) -> pd.DataFrame:
        if self.dataset_obj is None:
            if self.dataset_error_text:
                raise RuntimeError(
                    f"Scenario preview is unavailable because the dataset could not be rebuilt: "
                    f"{self.dataset_error_text}"
                )
            raise RuntimeError("Scenario preview requires a dataset object.")
        generated = self.scenarios(model_name)
        n_windows = int(getattr(self.dataset_obj, "contexts").shape[0])
        if evaluation_window < 0 or evaluation_window >= n_windows:
            raise IndexError(f"evaluation_window={evaluation_window} is out of range for {n_windows} windows.")
        asset_index, _ = self._asset_index(asset)
        return _scenario_band_dataframe(
            generated[evaluation_window],
            np.asarray(getattr(self.dataset_obj, "realized_futures"))[evaluation_window],
            asset_index,
        )

    def plot_scenario_band(
        self,
        model_name: str,
        *,
        evaluation_window: int = 0,
        asset: int | str = 0,
        ax: Any | None = None,
    ):
        plt = _import_matplotlib()
        frame = self.scenario_band(model_name, evaluation_window=evaluation_window, asset=asset)
        _, asset_name = self._asset_index(asset)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(frame))
        ax.plot(x, frame["realized"], label="realized", color="black", linewidth=1.5)
        ax.plot(x, frame["median"], label="median", color="tab:blue", linewidth=1.5)
        ax.fill_between(x, frame["p05"], frame["p95"], color="tab:blue", alpha=0.2, label="5-95% band")
        ax.set_title(f"{model_name} | {asset_name} | window {evaluation_window}")
        ax.set_xlabel("Forecast step")
        ax.legend(loc="best")
        return ax.figure

    def plot_per_window_metric(
        self,
        model_name: str,
        metric_name: str,
        *,
        ax: Any | None = None,
    ):
        plt = _import_matplotlib()
        frame = self.per_window_metrics(model_name=model_name)
        metric = str(metric_name)
        if metric not in frame.columns:
            available = [column for column in frame.columns if column not in {"model", "context_index", "evaluation_timestamp"}]
            raise KeyError(f"Unknown per-window metric '{metric_name}'. Available: {available}")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frame["context_index"], frame[metric], marker="o")
        ax.set_title(f"{model_name} | {metric}")
        ax.set_xlabel("Context index")
        ax.set_ylabel(metric)
        return ax.figure

    def debug_report(self, model_name: str) -> str:
        return _build_model_debug_report(
            model_name=str(model_name),
            benchmark_name=str(self.summary_payload.get("name") or self.config_payload.get("benchmark", {}).get("name") or "benchmark"),
            config=self.config_payload,
            metrics=self.metrics_payload,
            model_results=self.model_results_payload,
            diagnostics=self.diagnostics_payload,
            dataset=self.dataset_obj,
            generated_scenarios=self.generated_scenarios_payload,
            reference_scenarios=self.reference_scenarios_payload,
        )

    def compare_metrics(self, other: "NotebookRun | str | Path") -> pd.DataFrame:
        other_run = load_run(other) if isinstance(other, (str, Path)) else other
        if not isinstance(other_run, NotebookRun):
            raise TypeError("compare_metrics expects another NotebookRun or a saved run directory path.")
        left = self.metrics(include_metadata=False).add_suffix("_current")
        right = other_run.metrics(include_metadata=False).add_suffix("_compare")
        return left.join(right, how="outer")


def save_benchmark_definition(
    config_or_run: NotebookRun | BenchmarkConfig | Mapping[str, Any] | str | Path,
    path: str | Path,
) -> Path:
    """Persist a benchmark definition or notebook run config as JSON."""

    if isinstance(config_or_run, NotebookRun):
        payload = config_or_run.config()
    elif isinstance(config_or_run, BenchmarkConfig):
        payload = dump_benchmark_config(config_or_run)
    elif isinstance(config_or_run, Mapping):
        payload = to_jsonable(dict(config_or_run))
    else:
        payload = dump_benchmark_config(load_benchmark_config(config_or_run))
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def run_benchmark(
    config_or_path: BenchmarkConfig | str | Path | dict[str, Any],
    *,
    include: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    scheduler: str | None = None,
    with_model: NotebookModelSpec | Mapping[str, Any] | None = None,
    with_models: Sequence[NotebookModelSpec | Mapping[str, Any]] | None = None,
    with_dataset: NotebookDatasetSpec | Mapping[str, Any] | None = None,
) -> NotebookRun:
    """Run a benchmark for notebook use, overriding requested artifact flags.

    The optional notebook model arguments let a developer inject entrypoint,
    plugin, or builtin models for a single run without editing the underlying
    benchmark JSON. The optional dataset argument lets a developer swap in a
    CSV/Parquet dataset for a single run. These overrides are added only to the
    copied effective config used for execution and saved outputs.
    """

    config = config_or_path if isinstance(config_or_path, BenchmarkConfig) else load_benchmark_config(config_or_path)
    effective = copy.deepcopy(config)
    requested = _normalize_artifact_requests(include)
    _override_requested_artifacts(effective, requested)
    notebook_models = _notebook_models_to_add(with_model=with_model, with_models=with_models)
    forced_model_names = {model.name for model in notebook_models}
    _append_notebook_models(effective, notebook_models)
    if with_dataset is not None:
        effective.dataset = _dataset_config_from_spec(_normalize_notebook_dataset_spec(with_dataset))
    resolved_output_dir = _resolve_notebook_output_path(effective, output_dir)
    if resolved_output_dir is not None:
        effective.run.output.output_dir = str(resolved_output_dir)
    if device is not None:
        effective.run.device = str(device)
    if scheduler is not None:
        effective.run.scheduler = str(scheduler)

    previous_results_dir = _official_results_dir_for_config(effective)
    if previous_results_dir is not None and forced_model_names:
        merged = _execute_with_reused_results(
            effective=effective,
            forced_model_names=forced_model_names,
            previous_results_dir=previous_results_dir,
            output_dir=resolved_output_dir,
        )
        if merged is not None:
            return merged

    artifacts = run_benchmark_from_config(effective)
    return NotebookRun.from_benchmark_run(artifacts)


def load_run(run_dir: str | Path | NotebookRun) -> NotebookRun:
    """Load a previously saved benchmark run into the notebook result wrapper."""

    if isinstance(run_dir, NotebookRun):
        return run_dir
    return NotebookRun.from_saved_run(run_dir)
