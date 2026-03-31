"""Structured config editors for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ts_benchmark.metrics import available_metric_names, normalize_metric_config
from ts_benchmark.serialization import to_jsonable

from .renderers import render_float_input, render_key_value, render_structured_value
from .services.model_catalog import describe_catalog_model_entry

PROTOCOL_MODE_KEY = "benchmarks.protocol.generation_mode"
PROTOCOL_CONTEXT_LENGTH_KEY = "benchmarks.protocol.context_length"
PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY = "benchmarks.protocol.unconditional_train_data_mode"
PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY = "benchmarks.protocol.unconditional_n_train_paths"
PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY = "benchmarks.protocol.unconditional_n_eval_paths"
PROTOCOL_REGION_TRAIN_SIZE_KEY = "benchmarks.protocol.train_size_region"
PROTOCOL_REGION_TEST_SIZE_KEY = "benchmarks.protocol.test_size_region"
PREPROCESSING_PRESET_OPTIONS = ["raw", "standardized", "minmax", "custom"]
PREPROCESSING_STEP_TYPE_OPTIONS = [
    "identity",
    "demean",
    "standard_scale",
    "min_max_scale",
    "robust_scale",
    "clip",
    "winsorize",
]


def _dataset_supports_reference_scenarios(config: dict[str, Any]) -> bool:
    provider = (
        config.get("benchmark", {})
        .get("dataset", {})
        .get("provider", {})
    )
    return str(provider.get("kind") or "").strip().lower() == "synthetic"


def _count_sliding_windows(total_length: int, window_length: int, stride: int) -> int:
    if total_length < 1 or window_length < 1 or stride < 1 or total_length < window_length:
        return 0
    return 1 + (total_length - window_length) // stride


def _protocol_summary_lines(
    protocol: dict[str, Any],
    *,
    dataset_supports_reference_scenarios: bool,
) -> list[str]:
    kind = str(protocol.get("kind") or "forecast")
    horizon = int(protocol.get("horizon", 0) or 0)
    n_model_scenarios = int(protocol.get("n_model_scenarios", 0) or 0)
    evaluation_unit = "evaluation window"

    lines: list[str] = []
    if kind == "forecast":
        branch = dict(protocol.get("forecast") or {})
        train_size = int(branch.get("train_size", 0) or 0)
        test_size = int(branch.get("test_size", 0) or 0)
        context_length = int(branch.get("context_length", 0) or 0)
        train_stride = int(branch.get("train_stride", 1) or 1)
        eval_stride = int(branch.get("eval_stride", 1) or 1)
        n_train_examples = _count_sliding_windows(
            train_size,
            context_length + horizon,
            train_stride,
        )
        n_eval_windows = _count_sliding_windows(test_size, horizon, eval_stride)
        lines.append(
            f"Fit on {n_train_examples} forecast examples: benchmark-owned past ending in context {context_length} with target {horizon}."
        )
        lines.append(f"Evaluate on {n_eval_windows} forecast origins in the held-out future region.")
    elif kind == "unconditional_windowed":
        branch = dict(protocol.get("unconditional_windowed") or {})
        train_size = int(branch.get("train_size", 0) or 0)
        test_size = int(branch.get("test_size", 0) or 0)
        train_stride = int(branch.get("train_stride", 1) or 1)
        eval_stride = int(branch.get("eval_stride", 1) or 1)
        n_train_paths = _count_sliding_windows(train_size, horizon, train_stride)
        lines.append(
            f"Fit on {n_train_paths} unconditional training paths of length {horizon}."
        )
        n_eval_windows = _count_sliding_windows(test_size, horizon, eval_stride)
        lines.append(
            f"Evaluate on {n_eval_windows} unconditional origins with horizon {horizon}."
        )
    else:
        branch = dict(protocol.get("unconditional_path_dataset") or {})
        evaluation_unit = "realized path"
        n_train_paths = int(branch.get("n_train_paths", 0) or 0)
        n_realized_paths = int(branch.get("n_realized_paths", 0) or 0)
        lines.append(
            f"Fit on {n_train_paths} benchmark-provided unconditional training paths of length {horizon}."
        )
        lines.append(
            f"Evaluate on {n_realized_paths} held-out unconditional realized paths of length {horizon}."
        )
    lines.append(f"Request {n_model_scenarios} model scenarios per {evaluation_unit}.")
    if dataset_supports_reference_scenarios:
        n_reference_scenarios = int(protocol.get("n_reference_scenarios", 0) or 0)
        lines.append(
            f"Compare against {n_reference_scenarios} reference scenarios per {evaluation_unit}."
        )
    return lines


def _default_region_lengths(protocol: dict[str, Any]) -> tuple[int, int]:
    kind = str(protocol.get("kind") or "forecast")
    if kind == "unconditional_path_dataset":
        return 120, 40
    if kind == "forecast":
        branch = dict(protocol.get("forecast") or {})
    else:
        branch = dict(protocol.get("unconditional_windowed") or {})
    return int(branch.get("train_size", 120) or 120), int(branch.get("test_size", 40) or 40)


def _initial_mode_and_path_construction(protocol: dict[str, Any]) -> tuple[str, str]:
    kind = str(protocol.get("kind") or "forecast")
    if kind == "forecast":
        return "forecast", "windowed_path"
    if kind == "unconditional_path_dataset":
        return "unconditional", "path_dataset"
    return "unconditional", "windowed_path"


def _normalize_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.where(pd.notnull(frame), None)
    records: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        records.append({str(column): row[column] for column in normalized.columns})
    return records


def _raw_pipeline_config() -> dict[str, Any]:
    return {"name": "raw", "steps": []}


def _standardized_pipeline_config() -> dict[str, Any]:
    return {
        "name": "standardized",
        "steps": [
            {"type": "standard_scale", "params": {"with_mean": True, "with_std": True}},
        ],
    }


def _minmax_pipeline_config() -> dict[str, Any]:
    return {
        "name": "minmax",
        "steps": [
            {"type": "min_max_scale", "params": {"feature_min": 0.0, "feature_max": 1.0, "clip": False}},
        ],
    }


def _normalize_pipeline_dict(pipeline: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(pipeline or {})
    normalized_steps: list[dict[str, Any]] = []
    for step in list(resolved.get("steps") or []):
        normalized_steps.append(
            {
                "type": str(step.get("type") or "").strip(),
                "params": dict(step.get("params") or {}),
            }
        )
    return {
        "name": str(resolved.get("name") or "raw"),
        "steps": normalized_steps,
    }


def _pipeline_matches_preset(pipeline: dict[str, Any], preset_pipeline: dict[str, Any]) -> bool:
    normalized = _normalize_pipeline_dict(pipeline)
    expected = _normalize_pipeline_dict(preset_pipeline)
    return normalized == expected


def _infer_preprocessing_preset(pipeline: dict[str, Any] | None) -> str:
    normalized = _normalize_pipeline_dict(pipeline)
    if _pipeline_matches_preset(normalized, _raw_pipeline_config()):
        return "raw"
    if _pipeline_matches_preset(normalized, _standardized_pipeline_config()):
        return "standardized"
    if _pipeline_matches_preset(normalized, _minmax_pipeline_config()):
        return "minmax"
    return "custom"


def _pipeline_from_preprocessing_preset(pipeline: dict[str, Any] | None, preset: str) -> dict[str, Any]:
    if preset == "raw":
        return _raw_pipeline_config()
    if preset == "standardized":
        return _standardized_pipeline_config()
    if preset == "minmax":
        normalized = _normalize_pipeline_dict(pipeline)
        if normalized.get("steps"):
            first = dict(normalized["steps"][0] or {})
            if str(first.get("type") or "") == "min_max_scale":
                return {
                    "name": "minmax",
                    "steps": [
                        {
                            "type": "min_max_scale",
                            "params": {
                                "feature_min": float(dict(first.get("params") or {}).get("feature_min", 0.0)),
                                "feature_max": float(dict(first.get("params") or {}).get("feature_max", 1.0)),
                                "clip": bool(dict(first.get("params") or {}).get("clip", False)),
                            },
                        }
                    ],
                }
        return _minmax_pipeline_config()
    return _normalize_pipeline_dict(pipeline)


def render_config_general(config: dict[str, Any]) -> None:
    benchmark = config.setdefault("benchmark", {})
    run = config.setdefault("run", {})
    output = run.setdefault("output", {})
    config["version"] = st.text_input("Version", value=str(config.get("version", "1.0")))
    benchmark["name"] = st.text_input("Benchmark name", value=benchmark.get("name") or "benchmark")
    benchmark["description"] = st.text_area("Benchmark description", value=benchmark.get("description") or "")
    output["output_dir"] = st.text_input("Output directory", value=output.get("output_dir") or "")
    cols = st.columns(4)
    with cols[0]:
        output["keep_scenarios"] = st.checkbox("Keep scenarios in memory", value=bool(output.get("keep_scenarios", False)))
    with cols[1]:
        output["save_scenarios"] = st.checkbox("Save scenarios", value=bool(output.get("save_scenarios", False)))
    with cols[2]:
        output["save_model_info"] = st.checkbox("Save model info", value=bool(output.get("save_model_info", True)))
    with cols[3]:
        output["save_summary"] = st.checkbox("Save summary", value=bool(output.get("save_summary", True)))


def render_protocol_form(config: dict[str, Any]) -> None:
    protocol = config.setdefault("benchmark", {}).setdefault("protocol", {})
    dataset_supports_reference_scenarios = _dataset_supports_reference_scenarios(config)
    initial_mode, initial_unconditional_data_mode = _initial_mode_and_path_construction(protocol)
    initial_context_length = int(
        dict(protocol.get("forecast") or {}).get("context_length", 12 if initial_mode == "forecast" else 0)
    )
    initial_region_train_size, initial_region_test_size = _default_region_lengths(protocol)
    raw_unconditional_path_count = dict(protocol.get("unconditional_path_dataset") or {}).get("n_train_paths")
    initial_unconditional_path_count = (
        8
        if raw_unconditional_path_count in {None, ""}
        else int(raw_unconditional_path_count)
    )
    raw_unconditional_eval_path_count = dict(protocol.get("unconditional_path_dataset") or {}).get("n_realized_paths")
    initial_unconditional_eval_path_count = (
        4
        if raw_unconditional_eval_path_count in {None, ""}
        else int(raw_unconditional_eval_path_count)
    )
    st.session_state.setdefault(PROTOCOL_MODE_KEY, initial_mode)
    st.session_state.setdefault(PROTOCOL_CONTEXT_LENGTH_KEY, initial_context_length)
    st.session_state.setdefault(PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY, initial_unconditional_data_mode)
    st.session_state.setdefault(PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY, initial_unconditional_path_count)
    st.session_state.setdefault(PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY, initial_unconditional_eval_path_count)
    st.session_state.setdefault(PROTOCOL_REGION_TRAIN_SIZE_KEY, initial_region_train_size)
    st.session_state.setdefault(PROTOCOL_REGION_TEST_SIZE_KEY, initial_region_test_size)

    st.markdown("#### Task")
    previous_mode = str(st.session_state.get(PROTOCOL_MODE_KEY, initial_mode))
    task_cols = st.columns(2)
    with task_cols[0]:
        protocol["generation_mode"] = st.selectbox(
            "Generation mode",
            options=["forecast", "unconditional"],
            key=PROTOCOL_MODE_KEY,
        )
    if protocol["generation_mode"] != previous_mode:
        if protocol["generation_mode"] == "unconditional":
            st.session_state[PROTOCOL_CONTEXT_LENGTH_KEY] = 0
            if str(st.session_state.get(PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY, "")) not in {
                "windowed_path",
                "path_dataset",
            }:
                st.session_state[PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY] = initial_unconditional_data_mode
            if int(st.session_state.get(PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY, 0)) < 1:
                st.session_state[PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY] = initial_unconditional_path_count
            if int(st.session_state.get(PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY, 0)) < 1:
                st.session_state[PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY] = initial_unconditional_eval_path_count
        elif int(st.session_state.get(PROTOCOL_CONTEXT_LENGTH_KEY, 0)) < 1:
            fallback_context_length = initial_context_length if initial_context_length >= 1 else 12
            st.session_state[PROTOCOL_CONTEXT_LENGTH_KEY] = fallback_context_length
    if protocol["generation_mode"] == "forecast":
        cols = st.columns(2)
        with cols[0]:
            default_context_length = 12
            normalized_context_length = int(st.session_state.get(PROTOCOL_CONTEXT_LENGTH_KEY, default_context_length))
            if normalized_context_length < 1:
                normalized_context_length = initial_context_length if initial_context_length >= 1 else default_context_length
                st.session_state[PROTOCOL_CONTEXT_LENGTH_KEY] = normalized_context_length
            protocol["context_length"] = int(
                st.number_input(
                    "Context length",
                    min_value=1,
                    step=1,
                    key=PROTOCOL_CONTEXT_LENGTH_KEY,
                )
            )
        with cols[1]:
            protocol["horizon"] = int(
                st.number_input("Horizon", min_value=1, value=int(protocol.get("horizon", 4)), step=1)
            )
        protocol["kind"] = "forecast"
    else:
        with task_cols[1]:
            selected_unconditional_data_mode = st.selectbox(
                "Path construction",
                options=["windowed_path", "path_dataset"],
                key=PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY,
                format_func=lambda value: "Window one path" if value == "windowed_path" else "Use path dataset",
            )
        cols = st.columns(1)
        with cols[0]:
            protocol["horizon"] = int(
                st.number_input("Horizon", min_value=1, value=int(protocol.get("horizon", 4)), step=1)
            )
        protocol["kind"] = (
            "unconditional_path_dataset"
            if selected_unconditional_data_mode == "path_dataset"
            else "unconditional_windowed"
        )
    if protocol["generation_mode"] == "forecast":
        st.caption(
            "The benchmark fits on forecast examples that carry full benchmark-owned past plus the conditioning context, and evaluates sampled futures on held-out origins."
        )
    else:
        st.caption("The benchmark fits on unconditional paths and evaluates sampled futures at the configured horizon.")
    selected_unconditional_data_mode = (
        str(st.session_state.get(PROTOCOL_UNCONDITIONAL_DATA_MODE_KEY, initial_unconditional_data_mode))
        if protocol["generation_mode"] == "unconditional"
        else None
    )
    forecast_block = dict(protocol.get("forecast") or {})
    unconditional_windowed_block = dict(protocol.get("unconditional_windowed") or {})
    unconditional_path_dataset_block = dict(protocol.get("unconditional_path_dataset") or {})

    st.markdown("#### Data Split")
    unconditional_path_dataset = protocol["kind"] == "unconditional_path_dataset"
    if unconditional_path_dataset:
        cols = st.columns(2)
        with cols[0]:
            normalized_path_count = int(
                st.session_state.get(PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY, initial_unconditional_path_count)
            )
            if normalized_path_count < 1:
                normalized_path_count = initial_unconditional_path_count
                st.session_state[PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY] = normalized_path_count
            unconditional_path_dataset_block["n_train_paths"] = int(
                st.number_input(
                    "Training paths",
                    min_value=1,
                    step=1,
                    key=PROTOCOL_UNCONDITIONAL_PATH_COUNT_KEY,
                )
            )
        with cols[1]:
            normalized_eval_path_count = int(
                st.session_state.get(
                    PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY,
                    initial_unconditional_eval_path_count,
                )
            )
            if normalized_eval_path_count < 1:
                normalized_eval_path_count = initial_unconditional_eval_path_count
                st.session_state[PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY] = normalized_eval_path_count
            unconditional_path_dataset_block["n_realized_paths"] = int(
                st.number_input(
                    "Held-out realized paths",
                    min_value=1,
                    step=1,
                    key=PROTOCOL_UNCONDITIONAL_EVAL_PATH_COUNT_KEY,
                )
            )
        st.caption("Path-dataset mode does not use a temporal split. Horizon sets both training and held-out realized path length.")
    else:
        cols = st.columns(2)
        with cols[0]:
            normalized_train_size = int(st.session_state.get(PROTOCOL_REGION_TRAIN_SIZE_KEY, initial_region_train_size))
            if normalized_train_size < 1:
                normalized_train_size = initial_region_train_size
                st.session_state[PROTOCOL_REGION_TRAIN_SIZE_KEY] = normalized_train_size
            train_size = int(
                st.number_input(
                    "Training region length",
                    min_value=1,
                    step=1,
                    key=PROTOCOL_REGION_TRAIN_SIZE_KEY,
                )
            )
        with cols[1]:
            normalized_test_size = int(st.session_state.get(PROTOCOL_REGION_TEST_SIZE_KEY, initial_region_test_size))
            if normalized_test_size < 1:
                normalized_test_size = initial_region_test_size
                st.session_state[PROTOCOL_REGION_TEST_SIZE_KEY] = normalized_test_size
            test_size = int(
                st.number_input(
                    "Held-out future region length",
                    min_value=1,
                    step=1,
                    key=PROTOCOL_REGION_TEST_SIZE_KEY,
                )
            )
        if protocol["kind"] == "forecast":
            forecast_block["train_size"] = train_size
            forecast_block["test_size"] = test_size
        else:
            unconditional_windowed_block["train_size"] = train_size
            unconditional_windowed_block["test_size"] = test_size
        st.caption("Training examples are built from the training region. Realized futures come from the held-out future region.")

    if protocol["generation_mode"] == "forecast":
        st.markdown("#### Training Construction")
        cols = st.columns(1)
        with cols[0]:
            forecast_block["train_stride"] = int(
                st.number_input(
                    "Training window stride",
                    min_value=1,
                    value=int(forecast_block.get("train_stride", 1)),
                    step=1,
                )
            )
        forecast_block["context_length"] = int(protocol["context_length"])
        st.caption(
            "Forecast fit examples come from the training region using the benchmark-owned training stride, and each one carries the full past up to its target origin."
        )
    elif not unconditional_path_dataset:
        st.markdown("#### Training Construction")
        cols = st.columns(1)
        with cols[0]:
            unconditional_windowed_block["train_stride"] = int(
                st.number_input(
                    "Training path stride",
                    min_value=1,
                    value=int(unconditional_windowed_block.get("train_stride", 1)),
                    step=1,
                )
            )
        st.caption("Horizon sets the unconditional training path length in windowed-path mode.")

    st.markdown("#### Evaluation")
    if protocol["generation_mode"] == "unconditional":
        unconditional_path_dataset = protocol["kind"] == "unconditional_path_dataset"
        cols = st.columns(
            2 if unconditional_path_dataset and dataset_supports_reference_scenarios else 1 if unconditional_path_dataset else 3 if dataset_supports_reference_scenarios else 2
        )
        if unconditional_path_dataset:
            scenario_label = "Model scenarios per realized path"
            reference_label = "Reference scenarios per realized path"
            model_scenarios_col = 0
            reference_scenarios_col = 1
        else:
            scenario_label = "Model scenarios per evaluation window"
            reference_label = "Reference scenarios per evaluation window"
            with cols[0]:
                unconditional_windowed_block["eval_stride"] = int(
                    st.number_input(
                        "Evaluation origin stride",
                        min_value=1,
                        value=int(unconditional_windowed_block.get("eval_stride", 10)),
                        step=1,
                    )
                )
            model_scenarios_col = 1
            reference_scenarios_col = 2
    else:
        cols = st.columns(3 if dataset_supports_reference_scenarios else 2)
        eval_stride_col = 0
        model_scenarios_col = 1
        reference_scenarios_col = 2
        scenario_label = "Model scenarios per evaluation window"
        reference_label = "Reference scenarios per evaluation window"
        with cols[eval_stride_col]:
            forecast_block["eval_stride"] = int(
                st.number_input(
                    "Evaluation origin stride",
                    min_value=1,
                    value=int(forecast_block.get("eval_stride", 10)),
                    step=1,
                )
            )
    with cols[model_scenarios_col]:
        protocol["n_model_scenarios"] = int(
            st.number_input(
                scenario_label,
                min_value=1,
                value=int(protocol.get("n_model_scenarios", 64)),
                step=1,
            )
        )
    if dataset_supports_reference_scenarios:
        with cols[reference_scenarios_col]:
            protocol["n_reference_scenarios"] = int(
                st.number_input(
                    reference_label,
                    min_value=1,
                    value=int(protocol.get("n_reference_scenarios", 128)),
                    step=1,
                )
            )
    else:
        protocol["n_reference_scenarios"] = int(protocol.get("n_reference_scenarios", 128))
        st.caption("Reference scenarios are only used for synthetic datasets.")
    if protocol["generation_mode"] == "forecast":
        st.caption("Contexts may overlap the tail of the training region. Only realized futures are strictly held out.")
    elif unconditional_path_dataset:
        st.caption("Path-dataset evaluation compares generated samples against independent held-out realized paths, so origin stride is not used.")
    else:
        st.caption("Horizon controls the unconditional path length used at evaluation time.")

    for legacy_key in (
        "generation_mode",
        "context_length",
        "train_size",
        "test_size",
        "eval_stride",
        "train_stride",
        "unconditional_train_data_mode",
        "unconditional_train_window_length",
        "unconditional_n_train_paths",
        "unconditional_n_eval_paths",
    ):
        protocol.pop(legacy_key, None)

    if protocol["kind"] == "forecast":
        protocol["forecast"] = forecast_block
        protocol.pop("unconditional_windowed", None)
        protocol.pop("unconditional_path_dataset", None)
    elif protocol["kind"] == "unconditional_windowed":
        protocol["unconditional_windowed"] = unconditional_windowed_block
        protocol.pop("forecast", None)
        protocol.pop("unconditional_path_dataset", None)
    else:
        protocol["unconditional_path_dataset"] = unconditional_path_dataset_block
        protocol.pop("forecast", None)
        protocol.pop("unconditional_windowed", None)

    summary_lines = _protocol_summary_lines(
        protocol,
        dataset_supports_reference_scenarios=dataset_supports_reference_scenarios,
    )
    st.info("\n".join(summary_lines))


def render_benchmark_form(config: dict[str, Any], *, device_options: list[str]) -> None:
    run = config.setdefault("run", {})
    execution = run.setdefault("execution", {})
    run["name"] = st.text_input("Run name", value=run.get("name") or "")
    run["description"] = st.text_area("Run description", value=run.get("description") or "")
    run["seed"] = int(st.number_input("Seed", value=int(run.get("seed", 7)), step=1))
    scheduler = str(execution.get("scheduler") or "auto")
    execution["scheduler"] = st.selectbox(
        "Execution scheduler",
        options=["auto", "sequential", "model_parallel"],
        index=["auto", "sequential", "model_parallel"].index(scheduler)
        if scheduler in {"auto", "sequential", "model_parallel"}
        else 0,
    )
    current_device = execution.get("device")
    device_value = "auto" if current_device in {None, ""} else str(current_device)
    if device_value not in device_options:
        device_options = device_options + [device_value]
    chosen = st.selectbox("Execution device", options=device_options, index=device_options.index(device_value))
    execution["device"] = None if chosen == "auto" else chosen
    model_execution = dict(execution.get("model_execution") or {"mode": "inprocess"})
    selected_model_execution = st.selectbox(
        "Model execution",
        options=["inprocess", "subprocess"],
        index=0 if str(model_execution.get("mode") or "inprocess") == "inprocess" else 1,
        format_func=lambda value: "Inline" if value == "inprocess" else "Subprocess",
    )
    if selected_model_execution == "inprocess":
        execution["model_execution"] = {"mode": "inprocess"}
        return

    model_execution["mode"] = "subprocess"
    cols = st.columns(3)
    with cols[0]:
        model_execution["venv"] = st.text_input("Model venv", value=model_execution.get("venv") or "")
    with cols[1]:
        model_execution["python"] = st.text_input("Model python", value=model_execution.get("python") or "")
    with cols[2]:
        model_execution["cwd"] = st.text_input("Model working dir", value=model_execution.get("cwd") or "")
    st.caption("Model python path")
    pythonpath = render_structured_value(
        list(model_execution.get("pythonpath") or []),
        label="model_pythonpath",
        editable=True,
        key_prefix="run.execution.model_execution.pythonpath",
        allow_add_fields=False,
    )
    model_execution["pythonpath"] = list(pythonpath)
    st.caption("Model environment")
    model_execution["env"] = render_key_value(
        dict(model_execution.get("env") or {}),
        editable=True,
        key_prefix="run.execution.model_execution.env",
    )
    execution["model_execution"] = model_execution


def render_tracking_form(config: dict[str, Any], *, key_prefix: str = "tracking") -> None:
    run = config.setdefault("run", {})
    tracking = run.setdefault("tracking", {})
    mlflow = tracking.setdefault("mlflow", {})
    mlflow["enabled"] = st.checkbox(
        "Enable MLflow tracking",
        value=bool(mlflow.get("enabled", False)),
        key=f"{key_prefix}.enabled",
    )
    mlflow["tracking_uri"] = st.text_input(
        "Tracking URI",
        value=mlflow.get("tracking_uri") or "",
        key=f"{key_prefix}.tracking_uri",
    )
    cols = st.columns(2)
    with cols[0]:
        mlflow["experiment_name"] = st.text_input(
            "Experiment name",
            value=mlflow.get("experiment_name") or "ts-benchmark",
            key=f"{key_prefix}.experiment_name",
        )
    with cols[1]:
        mlflow["run_name"] = st.text_input(
            "Run name",
            value=mlflow.get("run_name") or "",
            key=f"{key_prefix}.run_name",
        )
    cols = st.columns(4)
    with cols[0]:
        mlflow["log_artifacts"] = st.checkbox(
            "Log artifacts",
            value=bool(mlflow.get("log_artifacts", True)),
            key=f"{key_prefix}.log_artifacts",
        )
    with cols[1]:
        mlflow["log_model_info"] = st.checkbox(
            "Log model info",
            value=bool(mlflow.get("log_model_info", True)),
            key=f"{key_prefix}.log_model_info",
        )
    with cols[2]:
        mlflow["log_diagnostics"] = st.checkbox(
            "Log diagnostics",
            value=bool(mlflow.get("log_diagnostics", True)),
            key=f"{key_prefix}.log_diagnostics",
        )
    with cols[3]:
        mlflow["log_scenarios"] = st.checkbox(
            "Log scenarios",
            value=bool(mlflow.get("log_scenarios", False)),
            key=f"{key_prefix}.log_scenarios",
        )
    st.caption("Tags")
    mlflow["tags"] = render_key_value(
        dict(mlflow.get("tags") or {}),
        editable=True,
        key_prefix=f"{key_prefix}.tags",
        allow_add_fields=False,
    )


def render_diagnostics_form(config: dict[str, Any]) -> None:
    diagnostics = config.setdefault("diagnostics", {})
    cols = st.columns(3)
    with cols[0]:
        diagnostics["save_model_debug_artifacts"] = st.checkbox(
            "Save model debug artifacts",
            value=bool(diagnostics.get("save_model_debug_artifacts", False)),
        )
    with cols[1]:
        diagnostics["save_distribution_summary"] = st.checkbox(
            "Save distribution summary",
            value=bool(diagnostics.get("save_distribution_summary", False)),
        )
    with cols[2]:
        diagnostics["save_per_window_metrics"] = st.checkbox(
            "Save per-window metrics",
            value=bool(diagnostics.get("save_per_window_metrics", False)),
        )
    smoke = diagnostics.setdefault("functional_smoke", {})
    smoke["enabled"] = st.checkbox("Enable sanity checks", value=bool(smoke.get("enabled", False)))
    smoke["finite_required"] = st.checkbox("Require finite samples", value=bool(smoke.get("finite_required", True)))
    cols = st.columns(3)
    with cols[0]:
        smoke["mean_abs_error_max"] = st.number_input(
            "Mean abs error max",
            value=float(smoke.get("mean_abs_error_max") or 0.005),
        )
        smoke["std_ratio_min"] = st.number_input(
            "Std ratio min",
            value=float(smoke.get("std_ratio_min") or 0.5),
        )
    with cols[1]:
        smoke["std_ratio_max"] = st.number_input(
            "Std ratio max",
            value=float(smoke.get("std_ratio_max") or 1.5),
        )
        smoke["crps_max"] = st.number_input(
            "CRPS max",
            value=float(smoke.get("crps_max") or 0.05),
        )
    with cols[2]:
        smoke["energy_score_max"] = st.number_input(
            "Energy score max",
            value=float(smoke.get("energy_score_max") or 0.1),
        )
        smoke["cross_correlation_error_max"] = st.number_input(
            "Cross-correlation error max",
            value=float(smoke.get("cross_correlation_error_max") or 1.0),
        )


def render_metrics_editor(config: dict[str, Any]) -> None:
    benchmark = config.setdefault("benchmark", {})
    metrics = list(benchmark.get("metrics") or [])
    if not metrics:
        metrics = [{"name": "crps"}, {"name": "energy_score"}]
    frame = pd.DataFrame([{"name": metric.get("name", "")} for metric in metrics])
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        use_container_width=True,
        key="config.metrics.rows",
        column_config={
            "name": st.column_config.SelectboxColumn(
                "Metric",
                options=available_metric_names(),
                required=True,
            )
        },
    )
    new_metrics: list[dict[str, Any]] = []
    for _, row in edited.iterrows():
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        new_metrics.append({"name": name})
    benchmark["metrics"] = new_metrics
    if not new_metrics:
        st.info("No explicit metrics selected. The benchmark will use its built-in default metric set.")
        return
    labels = [f"{idx + 1}. {metric['name']}" for idx, metric in enumerate(new_metrics)]
    selected = st.selectbox(
        "Metric detail",
        options=list(range(len(new_metrics))),
        format_func=lambda idx: labels[idx],
        key="config.metrics.detail",
    )
    resolved = to_jsonable(normalize_metric_config(new_metrics[selected]))
    render_structured_value(
        resolved,
        label="resolved_metric",
        editable=False,
        key_prefix=f"config.metrics.resolved.{selected}",
    )


def _parameter_widget_label(spec: dict[str, Any]) -> str:
    label = str(spec.get("name") or "param")
    annotation = str(spec.get("annotation") or "").strip()
    return label if not annotation else f"{label} ({annotation})"


def _parse_optional_numeric(text: str, *, numeric_type: type[int] | type[float]) -> int | float | None:
    cleaned = str(text).strip()
    if not cleaned:
        return None
    return numeric_type(cleaned)


def _render_model_parameter_field(spec: dict[str, Any], value: Any, *, key_prefix: str) -> Any:
    label = _parameter_widget_label(spec)
    description = str(spec.get("description") or "").strip()
    if description:
        st.caption(description)

    default = spec.get("default")
    value_type = str(spec.get("value_type") or "json")
    required = bool(spec.get("required", False))
    choices = list(spec.get("choices") or [])
    editable = bool(spec.get("editable", True))
    current = default if value is None else value

    if not editable:
        st.caption(f"{label}: this parameter is not editable through Benchmarks.")
        return value if value is not None else default

    if choices:
        options = choices if required else [""] + choices
        normalized = current if current in options else ("" if not required else options[0])
        selected = st.selectbox(label, options=options, index=options.index(normalized), key=key_prefix)
        return None if selected == "" and not required else selected

    if value_type == "bool":
        if current is None and not required:
            raw = st.selectbox(label, options=["", "true", "false"], key=key_prefix)
            return None if raw == "" else raw == "true"
        return bool(st.checkbox(label, value=bool(current), key=key_prefix))

    if value_type == "int":
        if current is None and not required:
            raw = st.text_input(label, value="", key=key_prefix)
            try:
                return _parse_optional_numeric(raw, numeric_type=int)
            except ValueError:
                st.warning(f"{label} must be an integer.")
                return value
        baseline = 0 if current is None else int(current)
        return int(st.number_input(label, value=baseline, step=1, key=key_prefix))

    if value_type == "float":
        if current is None and not required:
            raw = st.text_input(label, value="", key=key_prefix)
            try:
                return _parse_optional_numeric(raw, numeric_type=float)
            except ValueError:
                st.warning(f"{label} must be a number.")
                return value
        baseline = 0.0 if current is None else float(current)
        return render_float_input(label, baseline, key=key_prefix)

    if value_type == "string":
        raw = st.text_input(label, value="" if current is None else str(current), key=key_prefix)
        return None if raw == "" and not required and default is None else raw

    if value_type == "dict":
        payload = current if isinstance(current, dict) else (default if isinstance(default, dict) else {})
        return render_key_value(dict(payload), editable=True, key_prefix=key_prefix)

    if value_type == "list":
        payload = current if isinstance(current, list) else default
        if isinstance(payload, tuple):
            payload = list(payload)
        if not isinstance(payload, list):
            payload = []
        updated = render_structured_value(list(payload), label=label, editable=True, key_prefix=key_prefix)
        return list(updated or [])

    if isinstance(current, dict):
        return render_key_value(dict(current), editable=True, key_prefix=key_prefix)
    if isinstance(current, tuple):
        updated = render_structured_value(list(current), label=label, editable=True, key_prefix=key_prefix)
        return list(updated or [])
    if isinstance(current, list):
        updated = render_structured_value(list(current), label=label, editable=True, key_prefix=key_prefix)
        return list(updated or [])
    if isinstance(current, bool):
        return bool(st.checkbox(label, value=current, key=key_prefix))
    if isinstance(current, int) and not isinstance(current, bool):
        return int(st.number_input(label, value=current, step=1, key=key_prefix))
    if isinstance(current, float):
        return render_float_input(label, current, key=key_prefix)
    raw = st.text_input(label, value="" if current is None else str(current), key=key_prefix)
    return None if raw == "" and not required and default is None else raw


def _render_model_params_with_schema(model: dict[str, Any], *, key_prefix: str, allow_add_fields: bool) -> bool:
    detail = describe_catalog_model_entry(model)
    resolution = detail.get("resolution") or {}
    explicit_specs = [
        spec
        for spec in detail.get("parameters") or []
        if str(spec.get("parameter_type") or "") == "explicit"
    ]
    accepts_varargs = any(
        str(spec.get("parameter_type") or "") == "vararg"
        for spec in detail.get("parameters") or []
    )

    if resolution.get("error"):
        st.warning(f"Parameter schema unavailable: {resolution['error']}")
        return False
    if not explicit_specs:
        return False

    st.caption("Model params")
    params = dict(model.get("params") or {})
    resolved_params: dict[str, Any] = {}
    explicit_names = {str(spec.get("name") or "") for spec in explicit_specs}
    for spec in explicit_specs:
        name = str(spec.get("name") or "")
        if not name:
            continue
        resolved_params[name] = _render_model_parameter_field(
            spec,
            params.get(name),
            key_prefix=f"{key_prefix}.params.{name}",
        )

    extra_params = {
        key: value
        for key, value in params.items()
        if key not in explicit_names
    }
    if accepts_varargs or extra_params:
        st.caption("Additional params")
        extras = render_key_value(
            extra_params,
            editable=True,
            key_prefix=f"{key_prefix}.params.extra",
            allow_add_fields=allow_add_fields,
        )
        resolved_params.update(extras)

    model["params"] = resolved_params
    return True


def render_model_params_editor(
    model: dict[str, Any],
    *,
    key_prefix: str,
    allow_add_fields: bool = True,
    show_execution_controls: bool = True,
    show_description: bool = True,
    show_pipeline_steps: bool = True,
) -> None:
    if show_description:
        model["description"] = st.text_area(
            "Description",
            value=model.get("description") or "",
            key=f"{key_prefix}.description",
        )
    if not _render_model_params_with_schema(model, key_prefix=key_prefix, allow_add_fields=allow_add_fields):
        model["params"] = render_key_value(
            dict(model.get("params") or {}),
            editable=True,
            key_prefix=f"{key_prefix}.params",
            allow_add_fields=allow_add_fields,
        )
    st.caption("Metadata")
    model["metadata"] = render_key_value(
        dict(model.get("metadata") or {}),
        editable=True,
        key_prefix=f"{key_prefix}.metadata",
        allow_add_fields=allow_add_fields,
    )
    st.caption("Preprocessing")
    pipeline = _normalize_pipeline_dict(model.get("pipeline"))
    preset_key = f"{key_prefix}.pipeline.preset"
    initial_preset = _infer_preprocessing_preset(pipeline)
    st.session_state.setdefault(preset_key, initial_preset)
    if st.session_state.get(preset_key) not in PREPROCESSING_PRESET_OPTIONS:
        st.session_state[preset_key] = initial_preset
    preset = str(
        st.selectbox(
            "Preprocessing",
            options=PREPROCESSING_PRESET_OPTIONS,
            key=preset_key,
        )
    )
    pipeline = _pipeline_from_preprocessing_preset(pipeline, preset)
    if preset == "minmax":
        step_params = dict(pipeline["steps"][0].get("params") or {})
        cols = st.columns(3)
        with cols[0]:
            step_params["feature_min"] = float(
                st.number_input(
                    "Min value",
                    value=float(step_params.get("feature_min", 0.0)),
                    step=0.1,
                    key=f"{key_prefix}.pipeline.minmax.feature_min",
                )
            )
        with cols[1]:
            step_params["feature_max"] = float(
                st.number_input(
                    "Max value",
                    value=float(step_params.get("feature_max", 1.0)),
                    step=0.1,
                    key=f"{key_prefix}.pipeline.minmax.feature_max",
                )
            )
        with cols[2]:
            step_params["clip"] = bool(
                st.checkbox(
                    "Clip to range",
                    value=bool(step_params.get("clip", False)),
                    key=f"{key_prefix}.pipeline.minmax.clip",
                )
            )
        pipeline["steps"][0]["params"] = step_params
    if preset == "custom":
        pipeline["name"] = st.text_input(
            "Pipeline name",
            value=pipeline.get("name") or "custom",
            key=f"{key_prefix}.pipeline.name",
        )
    steps = list(pipeline.get("steps") or [])
    if preset == "custom":
        st.caption("Pipeline steps")
        st.caption(f"Available step types: {', '.join(PREPROCESSING_STEP_TYPE_OPTIONS)}")
        step_frame = pd.DataFrame([{"type": step.get("type", "")} for step in steps])
        edited_steps = st.data_editor(
            step_frame,
            num_rows="dynamic",
            use_container_width=True,
            key=f"{key_prefix}.pipeline.steps",
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "type",
                    options=PREPROCESSING_STEP_TYPE_OPTIONS,
                    required=False,
                )
            },
        )
        new_steps: list[dict[str, Any]] = []
        for idx, row in edited_steps.iterrows():
            base = steps[idx] if idx < len(steps) else {"params": {}}
            new_steps.append({"type": row.get("type") or f"step_{idx+1}", "params": dict(base.get("params") or {})})
        pipeline["steps"] = new_steps
        if new_steps:
            step_index = st.selectbox(
                "Edit step params",
                options=list(range(len(new_steps))),
                format_func=lambda idx: f"{idx + 1}. {new_steps[idx]['type']}",
                key=f"{key_prefix}.pipeline.step_select",
            )
            new_steps[step_index]["params"] = render_key_value(
                dict(new_steps[step_index].get("params") or {}),
                editable=True,
                key_prefix=f"{key_prefix}.pipeline.step_params.{step_index}",
                allow_add_fields=allow_add_fields,
            )
    model["pipeline"] = pipeline
    if not show_execution_controls:
        model.pop("execution", None)
        return
    execution_mode = st.selectbox(
        "Execution mode",
        options=["inprocess", "subprocess"],
        index=0 if dict(model.get("execution") or {}).get("mode", "inprocess") == "inprocess" else 1,
        key=f"{key_prefix}.execution_mode",
    )
    if execution_mode == "inprocess":
        model["execution"] = None
        return
    execution = dict(model.get("execution") or {})
    execution["mode"] = "subprocess"
    cols = st.columns(3)
    with cols[0]:
        execution["venv"] = st.text_input("Venv name", value=execution.get("venv") or "", key=f"{key_prefix}.venv")
    with cols[1]:
        execution["python"] = st.text_input("Python path", value=execution.get("python") or "", key=f"{key_prefix}.python")
    with cols[2]:
        execution["cwd"] = st.text_input("Working dir", value=execution.get("cwd") or "", key=f"{key_prefix}.cwd")
    st.caption("Python path entries")
    pythonpath = render_structured_value(
        list(execution.get("pythonpath") or []),
        label="pythonpath",
        editable=True,
        key_prefix=f"{key_prefix}.pythonpath",
    )
    execution["pythonpath"] = list(pythonpath)
    st.caption("Environment variables")
    execution["env"] = render_key_value(dict(execution.get("env") or {}), editable=True, key_prefix=f"{key_prefix}.env")
    model["execution"] = execution


def render_models_editor(config: dict[str, Any]) -> None:
    benchmark = config.setdefault("benchmark", {})
    models = list(benchmark.get("models") or [])
    frame = pd.DataFrame(
        [
            {
                "name": model.get("name", ""),
                "reference_kind": dict(model.get("reference") or {}).get("kind", "builtin"),
                "reference_value": dict(model.get("reference") or {}).get("value", ""),
                "preprocessing": _infer_preprocessing_preset(model.get("pipeline")),
                "execution_mode": dict(model.get("execution") or {}).get("mode", "inprocess"),
            }
            for model in models
        ]
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        use_container_width=True,
        key="config.models.rows",
        column_config={
            "reference_kind": st.column_config.SelectboxColumn(
                "Reference type",
                options=["builtin", "plugin", "entrypoint"],
                required=True,
            ),
            "preprocessing": st.column_config.SelectboxColumn(
                "Preprocessing",
                options=PREPROCESSING_PRESET_OPTIONS,
                required=True,
            ),
        },
    )
    new_models: list[dict[str, Any]] = []
    for idx, row in edited.iterrows():
        base = models[idx] if idx < len(models) else {"params": {}, "execution": None, "metadata": {}}
        updated = dict(base)
        updated["name"] = row.get("name") or f"model_{idx+1}"
        updated["reference"] = {
            "kind": row.get("reference_kind") or "builtin",
            "value": row.get("reference_value") or "",
        }
        updated["pipeline"] = _pipeline_from_preprocessing_preset(
            updated.get("pipeline"),
            str(row.get("preprocessing") or "raw"),
        )
        if (row.get("execution_mode") or "inprocess") == "inprocess":
            updated["execution"] = None
        else:
            execution = dict(updated.get("execution") or {})
            execution["mode"] = "subprocess"
            updated["execution"] = execution
        new_models.append(updated)
    benchmark["models"] = new_models
    if not new_models:
        return
    selected = st.selectbox(
        "Model detail",
        options=list(range(len(new_models))),
        format_func=lambda idx: f"{idx + 1}. {new_models[idx]['name']}",
        key="config.models.select",
    )
    render_model_params_editor(new_models[selected], key_prefix=f"config.models.detail.{selected}")
