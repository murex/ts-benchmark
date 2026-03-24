"""Benchmark authoring page for the Streamlit UI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ts_benchmark.metrics import available_metric_names

from ..schema_forms import (
    render_model_params_editor,
    render_protocol_form,
)
from ..services.configs import (
    build_effective_config,
    default_config_dict,
    list_saved_benchmarks,
    load_config_dict,
    load_saved_benchmark,
    validate_effective_config,
)
from ..services.datasets import list_saved_datasets, load_saved_dataset
from ..services.model_catalog import list_model_catalog, load_catalog_model
from ..state import (
    get_current_config,
    get_current_config_path,
    get_validation_result,
    set_current_config,
    set_current_config_path,
    set_current_run,
    set_current_run_artifacts,
    set_effective_config,
    set_page,
    set_validation_result,
)

BENCHMARKS_SECTION_KEY = "benchmarks.section"
BENCHMARKS_PENDING_SECTION_KEY = "benchmarks.pending_section"
BENCHMARKS_EDITOR_VIEW_KEY = "benchmarks.editor_view"
BENCHMARKS_PENDING_EDITOR_VIEW_KEY = "benchmarks.pending_editor_view"
BENCHMARKS_FLASH_KEY = "benchmarks.flash"
BENCHMARKS_UPLOAD_KEY = "benchmarks.catalog.upload"
BENCHMARKS_DATASET_SELECT_KEY = "benchmarks.definition.dataset_select"
BENCHMARKS_MODEL_SELECT_KEY = "benchmarks.definition.models"
BENCHMARKS_METRIC_SELECT_KEY = "benchmarks.definition.metrics"


def _set_flash(level: str, message: str) -> None:
    st.session_state[BENCHMARKS_FLASH_KEY] = {
        "level": str(level),
        "message": str(message),
    }


def _render_flash() -> None:
    payload = st.session_state.pop(BENCHMARKS_FLASH_KEY, None)
    if not payload:
        return
    level = payload.get("level")
    message = payload.get("message")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def _queue_navigation(section: str, editor_view: str | None = None) -> None:
    st.session_state[BENCHMARKS_PENDING_SECTION_KEY] = str(section)
    if editor_view is not None:
        st.session_state[BENCHMARKS_PENDING_EDITOR_VIEW_KEY] = str(editor_view)


def _apply_pending_navigation() -> None:
    pending_section = st.session_state.pop(BENCHMARKS_PENDING_SECTION_KEY, None)
    pending_view = st.session_state.pop(BENCHMARKS_PENDING_EDITOR_VIEW_KEY, None)
    if pending_section is not None:
        st.session_state[BENCHMARKS_SECTION_KEY] = str(pending_section)
    if pending_view is not None:
        st.session_state[BENCHMARKS_EDITOR_VIEW_KEY] = str(pending_view)


def _default_navigation() -> None:
    st.session_state.setdefault(BENCHMARKS_SECTION_KEY, "Catalog")
    st.session_state.setdefault(BENCHMARKS_EDITOR_VIEW_KEY, "Definition")


def _reset_benchmark_editor_widget_state() -> None:
    persistent = {
        BENCHMARKS_SECTION_KEY,
        BENCHMARKS_PENDING_SECTION_KEY,
        BENCHMARKS_EDITOR_VIEW_KEY,
        BENCHMARKS_PENDING_EDITOR_VIEW_KEY,
        BENCHMARKS_FLASH_KEY,
    }
    for key in list(st.session_state):
        if key in persistent:
            continue
        if key.startswith(("benchmarks.", "config.", "config_studio.")):
            st.session_state.pop(key, None)


def _ensure_current_benchmark() -> dict[str, Any]:
    current = get_current_config()
    if not current:
        current = default_config_dict()
    current.setdefault("version", "1.0")
    benchmark = current.setdefault("benchmark", {})
    defaults = default_config_dict()["benchmark"]
    benchmark.setdefault("name", defaults["name"])
    benchmark.setdefault("description", defaults["description"])
    benchmark.setdefault("dataset", copy.deepcopy(defaults["dataset"]))
    benchmark.setdefault("protocol", copy.deepcopy(defaults["protocol"]))
    benchmark.setdefault("metrics", copy.deepcopy(defaults["metrics"]))
    benchmark.setdefault("models", copy.deepcopy(defaults["models"]))
    run = current.setdefault("run", {})
    run.setdefault(
        "execution",
        {
            "scheduler": "auto",
            "device": None,
            "model_execution": {"mode": "inprocess"},
        },
    )
    run.setdefault("tracking", {"mlflow": {}})
    run.setdefault("output", {})
    current.setdefault("diagnostics", {})
    return current


def _open_benchmark(config: dict[str, Any], path: Path | None) -> None:
    _reset_benchmark_editor_widget_state()
    set_current_config(config)
    set_current_config_path(path)
    set_validation_result(None)
    _queue_navigation("Benchmark", "Definition")


def _new_benchmark() -> None:
    _open_benchmark(default_config_dict(), None)
    _set_flash("success", "Started a new benchmark.")
    st.rerun()


def _cloned_benchmark_name(name: str, existing_names: set[str]) -> str:
    base = str(name or "benchmark").strip() or "benchmark"
    candidate = f"{base}_copy"
    if candidate not in existing_names:
        return candidate
    index = 2
    while f"{candidate}_{index}" in existing_names:
        index += 1
    return f"{candidate}_{index}"


def _dataset_summary_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    provider = dict(dataset.get("provider") or {})
    schema = dict(dataset.get("schema") or {})
    return [
        {"field": "Dataset", "value": str(dataset.get("name") or "Unnamed dataset")},
        {"field": "Source", "value": str(provider.get("kind") or "")},
        {"field": "Layout", "value": str(schema.get("layout") or "")},
        {"field": "Frequency", "value": str(schema.get("frequency") or "")},
    ]


def _model_from_catalog(name: str) -> dict[str, Any]:
    entry = load_catalog_model(name)
    return {
        "name": entry.get("name") or name,
        "description": entry.get("description") or "",
        "reference": dict(entry.get("reference") or {}),
        "params": dict(entry.get("params") or {}),
        "metadata": dict(entry.get("metadata") or {}),
        "pipeline": {"name": "raw", "steps": []},
        "execution": None,
    }


def _sync_selected_models(current_models: list[dict[str, Any]], selected_names: list[str]) -> list[dict[str, Any]]:
    existing_by_name = {
        str(model.get("name") or ""): copy.deepcopy(model)
        for model in current_models
        if str(model.get("name") or "").strip()
    }
    synced: list[dict[str, Any]] = []
    for name in selected_names:
        if name in existing_by_name:
            synced.append(existing_by_name[name])
        else:
            synced.append(_model_from_catalog(name))
    return synced


def _catalog_badges(row: dict[str, Any]) -> str:
    badges = [
        f"`{row.get('dataset') or 'No dataset'}`",
        f"`{int(row.get('models') or 0)} models`",
        f"`{int(row.get('metrics') or 0)} metrics`",
    ]
    return " ".join(badges)


def _clone_catalog_benchmark(row: dict[str, Any]) -> None:
    config = load_saved_benchmark(str(row["name"]))
    existing_names = {str(item["name"]) for item in list_saved_benchmarks()}
    benchmark = config.setdefault("benchmark", {})
    benchmark["name"] = _cloned_benchmark_name(str(benchmark.get("name") or row["name"]), existing_names)
    _open_benchmark(config, None)
    _set_flash("success", f"Cloned benchmark '{row['name']}' into '{benchmark['name']}'.")


def _run_catalog_benchmark(row: dict[str, Any]) -> None:
    config = load_saved_benchmark(str(row["name"]))
    set_current_config(config)
    set_current_config_path(Path(row["path"]))
    set_effective_config(build_effective_config(config))
    set_validation_result(None)
    set_current_run(None)
    set_current_run_artifacts(None)
    set_page("Run Lab")
    _set_flash("success", f"Prepared benchmark '{row['name']}' in Run Lab.")


def _render_catalog() -> None:
    st.subheader("Catalog")
    action_cols = st.columns([1, 3])
    if action_cols[0].button("New benchmark", use_container_width=True):
        _new_benchmark()

    uploaded = st.file_uploader("Import benchmark JSON", type=["json"], key=BENCHMARKS_UPLOAD_KEY)
    if uploaded is not None:
        _open_benchmark(load_config_dict(uploaded.getvalue().decode("utf-8")), None)
        _set_flash("success", f"Imported benchmark '{uploaded.name}'.")
        st.rerun()

    st.subheader("Official benchmarks")
    rows = list_saved_benchmarks()
    if not rows:
        st.info("No official benchmarks are available.")
        return

    for row in rows:
        cols = st.columns([2.1, 2.9, 1.3, 0.9, 0.9, 0.9])
        with cols[0]:
            st.markdown(f"**{row['name']}**")
            st.markdown(_catalog_badges(row))
        with cols[1]:
            st.write(row.get("description") or "No description")
            st.caption(str(row["path"]))
        with cols[2]:
            dataset_name = str(row.get("dataset") or "No dataset")
            st.caption(f"Dataset: {dataset_name}")
            results_updated_at = row.get("results_updated_at")
            st.caption(
                "Results: none yet"
                if not results_updated_at
                else f"Results updated: {results_updated_at}"
            )
        if cols[3].button("Open", key=f"benchmarks.catalog.open.{row['name']}", use_container_width=True):
            _open_benchmark(load_saved_benchmark(str(row["name"])), Path(row["path"]))
            _set_flash("success", f"Opened benchmark '{row['name']}'.")
            st.rerun()
        if cols[4].button("Clone", key=f"benchmarks.catalog.clone.{row['name']}", use_container_width=True):
            _clone_catalog_benchmark(row)
            st.rerun()
        if cols[5].button(
            "Run",
            key=f"benchmarks.catalog.run.{row['name']}",
            use_container_width=True,
        ):
            _run_catalog_benchmark(row)
            st.rerun()
        st.divider()


def _render_definition(current: dict[str, Any]) -> None:
    benchmark = current.setdefault("benchmark", {})
    benchmark["name"] = st.text_input("Benchmark name", value=benchmark.get("name") or "benchmark")
    benchmark["description"] = st.text_area("Description", value=benchmark.get("description") or "")

    selected_dataset = dict(benchmark.get("dataset") or {})
    models = list(benchmark.get("models") or [])
    metrics = list(benchmark.get("metrics") or [])
    summary_cols = st.columns(3)
    summary_cols[0].metric("Dataset", str(selected_dataset.get("name") or "Not selected"))
    summary_cols[1].metric("Models", str(len(models)))
    summary_cols[2].metric("Metrics", str(len(metrics)))


def _render_dataset(current: dict[str, Any]) -> None:
    benchmark = current.setdefault("benchmark", {})

    quick = st.columns([1, 4])
    if quick[0].button("Manage datasets", use_container_width=True):
        set_page("Data Studio")
        st.rerun()

    st.subheader("Dataset")
    saved_datasets = list_saved_datasets()
    dataset_options = [str(row["name"]) for row in saved_datasets]
    current_dataset = dict(benchmark.get("dataset") or {})
    current_dataset_name = str(current_dataset.get("name") or "").strip()
    selection_options = list(dataset_options)
    embedded_option = "__embedded__"
    if current_dataset_name and current_dataset_name not in dataset_options:
        selection_options = [embedded_option] + selection_options
    elif not current_dataset_name and current_dataset:
        selection_options = [embedded_option] + selection_options

    if selection_options:
        if current_dataset_name in dataset_options:
            default_index = selection_options.index(current_dataset_name)
        else:
            default_index = 0
        selected_dataset = st.selectbox(
            "Benchmark dataset",
            options=selection_options,
            index=default_index,
            key=BENCHMARKS_DATASET_SELECT_KEY,
            format_func=lambda value: (
                current_dataset_name or "Embedded dataset"
                if value == embedded_option
                else value
            ),
        )
        if selected_dataset != embedded_option:
            benchmark["dataset"] = load_saved_dataset(selected_dataset)
            current_dataset = dict(benchmark["dataset"])
    else:
        st.info("No saved datasets are available yet. Create one in Data Studio.")

    if current_dataset:
        st.dataframe(pd.DataFrame(_dataset_summary_rows(current_dataset)), use_container_width=True, hide_index=True)
        if current_dataset.get("description"):
            st.caption(str(current_dataset["description"]))


def _render_models(current: dict[str, Any]) -> None:
    benchmark = current.setdefault("benchmark", {})

    quick = st.columns([1, 4])
    if quick[0].button("Manage models", use_container_width=True):
        set_page("Model Catalog")
        st.rerun()

    st.subheader("Models")
    catalog_rows = list_model_catalog()
    model_options = [str(row["name"]) for row in catalog_rows]
    current_models = list(benchmark.get("models") or [])
    current_names = [str(model.get("name") or "") for model in current_models if str(model.get("name") or "").strip()]
    for name in current_names:
        if name not in model_options:
            model_options.append(name)
    selected_model_names = st.multiselect(
        "Benchmark models",
        options=model_options,
        default=current_names,
        key=BENCHMARKS_MODEL_SELECT_KEY,
    )
    benchmark["models"] = _sync_selected_models(current_models, selected_model_names)
    if not benchmark["models"]:
        st.info("Select at least one model for this benchmark.")
    else:
        for index, model in enumerate(benchmark["models"]):
            with st.expander(f"{index + 1}. {model['name']}", expanded=index == 0 and len(benchmark["models"]) == 1):
                render_model_params_editor(
                    model,
                    key_prefix=f"benchmarks.models.{index}",
                    allow_add_fields=False,
                    show_execution_controls=False,
                )


def _render_metrics(current: dict[str, Any]) -> None:
    benchmark = current.setdefault("benchmark", {})

    st.subheader("Metrics")
    current_metric_names = [str(metric.get("name") or "") for metric in benchmark.get("metrics") or [] if str(metric.get("name") or "").strip()]
    selected_metric_names = st.multiselect(
        "Benchmark metrics",
        options=available_metric_names(),
        default=current_metric_names,
        key=BENCHMARKS_METRIC_SELECT_KEY,
    )
    benchmark["metrics"] = [{"name": name} for name in selected_metric_names]
    if selected_metric_names:
        st.dataframe(
            pd.DataFrame({"metric": selected_metric_names}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select at least one metric for this benchmark.")


def _render_benchmark_editor(current: dict[str, Any]) -> None:
    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("Validate benchmark", use_container_width=True):
        effective = build_effective_config(current)
        ok, message = validate_effective_config(effective)
        set_validation_result({"ok": ok, "message": message})
        if ok:
            _set_flash("success", "Benchmark is valid.")
        else:
            _set_flash("error", f"Validation failed: {message}")
        st.rerun()
    if action_cols[1].button("Open Run Lab", use_container_width=True):
        set_page("Run Lab")
        st.rerun()

    current_path = get_current_config_path()
    st.caption("Source benchmark" if current_path is not None else "Session benchmark: not saved by the library")
    if current_path is not None:
        st.code(str(current_path), language="text")

    validation = get_validation_result()
    if validation is not None:
        if validation["ok"]:
            st.success("Benchmark is valid.")
        else:
            st.error(f"Validation failed: {validation['message']}")

    editor_view = st.segmented_control(
        "Benchmark view",
        options=["Definition", "Dataset", "Models", "Metrics", "Protocol"],
        key=BENCHMARKS_EDITOR_VIEW_KEY,
        label_visibility="collapsed",
        width="stretch",
    )
    editor_view = str(editor_view or st.session_state.get(BENCHMARKS_EDITOR_VIEW_KEY, "Definition"))

    if editor_view == "Definition":
        _render_definition(current)
    elif editor_view == "Dataset":
        _render_dataset(current)
    elif editor_view == "Models":
        _render_models(current)
    elif editor_view == "Metrics":
        _render_metrics(current)
    elif editor_view == "Protocol":
        render_protocol_form(current)


def render() -> None:
    st.header("Benchmarks")
    st.caption("Browse official benchmarks and edit a benchmark configuration in the current UI session.")
    _apply_pending_navigation()
    _default_navigation()
    _render_flash()

    current = _ensure_current_benchmark()
    section = st.segmented_control(
        "Benchmarks section",
        options=["Catalog", "Benchmark"],
        key=BENCHMARKS_SECTION_KEY,
        label_visibility="collapsed",
        width="stretch",
    )
    section = str(section or st.session_state.get(BENCHMARKS_SECTION_KEY, "Catalog"))

    if section == "Catalog":
        _render_catalog()
    else:
        _render_benchmark_editor(current)

    set_current_config(current)
    set_effective_config(build_effective_config(current))
