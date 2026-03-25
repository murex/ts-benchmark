"""Execution page for the Streamlit UI."""

from __future__ import annotations

import copy
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from .. import OUTPUT_DIR
from ..renderers import render_status_badge, render_structured_value
from ..schema_forms import (
    render_benchmark_form,
    render_diagnostics_form,
    render_model_params_editor,
    render_tracking_form,
)
from ..services.configs import (
    build_effective_config,
    list_saved_benchmarks,
    load_saved_benchmark,
    validate_effective_config,
)
from ..services.environment import detect_devices
from ..services.runs import (
    launch_cli_run,
    load_run_artifacts,
    materialize_benchmark_results,
    poll_cli_run,
    read_log_tail,
)
from ..state import (
    get_current_config,
    get_current_config_path,
    get_current_run,
    get_current_run_artifacts,
    set_current_config,
    set_current_config_path,
    set_current_run,
    set_current_run_artifacts,
    set_effective_config,
    set_page,
    set_selected_run_dir,
    set_validation_result,
)

RUN_LAB_BENCHMARK_KEY = "run_lab.benchmark"
RUN_LAB_MODEL_SELECTION_KEY = "run_lab.models"
RUN_LAB_MODEL_SIGNATURE_KEY = "run_lab.model_signature"


def _render_section_heading(title: str, *, caption: str | None = None, level: str = "section") -> None:
    font_size = "1.05rem" if level == "section" else "0.96rem"
    margin_bottom = "0.35rem" if level == "section" else "0.25rem"
    st.markdown(
        (
            f"<div style='font-size:{font_size}; font-weight:600; margin:0.2rem 0 {margin_bottom} 0;'>"
            f"{title}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return slug or "benchmark"


def _default_run_fields(config: dict[str, Any], benchmark_name: str) -> None:
    run = config.setdefault("run", {})
    if not str(run.get("name") or "").strip():
        run["name"] = benchmark_name
    output = run.setdefault("output", {})
    if not str(output.get("output_dir") or "").strip():
        output["output_dir"] = str((OUTPUT_DIR / _slugify_name(benchmark_name)).resolve())


def _sync_selected_models(model_names: list[str]) -> list[str]:
    signature = tuple(model_names)
    previous_signature = tuple(st.session_state.get(RUN_LAB_MODEL_SIGNATURE_KEY, ()))
    if RUN_LAB_MODEL_SELECTION_KEY not in st.session_state or previous_signature != signature:
        st.session_state[RUN_LAB_MODEL_SELECTION_KEY] = list(model_names)
        st.session_state[RUN_LAB_MODEL_SIGNATURE_KEY] = signature
        return list(model_names)

    selected = [
        str(name)
        for name in st.session_state.get(RUN_LAB_MODEL_SELECTION_KEY, [])
        if str(name) in model_names
    ]
    if selected != st.session_state.get(RUN_LAB_MODEL_SELECTION_KEY):
        st.session_state[RUN_LAB_MODEL_SELECTION_KEY] = selected
    st.session_state[RUN_LAB_MODEL_SIGNATURE_KEY] = signature
    return selected


def _render_output_form(current: dict[str, Any]) -> None:
    run = current.setdefault("run", {})
    output = run.setdefault("output", {})
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


def _reset_run_lab_state(*, preserve: set[str] | None = None) -> None:
    keep = set(preserve or set())
    for key in list(st.session_state):
        if not key.startswith("run_lab."):
            continue
        if key in keep:
            continue
        st.session_state.pop(key, None)


def _selected_benchmark_row() -> dict[str, Any] | None:
    rows = list_saved_benchmarks()
    if not rows:
        return None
    options = {str(row["name"]): row for row in rows}
    current_path = get_current_config_path()
    default_name = str(rows[0]["name"])
    if current_path is not None:
        for row in rows:
            if Path(row["path"]) == current_path:
                default_name = str(row["name"])
                break
    elif RUN_LAB_BENCHMARK_KEY in st.session_state:
        candidate = str(st.session_state[RUN_LAB_BENCHMARK_KEY])
        if candidate in options:
            default_name = candidate
    selected_name = st.selectbox(
        "Benchmark",
        options=list(options),
        index=list(options).index(default_name),
        key=RUN_LAB_BENCHMARK_KEY,
    )
    return dict(options[str(selected_name)])


def _load_selected_benchmark(row: dict[str, Any]) -> dict[str, Any]:
    selected_path = Path(row["path"])
    current_path = get_current_config_path()
    current = get_current_config()
    if current_path is not None and current_path == selected_path and current:
        return current
    _reset_run_lab_state(preserve={RUN_LAB_BENCHMARK_KEY})
    current = load_saved_benchmark(str(row["name"]))
    set_current_config(current)
    set_current_config_path(selected_path)
    set_current_run(None)
    set_current_run_artifacts(None)
    set_validation_result(None)
    st.rerun()
    return current


def _attach_run_results(
    run_info: dict[str, Any],
    *,
    benchmark_config: dict[str, Any],
    previous_results_dir: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if run_info.get("results_attached"):
        latest_results_dir = run_info.get("latest_results_dir")
        artifacts = None if latest_results_dir is None else load_run_artifacts(Path(str(latest_results_dir)))
        return run_info, artifacts
    benchmark_path = run_info.get("benchmark_path")
    run_dir = run_info.get("output_dir")
    if benchmark_path is None or run_dir is None:
        return run_info, None
    merged_results_dir = materialize_benchmark_results(
        benchmark_path=Path(str(benchmark_path)),
        benchmark_config=benchmark_config,
        source_run_dir=Path(str(run_dir)),
        previous_results_dir=None if previous_results_dir is None else Path(str(previous_results_dir)),
    )
    merged_artifacts = load_run_artifacts(merged_results_dir)
    updated = dict(run_info)
    updated["results_attached"] = True
    updated["latest_results_dir"] = str(merged_results_dir)
    updated["latest_results_summary"] = dict(merged_artifacts.get("summary") or {})
    return updated, merged_artifacts


def render() -> None:
    st.header("Run Lab")
    st.caption("Select an official benchmark, choose the models to run, optionally adjust model params, and inspect its latest results.")

    benchmark_row = _selected_benchmark_row()
    if benchmark_row is None:
        st.info("No official benchmarks are available yet.")
        return

    _render_section_heading("Benchmark")
    current = _load_selected_benchmark(benchmark_row)
    benchmark = current.setdefault("benchmark", {})
    benchmark_name = str(benchmark.get("name") or benchmark_row["name"]).strip() or str(benchmark_row["name"])
    source_models = list(benchmark.get("models") or [])
    model_names = [str(model.get("name") or "") for model in source_models if str(model.get("name") or "").strip()]

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "benchmark": benchmark_name,
                    "dataset": str(benchmark_row.get("dataset") or "Not selected"),
                    "models": len(model_names),
                    "latest_results": "Available" if benchmark_row.get("results_run_dir") else "Not run yet",
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if benchmark_row.get("results_updated_at"):
        st.caption(f"Latest results updated: {benchmark_row['results_updated_at']}")

    if not model_names:
        st.info("This benchmark has no models. Add models in Benchmarks before running it.")
        return

    _sync_selected_models(model_names)
    selected_model_names = st.multiselect(
        "Models to run",
        options=model_names,
        default=model_names,
        key=RUN_LAB_MODEL_SELECTION_KEY,
    )

    run_config = copy.deepcopy(current)
    run_benchmark = run_config.setdefault("benchmark", {})
    _default_run_fields(run_config, benchmark_name)
    run_models = [
        copy.deepcopy(model)
        for model in source_models
        if str(model.get("name") or "") in selected_model_names
    ]
    run_benchmark["models"] = run_models

    editor_tabs = st.tabs(["Run Configuration", "Diagnostics", "Model Overrides"])

    with editor_tabs[0]:
        render_benchmark_form(run_config, device_options=detect_devices())
        _render_section_heading("Output", level="subsection")
        _render_output_form(run_config)
        _render_section_heading("Tracking", level="subsection")
        render_tracking_form(run_config)

    with editor_tabs[1]:
        st.caption("Diagnostics are optional debug outputs and sanity checks. Leave them off for normal benchmark runs.")
        render_diagnostics_form(run_config)

    with editor_tabs[2]:
        if not run_models:
            st.warning("Select at least one model to run this benchmark.")
        else:
            st.caption("Optional. Changes here apply only to the run launched from Run Lab.")
            for index, model in enumerate(run_models):
                with st.expander(f"{index + 1}. {model['name']}", expanded=False):
                    render_model_params_editor(
                        model,
                        key_prefix=f"run_lab.models.{index}",
                        allow_add_fields=False,
                        show_execution_controls=False,
                        show_description=False,
                        show_pipeline_steps=False,
                    )

    effective = build_effective_config(run_config)
    set_effective_config(effective)

    run_info = get_current_run()
    artifacts = get_current_run_artifacts()
    benchmark_results_dir = benchmark_row.get("results_run_dir")
    latest_results_dir = None if run_info is None else run_info.get("latest_results_dir")
    open_results_dir = latest_results_dir or (artifacts.get("run_dir") if artifacts is not None else benchmark_results_dir)

    actions = st.columns([1, 1, 1, 1])
    if actions[0].button("Validate", use_container_width=True):
        ok, message = validate_effective_config(effective)
        set_validation_result({"ok": ok, "message": message})
        if ok:
            st.success("Run configuration is valid.")
        else:
            st.error(message)

    if actions[1].button(
        "Run benchmark",
        use_container_width=True,
        disabled=(run_info is not None and run_info.get("status") == "running") or not run_models,
    ):
        ok, message = validate_effective_config(effective)
        set_validation_result({"ok": ok, "message": message})
        if not ok:
            st.error(f"Cannot run an invalid benchmark: {message}")
        else:
            run_info = launch_cli_run(effective)
            run_info["benchmark_name"] = str(benchmark_row["name"])
            run_info["benchmark_path"] = str(benchmark_row["path"])
            run_info["selected_models"] = list(selected_model_names)
            set_current_run(run_info)
            set_current_run_artifacts(None)
            st.rerun()

    if actions[2].button("Refresh status", use_container_width=True):
        st.rerun()

    if actions[3].button("Open results", use_container_width=True, disabled=open_results_dir is None):
        if open_results_dir is not None:
            set_selected_run_dir(open_results_dir)
            set_page("Results Explorer")
            st.rerun()

    run_info = get_current_run()
    if run_info is None:
        return

    updated = poll_cli_run(run_info)
    if updated != run_info:
        set_current_run(updated)
        run_info = updated

    _render_section_heading("Current Run")
    render_status_badge(str(run_info.get("status", "unknown")))
    render_structured_value(
        {
            "benchmark": run_info.get("benchmark_name"),
            "models": run_info.get("selected_models") or [],
            "pid": run_info.get("pid"),
            "output_dir": run_info.get("output_dir"),
            "config_path": run_info.get("config_path"),
            "log_path": run_info.get("log_path"),
            "returncode": run_info.get("returncode"),
        },
        label="current run",
        editable=False,
        key_prefix="run_lab.current_run",
    )
    log_path = Path(run_info["log_path"])
    _render_section_heading("CLI Log Tail", level="subsection")
    st.code(read_log_tail(log_path), language="text")

    if run_info.get("status") == "succeeded" and run_info.get("output_dir"):
        output_dir = Path(run_info["output_dir"])
        if output_dir.exists():
            updated_run_info, merged_artifacts = _attach_run_results(
                run_info,
                benchmark_config=current,
                previous_results_dir=benchmark_results_dir,
            )
            set_current_run_artifacts(merged_artifacts)
            if updated_run_info != run_info:
                set_current_run(updated_run_info)
                st.rerun()
            st.success(f"Benchmark results updated from {output_dir}")
