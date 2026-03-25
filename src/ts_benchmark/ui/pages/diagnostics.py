"""Diagnostics page for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..renderers import render_structured_value
from ..services.runs import discover_local_runs, load_run_artifacts
from ..state import get_current_run_artifacts, get_selected_run_dir, set_selected_run_dir


def render() -> None:
    st.header("Diagnostics")
    st.caption("Inspect sanity checks, distribution summaries, metric traces, and model debug artifacts.")

    runs = discover_local_runs()
    selected_run = get_selected_run_dir()
    current_artifacts = get_current_run_artifacts()
    if selected_run is None and current_artifacts is not None:
        selected_run = Path(current_artifacts["run_dir"])

    if not runs.empty:
        options = {row["name"]: Path(row["run_dir"]) for _, row in runs.iterrows()}
        names = list(options)
        default_index = 0
        if selected_run is not None and selected_run.name in options:
            default_index = names.index(selected_run.name)
        chosen = st.selectbox("Run", options=names, index=default_index)
        selected_run = options[chosen]
        set_selected_run_dir(selected_run)

    if selected_run is None:
        st.info("No run selected.")
        return

    artifacts = load_run_artifacts(selected_run)
    diagnostics = artifacts.get("diagnostics") or {}
    if not diagnostics:
        st.info("This run did not save diagnostics.")
        return

    tabs = st.tabs(["Sanity checks", "Distribution", "Metric traces", "Model Debug"])

    with tabs[0]:
        summary = diagnostics.get("functional_smoke_summary")
        checks = diagnostics.get("functional_smoke_checks")
        if summary is not None:
            st.dataframe(summary, use_container_width=True, hide_index=True)
        if checks is not None and not checks.empty:
            selected_model = st.selectbox(
                "Sanity check model",
                options=sorted(checks["model"].astype(str).unique()),
                key="diagnostics.sanity.model",
            )
            st.dataframe(
                checks[checks["model"].astype(str) == selected_model],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        summary = diagnostics.get("distribution_summary")
        by_asset = diagnostics.get("distribution_summary_by_asset")
        if summary is not None:
            st.dataframe(summary.round(6), use_container_width=True, hide_index=True)
        if by_asset is not None and not by_asset.empty:
            selected_model = st.selectbox("Distribution model", options=sorted(by_asset["model"].unique()), key="diagnostics.distribution.model")
            st.dataframe(by_asset[by_asset["model"] == selected_model].round(6), use_container_width=True, hide_index=True)

    with tabs[2]:
        per_window = diagnostics.get("per_window_metrics")
        if per_window is None or per_window.empty:
            st.info("No metric traces were saved.")
        else:
            model_name = st.selectbox("Model", options=sorted(per_window["model"].unique()), key="diagnostics.window.model")
            metric_columns = [column for column in per_window.columns if column not in {"model", "context_index", "evaluation_timestamp"}]
            metric_name = st.selectbox("Metric", options=metric_columns, key="diagnostics.window.metric")
            subset = per_window[per_window["model"] == model_name].copy()
            st.line_chart(subset.set_index("context_index")[[metric_name]], use_container_width=True)
            st.dataframe(subset.round(6), use_container_width=True, hide_index=True)

    with tabs[3]:
        debug_artifacts = diagnostics.get("model_debug_artifacts") or {}
        if not debug_artifacts:
            st.info("No model debug artifacts were saved.")
        else:
            model_name = st.selectbox("Model", options=sorted(debug_artifacts), key="diagnostics.debug.model")
            render_structured_value(debug_artifacts[model_name], label=model_name, editable=False, key_prefix=f"diagnostics.debug.{model_name}")
