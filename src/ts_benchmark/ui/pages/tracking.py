"""Experiment tracking page for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..renderers import render_structured_value
from ..services.tracking import (
    default_tracking_uri,
    download_artifact_path,
    get_run_payload,
    list_artifacts_df,
    list_experiments_df,
    search_runs_df,
    tracking_available,
)
from ..state import get_current_config, set_page, set_selected_run_dir


def render() -> None:
    st.header("Experiment Tracking")
    st.caption("Browse MLflow experiments and runs, inspect tracked artifacts, and open tracked benchmark outputs in the local explorer.")

    if not tracking_available():
        st.info("Install the optional MLflow dependency to use this page.")
        return

    uri = st.text_input("Tracking URI", value=default_tracking_uri(get_current_config()))
    try:
        experiments = list_experiments_df(uri)
    except Exception as exc:
        st.error(f"Could not load experiments: {exc}")
        return

    if experiments.empty:
        st.info("No experiments found.")
        return

    st.subheader("Experiments")
    st.dataframe(experiments, use_container_width=True, hide_index=True)

    options = {f"{row['name']} ({row['experiment_id']})": str(row["experiment_id"]) for _, row in experiments.iterrows()}
    selected_labels = st.multiselect("Experiments", options=list(options), default=list(options)[:1])
    if not selected_labels:
        st.info("Select at least one experiment.")
        return

    experiment_ids = [options[label] for label in selected_labels]
    runs = search_runs_df(uri, experiment_ids, max_results=100)
    if runs.empty:
        st.info("No runs found.")
        return

    st.subheader("Runs")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    run_options = {
        f"{row['run_name'] or row['run_id']} ({row['run_id']})": str(row["run_id"])
        for _, row in runs.iterrows()
    }
    selected_run = st.selectbox("Run detail", options=list(run_options))
    run_id = run_options[selected_run]
    payload = get_run_payload(uri, run_id)
    render_cols = st.columns([1, 1])
    with render_cols[0]:
        st.subheader("Run details")
        render_structured_value(payload["info"], label="run info", editable=False, key_prefix="tracking.run.info")
        render_structured_value(payload["metrics"], label="metrics", editable=False, key_prefix="tracking.run.metrics")
    with render_cols[1]:
        st.subheader("Run parameters and tags")
        render_structured_value(payload["params"], label="params", editable=False, key_prefix="tracking.run.params")
        render_structured_value(payload["tags"], label="tags", editable=False, key_prefix="tracking.run.tags")

    artifact_path = st.text_input("Artifact path", value="benchmark_outputs")
    artifacts = list_artifacts_df(uri, run_id, artifact_path or None)
    st.subheader("Artifacts")
    if artifacts.empty:
        st.info("No artifacts found at that path.")
        return
    st.dataframe(artifacts, use_container_width=True, hide_index=True)

    if st.button("Open benchmark outputs in Results Explorer"):
        local_dir = download_artifact_path(uri, run_id, artifact_path or "benchmark_outputs")
        set_selected_run_dir(local_dir)
        set_page("Results Explorer")
        st.rerun()

    previewable = [
        str(row["path"])
        for _, row in artifacts.iterrows()
        if not bool(row["is_dir"]) and Path(str(row["path"])).suffix.lower() in {".json", ".csv", ".txt", ".log"}
    ]
    if previewable:
        artifact_to_preview = st.selectbox("Preview artifact", options=previewable)
        local_path = download_artifact_path(uri, run_id, artifact_to_preview)
        suffix = local_path.suffix.lower()
        st.subheader("Artifact preview")
        if suffix == ".json":
            render_structured_value(
                json.loads(local_path.read_text(encoding="utf-8")),
                label="artifact json",
                editable=False,
                key_prefix="tracking.artifact.json",
            )
        elif suffix == ".csv":
            st.dataframe(pd.read_csv(local_path), use_container_width=True, hide_index=True)
        else:
            st.code(local_path.read_text(encoding="utf-8"), language="text")
