"""Home page for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..services.configs import load_saved_benchmark, list_saved_benchmarks
from ..services.datasets import list_saved_datasets
from ..services.model_catalog import list_model_catalog
from ..services.runs import load_run_artifacts
from ..state import (
    set_current_config,
    set_current_config_path,
    set_current_run,
    set_current_run_artifacts,
    set_page,
    set_selected_run_dir,
    set_validation_result,
)

RESULT_METADATA_COLUMNS = {
    "dataset_name",
    "dataset_source",
    "device",
    "has_reference_scenarios",
    "protocol_kind",
    "path_construction",
    "train_size",
    "test_size",
    "generation_mode",
    "context_length",
    "horizon",
    "eval_stride",
    "train_stride",
    "n_train_paths",
    "n_realized_paths",
    "n_model_scenarios",
    "n_reference_scenarios",
    "execution_mode",
    "average_rank",
}
BENCHMARKS_PENDING_SECTION_KEY = "benchmarks.pending_section"
BENCHMARKS_PENDING_EDITOR_VIEW_KEY = "benchmarks.pending_editor_view"
MODEL_CATALOG_RETURN_SOURCE_KEY = "model_catalog.return_source"


def _benchmark_catalog_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": row["name"],
                "dataset": row["dataset"] or "",
                "models": row["models"],
                "metrics": row["metrics"],
                "results": "Available" if row.get("results_run_dir") else "Not run yet",
                "updated": row.get("results_updated_at") or "",
            }
            for row in rows
        ]
    )


def _dataset_catalog_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": row["name"],
                "source": row["source"] or "",
                "frequency": row["frequency"] or "",
                "description": row["description"] or "",
            }
            for row in rows
        ]
    )


def _model_catalog_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": row["name"],
                "family": row["family"] or "",
                "origin": row["origin"] or "",
                "status": row["status"] or "",
                "description": row["description"] or "",
            }
            for row in rows
        ]
    )


def _result_metric_names(artifacts: dict[str, object], metrics: pd.DataFrame | None) -> list[str]:
    summary = dict(artifacts.get("summary") or {})
    configured = [
        str(metric.get("name") or "").strip()
        for metric in summary.get("metrics") or []
        if str(metric.get("name") or "").strip()
    ]
    if configured:
        return configured
    if metrics is None:
        return []
    return [
        str(column)
        for column in metrics.columns
        if str(column) not in RESULT_METADATA_COLUMNS and str(column) != "model"
    ]


def _result_model_table(artifacts: dict[str, object]) -> pd.DataFrame | None:
    metrics = artifacts.get("metrics")
    if metrics is None or metrics.empty:
        return None
    metric_names = _result_metric_names(artifacts, metrics)
    columns = ["model", *[name for name in metric_names if name in metrics.columns]]
    if "average_rank" in metrics.columns:
        columns.append("average_rank")
    frame = metrics[columns].copy()
    if "average_rank" in frame.columns:
        frame = frame.sort_values("average_rank", ascending=True, na_position="last")
    return frame


def _open_benchmark(row: dict[str, Any]) -> None:
    config = load_saved_benchmark(str(row["name"]))
    set_current_config(config)
    set_current_config_path(Path(str(row["path"])))
    set_current_run(None)
    set_current_run_artifacts(None)
    set_validation_result(None)
    st.session_state[BENCHMARKS_PENDING_SECTION_KEY] = "Benchmark"
    st.session_state[BENCHMARKS_PENDING_EDITOR_VIEW_KEY] = "Definition"
    set_page("Benchmarks")
    st.rerun()


def render() -> None:
    st.header("Home")
    st.caption("Workspace overview centered on benchmarks, datasets, and models.")

    benchmark_rows = sorted(list_saved_benchmarks(), key=lambda row: str(row.get("name") or "").lower())
    dataset_rows = sorted(list_saved_datasets(), key=lambda row: str(row.get("name") or "").lower())
    model_rows = sorted(list_model_catalog(), key=lambda row: str(row.get("name") or "").lower())
    benchmarks_with_results = [row for row in benchmark_rows if row.get("results_run_dir")]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Benchmarks", str(len(benchmark_rows)))
    summary_cols[1].metric("Benchmarks with results", str(len(benchmarks_with_results)))
    summary_cols[2].metric("Datasets", str(len(dataset_rows)))
    summary_cols[3].metric("Models", str(len(model_rows)))

    st.subheader("Benchmarks")
    if not benchmarks_with_results:
        st.info("No benchmark results are available yet.")
    else:
        options = {str(row["name"]): row for row in benchmarks_with_results}
        selected_name = st.selectbox(
            "Benchmarks",
            options=list(options),
            key="home.results.benchmark",
            label_visibility="collapsed",
        )
        selected_row = dict(options[selected_name])
        selected_run = Path(str(selected_row["results_run_dir"]))
        try:
            artifacts = load_run_artifacts(selected_run)
        except Exception as exc:
            st.error(f"Could not load saved results for '{selected_row['name']}': {exc}")
        else:
            model_table = _result_model_table(artifacts)
            if model_table is None or model_table.empty:
                st.info("No model metrics were saved for this benchmark result.")
            else:
                st.dataframe(model_table, use_container_width=True, hide_index=True)

            result_actions = st.columns(3)
            if result_actions[0].button("Open benchmark", key="home.results.benchmark_open", use_container_width=True):
                _open_benchmark(selected_row)
            if result_actions[1].button(
                "Open Results Explorer",
                key="home.results.open",
                use_container_width=True,
            ):
                set_selected_run_dir(selected_run)
                set_page("Results Explorer")
                st.rerun()
            if result_actions[2].button("Open Run Lab", key="home.results.run_lab", use_container_width=True):
                set_page("Run Lab")
                st.rerun()

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.subheader("Datasets")
        if not dataset_rows:
            st.info("No saved datasets are available yet.")
        else:
            st.dataframe(_dataset_catalog_frame(dataset_rows), use_container_width=True, hide_index=True)
            if st.button("Open Data Studio", key="home.datasets.open", use_container_width=True):
                set_page("Data Studio")
                st.rerun()

    with lower_right:
        st.subheader("Models")
        if not model_rows:
            st.info("No models are available in the catalog.")
        else:
            st.dataframe(_model_catalog_frame(model_rows), use_container_width=True, hide_index=True)
            if st.button("Open Model Catalog", key="home.models.open", use_container_width=True):
                st.session_state[MODEL_CATALOG_RETURN_SOURCE_KEY] = None
                set_page("Model Catalog")
                st.rerun()
