"""Home page for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..services.configs import list_saved_benchmarks
from ..services.datasets import list_saved_datasets
from ..services.model_catalog import list_model_catalog
from ..services.runs import load_run_artifacts
from ..state import set_page, set_selected_run_dir

RESULT_METADATA_COLUMNS = {
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


def render() -> None:
    st.header("Home")
    st.caption("Workspace overview centered on official benchmark definitions, their latest results, datasets, and models.")

    benchmark_rows = sorted(list_saved_benchmarks(), key=lambda row: str(row.get("name") or "").lower())
    dataset_rows = sorted(list_saved_datasets(), key=lambda row: str(row.get("name") or "").lower())
    model_rows = sorted(list_model_catalog(), key=lambda row: str(row.get("name") or "").lower())
    benchmarks_with_results = [row for row in benchmark_rows if row.get("results_run_dir")]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Benchmarks", str(len(benchmark_rows)))
    summary_cols[1].metric("Benchmarks with results", str(len(benchmarks_with_results)))
    summary_cols[2].metric("Datasets", str(len(dataset_rows)))
    summary_cols[3].metric("Models", str(len(model_rows)))

    top_left, top_right = st.columns([1.15, 1.35])
    with top_left:
        st.subheader("Official Benchmarks")
        if not benchmark_rows:
            st.info("No official benchmarks are available yet.")
        else:
            st.dataframe(_benchmark_catalog_frame(benchmark_rows), use_container_width=True, hide_index=True)
            actions = st.columns(2)
            if actions[0].button("Open Benchmarks", key="home.benchmarks.open", use_container_width=True):
                set_page("Benchmarks")
                st.rerun()
            if actions[1].button("Open Run Lab", key="home.benchmarks.run_lab", use_container_width=True):
                set_page("Run Lab")
                st.rerun()

    with top_right:
        st.subheader("Benchmark Results")
        if not benchmarks_with_results:
            st.info("No benchmark results are available yet.")
        else:
            options = {str(row["name"]): row for row in benchmarks_with_results}
            selected_name = st.selectbox(
                "Show latest results for",
                options=list(options),
                key="home.results.benchmark",
            )
            selected_row = dict(options[selected_name])
            selected_run = Path(str(selected_row["results_run_dir"]))
            try:
                artifacts = load_run_artifacts(selected_run)
            except Exception as exc:
                st.error(f"Could not load saved results for '{selected_row['name']}': {exc}")
            else:
                model_table = _result_model_table(artifacts)
                result_summary = dict(selected_row.get("results_summary") or {})
                status = str(dict(artifacts.get("run") or {}).get("status") or "unknown")
                metric_names = _result_metric_names(artifacts, artifacts.get("metrics"))
                summary = st.columns(4)
                summary[0].metric("Benchmark", str(selected_row["name"]))
                summary[1].metric("Status", status)
                summary[2].metric("Models", str(0 if model_table is None else len(model_table.index)))
                summary[3].metric("Metrics", str(len(metric_names)))
                if selected_row.get("results_updated_at"):
                    st.caption(f"Latest saved results updated on {selected_row['results_updated_at']}.")
                elif result_summary:
                    st.caption("Latest saved benchmark results are available.")

                if model_table is None or model_table.empty:
                    st.info("No model metrics were saved for this benchmark result.")
                else:
                    st.dataframe(model_table, use_container_width=True, hide_index=True)

                result_actions = st.columns(2)
                if result_actions[0].button(
                    "Open Results Explorer",
                    key="home.results.open",
                    use_container_width=True,
                ):
                    set_selected_run_dir(selected_run)
                    set_page("Results Explorer")
                    st.rerun()
                if result_actions[1].button("Open Run Lab", key="home.results.run_lab", use_container_width=True):
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
                set_page("Model Catalog")
                st.rerun()
