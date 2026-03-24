"""Results explorer page for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..renderers import render_structured_value
from ..services.configs import list_saved_benchmarks, load_config_dict
from ..services.runs import load_run_artifacts
from ..state import get_compare_run_dir, get_current_config_path, get_selected_run_dir, set_compare_run_dir, set_selected_run_dir

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


def _scenario_band_dataframe(samples: np.ndarray, realized: np.ndarray, asset_index: int) -> pd.DataFrame:
    asset_samples = np.asarray(samples[:, :, asset_index], dtype=float)
    q05, q50, q95 = np.quantile(asset_samples, [0.05, 0.50, 0.95], axis=0)
    return pd.DataFrame(
        {
            "realized": realized[:, asset_index],
            "p05": q05,
            "median": q50,
            "p95": q95,
        }
    )


def _default_selected_benchmark(rows: list[dict[str, object]]) -> str | None:
    if not rows:
        return None
    current_path = get_current_config_path()
    if current_path is not None:
        for row in rows:
            if Path(row["path"]) == current_path:
                return str(row["name"])
    selected_run = get_selected_run_dir()
    if selected_run is not None:
        for row in rows:
            if row.get("results_run_dir") and Path(str(row["results_run_dir"])) == selected_run:
                return str(row["name"])
    return str(rows[0]["name"])


def _metric_names(artifacts: dict[str, object], metrics: pd.DataFrame | None) -> list[str]:
    summary = dict(artifacts.get("summary") or {})
    names = [
        str(metric.get("name") or "").strip()
        for metric in summary.get("metrics") or []
        if str(metric.get("name") or "").strip()
    ]
    if names:
        return names
    if metrics is None:
        return []
    return [
        str(column)
        for column in metrics.columns
        if str(column) not in RESULTS_METADATA_COLUMNS and str(column) != "model"
    ]


def _configured_model_names(config: dict[str, object] | None) -> list[str]:
    benchmark = dict((config or {}).get("benchmark") or {})
    return [
        str(model.get("name") or "").strip()
        for model in benchmark.get("models") or []
        if str(model.get("name") or "").strip()
    ]


def _configured_dataset_name(config: dict[str, object] | None) -> str:
    benchmark = dict((config or {}).get("benchmark") or {})
    dataset = dict(benchmark.get("dataset") or {})
    provider = dict(dataset.get("provider") or {})
    return str(dataset.get("name") or provider.get("kind") or "").strip()


def _benchmark_results_alignment(
    benchmark_config: dict[str, object] | None,
    run_config: dict[str, object] | None,
) -> dict[str, object]:
    benchmark_models = _configured_model_names(benchmark_config)
    run_models = _configured_model_names(run_config)
    benchmark_dataset = _configured_dataset_name(benchmark_config)
    run_dataset = _configured_dataset_name(run_config)
    missing_models = [name for name in benchmark_models if name not in run_models]
    extra_models = [name for name in run_models if name not in benchmark_models]
    return {
        "benchmark_models": benchmark_models,
        "run_models": run_models,
        "benchmark_dataset": benchmark_dataset,
        "run_dataset": run_dataset,
        "missing_models": missing_models,
        "extra_models": extra_models,
        "dataset_changed": bool(benchmark_dataset and run_dataset and benchmark_dataset != run_dataset),
        "changed": bool(missing_models or extra_models or (benchmark_dataset and run_dataset and benchmark_dataset != run_dataset)),
    }


def _model_config_entry(config: dict[str, object] | None, model_name: str) -> dict[str, object] | None:
    benchmark = dict((config or {}).get("benchmark") or {})
    for model in benchmark.get("models") or []:
        if str(model.get("name") or "").strip() == model_name:
            return dict(model)
    return None


def _metrics_row_for_model(artifacts: dict[str, object], model_name: str) -> dict[str, object]:
    metrics = artifacts.get("metrics")
    if metrics is None or "model" not in metrics.columns:
        return {}
    subset = metrics[metrics["model"].astype(str) == model_name]
    if subset.empty:
        return {}
    return subset.iloc[0].replace({np.nan: None}).to_dict()


def _filter_frame_by_model(frame: pd.DataFrame | None, model_name: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    if "model" not in frame.columns:
        return frame.copy()
    subset = frame[frame["model"].astype(str) == model_name].copy()
    if subset.empty:
        return None
    return subset


def _diagnostics_for_model(artifacts: dict[str, object], model_name: str) -> dict[str, object]:
    diagnostics = dict(artifacts.get("diagnostics") or {})
    payload: dict[str, object] = {}
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
        payload["model_debug_artifacts"] = debug_artifacts[model_name]
    return payload


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


def _extract_model_log_text(artifacts: dict[str, object], model_name: str) -> str:
    debug_artifacts = dict((artifacts.get("diagnostics") or {}).get("model_debug_artifacts") or {})
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
    model_name: str,
    artifacts: dict[str, object],
    *,
    benchmark_name: str,
) -> str:
    config = dict(artifacts.get("config") or {})
    model_result = next(
        (
            dict(item)
            for item in (artifacts.get("model_results") or [])
            if str(item.get("model_name") or "").strip() == model_name
        ),
        {},
    )
    model_config = _model_config_entry(config, model_name) or {}
    diagnostics = _diagnostics_for_model(artifacts, model_name)
    dataset = artifacts.get("dataset")
    generated = dict(artifacts.get("generated_scenarios") or {}).get(model_name)
    training_payload = {
        "train_returns": None if dataset is None else np.asarray(dataset.train_returns),
        "contexts": None if dataset is None else np.asarray(dataset.contexts),
        "realized_futures": None if dataset is None else np.asarray(dataset.realized_futures),
        "reference_scenarios": artifacts.get("reference_scenarios"),
    }
    metric_payload = {
        "aggregated_metrics": _metrics_row_for_model(artifacts, model_name),
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
                    for key, value in diagnostics.items()
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
        _section("Model Logs", _extract_model_log_text(artifacts, model_name)),
    ]
    return "\n".join(sections)


def _model_overview_frame(artifacts: dict[str, object]) -> pd.DataFrame | None:
    metrics = artifacts.get("metrics")
    frame = None if metrics is None else metrics.copy()
    metric_names = _metric_names(artifacts, frame)
    model_results = artifacts.get("model_results") or []
    extras = []
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
    if "display_name" in frame.columns and any(str(value).strip() and str(value).strip() != str(model).strip() for value, model in zip(frame["display_name"], frame["model"])):
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
    return overview


def _render_diagnostics_tab(artifacts: dict[str, object]) -> None:
    diagnostics = artifacts.get("diagnostics") or {}
    if not diagnostics:
        st.info("This benchmark result did not save diagnostics.")
        return

    tabs = st.tabs(["Functional smoke", "Distribution", "Per-window"])

    with tabs[0]:
        summary = diagnostics.get("functional_smoke_summary")
        checks = diagnostics.get("functional_smoke_checks")
        if summary is None and checks is None:
            st.info("No functional smoke diagnostics were saved.")
        else:
            if summary is not None:
                st.dataframe(summary, use_container_width=True, hide_index=True)
            if checks is not None:
                st.dataframe(checks, use_container_width=True, hide_index=True)

    with tabs[1]:
        summary = diagnostics.get("distribution_summary")
        by_asset = diagnostics.get("distribution_summary_by_asset")
        if summary is None and by_asset is None:
            st.info("No distribution diagnostics were saved.")
        else:
            if summary is not None:
                st.dataframe(summary.round(6), use_container_width=True, hide_index=True)
            if by_asset is not None and not by_asset.empty:
                selected_model = st.selectbox(
                    "Distribution model",
                    options=sorted(by_asset["model"].unique()),
                    key="results.diagnostics.distribution.model",
                )
                st.dataframe(
                    by_asset[by_asset["model"] == selected_model].round(6),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[2]:
        per_window = diagnostics.get("per_window_metrics")
        if per_window is None or per_window.empty:
            st.info("No per-window diagnostics were saved.")
        else:
            model_name = st.selectbox(
                "Model",
                options=sorted(per_window["model"].unique()),
                key="results.diagnostics.window.model",
            )
            metric_columns = [
                column
                for column in per_window.columns
                if column not in {"model", "context_index", "evaluation_timestamp"}
            ]
            metric_name = st.selectbox(
                "Metric",
                options=metric_columns,
                key="results.diagnostics.window.metric",
            )
            subset = per_window[per_window["model"] == model_name].copy()
            st.line_chart(subset.set_index("context_index")[[metric_name]], use_container_width=True)
            st.dataframe(subset.round(6), use_container_width=True, hide_index=True)


def _render_technical_debug_tab(artifacts: dict[str, object], selected_row: dict[str, object], selected_run: Path) -> None:
    ranks = artifacts.get("ranks")
    if ranks is not None:
        st.subheader("Rank Table")
        st.dataframe(ranks, use_container_width=True, hide_index=True)

    model_results = artifacts.get("model_results") or []
    debug_artifacts = dict((artifacts.get("diagnostics") or {}).get("model_debug_artifacts") or {})
    metrics_frame = artifacts.get("metrics")
    metric_models = [] if metrics_frame is None else [
        str(name)
        for name in metrics_frame.get("model", pd.Series(dtype=object)).tolist()
        if str(name).strip()
    ]
    available_models = sorted(
        {
            *[str(item.get("model_name") or "") for item in model_results if str(item.get("model_name") or "").strip()],
            *[str(name) for name in debug_artifacts],
            *[
                str(name)
                for name in dict(artifacts.get("generated_scenarios") or {})
                if str(name).strip()
            ],
            *metric_models,
        }
    )
    if available_models:
        st.subheader("Model Result Details")
        by_name = {str(item.get("model_name")): item for item in model_results}
        selected_model = st.selectbox("Model", options=available_models, key="results.technical.model")
        report_text = _build_model_debug_report(
            selected_model,
            artifacts,
            benchmark_name=str(selected_row.get("name") or "benchmark"),
        )
        st.download_button(
            "Download model debug report",
            data=report_text,
            file_name=f"{str(selected_row.get('name') or 'benchmark').replace(' ', '_')}_{selected_model}_technical_debug.txt",
            mime="text/plain",
            key=f"results.technical.download.{selected_model}",
            use_container_width=True,
        )
        if selected_model in by_name:
            render_structured_value(
                by_name[selected_model],
                label=selected_model,
                editable=False,
                key_prefix=f"results.technical.model_result.{selected_model}",
            )
        else:
            st.info("No structured model result payload was saved for this model.")

    if debug_artifacts:
        st.subheader("Model Debug Artifacts")
        selected_model = str(st.session_state.get("results.technical.model") or sorted(debug_artifacts)[0])
        if selected_model in debug_artifacts:
            render_structured_value(
                debug_artifacts[selected_model],
                label=selected_model,
                editable=False,
                key_prefix=f"results.technical.debug_artifacts.{selected_model}",
            )
        else:
            st.info("No model-specific debug artifacts were saved for the selected model.")

    with st.expander("Run Record", expanded=False):
        render_structured_value(
            {
                "benchmark": {
                    "name": selected_row.get("name"),
                    "path": str(selected_row.get("path")),
                    "results_updated_at": selected_row.get("results_updated_at"),
                },
                "run_dir": str(selected_run),
                "run": artifacts.get("run") or {},
                "summary": artifacts.get("summary") or {},
                "tracking": artifacts.get("tracking") or {},
            },
            label="run",
            editable=False,
            key_prefix="results.technical.run_record",
        )

    with st.expander("Resolved Config", expanded=False):
        render_structured_value(
            artifacts.get("config") or {},
            label="config",
            editable=False,
            key_prefix="results.technical.config",
        )


def render() -> None:
    st.header("Results Explorer")
    st.caption("Inspect official benchmark outcomes, diagnostics, scenario previews, and technical run details.")

    benchmark_rows = list_saved_benchmarks()
    if not benchmark_rows:
        st.info("No official benchmarks are available yet.")
        return

    default_benchmark = _default_selected_benchmark(benchmark_rows)
    if default_benchmark is None:
        st.info("No benchmark results are available yet.")
        return
    benchmark_options = {str(row["name"]): row for row in benchmark_rows}
    selected_benchmark = st.selectbox(
        "Benchmark",
        options=list(benchmark_options),
        index=list(benchmark_options).index(default_benchmark),
    )
    selected_row = dict(benchmark_options[selected_benchmark])
    selected_run_raw = selected_row.get("results_run_dir")
    if not selected_run_raw:
        st.info("This benchmark has no saved results yet.")
        return
    selected_run = Path(str(selected_run_raw))
    set_selected_run_dir(selected_run)

    artifacts = load_run_artifacts(selected_run)
    benchmark_config = load_config_dict(Path(str(selected_row["path"])))
    dataset_error = artifacts.get("dataset_error")
    summary = dict(artifacts.get("summary") or {})
    run_record = dict(artifacts.get("run") or {})
    model_table = _model_overview_frame(artifacts)
    alignment = _benchmark_results_alignment(benchmark_config, dict(artifacts.get("config") or {}))
    dataset_label = (
        selected_row.get("dataset")
        or dict(summary.get("dataset") or {}).get("resolved_name")
        or dict(summary.get("dataset") or {}).get("name")
        or "Unknown dataset"
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Dataset", str(dataset_label))
    summary_cols[1].metric(
        "Models",
        str(len(model_table.index) if model_table is not None else len(artifacts.get("model_results") or [])),
    )
    summary_cols[2].metric("Run status", str(run_record.get("status") or "unknown"))
    best_model = "n/a"
    if model_table is not None and not model_table.empty:
        best_model = str(model_table.iloc[0]["model"])
    summary_cols[3].metric("Best model", best_model)
    if selected_row.get("results_updated_at"):
        st.caption(f"Results updated: {selected_row['results_updated_at']}")
    if dataset_error:
        st.warning(f"Dataset reconstruction for scenario preview failed: {dataset_error}")
    run_models = alignment["run_models"]
    if run_models:
        st.caption(f"Executed models in latest results: {', '.join(run_models)}")
    if alignment["changed"]:
        messages: list[str] = []
        if alignment["dataset_changed"]:
            messages.append(
                "dataset changed "
                f"({alignment['run_dataset']} in latest results vs {alignment['benchmark_dataset']} in the saved benchmark)"
            )
        if alignment["missing_models"]:
            messages.append(
                "current benchmark models missing from latest results: "
                + ", ".join(str(name) for name in alignment["missing_models"])
            )
        if alignment["extra_models"]:
            messages.append(
                "latest results still include models no longer in the saved benchmark: "
                + ", ".join(str(name) for name in alignment["extra_models"])
            )
        st.warning(
            "Latest saved results do not match the current saved benchmark definition: "
            + "; ".join(messages)
            + "."
        )
    failed_models = [
        item
        for item in (artifacts.get("model_results") or [])
        if str(dict(item.get("metadata") or {}).get("error") or "").strip()
    ]
    if failed_models:
        failed_names = ", ".join(str(item.get("model_name") or "") for item in failed_models)
        st.warning(
            f"This run is partial. Failed models: {failed_names}. "
            "See the Models or Technical debug tab for the recorded error details."
        )

    tabs = st.tabs(["Models", "Scenarios", "Diagnostics", "Technical debug", "Compare"])

    with tabs[0]:
        if model_table is None or model_table.empty:
            st.info("No model metrics were saved for this benchmark result.")
        else:
            st.dataframe(model_table, use_container_width=True, hide_index=True)

    with tabs[1]:
        generated = artifacts.get("generated_scenarios") or {}
        dataset = artifacts.get("dataset")
        if not generated:
            st.info("This run did not save scenarios.")
        elif dataset is None:
            if dataset_error:
                st.error(f"Scenario preview is unavailable because the dataset could not be rebuilt: {dataset_error}")
            else:
                st.info("Scenario preview requires a rebuildable dataset.")
        else:
            model_name = st.selectbox("Model", options=list(generated), key="results.scenarios.model")
            asset_names = list(dataset.asset_names)
            asset_name = st.selectbox("Asset", options=asset_names, key="results.scenarios.asset")
            eval_window = st.slider(
                "Evaluation window",
                min_value=0,
                max_value=int(dataset.contexts.shape[0] - 1),
                value=0,
                key="results.scenarios.window",
            )
            asset_index = asset_names.index(asset_name)
            band_df = _scenario_band_dataframe(
                generated[model_name][eval_window],
                dataset.realized_futures[eval_window],
                asset_index,
            )
            st.line_chart(band_df, use_container_width=True)

    with tabs[2]:
        _render_diagnostics_tab(artifacts)

    with tabs[3]:
        _render_technical_debug_tab(artifacts, selected_row, selected_run)

    with tabs[4]:
        compare_rows = [row for row in benchmark_rows if row.get("results_run_dir")]
        compare_options = {str(row["name"]): Path(str(row["results_run_dir"])) for row in compare_rows}
        if compare_options:
            compare_default = get_compare_run_dir()
            compare_names = list(compare_options)
            default_index = 0
            if compare_default is not None and compare_default.name in compare_options:
                default_index = compare_names.index(compare_default.name)
            compare_name = st.selectbox("Compare against", options=compare_names, index=default_index, key="results.compare.name")
            compare_dir = compare_options[compare_name]
            set_compare_run_dir(compare_dir)
            compare_artifacts = load_run_artifacts(compare_dir)
            left = artifacts.get("metrics")
            right = compare_artifacts.get("metrics")
            if left is not None and right is not None:
                left_indexed = left.set_index("model") if "model" in left.columns else left
                right_indexed = right.set_index("model") if "model" in right.columns else right
                merged = left_indexed.add_suffix("_current").join(right_indexed.add_suffix("_compare"), how="outer")
                st.dataframe(merged, use_container_width=True)
