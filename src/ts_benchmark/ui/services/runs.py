"""Run-management helpers for the Streamlit UI."""

from __future__ import annotations

import json
import os
import copy
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ts_benchmark.benchmark import benchmark_key_for_path, load_benchmark_config, packaged_baseline_dir
from ts_benchmark.dataset import build_dataset
from ts_benchmark.metrics import rank_metrics_table
from ts_benchmark.serialization import to_jsonable

from .. import APP_ROOT, BENCHMARK_RESULTS_DIR, OUTPUT_DIR, RUNTIME_DIR, SRC_ROOT

_EXTRA_RESULTS_DIRS_ENV = "TS_BENCHMARK_UI_EXTRA_RESULTS_DIRS"


def build_temp_config_file(config: dict[str, Any]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=RUNTIME_DIR,
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(config, tmp, indent=2)
    finally:
        tmp.close()
    return Path(tmp.name)


def read_log_tail(log_path: Path, max_lines: int = 200) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _resolve_cli_command() -> tuple[list[str], bool]:
    cli_path = shutil.which("ts-benchmark")
    if cli_path:
        return [cli_path], False
    return [sys.executable, "-m", "ts_benchmark.cli.main"], True


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    value = str(path)
    existing = env.get("PYTHONPATH")
    if not existing:
        env["PYTHONPATH"] = value
        return
    entries = existing.split(os.pathsep)
    if value in entries:
        return
    env["PYTHONPATH"] = os.pathsep.join([value, *entries])


def launch_cli_run(
    config: dict[str, Any],
    *,
    cwd: Path = APP_ROOT,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    config_path = build_temp_config_file(config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = RUNTIME_DIR / f"run_{stamp}.log"
    command_prefix, needs_source_pythonpath = _resolve_cli_command()
    command = [*command_prefix, "run", str(config_path)]
    env = os.environ.copy()
    if needs_source_pythonpath:
        _prepend_pythonpath(env, SRC_ROOT)
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "process": process,
        "pid": process.pid,
        "command": command,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "status": "running",
        "output_dir": dict(config.get("run") or {}).get("output", {}).get("output_dir"),
        "started_at": stamp,
    }


def poll_cli_run(run_handle: dict[str, Any]) -> dict[str, Any]:
    updated = dict(run_handle)
    process = updated.get("process")
    if process is None:
        updated["status"] = "unknown"
        return updated
    returncode = process.poll()
    if returncode is None:
        updated["status"] = "running"
    else:
        updated["returncode"] = int(returncode)
        updated["status"] = "succeeded" if returncode == 0 else "failed"
    return updated


def discover_local_runs(output_root: Path = OUTPUT_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return pd.DataFrame(rows)
    for path in sorted(output_root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        metrics_path = path / "metrics.csv"
        summary_path = path / "summary.json"
        if not metrics_path.exists():
            continue
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        rows.append(
            {
                "run_dir": str(path),
                "name": path.name,
                "dataset_name": (
                    summary.get("dataset", {}).get("resolved_name")
                    or summary.get("dataset", {}).get("name")
                ),
                "models": ", ".join(summary.get("models", [])) if isinstance(summary.get("models"), list) else None,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _extra_results_roots() -> list[Path]:
    raw = str(os.getenv(_EXTRA_RESULTS_DIRS_ENV) or "").strip()
    if not raw:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    for part in raw.split(os.pathsep):
        value = str(part).strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _looks_like_results_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metrics.csv").exists()


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_model_results(path: Path) -> list[dict[str, Any]] | None:
    payload = _read_optional_json(path)
    if payload is None:
        return None
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        rows: list[dict[str, Any]] = []
        for model_name, item in payload.items():
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("model_name", str(model_name))
            rows.append(row)
        return rows
    return None


def _load_diagnostics(run_dir: Path) -> dict[str, Any]:
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return {}
    payload: dict[str, Any] = {}
    for name in [
        "distribution_summary",
        "distribution_summary_by_asset",
        "per_window_metrics",
        "functional_smoke_summary",
        "functional_smoke_checks",
    ]:
        frame = _read_optional_csv(diagnostics_dir / f"{name}.csv")
        if frame is not None:
            payload[name] = frame
    debug_dir = diagnostics_dir / "model_debug_artifacts"
    if debug_dir.exists():
        payload["model_debug_artifacts"] = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(debug_dir.glob("*.json"))
        }
    return payload


def _load_scenarios(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "scenarios.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    generated = {
        key.replace("model__", "", 1): data[key]
        for key in data.files
        if key.startswith("model__")
    }
    reference = data["reference_scenarios"] if "reference_scenarios" in data.files else None
    return {
        "generated_scenarios": generated,
        "reference_scenarios": reference,
    }


def _rebuild_dataset(
    config_dict: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> tuple[Any | None, str | None]:
    try:
        config = load_benchmark_config(config_path) if config_path is not None else load_benchmark_config(config_dict)
        dataset = build_dataset(
            config.dataset,
            config.protocol,
            seed=config.run.seed,
            source_path=config.source_path,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return dataset, None


def _read_results_config(run_dir: Path) -> dict[str, Any] | None:
    return _read_optional_json(run_dir / "benchmark_config.json") or _read_optional_json(
        run_dir / "effective_benchmark.json"
    )


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = run_dir / "benchmark_config.json"
    if not config_path.exists():
        config_path = run_dir / "effective_benchmark.json"
    run_payload = _read_optional_json(run_dir / "run.json")
    payload = {
        "run_dir": run_dir,
        "metrics": _read_optional_csv(run_dir / "metrics.csv"),
        "ranks": _read_optional_csv(run_dir / "ranks.csv"),
        "config": _read_results_config(run_dir),
        "run": run_payload,
        "model_results": _read_model_results(run_dir / "model_results.json")
        or _read_model_results(run_dir / "model_infos.json"),
        "summary": _read_optional_json(run_dir / "summary.json"),
        "tracking": None if run_payload is None else dict(run_payload.get("tracking_result") or {}),
    }
    payload["diagnostics"] = _load_diagnostics(run_dir)
    payload.update(_load_scenarios(run_dir))
    if payload["config"] is None:
        payload["dataset"] = None
        payload["dataset_error"] = None
    else:
        payload["dataset"], payload["dataset_error"] = _rebuild_dataset(payload["config"], config_path=config_path)
    return payload


def benchmark_results_dir_for_path(benchmark_path: Path) -> Path:
    benchmark_path = Path(benchmark_path).expanduser().resolve()
    return (BENCHMARK_RESULTS_DIR / benchmark_key_for_path(benchmark_path)).resolve()


def previous_results_dir_for_path(benchmark_path: Path) -> Path | None:
    benchmark_path = Path(benchmark_path).expanduser().resolve()
    sibling_results_dir = benchmark_path.parent
    if _looks_like_results_dir(sibling_results_dir):
        return sibling_results_dir
    local_results_dir = benchmark_results_dir_for_path(benchmark_path)
    if local_results_dir.exists():
        return local_results_dir
    benchmark_key = benchmark_key_for_path(benchmark_path)
    for root in _extra_results_roots():
        if _looks_like_results_dir(root) and (root == sibling_results_dir or (root / benchmark_path.name).exists()):
            return root
        candidate = root / benchmark_key
        if _looks_like_results_dir(candidate):
            return candidate.resolve()
    baseline_dir = packaged_baseline_dir(benchmark_path)
    if baseline_dir is not None:
        return baseline_dir
    return None


def _strip_config_for_results_merge(config_dict: dict[str, Any]) -> dict[str, Any]:
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


def _configs_compatible_for_merge(previous_config: dict[str, Any] | None, current_config: dict[str, Any]) -> bool:
    if previous_config is None:
        return False
    return _strip_config_for_results_merge(previous_config) == _strip_config_for_results_merge(current_config)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = _read_model_results(path)
    if payload is None:
        return []
    return payload


def _read_model_results_list(run_dir: Path) -> list[dict[str, Any]]:
    return _read_json_list(run_dir / "model_results.json") or _read_json_list(run_dir / "model_infos.json")


def _merge_model_results(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {str(item.get("model_name")): copy.deepcopy(item) for item in previous if item.get("model_name")}
    for item in current:
        name = str(item.get("model_name") or "")
        if not name:
            continue
        merged[name] = copy.deepcopy(item)
    return list(merged.values())


def _configured_model_names(benchmark_config: dict[str, Any]) -> list[str]:
    benchmark = dict(benchmark_config.get("benchmark") or {})
    return [
        str(model.get("name") or "")
        for model in benchmark.get("models") or []
        if str(model.get("name") or "").strip()
    ]


def _filter_metrics_frame_by_models(
    frame: pd.DataFrame | None,
    allowed_model_names: set[str],
) -> pd.DataFrame | None:
    if frame is None or "model" not in frame.columns:
        return frame
    return frame[frame["model"].astype(str).isin(allowed_model_names)].copy()


def _filter_model_results_by_models(
    model_results: list[dict[str, Any]],
    allowed_model_names: set[str],
) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in model_results
        if str(item.get("model_name") or "") in allowed_model_names
    ]


def _merged_metrics_frames(
    previous_metrics: pd.DataFrame | None,
    current_metrics: pd.DataFrame | None,
    metric_configs,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if current_metrics is None and previous_metrics is None:
        return None, None
    frames = [frame for frame in [previous_metrics, current_metrics] if frame is not None]
    merged = pd.concat(frames, ignore_index=False)
    if "model" not in merged.columns:
        return merged, None
    merged = merged.drop_duplicates(subset=["model"], keep="last").set_index("model")
    metric_names = [metric.name for metric in metric_configs]
    metadata_columns = [column for column in merged.columns if column not in {*metric_names, "average_rank"}]
    if any(name not in merged.columns for name in metric_names):
        merged_metrics = merged.reset_index()
        if "average_rank" not in merged_metrics.columns:
            merged_metrics["average_rank"] = np.nan
        return merged_metrics, None

    merged_metrics = merged.copy()
    if "average_rank" not in merged_metrics.columns:
        merged_metrics["average_rank"] = np.nan
    merged_ranks = pd.DataFrame(index=merged.index)
    for name in metric_names:
        merged_ranks[name] = np.nan
    merged_ranks["average_rank"] = np.nan

    complete_mask = merged[metric_names].notna().all(axis=1)
    if complete_mask.any():
        filtered, rank_table = rank_metrics_table(merged.loc[complete_mask, metric_names], metric_configs)
        for column in metric_names + ["average_rank"]:
            merged_metrics.loc[filtered.index, column] = filtered[column]
        merged_ranks.loc[rank_table.index, metric_names] = rank_table[metric_names]
        merged_ranks.loc[rank_table.index, "average_rank"] = rank_table["average_rank"]

    merged_metrics = merged_metrics.reset_index()
    for column in reversed(metadata_columns):
        merged_ranks.insert(0, column, merged[column])
    merged_ranks = merged_ranks.reset_index()
    if "average_rank" in merged_metrics.columns:
        merged_metrics = merged_metrics.sort_values("average_rank", ascending=True, na_position="last")
    if "average_rank" in merged_ranks.columns:
        merged_ranks = merged_ranks.sort_values("average_rank", ascending=True, na_position="last")
    return merged_metrics, merged_ranks


def _apply_rankings_to_model_results(
    model_results: list[dict[str, Any]],
    merged_metrics: pd.DataFrame | None,
    merged_ranks: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if merged_metrics is None or merged_ranks is None or "model" not in merged_metrics.columns:
        return model_results
    metrics_by_model = merged_metrics.set_index("model").to_dict(orient="index")
    ranks_by_model = merged_ranks.set_index("model").to_dict(orient="index")
    ordered_names = list(merged_metrics["model"])
    merged_by_name = {str(item.get("model_name")): copy.deepcopy(item) for item in model_results if item.get("model_name")}
    updated: list[dict[str, Any]] = []
    updated_names: set[str] = set()
    for name in ordered_names:
        item = merged_by_name.get(str(name))
        if item is None:
            continue
        metric_values = metrics_by_model.get(str(name), {})
        rank_values = ranks_by_model.get(str(name), {})
        metric_names = {
            str(metric_result.get("metric_name") or "")
            for metric_result in item.get("metric_results") or []
            if str(metric_result.get("metric_name") or "")
        }
        metric_results = []
        for metric_result in item.get("metric_results") or []:
            metric_name = str(metric_result.get("metric_name") or "")
            if metric_name in metric_values:
                updated_metric = dict(metric_result)
                updated_metric["value"] = float(metric_values[metric_name])
                metric_results.append(updated_metric)
        item["metric_results"] = metric_results
        item["metric_rankings"] = [
            {
                "model_name": str(name),
                "metric_name": metric_name,
                "rank": float(rank),
            }
            for metric_name, rank in rank_values.items()
            if metric_name != "average_rank" and metric_name in metric_names and not pd.isna(rank)
        ]
        average_rank = rank_values.get("average_rank")
        item["average_rank"] = None if average_rank is None or pd.isna(average_rank) else float(average_rank)
        updated.append(item)
        updated_names.add(str(name))

    for item in model_results:
        name = str(item.get("model_name") or "")
        if not name or name in updated_names:
            continue
        residual = copy.deepcopy(item)
        residual["metric_rankings"] = []
        residual["average_rank"] = None
        updated.append(residual)
    return updated


def _merge_summary(
    previous_summary: dict[str, Any] | None,
    current_summary: dict[str, Any] | None,
    model_names: list[str],
) -> dict[str, Any]:
    summary = dict(previous_summary or {})
    summary.update(dict(current_summary or {}))
    summary["models"] = list(model_names)
    return summary


def _merge_run_record(
    previous_run: dict[str, Any] | None,
    current_run: dict[str, Any] | None,
    *,
    source_run_dir: Path,
    previous_results_dir: Path | None,
) -> dict[str, Any]:
    run_payload = dict(previous_run or {})
    run_payload.update(dict(current_run or {}))
    metadata = dict(run_payload.get("metadata") or {})
    benchmark_merge = dict(metadata.get("benchmark_merge") or {})
    if previous_results_dir is not None:
        benchmark_merge["previous_results_dir"] = str(previous_results_dir)
    benchmark_merge["source_run_dir"] = str(source_run_dir)
    metadata["benchmark_merge"] = benchmark_merge
    run_payload["metadata"] = metadata
    return run_payload


def _merge_scenarios(
    previous_dir: Path | None,
    current_dir: Path,
    destination_dir: Path,
    *,
    allowed_model_names: set[str],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for source_dir in [previous_dir, current_dir]:
        if source_dir is None:
            continue
        path = source_dir / "scenarios.npz"
        if not path.exists():
            continue
        loaded = np.load(path, allow_pickle=False)
        for key in loaded.files:
            if key.startswith("model__"):
                model_name = key.replace("model__", "", 1)
                if model_name not in allowed_model_names:
                    continue
            arrays[key] = loaded[key]
    if arrays:
        np.savez_compressed(destination_dir / "scenarios.npz", **arrays)


_DIAGNOSTIC_KEY_COLUMNS = {
    "distribution_summary.csv": ("model", "series_type"),
    "distribution_summary_by_asset.csv": ("model", "series_type", "asset"),
    "per_window_metrics.csv": ("model", "context_index"),
    "functional_smoke_summary.csv": ("model",),
    "functional_smoke_checks.csv": ("model", "check"),
}


def _merge_diagnostic_csv(
    previous_dir: Path | None,
    current_dir: Path,
    destination_dir: Path,
    filename: str,
) -> None:
    previous = None if previous_dir is None else _read_optional_csv(previous_dir / "diagnostics" / filename)
    current = _read_optional_csv(current_dir / "diagnostics" / filename)
    if previous is None and current is None:
        return
    target_dir = destination_dir / "diagnostics"
    target_dir.mkdir(parents=True, exist_ok=True)
    if previous is None:
        current.to_csv(target_dir / filename, index=False)  # type: ignore[union-attr]
        return
    if current is None:
        previous.to_csv(target_dir / filename, index=False)
        return
    key_columns = [column for column in _DIAGNOSTIC_KEY_COLUMNS.get(filename, ()) if column in current.columns]
    if key_columns:
        merged = pd.concat([previous, current], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep="last")
    else:
        merged = current
    merged.to_csv(target_dir / filename, index=False)


def _merge_debug_artifacts(previous_dir: Path | None, current_dir: Path, destination_dir: Path) -> None:
    previous = None if previous_dir is None else previous_dir / "diagnostics" / "model_debug_artifacts"
    current = current_dir / "diagnostics" / "model_debug_artifacts"
    if (previous is None or not previous.exists()) and not current.exists():
        return
    target = destination_dir / "diagnostics" / "model_debug_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    for source in [previous, current]:
        if source is None or not source.exists():
            continue
        for path in sorted(source.glob("*.json")):
            shutil.copy2(path, target / path.name)


def _merge_diagnostics(previous_dir: Path | None, current_dir: Path, destination_dir: Path) -> None:
    current_diagnostics = current_dir / "diagnostics"
    previous_diagnostics = None if previous_dir is None else previous_dir / "diagnostics"
    if (previous_diagnostics is None or not previous_diagnostics.exists()) and not current_diagnostics.exists():
        return
    for filename in _DIAGNOSTIC_KEY_COLUMNS:
        _merge_diagnostic_csv(previous_dir, current_dir, destination_dir, filename)
    _merge_debug_artifacts(previous_dir, current_dir, destination_dir)


def _prepare_previous_results_dir(
    previous_results_dir: Path | None,
    destination_dir: Path,
) -> tuple[Path | None, Path | None]:
    if previous_results_dir is None or not previous_results_dir.exists():
        return previous_results_dir, None
    if previous_results_dir != destination_dir:
        return previous_results_dir, None
    snapshot_root = Path(tempfile.mkdtemp(prefix="benchmark_results_previous_", dir=RUNTIME_DIR))
    snapshot_dir = snapshot_root / "snapshot"
    shutil.copytree(previous_results_dir, snapshot_dir)
    return snapshot_dir, snapshot_root


def materialize_benchmark_results(
    *,
    benchmark_path: Path,
    benchmark_config: dict[str, Any],
    source_run_dir: Path,
    previous_results_dir: Path | None = None,
    destination_dir: Path | None = None,
) -> Path:
    benchmark_path = Path(benchmark_path).expanduser().resolve()
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    previous_results_dir = None if previous_results_dir is None else Path(previous_results_dir).expanduser().resolve()
    previous_results_input_dir = previous_results_dir
    destination_dir = (
        benchmark_results_dir_for_path(benchmark_path)
        if destination_dir is None
        else Path(destination_dir).expanduser().resolve()
    )
    allowed_model_names = set(_configured_model_names(benchmark_config))
    previous_results_dir, snapshot_root = _prepare_previous_results_dir(previous_results_dir, destination_dir)
    try:
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        previous_config = _read_results_config(previous_results_dir) if previous_results_dir else None
        can_merge = previous_results_dir is not None and _configs_compatible_for_merge(previous_config, benchmark_config)

        current_config = _read_results_config(source_run_dir) or benchmark_config
        dumped_config = to_jsonable(benchmark_config)
        with (destination_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
            json.dump(dumped_config, handle, indent=2)
        with (destination_dir / "effective_benchmark.json").open("w", encoding="utf-8") as handle:
            json.dump(dumped_config, handle, indent=2)

        previous_metrics = _read_optional_csv(previous_results_dir / "metrics.csv") if can_merge and previous_results_dir else None
        current_metrics = _read_optional_csv(source_run_dir / "metrics.csv")
        previous_metrics = _filter_metrics_frame_by_models(previous_metrics, allowed_model_names)
        current_metrics = _filter_metrics_frame_by_models(current_metrics, allowed_model_names)
        metric_configs = load_benchmark_config(current_config).metrics
        merged_metrics, merged_ranks = _merged_metrics_frames(previous_metrics, current_metrics, metric_configs)
        if merged_metrics is not None:
            merged_metrics.to_csv(destination_dir / "metrics.csv", index=False)
        elif current_metrics is not None:
            current_metrics.to_csv(destination_dir / "metrics.csv", index=False)
        if merged_ranks is not None:
            merged_ranks.to_csv(destination_dir / "ranks.csv", index=False)
        else:
            current_ranks = _read_optional_csv(source_run_dir / "ranks.csv")
            if current_ranks is not None:
                current_ranks.to_csv(destination_dir / "ranks.csv", index=False)

        previous_model_results = _read_model_results_list(previous_results_dir) if can_merge and previous_results_dir else []
        current_model_results = _read_model_results_list(source_run_dir)
        previous_model_results = _filter_model_results_by_models(previous_model_results, allowed_model_names)
        current_model_results = _filter_model_results_by_models(current_model_results, allowed_model_names)
        merged_model_results = _merge_model_results(previous_model_results, current_model_results)
        merged_model_results = _apply_rankings_to_model_results(merged_model_results, merged_metrics, merged_ranks)
        if merged_model_results:
            with (destination_dir / "model_results.json").open("w", encoding="utf-8") as handle:
                json.dump(to_jsonable(merged_model_results), handle, indent=2)

        model_names: list[str]
        if merged_model_results:
            model_names = [str(item.get("model_name")) for item in merged_model_results if item.get("model_name")]
        elif merged_metrics is not None and "model" in merged_metrics.columns:
            model_names = [str(name) for name in merged_metrics["model"].tolist()]
        else:
            model_names = list(((_read_optional_json(source_run_dir / "summary.json") or {}).get("models") or []))

        summary = _merge_summary(
            _read_optional_json(previous_results_dir / "summary.json") if can_merge and previous_results_dir else None,
            _read_optional_json(source_run_dir / "summary.json"),
            model_names,
        )
        if summary:
            with (destination_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(to_jsonable(summary), handle, indent=2)

        run_payload = _merge_run_record(
            _read_optional_json(previous_results_dir / "run.json") if can_merge and previous_results_dir else None,
            _read_optional_json(source_run_dir / "run.json"),
            source_run_dir=source_run_dir,
            previous_results_dir=previous_results_input_dir if can_merge else None,
        )
        with (destination_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(run_payload), handle, indent=2)

        _merge_scenarios(
            previous_results_dir if can_merge else None,
            source_run_dir,
            destination_dir,
            allowed_model_names=allowed_model_names,
        )
        _merge_diagnostics(previous_results_dir if can_merge else None, source_run_dir, destination_dir)
        return destination_dir
    finally:
        if snapshot_root is not None and snapshot_root.exists():
            shutil.rmtree(snapshot_root, ignore_errors=True)
