"""Config-oriented services for the Streamlit UI."""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ts_benchmark.benchmark import benchmark_key_for_path, load_benchmark_config, shipped_benchmark_paths
from ts_benchmark.serialization import to_jsonable

from .. import BENCHMARK_CATALOG_DIR, OUTPUT_DIR, SAMPLE_DATA_DIR

_EXTRA_BENCHMARK_DIRS_ENV = "TS_BENCHMARK_UI_EXTRA_BENCHMARK_DIRS"
_IGNORED_EXTRA_BENCHMARK_FILENAMES = {
    "benchmark_config.json",
    "effective_benchmark.json",
    "manifest.json",
    "model_results.json",
    "model_infos.json",
    "run.json",
    "summary.json",
}


def _extra_catalog_paths(env_name: str) -> list[Path]:
    raw = str(os.getenv(env_name) or "").strip()
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


def _benchmark_paths_from_root(root: Path) -> dict[str, Path]:
    if root.is_file():
        if root.suffix.lower() != ".json":
            return {}
        return {benchmark_key_for_path(root): root.resolve()}
    return {
        key: path
        for key, path in shipped_benchmark_paths(config_dir=root).items()
        if path.name not in _IGNORED_EXTRA_BENCHMARK_FILENAMES and not path.name.endswith(".meta.json")
    }


def extra_benchmark_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for root in _extra_catalog_paths(_EXTRA_BENCHMARK_DIRS_ENV):
        for key, path in _benchmark_paths_from_root(root).items():
            paths.setdefault(key, path)
    return paths


def example_paths() -> dict[str, Path]:
    return shipped_benchmark_paths()


def dataset_example_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pattern in ("*.csv", "*.parquet"):
        for path in sorted(SAMPLE_DATA_DIR.glob(pattern)):
            out[path.stem] = path
    return out


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return slug or "benchmark"


def default_config_dict() -> dict[str, Any]:
    return {
        "version": "1.0",
        "benchmark": {
            "name": "new_benchmark",
            "description": "",
            "dataset": {
                "provider": {"kind": "synthetic", "config": {"generator": "regime_switching_factor_sv", "params": {}}},
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {},
            "metrics": [{"name": "crps"}, {"name": "energy_score"}],
            "models": [],
        },
        "run": {
            "execution": {
                "scheduler": "auto",
                "device": None,
                "model_execution": {"mode": "inprocess"},
            },
            "tracking": {"mlflow": {}},
            "output": {},
        },
    }


def current_config_summary(config: dict[str, Any] | None) -> dict[str, Any]:
    benchmark = {} if config is None else dict(config.get("benchmark") or {})
    models = benchmark.get("models") or []
    metrics = benchmark.get("metrics") or []
    dataset = dict(benchmark.get("dataset") or {})
    return {
        "name": benchmark.get("name") or "<unsaved benchmark>",
        "description": benchmark.get("description") or "",
        "dataset": dataset.get("name") or dict(dataset.get("provider") or {}).get("kind") or "",
        "models": len(models),
        "metrics": len(metrics),
    }


def load_config_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_config_dict(path_or_text: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_text, dict):
        return copy.deepcopy(path_or_text)
    if isinstance(path_or_text, Path):
        return json.loads(path_or_text.read_text(encoding="utf-8"))
    text = str(path_or_text).strip()
    if text.startswith("{"):
        return json.loads(text)
    return json.loads(Path(text).read_text(encoding="utf-8"))


def save_config_dict(path: Path, config: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(config), indent=2), encoding="utf-8")


def saved_benchmark_paths(benchmark_dir: Path | None = None) -> dict[str, Path]:
    if benchmark_dir is None:
        paths = dict(shipped_benchmark_paths())
        for key, path in extra_benchmark_paths().items():
            paths.setdefault(key, path)
        return paths
    return {
        path.stem: path
        for path in sorted(benchmark_dir.glob("*.json"))
        if not path.name.endswith(".meta.json")
    }


def _resolve_saved_benchmark_path(name: str, benchmark_dir: Path | None = None) -> Path | None:
    target = str(name).strip()
    if not target:
        return None
    if benchmark_dir is None:
        shipped = shipped_benchmark_paths()
        slug = _slugify_name(target)
        if slug in shipped:
            return shipped[slug]
    else:
        slug_match = benchmark_dir / f"{_slugify_name(target)}.json"
        if slug_match.exists():
            return slug_match
    for path in saved_benchmark_paths(benchmark_dir).values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        benchmark = dict(payload.get("benchmark") or {})
        if str(benchmark.get("name") or "").strip() == target:
            return path
    return None


def load_saved_benchmark(name: str, benchmark_dir: Path | None = None) -> dict[str, Any]:
    path = _resolve_saved_benchmark_path(name, benchmark_dir=benchmark_dir)
    if path is None:
        raise FileNotFoundError(f"No saved benchmark named '{name}'.")
    return load_config_dict(path)


def save_benchmark_definition(config: dict[str, Any], benchmark_dir: Path | None = None) -> Path:
    if benchmark_dir is None:
        raise ValueError("Official benchmarks are read-only. Save benchmark definitions to an explicit directory.")
    benchmark = dict(config.get("benchmark") or {})
    name = str(benchmark.get("name") or "").strip()
    if not name:
        raise ValueError("Saved benchmarks require a non-empty benchmark name.")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    path = benchmark_dir / f"{_slugify_name(name)}.json"
    save_config_dict(path, _apply_run_model_execution_defaults(config))
    return path


def _benchmark_metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.meta.json")


def _load_benchmark_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _benchmark_metadata_path(path)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_benchmark_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = _benchmark_metadata_path(path)
    metadata_path.write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata_path


def load_saved_benchmark_metadata(name_or_path: str | Path, benchmark_dir: Path | None = None) -> dict[str, Any]:
    if benchmark_dir is None:
        return {}
    if isinstance(name_or_path, Path):
        path = name_or_path
    else:
        path = _resolve_saved_benchmark_path(name_or_path, benchmark_dir=benchmark_dir)
        if path is None:
            raise FileNotFoundError(f"No saved benchmark named '{name_or_path}'.")
    return _load_benchmark_metadata(path)


def update_saved_benchmark_results(
    name_or_path: str | Path,
    run_dir: str | Path,
    *,
    summary: dict[str, Any] | None = None,
    benchmark_dir: Path | None = None,
) -> Path:
    if benchmark_dir is None:
        raise ValueError("Official benchmarks are read-only. Result metadata is derived from packaged or local run outputs.")
    if isinstance(name_or_path, Path):
        path = name_or_path
    else:
        path = _resolve_saved_benchmark_path(name_or_path, benchmark_dir=benchmark_dir)
        if path is None:
            raise FileNotFoundError(f"No saved benchmark named '{name_or_path}'.")
    metadata = _load_benchmark_metadata(path)
    metadata["latest_results"] = {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": to_jsonable(summary or {}),
    }
    _save_benchmark_metadata(path, metadata)
    return path


def delete_saved_benchmark(name: str, benchmark_dir: Path | None = None) -> Path:
    if benchmark_dir is None:
        raise ValueError("Official benchmarks are read-only and cannot be deleted.")
    path = _resolve_saved_benchmark_path(name, benchmark_dir=benchmark_dir)
    if path is None:
        raise FileNotFoundError(f"No saved benchmark named '{name}'.")
    metadata_path = _benchmark_metadata_path(path)
    path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    return path


def _result_summary_for_run_dir(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    summary_path = run_dir / "summary.json"
    try:
        return dict(json.loads(summary_path.read_text(encoding="utf-8"))) if summary_path.exists() else {}
    except Exception:
        return {}


def list_saved_benchmarks(benchmark_dir: Path | None = None) -> list[dict[str, Any]]:
    from .runs import previous_results_dir_for_path

    rows: list[dict[str, Any]] = []
    for path in saved_benchmark_paths(benchmark_dir).values():
        try:
            payload = load_config_dict(path)
        except Exception:
            continue
        benchmark = dict(payload.get("benchmark") or {})
        dataset = dict(benchmark.get("dataset") or {})
        metrics = list(benchmark.get("metrics") or [])
        models = list(benchmark.get("models") or [])
        if benchmark_dir is None:
            result_dir = previous_results_dir_for_path(path)
            updated_at = None
            if result_dir is not None and result_dir.exists():
                updated_at = datetime.fromtimestamp(result_dir.stat().st_mtime, tz=timezone.utc).isoformat()
            results_summary = _result_summary_for_run_dir(result_dir)
        else:
            metadata = _load_benchmark_metadata(path)
            latest_results = dict(metadata.get("latest_results") or {})
            result_dir = latest_results.get("run_dir")
            updated_at = latest_results.get("updated_at")
            results_summary = dict(latest_results.get("summary") or {})
        rows.append(
            {
                "name": benchmark.get("name") or path.stem,
                "description": benchmark.get("description") or "",
                "dataset": dataset.get("name") or dict(dataset.get("provider") or {}).get("kind") or "",
                "models": len(models),
                "metrics": len(metrics),
                "path": path,
                "results_run_dir": None if result_dir is None else str(result_dir),
                "results_updated_at": updated_at,
                "results_summary": results_summary,
            }
        )
        if benchmark_dir is None:
            rows[-1]["origin"] = "official"
            rows[-1]["read_only"] = True
    return rows


def validate_effective_config(config: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        load_benchmark_config(config)
    except Exception as exc:
        return False, str(exc)
    return True, None


def ensure_default_output_dir(config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    run = updated.setdefault("run", {})
    output = run.setdefault("output", {})
    if output.get("output_dir"):
        return updated
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output["output_dir"] = str((OUTPUT_DIR / "ui_runs" / stamp).resolve())
    return updated


def _normalize_model_execution_block(execution: Any) -> dict[str, Any]:
    block = dict(execution or {})
    mode = str(block.get("mode") or "inprocess")
    if mode != "subprocess":
        return {"mode": "inprocess"}
    return {
        "mode": "subprocess",
        "venv": block.get("venv"),
        "python": block.get("python"),
        "cwd": block.get("cwd"),
        "pythonpath": list(block.get("pythonpath") or []),
        "env": dict(block.get("env") or {}),
    }


def _common_model_execution_from_models(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    executions = [
        _normalize_model_execution_block(model.get("execution"))
        for model in models
        if model.get("execution") is not None
    ]
    if not executions:
        return None
    first = executions[0]
    if all(execution == first for execution in executions[1:]):
        return first
    return None


def _apply_run_model_execution_defaults(config: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    benchmark = updated.setdefault("benchmark", {})
    run = updated.setdefault("run", {})
    execution = run.setdefault("execution", {})

    models = list(benchmark.get("models") or [])
    has_explicit_run_model_execution = "model_execution" in execution and execution.get("model_execution") is not None
    derived_model_execution = _common_model_execution_from_models(models)
    model_execution = _normalize_model_execution_block(
        execution.get("model_execution") if has_explicit_run_model_execution else derived_model_execution
    )
    execution["model_execution"] = model_execution

    if has_explicit_run_model_execution or derived_model_execution is not None:
        for model in models:
            if model_execution["mode"] == "subprocess":
                model["execution"] = copy.deepcopy(model_execution)
            else:
                model.pop("execution", None)
    benchmark["models"] = models
    return updated


def build_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = ensure_default_output_dir(config)
    updated.setdefault("benchmark", {})
    updated["benchmark"].setdefault("dataset", {})
    updated["benchmark"].setdefault("protocol", {})
    updated["benchmark"].setdefault("metrics", [])
    updated["benchmark"].setdefault("models", [])
    run = updated.setdefault("run", {})
    run.setdefault("execution", {"scheduler": "auto", "device": None, "model_execution": {"mode": "inprocess"}})
    run.setdefault("tracking", {"mlflow": {}})
    run.setdefault("output", {})
    updated.setdefault("diagnostics", {})
    return _apply_run_model_execution_defaults(updated)


def cli_validate_command(config_path: Path | None) -> str:
    target = str(config_path) if config_path is not None else "<config.json>"
    return f"ts-benchmark validate {target}"


def cli_run_command(config_path: Path | None) -> str:
    target = str(config_path) if config_path is not None else "<config.json>"
    return f"ts-benchmark run {target}"
