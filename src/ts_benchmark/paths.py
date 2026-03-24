"""Filesystem locations for packaged assets and mutable workspace state."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

RESOURCES_ROOT = PACKAGE_ROOT / "resources"
PACKAGED_BENCHMARKS_DIR = RESOURCES_ROOT / "benchmarks"
PACKAGED_SAMPLE_DATA_DIR = RESOURCES_ROOT / "sample_data"


def workspace_root() -> Path:
    """Return the root directory for mutable ts-benchmark workspace state."""

    override = os.getenv("TS_BENCHMARK_HOME")
    if override:
        return Path(override).expanduser().resolve()
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return (Path(xdg_data_home).expanduser() / "ts-benchmark").resolve()
    return (Path.home() / ".local" / "share" / "ts-benchmark").resolve()


WORKSPACE_ROOT = workspace_root()
WORKSPACE_BENCHMARK_DIR = WORKSPACE_ROOT / "benchmarks"
SHIPPED_BENCHMARK_CACHE_DIR = WORKSPACE_ROOT / "shipped_benchmarks"
WORKSPACE_DATASET_DIR = WORKSPACE_ROOT / "datasets"
WORKSPACE_MODEL_CATALOG_DIR = WORKSPACE_ROOT / "model_catalog"
WORKSPACE_RUNS_DIR = WORKSPACE_ROOT / "runs"
WORKSPACE_BENCHMARK_RESULTS_DIR = WORKSPACE_ROOT / "benchmark_results"
WORKSPACE_UPLOAD_DIR = WORKSPACE_ROOT / "uploads"
WORKSPACE_MODEL_ADAPTER_UPLOAD_DIR = WORKSPACE_UPLOAD_DIR / "model_adapters"
WORKSPACE_RUNTIME_DIR = WORKSPACE_ROOT / "runtime" / "streamlit"
WORKSPACE_TRACKING_DIR = WORKSPACE_ROOT / "tracking"
WORKSPACE_MLFLOW_DIR = WORKSPACE_TRACKING_DIR / "mlruns"
WORKSPACE_SAMPLE_DATA_DIR = WORKSPACE_ROOT / "sample_data"

BENCHMARK_CATALOG_DIR = WORKSPACE_BENCHMARK_DIR
DATASET_CATALOG_DIR = WORKSPACE_DATASET_DIR
MODEL_CATALOG_DIR = WORKSPACE_MODEL_CATALOG_DIR
OUTPUT_DIR = WORKSPACE_RUNS_DIR
BENCHMARK_RESULTS_DIR = WORKSPACE_BENCHMARK_RESULTS_DIR
UPLOAD_DIR = WORKSPACE_UPLOAD_DIR
MODEL_ADAPTER_UPLOAD_DIR = WORKSPACE_MODEL_ADAPTER_UPLOAD_DIR
RUNTIME_DIR = WORKSPACE_RUNTIME_DIR
TRACKING_DIR = WORKSPACE_TRACKING_DIR
MLFLOW_DIR = WORKSPACE_MLFLOW_DIR
MLRUNS_DIR = WORKSPACE_MLFLOW_DIR
SAMPLE_DATA_DIR = WORKSPACE_SAMPLE_DATA_DIR


def ensure_workspace_dirs() -> None:
    """Create the mutable workspace directories used by the UI and notebook helpers."""

    for path in [
        SHIPPED_BENCHMARK_CACHE_DIR,
        WORKSPACE_DATASET_DIR,
        WORKSPACE_MODEL_CATALOG_DIR,
        WORKSPACE_RUNS_DIR,
        WORKSPACE_BENCHMARK_RESULTS_DIR,
        WORKSPACE_UPLOAD_DIR,
        WORKSPACE_MODEL_ADAPTER_UPLOAD_DIR,
        WORKSPACE_RUNTIME_DIR,
        WORKSPACE_TRACKING_DIR,
        WORKSPACE_MLFLOW_DIR,
        WORKSPACE_SAMPLE_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _merge_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                _merge_directory_contents(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            continue
        shutil.move(str(item), str(target))


def migrate_legacy_workspace() -> None:
    """Move legacy repo-local mutable state into the workspace directory."""

    ensure_workspace_dirs()
    migrations = [
        (REPO_ROOT / "datasets", WORKSPACE_DATASET_DIR),
        (REPO_ROOT / "model_catalog", WORKSPACE_MODEL_CATALOG_DIR),
        (REPO_ROOT / "outputs", WORKSPACE_RUNS_DIR),
        (REPO_ROOT / "mlruns", WORKSPACE_MLFLOW_DIR),
        (REPO_ROOT / ".streamlit_uploads", WORKSPACE_UPLOAD_DIR),
        (REPO_ROOT / ".streamlit_runtime", WORKSPACE_RUNTIME_DIR),
        (REPO_ROOT / "sample_data", WORKSPACE_SAMPLE_DATA_DIR),
    ]
    for source, destination in migrations:
        if not source.exists() or source.resolve() == destination.resolve():
            continue
        _merge_directory_contents(source, destination)
        try:
            source.rmdir()
        except OSError:
            pass


__all__ = [
    "PACKAGE_ROOT",
    "PACKAGED_BENCHMARKS_DIR",
    "PACKAGED_SAMPLE_DATA_DIR",
    "BENCHMARK_CATALOG_DIR",
    "BENCHMARK_RESULTS_DIR",
    "DATASET_CATALOG_DIR",
    "MLFLOW_DIR",
    "MODEL_ADAPTER_UPLOAD_DIR",
    "MODEL_CATALOG_DIR",
    "OUTPUT_DIR",
    "REPO_ROOT",
    "RESOURCES_ROOT",
    "RUNTIME_DIR",
    "SAMPLE_DATA_DIR",
    "SHIPPED_BENCHMARK_CACHE_DIR",
    "SRC_ROOT",
    "TRACKING_DIR",
    "UPLOAD_DIR",
    "MLRUNS_DIR",
    "WORKSPACE_BENCHMARK_DIR",
    "WORKSPACE_BENCHMARK_RESULTS_DIR",
    "WORKSPACE_DATASET_DIR",
    "WORKSPACE_MLFLOW_DIR",
    "WORKSPACE_MODEL_ADAPTER_UPLOAD_DIR",
    "WORKSPACE_MODEL_CATALOG_DIR",
    "WORKSPACE_ROOT",
    "WORKSPACE_RUNS_DIR",
    "WORKSPACE_RUNTIME_DIR",
    "WORKSPACE_SAMPLE_DATA_DIR",
    "WORKSPACE_TRACKING_DIR",
    "WORKSPACE_UPLOAD_DIR",
    "ensure_workspace_dirs",
    "migrate_legacy_workspace",
    "workspace_root",
]
