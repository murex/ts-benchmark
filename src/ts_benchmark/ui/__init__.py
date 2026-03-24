"""Shared helpers for the Streamlit UI."""

from __future__ import annotations

from ..benchmark import default_benchmark_config_dir
from ..paths import (
    BENCHMARK_CATALOG_DIR,
    BENCHMARK_RESULTS_DIR,
    DATASET_CATALOG_DIR,
    MODEL_ADAPTER_UPLOAD_DIR,
    MODEL_CATALOG_DIR,
    OUTPUT_DIR,
    REPO_ROOT as APP_ROOT,
    RUNTIME_DIR,
    SAMPLE_DATA_DIR,
    SRC_ROOT,
    UPLOAD_DIR,
    ensure_workspace_dirs,
    migrate_legacy_workspace,
)

ensure_workspace_dirs()
migrate_legacy_workspace()

CONFIG_DIR = default_benchmark_config_dir()

__all__ = [
    "APP_ROOT",
    "SRC_ROOT",
    "CONFIG_DIR",
    "BENCHMARK_CATALOG_DIR",
    "DATASET_CATALOG_DIR",
    "MODEL_CATALOG_DIR",
    "OUTPUT_DIR",
    "BENCHMARK_RESULTS_DIR",
    "SAMPLE_DATA_DIR",
    "UPLOAD_DIR",
    "MODEL_ADAPTER_UPLOAD_DIR",
    "RUNTIME_DIR",
]
