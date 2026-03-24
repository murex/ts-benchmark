"""MLflow-backed tracking services for the Streamlit UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ts_benchmark.tracking import (
    download_mlflow_artifact,
    get_mlflow_run_payload,
    list_mlflow_artifacts,
    list_mlflow_experiments,
    mlflow_available,
    search_mlflow_runs,
)

from ...paths import MLRUNS_DIR


def tracking_available() -> bool:
    return mlflow_available()


def default_tracking_uri(config: dict[str, Any] | None) -> str:
    run = {} if config is None else dict(config.get("run") or {})
    tracking = dict(run.get("tracking") or {})
    mlflow_cfg = dict(tracking.get("mlflow") or {})
    configured = mlflow_cfg.get("tracking_uri")
    if configured:
        return str(configured)
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return str(MLRUNS_DIR.resolve())


def list_experiments_df(uri: str) -> pd.DataFrame:
    return list_mlflow_experiments(tracking_uri=uri)


def search_runs_df(uri: str, experiment_ids: list[str], max_results: int) -> pd.DataFrame:
    return search_mlflow_runs(tracking_uri=uri, experiment_ids=experiment_ids, max_results=max_results)


def get_run_payload(uri: str, run_id: str) -> dict[str, Any]:
    return get_mlflow_run_payload(tracking_uri=uri, run_id=run_id)


def list_artifacts_df(uri: str, run_id: str, artifact_path: str | None) -> pd.DataFrame:
    return list_mlflow_artifacts(tracking_uri=uri, run_id=run_id, artifact_path=artifact_path)


def download_artifact_path(uri: str, run_id: str, artifact_path: str) -> Path:
    return download_mlflow_artifact(tracking_uri=uri, run_id=run_id, artifact_path=artifact_path)
