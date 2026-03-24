"""Dataset-catalog helpers for the Streamlit UI."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ts_benchmark.benchmark import shipped_benchmark_paths
from ts_benchmark.serialization import to_jsonable

from .. import BENCHMARK_CATALOG_DIR, CONFIG_DIR, DATASET_CATALOG_DIR, UPLOAD_DIR

SUPPORTED_DATASET_SOURCES = ("synthetic", "csv", "parquet")


def default_dataset_dict() -> dict[str, Any]:
    return {
        "name": "",
        "description": "",
        "provider": {
            "kind": "synthetic",
            "config": {
                "generator": "regime_switching_factor_sv",
                "params": {},
            },
        },
        "schema": {
            "layout": "tensor",
            "frequency": "B",
        },
        "semantics": {},
        "metadata": {},
    }


def normalize_dataset_dict(dataset: dict[str, Any] | None) -> dict[str, Any]:
    payload = copy.deepcopy(default_dataset_dict())
    if dataset:
        payload.update(copy.deepcopy(dataset))

    payload["name"] = str(payload.get("name") or "")
    payload["description"] = str(payload.get("description") or "")

    provider = dict(payload.get("provider") or {})
    kind = str(provider.get("kind") or "synthetic").strip().lower()
    if kind not in SUPPORTED_DATASET_SOURCES:
        kind = "synthetic"

    schema = dict(payload.get("schema") or {})
    semantics = dict(payload.get("semantics") or {})
    metadata = dict(payload.get("metadata") or {})
    provider_config = dict(provider.get("config") or {})

    if kind == "synthetic":
        payload["provider"] = {
            "kind": "synthetic",
            "config": {
                "generator": str(provider_config.get("generator") or "regime_switching_factor_sv"),
                "params": dict(provider_config.get("params") or {}),
            },
        }
        payload["schema"] = {
            "layout": "tensor",
            "frequency": str(schema.get("frequency") or "B"),
        }
    else:
        normalized_config = dict(provider_config)
        normalized_config.pop("generator", None)
        normalized_config.pop("params", None)
        normalized_config["path"] = str(normalized_config.get("path") or "")
        if normalized_config.get("value_column") is not None:
            normalized_config["value_column"] = str(normalized_config.get("value_column") or "")
        payload["provider"] = {
            "kind": kind,
            "config": normalized_config,
        }
        layout = str(schema.get("layout") or "wide").strip().lower()
        if layout not in {"wide", "long"}:
            layout = "wide"
        payload["schema"] = {
            "layout": layout,
            "time_column": schema.get("time_column"),
            "series_id_columns": list(schema.get("series_id_columns") or []),
            "target_columns": list(schema.get("target_columns") or []),
            "feature_columns": list(schema.get("feature_columns") or []),
            "static_feature_columns": list(schema.get("static_feature_columns") or []),
            "frequency": str(schema.get("frequency") or "B"),
        }

    payload["semantics"] = semantics
    payload["metadata"] = metadata
    return payload


def switch_dataset_source(dataset: dict[str, Any], source: str) -> dict[str, Any]:
    current = normalize_dataset_dict(dataset)
    target = str(source or "synthetic").strip().lower()
    if target not in SUPPORTED_DATASET_SOURCES:
        target = "synthetic"
    if target == current["provider"]["kind"]:
        return current

    updated = copy.deepcopy(current)
    provider_config = dict(updated.get("provider", {}).get("config") or {})
    schema = dict(updated.get("schema") or {})

    if target == "synthetic":
        updated["provider"] = {
            "kind": "synthetic",
            "config": {
                "generator": "regime_switching_factor_sv",
                "params": {},
            },
        }
        updated["schema"] = {
            "layout": "tensor",
            "frequency": str(schema.get("frequency") or "B"),
        }
        return updated

    file_config: dict[str, Any] = {"path": ""}
    if current["provider"]["kind"] in {"csv", "parquet"}:
        file_config.update(provider_config)
        file_config["path"] = str(file_config.get("path") or "")

    updated["provider"] = {
        "kind": target,
        "config": file_config,
    }
    layout = str(schema.get("layout") or "wide").strip().lower()
    if layout not in {"wide", "long"}:
        layout = "wide"
    updated["schema"] = {
        "layout": layout,
        "time_column": schema.get("time_column"),
        "series_id_columns": list(schema.get("series_id_columns") or []),
        "target_columns": list(schema.get("target_columns") or []),
        "feature_columns": list(schema.get("feature_columns") or []),
        "static_feature_columns": list(schema.get("static_feature_columns") or []),
        "frequency": str(schema.get("frequency") or "B"),
    }
    return updated


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return slug or "dataset"


def store_uploaded_dataset_file(
    *,
    filename: str,
    content: bytes,
    upload_dir: Path = UPLOAD_DIR,
) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{_slugify_name(Path(filename).stem)}{Path(filename).suffix.lower()}"
    target.write_bytes(content)
    return target


def inspect_tabular_source(
    *,
    path: str | Path,
    source: str,
    max_rows: int = 100,
) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if source == "csv":
        frame = pd.read_csv(dataset_path, nrows=max_rows)
    elif source == "parquet":
        frame = pd.read_parquet(dataset_path).head(max_rows)
    else:
        raise ValueError(f"Unsupported tabular source: {source}")

    columns = [str(column) for column in frame.columns]
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    return {
        "path": str(dataset_path),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "preview": frame,
    }


def saved_dataset_paths(dataset_dir: Path = DATASET_CATALOG_DIR) -> dict[str, Path]:
    return {path.stem: path for path in sorted(dataset_dir.glob("*.json"))}


def _resolve_saved_dataset_path(name: str, dataset_dir: Path = DATASET_CATALOG_DIR) -> Path | None:
    target = str(name).strip()
    if not target:
        return None
    slug_match = dataset_dir / f"{_slugify_name(target)}.json"
    if slug_match.exists():
        return slug_match
    for path in saved_dataset_paths(dataset_dir).values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("name") or "").strip() == target:
            return path
    return None


def load_saved_dataset(name: str, dataset_dir: Path = DATASET_CATALOG_DIR) -> dict[str, Any]:
    path = _resolve_saved_dataset_path(name, dataset_dir=dataset_dir)
    if path is None:
        raise FileNotFoundError(f"No saved dataset named '{name}'.")
    return normalize_dataset_dict(json.loads(path.read_text(encoding="utf-8")))


def save_dataset_definition(dataset: dict[str, Any], dataset_dir: Path = DATASET_CATALOG_DIR) -> Path:
    payload = normalize_dataset_dict(dataset)
    name = payload["name"].strip()
    if not name:
        raise ValueError("Saved datasets require a non-empty dataset name.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{_slugify_name(name)}.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return path


def list_saved_datasets(dataset_dir: Path = DATASET_CATALOG_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in saved_dataset_paths(dataset_dir).values():
        try:
            payload = load_saved_dataset(path.stem, dataset_dir=dataset_dir)
        except Exception:
            continue
        rows.append(
            {
                "name": payload.get("name") or path.stem,
                "description": payload.get("description") or "",
                "source": payload.get("provider", {}).get("kind"),
                "frequency": payload.get("schema", {}).get("frequency"),
                "path": str(path),
            }
        )
    return rows


def find_benchmark_configs_using_dataset(
    dataset_name: str,
    config_dir: Path = CONFIG_DIR,
    benchmark_dir: Path = BENCHMARK_CATALOG_DIR,
) -> list[Path]:
    target = str(dataset_name).strip()
    if not target:
        return []

    usages: list[Path] = []
    if config_dir == CONFIG_DIR:
        benchmark_paths = list(shipped_benchmark_paths().values()) + list(sorted(benchmark_dir.glob("*.json")))
    else:
        benchmark_paths = list(sorted(config_dir.glob("*.json")))
    for path in benchmark_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        configured_name = str(payload.get("benchmark", {}).get("dataset", {}).get("name") or "").strip()
        if configured_name == target:
            usages.append(path)
    return usages


def delete_saved_dataset(
    name: str,
    *,
    dataset_dir: Path = DATASET_CATALOG_DIR,
    config_dir: Path = CONFIG_DIR,
    benchmark_dir: Path = BENCHMARK_CATALOG_DIR,
) -> tuple[bool, list[Path]]:
    payload = load_saved_dataset(name, dataset_dir=dataset_dir)
    usages = find_benchmark_configs_using_dataset(
        payload["name"],
        config_dir=config_dir,
        benchmark_dir=benchmark_dir,
    )
    if usages:
        return False, usages

    path = _resolve_saved_dataset_path(name, dataset_dir=dataset_dir)
    if path is None:
        raise FileNotFoundError(f"No saved dataset named '{name}'.")
    path.unlink()
    return True, []
