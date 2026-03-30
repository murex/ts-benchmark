"""External dataset loaders for CSV and Parquet time-series return files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...benchmark.protocol import Protocol
from ..runtime import DatasetInstance


def _coerce_asset_columns(frame: pd.DataFrame, asset_columns: Sequence[str] | None, date_column: str | None) -> list[str]:
    if asset_columns:
        missing = [column for column in asset_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Requested asset column(s) not found: {missing}")
        return list(asset_columns)

    candidates = list(frame.columns)
    if date_column is not None and date_column in candidates:
        candidates.remove(date_column)
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric:
        raise ValueError("Could not infer any numeric asset columns from the dataset.")
    return numeric


def _load_frame(path: Path, source: str, params: dict[str, Any]) -> pd.DataFrame:
    if source == "csv":
        read_kwargs = dict(params.get("read_kwargs", {}))
        return pd.read_csv(path, **read_kwargs)
    if source == "parquet":
        read_kwargs = dict(params.get("read_kwargs", {}))
        return pd.read_parquet(path, **read_kwargs)
    raise ValueError(f"Unsupported tabular source: {source}")


def _pivot_long_frame(
    frame: pd.DataFrame,
    *,
    date_column: str,
    series_id_columns: Sequence[str],
    value_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [
        column
        for column in [date_column, value_column, *series_id_columns]
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Requested long-format column(s) not found: {missing}")

    long_frame = frame.loc[:, [date_column, *series_id_columns, value_column]].copy()
    long_frame[date_column] = pd.to_datetime(long_frame[date_column], errors="coerce")
    long_frame[value_column] = pd.to_numeric(long_frame[value_column], errors="coerce")
    long_frame = long_frame.dropna(subset=[date_column])
    if long_frame.empty:
        raise ValueError("Long-format dataset does not contain any valid timestamps.")

    if len(series_id_columns) == 1:
        series_key = long_frame[series_id_columns[0]].astype(str)
    else:
        series_key = long_frame.loc[:, list(series_id_columns)].astype(str).agg("::".join, axis=1)
    long_frame["_series_key"] = series_key

    duplicated = long_frame.duplicated(subset=[date_column, "_series_key"])
    if duplicated.any():
        duplicates = long_frame.loc[duplicated, [date_column, "_series_key"]].head(5)
        raise ValueError(
            "Long-format dataset has duplicate date/series rows, for example: "
            + ", ".join(f"({row[date_column].date()}, {row['_series_key']})" for _, row in duplicates.iterrows())
        )

    pivoted = long_frame.pivot(index=date_column, columns="_series_key", values=value_column).sort_index()
    timestamps = pd.Series(pivoted.index, name=date_column).reset_index(drop=True)
    pivoted = pivoted.reset_index(drop=True)
    pivoted.columns = [str(column) for column in pivoted.columns]
    return pivoted, timestamps


def load_returns_frame(
    *,
    path: str | Path,
    source: str,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = _load_frame(path, source, params)
    frame = frame.copy()

    date_column = params.get("date_column")
    sort_ascending = bool(params.get("sort_ascending", True))
    layout = str(params.get("layout", "wide") or "wide").strip().lower()
    if layout not in {"wide", "long"}:
        raise ValueError("dataset.schema.layout must be either 'wide' or 'long'.")
    timestamps = None
    if layout == "long":
        if date_column is None:
            raise ValueError("Long-format datasets require params.date_column.")
        series_id_columns = params.get("series_id_columns") or []
        if not series_id_columns:
            raise ValueError("Long-format datasets require params.series_id_columns.")
        value_column = params.get("value_column")
        if not value_column:
            raise ValueError("Long-format datasets require params.value_column.")
        values, timestamps = _pivot_long_frame(
            frame,
            date_column=str(date_column),
            series_id_columns=list(series_id_columns),
            value_column=str(value_column),
        )
        if not sort_ascending:
            values = values.iloc[::-1].reset_index(drop=True)
            timestamps = timestamps.iloc[::-1].reset_index(drop=True)
        asset_columns = list(values.columns)
    elif date_column is not None:
        if date_column not in frame.columns:
            raise KeyError(f"date_column '{date_column}' not found in dataset.")
        timestamps = pd.to_datetime(frame[date_column], errors="coerce")
        valid = timestamps.notna()
        frame = frame.loc[valid].reset_index(drop=True)
        timestamps = timestamps.loc[valid].reset_index(drop=True)
        order = np.argsort(timestamps.to_numpy())
        if not sort_ascending:
            order = order[::-1]
        frame = frame.iloc[order].reset_index(drop=True)
        timestamps = timestamps.iloc[order].reset_index(drop=True)
    elif not sort_ascending:
        frame = frame.iloc[::-1].reset_index(drop=True)

    if layout == "wide":
        asset_columns = _coerce_asset_columns(frame, params.get("asset_columns"), date_column)
        values = frame[asset_columns].apply(pd.to_numeric, errors="coerce")

    value_type = str(params.get("value_type", "returns")).lower()
    return_kind = str(params.get("return_kind", "simple")).lower()
    if return_kind not in {"simple", "log"}:
        raise ValueError(
            "dataset.semantics.return_kind or dataset.provider.config.return_kind "
            "must be either 'simple' or 'log'."
        )

    if value_type in {"price", "prices"}:
        value_mode = "price" if return_kind == "simple" else "price_to_log_return"
    elif value_type in {"log_price", "log_prices"}:
        value_mode = "log_price"
    elif value_type in {"return", "returns"}:
        value_mode = "return" if return_kind == "simple" else "log_return"
    elif value_type in {"log_return", "log_returns"}:
        value_mode = "log_return"
    else:
        raise ValueError(
            "dataset.semantics.target_kind or dataset.provider.config.value_type "
            "must be one of 'price', 'log_price', 'return', or 'log_return' "
            "(legacy 'prices'/'returns' are also supported)."
        )

    if value_mode == "price_to_log_return":
        values = np.log(values).diff()
        if timestamps is not None:
            timestamps = timestamps.iloc[1:].reset_index(drop=True)
        values = values.iloc[1:].reset_index(drop=True)
    elif value_mode == "price":
        values = values.pct_change()
        if timestamps is not None:
            timestamps = timestamps.iloc[1:].reset_index(drop=True)
        values = values.iloc[1:].reset_index(drop=True)
    elif value_mode == "log_price":
        values = values.diff()
        if timestamps is not None:
            timestamps = timestamps.iloc[1:].reset_index(drop=True)
        values = values.iloc[1:].reset_index(drop=True)

    dropna = str(params.get("dropna", "any")).lower()
    if dropna == "any":
        mask = values.notna().all(axis=1)
        values = values.loc[mask].reset_index(drop=True)
        if timestamps is not None:
            timestamps = timestamps.loc[mask].reset_index(drop=True)
    elif dropna == "none":
        if values.isna().any().any():
            raise ValueError("Dataset contains missing values and dataset.provider.config.dropna='none'.")
    else:
        raise ValueError("dataset.provider.config.dropna must be either 'any' or 'none'.")

    returns = values.astype(float)
    metadata = {
        "path": str(path),
        "date_column": date_column,
        "layout": layout,
        "asset_columns": list(asset_columns),
        "value_type": value_type,
        "value_mode": value_mode,
        "return_kind": return_kind,
        "n_rows_loaded": int(len(returns)),
    }
    return returns, timestamps, metadata


def make_tabular_benchmark_dataset(
    *,
    dataset_name: str,
    source: str,
    path: str | Path,
    freq: str,
    protocol: Protocol,
    params: dict[str, Any],
) -> DatasetInstance:
    returns_frame, timestamps, loader_metadata = load_returns_frame(path=path, source=source, params=params)
    returns = returns_frame.to_numpy(dtype=float)
    asset_names = list(returns_frame.columns)

    train_size = int(protocol.train_size)
    test_size = int(protocol.test_size)
    generation_mode = str(protocol.generation_mode)
    unconditional_train_data_mode = protocol.unconditional_train_data_mode
    context_length = int(protocol.context_length)
    horizon = int(protocol.horizon)
    eval_stride = int(protocol.eval_stride)

    if generation_mode == "forecast" and train_size <= context_length:
        raise ValueError("train_size must be larger than context_length.")
    if generation_mode == "unconditional" and unconditional_train_data_mode == "path_dataset":
        raise ValueError(
            "Tabular datasets do not support unconditional_train_data_mode='path_dataset'. "
            "Use 'windowed_path' instead."
        )
    if test_size < horizon:
        raise ValueError("test_size must be at least horizon.")
    if eval_stride <= 0:
        raise ValueError("eval_stride must be positive.")
    if returns.shape[0] < train_size + test_size:
        raise ValueError(
            f"Dataset has only {returns.shape[0]} rows, but train_size + test_size = {train_size + test_size}."
        )

    total_steps = train_size + test_size
    returns = returns[:total_steps]
    if timestamps is not None:
        timestamps = timestamps.iloc[:total_steps].reset_index(drop=True)

    contexts = []
    realized_futures = []
    evaluation_timestamps: list[str] = []

    for start in range(train_size, total_steps - horizon + 1, eval_stride):
        if generation_mode == "forecast":
            contexts.append(returns[start - context_length : start])
        else:
            contexts.append(np.zeros((0, returns.shape[1]), dtype=float))
        realized_futures.append(returns[start : start + horizon])
        if timestamps is not None:
            evaluation_timestamps.append(str(pd.Timestamp(timestamps.iloc[start]).date()))

    if not contexts:
        raise ValueError("No evaluation windows were generated. Adjust test_size / horizon / eval_stride.")

    metadata = {
        **loader_metadata,
        "dataset_name": dataset_name,
        "source": source,
        "file_format": source,
        "is_synthetic": False,
    }

    return DatasetInstance(
        name=dataset_name,
        source=source,
        full_returns=returns,
        train_returns=returns[:train_size],
        test_returns=returns[train_size:total_steps],
        contexts=np.stack(contexts, axis=0),
        realized_futures=np.stack(realized_futures, axis=0),
        asset_names=asset_names,
        protocol=protocol,
        freq=freq,
        metadata=metadata,
        evaluation_timestamps=evaluation_timestamps or None,
        reference_sampler=None,
    )
