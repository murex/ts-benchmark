"""Shared helpers for structural-contract model adapters."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ts_benchmark.model.model_contract import GenerationMode


def mode_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def require_forecast_task(task: object) -> tuple[int, int]:
    mode = mode_value(getattr(task, "mode", None))
    if mode != GenerationMode.FORECAST.value:
        raise ValueError(f"unsupported mode {mode!r}; this adapter supports forecast only.")

    raw_horizon = getattr(task, "horizon", None)
    if raw_horizon is None:
        raise ValueError("task.horizon is required for forecast adapters.")
    horizon = int(raw_horizon)
    if horizon < 1:
        raise ValueError("task.horizon must be positive.")

    raw_context_length = getattr(task, "context_length", None)
    context_length = horizon if raw_context_length is None else int(raw_context_length)
    if context_length < 1:
        raise ValueError("task.context_length must be positive when provided.")
    return horizon, context_length


def runtime_device(runtime: object | None) -> str | None:
    if runtime is None:
        return None
    return getattr(runtime, "device", None)


def coerce_series_values(series_like: object) -> np.ndarray:
    values = np.asarray(getattr(series_like, "values"), dtype=float)
    if values.ndim != 2:
        raise ValueError("series.values must be shaped [time, target_dim].")
    if values.shape[0] < 1:
        raise ValueError("series.values must contain at least one timestep.")
    return values


def coerce_training_series_values(series_like: object, *, target_dim: int) -> np.ndarray:
    values = coerce_series_values(series_like)
    if values.shape[-1] != target_dim:
        raise ValueError(
            f"schema.target_dim={target_dim} does not match training series target_dim={values.shape[-1]}."
        )
    if values.shape[0] < 2:
        raise ValueError("training series must contain at least two timesteps.")
    return values


def coerce_forecast_training_series_collection(
    train_like: object,
    *,
    target_dim: int,
    context_length: int,
    horizon: int,
) -> list[np.ndarray]:
    examples = getattr(train_like, "examples", None)
    if examples is None:
        raise ValueError("forecast adapters expect train.examples in TrainData.")
    series_collection: list[np.ndarray] = []
    for index, example in enumerate(examples):
        context_like = getattr(example, "context", None)
        target_like = getattr(example, "target", None)
        if context_like is None or target_like is None:
            raise ValueError(
                f"train.examples[{index}] must define both context and target series."
            )
        context_values = coerce_series_values(context_like)
        target_values = coerce_series_values(target_like)
        if context_values.shape[-1] != target_dim:
            raise ValueError(
                f"train.examples[{index}].context target_dim={context_values.shape[-1]} "
                f"does not match schema.target_dim={target_dim}."
            )
        if target_values.shape[-1] != target_dim:
            raise ValueError(
                f"train.examples[{index}].target target_dim={target_values.shape[-1]} "
                f"does not match schema.target_dim={target_dim}."
            )
        if context_values.shape[0] != context_length:
            raise ValueError(
                f"train.examples[{index}].context has length {context_values.shape[0]} "
                f"but expected context_length={context_length}."
            )
        if target_values.shape[0] != horizon:
            raise ValueError(
                f"train.examples[{index}].target has length {target_values.shape[0]} "
                f"but expected horizon={horizon}."
            )
        series_collection.append(np.concatenate([context_values, target_values], axis=0))
    if not series_collection:
        raise ValueError("forecast training data must contain at least one example.")
    return series_collection


def make_list_dataset_from_series_collection(
    values_collection: list[np.ndarray],
    *,
    freq: str,
    install_extra: str,
):
    try:
        import pandas as pd
        from gluonts.dataset.common import ListDataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "This adapter requires `gluonts` and `pandas`. Install the "
            f"`{install_extra}` extra."
        ) from exc

    start = pd.Period("2000-01-03", freq=freq)
    entries = [
        {"start": start, "target": np.asarray(values, dtype=np.float32).T}
        for values in values_collection
    ]
    return ListDataset(entries, freq=freq, one_dim_target=False)


def make_list_dataset_from_series(values: np.ndarray, *, freq: str, install_extra: str):
    return make_list_dataset_from_series_collection(
        [np.asarray(values, dtype=float)],
        freq=freq,
        install_extra=install_extra,
    )

def normalize_forecast_samples(
    samples: np.ndarray,
    *,
    num_samples: int,
    horizon: int,
    target_dim: int,
) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    if x.ndim != 3:
        raise ValueError(
            "Expected rank-3 forecast samples for a multivariate model, "
            f"got shape {x.shape}."
        )
    if x.shape[0] > num_samples:
        x = x[:num_samples]
    if x.shape[0] < num_samples:
        raise ValueError(
            f"Backend returned only {x.shape[0]} samples, but the caller requested {num_samples}."
        )
    if x.shape[1] == horizon and x.shape[2] == target_dim:
        return x
    if x.shape[1] == target_dim and x.shape[2] == horizon:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(
        "Could not reconcile forecast samples shape "
        f"{x.shape} with expected shape ({num_samples}, {horizon}, {target_dim})."
    )


def save_pickle_payload(path: str | Path, payload: object) -> None:
    target = Path(path)
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
