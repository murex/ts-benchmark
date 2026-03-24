"""Shared helpers for structural-contract model adapters."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

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


def coerce_values(batch_like: object) -> np.ndarray:
    values = np.asarray(getattr(batch_like, "values"), dtype=float)
    if values.ndim != 3:
        raise ValueError("batch.values must be shaped [batch, time, target_dim].")
    if values.shape[1] < 1:
        raise ValueError("batch.values must contain at least one timestep.")
    return values


def collect_training_batches(train: Iterable[object], *, target_dim: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    for batch_like in train:
        values = coerce_values(batch_like)
        if values.shape[-1] != target_dim:
            raise ValueError(
                f"schema.target_dim={target_dim} does not match training batch target_dim={values.shape[-1]}."
            )
        if values.shape[1] < 2:
            raise ValueError("training batches must contain at least two timesteps.")
        batches.append(values)
    if not batches:
        raise ValueError("at least one training batch is required.")
    return batches


def make_list_dataset_from_values(values: np.ndarray, *, freq: str, install_extra: str):
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
        {"start": start, "target": np.asarray(values[index], dtype=np.float32).T}
        for index in range(values.shape[0])
    ]
    return ListDataset(entries, freq=freq, one_dim_target=False)


def make_list_dataset_from_batches(
    batches: Iterable[np.ndarray],
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
    entries: list[dict[str, object]] = []
    for values in batches:
        entries.extend(
            {"start": start, "target": np.asarray(values[index], dtype=np.float32).T}
            for index in range(values.shape[0])
        )
    return ListDataset(entries, freq=freq, one_dim_target=False)


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
