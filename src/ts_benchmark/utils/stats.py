"""Statistical utilities used by metrics and models."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def safe_corrcoef(x: np.ndarray) -> np.ndarray:
    """Return a finite correlation matrix even if some columns are near-constant."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("safe_corrcoef expects a 2D array.")
    x_centered = x - x.mean(axis=0, keepdims=True)
    std = x_centered.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    z = x_centered / std
    corr = (z.T @ z) / max(1, x.shape[0] - 1)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def flatten_marginals(values: np.ndarray) -> np.ndarray:
    """Flatten sample arrays of any supported shape to `[n_rows, n_assets]`."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 4:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"Unsupported array shape {tuple(x.shape)} for marginal flattening.")


def to_paths(values: np.ndarray) -> np.ndarray:
    """Reshape sample arrays to `[n_paths, horizon, n_assets]`."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        return x[None, :, :]
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        n_contexts, n_scenarios, horizon, n_assets = x.shape
        return x.reshape(n_contexts * n_scenarios, horizon, n_assets)
    raise ValueError(f"Unsupported array shape {tuple(x.shape)} for path conversion.")


def flatten_path_samples(samples: np.ndarray) -> np.ndarray:
    """Flatten `[n_contexts, n_scenarios, horizon, n_assets]` into path vectors."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 4:
        raise ValueError("Expected samples with shape [n_contexts, n_scenarios, horizon, n_assets].")
    n_contexts, n_scenarios, horizon, n_assets = samples.shape
    return samples.reshape(n_contexts * n_scenarios, horizon * n_assets)


def lagged_autocorrelation_paths(
    paths: np.ndarray,
    lags: Sequence[int],
    square: bool = False,
) -> np.ndarray:
    """Compute per-asset autocorrelations aggregated over many paths.

    Parameters
    ----------
    paths:
        Array shaped `[n_paths, horizon, n_assets]`.
    lags:
        Lags at which autocorrelation is computed.
    square:
        If True, compute autocorrelation of squared returns.
    """
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3:
        raise ValueError("Expected paths with shape [n_paths, horizon, n_assets].")
    if square:
        x = x ** 2

    n_paths, horizon, n_assets = x.shape
    out = np.zeros((len(lags), n_assets), dtype=float)

    for i, lag in enumerate(lags):
        if lag <= 0 or lag >= horizon:
            continue
        left = x[:, :-lag, :].reshape(-1, n_assets)
        right = x[:, lag:, :].reshape(-1, n_assets)

        left = left - left.mean(axis=0, keepdims=True)
        right = right - right.mean(axis=0, keepdims=True)

        num = np.mean(left * right, axis=0)
        den = np.sqrt(np.mean(left ** 2, axis=0) * np.mean(right ** 2, axis=0))
        den = np.where(den < 1e-12, 1.0, den)
        out[i] = np.clip(num / den, -1.0, 1.0)
    return out


def max_drawdown_from_returns(paths: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each path and asset.

    Parameters
    ----------
    paths:
        Array of shape `[n_paths, horizon, n_assets]`.

    Returns
    -------
    np.ndarray
        Shape `[n_paths, n_assets]`.
    """
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3:
        raise ValueError("Expected paths with shape [n_paths, horizon, n_assets].")

    cumulative_curve = np.cumprod(1.0 + x, axis=1)
    running_max = np.maximum.accumulate(cumulative_curve, axis=1)
    running_max = np.where(running_max < 1e-12, 1.0, running_max)
    drawdowns = 1.0 - cumulative_curve / running_max
    return np.max(drawdowns, axis=1)
