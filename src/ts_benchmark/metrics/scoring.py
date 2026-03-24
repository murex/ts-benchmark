"""Proper scoring rules against the realized future path."""

from __future__ import annotations

import numpy as np


def sample_crps(samples: np.ndarray, realized: np.ndarray) -> float:
    """Compute marginal CRPS from predictive samples.

    Parameters
    ----------
    samples:
        Shape `[n_contexts, n_scenarios, horizon, n_assets]`.
    realized:
        Shape `[n_contexts, horizon, n_assets]`.
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(realized, dtype=float)

    if x.ndim != 4 or y.ndim != 3:
        raise ValueError("Unexpected shapes for samples or realized.")
    if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
        raise ValueError("samples and realized are not aligned.")

    values = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1])
    targets = y.reshape(-1)

    mean_abs = np.mean(np.abs(values - targets[:, None]), axis=1)

    values_sorted = np.sort(values, axis=1)
    n = values.shape[1]
    coeff = (2 * np.arange(1, n + 1) - n - 1).astype(float)
    pairwise_mean = 2.0 * np.sum(values_sorted * coeff[None, :], axis=1) / (n ** 2)

    crps = mean_abs - 0.5 * pairwise_mean
    return float(np.mean(crps))

def energy_score(
    samples: np.ndarray,
    realized: np.ndarray,
    beta: float = 1.0,
    max_scenarios: int = 64,
) -> float:
    """Multivariate energy score over whole forecast paths."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(realized, dtype=float)

    if x.ndim != 4 or y.ndim != 3:
        raise ValueError("Unexpected shapes for samples or realized.")

    n_contexts, n_scenarios, horizon, n_assets = x.shape
    n_used = min(max_scenarios, n_scenarios)

    paths = x[:, :n_used].reshape(n_contexts, n_used, horizon * n_assets)
    targets = y.reshape(n_contexts, horizon * n_assets)

    scores = []
    for i in range(n_contexts):
        x_i = paths[i]
        y_i = targets[i][None, :]

        d_xy = np.linalg.norm(x_i - y_i, axis=1) ** beta
        pairwise = np.linalg.norm(x_i[:, None, :] - x_i[None, :, :], axis=-1) ** beta

        score_i = np.mean(d_xy) - 0.5 * np.mean(pairwise)
        scores.append(score_i)

    return float(np.mean(scores))

def predictive_mean_mse(samples: np.ndarray, realized: np.ndarray) -> float:
    """MSE of the predictive mean against the realized future path."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(realized, dtype=float)
    mean_forecast = np.mean(x, axis=1)
    return float(np.mean((mean_forecast - y) ** 2))

def coverage_error(
    samples: np.ndarray,
    realized: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Absolute coverage error for the central prediction interval."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(realized, dtype=float)

    lower = np.quantile(x, alpha / 2.0, axis=1)
    upper = np.quantile(x, 1.0 - alpha / 2.0, axis=1)

    covered = (y >= lower) & (y <= upper)
    empirical_coverage = float(np.mean(covered))
    nominal = 1.0 - alpha
    return abs(empirical_coverage - nominal)

def compute_sample_scoring_metrics(
    samples: np.ndarray,
    realized: np.ndarray,
) -> dict[str, float]:
    """Bundle of sample-based proper scoring metrics."""
    return {
        "crps": sample_crps(samples, realized),
        "energy_score": energy_score(samples, realized),
        "predictive_mean_mse": predictive_mean_mse(samples, realized),
        "coverage_90_error": coverage_error(samples, realized, alpha=0.10),
    }


from .definition import register_metric_compute  # noqa: E402

register_metric_compute("crps", sample_crps)
register_metric_compute("energy_score", energy_score)
register_metric_compute("predictive_mean_mse", predictive_mean_mse)
register_metric_compute("coverage_90_error", lambda s, r: coverage_error(s, r, alpha=0.10))
