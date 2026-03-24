"""Distribution-matching metrics against synthetic reference scenarios."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from ..utils.stats import (
    flatten_marginals as _flatten_marginals,
    flatten_path_samples,
    lagged_autocorrelation_paths,
    max_drawdown_from_returns,
    safe_corrcoef,
    to_paths as _to_paths,
)

def moment_errors(samples: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    x = _flatten_marginals(samples)
    y = _flatten_marginals(reference)

    mean_err = np.mean(np.abs(x.mean(axis=0) - y.mean(axis=0)))
    vol_err = np.mean(np.abs(x.std(axis=0, ddof=1) - y.std(axis=0, ddof=1)))

    skew_x = np.nan_to_num(skew(x, axis=0, bias=False), nan=0.0)
    skew_y = np.nan_to_num(skew(y, axis=0, bias=False), nan=0.0)
    skew_err = np.mean(np.abs(skew_x - skew_y))

    kurt_x = np.nan_to_num(kurtosis(x, axis=0, fisher=True, bias=False), nan=0.0)
    kurt_y = np.nan_to_num(kurtosis(y, axis=0, fisher=True, bias=False), nan=0.0)
    kurt_err = np.mean(np.abs(kurt_x - kurt_y))

    return {
        "mean_error": float(mean_err),
        "volatility_error": float(vol_err),
        "skew_error": float(skew_err),
        "excess_kurtosis_error": float(kurt_err),
    }

def correlation_matrix_error(samples: np.ndarray, reference: np.ndarray) -> float:
    x = _flatten_marginals(samples)
    y = _flatten_marginals(reference)

    corr_x = safe_corrcoef(x)
    corr_y = safe_corrcoef(y)
    mask = ~np.eye(corr_x.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr_x[mask] - corr_y[mask])))

def autocorrelation_error(
    samples: np.ndarray,
    reference: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 5),
) -> float:
    x = _to_paths(samples)
    y = _to_paths(reference)
    acf_x = lagged_autocorrelation_paths(x, lags=lags, square=False)
    acf_y = lagged_autocorrelation_paths(y, lags=lags, square=False)
    return float(np.mean(np.abs(acf_x - acf_y)))

def squared_autocorrelation_error(
    samples: np.ndarray,
    reference: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 5),
) -> float:
    x = _to_paths(samples)
    y = _to_paths(reference)
    acf_x = lagged_autocorrelation_paths(x, lags=lags, square=True)
    acf_y = lagged_autocorrelation_paths(y, lags=lags, square=True)
    return float(np.mean(np.abs(acf_x - acf_y)))

def tail_risk_errors(
    samples: np.ndarray,
    reference: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    x = _flatten_marginals(samples)
    y = _flatten_marginals(reference)

    var_x = np.quantile(x, alpha, axis=0)
    var_y = np.quantile(y, alpha, axis=0)

    es_x = np.array([x[x[:, i] <= var_x[i], i].mean() for i in range(x.shape[1])], dtype=float)
    es_y = np.array([y[y[:, i] <= var_y[i], i].mean() for i in range(y.shape[1])], dtype=float)

    return {
        "var_95_error": float(np.mean(np.abs(var_x - var_y))),
        "es_95_error": float(np.mean(np.abs(es_x - es_y))),
    }

def max_drawdown_error(samples: np.ndarray, reference: np.ndarray) -> float:
    x = _to_paths(samples)
    y = _to_paths(reference)

    mdd_x = max_drawdown_from_returns(x)
    mdd_y = max_drawdown_from_returns(y)

    mean_mdd_x = np.mean(mdd_x, axis=0)
    mean_mdd_y = np.mean(mdd_y, axis=0)
    return float(np.mean(np.abs(mean_mdd_x - mean_mdd_y)))

def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)

def mmd_rbf(
    samples: np.ndarray,
    reference: np.ndarray,
    max_points: int = 512,
    seed: int = 123,
) -> float:
    x = flatten_path_samples(samples)
    y = flatten_path_samples(reference)

    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        x = x[rng.choice(len(x), size=max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), size=max_points, replace=False)]

    z = np.concatenate([x, y], axis=0)
    dists = _pairwise_sq_dists(z, z)
    positive = dists[dists > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / max(median, 1e-6)

    k_xx = np.exp(-gamma * _pairwise_sq_dists(x, x))
    k_yy = np.exp(-gamma * _pairwise_sq_dists(y, y))
    k_xy = np.exp(-gamma * _pairwise_sq_dists(x, y))

    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))

def compute_distributional_metrics(
    samples: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    metrics = {}
    metrics.update(moment_errors(samples, reference))
    metrics["cross_correlation_error"] = correlation_matrix_error(samples, reference)
    metrics["autocorrelation_error"] = autocorrelation_error(samples, reference)
    metrics["squared_autocorrelation_error"] = squared_autocorrelation_error(samples, reference)
    metrics.update(tail_risk_errors(samples, reference))
    metrics["max_drawdown_error"] = max_drawdown_error(samples, reference)
    metrics["mmd_rbf"] = mmd_rbf(samples, reference)
    return metrics


from .definition import register_metric_compute  # noqa: E402

register_metric_compute("mean_error", lambda s, r: moment_errors(s, r)["mean_error"])
register_metric_compute("volatility_error", lambda s, r: moment_errors(s, r)["volatility_error"])
register_metric_compute("skew_error", lambda s, r: moment_errors(s, r)["skew_error"])
register_metric_compute("excess_kurtosis_error", lambda s, r: moment_errors(s, r)["excess_kurtosis_error"])
register_metric_compute("cross_correlation_error", correlation_matrix_error)
register_metric_compute("autocorrelation_error", autocorrelation_error)
register_metric_compute("squared_autocorrelation_error", squared_autocorrelation_error)
register_metric_compute("var_95_error", lambda s, r: tail_risk_errors(s, r)["var_95_error"])
register_metric_compute("es_95_error", lambda s, r: tail_risk_errors(s, r)["es_95_error"])
register_metric_compute("max_drawdown_error", max_drawdown_error)
register_metric_compute("mmd_rbf", mmd_rbf)
