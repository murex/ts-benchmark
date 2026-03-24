"""Distribution summary statistics for diagnostic analysis."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from ..dataset.runtime import DatasetInstance
from ..utils.stats import flatten_marginals as _flatten_marginals, lagged_autocorrelation_paths, safe_corrcoef, to_paths as _to_paths


def _off_diagonal_corr_mean(flattened: np.ndarray) -> float:
    x = np.asarray(flattened, dtype=float)
    if x.shape[1] <= 1 or x.shape[0] <= 1:
        return 0.0
    corr = safe_corrcoef(x)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(corr[mask]))


def _lag1_autocorr_mean(paths: np.ndarray) -> float:
    x = np.asarray(paths, dtype=float)
    if x.shape[1] <= 1:
        return 0.0
    acf = lagged_autocorrelation_paths(x, lags=(1,), square=False)
    return float(np.mean(acf[0]))


def _series_summary_row(
    *,
    model: str,
    series_type: str,
    values: np.ndarray,
) -> dict[str, object]:
    flat = _flatten_marginals(values)
    paths = _to_paths(values)
    asset_means = flat.mean(axis=0)
    asset_stds = flat.std(axis=0, ddof=1) if flat.shape[0] > 1 else np.zeros(flat.shape[1], dtype=float)
    asset_skews = np.nan_to_num(skew(flat, axis=0, bias=False), nan=0.0)
    asset_kurts = np.nan_to_num(kurtosis(flat, axis=0, fisher=True, bias=False), nan=0.0)
    quantiles = np.quantile(flat, [0.05, 0.5, 0.95], axis=0)
    return {
        "model": model,
        "series_type": series_type,
        "n_rows": int(flat.shape[0]),
        "n_assets": int(flat.shape[1]),
        "n_paths": int(paths.shape[0]),
        "mean_mean": float(np.mean(asset_means)),
        "mean_abs": float(np.mean(np.abs(asset_means))),
        "std_mean": float(np.mean(asset_stds)),
        "std_min": float(np.min(asset_stds)) if asset_stds.size else 0.0,
        "std_max": float(np.max(asset_stds)) if asset_stds.size else 0.0,
        "skew_mean": float(np.mean(asset_skews)),
        "excess_kurtosis_mean": float(np.mean(asset_kurts)),
        "q05_mean": float(np.mean(quantiles[0])),
        "q50_mean": float(np.mean(quantiles[1])),
        "q95_mean": float(np.mean(quantiles[2])),
        "lag1_autocorrelation_mean": _lag1_autocorr_mean(paths),
        "cross_asset_correlation_mean": _off_diagonal_corr_mean(flat),
    }


def _series_asset_rows(
    *,
    model: str,
    series_type: str,
    values: np.ndarray,
    asset_names: list[str],
) -> list[dict[str, object]]:
    flat = _flatten_marginals(values)
    paths = _to_paths(values)
    asset_skews = np.nan_to_num(skew(flat, axis=0, bias=False), nan=0.0)
    asset_kurts = np.nan_to_num(kurtosis(flat, axis=0, fisher=True, bias=False), nan=0.0)
    quantiles = np.quantile(flat, [0.05, 0.5, 0.95], axis=0)
    rows: list[dict[str, object]] = []
    for asset_index, asset_name in enumerate(asset_names):
        lag1 = 0.0
        if paths.shape[1] > 1:
            lag1 = float(
                np.mean(
                    lagged_autocorrelation_paths(
                        paths[:, :, asset_index : asset_index + 1],
                        lags=(1,),
                        square=False,
                    )[0]
                )
            )
        rows.append(
            {
                "model": model,
                "series_type": series_type,
                "asset": asset_name,
                "mean": float(np.mean(flat[:, asset_index])),
                "std": float(np.std(flat[:, asset_index], ddof=1)) if flat.shape[0] > 1 else 0.0,
                "skew": float(asset_skews[asset_index]),
                "excess_kurtosis": float(asset_kurts[asset_index]),
                "q05": float(quantiles[0, asset_index]),
                "q50": float(quantiles[1, asset_index]),
                "q95": float(quantiles[2, asset_index]),
                "lag1_autocorrelation": lag1,
            }
        )
    return rows


def build_distribution_summaries(
    *,
    dataset: DatasetInstance,
    generated_scenarios: Mapping[str, np.ndarray],
    reference_scenarios: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = [
        _series_summary_row(model="__dataset__", series_type="train_returns", values=dataset.train_returns),
        _series_summary_row(model="__dataset__", series_type="realized_futures", values=dataset.realized_futures),
    ]
    asset_rows = [
        *_series_asset_rows(
            model="__dataset__",
            series_type="train_returns",
            values=dataset.train_returns,
            asset_names=list(dataset.asset_names),
        ),
        *_series_asset_rows(
            model="__dataset__",
            series_type="realized_futures",
            values=dataset.realized_futures,
            asset_names=list(dataset.asset_names),
        ),
    ]

    if reference_scenarios is not None:
        summary_rows.append(
            _series_summary_row(
                model="__dataset__",
                series_type="reference_scenarios",
                values=reference_scenarios,
            )
        )
        asset_rows.extend(
            _series_asset_rows(
                model="__dataset__",
                series_type="reference_scenarios",
                values=reference_scenarios,
                asset_names=list(dataset.asset_names),
            )
        )

    for model_name, samples in generated_scenarios.items():
        summary_rows.append(
            _series_summary_row(
                model=model_name,
                series_type="generated_scenarios",
                values=samples,
            )
        )
        asset_rows.extend(
            _series_asset_rows(
                model=model_name,
                series_type="generated_scenarios",
                values=samples,
                asset_names=list(dataset.asset_names),
            )
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(asset_rows)
