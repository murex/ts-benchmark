"""Functional smoke-test checks for model output quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset.runtime import DatasetInstance
from ..run.definition import FunctionalSmokeConfig
from ..utils.stats import flatten_marginals as _flatten_marginals, safe_corrcoef
from .types import BenchmarkResults


def _functional_target(
    dataset: DatasetInstance,
    reference_scenarios: np.ndarray | None,
) -> tuple[str, np.ndarray]:
    if reference_scenarios is not None:
        return "reference_scenarios", np.asarray(reference_scenarios, dtype=float)
    return "realized_futures", np.asarray(dataset.realized_futures, dtype=float)


def _mean_abs_error(generated: np.ndarray, target: np.ndarray) -> float:
    generated_flat = _flatten_marginals(generated)
    target_flat = _flatten_marginals(target)
    generated_mean = generated_flat.mean(axis=0)
    target_mean = target_flat.mean(axis=0)
    return float(np.mean(np.abs(generated_mean - target_mean)))


def _std_ratio_bounds(generated: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    generated_flat = _flatten_marginals(generated)
    target_flat = _flatten_marginals(target)
    generated_std = generated_flat.std(axis=0, ddof=1)
    target_std = target_flat.std(axis=0, ddof=1)
    ratios = generated_std / np.maximum(target_std, 1e-12)
    return float(np.min(ratios)), float(np.max(ratios))


def _cross_correlation_error(generated: np.ndarray, target: np.ndarray) -> float:
    generated_flat = _flatten_marginals(generated)
    target_flat = _flatten_marginals(target)
    if generated_flat.shape[1] <= 1:
        return 0.0
    corr_generated = safe_corrcoef(generated_flat)
    corr_target = safe_corrcoef(target_flat)
    mask = ~np.eye(corr_generated.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr_generated[mask] - corr_target[mask])))


def build_functional_smoke_tables(
    *,
    smoke_config: FunctionalSmokeConfig,
    dataset: DatasetInstance,
    results: BenchmarkResults,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generated_scenarios = results.scenario_outputs()
    if not generated_scenarios:
        raise ValueError("Functional smoke diagnostics require generated_scenarios.")

    comparator_name, comparator_values = _functional_target(dataset, results.reference_scenarios)
    checks_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    metrics_frame = results.metrics_frame().drop(columns=["average_rank"], errors="ignore")

    for model_name, samples in generated_scenarios.items():
        model_metrics = (
            {str(key): value for key, value in metrics_frame.loc[model_name].items()}
            if model_name in metrics_frame.index
            else {}
        )

        model_checks: list[dict[str, object]] = []

        def add_check(
            *,
            check: str,
            enabled: bool,
            observed: float | bool | None,
            threshold: float | bool | None,
            passed: bool | None,
            comparator: str | None = None,
            message: str | None = None,
        ) -> None:
            status = "skipped"
            if enabled:
                status = "pass" if bool(passed) else "fail"
            model_checks.append(
                {
                    "model": model_name,
                    "check": check,
                    "status": status,
                    "observed": observed,
                    "threshold": threshold,
                    "comparator": comparator,
                    "message": message,
                }
            )

        finite_observed = bool(np.isfinite(samples).all())
        add_check(
            check="finite_required",
            enabled=bool(smoke_config.finite_required),
            observed=finite_observed,
            threshold=True,
            passed=finite_observed,
            message="All generated samples must be finite.",
        )

        if smoke_config.mean_abs_error_max is not None:
            observed = _mean_abs_error(samples, comparator_values)
            add_check(
                check="mean_abs_error_max",
                enabled=True,
                observed=observed,
                threshold=smoke_config.mean_abs_error_max,
                passed=observed <= smoke_config.mean_abs_error_max,
                comparator=comparator_name,
            )

        if smoke_config.std_ratio_min is not None:
            observed_min, _ = _std_ratio_bounds(samples, comparator_values)
            add_check(
                check="std_ratio_min",
                enabled=True,
                observed=observed_min,
                threshold=smoke_config.std_ratio_min,
                passed=observed_min >= smoke_config.std_ratio_min,
                comparator=comparator_name,
            )

        if smoke_config.std_ratio_max is not None:
            _, observed_max = _std_ratio_bounds(samples, comparator_values)
            add_check(
                check="std_ratio_max",
                enabled=True,
                observed=observed_max,
                threshold=smoke_config.std_ratio_max,
                passed=observed_max <= smoke_config.std_ratio_max,
                comparator=comparator_name,
            )

        if smoke_config.crps_max is not None:
            observed = model_metrics.get("crps")
            add_check(
                check="crps_max",
                enabled=observed is not None,
                observed=observed,
                threshold=smoke_config.crps_max,
                passed=None if observed is None else observed <= smoke_config.crps_max,
                comparator="realized_futures",
                message=None if observed is not None else "crps metric unavailable for this run.",
            )

        if smoke_config.energy_score_max is not None:
            observed = model_metrics.get("energy_score")
            add_check(
                check="energy_score_max",
                enabled=observed is not None,
                observed=observed,
                threshold=smoke_config.energy_score_max,
                passed=None if observed is None else observed <= smoke_config.energy_score_max,
                comparator="realized_futures",
                message=None if observed is not None else "energy_score metric unavailable for this run.",
            )

        if smoke_config.cross_correlation_error_max is not None:
            observed = model_metrics.get("cross_correlation_error")
            if observed is None and comparator_values is not None:
                observed = _cross_correlation_error(samples, comparator_values)
            add_check(
                check="cross_correlation_error_max",
                enabled=observed is not None,
                observed=observed,
                threshold=smoke_config.cross_correlation_error_max,
                passed=None if observed is None else observed <= smoke_config.cross_correlation_error_max,
                comparator=comparator_name,
                message=None if observed is not None else "cross-correlation comparison unavailable.",
            )

        checks_rows.extend(model_checks)

        applicable = [row for row in model_checks if row["status"] != "skipped"]
        failed = [row for row in applicable if row["status"] == "fail"]
        passed = [row for row in applicable if row["status"] == "pass"]
        overall_status = "warn"
        if failed:
            overall_status = "fail"
        elif applicable:
            overall_status = "pass"
        summary_rows.append(
            {
                "model": model_name,
                "overall_status": overall_status,
                "applicable_checks": int(len(applicable)),
                "passed_checks": int(len(passed)),
                "failed_checks": int(len(failed)),
                "skipped_checks": int(len(model_checks) - len(applicable)),
                "failed_check_names": ",".join(row["check"] for row in failed),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by=["overall_status", "failed_checks", "model"],
        ascending=[True, False, True],
        kind="stable",
    )
    check_frame = pd.DataFrame(checks_rows)
    return summary_frame, check_frame
