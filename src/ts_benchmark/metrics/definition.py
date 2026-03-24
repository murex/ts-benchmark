"""Metric definition objects, registry, and ranking helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..serialization import to_jsonable
from ..utils import JsonObject

MetricComputeFn = Callable[[np.ndarray, np.ndarray], float]

METRIC_COMPUTE_REGISTRY: dict[str, MetricComputeFn] = {}


def register_metric_compute(name: str, fn: MetricComputeFn) -> None:
    """Register a compute function for a named metric."""
    METRIC_COMPUTE_REGISTRY[name] = fn


def compute_metric(name: str, samples: np.ndarray, target: np.ndarray) -> float:
    """Compute a single metric by name using the registry."""
    if name not in METRIC_COMPUTE_REGISTRY:
        raise KeyError(f"No compute function registered for metric '{name}'.")
    return METRIC_COMPUTE_REGISTRY[name](samples, target)


@dataclass(frozen=True)
class MetricRequirements:
    requires_realized_path: bool = False
    requires_reference_scenarios: bool = False
    multivariate_only: bool = False
    synthetic_only: bool = False



@dataclass(frozen=True)
class MetricConfig:
    name: str
    description: str | None = None
    category: str | None = None
    direction: str = "minimize"
    target_value: float | None = None
    params: JsonObject = field(default_factory=JsonObject)
    requirements: MetricRequirements = field(default_factory=MetricRequirements)
    granularity: str = "per_window"
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.params, JsonObject):
            self.params = JsonObject(self.params)

    def validate(self) -> None:
        if self.direction not in {"minimize", "maximize", "target"}:
            raise ValueError(
                f"Metric '{self.name}' has unsupported direction '{self.direction}'. "
                "Use 'minimize', 'maximize', or 'target'."
            )
        if self.direction == "target" and self.target_value is None:
            raise ValueError(f"Metric '{self.name}' uses direction='target' but has no target_value.")



def _metric(
    *,
    name: str,
    description: str,
    category: str,
    direction: str = "minimize",
    target_value: float | None = None,
    params: Mapping[str, Any] | None = None,
    requirements: MetricRequirements | None = None,
    granularity: str = "per_window",
    aggregation: str = "mean",
) -> MetricConfig:
    metric = MetricConfig(
        name=name,
        description=description,
        category=category,
        direction=direction,
        target_value=target_value,
        params=JsonObject(params),
        requirements=requirements or MetricRequirements(),
        granularity=granularity,
        aggregation=aggregation,
    )
    metric.validate()
    return metric


BUILTIN_METRIC_DEFINITIONS: dict[str, MetricConfig] = {
    "crps": _metric(
        name="crps",
        description="Continuous ranked probability score against realized future paths.",
        category="distribution",
        requirements=MetricRequirements(requires_realized_path=True),
    ),
    "energy_score": _metric(
        name="energy_score",
        description="Multivariate energy score against realized future paths.",
        category="distribution",
        requirements=MetricRequirements(requires_realized_path=True),
    ),
    "predictive_mean_mse": _metric(
        name="predictive_mean_mse",
        description="Mean squared error of the predictive mean against realized future paths.",
        category="point",
        requirements=MetricRequirements(requires_realized_path=True),
    ),
    "coverage_90_error": _metric(
        name="coverage_90_error",
        description="Absolute error of 90% predictive interval coverage against realized future paths.",
        category="calibration",
        params={"alpha": 0.10, "nominal_coverage": 0.90},
        requirements=MetricRequirements(requires_realized_path=True),
    ),
    "mean_error": _metric(
        name="mean_error",
        description="Absolute error in marginal means against reference scenarios.",
        category="moment",
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "volatility_error": _metric(
        name="volatility_error",
        description="Absolute error in marginal volatilities against reference scenarios.",
        category="moment",
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "skew_error": _metric(
        name="skew_error",
        description="Absolute error in marginal skewness against reference scenarios.",
        category="moment",
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "excess_kurtosis_error": _metric(
        name="excess_kurtosis_error",
        description="Absolute error in marginal excess kurtosis against reference scenarios.",
        category="moment",
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "cross_correlation_error": _metric(
        name="cross_correlation_error",
        description="Absolute error in cross-asset correlation structure against reference scenarios.",
        category="dependence",
        requirements=MetricRequirements(
            requires_reference_scenarios=True,
            multivariate_only=True,
            synthetic_only=True,
        ),
    ),
    "autocorrelation_error": _metric(
        name="autocorrelation_error",
        description="Absolute error in path autocorrelation against reference scenarios.",
        category="dependence",
        params={"lags": [1, 2, 5]},
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "squared_autocorrelation_error": _metric(
        name="squared_autocorrelation_error",
        description="Absolute error in squared-path autocorrelation against reference scenarios.",
        category="dependence",
        params={"lags": [1, 2, 5]},
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "var_95_error": _metric(
        name="var_95_error",
        description="Absolute error in 95% value-at-risk against reference scenarios.",
        category="tail",
        params={"alpha": 0.05},
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "es_95_error": _metric(
        name="es_95_error",
        description="Absolute error in 95% expected shortfall against reference scenarios.",
        category="tail",
        params={"alpha": 0.05},
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "max_drawdown_error": _metric(
        name="max_drawdown_error",
        description="Absolute error in expected max drawdown against reference scenarios.",
        category="tail",
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
    "mmd_rbf": _metric(
        name="mmd_rbf",
        description="Maximum mean discrepancy with an RBF kernel against reference scenarios.",
        category="distribution",
        params={"kernel": "rbf"},
        requirements=MetricRequirements(requires_reference_scenarios=True, synthetic_only=True),
    ),
}


def available_metric_names() -> list[str]:
    return list(BUILTIN_METRIC_DEFINITIONS)


def _normalize_requirements(
    value: MetricRequirements | Mapping[str, Any] | None,
) -> MetricRequirements:
    if isinstance(value, MetricRequirements):
        return value
    if value is None:
        return MetricRequirements()
    if not isinstance(value, Mapping):
        raise TypeError("Metric requirements must be a mapping, MetricRequirements, or None.")
    return MetricRequirements(**dict(value))


def normalize_metric_config(value: MetricConfig | Mapping[str, Any]) -> MetricConfig:
    if isinstance(value, MetricConfig):
        name = value.name
        payload = to_jsonable(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
        name = str(payload.get("name") or "").strip()
    else:
        raise TypeError("Metric definitions must be provided as objects.")

    if not name:
        raise ValueError("Each metric object must define a non-empty 'name'.")
    if name not in BUILTIN_METRIC_DEFINITIONS:
        raise ValueError(
            f"Unsupported metric '{name}'. Available metrics: {available_metric_names()}"
        )

    canonical = BUILTIN_METRIC_DEFINITIONS[name]
    if isinstance(value, MetricConfig):
        if value != canonical:
            raise ValueError(
                f"Metric '{name}' differs from the built-in definition. "
                "Metric objects may select built-in metrics by name but may not redefine them."
            )
        return canonical

    comparison_payload = dict(payload)
    comparison_payload.pop("name", None)
    if "requirements" in comparison_payload:
        comparison_payload["requirements"] = to_jsonable(
            _normalize_requirements(comparison_payload["requirements"])
        )

    canonical_payload = to_jsonable(canonical)
    canonical_payload.pop("name", None)
    for key, provided in comparison_payload.items():
        if provided != canonical_payload.get(key):
            raise ValueError(
                f"Metric '{name}' sets '{key}' to {provided!r}, but the built-in definition uses "
                f"{canonical_payload.get(key)!r}. Metric objects may not redefine built-in metric semantics."
            )
    return canonical


def resolve_metric_configs(values: Iterable[MetricConfig | Mapping[str, Any]] | None) -> list[MetricConfig]:
    if not values:
        return []
    resolved = [normalize_metric_config(value) for value in values]
    names = [metric.name for metric in resolved]
    if len(names) != len(set(names)):
        raise ValueError("Metric names must be unique.")
    return resolved


def default_metric_configs() -> list[MetricConfig]:
    return [BUILTIN_METRIC_DEFINITIONS[name] for name in available_metric_names()]


def _metric_applicability_reasons(
    metric: MetricConfig,
    *,
    has_reference_scenarios: bool,
    n_assets: int | None = None,
    dataset_source: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    requirements = metric.requirements
    if requirements.requires_reference_scenarios and not has_reference_scenarios:
        reasons.append("reference scenarios")
    if requirements.multivariate_only and n_assets is not None and n_assets < 2:
        reasons.append("multiple assets")
    if requirements.synthetic_only and dataset_source is not None and dataset_source != "synthetic":
        reasons.append("synthetic datasets")
    return reasons


def select_metric_configs_for_run(
    requested: Iterable[MetricConfig | Mapping[str, Any]] | None,
    *,
    has_reference_scenarios: bool,
    n_assets: int | None = None,
    dataset_source: str | None = None,
) -> list[MetricConfig]:
    requested_list = [] if requested is None else list(requested)
    explicit = bool(requested_list)
    metrics = resolve_metric_configs(requested_list) if explicit else default_metric_configs()
    selected: list[MetricConfig] = []
    invalid: list[str] = []
    for metric in metrics:
        reasons = _metric_applicability_reasons(
            metric,
            has_reference_scenarios=has_reference_scenarios,
            n_assets=n_assets,
            dataset_source=dataset_source,
        )
        if reasons:
            if explicit:
                invalid.append(f"{metric.name} (requires {', '.join(reasons)})")
            continue
        selected.append(metric)
    if invalid:
        raise ValueError(
            "Selected metric(s) are not applicable to this dataset/run: "
            + ", ".join(invalid)
        )
    return selected


def rank_metrics_table(
    metrics_table: pd.DataFrame,
    metric_configs: Iterable[MetricConfig],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_list = list(metric_configs)
    ordered_names = [metric.name for metric in metric_list]
    missing = [name for name in ordered_names if name not in metrics_table.columns]
    if missing:
        raise KeyError(
            f"Requested metric(s) not found: {missing}. Available: {list(metrics_table.columns)}"
        )

    filtered = metrics_table[ordered_names].copy()
    rank_columns: dict[str, pd.Series] = {}
    for metric in metric_list:
        metric.validate()
        values = filtered[metric.name]
        if metric.direction == "minimize":
            rank_columns[metric.name] = values.rank(method="average", ascending=True)
        elif metric.direction == "maximize":
            rank_columns[metric.name] = values.rank(method="average", ascending=False)
        else:
            if metric.target_value is None:
                raise ValueError(
                    f"Metric '{metric.name}' uses direction='target' but has no target_value."
                )
            rank_columns[metric.name] = (values - float(metric.target_value)).abs().rank(
                method="average",
                ascending=True,
            )

    rank_table = pd.DataFrame(rank_columns, index=filtered.index)
    rank_table["average_rank"] = rank_table.mean(axis=1)
    filtered["average_rank"] = rank_table["average_rank"]
    filtered = filtered.sort_values("average_rank", ascending=True)
    rank_table = rank_table.loc[filtered.index]
    return filtered, rank_table
