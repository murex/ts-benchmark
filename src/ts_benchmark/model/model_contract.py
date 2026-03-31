"""Minimal structural contract for external time-series generators.

This module is intentionally independent of any benchmark implementation.
It defines a small, duck-typed interface that an external model package
can satisfy without inheriting from framework classes.

The intended split is:
- this file describes the model-side contract
- the caller or benchmark owns data preparation, task construction,
  evaluation, and process orchestration


In this setup, `build_estimator(...)` should return an object that
structurally satisfies `TSGeneratorEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

Array = Any  # Dense backend array, e.g. np.ndarray, torch.Tensor, or jax.Array.


class GenerationMode(str, Enum):
    """Task family requested from the model."""

    UNCONDITIONAL = "unconditional"
    FORECAST = "forecast"


@dataclass(frozen=True)
class RuntimeContext:
    """Optional execution hints supplied by the caller.

    These are caller-owned runtime hints, not model parameters.
    A model may ignore them, but if it honors them it should do so
    consistently.
    """

    device: Optional[str] = None  # Requested device identifier, e.g. "cpu" or "cuda:0".
    seed: Optional[int] = None  # Requested random seed for reproducible sampling.


@dataclass(frozen=True)
class DataSchema:
    """Static description of the target series and available covariates.

    This object describes what the model should expect structurally.
    It does not carry actual series values.
    """

    target_dim: int  # Number of target channels per timestep.
    freq: Optional[str] = None  # Optional frequency label such as "B" or "D".
    known_covariates: Optional[Mapping[str, int]] = None  # Name -> feature dimension for future-known covariates.
    observed_covariates: Optional[Mapping[str, int]] = None  # Name -> feature dimension for observed time-varying covariates.
    static_covariates: Optional[Mapping[str, int]] = None  # Name -> feature dimension for non-time-varying covariates.


@dataclass
class TSSeries:
    """Canonical single-series payload passed to the model.

    Expected shapes:
    - `values`: [time, target_dim]
    - `known_covariates[name]`: [time, covariate_dim]
    - `observed_covariates[name]`: [time, covariate_dim]
    - `static_covariates[name]`: [covariate_dim]

    The caller is expected to provide clean dense inputs. Handling of
    missing values, ragged histories, and irregular timestamps is outside
    the scope of this minimal contract.
    """

    values: Array
    known_covariates: Optional[Mapping[str, Array]] = None
    observed_covariates: Optional[Mapping[str, Array]] = None
    static_covariates: Optional[Mapping[str, Array]] = None


@dataclass
class TrainExample:
    """Single benchmark-prepared training example.

    `target` is always the model's supervised training target.

    `history` is:
    - present in forecast mode and shaped `[history_time, target_dim]`
    - the full benchmark-owned past available up to the forecast origin
    - omitted in unconditional mode

    `context` is:
    - present in forecast mode and shaped `[context_length, target_dim]`
    - the conditioning suffix exposed at generation time
    - a trailing slice of `history`
    - omitted in unconditional mode
    """

    history: Optional[TSSeries]
    context: Optional[TSSeries]
    target: TSSeries


@dataclass
class TrainData:
    """Benchmark-prepared fit dataset.

    The benchmark may construct these examples from one long path or from a
    dataset of paths. That provenance is intentionally hidden from the
    model-facing contract.

    Semantics:
    - `FORECAST`: every example defines `history`, `context`, and `target`
    - `UNCONDITIONAL`: every example defines `history=None`, `context=None`,
      and `target`
    """

    examples: Sequence[TrainExample]


@dataclass(frozen=True)
class TaskSpec:
    """What task the caller wants the model to perform.

    For `FORECAST`, `horizon` is the requested number of future steps to
    generate conditional on the history embedded in the series.

    For `UNCONDITIONAL`, `horizon` may be used as the desired generated
    sequence length if the caller chooses that convention.
    """

    mode: GenerationMode
    horizon: Optional[int] = None


@dataclass(frozen=True)
class Constraint:
    """Optional caller-side generation constraint.

    This is intentionally generic. If used, both caller and model are
    expected to agree out-of-band on supported constraint names and payload
    schemas.
    """

    name: str
    payload: Mapping[str, Any]


@dataclass
class GenerationRequest:
    """Dynamic generation request for a fitted model.

    `series` carries the conditioning data and optional covariates.
    `num_samples` is the number of trajectories the caller wants back.
    """

    series: TSSeries
    task: TaskSpec
    num_samples: int
    constraints: Optional[Sequence[Constraint]] = None
    runtime: Optional[RuntimeContext] = None


@dataclass
class GenerationResult:
    """Samples produced by the fitted model.

    Expected shape:
    - `samples`: [num_samples, generated_time, target_dim]

    For `FORECAST`, `generated_time` is typically the requested forecast
    horizon. For `UNCONDITIONAL`, it is the generated sequence length
    chosen by the caller/model convention.

    `diagnostics` is reserved for optional non-essential outputs such as
    timing summaries or model-specific debug information.
    """

    samples: Array
    diagnostics: Optional[Mapping[str, Any]] = None


@dataclass
class ModelCapabilities:
    """Feature support declared against the explicit contract objects.

    These flags are routing hints for the caller. They should reflect the
    model's real supported input surface, not aspirational features.
    """

    supported_modes: frozenset[GenerationMode]
    supports_multivariate_targets: bool = True
    supports_known_covariates: bool = False
    supports_observed_covariates: bool = False
    supports_static_covariates: bool = False
    supports_constraints: bool = False


@dataclass
class FitReport:
    """Optional summary returned by estimator fitting."""

    train_metrics: Optional[Mapping[str, float]] = None
    val_metrics: Optional[Mapping[str, float]] = None
    fit_time_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    n_parameters: Optional[int] = None
    diagnostics: Optional[Mapping[str, Any]] = None


@runtime_checkable
class FittedTSGenerator(Protocol):
    """Structural interface for a fitted generator.

    Any object with these methods satisfies the contract; inheritance from
    this protocol is not required.
    """

    def capabilities(self) -> ModelCapabilities: ...

    def sample(self, request: GenerationRequest) -> GenerationResult: ...

    def save(self, path: str | Path) -> None: ...


@runtime_checkable
class TSGeneratorEstimator(Protocol):
    """Structural interface for a trainable generator estimator.

    The estimator consumes caller-prepared training data and returns a
    fitted generator plus an optional fit summary.

    `train.examples` depends on the task family:
    - `FORECAST`: each example defines `history`, `context`, and `target`
    - `UNCONDITIONAL`: each example defines `history=None`, `context=None`,
      and `target`
    """

    def fit(
        self,
        train: TrainData,
        *,
        schema: DataSchema,
        task: TaskSpec,
        valid: Optional[TrainData] = None,
        runtime: Optional[RuntimeContext] = None,
    ) -> tuple[FittedTSGenerator, FitReport]:
        ...
