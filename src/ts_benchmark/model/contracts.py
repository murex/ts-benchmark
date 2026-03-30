"""Formal contract between the benchmark and scenario-generation models.

The benchmark owns the protocol definition (split sizes, generation mode,
context length when relevant, horizon, evaluation stride, supervised
training-window stride, unconditional training-path length when needed,
and evaluation scenario counts). Models receive that protocol from the
benchmark rather than redeclaring those values in their own parameter
blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from ..benchmark.protocol import Protocol
from ..utils import JsonObject


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime information owned by the benchmark/UI layer."""

    device: str | None = None
    seed: int | None = None


@dataclass
class ForecastWindowCollection:
    """Normalized forecast training windows owned by the benchmark.

    `source_kind` is either:
    - `single_path`: windows extracted from one realized training path
    - `path_dataset`: windows extracted from multiple independent training paths
    """

    contexts: np.ndarray
    targets: np.ndarray
    source_kind: str
    stride: int

    def validate(
        self,
        *,
        expected_n_assets: int | None = None,
        expected_context_length: int | None = None,
        expected_horizon: int | None = None,
    ) -> None:
        if self.source_kind not in {"single_path", "path_dataset"}:
            raise ValueError(
                "source_kind must be either 'single_path' or 'path_dataset'."
            )
        contexts = np.asarray(self.contexts, dtype=float)
        targets = np.asarray(self.targets, dtype=float)
        if contexts.ndim != 3:
            raise ValueError("contexts must be shaped [n_windows, context_length, n_assets].")
        if targets.ndim != 3:
            raise ValueError("targets must be shaped [n_windows, horizon, n_assets].")
        if contexts.shape[0] != targets.shape[0]:
            raise ValueError("contexts and targets must have the same number of windows.")
        if contexts.shape[0] < 1:
            raise ValueError("forecast windows must contain at least one window.")
        if contexts.shape[1] < 1:
            raise ValueError("forecast contexts must contain at least one timestep.")
        if targets.shape[1] < 1:
            raise ValueError("forecast targets must contain at least one timestep.")
        if contexts.shape[2] != targets.shape[2]:
            raise ValueError("forecast contexts and targets must have the same n_assets.")
        if expected_n_assets is not None and contexts.shape[2] != expected_n_assets:
            raise ValueError(
                f"forecast windows have n_assets={contexts.shape[2]} but expected {expected_n_assets}."
            )
        if expected_context_length is not None and contexts.shape[1] != expected_context_length:
            raise ValueError(
                "forecast contexts have length "
                f"{contexts.shape[1]} but expected context_length={expected_context_length}."
            )
        if expected_horizon is not None and targets.shape[1] != expected_horizon:
            raise ValueError(
                f"forecast targets have horizon {targets.shape[1]} but expected horizon={expected_horizon}."
            )
        if not np.isfinite(contexts).all():
            raise ValueError("forecast contexts contain non-finite values.")
        if not np.isfinite(targets).all():
            raise ValueError("forecast targets contain non-finite values.")
        if self.stride < 1:
            raise ValueError("forecast window stride must be positive.")
        self.contexts = contexts
        self.targets = targets

    def as_paths(self) -> list[np.ndarray]:
        self.validate()
        return [
            np.concatenate([self.contexts[index], self.targets[index]], axis=0)
            for index in range(self.contexts.shape[0])
        ]


@dataclass
class TrainPathCollection:
    """Normalized unconditional training paths owned by the benchmark.

    `source_kind` is either:
    - `path_dataset`: genuine independent training trajectories
    - `windowed_path`: benchmark-extracted paths from a longer training series
    """

    paths: list[np.ndarray]
    source_kind: str
    window_length: int | None = None
    stride: int | None = None

    def validate(self, *, expected_n_assets: int | None = None) -> None:
        if self.source_kind not in {"path_dataset", "windowed_path"}:
            raise ValueError(
                "source_kind must be either 'path_dataset' or 'windowed_path'."
            )
        normalized: list[np.ndarray] = []
        if not self.paths:
            raise ValueError("paths must contain at least one training path.")
        for index, path in enumerate(self.paths):
            x = np.asarray(path, dtype=float)
            if x.ndim != 2:
                raise ValueError(f"paths[{index}] must be shaped [time, n_assets].")
            if x.shape[0] < 1:
                raise ValueError(f"paths[{index}] must contain at least one timestep.")
            if expected_n_assets is not None and x.shape[1] != expected_n_assets:
                raise ValueError(
                    f"paths[{index}] has n_assets={x.shape[1]} but expected {expected_n_assets}."
                )
            if not np.isfinite(x).all():
                raise ValueError(f"paths[{index}] contains non-finite values.")
            normalized.append(x)
        self.paths = normalized
        if self.source_kind == "windowed_path":
            if self.window_length is None or self.window_length < 1:
                raise ValueError(
                    "windowed_path collections require a positive window_length."
                )
            if self.stride is None or self.stride < 1:
                raise ValueError("windowed_path collections require a positive stride.")
            for index, path in enumerate(self.paths):
                if path.shape[0] != self.window_length:
                    raise ValueError(
                        f"paths[{index}] has length {path.shape[0]} but window_length is {self.window_length}."
                    )
        elif self.window_length is not None or self.stride is not None:
            raise ValueError(
                "path_dataset collections must not define window_length or stride."
            )


@dataclass
class TrainingData:
    """Training data passed to a benchmark-compatible model.

    `returns` is always the benchmark's realized training history.
    `forecast_windows` is populated in forecast mode when the benchmark
    normalizes training into benchmark-owned supervised windows.
    `path_collection` is additionally populated in unconditional mode when
    the benchmark normalizes training into a dataset of independent paths.
    """

    returns: np.ndarray
    protocol: Protocol
    asset_names: list[str] | None = None
    freq: str | None = None
    forecast_windows: ForecastWindowCollection | None = None
    path_collection: TrainPathCollection | None = None
    runtime: RuntimeContext = field(default_factory=RuntimeContext)
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    def validate(self) -> None:
        x = np.asarray(self.returns, dtype=float)
        if x.ndim != 2:
            raise ValueError("returns must be shaped [time, n_assets].")
        if x.shape[0] < 2:
            raise ValueError("returns must contain at least two timesteps.")
        self.protocol.validate()
        self.returns = x
        if self.protocol.generation_mode == "forecast":
            if self.forecast_windows is None:
                raise ValueError(
                    "forecast training data must include forecast_windows."
                )
            if self.path_collection is not None:
                raise ValueError(
                    "path_collection must be omitted in forecast mode."
                )
            self.forecast_windows.validate(
                expected_n_assets=int(x.shape[1]),
                expected_context_length=int(self.protocol.context_length),
                expected_horizon=int(self.protocol.horizon),
            )
            return
        if self.forecast_windows is not None:
            raise ValueError(
                "forecast_windows must be omitted in unconditional mode."
            )
        if self.path_collection is None:
            raise ValueError(
                "unconditional training data must include a path_collection."
            )
        self.path_collection.validate(expected_n_assets=int(x.shape[1]))

    @property
    def n_timesteps(self) -> int:
        return int(np.asarray(self.returns).shape[0])

    @property
    def n_assets(self) -> int:
        return int(np.asarray(self.returns).shape[1])

    def benchmark_training_paths(self) -> list[np.ndarray]:
        """Return the benchmark-owned fit payload as a list of full paths."""

        if self.forecast_windows is not None:
            return self.forecast_windows.as_paths()
        if self.path_collection is not None:
            return [np.asarray(path, dtype=float) for path in self.path_collection.paths]
        return [np.asarray(self.returns, dtype=float)]

    def concatenated_training_values(self) -> np.ndarray:
        """Return benchmark-owned fit values concatenated across training examples."""

        return np.concatenate(self.benchmark_training_paths(), axis=0)


@dataclass
class ScenarioRequest:
    """Single generation request owned by the benchmark runtime."""

    context: np.ndarray
    horizon: int
    n_scenarios: int
    protocol: Protocol
    mode: str = "forecast"
    seed: int | None = None
    asset_names: list[str] | None = None
    freq: str | None = None
    runtime: RuntimeContext = field(default_factory=RuntimeContext)
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    def validate(self) -> None:
        x = np.asarray(self.context, dtype=float)
        if x.ndim != 2:
            raise ValueError("context must be shaped [context_length, n_assets].")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive.")
        self.protocol.validate()
        if self.mode not in {"forecast", "unconditional"}:
            raise ValueError("mode must be either 'forecast' or 'unconditional'.")
        if self.mode != self.protocol.generation_mode:
            raise ValueError(
                f"request mode {self.mode!r} does not match benchmark protocol generation_mode "
                f"{self.protocol.generation_mode!r}."
            )
        if self.mode == "forecast":
            if x.shape[0] < 1:
                raise ValueError("context must contain at least one timestep in forecast mode.")
            if x.shape[0] != self.protocol.context_length:
                raise ValueError(
                    f"context has length {x.shape[0]} but benchmark protocol requires "
                    f"context_length={self.protocol.context_length}."
                )
        elif x.shape[0] != 0:
            raise ValueError("context must be empty in unconditional mode.")
        if self.horizon != self.protocol.horizon:
            raise ValueError(
                f"request horizon {self.horizon} does not match benchmark protocol horizon "
                f"{self.protocol.horizon}."
            )
        self.context = x

    @property
    def n_assets(self) -> int:
        return int(np.asarray(self.context).shape[1])


@dataclass
class ScenarioSamples:
    """Model output understood by the benchmark."""

    samples: np.ndarray
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    def validate(
        self,
        *,
        expected_horizon: Optional[int] = None,
        expected_n_assets: Optional[int] = None,
    ) -> None:
        x = np.asarray(self.samples, dtype=float)
        if x.ndim != 3:
            raise ValueError("samples must be shaped [n_scenarios, horizon, n_assets].")
        if x.shape[0] < 1:
            raise ValueError("samples must contain at least one scenario.")
        if expected_horizon is not None and x.shape[1] != expected_horizon:
            raise ValueError(
                f"samples horizon {x.shape[1]} does not match expected horizon {expected_horizon}."
            )
        if expected_n_assets is not None and x.shape[2] != expected_n_assets:
            raise ValueError(
                f"samples asset dimension {x.shape[2]} does not match expected n_assets {expected_n_assets}."
            )
        if not np.isfinite(x).all():
            raise ValueError("samples contain non-finite values.")
        self.samples = x


class ScenarioModel(ABC):
    """Formal benchmark/model contract.

    To integrate a new model into the benchmark, subclass this class and
    implement `fit` plus `sample`.
    """

    name: str = "scenario_model"

    @abstractmethod
    def fit(self, train_data: TrainingData) -> "ScenarioModel":
        """Fit the model on benchmark-owned training data and protocol."""

    @abstractmethod
    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        """Generate conditional future scenarios for one context window."""

    def model_info(self) -> Dict[str, Any]:
        """Optional metadata displayed or logged by the benchmark."""
        return {
            "name": getattr(self, "name", self.__class__.__name__),
            "class": self.__class__.__name__,
        }

    def debug_artifacts(self) -> Dict[str, Any] | None:
        """Optional rich diagnostics persisted by the benchmark when enabled."""
        return None
