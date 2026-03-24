"""Formal contract between the benchmark and scenario-generation models.

The benchmark owns the protocol definition (split sizes, generation mode,
context length when relevant, horizon, evaluation stride, supervised
training-window stride, and evaluation scenario counts). Models receive
that protocol from the benchmark rather than redeclaring those values in
their own parameter blocks.
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
class TrainingData:
    """Training data passed to a benchmark-compatible model."""

    returns: np.ndarray
    protocol: Protocol
    asset_names: list[str] | None = None
    freq: str | None = None
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

    @property
    def n_timesteps(self) -> int:
        return int(np.asarray(self.returns).shape[0])

    @property
    def n_assets(self) -> int:
        return int(np.asarray(self.returns).shape[1])


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
        """Fit the model on the training history and benchmark-owned protocol."""

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
