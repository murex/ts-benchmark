"""Typed runtime dataset objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..benchmark.protocol import Protocol, protocol_metadata_payload
from ..serialization import to_jsonable
from ..utils import JsonObject


ReferenceSampler = Callable[[int, int | None], np.ndarray]


@dataclass
class DatasetInstance:
    """Realized dataset consumed by :class:`ScenarioBenchmark`.

    This is the materialized runtime counterpart to the benchmark's
    dataset definition. It carries the realized train split, rolling
    evaluation windows, and optionally a callable able to sample
    reference scenarios from a known data-generating process.
    """

    name: str
    source: str
    full_returns: np.ndarray
    train_returns: np.ndarray
    test_returns: np.ndarray
    contexts: np.ndarray
    realized_futures: np.ndarray
    asset_names: list[str]
    protocol: Protocol
    freq: str = "B"
    train_paths: list[np.ndarray] | None = None
    metadata: JsonObject = field(default_factory=JsonObject)
    evaluation_timestamps: list[str] | None = None
    reference_sampler: ReferenceSampler | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)
        if self.train_paths is not None:
            self.train_paths = [np.asarray(path, dtype=float) for path in self.train_paths]

    @property
    def context_length(self) -> int:
        return int(self.protocol.context_length)

    @property
    def horizon(self) -> int:
        return int(self.protocol.horizon)

    @property
    def train_size(self) -> int:
        return int(self.protocol.train_size)

    @property
    def test_size(self) -> int:
        return int(self.protocol.test_size)

    @property
    def eval_stride(self) -> int:
        return int(self.protocol.eval_stride)

    def has_reference_scenarios(self) -> bool:
        return self.reference_sampler is not None

    def sample_reference_scenarios(
        self,
        n_scenarios: int,
        seed: int | None = None,
    ) -> np.ndarray:
        if self.reference_sampler is None:
            raise RuntimeError(
                "This dataset does not expose reference scenarios. "
                "Distribution-matching metrics are only available when the dataset "
                "comes with a reference scenario generator."
            )
        return np.asarray(self.reference_sampler(int(n_scenarios), seed), dtype=float)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "freq": self.freq,
            "n_timesteps": int(np.asarray(self.full_returns).shape[0]),
            "n_assets": int(np.asarray(self.train_returns).shape[1]),
            "n_train_paths": None if self.train_paths is None else len(self.train_paths),
            "asset_names": list(self.asset_names),
            **protocol_metadata_payload(self.protocol),
            "n_eval_windows": int(np.asarray(self.contexts).shape[0]),
            "has_reference_scenarios": bool(self.has_reference_scenarios()),
            "metadata": self.metadata.to_builtin(),
        }
