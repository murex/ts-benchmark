"""Shared benchmark protocol definition used in config and runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class _ProtocolBase:
    horizon: int
    n_model_scenarios: int = 64
    n_reference_scenarios: int = 128
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name in ("horizon", "n_model_scenarios", "n_reference_scenarios"):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if self.horizon < 1:
            raise ValueError("horizon must be positive.")
        if self.n_model_scenarios < 1:
            raise ValueError("n_model_scenarios must be positive.")
        if self.n_reference_scenarios < 1:
            raise ValueError("n_reference_scenarios must be positive.")
        self._validate_branch()

    def _validate_branch(self) -> None:
        raise NotImplementedError

    @property
    def generation_mode(self) -> str:
        raise NotImplementedError

    @property
    def unconditional_train_data_mode(self) -> str | None:
        return None

    @property
    def unconditional_train_window_length(self) -> int | None:
        return None

    @property
    def unconditional_n_train_paths(self) -> int | None:
        return None

    @property
    def unconditional_n_realized_paths(self) -> int | None:
        return None

    @property
    def unconditional_n_eval_paths(self) -> int | None:
        return self.unconditional_n_realized_paths


@dataclass(frozen=True, kw_only=True)
class ForecastProtocol(_ProtocolBase):
    context_length: int
    train_size: int
    test_size: int
    eval_stride: int = 1
    train_stride: int = 1
    kind: str = field(default="forecast", init=False)

    @property
    def generation_mode(self) -> str:
        return "forecast"

    def _validate_branch(self) -> None:
        for field_name in ("context_length", "train_size", "test_size", "eval_stride", "train_stride"):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if self.context_length < 1:
            raise ValueError("context_length must be positive for forecast mode.")
        if self.train_size < 2:
            raise ValueError("train_size must be at least 2.")
        if self.test_size < 1:
            raise ValueError("test_size must be positive.")
        if self.eval_stride < 1:
            raise ValueError("eval_stride must be positive.")
        if self.train_stride < 1:
            raise ValueError("train_stride must be positive.")


@dataclass(frozen=True, kw_only=True)
class UnconditionalWindowedProtocol(_ProtocolBase):
    train_size: int
    test_size: int
    eval_stride: int = 1
    train_stride: int = 1
    kind: str = field(default="unconditional_windowed", init=False)

    @property
    def generation_mode(self) -> str:
        return "unconditional"

    @property
    def context_length(self) -> int:
        return 0

    @property
    def unconditional_train_data_mode(self) -> str | None:
        return "windowed_path"

    @property
    def unconditional_train_window_length(self) -> int | None:
        return self.horizon

    def _validate_branch(self) -> None:
        for field_name in ("train_size", "test_size", "eval_stride", "train_stride"):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if self.train_size < 2:
            raise ValueError("train_size must be at least 2.")
        if self.test_size < 1:
            raise ValueError("test_size must be positive.")
        if self.train_size < self.horizon:
            raise ValueError("train_size must be greater than or equal to horizon.")
        if self.test_size < self.horizon:
            raise ValueError("test_size must be greater than or equal to horizon.")
        if self.eval_stride < 1:
            raise ValueError("eval_stride must be positive.")
        if self.train_stride < 1:
            raise ValueError("train_stride must be positive.")


@dataclass(frozen=True, kw_only=True)
class UnconditionalPathDatasetProtocol(_ProtocolBase):
    n_train_paths: int
    n_realized_paths: int
    kind: str = field(default="unconditional_path_dataset", init=False)

    @property
    def generation_mode(self) -> str:
        return "unconditional"

    @property
    def context_length(self) -> int:
        return 0

    @property
    def train_size(self) -> int:
        return self.horizon

    @property
    def test_size(self) -> int:
        return 0

    @property
    def eval_stride(self) -> int:
        return 1

    @property
    def train_stride(self) -> int:
        return 1

    @property
    def unconditional_train_data_mode(self) -> str | None:
        return "path_dataset"

    @property
    def unconditional_n_train_paths(self) -> int | None:
        return self.n_train_paths

    @property
    def unconditional_n_realized_paths(self) -> int | None:
        return self.n_realized_paths

    def _validate_branch(self) -> None:
        for field_name in ("n_train_paths", "n_realized_paths"):
            value = getattr(self, field_name)
            if not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
        if self.n_train_paths < 1:
            raise ValueError("n_train_paths must be positive.")
        if self.n_realized_paths < 1:
            raise ValueError("n_realized_paths must be positive.")


Protocol = ForecastProtocol | UnconditionalWindowedProtocol | UnconditionalPathDatasetProtocol


def protocol_from_mapping(value: Mapping[str, Any]) -> Protocol:
    kind = str(value.get("kind") or "").strip()
    if kind == "forecast":
        forecast = _as_mapping(value.get("forecast"), field_name="benchmark.protocol.forecast")
        return ForecastProtocol(
            horizon=int(value["horizon"]),
            n_model_scenarios=int(value.get("n_model_scenarios", 64)),
            n_reference_scenarios=int(value.get("n_reference_scenarios", 128)),
            context_length=int(forecast["context_length"]),
            train_size=int(forecast["train_size"]),
            test_size=int(forecast["test_size"]),
            eval_stride=int(forecast.get("eval_stride", 1)),
            train_stride=int(forecast.get("train_stride", 1)),
        )
    if kind == "unconditional_windowed":
        branch = _as_mapping(
            value.get("unconditional_windowed"),
            field_name="benchmark.protocol.unconditional_windowed",
        )
        return UnconditionalWindowedProtocol(
            horizon=int(value["horizon"]),
            n_model_scenarios=int(value.get("n_model_scenarios", 64)),
            n_reference_scenarios=int(value.get("n_reference_scenarios", 128)),
            train_size=int(branch["train_size"]),
            test_size=int(branch["test_size"]),
            eval_stride=int(branch.get("eval_stride", 1)),
            train_stride=int(branch.get("train_stride", 1)),
        )
    if kind == "unconditional_path_dataset":
        branch = _as_mapping(
            value.get("unconditional_path_dataset"),
            field_name="benchmark.protocol.unconditional_path_dataset",
        )
        return UnconditionalPathDatasetProtocol(
            horizon=int(value["horizon"]),
            n_model_scenarios=int(value.get("n_model_scenarios", 64)),
            n_reference_scenarios=int(value.get("n_reference_scenarios", 128)),
            n_train_paths=int(branch["n_train_paths"]),
            n_realized_paths=int(branch["n_realized_paths"]),
        )
    raise ValueError(
        "benchmark.protocol.kind must be one of 'forecast', "
        "'unconditional_windowed', or 'unconditional_path_dataset'."
    )


def protocol_config_payload(protocol: Protocol) -> dict[str, Any]:
    common = {
        "kind": protocol.kind,
        "horizon": int(protocol.horizon),
        "n_model_scenarios": int(protocol.n_model_scenarios),
        "n_reference_scenarios": int(protocol.n_reference_scenarios),
    }
    if isinstance(protocol, ForecastProtocol):
        return {
            **common,
            "forecast": {
                "train_size": int(protocol.train_size),
                "test_size": int(protocol.test_size),
                "context_length": int(protocol.context_length),
                "eval_stride": int(protocol.eval_stride),
                "train_stride": int(protocol.train_stride),
            },
        }
    if isinstance(protocol, UnconditionalWindowedProtocol):
        return {
            **common,
            "unconditional_windowed": {
                "train_size": int(protocol.train_size),
                "test_size": int(protocol.test_size),
                "eval_stride": int(protocol.eval_stride),
                "train_stride": int(protocol.train_stride),
            },
        }
    return {
        **common,
        "unconditional_path_dataset": {
            "n_train_paths": int(protocol.n_train_paths),
            "n_realized_paths": int(protocol.n_realized_paths),
        },
    }


def protocol_metadata_payload(protocol: Protocol) -> dict[str, Any]:
    common = {
        "protocol_kind": protocol.kind,
        "generation_mode": protocol.generation_mode,
        "horizon": int(protocol.horizon),
        "n_model_scenarios": int(protocol.n_model_scenarios),
        "n_reference_scenarios": int(protocol.n_reference_scenarios),
    }
    if isinstance(protocol, ForecastProtocol):
        return {
            **common,
            "context_length": int(protocol.context_length),
            "train_size": int(protocol.train_size),
            "test_size": int(protocol.test_size),
            "eval_stride": int(protocol.eval_stride),
            "train_stride": int(protocol.train_stride),
        }
    if isinstance(protocol, UnconditionalWindowedProtocol):
        return {
            **common,
            "path_construction": "windowed_path",
            "train_size": int(protocol.train_size),
            "test_size": int(protocol.test_size),
            "eval_stride": int(protocol.eval_stride),
            "train_stride": int(protocol.train_stride),
        }
    return {
        **common,
        "path_construction": "path_dataset",
        "n_train_paths": int(protocol.n_train_paths),
        "n_realized_paths": int(protocol.n_realized_paths),
    }


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    return value
