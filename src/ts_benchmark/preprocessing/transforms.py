"""Transform objects used by the benchmark preprocessing pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


class Transform(ABC):
    """Small fit/transform/inverse_transform interface.

    Transforms operate asset-wise on arrays whose last dimension is the asset axis.
    They are designed to work on both rank-2 `[time, n_assets]` arrays and rank-3
    `[n_scenarios, horizon, n_assets]` arrays.
    """

    name: str = "transform"
    is_fitted: bool = False

    def fit(self, x: np.ndarray) -> "Transform":
        self.is_fitted = True
        return self

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the forward transform."""

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Map transformed data back to the original scale when possible."""
        return np.asarray(x, dtype=float)

    def summary(self) -> Dict[str, Any]:
        return {"type": self.name}

    @staticmethod
    def _flatten_last_axis(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(x, dtype=float)
        if arr.ndim < 2:
            raise ValueError("Expected an array with at least two dimensions.")
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        return flat, shape

    @staticmethod
    def _restore_last_axis(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(flat, dtype=float).reshape(shape)


class IdentityTransform(Transform):
    name = "identity"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


@dataclass
class DemeanTransform(Transform):
    name: str = "demean"
    mean_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "DemeanTransform":
        flat, _ = self._flatten_last_axis(np.asarray(x, dtype=float))
        self.mean_ = flat.mean(axis=0)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("DemeanTransform must be fit before transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis(flat - self.mean_, shape)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("DemeanTransform must be fit before inverse_transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis(flat + self.mean_, shape)

    def summary(self) -> Dict[str, Any]:
        return {"type": self.name}


@dataclass
class StandardScalerTransform(Transform):
    eps: float = 1e-8
    with_mean: bool = True
    with_std: bool = True
    name: str = "standard_scale"
    mean_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "StandardScalerTransform":
        flat, _ = self._flatten_last_axis(np.asarray(x, dtype=float))
        self.mean_ = flat.mean(axis=0) if self.with_mean else np.zeros(flat.shape[1], dtype=float)
        std = flat.std(axis=0, ddof=1) if self.with_std else np.ones(flat.shape[1], dtype=float)
        self.scale_ = np.where(std < self.eps, 1.0, std)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerTransform must be fit before transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis((flat - self.mean_) / self.scale_, shape)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerTransform must be fit before inverse_transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis(flat * self.scale_ + self.mean_, shape)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
        }


@dataclass
class RobustScalerTransform(Transform):
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    eps: float = 1e-8
    name: str = "robust_scale"
    center_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "RobustScalerTransform":
        flat, _ = self._flatten_last_axis(np.asarray(x, dtype=float))
        self.center_ = np.median(flat, axis=0)
        q_low = np.quantile(flat, self.lower_quantile, axis=0)
        q_high = np.quantile(flat, self.upper_quantile, axis=0)
        iqr = q_high - q_low
        self.scale_ = np.where(iqr < self.eps, 1.0, iqr)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScalerTransform must be fit before transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis((flat - self.center_) / self.scale_, shape)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScalerTransform must be fit before inverse_transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        return self._restore_last_axis(flat * self.scale_ + self.center_, shape)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
        }


@dataclass
class MinMaxScalerTransform(Transform):
    feature_min: float = 0.0
    feature_max: float = 1.0
    clip: bool = False
    eps: float = 1e-8
    name: str = "min_max_scale"
    data_min_: np.ndarray | None = field(default=None, init=False)
    data_max_: np.ndarray | None = field(default=None, init=False)
    scale_range_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "MinMaxScalerTransform":
        if self.feature_max <= self.feature_min:
            raise ValueError("feature_max must be strictly greater than feature_min.")
        flat, _ = self._flatten_last_axis(np.asarray(x, dtype=float))
        self.data_min_ = flat.min(axis=0)
        self.data_max_ = flat.max(axis=0)
        raw_range = self.data_max_ - self.data_min_
        self.scale_range_ = np.where(raw_range < self.eps, 1.0, raw_range)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.scale_range_ is None:
            raise RuntimeError("MinMaxScalerTransform must be fit before transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        feature_range = self.feature_max - self.feature_min
        transformed = ((flat - self.data_min_) / self.scale_range_) * feature_range + self.feature_min
        if self.clip:
            transformed = np.clip(transformed, self.feature_min, self.feature_max)
        return self._restore_last_axis(transformed, shape)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.scale_range_ is None:
            raise RuntimeError("MinMaxScalerTransform must be fit before inverse_transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        feature_range = self.feature_max - self.feature_min
        restored = ((flat - self.feature_min) / feature_range) * self.scale_range_ + self.data_min_
        return self._restore_last_axis(restored, shape)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
            "clip": self.clip,
        }


@dataclass
class ClipTransform(Transform):
    min_value: float = -0.25
    max_value: float = 0.25
    name: str = "clip"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), self.min_value, self.max_value)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass
class WinsorizeTransform(Transform):
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    name: str = "winsorize"
    lower_: np.ndarray | None = field(default=None, init=False)
    upper_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "WinsorizeTransform":
        flat, _ = self._flatten_last_axis(np.asarray(x, dtype=float))
        self.lower_ = np.quantile(flat, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(flat, self.upper_quantile, axis=0)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("WinsorizeTransform must be fit before transform.")
        flat, shape = self._flatten_last_axis(np.asarray(x, dtype=float))
        clipped = np.minimum(np.maximum(flat, self.lower_), self.upper_)
        return self._restore_last_axis(clipped, shape)

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
        }
