"""Simple historical return bootstrap baseline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


def _resolve_torch_device(requested: str | None) -> tuple[Any | None, Any | None]:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is an optional backend here
        return None, None

    text = None if requested is None else str(requested).strip().lower()
    if text in {None, "", "auto"}:
        if torch.cuda.is_available():
            return torch, torch.device("cuda")
        backends = getattr(torch, "backends", None)
        mps_backend = None if backends is None else getattr(backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch, torch.device("mps")
        return torch, torch.device("cpu")

    if text == "mps":
        backends = getattr(torch, "backends", None)
        mps_backend = None if backends is None else getattr(backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch, torch.device("mps")
        return torch, torch.device("cpu")

    try:
        device = torch.device(text)
    except Exception:
        return torch, torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch, torch.device("cpu")
    return torch, device


def _block_bootstrap_indices(
    *,
    n_rows: int,
    horizon: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size == 1:
        return rng.integers(0, n_rows, size=(n_paths, horizon), dtype=np.int64)

    n_blocks = int(math.ceil(horizon / block_size))
    max_start = n_rows - block_size
    starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks), dtype=np.int64)
    offsets = np.arange(block_size, dtype=np.int64)
    idx = (starts[..., None] + offsets).reshape(n_paths, -1)
    return idx[:, :horizon]


class HistoricalBootstrapModel(ScenarioModel):
    """Resample historical return vectors with optional block bootstrap."""

    name = "historical_bootstrap"

    def __init__(self, block_size: int = 5):
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.block_size = int(block_size)
        self.train_returns: np.ndarray | None = None
        self.n_assets: int | None = None
        self._runtime_device: str | None = None
        self._resolved_device: str | None = None
        self._train_returns_torch: Any | None = None

    def fit(self, train_data: TrainingData) -> "HistoricalBootstrapModel":
        train_data.validate()
        x = np.asarray(train_data.returns, dtype=float)
        if x.ndim != 2:
            raise ValueError("train_returns must be shaped [time, n_assets].")
        if len(x) < self.block_size:
            raise ValueError("train_returns are shorter than block_size.")

        self._runtime_device = None if train_data.runtime is None else train_data.runtime.device
        self.train_returns = x
        self.n_assets = x.shape[1]

        torch, device = _resolve_torch_device(self._runtime_device)
        self._resolved_device = None if device is None else str(device)
        self._train_returns_torch = None
        if torch is not None and device is not None and device.type != "cpu":
            self._train_returns_torch = torch.as_tensor(x, dtype=torch.float32, device=device)
        return self

    def _sample_single_path(self, horizon: int, rng: np.random.Generator) -> np.ndarray:
        assert self.train_returns is not None
        if self.block_size == 1:
            idx = rng.integers(0, len(self.train_returns), size=horizon)
            return self.train_returns[idx]

        n_blocks = int(math.ceil(horizon / self.block_size))
        max_start = len(self.train_returns) - self.block_size
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [self.train_returns[s : s + self.block_size] for s in starts]
        path = np.concatenate(blocks, axis=0)[:horizon]
        return path

    def _sample_torch(
        self,
        *,
        horizon: int,
        n_scenarios: int,
        seed: int | None,
    ) -> np.ndarray:
        assert self.train_returns is not None
        assert self._train_returns_torch is not None

        rng = np.random.default_rng(seed)
        idx_np = _block_bootstrap_indices(
            n_rows=len(self.train_returns),
            horizon=horizon,
            n_paths=n_scenarios,
            block_size=self.block_size,
            rng=rng,
        )
        torch, _ = _resolve_torch_device(self._runtime_device)
        assert torch is not None
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=self._train_returns_torch.device)
        return self._train_returns_torch[idx].detach().cpu().numpy().astype(float, copy=False)

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.train_returns is None or self.n_assets is None:
            raise RuntimeError("The model must be fit before sampling.")

        horizon = request.horizon
        n_scenarios = request.n_scenarios
        seed = request.seed

        if self._train_returns_torch is not None:
            samples = self._sample_torch(horizon=horizon, n_scenarios=n_scenarios, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            samples = np.zeros((n_scenarios, horizon, self.n_assets), dtype=float)
            for s in range(n_scenarios):
                samples[s] = self._sample_single_path(horizon, rng)

        result = ScenarioSamples(samples=samples)
        result.validate(
            expected_horizon=horizon,
            expected_n_assets=request.n_assets,
        )
        return result

    def model_info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "block_size": self.block_size,
            "runtime_device": self._runtime_device,
            "resolved_device": self._resolved_device,
        }
