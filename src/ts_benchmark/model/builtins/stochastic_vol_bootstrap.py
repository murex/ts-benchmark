"""Historical bootstrap with stochastic volatility dynamics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData
from .historical_bootstrap import _block_bootstrap_indices, _resolve_torch_device


class StochasticVolatilityBootstrapModel(ScenarioModel):
    """Resample standardized residuals and evolve volatility recursively.

    The model:
    1. estimates historical volatilities with EWMA,
    2. standardizes returns into residuals,
    3. resamples residual vectors jointly across assets,
    4. simulates future volatility using the recent context and multiplicative vol shocks.
    """

    name = "historical_sv_bootstrap"

    def __init__(
        self,
        ewma_lambda: float = 0.97,
        block_size: int = 5,
        vol_of_vol: float = 0.10,
        long_run_blend: float = 0.02,
        min_vol: float = 1e-4,
    ):
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1).")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if vol_of_vol < 0.0:
            raise ValueError("vol_of_vol must be non-negative.")
        if not 0.0 <= long_run_blend <= 1.0:
            raise ValueError("long_run_blend must be in [0, 1].")

        self.ewma_lambda = float(ewma_lambda)
        self.block_size = int(block_size)
        self.vol_of_vol = float(vol_of_vol)
        self.long_run_blend = float(long_run_blend)
        self.min_vol = float(min_vol)

        self.train_returns: np.ndarray | None = None
        self.residual_pool: np.ndarray | None = None
        self.long_run_var: np.ndarray | None = None
        self.mean_vector: np.ndarray | None = None
        self.n_assets: int | None = None
        self._runtime_device: str | None = None
        self._resolved_device: str | None = None
        self._residual_pool_torch: Any | None = None
        self._long_run_var_torch: Any | None = None
        self._mean_vector_torch: Any | None = None

    def _ewma_variances(self, returns: np.ndarray) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        sigma2 = np.zeros_like(x)
        warmup = min(20, len(x))
        sigma2[0] = np.var(x[:warmup], axis=0) + self.min_vol ** 2

        for t in range(1, len(x)):
            sigma2[t] = self.ewma_lambda * sigma2[t - 1] + (1.0 - self.ewma_lambda) * (x[t - 1] ** 2)
            sigma2[t] = np.maximum(sigma2[t], self.min_vol ** 2)
        return sigma2

    def fit(self, train_data: TrainingData) -> "StochasticVolatilityBootstrapModel":
        train_data.validate()
        x = np.asarray(train_data.returns, dtype=float)
        if x.ndim != 2:
            raise ValueError("train_returns must be shaped [time, n_assets].")
        if len(x) < max(25, self.block_size):
            raise ValueError("train_returns are too short for the stochastic-volatility bootstrap.")

        self._runtime_device = None if train_data.runtime is None else train_data.runtime.device

        sigma2 = self._ewma_variances(x)
        sigma = np.sqrt(np.maximum(sigma2, self.min_vol ** 2))
        residuals = x / sigma
        residuals = residuals - residuals.mean(axis=0, keepdims=True)

        self.train_returns = x
        self.residual_pool = residuals[5:]
        self.long_run_var = np.mean(sigma2, axis=0)
        self.mean_vector = np.mean(x, axis=0)
        self.n_assets = x.shape[1]

        torch, device = _resolve_torch_device(self._runtime_device)
        self._resolved_device = None if device is None else str(device)
        self._residual_pool_torch = None
        self._long_run_var_torch = None
        self._mean_vector_torch = None
        if torch is not None and device is not None and device.type != "cpu":
            self._residual_pool_torch = torch.as_tensor(self.residual_pool, dtype=torch.float32, device=device)
            self._long_run_var_torch = torch.as_tensor(self.long_run_var, dtype=torch.float32, device=device)
            self._mean_vector_torch = torch.as_tensor(self.mean_vector, dtype=torch.float32, device=device)
        return self

    def _initial_sigma2_from_context(self, context: np.ndarray) -> np.ndarray:
        assert self.long_run_var is not None
        if context is None or len(context) == 0:
            return np.array(self.long_run_var, dtype=float, copy=True)

        x = np.asarray(context, dtype=float)
        warmup = min(10, len(x))
        sigma2 = np.var(x[:warmup], axis=0) + self.min_vol ** 2
        for t in range(len(x)):
            sigma2 = self.ewma_lambda * sigma2 + (1.0 - self.ewma_lambda) * (x[t] ** 2)
            sigma2 = np.maximum(sigma2, self.min_vol ** 2)
        return sigma2

    def _sample_residual_path(self, horizon: int, rng: np.random.Generator) -> np.ndarray:
        assert self.residual_pool is not None
        if self.block_size == 1:
            idx = rng.integers(0, len(self.residual_pool), size=horizon)
            return self.residual_pool[idx]

        n_blocks = int(math.ceil(horizon / self.block_size))
        max_start = len(self.residual_pool) - self.block_size
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [self.residual_pool[s : s + self.block_size] for s in starts]
        return np.concatenate(blocks, axis=0)[:horizon]

    def _sample_torch(
        self,
        *,
        context: np.ndarray,
        horizon: int,
        n_scenarios: int,
        seed: int | None,
    ) -> np.ndarray:
        assert self.residual_pool is not None
        assert self.long_run_var is not None
        assert self.mean_vector is not None
        assert self.n_assets is not None
        assert self._residual_pool_torch is not None
        assert self._long_run_var_torch is not None
        assert self._mean_vector_torch is not None

        rng = np.random.default_rng(seed)
        init_sigma2 = self._initial_sigma2_from_context(context)
        idx_np = _block_bootstrap_indices(
            n_rows=len(self.residual_pool),
            horizon=horizon,
            n_paths=n_scenarios,
            block_size=self.block_size,
            rng=rng,
        )
        torch, _ = _resolve_torch_device(self._runtime_device)
        assert torch is not None

        device = self._residual_pool_torch.device
        residual_idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        residuals = self._residual_pool_torch[residual_idx]

        sigma2 = torch.as_tensor(
            np.broadcast_to(init_sigma2, (n_scenarios, self.n_assets)).copy(),
            dtype=torch.float32,
            device=device,
        )
        paths = torch.empty((n_scenarios, horizon, self.n_assets), dtype=torch.float32, device=device)
        min_var = float(self.min_vol ** 2)

        if self.vol_of_vol > 0.0:
            common_vol_shocks = torch.as_tensor(
                np.exp(self.vol_of_vol * rng.normal(size=(n_scenarios, horizon, 1))),
                dtype=torch.float32,
                device=device,
            )
            idio_vol_shocks = torch.as_tensor(
                np.exp(0.5 * self.vol_of_vol * rng.normal(size=(n_scenarios, horizon, self.n_assets))),
                dtype=torch.float32,
                device=device,
            )
        else:
            common_vol_shocks = None
            idio_vol_shocks = None

        for t in range(horizon):
            sigma_t = torch.sqrt(torch.clamp(sigma2, min=min_var))
            if common_vol_shocks is not None and idio_vol_shocks is not None:
                sigma_t = sigma_t * common_vol_shocks[:, t, :] * idio_vol_shocks[:, t, :]

            returns_t = self._mean_vector_torch + sigma_t * residuals[:, t, :]
            paths[:, t, :] = returns_t

            sigma2 = self.ewma_lambda * sigma2 + (1.0 - self.ewma_lambda) * (returns_t ** 2)
            sigma2 = (1.0 - self.long_run_blend) * sigma2 + self.long_run_blend * self._long_run_var_torch
            sigma2 = torch.clamp(sigma2, min=min_var)

        return paths.detach().cpu().numpy().astype(float, copy=False)

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.residual_pool is None or self.long_run_var is None or self.mean_vector is None or self.n_assets is None:
            raise RuntimeError("The model must be fit before sampling.")

        context = request.context
        horizon = request.horizon
        n_scenarios = request.n_scenarios
        seed = request.seed

        if self._residual_pool_torch is not None:
            samples = self._sample_torch(
                context=context,
                horizon=horizon,
                n_scenarios=n_scenarios,
                seed=seed,
            )
        else:
            rng = np.random.default_rng(seed)
            init_sigma2 = self._initial_sigma2_from_context(context)

            samples = np.zeros((n_scenarios, horizon, self.n_assets), dtype=float)
            for s in range(n_scenarios):
                sigma2 = np.array(init_sigma2, dtype=float, copy=True)
                residuals = self._sample_residual_path(horizon, rng)

                for t in range(horizon):
                    common_vol_shock = np.exp(self.vol_of_vol * rng.normal())
                    idio_vol_shock = np.exp(0.5 * self.vol_of_vol * rng.normal(size=self.n_assets))
                    sigma_t = np.sqrt(np.maximum(sigma2, self.min_vol ** 2)) * common_vol_shock * idio_vol_shock

                    returns_t = self.mean_vector + sigma_t * residuals[t]
                    samples[s, t] = returns_t

                    sigma2 = self.ewma_lambda * sigma2 + (1.0 - self.ewma_lambda) * (returns_t ** 2)
                    sigma2 = (1.0 - self.long_run_blend) * sigma2 + self.long_run_blend * self.long_run_var
                    sigma2 = np.maximum(sigma2, self.min_vol ** 2)

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
            "vol_of_vol": self.vol_of_vol,
            "ewma_lambda": self.ewma_lambda,
        }
