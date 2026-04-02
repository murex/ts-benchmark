"""Filtered historical simulation baseline."""

from __future__ import annotations

import math

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


class FilteredHistoricalSimulationModel(ScenarioModel):
    """EWMA-filtered residual bootstrap with deterministic volatility reinflation."""

    name = "filtered_historical_simulation"

    def __init__(
        self,
        ewma_lambda: float = 0.97,
        block_size: int = 5,
        long_run_blend: float = 0.02,
        min_vol: float = 1e-4,
        use_empirical_mean: bool = True,
    ):
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1).")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if not 0.0 <= long_run_blend <= 1.0:
            raise ValueError("long_run_blend must be in [0, 1].")
        if min_vol <= 0.0:
            raise ValueError("min_vol must be positive.")

        self.ewma_lambda = float(ewma_lambda)
        self.block_size = int(block_size)
        self.long_run_blend = float(long_run_blend)
        self.min_vol = float(min_vol)
        self.use_empirical_mean = bool(use_empirical_mean)

        self.train_returns: np.ndarray | None = None
        self.residual_pool: np.ndarray | None = None
        self.residual_paths: list[np.ndarray] | None = None
        self.long_run_var: float | None = None
        self.mean_vector: np.ndarray | None = None
        self.n_assets: int | None = None
        self._n_train_windows: int = 0
        self._n_train_paths: int = 0

    def _ewma_variances(self, returns: np.ndarray, *, mean_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        centered = x - mean_vector
        sigma2 = np.zeros(len(centered), dtype=float)
        warmup = min(20, len(centered))
        sigma2[0] = float(np.mean(np.var(centered[:warmup], axis=0))) + self.min_vol ** 2

        for t in range(1, len(centered)):
            realized_var = float(np.mean(centered[t - 1] ** 2))
            sigma2[t] = self.ewma_lambda * sigma2[t - 1] + (1.0 - self.ewma_lambda) * realized_var
            sigma2[t] = max(sigma2[t], self.min_vol ** 2)
        return sigma2

    def _trim_residuals(self, residuals: np.ndarray) -> np.ndarray:
        drop = min(5, max(0, len(residuals) - self.block_size))
        trimmed = residuals[drop:]
        if len(trimmed) < self.block_size:
            raise ValueError("filtered historical simulation needs more residual observations per training path.")
        return trimmed

    def fit(self, train_data: TrainingData) -> "FilteredHistoricalSimulationModel":
        train_data.validate()
        x = np.asarray(train_data.returns, dtype=float)
        if x.ndim != 2:
            raise ValueError("train_returns must be shaped [time, n_assets].")

        self._n_train_windows = 0
        self._n_train_paths = 0
        self.residual_paths = None
        self.n_assets = int(x.shape[1])

        if train_data.forecast_windows is not None or train_data.path_collection is not None:
            paths = [np.asarray(path, dtype=float) for path in train_data.benchmark_training_paths()]
            if any(len(path) < max(6, self.block_size) for path in paths):
                raise ValueError("training paths are too short for filtered historical simulation.")

            self.train_returns = np.concatenate(paths, axis=0)
            self.mean_vector = (
                self.train_returns.mean(axis=0) if self.use_empirical_mean else np.zeros(self.n_assets, dtype=float)
            )

            sigma2_parts: list[np.ndarray] = []
            raw_residual_paths: list[np.ndarray] = []
            for path in paths:
                sigma2 = self._ewma_variances(path, mean_vector=self.mean_vector)
                sigma = np.sqrt(np.maximum(sigma2, self.min_vol ** 2))[:, None]
                centered = path - self.mean_vector
                raw_residual_paths.append(self._trim_residuals(centered / sigma))
                sigma2_parts.append(sigma2)

            residual_mean = np.concatenate(raw_residual_paths, axis=0).mean(axis=0, keepdims=True)
            self.residual_paths = [path - residual_mean for path in raw_residual_paths]
            self.residual_pool = np.concatenate(self.residual_paths, axis=0)
            self.long_run_var = float(np.mean(np.concatenate(sigma2_parts, axis=0)))
            if train_data.forecast_windows is not None:
                self._n_train_windows = len(paths)
            else:
                self._n_train_paths = len(paths)
            return self

        if len(x) < max(25, self.block_size):
            raise ValueError("train_returns are too short for filtered historical simulation.")

        self.train_returns = x
        self.mean_vector = x.mean(axis=0) if self.use_empirical_mean else np.zeros(self.n_assets, dtype=float)
        sigma2 = self._ewma_variances(x, mean_vector=self.mean_vector)
        sigma = np.sqrt(np.maximum(sigma2, self.min_vol ** 2))[:, None]
        centered = x - self.mean_vector
        residuals = self._trim_residuals(centered / sigma)
        residuals = residuals - residuals.mean(axis=0, keepdims=True)
        self.residual_pool = residuals
        self.long_run_var = float(np.mean(sigma2))
        return self

    def _initial_sigma2_from_context(self, context: np.ndarray | None) -> float:
        assert self.long_run_var is not None
        assert self.mean_vector is not None
        if context is None or len(context) == 0:
            return float(self.long_run_var)

        centered = np.asarray(context, dtype=float) - self.mean_vector
        warmup = min(10, len(centered))
        sigma2 = float(np.mean(np.var(centered[:warmup], axis=0))) + self.min_vol ** 2
        for step in range(len(centered)):
            sigma2 = self.ewma_lambda * sigma2 + (1.0 - self.ewma_lambda) * float(np.mean(centered[step] ** 2))
            sigma2 = max(sigma2, self.min_vol ** 2)
        return sigma2

    def _sample_residual_path(self, horizon: int, rng: np.random.Generator) -> np.ndarray:
        if self.residual_paths is not None:
            if self.block_size == 1:
                out = np.empty((horizon, self.n_assets), dtype=float)
                for step in range(horizon):
                    path = self.residual_paths[int(rng.integers(0, len(self.residual_paths)))]
                    out[step] = path[int(rng.integers(0, len(path)))]
                return out

            n_blocks = int(math.ceil(horizon / self.block_size))
            blocks: list[np.ndarray] = []
            for _ in range(n_blocks):
                path = self.residual_paths[int(rng.integers(0, len(self.residual_paths)))]
                start = int(rng.integers(0, len(path) - self.block_size + 1))
                blocks.append(path[start : start + self.block_size])
            return np.concatenate(blocks, axis=0)[:horizon]

        assert self.residual_pool is not None
        if self.block_size == 1:
            idx = rng.integers(0, len(self.residual_pool), size=horizon)
            return self.residual_pool[idx]

        n_blocks = int(math.ceil(horizon / self.block_size))
        max_start = len(self.residual_pool) - self.block_size
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [self.residual_pool[start : start + self.block_size] for start in starts]
        return np.concatenate(blocks, axis=0)[:horizon]

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.residual_pool is None or self.long_run_var is None or self.mean_vector is None or self.n_assets is None:
            raise RuntimeError("The model must be fit before sampling.")

        rng = np.random.default_rng(request.seed)
        samples = np.empty((request.n_scenarios, request.horizon, self.n_assets), dtype=float)
        min_var = self.min_vol ** 2

        for scenario in range(request.n_scenarios):
            sigma2 = self._initial_sigma2_from_context(request.context)
            residual_path = self._sample_residual_path(request.horizon, rng)

            for step in range(request.horizon):
                sigma_t = math.sqrt(max(sigma2, min_var))
                centered = sigma_t * residual_path[step]
                returns_t = self.mean_vector + centered
                samples[scenario, step] = returns_t

                sigma2 = self.ewma_lambda * sigma2 + (1.0 - self.ewma_lambda) * float(np.mean(centered ** 2))
                sigma2 = (1.0 - self.long_run_blend) * sigma2 + self.long_run_blend * self.long_run_var
                sigma2 = max(float(sigma2), min_var)

        result = ScenarioSamples(samples=samples)
        result.validate(
            expected_horizon=request.horizon,
            expected_n_assets=request.n_assets,
        )
        return result

    def model_info(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "ewma_lambda": self.ewma_lambda,
            "block_size": self.block_size,
            "long_run_blend": self.long_run_blend,
            "min_vol": self.min_vol,
            "use_empirical_mean": self.use_empirical_mean,
            "n_train_windows": self._n_train_windows,
            "n_train_paths": self._n_train_paths,
        }
