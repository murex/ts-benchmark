"""EWMA multivariate Gaussian return baseline."""

from __future__ import annotations

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData
from .gaussian_covariance import _stabilize_covariance


class EWMAGaussianModel(ScenarioModel):
    """RiskMetrics-style Gaussian baseline with recursively updated covariance."""

    name = "ewma_gaussian"

    def __init__(
        self,
        ewma_lambda: float = 0.97,
        covariance_jitter: float = 1e-6,
        use_empirical_mean: bool = True,
        long_run_blend: float = 0.02,
    ):
        if not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1).")
        if covariance_jitter < 0.0:
            raise ValueError("covariance_jitter must be non-negative.")
        if not 0.0 <= long_run_blend <= 1.0:
            raise ValueError("long_run_blend must be in [0, 1].")

        self.ewma_lambda = float(ewma_lambda)
        self.covariance_jitter = float(covariance_jitter)
        self.use_empirical_mean = bool(use_empirical_mean)
        self.long_run_blend = float(long_run_blend)

        self.mean_vector: np.ndarray | None = None
        self.long_run_covariance: np.ndarray | None = None
        self.recent_covariance: np.ndarray | None = None
        self.n_assets: int | None = None
        self._n_train_windows: int = 0
        self._n_train_paths: int = 0

    def _ewma_covariance_state(
        self,
        values: np.ndarray,
        *,
        initial_covariance: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.mean_vector is None or self.long_run_covariance is None:
            raise RuntimeError("The model must be fit before estimating covariance state.")

        x = np.asarray(values, dtype=float)
        if x.ndim != 2:
            raise ValueError("values must be shaped [time, n_assets].")

        state = np.array(
            self.long_run_covariance if initial_covariance is None else initial_covariance,
            dtype=float,
            copy=True,
        )
        for row in x:
            centered = row - self.mean_vector
            state = self.ewma_lambda * state + (1.0 - self.ewma_lambda) * np.outer(centered, centered)
            if self.long_run_blend > 0.0:
                state = (1.0 - self.long_run_blend) * state + self.long_run_blend * self.long_run_covariance
            state = 0.5 * (state + state.T)
        return _stabilize_covariance(state, jitter=max(self.covariance_jitter, 1e-12))

    def fit(self, train_data: TrainingData) -> "EWMAGaussianModel":
        train_data.validate()

        training_paths = train_data.benchmark_training_paths()
        x = np.asarray(np.concatenate(training_paths, axis=0), dtype=float)
        if x.ndim != 2:
            raise ValueError("training values must be shaped [time, n_assets].")
        if len(x) < 2:
            raise ValueError("at least two training observations are required.")

        self.mean_vector = x.mean(axis=0) if self.use_empirical_mean else np.zeros(x.shape[1], dtype=float)
        centered = x - self.mean_vector
        covariance = np.cov(centered, rowvar=False)
        self.long_run_covariance = _stabilize_covariance(covariance, jitter=max(self.covariance_jitter, 1e-12))
        self.n_assets = int(x.shape[1])

        terminal_covariances = [self._ewma_covariance_state(path) for path in training_paths]
        self.recent_covariance = _stabilize_covariance(
            np.mean(np.stack(terminal_covariances, axis=0), axis=0),
            jitter=max(self.covariance_jitter, 1e-12),
        )
        self._n_train_windows = (
            0 if train_data.forecast_windows is None else int(train_data.forecast_windows.contexts.shape[0])
        )
        self._n_train_paths = 0 if train_data.path_collection is None else len(train_data.path_collection.paths)
        return self

    def _initial_covariance(self, request: ScenarioRequest) -> np.ndarray:
        if self.long_run_covariance is None or self.recent_covariance is None:
            raise RuntimeError("The model must be fit before sampling.")
        if request.mode == "forecast":
            return self._ewma_covariance_state(request.context)
        return np.array(self.recent_covariance, dtype=float, copy=True)

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.mean_vector is None or self.long_run_covariance is None or self.recent_covariance is None:
            raise RuntimeError("The model must be fit before sampling.")

        rng = np.random.default_rng(request.seed)
        initial_covariance = self._initial_covariance(request)
        samples = np.empty((request.n_scenarios, request.horizon, request.n_assets), dtype=float)

        for scenario_index in range(request.n_scenarios):
            covariance_state = np.array(initial_covariance, dtype=float, copy=True)
            for step in range(request.horizon):
                draw = rng.multivariate_normal(mean=self.mean_vector, cov=covariance_state)
                samples[scenario_index, step] = draw
                covariance_state = self._ewma_covariance_state(
                    draw[None, :],
                    initial_covariance=covariance_state,
                )

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
            "n_assets": self.n_assets,
            "ewma_lambda": self.ewma_lambda,
            "covariance_jitter": self.covariance_jitter,
            "use_empirical_mean": self.use_empirical_mean,
            "long_run_blend": self.long_run_blend,
            "n_train_windows": self._n_train_windows,
            "n_train_paths": self._n_train_paths,
        }
