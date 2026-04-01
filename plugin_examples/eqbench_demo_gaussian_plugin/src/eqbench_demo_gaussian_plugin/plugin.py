from __future__ import annotations

import numpy as np

from ts_benchmark.model import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


class GaussianPluginModel(ScenarioModel):
    name = "demo_gaussian_plugin"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = float(ridge)
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.protocol_ = None

    def fit(self, train_data: TrainingData) -> "GaussianPluginModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        self.mean_ = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov) + self.ridge * np.eye(x.shape[1])
        self.cov_ = cov
        self.protocol_ = train_data.protocol
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.mean_ is None or self.cov_ is None:
            raise RuntimeError("Model must be fit before sampling.")
        rng = np.random.default_rng(request.seed)
        draws = rng.multivariate_normal(
            mean=self.mean_,
            cov=self.cov_,
            size=(request.n_scenarios, request.horizon),
        )
        result = ScenarioSamples(samples=np.asarray(draws, dtype=float))
        result.validate(expected_horizon=request.horizon, expected_n_assets=request.n_assets)
        return result


def build_model(*, ridge: float = 1e-6):
    return GaussianPluginModel(ridge=ridge)
