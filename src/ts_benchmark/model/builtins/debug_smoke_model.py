"""Small contract model used by diagnostics tests."""

from __future__ import annotations

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


class DebugSmokeModel(ScenarioModel):
    name = "debug_smoke_model"

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fit_log: list[dict[str, float]] = []

    def fit(self, train_data: TrainingData) -> "DebugSmokeModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0, ddof=1) + 1e-6
        self._fit_log = [
            {
                "n_timesteps": float(x.shape[0]),
                "n_assets": float(x.shape[1]),
                "mean_abs_return": float(np.mean(np.abs(x))),
                "n_train_windows": float(0 if train_data.forecast_windows is None else train_data.forecast_windows.contexts.shape[0]),
                "n_train_paths": float(0 if train_data.path_collection is None else len(train_data.path_collection.paths)),
                "scale": self.scale,
            }
        ]
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self._mean is None or self._std is None:
            raise RuntimeError("Model must be fit before sampling.")
        rng = np.random.default_rng(request.seed)
        noise = rng.normal(
            loc=0.0,
            scale=self._std[None, None, :] * max(self.scale, 1e-6),
            size=(request.n_scenarios, request.horizon, request.n_assets),
        )
        samples = self._mean[None, None, :] + noise
        result = ScenarioSamples(
            samples=samples,
            metadata={"fit_log_tail": list(self._fit_log[-1:])},
        )
        result.validate(
            expected_horizon=request.horizon,
            expected_n_assets=request.n_assets,
        )
        return result

    def debug_artifacts(self) -> dict[str, object] | None:
        return {
            "name": self.name,
            "scale": self.scale,
            "fit_log": list(self._fit_log),
        }
