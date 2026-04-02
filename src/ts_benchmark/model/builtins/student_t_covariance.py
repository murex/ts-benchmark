"""Static multivariate Student-t return baseline."""

from __future__ import annotations

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData
from .gaussian_covariance import _stabilize_covariance


class StudentTCovarianceModel(ScenarioModel):
    """Fit a static multivariate Student-t distribution to historical returns."""

    name = "student_t_covariance"

    def __init__(
        self,
        degrees_of_freedom: float = 8.0,
        covariance_jitter: float = 1e-6,
        use_empirical_mean: bool = True,
    ):
        if degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must be > 2.")
        if covariance_jitter < 0.0:
            raise ValueError("covariance_jitter must be non-negative.")

        self.degrees_of_freedom = float(degrees_of_freedom)
        self.covariance_jitter = float(covariance_jitter)
        self.use_empirical_mean = bool(use_empirical_mean)
        self.mean_vector: np.ndarray | None = None
        self.scale_matrix: np.ndarray | None = None
        self.n_assets: int | None = None
        self._n_train_windows: int = 0
        self._n_train_paths: int = 0

    def fit(self, train_data: TrainingData) -> "StudentTCovarianceModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        if x.ndim != 2:
            raise ValueError("training values must be shaped [time, n_assets].")
        if len(x) < 2:
            raise ValueError("at least two training observations are required.")

        self.mean_vector = x.mean(axis=0) if self.use_empirical_mean else np.zeros(x.shape[1], dtype=float)
        centered = x - self.mean_vector
        empirical_covariance = np.cov(centered, rowvar=False)
        empirical_covariance = _stabilize_covariance(empirical_covariance, jitter=max(self.covariance_jitter, 1e-12))

        scale = empirical_covariance * ((self.degrees_of_freedom - 2.0) / self.degrees_of_freedom)
        self.scale_matrix = _stabilize_covariance(scale, jitter=max(self.covariance_jitter, 1e-12))
        self.n_assets = int(x.shape[1])

        self._n_train_windows = (
            0 if train_data.forecast_windows is None else int(train_data.forecast_windows.contexts.shape[0])
        )
        self._n_train_paths = 0 if train_data.path_collection is None else len(train_data.path_collection.paths)
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.mean_vector is None or self.scale_matrix is None or self.n_assets is None:
            raise RuntimeError("The model must be fit before sampling.")

        rng = np.random.default_rng(request.seed)
        gaussian_draws = rng.multivariate_normal(
            mean=np.zeros(self.n_assets, dtype=float),
            cov=self.scale_matrix,
            size=(request.n_scenarios, request.horizon),
        )
        chi_square = rng.chisquare(self.degrees_of_freedom, size=(request.n_scenarios, request.horizon, 1))
        scales = np.sqrt(self.degrees_of_freedom / chi_square)
        samples = self.mean_vector + gaussian_draws * scales

        result = ScenarioSamples(samples=np.asarray(samples, dtype=float))
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
            "degrees_of_freedom": self.degrees_of_freedom,
            "covariance_jitter": self.covariance_jitter,
            "use_empirical_mean": self.use_empirical_mean,
            "n_train_windows": self._n_train_windows,
            "n_train_paths": self._n_train_paths,
        }
