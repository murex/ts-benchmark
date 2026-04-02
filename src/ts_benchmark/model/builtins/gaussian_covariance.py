"""Static multivariate Gaussian return baseline."""

from __future__ import annotations

import numpy as np

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


def _stabilize_covariance(covariance: np.ndarray, *, jitter: float) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be square.")
    if cov.shape[0] == 0:
        raise ValueError("covariance matrix must be non-empty.")

    eigvals, eigvecs = np.linalg.eigh(cov)
    floor = float(jitter)
    eigvals = np.maximum(eigvals, floor)
    stabilized = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (stabilized + stabilized.T)


class GaussianCovarianceModel(ScenarioModel):
    """Fit a static Gaussian distribution to historical returns."""

    name = "gaussian_covariance"

    def __init__(
        self,
        covariance_jitter: float = 1e-6,
        use_empirical_mean: bool = True,
    ):
        if covariance_jitter < 0.0:
            raise ValueError("covariance_jitter must be non-negative.")
        self.covariance_jitter = float(covariance_jitter)
        self.use_empirical_mean = bool(use_empirical_mean)
        self.mean_vector: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.n_assets: int | None = None
        self._n_train_windows: int = 0
        self._n_train_paths: int = 0

    def fit(self, train_data: TrainingData) -> "GaussianCovarianceModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        if x.ndim != 2:
            raise ValueError("training values must be shaped [time, n_assets].")
        if len(x) < 2:
            raise ValueError("at least two training observations are required.")

        self.mean_vector = x.mean(axis=0) if self.use_empirical_mean else np.zeros(x.shape[1], dtype=float)
        centered = x - self.mean_vector
        covariance = np.cov(centered, rowvar=False)
        self.covariance = _stabilize_covariance(covariance, jitter=max(self.covariance_jitter, 1e-12))
        self.n_assets = int(x.shape[1])

        self._n_train_windows = (
            0 if train_data.forecast_windows is None else int(train_data.forecast_windows.contexts.shape[0])
        )
        self._n_train_paths = 0 if train_data.path_collection is None else len(train_data.path_collection.paths)
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.mean_vector is None or self.covariance is None or self.n_assets is None:
            raise RuntimeError("The model must be fit before sampling.")

        rng = np.random.default_rng(request.seed)
        samples = rng.multivariate_normal(
            mean=self.mean_vector,
            cov=self.covariance,
            size=(request.n_scenarios, request.horizon),
        )
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
            "covariance_jitter": self.covariance_jitter,
            "use_empirical_mean": self.use_empirical_mean,
            "n_train_windows": self._n_train_windows,
            "n_train_paths": self._n_train_paths,
        }
