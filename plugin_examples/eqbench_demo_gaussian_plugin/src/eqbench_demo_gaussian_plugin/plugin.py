from __future__ import annotations

import numpy as np

from ts_benchmark.model import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData
from ts_benchmark.model.catalog import (
    ModelParameterSchema,
    ModelParameterSpec,
    ModelPluginManifest,
    PluginCapabilities,
)


class GaussianPluginModel(ScenarioModel):
    name = "demo_gaussian_plugin"

    PLUGIN_MANIFEST = ModelPluginManifest(
        name="demo_gaussian_plugin",
        display_name="Demo Gaussian plugin",
        version="0.1.0",
        family="gaussian",
        description="Minimal external plugin example that samples Gaussian future returns.",
        runtime_device_hints=("cpu",),
        supported_dataset_sources=("synthetic", "csv", "parquet"),
        required_input="returns",
        default_pipeline="raw",
        required_pipeline="raw",
        tags=("example", "plugin", "gaussian"),
        notes="Intended as a developer template for external benchmark integrations.",
        capabilities=PluginCapabilities(
            multivariate=True,
            probabilistic_sampling=True,
            benchmark_protocol_contract=True,
            explicit_preprocessing=True,
            uses_benchmark_device=False,
        ),
        manifest_source="builder_attribute",
    )
    PARAMETER_SCHEMA = ModelParameterSchema(
        name="demo_gaussian_plugin",
        fields=(
            ModelParameterSpec(
                name="ridge",
                value_type="float",
                required=False,
                default=1e-6,
                annotation="float",
                description="Diagonal ridge added to the empirical covariance estimate.",
                schema_source="builder_attribute",
            ),
        ),
        schema_source="builder_attribute",
    )

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


def build_model(**params):
    return GaussianPluginModel(**params)


def get_plugin_manifest():
    return GaussianPluginModel.PLUGIN_MANIFEST


def get_parameter_schema():
    return GaussianPluginModel.PARAMETER_SCHEMA


build_model.PARAMETER_SCHEMA = GaussianPluginModel.PARAMETER_SCHEMA
