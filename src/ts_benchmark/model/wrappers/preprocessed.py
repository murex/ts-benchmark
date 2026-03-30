"""Wrapper that makes preprocessing explicit in the benchmark/model contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...preprocessing import PreprocessingPipeline
from ...utils import JsonObject
from ..contracts import (
    ForecastWindowCollection,
    ScenarioModel,
    ScenarioRequest,
    ScenarioSamples,
    TrainPathCollection,
    TrainingData,
)


class PreprocessedScenarioModel(ScenarioModel):
    """Apply a preprocessing pipeline around any benchmark-compatible model."""

    def __init__(
        self,
        model: ScenarioModel,
        pipeline: PreprocessingPipeline,
        name: str | None = None,
    ):
        self.model = model
        self.pipeline = pipeline
        self.name = name or getattr(model, "name", model.__class__.__name__)

    def fit(self, train_data: TrainingData) -> "PreprocessedScenarioModel":
        train_data.validate()
        fit_values = train_data.concatenated_training_values()
        self.pipeline.fit(fit_values)
        transformed_forecast_windows = None
        if train_data.forecast_windows is not None:
            transformed_forecast_windows = ForecastWindowCollection(
                contexts=np.asarray(
                    [self.pipeline.transform(context) for context in train_data.forecast_windows.contexts],
                    dtype=float,
                ),
                targets=np.asarray(
                    [self.pipeline.transform(target) for target in train_data.forecast_windows.targets],
                    dtype=float,
                ),
                source_kind=train_data.forecast_windows.source_kind,
                stride=train_data.forecast_windows.stride,
            )
        transformed_path_collection = None
        if train_data.path_collection is not None:
            transformed_path_collection = TrainPathCollection(
                paths=[self.pipeline.transform(path) for path in train_data.path_collection.paths],
                source_kind=train_data.path_collection.source_kind,
                window_length=train_data.path_collection.window_length,
                stride=train_data.path_collection.stride,
            )
        transformed = TrainingData(
            returns=self.pipeline.transform(train_data.returns),
            protocol=train_data.protocol,
            asset_names=train_data.asset_names,
            freq=train_data.freq,
            forecast_windows=transformed_forecast_windows,
            path_collection=transformed_path_collection,
            runtime=train_data.runtime,
            metadata=JsonObject(
                {
                    **train_data.metadata.to_builtin(),
                    "preprocessing": self.pipeline.summary(),
                }
            ),
        )
        self.model.fit(transformed)
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        transformed_request = ScenarioRequest(
            context=self.pipeline.transform(request.context),
            horizon=request.horizon,
            n_scenarios=request.n_scenarios,
            protocol=request.protocol,
            mode=request.mode,
            seed=request.seed,
            asset_names=request.asset_names,
            freq=request.freq,
            runtime=request.runtime,
            metadata=JsonObject(
                {
                    **request.metadata.to_builtin(),
                    "preprocessing": self.pipeline.summary(),
                }
            ),
        )
        transformed_samples = self.model.sample(transformed_request)
        samples = self.pipeline.inverse_transform(transformed_samples.samples)
        result = ScenarioSamples(
            samples=samples,
            metadata=JsonObject(
                {
                    **transformed_samples.metadata.to_builtin(),
                    "preprocessing": self.pipeline.summary(),
                    "wrapped_model": self.name,
                }
            ),
        )
        result.validate(
            expected_horizon=request.horizon,
            expected_n_assets=request.n_assets,
        )
        return result

    def model_info(self) -> dict[str, Any]:
        info = self.model.model_info()
        return {
            **dict(info),
            "name": self.name,
            "preprocessing": self.pipeline.summary(),
        }

    def debug_artifacts(self) -> dict[str, Any] | None:
        wrapped_debug = getattr(self.model, "debug_artifacts", None)
        payload = None
        if callable(wrapped_debug):
            payload = wrapped_debug()
        if payload is None:
            return None
        return {
            "name": self.name,
            "preprocessing": self.pipeline.summary(),
            "wrapped_debug_artifacts": payload,
        }
