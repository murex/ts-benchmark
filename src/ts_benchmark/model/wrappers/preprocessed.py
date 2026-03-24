"""Wrapper that makes preprocessing explicit in the benchmark/model contract."""

from __future__ import annotations

from typing import Any

from ...preprocessing import PreprocessingPipeline
from ...utils import JsonObject
from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData


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
        self.pipeline.fit(train_data.returns)
        transformed = TrainingData(
            returns=self.pipeline.transform(train_data.returns),
            protocol=train_data.protocol,
            asset_names=train_data.asset_names,
            freq=train_data.freq,
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
