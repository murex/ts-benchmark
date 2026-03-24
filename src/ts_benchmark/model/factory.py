"""Model and preprocessing construction helpers."""

from __future__ import annotations

from pathlib import Path

from ..preprocessing import PreprocessingPipeline, build_pipeline_from_config
from .definition import ModelConfig, PipelineConfig
from .resolution import instantiate_model_target
from .wrappers.duck_typed import coerce_model_target
from .wrappers.external_process import ExternalProcessScenarioModel


def build_pipeline(config: PipelineConfig) -> PreprocessingPipeline:
    return build_pipeline_from_config(config.name, config.steps)


def build_model(
    model_config: ModelConfig,
    *,
    default_device: str | None = None,
    default_seed: int | None = None,
    source_path: Path | None = None,
) -> object:
    execution = model_config.execution
    if execution is not None and execution.mode == "subprocess":
        return ExternalProcessScenarioModel(
            name=model_config.name,
            reference=model_config.reference,
            params=model_config.params.to_builtin(),
            execution=execution,
            source_path=source_path,
        )

    target = instantiate_model_target(
        reference=model_config.reference,
        params=model_config.params.to_builtin(),
    )
    return coerce_model_target(target, name=model_config.name)
