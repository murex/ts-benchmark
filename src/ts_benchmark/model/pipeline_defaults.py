"""Helpers for materializing default preprocessing pipelines."""

from __future__ import annotations

from .definition import (
    MinMaxScalePipelineStepConfig,
    ModelReferenceConfig,
    PipelineConfig,
    StandardScalePipelineStepConfig,
)


def pipeline_config_from_name(name: str | None) -> PipelineConfig:
    text = str(name or "raw").strip()
    normalized = text.lower()
    if normalized in {"", "raw"}:
        return PipelineConfig(name="raw", steps=[])
    if normalized == "standardized":
        return PipelineConfig(
            name="standardized",
            steps=[StandardScalePipelineStepConfig()],
        )
    if normalized == "minmax":
        return PipelineConfig(
            name="minmax",
            steps=[MinMaxScalePipelineStepConfig()],
        )
    return PipelineConfig(name=text, steps=[])


def default_pipeline_name_from_manifest(
    manifest,
) -> str | None:
    if manifest is None:
        return None
    manifest_payload = getattr(manifest, "manifest", manifest)
    if manifest_payload is None:
        return None
    default_pipeline = str(manifest_payload.default_pipeline or "").strip()
    return default_pipeline or None


def resolve_default_pipeline_config(
    reference: ModelReferenceConfig,
    *,
    default_name: str | None = None,
) -> PipelineConfig:
    manifest = None
    try:
        from .catalog.plugins import extract_model_plugin_manifest, get_model_plugin_info
        from .resolution import import_object

        if str(reference.kind) in {"builtin", "plugin"}:
            manifest = get_model_plugin_info(str(reference.value))
        elif str(reference.kind) == "entrypoint":
            builder = import_object(str(reference.value))
            manifest = extract_model_plugin_manifest(builder, default_name=default_name)
    except Exception:
        manifest = None
    return pipeline_config_from_name(default_pipeline_name_from_manifest(manifest))
