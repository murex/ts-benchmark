"""Model construction and model-side metadata helpers for benchmark runs."""

from __future__ import annotations

from typing import Any

from ..benchmark.definition import BenchmarkConfig
from ..model.catalog.plugins import ModelPluginManifest, PluginInfo, extract_model_plugin_manifest, get_model_plugin_info
from ..model.factory import build_model, build_pipeline
from ..model.wrappers.preprocessed import PreprocessedScenarioModel
from ..results import ModelExecutionRecord, ModelResult
from ..utils import JsonObject

def resolve_declared_manifest(
    model_config,
    model_object: object,
) -> PluginInfo | None:
    if model_config.reference.kind in {"builtin", "plugin"}:
        plugin_key = model_config.reference.value
        try:
            return get_model_plugin_info(plugin_key)
        except KeyError:
            pass

    manifest = extract_model_plugin_manifest(model_object, default_name=model_config.name)
    if manifest is None:
        return None
    return PluginInfo(
        name=model_config.name,
        source="object",
        target=model_config.reference.value,
        manifest=manifest,
    )


def enforce_manifest_preprocessing_contract(
    *,
    model_name: str,
    configured_pipeline: str,
    manifest: PluginInfo | ModelPluginManifest | None,
) -> None:
    if not manifest:
        return
    manifest_payload = manifest.manifest if isinstance(manifest, PluginInfo) else manifest
    if manifest_payload is None:
        return
    required_pipeline = manifest_payload.required_pipeline
    if not required_pipeline:
        return
    if configured_pipeline != required_pipeline:
        raise ValueError(
            f"Model '{model_name}' requires preprocessing pipeline '{required_pipeline}' "
            f"according to its manifest, but the config uses pipeline '{configured_pipeline}'."
        )


def build_models(
    config: BenchmarkConfig,
    *,
    runtime_device: str | None,
) -> tuple[dict[str, object], dict[str, ModelResult]]:
    models: dict[str, object] = {}
    model_results: dict[str, ModelResult] = {}

    for model_config in config.models:
        model = build_model(
            model_config,
            default_device=runtime_device,
            default_seed=config.run.seed,
            source_path=config.source_path,
        )
        manifest = resolve_declared_manifest(model_config, model)
        enforce_manifest_preprocessing_contract(
            model_name=model_config.name,
            configured_pipeline=model_config.pipeline.name,
            manifest=manifest,
        )
        pipeline = build_pipeline(model_config.pipeline)
        wrapped = PreprocessedScenarioModel(model=model, pipeline=pipeline, name=model_config.name)
        models[model_config.name] = wrapped
        model_results[model_config.name] = ModelResult(
            model_name=model_config.name,
            description=model_config.description,
            reference=model_config.reference,
            params=model_config.params,
            execution=ModelExecutionRecord(
                requested=model_config.execution,
                assigned_device=runtime_device,
                runtime_device=runtime_device,
            ),
            pipeline=model_config.pipeline,
            metadata=model_config.metadata,
            plugin_info=manifest,
        )
    return models, model_results


def collect_model_debug_artifacts(models: dict[str, object]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, wrapped in models.items():
        getter = getattr(wrapped, "debug_artifacts", None)
        if callable(getter):
            artifacts = getter()
            if artifacts is not None:
                payload[name] = artifacts
    return payload


def close_models(models: dict[str, object]) -> None:
    for wrapped in models.values():
        base_model = getattr(wrapped, "model", wrapped)
        closer = getattr(base_model, "close", None)
        if callable(closer):
            closer()
