"""Model definitions, contracts, factories, and catalog metadata."""

from __future__ import annotations

from importlib import import_module

_EXPORT_TO_MODULE = {
    "DebugSmokeModel": ".builtins",
    "ExternalProcessScenarioModel": ".wrappers",
    "ForecastWindowCollection": ".contracts",
    "HistoricalBootstrapModel": ".builtins",
    "ModelConfig": ".definition",
    "ModelExecutionConfig": ".definition",
    "ModelReferenceConfig": ".definition",
    "PipelineConfig": ".definition",
    "PreprocessedScenarioModel": ".wrappers",
    "Protocol": "..benchmark",
    "RuntimeContext": ".contracts",
    "ScenarioModel": ".contracts",
    "ScenarioRequest": ".contracts",
    "ScenarioSamples": ".contracts",
    "StochasticVolatilityBootstrapModel": ".builtins",
    "TrainPathCollection": ".contracts",
    "TrainingData": ".contracts",
    "build_model": ".factory",
    "build_pipeline": ".factory",
    "import_object": ".resolution",
    "instantiate_model_target": ".resolution",
    "resolve_model_builder": ".resolution",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
