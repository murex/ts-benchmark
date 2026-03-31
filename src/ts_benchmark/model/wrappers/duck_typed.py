"""Generic bridge for structurally compatible external time-series generators."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ...serialization import to_jsonable
from ...utils import JsonObject
from ..contracts import (
    RuntimeContext as BenchmarkRuntimeContext,
    ScenarioModel,
    ScenarioRequest,
    ScenarioSamples,
    TrainingData,
)
from ..model_contract import (
    DataSchema as ExternalSchema,
    GenerationMode as ExternalMode,
    GenerationRequest as ExternalRequest,
    RuntimeContext as ExternalRuntime,
    TrainData as ExternalTrainData,
    TrainExample as ExternalTrainExample,
    TSSeries as ExternalSeries,
    TaskSpec as ExternalTask,
)


def _looks_like_structured_estimator(candidate: object) -> bool:
    fit = getattr(candidate, "fit", None)
    if not callable(fit):
        return False
    try:
        signature = inspect.signature(fit)
    except (TypeError, ValueError):
        return False
    return "schema" in signature.parameters and "task" in signature.parameters


def _mode_value(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _capabilities_to_builtin(capabilities: object | None) -> dict[str, object] | None:
    if capabilities is None:
        return None
    supported = getattr(capabilities, "supported_modes", None)
    if supported is None:
        return None
    return {
        "supported_modes": sorted(_mode_value(mode) for mode in supported),
        "supports_multivariate_targets": bool(getattr(capabilities, "supports_multivariate_targets", False)),
        "supports_known_covariates": bool(getattr(capabilities, "supports_known_covariates", False)),
        "supports_observed_covariates": bool(getattr(capabilities, "supports_observed_covariates", False)),
        "supports_static_covariates": bool(getattr(capabilities, "supports_static_covariates", False)),
        "supports_constraints": bool(getattr(capabilities, "supports_constraints", False)),
    }


def _fit_report_to_builtin(report: object | None) -> dict[str, object] | None:
    if report is None:
        return None
    payload = {
        "train_metrics": getattr(report, "train_metrics", None),
        "val_metrics": getattr(report, "val_metrics", None),
        "fit_time_sec": getattr(report, "fit_time_sec", None),
        "peak_memory_mb": getattr(report, "peak_memory_mb", None),
        "n_parameters": getattr(report, "n_parameters", None),
        "diagnostics": getattr(report, "diagnostics", None),
    }
    return {key: to_jsonable(value) for key, value in payload.items() if value is not None}


def _runtime_hints(runtime: BenchmarkRuntimeContext | None, *, seed_override: int | None = None) -> ExternalRuntime:
    return ExternalRuntime(
        device=None if runtime is None else runtime.device,
        seed=seed_override if seed_override is not None else (None if runtime is None else runtime.seed),
    )


def _training_payload(
    train_data: TrainingData,
) -> ExternalTrainData:
    if str(train_data.protocol.generation_mode) == "forecast":
        if train_data.forecast_windows is None:
            raise ValueError(
                "Forecast training data must include forecast_windows before bridging to the structural contract."
            )
        return ExternalTrainData(
            examples=[
                ExternalTrainExample(
                    history=ExternalSeries(values=np.asarray(history, dtype=float)),
                    context=ExternalSeries(values=np.asarray(context, dtype=float)),
                    target=ExternalSeries(values=np.asarray(target, dtype=float)),
                )
                for history, context, target in zip(
                    train_data.forecast_windows.histories,
                    train_data.forecast_windows.contexts,
                    train_data.forecast_windows.targets,
                    strict=True,
                )
            ],
        )
    if train_data.path_collection is None:
        raise ValueError(
            "Unconditional training data must include path_collection before bridging to the structural contract."
        )
    return ExternalTrainData(
        examples=[
            ExternalTrainExample(
                history=None,
                context=None,
                target=ExternalSeries(values=np.asarray(path, dtype=float)),
            )
            for path in train_data.path_collection.paths
        ],
    )


class DuckTypedGeneratorScenarioModel(ScenarioModel):
    """Wrap an external estimator-like object in the benchmark's ScenarioModel contract."""

    def __init__(self, *, estimator: object, name: str):
        self.estimator = estimator
        self.name = name
        self._engine: object | None = None
        self._fit_report: object | None = None

    def fit(self, train_data: TrainingData) -> "DuckTypedGeneratorScenarioModel":
        train_data.validate()
        schema = ExternalSchema(
            target_dim=train_data.n_assets,
            freq=train_data.freq,
            known_covariates=None,
            observed_covariates=None,
            static_covariates=None,
        )
        task = ExternalTask(
            mode=ExternalMode(str(train_data.protocol.generation_mode)),
            horizon=int(train_data.protocol.horizon),
        )
        engine, fit_report = self.estimator.fit(
            _training_payload(train_data),
            schema=schema,
            task=task,
            valid=None,
            runtime=_runtime_hints(train_data.runtime),
        )
        self._engine = engine
        self._fit_report = fit_report
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self._engine is None:
            raise RuntimeError("Model must be fit before sampling.")

        series = ExternalSeries(values=np.asarray(request.context, dtype=float))
        task = ExternalTask(
            mode=ExternalMode(str(request.mode)),
            horizon=int(request.horizon),
        )
        external_request = ExternalRequest(
            series=series,
            task=task,
            num_samples=int(request.n_scenarios),
            constraints=None,
            runtime=_runtime_hints(request.runtime, seed_override=request.seed),
        )
        result = self._engine.sample(external_request)
        samples = np.asarray(getattr(result, "samples"), dtype=float)
        if samples.ndim != 3:
            raise ValueError(
                "External generator returned samples with unsupported shape; expected "
                "[num_samples, horizon, target_dim]."
            )

        diagnostics = getattr(result, "diagnostics", None)
        payload = {
            "adapter": "duck_typed_estimator",
            "wrapped_model": self.name,
            "diagnostics": None if diagnostics is None else to_jsonable(diagnostics),
        }
        out = ScenarioSamples(samples=samples, metadata=JsonObject(payload))
        out.validate(expected_horizon=request.horizon, expected_n_assets=request.n_assets)
        return out

    def model_info(self) -> dict[str, Any]:
        capabilities = None
        if self._engine is not None and hasattr(self._engine, "capabilities"):
            try:
                capabilities = self._engine.capabilities()
            except Exception:
                capabilities = None
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "adapter": "duck_typed_estimator",
            "wrapped_estimator_class": self.estimator.__class__.__name__,
            "wrapped_generator_class": None if self._engine is None else self._engine.__class__.__name__,
            "capabilities": _capabilities_to_builtin(capabilities),
            "fit_report": _fit_report_to_builtin(self._fit_report),
        }

    def close(self) -> None:
        for target in (self._engine, self.estimator):
            closer = getattr(target, "close", None)
            if callable(closer):
                closer()


def coerce_model_target(target: object, *, name: str) -> ScenarioModel:
    """Coerce a loaded model target into the benchmark's ScenarioModel contract."""

    if isinstance(target, ScenarioModel):
        return target
    if _looks_like_structured_estimator(target):
        return DuckTypedGeneratorScenarioModel(estimator=target, name=name)
    raise TypeError(
        "Resolved model target is not benchmark-compatible. Expected either a "
        "ScenarioModel instance or an estimator-like object with a fit(..., schema=..., task=...) method."
    )
