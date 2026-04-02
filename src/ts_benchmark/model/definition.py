"""Model definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

from ..serialization import to_jsonable
from ..utils import JsonObject, StringMap


def _json_object(value: Mapping[str, Any] | JsonObject | None) -> JsonObject:
    return value if isinstance(value, JsonObject) else JsonObject(value)


def _typed_dataclass_payload(value: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    blocked = set() if exclude is None else set(exclude)
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in blocked
    }


class _TypedConfigPayload:
    def to_builtin(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class PipelineStepConfig(_TypedConfigPayload):
    """Generic preprocessing step config for extension-owned or unknown steps."""

    type: str
    params: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.params, JsonObject):
            self.params = JsonObject(self.params)
        self.type = str(self.type or "").strip()


class _TypedPipelineStepConfig(_TypedConfigPayload):
    type: str

    @property
    def params(self) -> JsonObject:
        return JsonObject(to_jsonable(_typed_dataclass_payload(self, exclude={"type"})))

    def to_builtin(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "params": self.params.to_builtin(),
        }


@dataclass(frozen=True)
class IdentityPipelineStepConfig(_TypedPipelineStepConfig):
    type: str = field(default="identity", init=False)


@dataclass(frozen=True)
class DemeanPipelineStepConfig(_TypedPipelineStepConfig):
    type: str = field(default="demean", init=False)


@dataclass(frozen=True)
class StandardScalePipelineStepConfig(_TypedPipelineStepConfig):
    with_mean: bool = True
    with_std: bool = True
    type: str = field(default="standard_scale", init=False)


@dataclass(frozen=True)
class MinMaxScalePipelineStepConfig(_TypedPipelineStepConfig):
    feature_min: float = 0.0
    feature_max: float = 1.0
    clip: bool = False
    type: str = field(default="min_max_scale", init=False)


@dataclass(frozen=True)
class RobustScalePipelineStepConfig(_TypedPipelineStepConfig):
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    type: str = field(default="robust_scale", init=False)


@dataclass(frozen=True)
class ClipPipelineStepConfig(_TypedPipelineStepConfig):
    min_value: float = -0.25
    max_value: float = 0.25
    type: str = field(default="clip", init=False)


@dataclass(frozen=True)
class WinsorizePipelineStepConfig(_TypedPipelineStepConfig):
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    type: str = field(default="winsorize", init=False)


BUILTIN_PIPELINE_STEP_TYPES: dict[str, type[_TypedPipelineStepConfig]] = {
    "identity": IdentityPipelineStepConfig,
    "demean": DemeanPipelineStepConfig,
    "standard_scale": StandardScalePipelineStepConfig,
    "min_max_scale": MinMaxScalePipelineStepConfig,
    "robust_scale": RobustScalePipelineStepConfig,
    "clip": ClipPipelineStepConfig,
    "winsorize": WinsorizePipelineStepConfig,
}


def pipeline_step_from_object(value: PipelineStepConfig | Mapping[str, Any] | Any) -> PipelineStepConfig | _TypedPipelineStepConfig:
    if isinstance(
        value,
        (
            PipelineStepConfig,
            IdentityPipelineStepConfig,
            DemeanPipelineStepConfig,
            StandardScalePipelineStepConfig,
            MinMaxScalePipelineStepConfig,
            RobustScalePipelineStepConfig,
            ClipPipelineStepConfig,
            WinsorizePipelineStepConfig,
        ),
    ):
        return value
    if hasattr(value, "type") and hasattr(value, "params"):
        step_type = str(value.type).strip()
        params = value.params.to_builtin() if hasattr(value.params, "to_builtin") else dict(value.params)
    elif isinstance(value, Mapping):
        step_type = str(value.get("type", "")).strip()
        params = dict(value.get("params", {}) or {})
    else:
        raise TypeError("Pipeline steps must be provided as objects or mappings.")

    if step_type in BUILTIN_PIPELINE_STEP_TYPES:
        return BUILTIN_PIPELINE_STEP_TYPES[step_type](**params)
    return PipelineStepConfig(type=step_type, params=JsonObject(params))


def pipeline_step_payload(step: PipelineStepConfig | _TypedPipelineStepConfig) -> dict[str, Any]:
    return step.to_builtin() if hasattr(step, "to_builtin") else {
        "type": str(step.type),
        "params": step.params.to_builtin() if hasattr(step.params, "to_builtin") else dict(step.params),
    }


@dataclass
class PipelineConfig:
    name: str
    steps: list[
        PipelineStepConfig
        | IdentityPipelineStepConfig
        | DemeanPipelineStepConfig
        | StandardScalePipelineStepConfig
        | MinMaxScalePipelineStepConfig
        | RobustScalePipelineStepConfig
        | ClipPipelineStepConfig
        | WinsorizePipelineStepConfig
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "raw")
        self.steps = [pipeline_step_from_object(step) for step in list(self.steps or [])]

    def to_builtin(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "steps": [pipeline_step_payload(step) for step in self.steps],
        }


@dataclass
class ModelReferenceConfig:
    kind: str
    value: str


@dataclass
class ModelExecutionConfig:
    mode: str = "inprocess"
    venv: str | None = None
    python: str | None = None
    cwd: str | None = None
    pythonpath: list[str] = field(default_factory=list)
    env: StringMap = field(default_factory=StringMap)

    def __post_init__(self) -> None:
        if not isinstance(self.env, StringMap):
            self.env = StringMap(self.env)


@dataclass(frozen=True)
class HistoricalBootstrapParams(_TypedConfigPayload):
    block_size: int = 5


@dataclass(frozen=True)
class GaussianCovarianceParams(_TypedConfigPayload):
    covariance_jitter: float = 1e-6
    use_empirical_mean: bool = True


@dataclass(frozen=True)
class StochasticVolatilityBootstrapParams(_TypedConfigPayload):
    ewma_lambda: float = 0.97
    block_size: int = 5
    vol_of_vol: float = 0.10
    long_run_blend: float = 0.02
    min_vol: float = 1e-4


@dataclass(frozen=True)
class DebugSmokeModelParams(_TypedConfigPayload):
    scale: float = 1.0


BUILTIN_MODEL_PARAM_TYPES: dict[tuple[str, str], type[_TypedConfigPayload]] = {
    ("builtin", "gaussian_covariance"): GaussianCovarianceParams,
    ("builtin", "historical_bootstrap"): HistoricalBootstrapParams,
    ("builtin", "stochastic_volatility_bootstrap"): StochasticVolatilityBootstrapParams,
    ("entrypoint", "ts_benchmark.model.builtins.gaussian_covariance:GaussianCovarianceModel"): GaussianCovarianceParams,
    ("entrypoint", "ts_benchmark.model.builtins.historical_bootstrap:HistoricalBootstrapModel"): HistoricalBootstrapParams,
    ("entrypoint", "ts_benchmark.model.builtins.stochastic_vol_bootstrap:StochasticVolatilityBootstrapModel"): StochasticVolatilityBootstrapParams,
    ("entrypoint", "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel"): DebugSmokeModelParams,
}


ModelParamValue = (
    GaussianCovarianceParams
    | HistoricalBootstrapParams
    | StochasticVolatilityBootstrapParams
    | DebugSmokeModelParams
    | JsonObject
)


def builtin_model_param_type(reference: ModelReferenceConfig) -> type[_TypedConfigPayload] | None:
    return BUILTIN_MODEL_PARAM_TYPES.get((str(reference.kind), str(reference.value)))


def model_params_from_object(
    reference: ModelReferenceConfig,
    value: ModelParamValue | Mapping[str, Any] | None,
) -> ModelParamValue:
    expected = builtin_model_param_type(reference)
    if expected is not None:
        if isinstance(value, expected):
            return value
        payload = {} if value is None else (
            value.to_builtin() if hasattr(value, "to_builtin") and not isinstance(value, JsonObject) else dict(value)
        )
        return expected(**payload)
    if isinstance(value, JsonObject):
        return value
    if value is None:
        return JsonObject()
    if is_dataclass(value) and not isinstance(value, type):
        return JsonObject(to_jsonable(value))
    return JsonObject(value)


def model_params_to_builtin(value: ModelParamValue | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(value, "to_builtin"):
        return value.to_builtin()
    if is_dataclass(value) and not isinstance(value, type):
        return dict(to_jsonable(value))
    return dict(value)


@dataclass
class ModelConfig:
    name: str
    reference: ModelReferenceConfig
    params: ModelParamValue = field(default_factory=JsonObject)
    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig(name="raw", steps=[]))
    execution: ModelExecutionConfig | None = None
    description: str | None = None
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ModelReferenceConfig):
            self.reference = ModelReferenceConfig(**self.reference)
        self.params = model_params_from_object(self.reference, self.params)
        if not isinstance(self.pipeline, PipelineConfig):
            pipeline = dict(self.pipeline or {})
            self.pipeline = PipelineConfig(
                name=str(pipeline.get("name", "raw")),
                steps=list(pipeline.get("steps") or []),
            )
        if self.execution is not None and not isinstance(self.execution, ModelExecutionConfig):
            self.execution = ModelExecutionConfig(**self.execution)
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)
