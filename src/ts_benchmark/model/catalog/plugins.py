"""Plugin discovery helpers and plugin manifest/capabilities contracts.

The recommended way to benchmark a model under active development is to
keep it in its own package/repository and expose a factory through the
`ts_benchmark.models` entry-point group.

To make plugin discovery useful in the CLI and UI, plugins can also
publish a lightweight manifest through the
`ts_benchmark.model_manifests` entry-point group.
"""

from __future__ import annotations

import inspect
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from importlib import metadata
from typing import Any, Callable, Iterable, Mapping, get_args, get_origin

from ..builtins.historical_bootstrap import HistoricalBootstrapModel
from ..builtins.stochastic_vol_bootstrap import StochasticVolatilityBootstrapModel
from ...serialization import to_jsonable

MODEL_ENTRYPOINT_GROUP = "ts_benchmark.models"
MODEL_MANIFEST_ENTRYPOINT_GROUP = "ts_benchmark.model_manifests"


@dataclass(frozen=True)
class PluginCapabilities:
    """Capability flags surfaced by the benchmark UI and metadata outputs."""

    multivariate: bool | None = None
    probabilistic_sampling: bool | None = None
    benchmark_protocol_contract: bool | None = None
    explicit_preprocessing: bool | None = None
    uses_benchmark_device: bool | None = None



@dataclass(frozen=True)
class ModelPluginManifest:
    """Descriptive metadata about a model plugin.

    This is intentionally lightweight. The benchmark uses it for:
    - plugin listings in the CLI/UI,
    - saved run metadata,
    - onboarding guidance for model developers.

    The manifest is descriptive rather than a hard compatibility gate; it
    is meant to help users understand what a plugin supports.
    """

    name: str
    display_name: str | None = None
    version: str | None = None
    family: str | None = None
    description: str | None = None
    runtime_device_hints: tuple[str, ...] = field(default_factory=tuple)
    supported_dataset_sources: tuple[str, ...] = field(default_factory=tuple)
    required_input: str | None = None
    default_pipeline: str | None = None
    required_pipeline: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None
    capabilities: PluginCapabilities = field(default_factory=PluginCapabilities)
    manifest_source: str = "default"


@dataclass(frozen=True)
class ModelParameterSpec:
    name: str
    parameter_type: str = "explicit"
    value_type: str = "json"
    required: bool = False
    default: Any = None
    annotation: str | None = None
    description: str | None = None
    choices: tuple[Any, ...] = field(default_factory=tuple)
    editable: bool = True
    schema_source: str = "default"


@dataclass(frozen=True)
class ModelParameterSchema:
    name: str
    fields: tuple[ModelParameterSpec, ...] = field(default_factory=tuple)
    notes: str | None = None
    schema_source: str = "default"


@dataclass(frozen=True)
class PluginInfo:
    name: str
    source: str
    target: str
    package: str | None = None
    package_version: str | None = None
    manifest: ModelPluginManifest | None = None


BUILTIN_MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "historical_bootstrap": HistoricalBootstrapModel,
    "stochastic_volatility_bootstrap": StochasticVolatilityBootstrapModel,
}


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


_BUILTIN_PACKAGE_VERSION = _package_version("ts-benchmark") or "local"

BUILTIN_MODEL_MANIFESTS: dict[str, ModelPluginManifest] = {
    "historical_bootstrap": ModelPluginManifest(
        name="historical_bootstrap",
        display_name="Historical bootstrap",
        version=_BUILTIN_PACKAGE_VERSION,
        family="bootstrap",
        description="Simple historical block bootstrap over multivariate return vectors.",
        runtime_device_hints=("cpu", "cuda", "mps"),
        supported_dataset_sources=("synthetic", "csv", "parquet"),
        required_input="returns",
        default_pipeline="raw",
        tags=("baseline", "historical", "bootstrap"),
        notes="Resamples empirical return vectors and preserves same-date cross-sectional dependence.",
        capabilities=PluginCapabilities(
            multivariate=True,
            probabilistic_sampling=True,
            benchmark_protocol_contract=True,
            explicit_preprocessing=True,
            uses_benchmark_device=True,
        ),
        manifest_source="builtin",
    ),
    "stochastic_volatility_bootstrap": ModelPluginManifest(
        name="stochastic_volatility_bootstrap",
        display_name="Historical SV bootstrap",
        version=_BUILTIN_PACKAGE_VERSION,
        family="bootstrap",
        description="Historical residual bootstrap with recursively simulated stochastic volatility.",
        runtime_device_hints=("cpu", "cuda", "mps"),
        supported_dataset_sources=("synthetic", "csv", "parquet"),
        required_input="returns",
        default_pipeline="raw",
        tags=("baseline", "historical", "bootstrap", "stochastic-volatility"),
        notes="Uses EWMA volatility scaling and bootstrapped standardized residual vectors.",
        capabilities=PluginCapabilities(
            multivariate=True,
            probabilistic_sampling=True,
            benchmark_protocol_contract=True,
            explicit_preprocessing=True,
            uses_benchmark_device=True,
        ),
        manifest_source="builtin",
    ),
}


def _normalize_string_seq(value: Iterable[Any] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return () if not text else (text,)
    out = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _annotation_label(annotation: Any) -> str | None:
    if annotation is inspect.Signature.empty or annotation is None:
        return None
    if isinstance(annotation, str):
        text = annotation.strip()
        return text or None
    text = getattr(annotation, "__name__", None)
    if text:
        module = getattr(annotation, "__module__", "")
        return text if module == "builtins" else f"{module}.{text}"
    rendered = str(annotation)
    if rendered.startswith("typing."):
        rendered = rendered[len("typing.") :]
    return rendered or None


def _normalize_parameter_choices(value: Iterable[Any] | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return () if not text else (text,)
    return tuple(to_jsonable(item) for item in value)


def _infer_value_type(annotation: Any, default: Any) -> tuple[str, bool]:
    if isinstance(annotation, str):
        text = annotation.strip().lower()
        if "callable" in text:
            return "callable", False
        if "dict" in text or "mapping" in text:
            return "dict", True
        if "list" in text or "tuple" in text or "set" in text:
            return "list", True
        if "bool" in text:
            return "bool", True
        if "int" in text:
            return "int", True
        if "float" in text:
            return "float", True
        if "str" in text or "string" in text:
            return "string", True

    target = annotation
    origin = get_origin(annotation)
    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if origin is not None:
        target = origin
        if origin in {list, tuple, set, frozenset}:
            return "list", True
        if origin in {dict, Mapping}:
            return "dict", True
        if origin in {Callable} or str(origin) in {"collections.abc.Callable", "typing.Callable"}:
            return "callable", False
        if origin is type and args:
            target = args[0]
    elif args:
        target = args[0]

    if target is bool:
        return "bool", True
    if target is int:
        return "int", True
    if target is float:
        return "float", True
    if target is str:
        return "string", True

    if default is not inspect.Signature.empty and default is not None:
        if isinstance(default, bool):
            return "bool", True
        if isinstance(default, int) and not isinstance(default, bool):
            return "int", True
        if isinstance(default, float):
            return "float", True
        if isinstance(default, str):
            return "string", True
        if isinstance(default, (list, tuple, set, frozenset)):
            return "list", True
        if isinstance(default, dict):
            return "dict", True

    return "json", True


def normalize_model_parameter_spec(
    value: ModelParameterSpec | Mapping[str, Any],
    *,
    default_source: str = "default",
) -> ModelParameterSpec:
    if isinstance(value, ModelParameterSpec):
        spec = value
    elif isinstance(value, Mapping):
        payload = dict(value)
        default = payload.get("default")
        value_type = str(payload.get("value_type") or "").strip().lower()
        if not value_type:
            value_type, editable = _infer_value_type(payload.get("annotation"), default)
        else:
            editable = bool(payload.get("editable", value_type != "callable"))
        spec = ModelParameterSpec(
            name=str(payload.get("name") or "").strip(),
            parameter_type=str(payload.get("parameter_type") or "explicit").strip().lower(),
            value_type=value_type,
            required=bool(payload.get("required", False)),
            default=to_jsonable(default),
            annotation=_annotation_label(payload.get("annotation")),
            description=payload.get("description"),
            choices=_normalize_parameter_choices(payload.get("choices")),
            editable=editable,
            schema_source=str(payload.get("schema_source") or default_source),
        )
    else:
        raise TypeError("Model parameter specs must be ModelParameterSpec instances or mappings.")

    if not spec.name:
        raise ValueError("Model parameter specs require a non-empty name.")
    if not spec.parameter_type:
        spec = replace(spec, parameter_type="explicit")
    if not spec.value_type:
        inferred_value_type, inferred_editable = _infer_value_type(spec.annotation, spec.default)
        spec = replace(spec, value_type=inferred_value_type, editable=inferred_editable)
    if not spec.schema_source:
        spec = replace(spec, schema_source=default_source)
    return spec


def normalize_model_parameter_schema(
    value: ModelParameterSchema | Mapping[str, Any] | Iterable[Any] | None,
    *,
    name: str,
    default_source: str = "default",
) -> ModelParameterSchema:
    if isinstance(value, ModelParameterSchema):
        schema = value
    elif isinstance(value, Mapping):
        payload = dict(value)
        fields_value = payload.get("fields")
        if fields_value is None:
            fields_value = payload.get("parameters")
        if fields_value is None:
            fields_value = []
        schema = ModelParameterSchema(
            name=str(payload.get("name") or name),
            fields=tuple(
                normalize_model_parameter_spec(field_value, default_source=str(payload.get("schema_source") or default_source))
                for field_value in fields_value
            ),
            notes=payload.get("notes"),
            schema_source=str(payload.get("schema_source") or default_source),
        )
    elif value is None:
        schema = ModelParameterSchema(name=name, schema_source=default_source)
    else:
        schema = ModelParameterSchema(
            name=name,
            fields=tuple(
                normalize_model_parameter_spec(field_value, default_source=default_source)
                for field_value in value
            ),
            schema_source=default_source,
        )

    if not schema.name:
        schema = replace(schema, name=name)
    if not schema.schema_source:
        schema = replace(schema, schema_source=default_source)
    return schema


def _dataclass_field_default(field_info: Any) -> tuple[Any, bool]:
    if field_info.default is not MISSING:
        return field_info.default, True
    if field_info.default_factory is not MISSING:
        try:
            return field_info.default_factory(), True
        except Exception:
            return None, False
    return None, False


def _parameter_schema_from_dataclass_cls(
    config_cls: type[Any],
    *,
    name: str,
    default_source: str,
) -> ModelParameterSchema | None:
    if not inspect.isclass(config_cls) or not is_dataclass(config_cls):
        return None
    specs: list[ModelParameterSpec] = []
    for field_info in fields(config_cls):
        default, has_default = _dataclass_field_default(field_info)
        value_type, editable = _infer_value_type(field_info.type, default if has_default else inspect.Signature.empty)
        specs.append(
            ModelParameterSpec(
                name=str(field_info.name),
                parameter_type="explicit",
                value_type=value_type,
                required=not has_default,
                default=to_jsonable(default) if has_default else None,
                annotation=_annotation_label(field_info.type),
                editable=editable,
                schema_source=default_source,
            )
        )
    return ModelParameterSchema(name=name, fields=tuple(specs), schema_source=default_source)


def _parameter_schema_from_signature(
    target: Any,
    *,
    name: str,
    default_source: str,
) -> ModelParameterSchema | None:
    callable_target = target.__init__ if inspect.isclass(target) else target
    try:
        signature = inspect.signature(callable_target)
    except Exception:
        return None

    specs: list[ModelParameterSpec] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        parameter_type = "vararg" if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD} else "explicit"
        default = parameter.default
        value_type, editable = _infer_value_type(parameter.annotation, default)
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            display_name = f"*{parameter.name}"
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            display_name = f"**{parameter.name}"
        else:
            display_name = parameter.name
        specs.append(
            ModelParameterSpec(
                name=display_name,
                parameter_type=parameter_type,
                value_type=value_type,
                required=default is inspect.Signature.empty and parameter_type == "explicit",
                default=None if default is inspect.Signature.empty else to_jsonable(default),
                annotation=_annotation_label(parameter.annotation),
                editable=editable and parameter_type == "explicit",
                schema_source=default_source,
            )
        )
    if not specs:
        return None
    return ModelParameterSchema(name=name, fields=tuple(specs), schema_source=default_source)


def _parameter_schema_from_object_attr(
    value: Any,
    *,
    name: str,
) -> ModelParameterSchema | None:
    if value is None:
        return None
    try:
        schema_value = value() if callable(value) else value
    except TypeError:
        return None
    return normalize_model_parameter_schema(
        schema_value,
        name=name,
        default_source="builder_attribute",
    )


def _builder_referenced_parameter_schema(builder: Any, *, name: str) -> ModelParameterSchema | None:
    if not inspect.isfunction(builder):
        return None
    globals_map = getattr(builder, "__globals__", {})
    closure_cells = [
        cell.cell_contents
        for cell in (getattr(builder, "__closure__", None) or ())
        if hasattr(cell, "cell_contents")
    ]
    dataclass_candidates: list[type[Any]] = []
    class_candidates: list[type[Any]] = []
    seen: set[int] = set()
    for symbol in getattr(builder.__code__, "co_names", ()):
        candidate = globals_map.get(symbol)
        if not inspect.isclass(candidate):
            continue
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        class_candidates.append(candidate)
        if is_dataclass(candidate):
            dataclass_candidates.append(candidate)
    for candidate in closure_cells:
        if not inspect.isclass(candidate):
            continue
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        class_candidates.append(candidate)
        if is_dataclass(candidate):
            dataclass_candidates.append(candidate)

    preferred_dataclasses = [candidate for candidate in dataclass_candidates if candidate.__name__.endswith("Config")]
    if len(preferred_dataclasses) == 1:
        return _parameter_schema_from_dataclass_cls(
            preferred_dataclasses[0],
            name=name,
            default_source="builder_referenced_dataclass",
        )
    if len(dataclass_candidates) == 1:
        return _parameter_schema_from_dataclass_cls(
            dataclass_candidates[0],
            name=name,
            default_source="builder_referenced_dataclass",
        )

    preferred_classes = [
        candidate
        for candidate in class_candidates
        if candidate.__module__ == builder.__module__ and candidate.__name__.endswith(("Model", "Adapter"))
    ]
    if len(preferred_classes) == 1:
        return _parameter_schema_from_signature(
            preferred_classes[0],
            name=name,
            default_source="builder_referenced_class",
        )
    if len(class_candidates) == 1:
        return _parameter_schema_from_signature(
            class_candidates[0],
            name=name,
            default_source="builder_referenced_class",
        )
    return None


def normalize_model_plugin_manifest(
    value: ModelPluginManifest | Mapping[str, Any] | None,
    *,
    name: str,
    default_version: str | None = None,
    default_source: str = "default",
) -> ModelPluginManifest:
    """Normalize manifest values returned by plugins.

    Accepted inputs:
    - `ModelPluginManifest`
    - mapping/dict with compatible keys
    - `None`, which yields a minimal default manifest
    """

    if isinstance(value, ModelPluginManifest):
        manifest = value
    elif isinstance(value, Mapping):
        payload = dict(value)
        capabilities_value = payload.get("capabilities")
        if isinstance(capabilities_value, PluginCapabilities):
            capabilities = capabilities_value
        elif isinstance(capabilities_value, Mapping):
            capabilities = PluginCapabilities(**dict(capabilities_value))
        elif capabilities_value is None:
            capabilities = PluginCapabilities()
        else:
            raise TypeError("capabilities must be a mapping, PluginCapabilities, or None.")

        manifest = ModelPluginManifest(
            name=str(payload.get("name") or name),
            display_name=payload.get("display_name"),
            version=payload.get("version"),
            family=payload.get("family"),
            description=payload.get("description"),
            runtime_device_hints=_normalize_string_seq(payload.get("runtime_device_hints")),
            supported_dataset_sources=_normalize_string_seq(payload.get("supported_dataset_sources")),
            required_input=payload.get("required_input"),
            default_pipeline=payload.get("default_pipeline"),
            required_pipeline=payload.get("required_pipeline"),
            tags=_normalize_string_seq(payload.get("tags")),
            notes=payload.get("notes"),
            capabilities=capabilities,
            manifest_source=str(payload.get("manifest_source") or default_source),
        )
    elif value is None:
        manifest = ModelPluginManifest(
            name=name,
            display_name=name,
            version=default_version,
            description="No explicit plugin manifest was provided by this package.",
            manifest_source=default_source,
        )
    else:
        raise TypeError(
            "Plugin manifest must be a ModelPluginManifest, a mapping, or None."
        )

    if not manifest.name:
        manifest = replace(manifest, name=name)
    if manifest.version is None and default_version is not None:
        manifest = replace(manifest, version=default_version)
    if not manifest.manifest_source:
        manifest = replace(manifest, manifest_source=default_source)
    return manifest


@lru_cache(maxsize=1)
def _model_entry_points() -> dict[str, metadata.EntryPoint]:
    try:
        entry_points = metadata.entry_points(group=MODEL_ENTRYPOINT_GROUP)
    except TypeError:  # pragma: no cover - older importlib.metadata fallback
        entry_points = metadata.entry_points().get(MODEL_ENTRYPOINT_GROUP, [])
    return {ep.name: ep for ep in entry_points}


@lru_cache(maxsize=1)
def _manifest_entry_points() -> dict[str, metadata.EntryPoint]:
    try:
        entry_points = metadata.entry_points(group=MODEL_MANIFEST_ENTRYPOINT_GROUP)
    except TypeError:  # pragma: no cover - older importlib.metadata fallback
        entry_points = metadata.entry_points().get(MODEL_MANIFEST_ENTRYPOINT_GROUP, [])
    return {ep.name: ep for ep in entry_points}


@lru_cache(maxsize=1)
def _entrypoint_model_factories() -> dict[str, Callable[..., Any]]:
    factories: dict[str, Callable[..., Any]] = {}
    for ep in _model_entry_points().values():
        factories[ep.name] = ep.load()
    return factories



def _entrypoint_dist_info(ep: metadata.EntryPoint) -> tuple[str | None, str | None]:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None, None
    name = getattr(dist, "name", None)
    version = getattr(dist, "version", None)
    return name, version



def _call_manifest_provider(provider: Any) -> Any:
    return provider() if callable(provider) else provider


@lru_cache(maxsize=1)
def _entrypoint_manifests() -> dict[str, ModelPluginManifest]:
    manifests: dict[str, ModelPluginManifest] = {}
    for ep in _manifest_entry_points().values():
        package, version = _entrypoint_dist_info(ep)
        provider = ep.load()
        manifest_value = _call_manifest_provider(provider)
        manifests[ep.name] = normalize_model_plugin_manifest(
            manifest_value,
            name=ep.name,
            default_version=version,
            default_source="entry_point",
        )
    return manifests



def _manifest_from_object_attr(value: Any, *, name: str) -> ModelPluginManifest | None:
    if value is None:
        return None
    try:
        manifest_value = value() if callable(value) else value
    except TypeError:
        return None
    return normalize_model_plugin_manifest(
        manifest_value,
        name=name,
        default_source="builder_attribute",
    )



def extract_model_plugin_manifest(model_or_builder: Any, *, default_name: str | None = None) -> ModelPluginManifest | None:
    """Best-effort manifest extraction from a model instance or builder.

    This is used primarily for direct entrypoint models, where the object is
    available only after instantiation.
    """

    candidates = [
        getattr(model_or_builder, "get_plugin_manifest", None),
        getattr(model_or_builder, "plugin_manifest", None),
        getattr(model_or_builder, "PLUGIN_MANIFEST", None),
        getattr(model_or_builder, "MANIFEST", None),
    ]
    cls = getattr(model_or_builder, "__class__", None)
    if cls is not None:
        candidates.extend(
            [
                getattr(cls, "get_plugin_manifest", None),
                getattr(cls, "plugin_manifest", None),
                getattr(cls, "PLUGIN_MANIFEST", None),
                getattr(cls, "MANIFEST", None),
            ]
        )

    name = default_name or getattr(model_or_builder, "name", None) or model_or_builder.__class__.__name__
    for candidate in candidates:
        manifest = _manifest_from_object_attr(candidate, name=str(name))
        if manifest is not None:
            return manifest
    return None


def extract_model_parameter_schema(
    model_or_builder: Any,
    *,
    default_name: str | None = None,
) -> ModelParameterSchema | None:
    """Best-effort parameter schema extraction from a model instance or builder."""

    name = default_name or getattr(model_or_builder, "name", None) or model_or_builder.__class__.__name__
    candidates = [
        getattr(model_or_builder, "get_parameter_schema", None),
        getattr(model_or_builder, "parameter_schema", None),
        getattr(model_or_builder, "PARAMETER_SCHEMA", None),
        getattr(model_or_builder, "PARAMETERS", None),
    ]
    cls = getattr(model_or_builder, "__class__", None)
    if cls is not None:
        candidates.extend(
            [
                getattr(cls, "get_parameter_schema", None),
                getattr(cls, "parameter_schema", None),
                getattr(cls, "PARAMETER_SCHEMA", None),
                getattr(cls, "PARAMETERS", None),
            ]
        )

    for candidate in candidates:
        schema = _parameter_schema_from_object_attr(candidate, name=str(name))
        if schema is not None and schema.fields:
            return schema

    for config_cls in (
        getattr(model_or_builder, "CONFIG_CLS", None),
        None if cls is None else getattr(cls, "CONFIG_CLS", None),
    ):
        schema = _parameter_schema_from_dataclass_cls(
            config_cls,
            name=str(name),
            default_source="config_dataclass",
        )
        if schema is not None and schema.fields:
            return schema

    schema = _builder_referenced_parameter_schema(model_or_builder, name=str(name))
    if schema is not None and schema.fields:
        return schema

    schema = _parameter_schema_from_signature(
        model_or_builder,
        name=str(name),
        default_source="call_signature",
    )
    if schema is not None and schema.fields:
        return schema
    return None


@lru_cache(maxsize=1)
def discover_model_plugins() -> dict[str, PluginInfo]:
    discovered: dict[str, PluginInfo] = {
        name: PluginInfo(
            name=name,
            source="builtin",
            target=factory.__module__ + ":" + getattr(factory, "__name__", factory.__class__.__name__),
            package="ts-benchmark",
            package_version=_BUILTIN_PACKAGE_VERSION,
            manifest=BUILTIN_MODEL_MANIFESTS.get(name),
        )
        for name, factory in BUILTIN_MODEL_FACTORIES.items()
    }

    entrypoint_manifests = _entrypoint_manifests()
    for ep in _model_entry_points().values():
        package, version = _entrypoint_dist_info(ep)
        discovered[ep.name] = PluginInfo(
            name=ep.name,
            source="entry_point",
            target=ep.value,
            package=package,
            package_version=version,
            manifest=entrypoint_manifests.get(
                ep.name,
                normalize_model_plugin_manifest(
                    None,
                    name=ep.name,
                    default_version=version,
                    default_source="default",
                ),
            ),
        )
    return discovered



def list_model_plugins() -> dict[str, dict[str, Any]]:
    return to_jsonable(discover_model_plugins())



def get_model_plugin_info(name: str) -> PluginInfo:
    discovered = discover_model_plugins()
    if name not in discovered:
        available = sorted(discover_model_plugins())
        raise KeyError(f"Unknown model plugin '{name}'. Available: {available}")
    return discovered[name]



def resolve_model_plugin_manifest(name: str) -> ModelPluginManifest | None:
    return get_model_plugin_info(name).manifest


def resolve_model_plugin_parameter_schema(name: str) -> ModelParameterSchema | None:
    builder = resolve_model_plugin(name)
    return extract_model_parameter_schema(builder, default_name=name)



def resolve_model_plugin(name: str) -> Callable[..., Any]:
    if name in BUILTIN_MODEL_FACTORIES:
        return BUILTIN_MODEL_FACTORIES[name]
    factories = _entrypoint_model_factories()
    if name in factories:
        return factories[name]
    available = sorted(discover_model_plugins())
    raise KeyError(f"Unknown model plugin '{name}'. Available: {available}")



def clear_plugin_caches() -> None:
    discover_model_plugins.cache_clear()
    _entrypoint_model_factories.cache_clear()
    _model_entry_points.cache_clear()
    _manifest_entry_points.cache_clear()
    _entrypoint_manifests.cache_clear()
