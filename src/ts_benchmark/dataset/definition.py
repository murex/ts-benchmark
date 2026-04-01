"""Dataset definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..serialization import to_jsonable
from ..utils import JsonObject
from .providers.synthetic import RegimeSwitchingFactorSVConfig


def _json_object(value: Mapping[str, Any] | JsonObject | None) -> JsonObject:
    return value if isinstance(value, JsonObject) else JsonObject(value)


def _coerce_regime_switching_params(
    value: RegimeSwitchingFactorSVConfig | Mapping[str, Any] | JsonObject | None,
) -> RegimeSwitchingFactorSVConfig | JsonObject:
    if isinstance(value, RegimeSwitchingFactorSVConfig):
        return value
    if isinstance(value, JsonObject):
        payload = value.to_builtin()
    else:
        payload = {} if value is None else dict(value)
    try:
        return RegimeSwitchingFactorSVConfig(**payload)
    except TypeError:
        return JsonObject(payload)


@dataclass
class DatasetProviderConfig:
    """Generic provider config for extension-owned or unknown dataset sources."""

    kind: str
    config: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if not isinstance(self.config, JsonObject):
            self.config = JsonObject(self.config)

    def config_payload(self) -> dict[str, Any]:
        return self.config.to_builtin()


@dataclass
class SyntheticDatasetProviderConfig:
    generator: str = "regime_switching_factor_sv"
    params: RegimeSwitchingFactorSVConfig | JsonObject = field(
        default_factory=RegimeSwitchingFactorSVConfig
    )
    kind: str = field(default="synthetic", init=False)

    def __post_init__(self) -> None:
        if str(self.generator) == "regime_switching_factor_sv":
            self.params = _coerce_regime_switching_params(self.params)
        elif not isinstance(self.params, JsonObject):
            self.params = JsonObject(self.params)

    @property
    def config(self) -> JsonObject:
        return JsonObject(self.config_payload())

    def config_payload(self) -> dict[str, Any]:
        return {
            "generator": str(self.generator),
            "params": to_jsonable(self.params),
        }


@dataclass
class _TabularDatasetProviderConfigBase:
    path: str
    value_column: str | None = None
    sort_ascending: bool = True
    dropna: str = "any"
    read_kwargs: JsonObject = field(default_factory=JsonObject)
    extra: JsonObject = field(default_factory=JsonObject)
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.read_kwargs, JsonObject):
            self.read_kwargs = JsonObject(self.read_kwargs)
        if not isinstance(self.extra, JsonObject):
            self.extra = JsonObject(self.extra)
        self.path = str(self.path)
        if self.value_column is not None:
            self.value_column = str(self.value_column)
        self.dropna = str(self.dropna or "any")

    @property
    def config(self) -> JsonObject:
        return JsonObject(self.config_payload())

    def config_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": str(self.path),
            "sort_ascending": bool(self.sort_ascending),
            "dropna": str(self.dropna),
        }
        if self.value_column is not None:
            payload["value_column"] = str(self.value_column)
        if self.read_kwargs:
            payload["read_kwargs"] = self.read_kwargs.to_builtin()
        payload.update(self.extra.to_builtin())
        return payload


@dataclass
class CsvDatasetProviderConfig(_TabularDatasetProviderConfigBase):
    kind: str = field(default="csv", init=False)


@dataclass
class ParquetDatasetProviderConfig(_TabularDatasetProviderConfigBase):
    kind: str = field(default="parquet", init=False)


DatasetProvider = (
    DatasetProviderConfig
    | SyntheticDatasetProviderConfig
    | CsvDatasetProviderConfig
    | ParquetDatasetProviderConfig
)


def dataset_provider_from_mapping(provider_block: Mapping[str, Any] | DatasetProvider) -> DatasetProvider:
    if isinstance(
        provider_block,
        (
            DatasetProviderConfig,
            SyntheticDatasetProviderConfig,
            CsvDatasetProviderConfig,
            ParquetDatasetProviderConfig,
        ),
    ):
        return provider_block
    provider = dict(provider_block)
    kind = str(provider.get("kind") or "").strip()
    config = dict(provider.get("config") or {})
    if kind == "synthetic":
        return SyntheticDatasetProviderConfig(
            generator=str(config.get("generator") or "regime_switching_factor_sv"),
            params=config.get("params"),
        )
    if kind == "csv":
        extra = {
            key: value
            for key, value in config.items()
            if key not in {"path", "value_column", "sort_ascending", "dropna", "read_kwargs"}
        }
        return CsvDatasetProviderConfig(
            path=str(config.get("path") or ""),
            value_column=None if config.get("value_column") is None else str(config.get("value_column")),
            sort_ascending=bool(config.get("sort_ascending", True)),
            dropna=str(config.get("dropna", "any")),
            read_kwargs=_json_object(config.get("read_kwargs")),
            extra=_json_object(extra),
        )
    if kind == "parquet":
        extra = {
            key: value
            for key, value in config.items()
            if key not in {"path", "value_column", "sort_ascending", "dropna", "read_kwargs"}
        }
        return ParquetDatasetProviderConfig(
            path=str(config.get("path") or ""),
            value_column=None if config.get("value_column") is None else str(config.get("value_column")),
            sort_ascending=bool(config.get("sort_ascending", True)),
            dropna=str(config.get("dropna", "any")),
            read_kwargs=_json_object(config.get("read_kwargs")),
            extra=_json_object(extra),
        )
    return DatasetProviderConfig(kind=kind, config=_json_object(config))


@dataclass
class DatasetConfig:
    name: str | None = None
    description: str | None = None
    provider: DatasetProvider = field(default_factory=SyntheticDatasetProviderConfig)
    layout: str = "wide"
    time_column: str | None = None
    series_id_columns: list[str] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    static_feature_columns: list[str] = field(default_factory=list)
    frequency: str | None = None
    semantics: JsonObject = field(default_factory=JsonObject)
    metadata: JsonObject = field(default_factory=JsonObject)

    def __post_init__(self) -> None:
        if isinstance(self.provider, Mapping):
            self.provider = dataset_provider_from_mapping(self.provider)
        if not isinstance(self.semantics, JsonObject):
            self.semantics = JsonObject(self.semantics)
        if not isinstance(self.metadata, JsonObject):
            self.metadata = JsonObject(self.metadata)

    @property
    def source(self) -> str:
        return str(self.provider.kind)

    @property
    def generator(self) -> str | None:
        if isinstance(self.provider, SyntheticDatasetProviderConfig):
            return str(self.provider.generator)
        return self.provider.config.get("generator")

    @property
    def path(self) -> str | None:
        if isinstance(self.provider, CsvDatasetProviderConfig | ParquetDatasetProviderConfig):
            return str(self.provider.path)
        value = self.provider.config.get("path")
        return None if value is None else str(value)

    @property
    def params(self) -> dict[str, Any]:
        if isinstance(self.provider, SyntheticDatasetProviderConfig):
            return to_jsonable(self.provider.params)
        params = self.provider.config.to_builtin()
        params.pop("path", None)
        return params

    @property
    def freq(self) -> str:
        return str(self.frequency or "B")

    @freq.setter
    def freq(self, value: str | None) -> None:
        self.frequency = None if value is None else str(value)
