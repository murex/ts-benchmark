"""Dataset definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import JsonObject


@dataclass
class DatasetProviderConfig:
    kind: str
    config: JsonObject = field(default_factory=JsonObject)



@dataclass
class DatasetConfig:
    name: str | None = None
    description: str | None = None
    provider: DatasetProviderConfig = field(default_factory=lambda: DatasetProviderConfig(kind="synthetic"))
    layout: str = "wide"
    time_column: str | None = None
    series_id_columns: list[str] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    static_feature_columns: list[str] = field(default_factory=list)
    frequency: str | None = None
    semantics: JsonObject = field(default_factory=JsonObject)
    metadata: JsonObject = field(default_factory=JsonObject)

    @property
    def source(self) -> str:
        return self.provider.kind

    @property
    def generator(self) -> str | None:
        return self.provider.config.get("generator")

    @property
    def path(self) -> str | None:
        value = self.provider.config.get("path")
        return None if value is None else str(value)

    @property
    def params(self) -> dict[str, Any]:
        if self.provider.kind == "synthetic":
            params = self.provider.config.get("params")
            if isinstance(params, JsonObject):
                return params.to_builtin()
            return dict(params or {})
        params = self.provider.config.to_builtin()
        params.pop("path", None)
        return params

    @property
    def freq(self) -> str:
        return str(self.frequency or "B")

    @freq.setter
    def freq(self, value: str | None) -> None:
        self.frequency = None if value is None else str(value)
