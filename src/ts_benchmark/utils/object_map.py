"""Typed wrappers for dynamic key-value objects at the system boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, JsonObject | StringMap):
        return value
    if isinstance(value, Mapping):
        return JsonObject(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_json_value(item) for item in value)
    return value


def _to_builtin_json(value: Any) -> Any:
    if isinstance(value, JsonObject | StringMap):
        return value.to_builtin()
    if isinstance(value, list):
        return [_to_builtin_json(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin_json(item) for item in value]
    return value


class JsonObject(Mapping[str, Any]):
    """Immutable mapping wrapper used for dynamic JSON-like payloads."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any] | None = None):
        source = {} if value is None else value
        self._data = {str(key): _normalize_json_value(item) for key, item in source.items()}

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __bool__(self) -> bool:
        return bool(self._data)

    def to_builtin(self) -> dict[str, Any]:
        return {key: _to_builtin_json(value) for key, value in self._data.items()}


class StringMap(Mapping[str, str]):
    """Immutable mapping wrapper for string-to-string configuration maps."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any] | None = None):
        source = {} if value is None else value
        self._data = {str(key): str(item) for key, item in source.items()}

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __bool__(self) -> bool:
        return bool(self._data)

    def to_builtin(self) -> dict[str, str]:
        return dict(self._data)
