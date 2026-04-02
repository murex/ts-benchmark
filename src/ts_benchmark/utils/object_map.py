"""Typed wrappers for dynamic key-value objects at the system boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
import copy
from dataclasses import fields, is_dataclass
from typing import Any


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
    }


def _mapping_source(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_builtin") and callable(value.to_builtin):
        built = value.to_builtin()
        if isinstance(built, Mapping):
            return built
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_dict(value)
    raise TypeError(f"Expected a mapping-like value, got {type(value).__name__}.")


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, JsonObject | StringMap):
        return value
    if hasattr(value, "to_builtin") and callable(value.to_builtin):
        built = value.to_builtin()
        if isinstance(built, Mapping):
            return JsonObject(built)
    if is_dataclass(value) and not isinstance(value, type):
        return JsonObject(_dataclass_to_dict(value))
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
    if hasattr(value, "to_builtin") and callable(value.to_builtin):
        return _to_builtin_json(value.to_builtin())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: _to_builtin_json(item)
            for key, item in _dataclass_to_dict(value).items()
        }
    if isinstance(value, list):
        return [_to_builtin_json(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin_json(item) for item in value]
    return value


class JsonObject(MutableMapping[str, Any]):
    """Mutable mapping wrapper used for dynamic JSON-like payloads.

    JsonObject supports both mapping access and attribute-style access for
    key names that are valid Python identifiers and do not collide with
    wrapper methods.
    """

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any] | None = None):
        source = _mapping_source(value)
        object.__setattr__(self, "_data", {str(key): _normalize_json_value(item) for key, item in source.items()})

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[str(key)] = _normalize_json_value(value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __getattr__(self, name: str) -> Any:
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
            return
        if hasattr(type(self), name):
            raise AttributeError(
                f"{name!r} is reserved on {type(self).__name__}; use item access instead."
            )
        self._data[name] = _normalize_json_value(value)

    def __delattr__(self, name: str) -> None:
        if name == "_data" or hasattr(type(self), name):
            raise AttributeError(
                f"{name!r} is reserved on {type(self).__name__}; use item access instead."
            )
        try:
            del self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __deepcopy__(self, memo: dict[int, Any]) -> "JsonObject":
        copied = type(self)()
        memo[id(self)] = copied
        object.__setattr__(copied, "_data", copy.deepcopy(self._data, memo))
        return copied

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
        source = _mapping_source(value)
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
