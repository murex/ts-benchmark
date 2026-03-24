"""JSON-safe serialization for benchmark domain objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: to_jsonable(getattr(value, field.name))
        for field in fields(value)
    }


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_dataclass(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
