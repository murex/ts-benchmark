from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.utils import JsonObject


def test_json_object_supports_attribute_access_and_mutation() -> None:
    payload = JsonObject({"h": 0.2, "nested": {"clip": False}})

    assert payload.h == 0.2
    assert payload["nested"].clip is False

    payload.h = 0.05
    payload.random_seed = 7

    assert payload["h"] == 0.05
    assert payload.random_seed == 7
    assert payload.to_builtin() == {
        "h": 0.05,
        "nested": {"clip": False},
        "random_seed": 7,
    }


def test_json_object_reserved_names_require_item_access() -> None:
    payload = JsonObject()

    with pytest.raises(AttributeError, match="reserved"):
        payload.items = 3

    payload["items"] = 3

    assert payload["items"] == 3
