"""Child-process RPC worker for externally executed benchmark models."""

from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

# NumPy 2 pickles may reference internal modules under ``numpy._core.*``.
# Older NumPy 1.x runtimes still expose the same objects under ``numpy.core.*``.
# Register aliases before unpickling cross-env payloads.
if hasattr(np, "core"):
    sys.modules.setdefault("numpy._core", np.core)
    numeric = getattr(np.core, "numeric", None)
    if numeric is not None:
        sys.modules.setdefault("numpy._core.numeric", numeric)
    multiarray = getattr(np.core, "multiarray", None)
    if multiarray is not None:
        sys.modules.setdefault("numpy._core.multiarray", multiarray)

from ..definition import ModelReferenceConfig
from ..resolution import instantiate_model_target
from .duck_typed import coerce_model_target


def _send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _ok(**payload: Any) -> None:
    _send({"status": "ok", **payload})


def _error(exc: BaseException) -> None:
    _send(
        {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def main() -> int:
    model = None

    for line in sys.stdin:
        message = json.loads(line)
        action = message.get("action")

        try:
            if action == "init":
                spec = dict(message["model_spec"])
                target = instantiate_model_target(
                    reference=ModelReferenceConfig(**dict(spec["reference"])),
                    params=spec.get("params") or {},
                )
                model = coerce_model_target(target, name=str(spec.get("name") or "external_model"))
                _ok()
                continue

            if model is None:
                raise RuntimeError("Model worker received a command before initialization.")

            if action == "fit":
                with Path(message["input_path"]).open("rb") as f:
                    train_data = pickle.load(f)
                model.fit(train_data)
                _ok()
                continue

            if action == "sample":
                with Path(message["input_path"]).open("rb") as f:
                    request = pickle.load(f)
                result = model.sample(request)
                with Path(message["output_path"]).open("wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                _ok()
                continue

            if action == "model_info":
                info = model.model_info() if hasattr(model, "model_info") else {}
                _ok(payload=info)
                continue

            if action == "debug_artifacts":
                payload = None
                getter = getattr(model, "debug_artifacts", None)
                if callable(getter):
                    payload = getter()
                _ok(payload=payload)
                continue

            if action == "close":
                closer = getattr(model, "close", None)
                if callable(closer):
                    closer()
                _ok()
                return 0

            raise ValueError(f"Unsupported worker action '{action}'.")
        except BaseException as exc:
            _error(exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
