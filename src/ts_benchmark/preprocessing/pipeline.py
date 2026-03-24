"""Configurable preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .transforms import (
    ClipTransform,
    DemeanTransform,
    IdentityTransform,
    RobustScalerTransform,
    StandardScalerTransform,
    Transform,
    WinsorizeTransform,
)


TRANSFORM_REGISTRY = {
    "identity": IdentityTransform,
    "demean": DemeanTransform,
    "standard_scale": StandardScalerTransform,
    "robust_scale": RobustScalerTransform,
    "clip": ClipTransform,
    "winsorize": WinsorizeTransform,
}


@dataclass
class PreprocessingPipeline:
    """Ordered transform sequence applied before model fit and sampling.

    The benchmark always evaluates models in the original return scale. This means
    the pipeline transforms the training set and generation context before passing
    them to the model, and it inverse-transforms generated scenarios back into the
    original scale before metrics are computed.
    """

    name: str = "raw"
    steps: list[Transform] = field(default_factory=list)

    def fit(self, x: np.ndarray) -> "PreprocessingPipeline":
        current = np.asarray(x, dtype=float)
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        current = np.asarray(x, dtype=float)
        for step in self.steps:
            current = step.transform(current)
        return current

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        current = np.asarray(x, dtype=float)
        for step in reversed(self.steps):
            current = step.inverse_transform(current)
        return current

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [step.summary() for step in self.steps],
        }


def build_pipeline_from_config(
    name: str,
    step_specs: Iterable[Mapping[str, Any] | Any] | None,
) -> PreprocessingPipeline:
    steps: list[Transform] = []
    for spec in step_specs or []:
        if hasattr(spec, "type") and hasattr(spec, "params"):
            step_type = str(spec.type).strip()
            params = spec.params.to_builtin() if hasattr(spec.params, "to_builtin") else dict(spec.params)
        else:
            step_type = str(spec.get("type", "")).strip()
            params = dict(spec.get("params", {}))
        if not step_type:
            raise ValueError("Each preprocessing step needs a non-empty 'type'.")
        if step_type not in TRANSFORM_REGISTRY:
            raise KeyError(
                f"Unknown preprocessing step '{step_type}'. Supported: {sorted(TRANSFORM_REGISTRY)}"
            )
        steps.append(TRANSFORM_REGISTRY[step_type](**params))
    return PreprocessingPipeline(name=name, steps=steps)
