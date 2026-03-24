"""Model definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import JsonObject, StringMap


@dataclass
class PipelineStepConfig:
    type: str
    params: JsonObject = field(default_factory=JsonObject)



@dataclass
class PipelineConfig:
    name: str
    steps: list[PipelineStepConfig] = field(default_factory=list)



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



@dataclass
class ModelConfig:
    name: str
    reference: ModelReferenceConfig
    params: JsonObject = field(default_factory=JsonObject)
    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig(name="raw", steps=[]))
    execution: ModelExecutionConfig | None = None
    description: str | None = None
    metadata: JsonObject = field(default_factory=JsonObject)
