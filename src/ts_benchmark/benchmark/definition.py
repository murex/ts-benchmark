"""Benchmark definition objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..dataset.definition import DatasetConfig
from ..metrics.definition import MetricConfig
from ..model.definition import ModelConfig
from ..run.definition import DiagnosticsConfig, RunConfig
from .protocol import Protocol


@dataclass
class BenchmarkConfig:
    version: str
    name: str
    dataset: DatasetConfig
    protocol: Protocol
    metrics: list[MetricConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    description: str | None = None
    run: RunConfig = field(default_factory=RunConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    source_path: Path | None = field(default=None, repr=False, compare=False)
