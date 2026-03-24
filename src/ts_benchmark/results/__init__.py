"""Benchmark results, diagnostics, and reporting."""

from .distribution_summary import build_distribution_summaries
from .functional_smoke import build_functional_smoke_tables
from .per_window import build_per_window_metrics
from .report import metrics_table_to_markdown
from .types import (
    BenchmarkDiagnostics,
    BenchmarkResults,
    MetricRanking,
    MetricResult,
    ModelDeviceAssignment,
    ModelExecutionRecord,
    ModelResult,
    ResolvedRunExecution,
    RunRecord,
    RunTrackingRecord,
    ScenarioOutput,
)

__all__ = [
    "BenchmarkDiagnostics",
    "BenchmarkResults",
    "MetricRanking",
    "MetricResult",
    "ModelDeviceAssignment",
    "ModelExecutionRecord",
    "ModelResult",
    "ResolvedRunExecution",
    "RunRecord",
    "RunTrackingRecord",
    "ScenarioOutput",
    "build_distribution_summaries",
    "build_functional_smoke_tables",
    "build_per_window_metrics",
    "metrics_table_to_markdown",
]
