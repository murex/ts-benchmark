"""Benchmark definitions, protocol, and config IO."""

from .catalog import (
    BenchmarkSummary,
    benchmark_key_for_path,
    default_benchmark_config_dir,
    has_packaged_baseline,
    list_benchmark_summaries,
    packaged_baseline_dir,
    resolve_benchmark_reference,
    shipped_benchmark_paths,
    summarize_benchmark,
)
from .definition import BenchmarkConfig
from .io import dump_benchmark_config, load_benchmark_config, validate_benchmark_config
from .protocol import (
    ForecastProtocol,
    Protocol,
    UnconditionalPathDatasetProtocol,
    UnconditionalWindowedProtocol,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkSummary",
    "ForecastProtocol",
    "Protocol",
    "UnconditionalPathDatasetProtocol",
    "UnconditionalWindowedProtocol",
    "benchmark_key_for_path",
    "default_benchmark_config_dir",
    "dump_benchmark_config",
    "has_packaged_baseline",
    "list_benchmark_summaries",
    "load_benchmark_config",
    "packaged_baseline_dir",
    "resolve_benchmark_reference",
    "shipped_benchmark_paths",
    "summarize_benchmark",
    "validate_benchmark_config",
]
