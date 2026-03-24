"""Time-series generative benchmarking framework."""

from __future__ import annotations

from importlib import import_module

_EXPORT_TO_MODULE = {
    "BenchmarkConfig": ".benchmark",
    "BenchmarkDiagnostics": ".results",
    "BenchmarkResults": ".results",
    "BenchmarkRunArtifacts": ".run",
    "Protocol": ".benchmark",
    "dump_benchmark_config": ".benchmark",
    "list_benchmark_summaries": ".benchmark",
    "load_benchmark_config": ".benchmark",
    "run_benchmark_from_config": ".run",
    "summarize_benchmark": ".benchmark",
    "validate_benchmark_config": ".benchmark",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
