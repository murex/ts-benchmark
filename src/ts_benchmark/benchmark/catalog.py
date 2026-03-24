"""Catalog helpers for shipped benchmark configs and packaged baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..paths import SAMPLE_DATA_DIR, SHIPPED_BENCHMARK_CACHE_DIR, ensure_workspace_dirs
from .definition import BenchmarkConfig
from .io import load_benchmark_config

_RESOURCE_PACKAGE = "ts_benchmark.resources"
_RESOURCE_BENCHMARKS_DIR = "benchmarks"
_RESOURCE_SAMPLE_DATA_DIR = "sample_data"


@dataclass(frozen=True)
class BenchmarkSummary:
    """Small summary view over a benchmark config."""

    key: str
    name: str
    description: str | None
    path: Path | None
    dataset_name: str | None
    dataset_provider: str
    model_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    has_baseline: bool = False

    @property
    def n_models(self) -> int:
        return len(self.model_names)

    @property
    def n_metrics(self) -> int:
        return len(self.metric_names)


def benchmark_key_for_path(path: str | Path) -> str:
    """Return the storage key for a benchmark config path."""

    benchmark_path = Path(path).expanduser().resolve()
    if benchmark_path.name == "benchmark.json" and benchmark_path.parent.name:
        return benchmark_path.parent.name
    return benchmark_path.stem


def _resource_benchmark_root():
    return resources.files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_BENCHMARKS_DIR)


def _resource_sample_data_root():
    return resources.files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_SAMPLE_DATA_DIR)


def _sync_resource_tree(source, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    children = list(source.iterdir())
    child_names = {child.name for child in children}
    for existing in destination.iterdir():
        if existing.name in child_names:
            continue
        if existing.is_dir():
            for nested in sorted(existing.rglob("*"), reverse=True):
                if nested.is_file():
                    nested.unlink()
                elif nested.is_dir():
                    nested.rmdir()
            existing.rmdir()
        else:
            existing.unlink()
    for child in children:
        target = destination / child.name
        if child.is_dir():
            _sync_resource_tree(child, target)
            continue
        data = child.read_bytes()
        if target.exists() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def default_benchmark_config_dir() -> Path:
    """Return the cached directory of packaged benchmark configs."""

    ensure_workspace_dirs()
    _sync_resource_tree(_resource_benchmark_root(), SHIPPED_BENCHMARK_CACHE_DIR)
    _sync_resource_tree(_resource_sample_data_root(), SAMPLE_DATA_DIR)
    return SHIPPED_BENCHMARK_CACHE_DIR


def shipped_benchmark_paths(config_dir: str | Path | None = None) -> dict[str, Path]:
    """Return shipped benchmark config paths keyed by benchmark id."""

    if config_dir is not None:
        root = Path(config_dir).expanduser().resolve()
        if not root.exists():
            return {}
        nested = {
            path.parent.name: path.resolve()
            for path in sorted(root.glob("*/benchmark.json"))
        }
        if nested:
            return nested
        return {
            path.stem: path.resolve()
            for path in sorted(root.glob("*.json"))
        }
    root = default_benchmark_config_dir()
    return {
        path.parent.name: path.resolve()
        for path in sorted(root.glob("*/benchmark.json"))
    }


def packaged_baseline_dir(
    benchmark: str | Path | dict[str, Any] | BenchmarkConfig,
    *,
    config_dir: str | Path | None = None,
) -> Path | None:
    """Return the packaged baseline directory for a shipped benchmark when available."""

    if config_dir is not None:
        return None
    if isinstance(benchmark, BenchmarkConfig):
        if benchmark.source_path is None:
            return None
        key = benchmark_key_for_path(benchmark.source_path)
    elif isinstance(benchmark, dict):
        config = load_benchmark_config(benchmark)
        if config.source_path is None:
            return None
        key = benchmark_key_for_path(config.source_path)
    else:
        candidate_path = Path(str(benchmark)).expanduser()
        if candidate_path.exists():
            key = benchmark_key_for_path(candidate_path)
        else:
            key = str(benchmark).strip()
    candidate = default_benchmark_config_dir() / key / "baseline"
    if not candidate.exists():
        return None
    return candidate.resolve()


def has_packaged_baseline(
    benchmark: str | Path | dict[str, Any] | BenchmarkConfig,
    *,
    config_dir: str | Path | None = None,
) -> bool:
    """Return whether a packaged baseline is available for the benchmark."""

    return packaged_baseline_dir(benchmark, config_dir=config_dir) is not None


def resolve_benchmark_reference(
    benchmark: str | Path | dict[str, Any] | BenchmarkConfig,
    *,
    config_dir: str | Path | None = None,
) -> str | Path | dict[str, Any] | BenchmarkConfig:
    if isinstance(benchmark, BenchmarkConfig | dict):
        return benchmark
    path = Path(str(benchmark)).expanduser()
    if path.exists():
        return path.resolve()
    options = shipped_benchmark_paths(config_dir=config_dir)
    key = str(benchmark).strip()
    if key in options:
        return options[key]
    candidate = Path(key)
    if candidate.suffix != ".json" and f"{key}.json" in {p.name for p in options.values()}:
        return next(path for path in options.values() if path.name == f"{key}.json")
    raise FileNotFoundError(f"Unknown benchmark '{benchmark}'.")


def summarize_benchmark(
    benchmark: str | Path | dict[str, Any] | BenchmarkConfig,
    *,
    config_dir: str | Path | None = None,
) -> BenchmarkSummary:
    """Load a benchmark config and return a small summary view."""

    resolved = resolve_benchmark_reference(benchmark, config_dir=config_dir)
    config = resolved if isinstance(resolved, BenchmarkConfig) else load_benchmark_config(resolved)
    source_path = config.source_path
    dataset_name = config.dataset.name or None
    key = (
        benchmark_key_for_path(source_path)
        if source_path is not None
        else str(config.name).strip().lower().replace(" ", "_")
    )
    description = config.description
    if source_path is not None:
        manifest_path = source_path.with_name("manifest.json")
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            if not description:
                description = manifest.get("description")
    return BenchmarkSummary(
        key=key,
        name=str(config.name),
        description=None if description is None else str(description),
        path=None if source_path is None else Path(source_path).resolve(),
        dataset_name=dataset_name,
        dataset_provider=str(config.dataset.provider.kind),
        model_names=tuple(str(model.name) for model in config.models),
        metric_names=tuple(str(metric.name) for metric in config.metrics),
        has_baseline=has_packaged_baseline(config, config_dir=config_dir),
    )


def list_benchmark_summaries(config_dir: str | Path | None = None) -> list[BenchmarkSummary]:
    """List shipped benchmark configs as summary objects."""

    return [
        summarize_benchmark(path, config_dir=config_dir)
        for _, path in shipped_benchmark_paths(config_dir=config_dir).items()
    ]
