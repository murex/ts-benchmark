"""Command-line entry point for config-driven benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..benchmark import (
    dump_benchmark_config,
    list_benchmark_summaries,
    load_benchmark_config,
    resolve_benchmark_reference,
    validate_benchmark_config,
)
from ..model.catalog.plugins import list_model_plugins
from ..run import run_benchmark_from_config
from ..serialization import to_jsonable


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_benchmark_config(resolve_benchmark_reference(args.config))
    validate_benchmark_config(dump_benchmark_config(config))
    print(f"Config is valid: {args.config}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config_path = resolve_benchmark_reference(args.config)
    config = load_benchmark_config(config_path)
    if args.output_dir is not None:
        config.run.output.output_dir = args.output_dir
    if args.device is not None:
        config.run.device = args.device
    artifacts = run_benchmark_from_config(config)
    print("Run metadata:")
    print(json.dumps(to_jsonable(artifacts.run), indent=2))
    print("\nMetrics:")
    print(artifacts.results.metrics_frame().round(6).to_string())
    if artifacts.output_dir is not None:
        print(f"\nSaved outputs to: {artifacts.output_dir}")
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    plugins = list_model_plugins()
    if args.json:
        print(json.dumps(plugins, indent=2))
        return 0
    print("Available model plugins:")
    for name, info in sorted(plugins.items()):
        manifest = info.get("manifest") or {}
        display_name = manifest.get("display_name") or name
        device_hints = ",".join(manifest.get("runtime_device_hints") or []) or "n/a"
        caps = manifest.get("capabilities") or {}
        probabilistic = caps.get("probabilistic_sampling")
        multivariate = caps.get("multivariate")
        print(
            f"- {name:28s} [{info['source']}] {display_name} | "
            f"devices={device_hints} | multivariate={multivariate} | probabilistic={probabilistic}"
        )
        print(f"  target={info['target']}")
    return 0


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    summaries = list_benchmark_summaries(config_dir=args.config_dir)
    if args.json:
        print(json.dumps(to_jsonable(summaries), indent=2))
        return 0
    if not summaries:
        print("No benchmark configs were found.")
        return 0
    print("Available benchmarks:")
    for summary in summaries:
        print(
            f"- {summary.key:36s} {summary.name} | "
            f"models={summary.n_models} | metrics={summary.n_metrics} | "
            f"dataset={summary.dataset_provider} | baseline={'yes' if summary.has_baseline else 'no'}"
        )
        if summary.description:
            print(f"  description={summary.description}")
        print(f"  models={', '.join(summary.model_names) or 'n/a'}")
        print(f"  metrics={', '.join(summary.metric_names) or 'n/a'}")
        if summary.path is not None:
            print(f"  path={summary.path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Time-series generative benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a benchmark JSON config")
    validate_parser.add_argument("config", help="Path to a benchmark config JSON file")
    validate_parser.set_defaults(func=_cmd_validate)

    run_parser = subparsers.add_parser("run", help="Run a benchmark from a JSON config")
    run_parser.add_argument("config", help="Path to a benchmark config JSON file")
    run_parser.add_argument("--output-dir", default=None, help="Optional output directory override")
    run_parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device override. Use cpu, cuda:0, mps, a comma-separated list like "
            "cuda:0,cuda:1, or omit for auto device selection."
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    plugins_parser = subparsers.add_parser("plugins", help="List built-in and discovered model plugins")
    plugins_parser.add_argument("--json", action="store_true", help="Emit plugin listing as JSON")
    plugins_parser.set_defaults(func=_cmd_plugins)

    benchmarks_parser = subparsers.add_parser("benchmarks", help="List shipped benchmark configs")
    benchmarks_parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional benchmark-config directory override. Defaults to the packaged benchmark catalog.",
    )
    benchmarks_parser.add_argument("--json", action="store_true", help="Emit benchmark listing as JSON")
    benchmarks_parser.set_defaults(func=_cmd_benchmarks)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
