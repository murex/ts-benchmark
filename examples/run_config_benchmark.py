from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark import shipped_benchmark_paths
from ts_benchmark.run import run_benchmark_from_config


if __name__ == "__main__":
    config_path = shipped_benchmark_paths()["smoke_test"]
    artifacts = run_benchmark_from_config(config_path)
    print(artifacts.results.metrics_frame().round(6).to_string())
    if artifacts.output_dir is not None:
        print(f"\nSaved outputs to: {artifacts.output_dir}")
