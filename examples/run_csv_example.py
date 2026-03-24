from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.paths import SAMPLE_DATA_DIR
from ts_benchmark.run import run_benchmark_from_config
from ts_benchmark.serialization import to_jsonable


if __name__ == "__main__":
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "csv_example",
            "dataset": {
                "name": "demo_returns",
                "provider": {
                    "kind": "csv",
                    "config": {"path": str(SAMPLE_DATA_DIR / "demo_returns.csv")},
                },
                "schema": {
                    "layout": "wide",
                    "time_column": "date",
                    "target_columns": [
                        "EU_Banks",
                        "US_Tech",
                        "Global_Industrials",
                        "EM_Consumer",
                    ],
                    "frequency": "B",
                },
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "train_size": 80,
                "test_size": 20,
                "context_length": 8,
                "horizon": 3,
                "eval_stride": 5,
                "train_stride": 1,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 12,
            },
            "metrics": [{"name": "crps"}, {"name": "energy_score"}],
            "models": [
                {
                    "name": "historical_bootstrap",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 3},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        },
        "run": {
            "seed": 19,
            "execution": {"scheduler": "auto"},
            "output": {"keep_scenarios": False, "output_dir": None},
        },
    }
    artifacts = run_benchmark_from_config(config)
    print("Run metadata:")
    print(json.dumps(to_jsonable(artifacts.run), indent=2))
    print("\nMetrics:")
    print(artifacts.results.metrics_frame().round(6).to_string())
