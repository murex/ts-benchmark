from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark import (
    list_benchmark_summaries,
    shipped_benchmark_paths,
    summarize_benchmark,
)
from ts_benchmark.cli.main import main


def _benchmark_config_payload(
    *,
    name: str,
    description: str,
    model_name: str = "debug_model",
    metric_names: list[str] | None = None,
) -> dict[str, object]:
    return {
        "version": "1.0",
        "benchmark": {
            "name": name,
            "description": description,
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 19},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
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
            "metrics": [{"name": metric_name} for metric_name in (metric_names or ["crps"])],
            "models": [
                {
                    "name": model_name,
                    "reference": {
                        "kind": "entrypoint",
                        "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                    },
                    "params": {"scale": 1.0},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        },
        "run": {
            "seed": 19,
            "execution": {"device": "cpu", "scheduler": "auto"},
            "output": {
                "keep_scenarios": False,
                "save_scenarios": False,
                "save_model_info": False,
                "save_summary": False,
            },
        },
        "diagnostics": {
            "save_model_debug_artifacts": False,
            "save_distribution_summary": False,
            "save_per_window_metrics": False,
            "functional_smoke": {"enabled": False},
        },
    }


def test_list_benchmark_summaries_from_config_dir(tmp_path: Path) -> None:
    alpha_path = tmp_path / "alpha.json"
    beta_path = tmp_path / "beta.json"
    alpha_path.write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Alpha Benchmark",
                description="Alpha description",
                metric_names=["crps", "energy_score"],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    beta_path.write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Beta Benchmark",
                description="Beta description",
                model_name="alt_model",
                metric_names=["crps"],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    summaries = list_benchmark_summaries(config_dir=tmp_path)

    assert [summary.key for summary in summaries] == ["alpha", "beta"]
    assert summaries[0].name == "Alpha Benchmark"
    assert summaries[0].description == "Alpha description"
    assert summaries[0].dataset_provider == "synthetic"
    assert summaries[0].model_names == ("debug_model",)
    assert summaries[0].metric_names == ("crps", "energy_score")
    assert summaries[0].path == alpha_path.resolve()
    assert summaries[1].model_names == ("alt_model",)


def test_summarize_benchmark_accepts_key_and_path(tmp_path: Path) -> None:
    config_path = tmp_path / "gamma.json"
    config_path.write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Gamma Benchmark",
                description="Gamma description",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    by_key = summarize_benchmark("gamma", config_dir=tmp_path)
    by_path = summarize_benchmark(config_path)

    assert by_key.name == "Gamma Benchmark"
    assert by_key.path == config_path.resolve()
    assert by_path.key == "gamma"
    assert by_path.metric_names == ("crps",)


def test_cli_benchmarks_json(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "delta.json"
    config_path.write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Delta Benchmark",
                description="Delta description",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ts-benchmark",
            "benchmarks",
            "--config-dir",
            str(tmp_path),
            "--json",
        ],
    )

    exit_code = main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(output) == 1
    assert output[0]["key"] == "delta"
    assert output[0]["name"] == "Delta Benchmark"
    assert output[0]["model_names"] == ["debug_model"]


def test_nested_benchmark_catalog_paths_are_discoverable(tmp_path: Path) -> None:
    alpha_dir = tmp_path / "alpha"
    beta_dir = tmp_path / "beta"
    alpha_dir.mkdir()
    beta_dir.mkdir()
    (alpha_dir / "benchmark.json").write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Alpha Nested",
                description="Alpha nested benchmark",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (beta_dir / "benchmark.json").write_text(
        json.dumps(
            _benchmark_config_payload(
                name="Beta Nested",
                description="Beta nested benchmark",
                model_name="beta_model",
                metric_names=["crps", "energy_score"],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    shipped = shipped_benchmark_paths(config_dir=tmp_path)
    summary = summarize_benchmark("alpha", config_dir=tmp_path)

    assert set(shipped) == {"alpha", "beta"}
    assert shipped["alpha"].name == "benchmark.json"
    assert summary.key == "alpha"
    assert summary.name == "Alpha Nested"
    assert summary.has_baseline is False
