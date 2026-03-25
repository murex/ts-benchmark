from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui import SRC_ROOT
from ts_benchmark.ui.pages import results as results_page
from ts_benchmark.ui.services import configs, datasets, runs
from ts_benchmark.ui.services.configs import current_config_summary


class _DummyProcess:
    def __init__(self, pid: int = 4321):
        self.pid = pid

    def poll(self):
        return None


def _valid_benchmark_config(model_names: list[str]) -> dict[str, object]:
    config = configs.default_config_dict()
    config["benchmark"]["name"] = "merge_benchmark"
    config["benchmark"]["dataset"] = {
        "name": "synthetic_default",
        "provider": {
            "kind": "synthetic",
            "config": {
                "generator": "regime_switching_factor_sv",
                "params": {"n_assets": 2, "seed": 7},
            },
        },
        "schema": {"layout": "tensor", "frequency": "B"},
        "semantics": {},
        "metadata": {},
    }
    config["benchmark"]["protocol"] = {
        "train_size": 100,
        "test_size": 20,
        "context_length": 8,
        "horizon": 2,
        "eval_stride": 5,
        "train_stride": 1,
        "n_model_scenarios": 8,
        "n_reference_scenarios": 16,
    }
    config["benchmark"]["models"] = [
        {
            "name": name,
            "reference": {"kind": "builtin", "value": "historical_bootstrap"},
            "params": {"block_size": index + 2},
            "pipeline": {"name": "raw", "steps": []},
        }
        for index, name in enumerate(model_names)
    ]
    config["run"]["execution"] = {"scheduler": "auto", "device": "cpu", "model_execution": {"mode": "inprocess"}}
    config["run"]["output"] = {}
    config["diagnostics"] = {}
    return config


def _write_run_artifacts(
    run_dir: Path,
    *,
    benchmark_config: dict[str, object],
    model_name: str,
    crps: float,
    energy_score: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "dataset_name": "synthetic::regime_switching_factor_sv",
                "dataset_source": "synthetic",
                "device": "cpu",
                "has_reference_scenarios": True,
                "train_size": 100,
                "test_size": 20,
                "context_length": 8,
                "horizon": 2,
                "eval_stride": 5,
                "train_stride": 1,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "execution_mode": "sequential",
                "crps": crps,
                "energy_score": energy_score,
                "average_rank": 1.0,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "dataset_name": "synthetic::regime_switching_factor_sv",
                "dataset_source": "synthetic",
                "device": "cpu",
                "has_reference_scenarios": True,
                "train_size": 100,
                "test_size": 20,
                "context_length": 8,
                "horizon": 2,
                "eval_stride": 5,
                "train_stride": 1,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "execution_mode": "sequential",
                "crps": 1.0,
                "energy_score": 1.0,
                "average_rank": 1.0,
            }
        ]
    ).to_csv(run_dir / "ranks.csv", index=False)
    (run_dir / "benchmark_config.json").write_text(json.dumps(benchmark_config, indent=2), encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "name": "ui-run",
                "status": "succeeded",
                "metadata": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "name": benchmark_config["benchmark"]["name"],
                "dataset": {"resolved_name": "synthetic::regime_switching_factor_sv"},
                "protocol": {"n_assets": 2},
                "metrics": [{"name": "crps"}, {"name": "energy_score"}],
                "models": [model_name],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "model_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": model_name,
                    "params": {"block_size": 2},
                    "metric_results": [
                        {
                            "model_name": model_name,
                            "metric_name": "crps",
                            "value": crps,
                            "direction": "minimize",
                            "category": None,
                            "granularity": "per_window",
                            "aggregation": "mean",
                            "metadata": {},
                        },
                        {
                            "model_name": model_name,
                            "metric_name": "energy_score",
                            "value": energy_score,
                            "direction": "minimize",
                            "category": None,
                            "granularity": "per_window",
                            "aggregation": "mean",
                            "metadata": {},
                        },
                    ],
                    "metric_rankings": [],
                    "average_rank": 1.0,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "scenarios.npz",
        **{
            f"model__{model_name}": np.ones((1, 1, 2, 2), dtype=float) * crps,
            "reference_scenarios": np.zeros((1, 1, 2, 2), dtype=float),
        },
    )


def test_launch_cli_run_falls_back_to_python_module(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr(runs.shutil, "which", lambda name: None)
    monkeypatch.setattr(runs.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "existing"))

    run_info = runs.launch_cli_run(
        {"run": {"output": {"output_dir": str(tmp_path / "out")}}},
    )

    command = captured["command"]
    kwargs = captured["kwargs"]
    env = kwargs["env"]

    assert command[:3] == [sys.executable, "-m", "ts_benchmark.cli.main"]
    assert command[3] == "run"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(SRC_ROOT)
    assert run_info["pid"] == 4321


def test_launch_cli_run_prefers_installed_console_script(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _DummyProcess(pid=9876)

    monkeypatch.setattr(runs.shutil, "which", lambda name: "/tmp/ts-benchmark")
    monkeypatch.setattr(runs.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    run_info = runs.launch_cli_run(
        {"run": {"output": {"output_dir": str(tmp_path / "out")}}},
    )

    command = captured["command"]
    kwargs = captured["kwargs"]

    assert command[:2] == ["/tmp/ts-benchmark", "run"]
    assert "PYTHONPATH" not in kwargs["env"] or kwargs["env"]["PYTHONPATH"] != str(SRC_ROOT)
    assert run_info["pid"] == 9876


def test_current_config_summary_reads_benchmark_block() -> None:
    summary = current_config_summary(
        {
            "benchmark": {
                "name": "ui_summary_smoke",
                "description": "Nested benchmark description",
                "dataset": {"name": "dj30"},
                "metrics": [{"name": "crps"}],
                "models": [{"name": "a"}, {"name": "b"}],
            }
        }
    )

    assert summary == {
        "name": "ui_summary_smoke",
        "description": "Nested benchmark description",
        "dataset": "dj30",
        "models": 2,
        "metrics": 1,
    }


def test_saved_benchmark_catalog_crud(tmp_path: Path) -> None:
    config = configs.default_config_dict()
    config["benchmark"]["name"] = "catalog_smoke"
    config["benchmark"]["description"] = "Benchmark catalog smoke test"
    config["benchmark"]["dataset"]["name"] = "synthetic_default"

    path = configs.save_benchmark_definition(config, benchmark_dir=tmp_path)
    loaded = configs.load_saved_benchmark("catalog_smoke", benchmark_dir=tmp_path)
    rows = configs.list_saved_benchmarks(benchmark_dir=tmp_path)

    assert path.exists()
    assert loaded["benchmark"]["name"] == "catalog_smoke"
    assert rows == [
        {
            "name": "catalog_smoke",
            "description": "Benchmark catalog smoke test",
            "dataset": "synthetic_default",
            "models": 0,
            "metrics": 2,
            "path": path,
            "results_run_dir": None,
            "results_updated_at": None,
            "results_summary": {},
        }
    ]

    deleted = configs.delete_saved_benchmark("catalog_smoke", benchmark_dir=tmp_path)

    assert deleted == path
    assert not path.exists()


def test_saved_benchmark_results_metadata(tmp_path: Path) -> None:
    config = configs.default_config_dict()
    config["benchmark"]["name"] = "catalog_results"

    path = configs.save_benchmark_definition(config, benchmark_dir=tmp_path)
    configs.update_saved_benchmark_results(
        path,
        tmp_path / "outputs" / "catalog_results_latest",
        summary={"models": ["hist_a", "hist_b"]},
        benchmark_dir=tmp_path,
    )

    rows = configs.list_saved_benchmarks(benchmark_dir=tmp_path)
    metadata = configs.load_saved_benchmark_metadata(path, benchmark_dir=tmp_path)

    assert rows[0]["results_run_dir"] == str((tmp_path / "outputs" / "catalog_results_latest").resolve())
    assert rows[0]["results_updated_at"] is not None
    assert rows[0]["results_summary"] == {"models": ["hist_a", "hist_b"]}
    assert metadata["latest_results"]["summary"] == {"models": ["hist_a", "hist_b"]}


def test_list_saved_benchmarks_includes_extra_benchmark_dirs(tmp_path: Path, monkeypatch) -> None:
    extra_root = tmp_path / "external_benchmarks"
    extra_root.mkdir(parents=True, exist_ok=True)
    config = configs.default_config_dict()
    config["benchmark"]["name"] = "story2_generated"
    config["benchmark"]["description"] = "Notebook-generated benchmark"
    config["benchmark"]["dataset"]["name"] = "story2_dataset"
    benchmark_path = extra_root / "story2_generated_benchmark.json"
    benchmark_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    original_shipped_benchmark_paths = configs.shipped_benchmark_paths
    monkeypatch.setattr(
        configs,
        "shipped_benchmark_paths",
        lambda config_dir=None: {} if config_dir is None else original_shipped_benchmark_paths(config_dir=config_dir),
    )
    monkeypatch.setenv("TS_BENCHMARK_UI_EXTRA_BENCHMARK_DIRS", str(extra_root))

    rows = configs.list_saved_benchmarks()

    assert rows == [
        {
            "name": "story2_generated",
            "description": "Notebook-generated benchmark",
            "dataset": "story2_dataset",
            "models": 0,
            "metrics": 2,
            "path": benchmark_path.resolve(),
            "results_run_dir": None,
            "results_updated_at": None,
            "results_summary": {},
            "origin": "official",
            "read_only": True,
        }
    ]


def test_previous_results_dir_for_path_prefers_sibling_run_dir(tmp_path: Path) -> None:
    benchmark_config = _valid_benchmark_config(["hist_a"])
    run_dir = tmp_path / "story2_custom_dataset"
    benchmark_path = run_dir / "story2_generated_benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(benchmark_config, indent=2), encoding="utf-8")
    _write_run_artifacts(run_dir, benchmark_config=benchmark_config, model_name="hist_a", crps=0.4, energy_score=0.5)

    previous = runs.previous_results_dir_for_path(benchmark_path)

    assert previous == run_dir.resolve()


def test_previous_results_dir_for_path_checks_extra_result_roots(tmp_path: Path, monkeypatch) -> None:
    benchmark_path = tmp_path / "benchmarks" / "story2_generated_benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(_valid_benchmark_config(["hist_a"]), indent=2), encoding="utf-8")

    extra_results_root = tmp_path / "external_results"
    run_dir = extra_results_root / "story2_generated_benchmark"
    _write_run_artifacts(run_dir, benchmark_config=_valid_benchmark_config(["hist_a"]), model_name="hist_a", crps=0.4, energy_score=0.5)

    monkeypatch.setenv("TS_BENCHMARK_UI_EXTRA_RESULTS_DIRS", str(extra_results_root))

    previous = runs.previous_results_dir_for_path(benchmark_path)

    assert previous == run_dir.resolve()


def test_list_saved_datasets_includes_extra_dataset_dirs(tmp_path: Path, monkeypatch) -> None:
    extra_root = tmp_path / "external_datasets"
    extra_root.mkdir(parents=True, exist_ok=True)
    dataset_path = extra_root / "story2_saved_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "Story 2 Dataset",
                "description": "Notebook dataset",
                "provider": {
                    "kind": "csv",
                    "config": {"path": "/tmp/story2.csv"},
                },
                "schema": {
                    "layout": "wide",
                    "frequency": "B",
                    "target_columns": ["asset_a", "asset_b"],
                },
                "semantics": {},
                "metadata": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("TS_BENCHMARK_UI_EXTRA_DATASET_DIRS", str(extra_root))

    rows = datasets.list_saved_datasets()

    assert any(row["name"] == "Story 2 Dataset" and Path(row["path"]) == dataset_path.resolve() for row in rows)


def test_materialize_benchmark_results_merges_partial_model_run(tmp_path: Path, monkeypatch) -> None:
    benchmark_path = tmp_path / "benchmarks" / "merge_benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    full_config = _valid_benchmark_config(["hist_a", "hist_b"])
    previous_config = _valid_benchmark_config(["hist_a"])
    current_subset_config = _valid_benchmark_config(["hist_b"])
    previous_run_dir = tmp_path / "outputs" / "previous"
    current_run_dir = tmp_path / "outputs" / "current"
    merged_root = tmp_path / "benchmark_results"
    monkeypatch.setattr(runs, "BENCHMARK_RESULTS_DIR", merged_root)

    _write_run_artifacts(previous_run_dir, benchmark_config=previous_config, model_name="hist_a", crps=0.4, energy_score=0.5)
    _write_run_artifacts(current_run_dir, benchmark_config=current_subset_config, model_name="hist_b", crps=0.2, energy_score=0.3)

    merged_dir = runs.materialize_benchmark_results(
        benchmark_path=benchmark_path,
        benchmark_config=full_config,
        source_run_dir=current_run_dir,
        previous_results_dir=previous_run_dir,
    )

    metrics = pd.read_csv(merged_dir / "metrics.csv")
    ranks = pd.read_csv(merged_dir / "ranks.csv")
    model_results = json.loads((merged_dir / "model_results.json").read_text(encoding="utf-8"))
    summary = json.loads((merged_dir / "summary.json").read_text(encoding="utf-8"))
    scenarios = np.load(merged_dir / "scenarios.npz", allow_pickle=False)

    assert list(metrics["model"]) == ["hist_b", "hist_a"]
    assert list(ranks["model"]) == ["hist_b", "hist_a"]
    assert summary["models"] == ["hist_b", "hist_a"]
    assert [item["model_name"] for item in model_results] == ["hist_b", "hist_a"]
    assert scenarios["model__hist_a"].shape == (1, 1, 2, 2)
    assert scenarios["model__hist_b"].shape == (1, 1, 2, 2)


def test_materialize_benchmark_results_keeps_previous_models_when_previous_dir_is_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_path = tmp_path / "benchmarks" / "merge_benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    merged_root = tmp_path / "benchmark_results"
    monkeypatch.setattr(runs, "BENCHMARK_RESULTS_DIR", merged_root)

    full_config = _valid_benchmark_config(["hist_a", "hist_b"])
    previous_config = _valid_benchmark_config(["hist_a"])
    current_subset_config = _valid_benchmark_config(["hist_b"])
    previous_results_dir = runs.benchmark_results_dir_for_path(benchmark_path)
    current_run_dir = tmp_path / "outputs" / "current"

    _write_run_artifacts(previous_results_dir, benchmark_config=previous_config, model_name="hist_a", crps=0.4, energy_score=0.5)
    _write_run_artifacts(current_run_dir, benchmark_config=current_subset_config, model_name="hist_b", crps=0.2, energy_score=0.3)

    merged_dir = runs.materialize_benchmark_results(
        benchmark_path=benchmark_path,
        benchmark_config=full_config,
        source_run_dir=current_run_dir,
        previous_results_dir=previous_results_dir,
    )

    metrics = pd.read_csv(merged_dir / "metrics.csv")
    model_results = json.loads((merged_dir / "model_results.json").read_text(encoding="utf-8"))
    summary = json.loads((merged_dir / "summary.json").read_text(encoding="utf-8"))
    run_payload = json.loads((merged_dir / "run.json").read_text(encoding="utf-8"))

    assert list(metrics["model"]) == ["hist_b", "hist_a"]
    assert [item["model_name"] for item in model_results] == ["hist_b", "hist_a"]
    assert summary["models"] == ["hist_b", "hist_a"]
    assert run_payload["metadata"]["benchmark_merge"]["previous_results_dir"] == str(previous_results_dir)


def test_materialize_benchmark_results_drops_models_removed_from_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_path = tmp_path / "benchmarks" / "merge_benchmark.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    merged_root = tmp_path / "benchmark_results"
    monkeypatch.setattr(runs, "BENCHMARK_RESULTS_DIR", merged_root)

    benchmark_config = _valid_benchmark_config(["hist_b"])
    previous_config = _valid_benchmark_config(["hist_a"])
    current_subset_config = _valid_benchmark_config(["hist_b"])
    previous_run_dir = tmp_path / "outputs" / "previous"
    current_run_dir = tmp_path / "outputs" / "current"

    _write_run_artifacts(previous_run_dir, benchmark_config=previous_config, model_name="hist_a", crps=0.4, energy_score=0.5)
    _write_run_artifacts(current_run_dir, benchmark_config=current_subset_config, model_name="hist_b", crps=0.2, energy_score=0.3)

    merged_dir = runs.materialize_benchmark_results(
        benchmark_path=benchmark_path,
        benchmark_config=benchmark_config,
        source_run_dir=current_run_dir,
        previous_results_dir=previous_run_dir,
    )

    metrics = pd.read_csv(merged_dir / "metrics.csv")
    model_results = json.loads((merged_dir / "model_results.json").read_text(encoding="utf-8"))
    summary = json.loads((merged_dir / "summary.json").read_text(encoding="utf-8"))
    scenarios = np.load(merged_dir / "scenarios.npz", allow_pickle=False)

    assert list(metrics["model"]) == ["hist_b"]
    assert [item["model_name"] for item in model_results] == ["hist_b"]
    assert summary["models"] == ["hist_b"]
    assert "model__hist_a" not in scenarios.files
    assert "model__hist_b" in scenarios.files


def test_results_overview_frame_includes_failed_models_without_metrics() -> None:
    frame = results_page._model_overview_frame(
        {
            "summary": {"metrics": [{"name": "crps"}]},
            "metrics": pd.DataFrame(
                [
                    {
                        "model": "good_model",
                        "crps": 0.1,
                        "average_rank": 1.0,
                    }
                ]
            ),
            "model_results": [
                {
                    "model_name": "good_model",
                    "metadata": {},
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                },
                {
                    "model_name": "failed_model",
                    "metadata": {"error": "ValueError: training exploded"},
                    "reference": {"kind": "entrypoint", "value": "demo:build_model"},
                },
            ],
        }
    )

    assert frame is not None
    assert list(frame["model"]) == ["good_model", "failed_model"]
    failed_row = frame[frame["model"] == "failed_model"].iloc[0]
    assert failed_row["status"] == "Failed"
    assert failed_row["error"] == "ValueError: training exploded"
    assert pd.isna(failed_row["crps"])


def test_results_alignment_detects_missing_model_from_latest_results() -> None:
    benchmark_config = _valid_benchmark_config(["hist_a", "path_mfc"])
    run_config = _valid_benchmark_config(["hist_a"])

    alignment = results_page._benchmark_results_alignment(benchmark_config, run_config)

    assert alignment["changed"] is True
    assert alignment["missing_models"] == ["path_mfc"]
    assert alignment["extra_models"] == []
    assert alignment["run_models"] == ["hist_a"]


def test_model_debug_report_contains_selected_model_sections(tmp_path: Path) -> None:
    artifacts = {
        "config": _valid_benchmark_config(["PathMfcMultidimScenarioModel"]),
        "run": {"resolved_output_dir": str(tmp_path / "out"), "metadata": {}},
        "metrics": pd.DataFrame(
            [
                {
                    "model": "PathMfcMultidimScenarioModel",
                    "crps": 0.11,
                    "energy_score": 0.22,
                    "average_rank": 1.0,
                }
            ]
        ),
        "model_results": [
            {
                "model_name": "PathMfcMultidimScenarioModel",
                "params": {"batch_size": 32},
                "metric_results": [{"metric_name": "crps", "value": 0.11}],
                "metric_rankings": [{"metric_name": "crps", "rank": 1.0}],
                "average_rank": 1.0,
            }
        ],
        "diagnostics": {
            "model_debug_artifacts": {
                "PathMfcMultidimScenarioModel": {
                    "wrapped_debug_artifacts": {
                        "training_log": [
                            {"step": 1, "loss": 0.25},
                            {"step": 2, "loss": 0.12},
                        ]
                    }
                }
            },
            "per_window_metrics": pd.DataFrame(
                [
                    {
                        "model": "PathMfcMultidimScenarioModel",
                        "context_index": 0,
                        "crps": 0.11,
                    }
                ]
            )
        },
        "dataset": SimpleNamespace(
            train_returns=np.array([[0.1, 0.2]]),
            contexts=np.array([[[0.1, 0.2]]]),
            realized_futures=np.array([[[0.3, 0.4]]]),
        ),
        "generated_scenarios": {
            "PathMfcMultidimScenarioModel": np.ones((1, 1, 1, 2), dtype=float),
        },
        "reference_scenarios": np.zeros((1, 1, 1, 2), dtype=float),
    }

    report = results_page._build_model_debug_report(
        "PathMfcMultidimScenarioModel",
        artifacts,
        benchmark_name="smoke test",
    )

    assert "Benchmark" in report
    assert "Model Hyperparameters" in report
    assert "Diagnostics" in report
    assert report.index("Diagnostics") < report.index("Metrics")
    assert "Metrics" in report
    assert "Training Scenarios" in report
    assert "Generated Scenarios" in report
    assert "Model Logs" in report
    assert "\"training_log\"" in report
    assert "\"loss\": 0.25" in report


def test_build_effective_config_applies_run_level_model_execution() -> None:
    config = configs.default_config_dict()
    config["benchmark"]["models"] = [
        {
            "name": "hist",
            "reference": {"kind": "builtin", "value": "historical_bootstrap"},
            "params": {"block_size": 2},
            "pipeline": {"name": "raw", "steps": []},
        }
    ]
    config["run"]["execution"]["model_execution"] = {
        "mode": "subprocess",
        "venv": "demo-env",
    }

    effective = configs.build_effective_config(config)

    assert effective["benchmark"]["models"][0]["execution"] == {
        "mode": "subprocess",
        "venv": "demo-env",
        "python": None,
        "cwd": None,
        "pythonpath": [],
        "env": {},
    }


def test_load_run_artifacts_surfaces_dataset_rebuild_error(tmp_path) -> None:
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir()
    (run_dir / "benchmark_config.json").write_text(
        """
        {
          "version": "1.0",
          "benchmark": {
            "name": "broken_csv_run",
            "dataset": {
              "provider": {
                "kind": "csv",
                "config": {
                  "path": "/tmp/does_not_exist_returns.csv",
                  "dropna": "any"
                }
              },
              "schema": {
                "layout": "wide",
                "frequency": "B"
              },
              "semantics": {},
              "metadata": {}
            },
            "protocol": {
              "train_size": 20,
              "test_size": 5,
              "context_length": 4,
              "horizon": 2,
              "eval_stride": 1,
              "train_stride": 1,
              "n_model_scenarios": 4,
              "n_reference_scenarios": 8
            },
            "metrics": [{"name": "crps"}],
            "models": [
              {
                "name": "historical",
                "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                "params": {"block_size": 2},
                "pipeline": {"name": "raw", "steps": []}
              }
            ]
          },
          "run": {
            "seed": 7,
            "execution": {"device": "cpu", "scheduler": "auto"},
            "tracking": {"mlflow": {}},
            "output": {}
          },
          "diagnostics": {}
        }
        """.strip(),
        encoding="utf-8",
    )

    payload = runs.load_run_artifacts(run_dir)

    assert payload["dataset"] is None
    assert payload["dataset_error"] is not None
    assert "FileNotFoundError" in payload["dataset_error"]


def test_data_studio_page_import_smoke() -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    assert callable(data_studio.render)


def test_benchmarks_page_import_smoke() -> None:
    import ts_benchmark.ui.pages.config_studio as config_studio

    assert callable(config_studio.render)


def test_run_lab_page_import_smoke() -> None:
    import ts_benchmark.ui.pages.run_lab as run_lab

    assert callable(run_lab.render)


def test_results_page_import_smoke() -> None:
    import ts_benchmark.ui.pages.results as results

    assert callable(results.render)


def test_benchmark_clone_name_is_unique() -> None:
    import ts_benchmark.ui.pages.config_studio as config_studio

    assert config_studio._cloned_benchmark_name("demo", {"demo", "demo_copy"}) == "demo_copy_2"


def test_benchmark_catalog_run_loads_run_lab(monkeypatch, tmp_path: Path) -> None:
    import ts_benchmark.ui.pages.config_studio as config_studio

    captured: dict[str, object] = {}

    config = configs.default_config_dict()
    config["benchmark"]["name"] = "catalog_run"
    saved_path = configs.save_benchmark_definition(config, benchmark_dir=tmp_path)

    monkeypatch.setattr(config_studio, "load_saved_benchmark", lambda name: config)
    monkeypatch.setattr(config_studio, "build_effective_config", lambda payload: payload)
    monkeypatch.setattr(config_studio, "set_current_config", lambda value: captured.__setitem__("config", value))
    monkeypatch.setattr(config_studio, "set_current_config_path", lambda value: captured.__setitem__("config_path", value))
    monkeypatch.setattr(config_studio, "set_effective_config", lambda value: captured.__setitem__("effective", value))
    monkeypatch.setattr(config_studio, "set_current_run", lambda value: captured.__setitem__("run", value))
    monkeypatch.setattr(config_studio, "set_current_run_artifacts", lambda value: captured.__setitem__("artifacts", value))
    monkeypatch.setattr(config_studio, "set_page", lambda value: captured.__setitem__("page", value))
    monkeypatch.setattr(config_studio, "set_validation_result", lambda value: captured.__setitem__("validation", value))

    config_studio._run_catalog_benchmark(
        {
            "name": "catalog_run",
            "path": saved_path,
        }
    )

    assert captured["page"] == "Run Lab"
    assert captured["config"]["benchmark"]["name"] == "catalog_run"
    assert captured["run"] is None


def test_model_catalog_page_import_smoke() -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    assert callable(model_catalog.render)


def test_model_catalog_pending_selection_is_applied_before_widget(monkeypatch) -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    session_state: dict[str, object] = {
        model_catalog.MODEL_CATALOG_SELECTED_KEY: "historical_bootstrap",
        model_catalog.MODEL_CATALOG_PENDING_SELECTED_KEY: "debug_entrypoint",
    }
    monkeypatch.setattr(model_catalog.st, "session_state", session_state)

    model_catalog._apply_pending_selection(["historical_bootstrap", "debug_entrypoint"])

    assert session_state[model_catalog.MODEL_CATALOG_SELECTED_KEY] == "debug_entrypoint"
    assert model_catalog.MODEL_CATALOG_PENDING_SELECTED_KEY not in session_state


def test_model_catalog_plugin_name_tracks_selected_plugin_without_overwriting_custom_name(monkeypatch) -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    session_state: dict[str, object] = {}
    monkeypatch.setattr(model_catalog.st, "session_state", session_state)

    model_catalog._sync_name_default(
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY,
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_DEFAULT_KEY,
        "plugin_alpha",
    )
    assert session_state[model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY] == "plugin_alpha"

    model_catalog._sync_name_default(
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY,
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_DEFAULT_KEY,
        "plugin_beta",
    )
    assert session_state[model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY] == "plugin_beta"

    session_state[model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY] = "custom_name"
    model_catalog._sync_name_default(
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY,
        model_catalog.MODEL_CATALOG_PLUGIN_NAME_DEFAULT_KEY,
        "plugin_gamma",
    )
    assert session_state[model_catalog.MODEL_CATALOG_PLUGIN_NAME_KEY] == "custom_name"


def test_model_catalog_simplified_parameter_rows_hide_varargs() -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    rows, accepts_varargs = model_catalog._simplified_parameter_rows(
        [
            {"name": "ridge", "parameter_type": "explicit", "value_type": "float", "default": 1e-6, "required": False},
            {"name": "**kwargs", "parameter_type": "vararg", "value_type": "json", "default": None, "required": False},
        ]
    )

    assert rows == [
        {"parameter": "ridge", "type": "float", "default": "1e-06", "required": False}
    ]
    assert accepts_varargs is True


def test_model_catalog_load_detail_rows_are_compact() -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    rows = model_catalog._load_detail_rows(
        {
            "reference": {
                "kind": "entrypoint",
                "value": "/tmp/demo_adapter.py:DemoAdapter",
            }
        },
        {
            "status": "importable",
            "target": "/tmp/demo_adapter.py:DemoAdapter",
            "version": "",
        },
    )

    assert rows == [
        {"field": "Reference kind", "value": "entrypoint"},
        {"field": "Reference", "value": "/tmp/demo_adapter.py:DemoAdapter"},
        {"field": "Status", "value": "importable"},
    ]


def test_model_catalog_prefers_scenario_model_symbols() -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    preferred = model_catalog._preferred_symbol_name(
        {
            "scenario_model_classes": ["RepoModel"],
            "symbols": [
                {"name": "build_model", "kind": "function"},
                {"name": "RepoModel", "kind": "class", "extends_scenario_model": True},
            ],
        }
    )

    assert preferred == "RepoModel"


def test_model_catalog_uploaded_entrypoint_file_is_persisted(monkeypatch, tmp_path: Path) -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    class _UploadedFile:
        name = "demo_adapter.py"
        size = 24
        file_id = "upload-1"

        @staticmethod
        def getvalue() -> bytes:
            return b"class DemoAdapter:\n    pass\n"

    session_state: dict[str, object] = {}
    monkeypatch.setattr(model_catalog.st, "session_state", session_state)

    def fake_store_uploaded_entrypoint_file(*, filename: str, content: bytes):
        assert filename == "demo_adapter.py"
        assert content == b"class DemoAdapter:\n    pass\n"
        return tmp_path / "demo_adapter_saved.py"

    monkeypatch.setattr(
        model_catalog,
        "store_uploaded_entrypoint_file",
        fake_store_uploaded_entrypoint_file,
    )

    saved = model_catalog._handle_uploaded_entrypoint_file(_UploadedFile())

    assert saved == str(tmp_path / "demo_adapter_saved.py")
    assert session_state[model_catalog.MODEL_CATALOG_ENTRYPOINT_UPLOADED_PATH_KEY] == saved


def test_model_catalog_repo_root_key_exists() -> None:
    import ts_benchmark.ui.pages.model_catalog as model_catalog

    assert model_catalog.MODEL_CATALOG_ENTRYPOINT_REPO_ROOT_KEY == "model_catalog.add.entrypoint_repo_root"


def test_discover_plugins_df_normalizes_entry_point_source(monkeypatch) -> None:
    from ts_benchmark.ui.services import environment

    monkeypatch.setattr(environment, "clear_plugin_caches", lambda: None)
    monkeypatch.setattr(
        environment,
        "list_model_plugins",
        lambda: {
            "demo_plugin": {
                "source": "entry_point",
                "manifest": {
                    "display_name": "Demo Plugin",
                    "capabilities": {},
                },
                "package_version": "1.2.3",
            }
        },
    )

    frame = environment.discover_plugins_df()

    assert frame.iloc[0]["source"] == "plugin"


def test_data_studio_frequency_labels_are_human_readable() -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    assert data_studio._format_frequency("B") == "Business day (B)"
    assert data_studio._format_frequency("M") == "Month end (M)"


def test_data_studio_tabular_value_mode_maps_legacy_and_explicit_values() -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    assert data_studio._tabular_value_mode_from_config({"value_type": "price"}, {}) == "price"
    assert data_studio._tabular_value_mode_from_config({"value_type": "log_price"}, {}) == "log_price"
    assert data_studio._tabular_value_mode_from_config({"value_type": "returns", "return_kind": "log"}, {}) == "log_return"
    assert data_studio._tabular_value_mode_from_config({}, {"target_kind": "prices", "return_kind": "simple"}) == "price"


def test_data_studio_open_navigation_uses_pending_state(monkeypatch) -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    session_state: dict[str, object] = {
        data_studio.DATA_STUDIO_SECTION_KEY: "Catalog",
        data_studio.DATA_STUDIO_DATASET_VIEW_KEY: "Statistics",
        "data_studio.dataset.name": "stale",
    }
    synced: dict[str, object] = {}

    monkeypatch.setattr(data_studio.st, "session_state", session_state)
    monkeypatch.setattr(data_studio, "_sync_dataset_to_config", lambda dataset: synced.setdefault("dataset", dataset))

    data_studio._open_dataset_definition({"name": "demo"})

    assert session_state[data_studio.DATA_STUDIO_PENDING_SECTION_KEY] == "Dataset"
    assert session_state[data_studio.DATA_STUDIO_PENDING_DATASET_VIEW_KEY] == "Definition"
    assert session_state[data_studio.DATA_STUDIO_SECTION_KEY] == "Catalog"
    assert "data_studio.dataset.name" not in session_state
    assert synced["dataset"] == {"name": "demo"}


def test_data_studio_uploaded_file_sets_path_widget_state(monkeypatch) -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    class _Uploaded:
        name = "demo.csv"
        size = 12
        file_id = "abc123"

        def getvalue(self) -> bytes:
            return b"date,a\n2024-01-01,1\n"

    session_state: dict[str, object] = {"data_studio.csv.path": ""}
    monkeypatch.setattr(data_studio.st, "session_state", session_state)
    monkeypatch.setattr(
        data_studio,
        "store_uploaded_dataset_file",
        lambda *, filename, content: Path("/tmp/uploaded_demo.csv"),
    )

    saved_path = data_studio._handle_uploaded_tabular_file("csv", _Uploaded())

    assert saved_path == "/tmp/uploaded_demo.csv"
    assert session_state["data_studio.csv.path"] == "/tmp/uploaded_demo.csv"
    assert session_state["data_studio.csv.uploaded_path"] == "/tmp/uploaded_demo.csv"


def test_data_studio_series_frame_prefers_targets_and_excludes_regime() -> None:
    import pandas as pd
    import ts_benchmark.ui.pages.data_studio as data_studio

    preview = pd.DataFrame(
        {
            "regime": [0, 1, 1],
            "asset_a": [0.1, 0.2, 0.3],
            "asset_b": [0.4, 0.5, 0.6],
        }
    )

    targeted = data_studio._series_frame_for_statistics(
        {"schema": {"target_columns": ["asset_b"]}},
        preview,
    )
    fallback = data_studio._series_frame_for_statistics(
        {"schema": {"target_columns": []}},
        preview,
    )

    assert list(targeted.columns) == ["asset_b"]
    assert list(fallback.columns) == ["asset_a", "asset_b"]


def test_data_studio_configured_tabular_preview_respects_long_mapping() -> None:
    import pandas as pd
    import ts_benchmark.ui.pages.data_studio as data_studio

    raw_preview = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "price": [100.0, 200.0, 110.0, 210.0],
            "volume": [10, 20, 30, 40],
        }
    )

    mapped = data_studio._configured_tabular_preview(
        {
            "provider": {"config": {"value_column": "price"}},
            "schema": {
                "layout": "long",
                "time_column": "date",
                "series_id_columns": ["ticker"],
            },
        },
        raw_preview,
    )

    assert list(mapped.columns) == ["date", "AAPL", "MSFT"]


def test_data_studio_configured_tabular_preview_respects_selected_wide_columns() -> None:
    import pandas as pd
    import ts_benchmark.ui.pages.data_studio as data_studio

    raw_preview = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "asset_a": [1.0, 2.0],
            "asset_b": [3.0, 4.0],
            "feature_x": [5.0, 6.0],
        }
    )

    mapped = data_studio._configured_tabular_preview(
        {
            "provider": {"config": {}},
            "schema": {
                "layout": "wide",
                "time_column": "date",
                "target_columns": ["asset_b"],
                "feature_columns": [],
                "static_feature_columns": [],
            },
        },
        raw_preview,
    )

    assert list(mapped.columns) == ["date", "asset_b"]


def test_data_studio_configured_tabular_preview_from_source_reads_long_layout_across_series(tmp_path: Path) -> None:
    import ts_benchmark.ui.pages.data_studio as data_studio

    path = tmp_path / "long_sorted_by_ticker.csv"
    path.write_text(
        "date,ticker,price\n"
        "2024-01-01,AAPL,100.0\n"
        "2024-01-02,AAPL,110.0\n"
        "2024-01-01,MSFT,200.0\n"
        "2024-01-02,MSFT,210.0\n",
        encoding="utf-8",
    )

    preview = data_studio._configured_tabular_preview_from_source(
        {
            "provider": {
                "kind": "csv",
                "config": {"path": str(path), "value_column": "price"},
            },
            "schema": {
                "layout": "long",
                "time_column": "date",
                "series_id_columns": ["ticker"],
            },
        },
        max_rows=10,
    )

    assert list(preview.columns) == ["date", "AAPL", "MSFT"]
