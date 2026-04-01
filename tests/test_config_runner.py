from __future__ import annotations

import json
import sys
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark import dump_benchmark_config, load_benchmark_config
from ts_benchmark.cli.main import main as cli_main
from ts_benchmark.paths import SAMPLE_DATA_DIR
from ts_benchmark.tracking import (
    get_mlflow_run_payload,
    list_mlflow_artifacts,
    list_mlflow_experiments,
    search_mlflow_runs,
)
from ts_benchmark.model.catalog.plugins import ModelPluginManifest
from ts_benchmark.preprocessing import build_pipeline_from_config
from ts_benchmark.run import run_benchmark_from_config
from ts_benchmark.run.model_runtime import enforce_manifest_preprocessing_contract
from ts_benchmark.ui.services.runs import load_run_artifacts

def _small_synthetic_config(
    *,
    models: list[dict[str, object]],
    seed: int = 7,
    generation_mode: str = "forecast",
    context_length: int | None = None,
    unconditional_train_data_mode: str | None = None,
    unconditional_n_train_paths: int | None = None,
    unconditional_n_eval_paths: int | None = None,
    device: str | None = None,
    scheduler: str = "auto",
    output_dir: str | None = None,
) -> dict[str, object]:
    execution: dict[str, object] = {"scheduler": scheduler}
    if device is not None:
        execution["device"] = device
    resolved_context_length = 6 if context_length is None and generation_mode == "forecast" else 0 if context_length is None else int(context_length)
    resolved_unconditional_train_data_mode = unconditional_train_data_mode
    if generation_mode == "unconditional" and resolved_unconditional_train_data_mode is None:
        resolved_unconditional_train_data_mode = "windowed_path"
    resolved_horizon = 2
    resolved_unconditional_n_train_paths = (
        6
        if generation_mode == "unconditional"
        and resolved_unconditional_train_data_mode == "path_dataset"
        and unconditional_n_train_paths is None
        else unconditional_n_train_paths
    )
    resolved_unconditional_n_eval_paths = (
        4
        if generation_mode == "unconditional"
        and resolved_unconditional_train_data_mode == "path_dataset"
        and unconditional_n_eval_paths is None
        else unconditional_n_eval_paths
    )
    resolved_protocol: dict[str, object]
    if generation_mode == "forecast":
        resolved_protocol = {
            "kind": "forecast",
            "horizon": resolved_horizon,
            "n_model_scenarios": 6,
            "n_reference_scenarios": 10,
            "forecast": {
                "train_size": 60,
                "test_size": 12,
                "context_length": resolved_context_length,
                "eval_stride": 6,
                "train_stride": 1,
            },
        }
    elif resolved_unconditional_train_data_mode == "path_dataset":
        resolved_protocol = {
            "kind": "unconditional_path_dataset",
            "horizon": resolved_horizon,
            "n_model_scenarios": 6,
            "n_reference_scenarios": 10,
            "unconditional_path_dataset": {
                "n_train_paths": resolved_unconditional_n_train_paths,
                "n_realized_paths": resolved_unconditional_n_eval_paths,
            },
        }
    else:
        resolved_protocol = {
            "kind": "unconditional_windowed",
            "horizon": resolved_horizon,
            "n_model_scenarios": 6,
            "n_reference_scenarios": 10,
            "unconditional_windowed": {
                "train_size": 60,
                "test_size": 12,
                "eval_stride": 6,
                "train_stride": 1,
            },
        }

    return {
        "version": "1.0",
        "benchmark": {
            "name": f"small_synthetic_{seed}",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": seed},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": resolved_protocol,
            "metrics": [{"name": "crps"}],
            "models": models,
        },
        "run": {
            "seed": seed,
            "execution": execution,
            "output": {
                "keep_scenarios": False,
                "output_dir": output_dir,
                "save_scenarios": False,
                "save_model_info": True,
                "save_summary": True,
            },
        },
    }


def _write_external_structural_model(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass
            from pathlib import Path
            from types import SimpleNamespace

            import numpy as np


            @dataclass
            class TinyGenerator:
                shift: float = 0.0

                def capabilities(self):
                    return SimpleNamespace(
                        supported_modes={"forecast"},
                        supports_multivariate_targets=True,
                        supports_known_covariates=False,
                        supports_observed_covariates=False,
                        supports_static_covariates=False,
                        supports_constraints=False,
                    )

                def sample(self, request):
                    values = np.asarray(request.series.values, dtype=float)
                    horizon = int(request.task.horizon or 1)
                    num_samples = int(request.num_samples)
                    last = values[-1:, :] + self.shift
                    tiled = np.repeat(last, horizon, axis=0)
                    samples = np.repeat(tiled[None, :, :], num_samples, axis=0)
                    return SimpleNamespace(
                        samples=samples,
                        diagnostics={"source": "temp_external_structural_model"},
                    )

                def save(self, path):
                    Path(path).write_text("tiny-generator", encoding="utf-8")


            @dataclass
            class TinyTrainer:
                shift: float = 0.0

                def fit(self, train, *, schema, task, valid=None, runtime=None):
                    del train, schema, task, valid, runtime
                    return TinyGenerator(shift=float(self.shift)), SimpleNamespace(
                        train_metrics={"fit_called": 1.0, "shift": float(self.shift)}
                    )


            def build_estimator(**params):
                return TinyTrainer(**params)
            """
        ),
        encoding="utf-8",
    )


def test_load_config_and_pipeline_smoke() -> None:
    config = load_benchmark_config(
        _small_synthetic_config(
            seed=13,
            models=[
                {
                    "name": "historical_bootstrap",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        )
    )
    assert config.version == "1.0"
    assert config.dataset.provider.kind == "synthetic"
    assert config.dataset.provider.generator == "regime_switching_factor_sv"
    assert config.protocol.train_stride == 1
    pipeline = build_pipeline_from_config(
        "std",
        [{"type": "standard_scale", "params": {"with_mean": True, "with_std": True}}],
    )
    x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=float)
    pipeline.fit(x)
    y = pipeline.transform(x)
    x_back = pipeline.inverse_transform(y)
    assert np.allclose(x, x_back)


def test_dump_benchmark_config_round_trip() -> None:
    config = load_benchmark_config(
        {
            "version": "1.0",
            "benchmark": {
                "name": "round_trip_smoke",
                "dataset": {
                    "provider": {
                        "kind": "synthetic",
                        "config": {
                            "generator": "regime_switching_factor_sv",
                            "params": {"n_assets": 4, "seed": 23},
                        },
                    },
                    "schema": {"layout": "tensor", "frequency": "B"},
                    "semantics": {},
                    "metadata": {},
                },
                "protocol": {
                    "kind": "forecast",
                    "horizon": 3,
                    "n_model_scenarios": 8,
                    "n_reference_scenarios": 16,
                    "forecast": {
                        "train_size": 100,
                        "test_size": 24,
                        "context_length": 8,
                        "eval_stride": 6,
                        "train_stride": 1,
                    },
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
                "seed": 23,
                "execution": {"device": "cpu", "scheduler": "auto"},
                "output": {"keep_scenarios": False, "output_dir": None},
            },
        }
    )
    payload = dump_benchmark_config(config)

    assert payload["version"] == config.version
    assert payload["benchmark"]["name"] == config.name
    assert payload["run"]["execution"]["device"] == config.run.device
    assert payload["run"]["execution"]["scheduler"] == config.run.scheduler
    assert payload["run"]["execution"]["model_execution"]["mode"] == config.run.model_execution.mode
    assert "device" not in payload["run"]
    assert "scheduler" not in payload["run"]

    round_tripped = load_benchmark_config(payload)
    assert round_tripped.name == config.name
    assert round_tripped.dataset.provider.kind == config.dataset.provider.kind
    assert round_tripped.run.device == config.run.device
    assert round_tripped.run.scheduler == config.run.scheduler
    assert round_tripped.run.model_execution.mode == config.run.model_execution.mode


def test_structured_dataset_config_smoke() -> None:
    config = load_benchmark_config(
        {
            "version": "1.0",
            "benchmark": {
                "name": "structured_dataset_smoke",
                "description": "Structured dataset object smoke test",
                "dataset": {
                    "name": "structured_synth",
                    "description": "Structured dataset object smoke test",
                    "provider": {
                        "kind": "synthetic",
                        "config": {
                            "generator": "regime_switching_factor_sv",
                            "params": {"n_assets": 3, "seed": 13},
                        },
                    },
                    "schema": {
                        "layout": "tensor",
                        "frequency": "B",
                    },
                    "semantics": {"domain": "time_series", "target_kind": "returns"},
                    "metadata": {"provenance": "unit-test"},
                },
                "protocol": {
                    "kind": "forecast",
                    "horizon": 3,
                    "n_model_scenarios": 8,
                    "n_reference_scenarios": 16,
                    "forecast": {
                        "train_size": 90,
                        "test_size": 24,
                        "context_length": 8,
                        "eval_stride": 8,
                        "train_stride": 1,
                    },
                },
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
                "execution": {"scheduler": "auto"},
                "output": {},
            },
        }
    )
    assert config.dataset.name == "structured_synth"
    assert config.dataset.layout == "tensor"
    assert config.dataset.source == "synthetic"
    assert config.dataset.generator == "regime_switching_factor_sv"


def test_config_runner_smoke(tmp_path: Path) -> None:
    artifacts = run_benchmark_from_config(
        _small_synthetic_config(
            seed=17,
            output_dir=str(tmp_path / "run"),
            models=[
                {
                    "name": "historical_bootstrap",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        )
    )
    metrics = artifacts.results.metrics_frame()
    assert not metrics.empty
    assert artifacts.output_dir is not None
    assert (artifacts.output_dir / "metrics.csv").exists()
    saved_config = json.loads((artifacts.output_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    assert saved_config["benchmark"]["name"] == artifacts.config.name
    assert saved_config["run"]["execution"]["device"] == artifacts.config.run.device
    assert saved_config["run"]["execution"]["scheduler"] == artifacts.config.run.scheduler
    assert saved_config["run"]["execution"]["model_execution"]["mode"] == artifacts.config.run.model_execution.mode
    assert "device" not in saved_config["run"]
    assert "scheduler" not in saved_config["run"]
    assert "historical_bootstrap" in metrics.index
    model_results = {result.model_name: result for result in artifacts.results.model_results}
    assert model_results["historical_bootstrap"].plugin_info is not None
    assert model_results["historical_bootstrap"].plugin_info.manifest is not None
    assert model_results["historical_bootstrap"].plugin_info.manifest.display_name == "Historical bootstrap"
    saved = pd.read_csv(artifacts.output_dir / "metrics.csv")
    assert "dataset_name" in saved.columns
    assert saved["dataset_source"].iloc[0] == "synthetic"
    assert "train_stride" in saved.columns
    loaded_artifacts = load_run_artifacts(artifacts.output_dir)
    assert loaded_artifacts["dataset"] is not None
    assert loaded_artifacts["config"] is not None


def test_cli_validate_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "validate_config.json"
    config = _small_synthetic_config(
        seed=23,
        models=[
            {
                "name": "validate_debug_model",
                "reference": {
                    "kind": "entrypoint",
                    "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                },
                "params": {"scale": 1.0},
                "pipeline": {"name": "raw", "steps": []},
            }
        ],
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ts-benchmark",
            "validate",
            str(config_path),
        ],
    )

    assert cli_main() == 0
    assert f"Config is valid: {config_path}" in capsys.readouterr().out


def test_cli_run_device_override_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "cli_config.json"
    output_dir = tmp_path / "cli_run"
    config = _small_synthetic_config(
        seed=29,
        models=[
            {
                "name": "cli_debug_model",
                "reference": {
                    "kind": "entrypoint",
                    "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                },
                "params": {"scale": 1.0},
                "pipeline": {"name": "raw", "steps": []},
            }
        ],
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ts-benchmark",
            "run",
            str(config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert cli_main() == 0
    run_payload = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run_payload["device"] == "cpu"
    assert run_payload["resolved_execution"]["requested_device"] == "cpu"
    assert "Run metadata:" in capsys.readouterr().out


def test_model_parallel_execution_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "parallel_run"
    config = _small_synthetic_config(
        seed=31,
        device="cpu,not_a_real_device",
        scheduler="model_parallel",
        output_dir=str(output_dir),
        models=[
            {
                "name": "hist_a",
                "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                "params": {"block_size": 2},
                "pipeline": {"name": "raw", "steps": []},
            },
            {
                "name": "hist_b",
                "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                "params": {"block_size": 3},
                "pipeline": {"name": "raw", "steps": []},
            },
        ],
    )

    artifacts = run_benchmark_from_config(config)
    assert artifacts.output_dir == output_dir.resolve()
    assert artifacts.run.status == "succeeded"
    assert artifacts.run.resolved_execution.execution_mode == "model_parallel"
    assert artifacts.run.resolved_execution.parallel_workers == 2
    assert artifacts.run.resolved_execution.assigned_device_map() == {
        "hist_a": "cpu",
        "hist_b": "not_a_real_device",
    }
    assert set(artifacts.results.metrics_frame().index) == {"hist_a", "hist_b"}

    model_results = {result.model_name: result for result in artifacts.results.model_results}
    assert model_results["hist_a"].execution is not None
    assert model_results["hist_a"].execution.assigned_device == "cpu"
    assert model_results["hist_b"].execution is not None
    assert model_results["hist_b"].execution.assigned_device == "not_a_real_device"
    assert model_results["hist_b"].execution.runtime_device == "not_a_real_device"
    assert (output_dir / "run.json").exists()


def test_unconditional_config_runner_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "unconditional_run"
    config = _small_synthetic_config(
        seed=37,
        generation_mode="unconditional",
        models=[
            {
                "name": "unconditional_debug_model",
                "reference": {
                    "kind": "entrypoint",
                    "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                },
                "params": {"scale": 1.0},
                "pipeline": {"name": "raw", "steps": []},
            }
        ],
        output_dir=str(output_dir),
    )

    artifacts = run_benchmark_from_config(config)
    metrics = artifacts.results.metrics_frame(include_metadata=True)
    assert artifacts.run.status == "succeeded"
    assert metrics.loc["unconditional_debug_model", "generation_mode"] == "unconditional"
    assert metrics.loc["unconditional_debug_model", "path_construction"] == "windowed_path"
    assert metrics.loc["unconditional_debug_model", "crps"] >= 0.0
    scenarios = artifacts.results.model_results[0].scenario_output
    assert scenarios is None


def test_unconditional_path_dataset_config_runner_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "unconditional_path_dataset_run"
    config = _small_synthetic_config(
        seed=39,
        generation_mode="unconditional",
        unconditional_train_data_mode="path_dataset",
        unconditional_n_train_paths=5,
        unconditional_n_eval_paths=3,
        models=[
            {
                "name": "unconditional_debug_model",
                "reference": {
                    "kind": "entrypoint",
                    "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                },
                "params": {"scale": 1.0},
                "pipeline": {"name": "raw", "steps": []},
            }
        ],
        output_dir=str(output_dir),
    )

    artifacts = run_benchmark_from_config(config)
    metrics = artifacts.results.metrics_frame(include_metadata=True)
    assert artifacts.run.status == "succeeded"
    assert metrics.loc["unconditional_debug_model", "generation_mode"] == "unconditional"
    assert metrics.loc["unconditional_debug_model", "path_construction"] == "path_dataset"
    assert metrics.loc["unconditional_debug_model", "n_train_paths"] == 5
    assert metrics.loc["unconditional_debug_model", "n_realized_paths"] == 3
    assert metrics.loc["unconditional_debug_model", "crps"] >= 0.0


def test_entrypoint_model_smoke() -> None:
    artifacts = run_benchmark_from_config(
        _small_synthetic_config(
            seed=41,
            models=[
                {
                    "name": "entrypoint_historical",
                    "reference": {
                        "kind": "entrypoint",
                        "value": "ts_benchmark.model.builtins.historical_bootstrap:HistoricalBootstrapModel",
                    },
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        )
    )
    assert "entrypoint_historical" in artifacts.results.metrics_frame().index


def test_duck_typed_external_entrypoint_model_smoke(tmp_path: Path) -> None:
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_structural_model(external_model_path)

    artifacts = run_benchmark_from_config(
        _small_synthetic_config(
            seed=41,
            models=[
                {
                    "name": "duck_typed_external",
                    "reference": {
                        "kind": "entrypoint",
                        "value": f"{external_model_path}:build_estimator",
                    },
                    "params": {"shift": 0.25},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        )
    )
    metrics = artifacts.results.metrics_frame()
    assert "duck_typed_external" in metrics.index
    model_results = {result.model_name: result for result in artifacts.results.model_results}
    info = model_results["duck_typed_external"].fitted_model_info
    assert info["adapter"] == "duck_typed_estimator"
    assert info["wrapped_estimator_class"] == "TinyTrainer"
    assert info["wrapped_generator_class"] == "TinyGenerator"


def test_file_entrypoint_model_smoke(tmp_path: Path) -> None:
    config = _small_synthetic_config(
        models=[
            {
                "name": "file_entrypoint_debug",
                "reference": {
                    "kind": "entrypoint",
                    "value": str(ROOT / "src" / "ts_benchmark" / "model" / "builtins" / "debug_smoke_model.py")
                    + ":DebugSmokeModel",
                },
                "params": {"scale": 1.5},
                "pipeline": {"name": "raw", "steps": []},
            }
        ],
        output_dir=str(tmp_path / "out"),
    )
    path = tmp_path / "file_entrypoint.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    artifacts = run_benchmark_from_config(path)

    assert "file_entrypoint_debug" in artifacts.results.metrics_frame().index


def test_subprocess_model_execution_smoke() -> None:
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "subprocess_model_execution_smoke",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 3, "seed": 19},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 3,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "forecast": {
                    "train_size": 90,
                    "test_size": 24,
                    "context_length": 8,
                    "eval_stride": 8,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "subprocess_historical",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                    "execution": {
                        "mode": "subprocess",
                        "venv": Path(sys.prefix).name,
                        "pythonpath": [str(ROOT / "src")],
                    }
                }
            ]
        },
        "run": {
            "seed": 19,
            "execution": {"device": "cpu", "scheduler": "auto"},
            "output": {"keep_scenarios": False, "output_dir": None},
        },
    }
    artifacts = run_benchmark_from_config(config)
    assert "subprocess_historical" in artifacts.results.metrics_frame().index
    model_results = {result.model_name: result for result in artifacts.results.model_results}
    info = model_results["subprocess_historical"].fitted_model_info
    assert info["execution_mode"] == "subprocess"
    assert Path(info["python_executable"]).exists()
    assert info["child_model_info"]["name"] == "historical_bootstrap"


def test_duck_typed_external_subprocess_model_smoke(tmp_path: Path) -> None:
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_structural_model(external_model_path)
    config = _small_synthetic_config(
        seed=47,
        models=[
            {
                "name": "duck_typed_external_subprocess",
                "reference": {
                    "kind": "entrypoint",
                    "value": f"{external_model_path}:build_estimator",
                },
                "params": {"shift": 0.5},
                "pipeline": {"name": "raw", "steps": []},
                "execution": {
                    "mode": "subprocess",
                    "venv": Path(sys.prefix).name,
                    "pythonpath": [str(ROOT / "src")],
                },
            }
        ],
    )

    artifacts = run_benchmark_from_config(config)
    metrics = artifacts.results.metrics_frame()
    assert "duck_typed_external_subprocess" in metrics.index
    model_results = {result.model_name: result for result in artifacts.results.model_results}
    info = model_results["duck_typed_external_subprocess"].fitted_model_info
    assert info["execution_mode"] == "subprocess"
    assert info["child_model_info"]["adapter"] == "duck_typed_estimator"
    assert info["child_model_info"]["wrapped_estimator_class"] == "TinyTrainer"
    assert info["child_model_info"]["wrapped_generator_class"] == "TinyGenerator"


def test_run_level_subprocess_execution_smoke() -> None:
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "run_level_subprocess_execution_smoke",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 3, "seed": 21},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 3,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "forecast": {
                    "train_size": 90,
                    "test_size": 24,
                    "context_length": 8,
                    "eval_stride": 8,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "subprocess_historical_run_level",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ]
        },
        "run": {
            "seed": 21,
            "execution": {
                "device": "cpu",
                "scheduler": "auto",
                "model_execution": {
                    "mode": "subprocess",
                    "venv": Path(sys.prefix).name,
                    "pythonpath": [str(ROOT / "src")],
                },
            },
            "output": {"keep_scenarios": False, "output_dir": None},
        },
    }
    artifacts = run_benchmark_from_config(config)
    assert "subprocess_historical_run_level" in artifacts.results.metrics_frame().index
    model_results = {result.model_name: result for result in artifacts.results.model_results}
    info = model_results["subprocess_historical_run_level"].fitted_model_info
    assert info["execution_mode"] == "subprocess"
    assert Path(info["python_executable"]).exists()


def test_csv_dataset_smoke() -> None:
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "csv_dataset_smoke",
            "dataset": {
                    "name": "demo_returns",
                    "provider": {
                        "kind": "csv",
                        "config": {
                            "path": str(SAMPLE_DATA_DIR / "demo_returns.csv"),
                        },
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
                "kind": "forecast",
                "horizon": 3,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 12,
                "forecast": {
                    "train_size": 80,
                    "test_size": 20,
                    "context_length": 8,
                    "eval_stride": 5,
                    "train_stride": 1,
                },
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
    assert artifacts.dataset.name == "demo_returns"
    assert artifacts.dataset.source == "csv"
    assert not artifacts.dataset.has_reference_scenarios()
    assert set(artifacts.results.metadata) >= {"dataset_name", "dataset_source", "device"}
    assert "crps" in artifacts.results.metrics_frame().columns
    assert "volatility_error" not in artifacts.results.metrics_frame().columns


def test_loader_rejects_task_fields_inside_model_params() -> None:
    bad = {
        "version": "1.0",
        "benchmark": {
            "name": "loader_rejects_task_fields_inside_model_params",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 11},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 2,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "forecast": {
                    "train_size": 100,
                    "test_size": 20,
                    "context_length": 8,
                    "eval_stride": 5,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "bad_historical_bootstrap",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"context_length": 8, "block_size": 2},
                    "pipeline": {"name": "raw", "steps": []}
                }
            ]
        },
        "run": {
            "execution": {"scheduler": "auto"},
            "output": {},
        },
    }
    with pytest.raises(ValueError, match="benchmark-owned protocol fields"):
        load_benchmark_config(bad)


def test_manifest_required_pipeline_enforced() -> None:
    with pytest.raises(ValueError, match="requires preprocessing pipeline 'raw'"):
        enforce_manifest_preprocessing_contract(
            model_name="strict_model",
            configured_pipeline="standardized",
            manifest=ModelPluginManifest(name="strict_model", required_pipeline="raw"),
        )


def test_manifest_default_pipeline_is_not_enforced() -> None:
    enforce_manifest_preprocessing_contract(
        model_name="advisory_model",
        configured_pipeline="standardized",
        manifest=ModelPluginManifest(name="advisory_model", default_pipeline="raw"),
    )


def test_loader_rejects_invalid_execution_block() -> None:
    bad = {
        "version": "1.0",
        "benchmark": {
            "name": "loader_rejects_invalid_execution_block",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 11},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 2,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 16,
                "forecast": {
                    "train_size": 100,
                    "test_size": 20,
                    "context_length": 8,
                    "eval_stride": 5,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "bad_execution",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                    "execution": {"mode": "inprocess", "venv": "mxnet-gluonts"}
                }
            ]
        },
        "run": {
            "execution": {"scheduler": "auto"},
            "output": {},
        },
    }
    with pytest.raises(ValueError, match="mode='subprocess'"):
        load_benchmark_config(bad)


def test_diagnostics_outputs_smoke(tmp_path: Path) -> None:
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "diagnostics_outputs_smoke",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 17},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 3,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 12,
                "forecast": {
                    "train_size": 100,
                    "test_size": 24,
                    "context_length": 8,
                    "eval_stride": 8,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "debug_model",
                    "reference": {
                        "kind": "entrypoint",
                        "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel"
                    },
                    "params": {"scale": 1.0},
                    "pipeline": {"name": "raw", "steps": []}
                }
            ]
        },
        "run": {
            "seed": 17,
            "execution": {"device": "cpu", "scheduler": "auto"},
            "output": {"keep_scenarios": False, "output_dir": str(tmp_path / "diagnostics_run"), "save_scenarios": False, "save_model_info": True, "save_summary": True},
        },
        "diagnostics": {
            "save_model_debug_artifacts": True,
            "save_distribution_summary": True,
            "save_per_window_metrics": True,
            "functional_smoke": {
                "enabled": True,
                "mean_abs_error_max": 1.0,
                "std_ratio_min": 0.0,
                "std_ratio_max": 10.0,
                "crps_max": 1.0,
                "energy_score_max": 1.0,
                "cross_correlation_error_max": 1.0,
            },
        },
    }

    artifacts = run_benchmark_from_config(config)
    assert artifacts.results.diagnostics is not None
    assert artifacts.results.diagnostics.distribution_summary is not None
    assert artifacts.results.diagnostics.distribution_summary_by_asset is not None
    assert artifacts.results.diagnostics.per_window_metrics is not None
    assert artifacts.results.diagnostics.functional_smoke_summary is not None
    assert artifacts.results.diagnostics.functional_smoke_checks is not None
    assert artifacts.results.diagnostics.model_debug_artifacts["debug_model"]["wrapped_debug_artifacts"]["fit_log"]
    assert artifacts.results.scenario_outputs() == {}

    diagnostics_dir = artifacts.output_dir / "diagnostics"
    assert (diagnostics_dir / "distribution_summary.csv").exists()
    assert (diagnostics_dir / "distribution_summary_by_asset.csv").exists()
    assert (diagnostics_dir / "per_window_metrics.csv").exists()
    assert (diagnostics_dir / "functional_smoke_summary.csv").exists()
    assert (diagnostics_dir / "functional_smoke_checks.csv").exists()
    debug_json = diagnostics_dir / "model_debug_artifacts" / "debug_model.json"
    assert debug_json.exists()
    payload = json.loads(debug_json.read_text(encoding="utf-8"))
    assert payload["wrapped_debug_artifacts"]["fit_log"]


def test_mlflow_tracking_smoke(tmp_path: Path) -> None:
    tracking_uri = f"file://{(tmp_path / 'mlruns').resolve()}"
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "mlflow_tracking_smoke",
            "description": "mlflow smoke",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 23},
                    },
                },
                "schema": {"layout": "tensor", "frequency": "B"},
                "semantics": {},
                "metadata": {},
            },
            "protocol": {
                "kind": "forecast",
                "horizon": 3,
                "n_model_scenarios": 8,
                "n_reference_scenarios": 12,
                "forecast": {
                    "train_size": 80,
                    "test_size": 20,
                    "context_length": 8,
                    "eval_stride": 5,
                    "train_stride": 1,
                },
            },
            "models": [
                {
                    "name": "historical_bootstrap",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []}
                }
            ]
        },
        "run": {
            "seed": 23,
            "execution": {"device": "cpu", "scheduler": "auto"},
            "tracking": {
                "mlflow": {
                    "enabled": True,
                    "tracking_uri": tracking_uri,
                    "experiment_name": "eqbench-pytest",
                    "run_name": "mlflow-smoke",
                    "log_artifacts": True,
                    "log_model_info": True,
                    "log_diagnostics": False,
                }
            },
            "output": {
                "keep_scenarios": False,
                "output_dir": str(tmp_path / "mlflow_run"),
                "save_scenarios": False,
                "save_model_info": True,
                "save_summary": True,
            },
        },
    }

    artifacts = run_benchmark_from_config(config)
    assert artifacts.run.tracking_result is not None
    assert artifacts.run.tracking_result.backend == "mlflow"
    assert artifacts.run.tracking_result.experiment_name == "eqbench-pytest"
    assert (artifacts.output_dir / "run.json").exists()

    experiments = list_mlflow_experiments(tracking_uri=tracking_uri)
    assert "eqbench-pytest" in experiments["name"].tolist()

    runs = search_mlflow_runs(
        tracking_uri=tracking_uri,
        experiment_ids=[artifacts.run.tracking_result.experiment_id],
        max_results=20,
    )
    assert artifacts.run.tracking_result.run_id in runs["run_id"].tolist()

    payload = get_mlflow_run_payload(
        tracking_uri=tracking_uri,
        run_id=artifacts.run.tracking_result.run_id,
    )
    assert payload["info"]["run_id"] == artifacts.run.tracking_result.run_id
    assert payload["tags"]["benchmark.framework"] == "ts-benchmark"

    artifact_rows = list_mlflow_artifacts(
        tracking_uri=tracking_uri,
        run_id=artifacts.run.tracking_result.run_id,
        artifact_path="benchmark_outputs",
    )
    assert "benchmark_outputs/metrics.csv" in artifact_rows["path"].tolist()
