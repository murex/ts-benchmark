from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ts_benchmark.notebook as notebook_module
from ts_benchmark.notebook import (
    BenchmarkConfig,
    DatasetConfig,
    DebugSmokeModelParams,
    DiagnosticsConfig,
    ForecastProtocol,
    FunctionalSmokeConfig,
    JsonObject,
    ModelConfig,
    ModelReferenceConfig,
    OutputConfig,
    PipelineConfig,
    RegimeSwitchingFactorSVConfig,
    RunConfig,
    SyntheticDatasetProviderConfig,
    catalog_model,
    csv_dataset,
    dataset_frame,
    default_metric_configs,
    entrypoint_model,
    list_models,
    load_run,
    model_info,
    model_parameter_schema,
    provision_adapter_venv,
    run_benchmark,
    save_benchmark_definition,
    save_dataset_definition,
    save_model_to_catalog,
    synthetic_dataset,
)
from ts_benchmark.ui.services.runs import benchmark_results_dir_for_path, materialize_benchmark_results


def test_notebook_facade_exposes_typed_surfaces_not_generic_fallbacks() -> None:
    assert "DatasetProviderConfig" not in notebook_module.__all__
    assert "PipelineStepConfig" not in notebook_module.__all__


def _notebook_smoke_config(*, output_dir: Path) -> dict[str, object]:
    return {
        "version": "1.0",
        "benchmark": {
            "name": "notebook_smoke",
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
            "metrics": [{"name": "crps"}],
            "models": [
                {
                    "name": "debug_model",
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
                "output_dir": str(output_dir),
                "save_scenarios": False,
                "save_model_info": False,
                "save_summary": False,
            },
        },
        "diagnostics": {
            "save_model_debug_artifacts": False,
            "save_distribution_summary": False,
            "save_per_window_metrics": False,
            "functional_smoke": {
                "enabled": False,
            },
        },
    }


def _write_external_entrypoint_model(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass
            from types import SimpleNamespace

            import numpy as np


            @dataclass
            class TinyGenerator:
                target_dim: int
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
                        diagnostics={"source": "temp_external_entrypoint"},
                    )


            @dataclass
            class TinyTrainer:
                shift: float = 0.0

                def fit(self, train, *, schema, task, valid=None, runtime=None):
                    return TinyGenerator(target_dim=int(schema.target_dim), shift=float(self.shift)), SimpleNamespace(
                        train_metrics={"fit_called": 1.0, "shift": float(self.shift)}
                    )


            def build_estimator(**params):
                return TinyTrainer(**params)
            """
        ),
        encoding="utf-8",
    )


def _write_returns_csv(path: Path, *, n_rows: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n_rows, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "asset_a": np.linspace(-0.02, 0.03, n_rows),
            "asset_b": np.linspace(0.01, -0.015, n_rows),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_notebook_run_overrides_requested_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "notebook_run"
    run = run_benchmark(
        _notebook_smoke_config(output_dir=output_dir),
        include=["scenarios", "diagnostics"],
    )

    metrics = run.metrics()
    overview = run.model_overview()
    scenarios = run.scenarios("debug_model")
    dist = run.distribution_summary()
    per_window = run.per_window_metrics(model_name="debug_model")
    smoke_summary = run.functional_smoke_summary(model_name="debug_model")
    smoke_checks = run.functional_smoke_checks(model_name="debug_model")
    debug_artifacts = run.model_debug_artifacts("debug_model")

    assert not metrics.empty
    assert "crps" in metrics.columns
    assert not overview.empty
    assert scenarios.shape[0] > 0
    assert not dist.empty
    assert not per_window.empty
    assert not smoke_summary.empty
    assert not smoke_checks.empty
    assert debug_artifacts["wrapped_debug_artifacts"]["fit_log"]

    assert run.output_dir == output_dir.resolve()
    assert (output_dir / "scenarios.npz").exists()
    assert (output_dir / "model_results.json").exists()
    assert (output_dir / "summary.json").exists()
    diagnostics_dir = output_dir / "diagnostics"
    assert (diagnostics_dir / "distribution_summary.csv").exists()
    assert (diagnostics_dir / "distribution_summary_by_asset.csv").exists()
    assert (diagnostics_dir / "per_window_metrics.csv").exists()
    assert (diagnostics_dir / "functional_smoke_summary.csv").exists()
    assert (diagnostics_dir / "functional_smoke_checks.csv").exists()
    assert (diagnostics_dir / "model_debug_artifacts" / "debug_model.json").exists()


def test_notebook_run_accepts_typed_benchmark_config_from_notebook_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "typed_notebook_run"
    crps = next(metric for metric in default_metric_configs() if metric.name == "crps")

    config = BenchmarkConfig(
        version="1.0",
        name="typed_notebook_smoke",
        dataset=DatasetConfig(
            provider=SyntheticDatasetProviderConfig(
                generator="regime_switching_factor_sv",
                params=RegimeSwitchingFactorSVConfig(n_assets=2, seed=19),
            ),
            layout="tensor",
            frequency="B",
        ),
        protocol=ForecastProtocol(
            horizon=3,
            n_model_scenarios=8,
            n_reference_scenarios=12,
            context_length=8,
            train_size=80,
            test_size=20,
            eval_stride=5,
            train_stride=1,
        ),
        metrics=[crps],
        models=[
            ModelConfig(
                name="debug_model",
                reference=ModelReferenceConfig(
                    kind="entrypoint",
                    value="ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                ),
                params=DebugSmokeModelParams(scale=1.0),
                pipeline=PipelineConfig(name="raw", steps=[]),
            )
        ],
        run=RunConfig(
            seed=19,
            device="cpu",
            scheduler="auto",
            output=OutputConfig(
                output_dir=str(output_dir),
                keep_scenarios=False,
                save_scenarios=False,
                save_model_info=False,
                save_summary=False,
            ),
        ),
        diagnostics=DiagnosticsConfig(
            save_model_debug_artifacts=False,
            save_distribution_summary=False,
            save_per_window_metrics=False,
            functional_smoke=FunctionalSmokeConfig(enabled=False),
        ),
    )

    run = run_benchmark(config)

    metrics = run.metrics()

    assert run.config()["benchmark"]["name"] == "typed_notebook_smoke"
    assert run.config()["benchmark"]["protocol"]["kind"] == "forecast"
    assert run.output_dir == output_dir.resolve()
    assert "debug_model" in metrics.index
    assert "crps" in metrics.columns


def test_dataset_frame_exposes_synthetic_generator_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset_frame_run"
    config = _notebook_smoke_config(output_dir=output_dir)

    view = dataset_frame(config)

    assert list(view.frame.columns) == ["Asset_01", "Asset_02"]
    assert view.frame.shape[0] == 100
    assert view.info["source"] == "synthetic"
    assert view.info["synthetic"]["generator"] == "regime_switching_factor_sv"
    assert view.info["synthetic"]["params"]["n_assets"] == 2
    assert view.info["synthetic"]["n_points_to_generate"] == 100


def test_notebook_can_list_models_and_parameter_schemas() -> None:
    models = list_models()
    info = model_info("historical_bootstrap")
    schema = model_parameter_schema("historical_bootstrap")

    assert "historical_bootstrap" in models
    assert info["name"] == "historical_bootstrap"
    assert info["source"] == "builtin"
    assert info["manifest"]["name"] == "historical_bootstrap"
    assert schema is not None
    field_names = [field["name"] for field in schema["fields"]]
    assert "block_size" in field_names


def test_dataset_frame_accepts_tabular_notebook_dataset_spec(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    source = _write_returns_csv(csv_path)
    spec = csv_dataset(
        csv_path,
        name="local_returns",
        time_column="date",
        target_columns=["asset_a", "asset_b"],
        frequency="B",
        semantics={"target_kind": "returns", "return_kind": "simple"},
    )

    view = dataset_frame(spec)

    assert list(view.frame.columns) == ["__timestamp__", "asset_a", "asset_b"]
    assert view.frame.shape == (100, 3)
    pd.testing.assert_series_equal(
        view.frame["asset_a"].reset_index(drop=True),
        source["asset_a"],
        check_names=False,
    )
    assert view.info["source"] == "csv"
    assert view.info["n_assets"] == 2
    assert view.info["provider"]["config"]["date_column"] == "date"


def test_dataset_frame_accepts_synthetic_notebook_dataset_spec() -> None:
    spec = synthetic_dataset(
        "regime_switching_factor_sv",
        params={"n_assets": 3, "seed": 23},
        n_points=50,
        name="synthetic_preview",
    )

    view = dataset_frame(spec)

    assert view.frame.shape == (50, 3)
    assert view.info["source"] == "synthetic"
    assert view.info["synthetic"]["generator"] == "regime_switching_factor_sv"
    assert view.info["synthetic"]["params"]["n_assets"] == 3
    assert view.info["synthetic"]["n_points_to_generate"] == 50


def test_dataset_frame_accepts_typed_synthetic_notebook_dataset_spec() -> None:
    spec = synthetic_dataset(
        "regime_switching_factor_sv",
        params=RegimeSwitchingFactorSVConfig(n_assets=4, seed=29),
        n_points=40,
        name="typed_synthetic_preview",
    )

    view = dataset_frame(spec)

    assert view.frame.shape == (40, 4)
    assert view.info["source"] == "synthetic"
    assert view.info["synthetic"]["params"]["n_assets"] == 4
    assert view.info["synthetic"]["params"]["seed"] == 29


def test_provision_adapter_venv_returns_subprocess_execution_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_root = tmp_path / "ts-benchmark"
    adapters_root = tmp_path / "official_adapters"
    benchmark_root.mkdir()
    adapters_root.mkdir()
    (adapters_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "ts-benchmark-official-adapters"
            version = "0.1.0"
            dependencies = [
              "numpy>=1.24",
              "ts-benchmark>=0.4.0",
            ]

            [project.optional-dependencies]
            timegrad = [
              "gluonts>=0.14",
              "pytorchts>=0.6,<0.7",
              "lightning>=2,<3",
              "diffusers>=0.24",
            ]
            """
        ),
        encoding="utf-8",
    )
    venv_dir = tmp_path / "venvs" / "timegrad"
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = True):
        commands.append(list(cmd))
        if "-m" in cmd and "venv" in cmd:
            python_exe = venv_dir / "bin" / "python"
            python_exe.parent.mkdir(parents=True, exist_ok=True)
            python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ts_benchmark.notebook.api.subprocess.run", fake_run)

    env = provision_adapter_venv(
        venv_dir,
        "pytorchts_timegrad",
        benchmark_root=benchmark_root,
        adapters_root=adapters_root,
        python=sys.executable,
    )

    assert env.adapter_name == "pytorchts_timegrad"
    assert env.extra_name == "timegrad"
    assert env.venv_dir == venv_dir.resolve()
    assert env.execution.mode == "subprocess"
    assert env.execution.venv == str(venv_dir.resolve())
    assert commands[0] == [str(Path(sys.executable).resolve()), "-m", "venv", str(venv_dir.resolve())]
    assert commands[1] == [str(venv_dir.resolve() / "bin" / "python"), "-m", "pip", "install", "-U", "pip"]
    assert commands[2] == [
        str(venv_dir.resolve() / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "-e",
        str(benchmark_root.resolve()),
        "--no-deps",
    ]
    assert commands[3] == [
        str(venv_dir.resolve() / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "-e",
        str(adapters_root.resolve()),
        "--no-deps",
    ]
    assert commands[4] == [
        str(venv_dir.resolve() / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "numpy>=1.24",
        "gluonts>=0.14",
        "pytorchts>=0.6,<0.7",
        "lightning>=2,<3",
        "diffusers>=0.24",
    ]


def test_provision_adapter_venv_reuses_existing_environment_without_reinstall(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv_dir = tmp_path / "venvs" / "timegrad"
    python_exe = venv_dir / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (venv_dir / ".ts_benchmark_adapter_env.json").write_text(
        json.dumps(
            {
                "adapter_name": "pytorchts_timegrad",
                "extra_name": "timegrad",
                "creator_python": str(Path(sys.executable).resolve()),
            }
        ),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when reusing an existing adapter venv")

    monkeypatch.setattr("ts_benchmark.notebook.api.subprocess.run", fail_run)

    def fail_validate(*args, **kwargs):
        raise AssertionError("validation should not run unless validate=True")

    monkeypatch.setattr("ts_benchmark.notebook.api._adapter_env_is_usable", fail_validate)

    env = provision_adapter_venv(
        venv_dir,
        "pytorchts_timegrad",
        benchmark_root=tmp_path / "ts-benchmark",
        adapters_root=tmp_path / "official_adapters",
    )

    assert env.adapter_name == "pytorchts_timegrad"
    assert env.extra_name == "timegrad"
    assert env.venv_dir == venv_dir.resolve()
    assert env.python_executable == python_exe.resolve()
    assert env.execution.mode == "subprocess"
    assert env.execution.venv == str(venv_dir.resolve())


def test_provision_adapter_venv_can_explicitly_validate_existing_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv_dir = tmp_path / "venvs" / "timegrad"
    python_exe = venv_dir / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (venv_dir / ".ts_benchmark_adapter_env.json").write_text(
        json.dumps(
            {
                "adapter_name": "pytorchts_timegrad",
                "extra_name": "timegrad",
                "creator_python": str(Path(sys.executable).resolve()),
            }
        ),
        encoding="utf-8",
    )
    validated = {"called": False}

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when validation succeeds")

    def validate_env(_python, _adapter):
        validated["called"] = True
        return True

    monkeypatch.setattr("ts_benchmark.notebook.api.subprocess.run", fail_run)
    monkeypatch.setattr("ts_benchmark.notebook.api._adapter_env_is_usable", validate_env)

    env = provision_adapter_venv(
        venv_dir,
        "pytorchts_timegrad",
        benchmark_root=tmp_path / "ts-benchmark",
        adapters_root=tmp_path / "official_adapters",
        validate=True,
    )

    assert validated["called"] is True
    assert env.adapter_name == "pytorchts_timegrad"
    assert env.extra_name == "timegrad"


def test_notebook_load_run_rehydrates_views(tmp_path: Path) -> None:
    output_dir = tmp_path / "saved_notebook_run"
    live = run_benchmark(
        _notebook_smoke_config(output_dir=output_dir),
        include=["scenarios", "diagnostics"],
    )
    loaded = load_run(output_dir)

    pd.testing.assert_frame_equal(live.metrics(), loaded.metrics())
    pd.testing.assert_frame_equal(
        live.per_window_metrics(model_name="debug_model"),
        loaded.per_window_metrics(model_name="debug_model"),
    )

    band = loaded.scenario_band("debug_model", evaluation_window=0, asset=0)
    report = loaded.debug_report("debug_model")
    compare = loaded.compare_metrics(live)
    dataset_view = loaded.dataset_frame()

    assert list(band.columns) == ["realized", "p05", "median", "p95"]
    assert dataset_view.frame.shape == (100, 2)
    assert dataset_view.info["synthetic"]["n_points_to_generate"] == 100
    assert "Benchmark" in report
    assert "Model Hyperparameters" in report
    assert "Generated Scenarios" in report
    assert "Model Logs" in report
    assert "crps_current" in compare.columns
    assert "crps_compare" in compare.columns


def test_notebook_run_can_inject_entrypoint_model_without_mutating_source_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "notebook_injected_model_run"
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_entrypoint_model(external_model_path)

    config = _notebook_smoke_config(output_dir=output_dir)
    config["run"]["execution"]["model_execution"] = {"mode": "subprocess", "python": sys.executable}
    config["benchmark"]["models"][0]["execution"] = {"mode": "inprocess"}
    original_models = list(config["benchmark"]["models"])

    run = run_benchmark(
        config,
        with_model=entrypoint_model(
            "notebook_external",
            f"{external_model_path}:build_estimator",
            shift=0.25,
        ),
    )

    assert len(config["benchmark"]["models"]) == len(original_models)
    assert [item["name"] for item in config["benchmark"]["models"]] == [item["name"] for item in original_models]

    effective_models = run.config()["benchmark"]["models"]
    assert [item["name"] for item in effective_models] == ["debug_model", "notebook_external"]
    injected = next(item for item in effective_models if item["name"] == "notebook_external")
    assert injected["reference"]["kind"] == "entrypoint"
    assert injected["reference"]["value"] == f"{external_model_path}:build_estimator"
    assert injected["params"]["shift"] == 0.25
    assert injected["execution"]["mode"] == "inprocess"

    metrics = run.metrics()
    assert "notebook_external" in metrics.index

    model_result = run.model_result("notebook_external")
    assert model_result["params"]["shift"] == 0.25
    assert model_result["execution"]["requested"]["mode"] == "inprocess"
    assert model_result["fitted_model_info"]["adapter"] == "duck_typed_estimator"


def test_notebook_injected_model_always_runs_while_official_results_are_reused(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmarks" / "notebook_smoke.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    previous_output_dir = tmp_path / "outputs" / "official_previous_run"
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_entrypoint_model(external_model_path)

    benchmark_config = _notebook_smoke_config(output_dir=previous_output_dir)
    benchmark_path.write_text(json.dumps(benchmark_config, indent=2), encoding="utf-8")

    baseline = run_benchmark(benchmark_path)
    assert baseline.model_names() == ["debug_model"]
    official_results_dir = materialize_benchmark_results(
        benchmark_path=benchmark_path,
        benchmark_config=baseline.config(),
        source_run_dir=baseline.output_dir,
        previous_results_dir=None,
    )
    assert official_results_dir == benchmark_results_dir_for_path(benchmark_path)

    notebook_output_dir = tmp_path / "outputs" / "notebook_merged_run"
    first = run_benchmark(
        benchmark_path,
        output_dir=notebook_output_dir,
        with_model=entrypoint_model(
            "notebook_external",
            f"{external_model_path}:build_estimator",
            shift=0.25,
        ),
    )
    second = run_benchmark(
        benchmark_path,
        output_dir=notebook_output_dir,
        with_model=entrypoint_model(
            "notebook_external",
            f"{external_model_path}:build_estimator",
            shift=0.75,
        ),
    )

    assert sorted(first.model_names()) == ["debug_model", "notebook_external"]
    assert sorted(second.model_names()) == ["debug_model", "notebook_external"]
    assert "notebook_external" not in load_run(official_results_dir).model_names()
    assert first.model_result("notebook_external")["fitted_model_info"]["fit_report"]["train_metrics"]["shift"] == 0.25
    assert second.model_result("notebook_external")["fitted_model_info"]["fit_report"]["train_metrics"]["shift"] == 0.75
    assert second.output_dir == notebook_output_dir.resolve()


def test_run_benchmark_can_override_dataset_with_notebook_dataset_spec(tmp_path: Path) -> None:
    output_dir = tmp_path / "notebook_tabular_run"
    csv_path = tmp_path / "returns.csv"
    _write_returns_csv(csv_path)
    spec = csv_dataset(
        csv_path,
        name="local_returns",
        time_column="date",
        target_columns=["asset_a", "asset_b"],
        frequency="B",
        semantics={"target_kind": "returns", "return_kind": "simple"},
    )

    run = run_benchmark(
        _notebook_smoke_config(output_dir=output_dir),
        with_dataset=spec,
    )

    assert run.config()["benchmark"]["dataset"]["provider"]["kind"] == "csv"
    assert run.config()["benchmark"]["dataset"]["provider"]["config"]["path"] == str(csv_path)
    assert run.summary()["dataset"]["resolved_source"] == "csv"
    assert run.model_names() == ["debug_model"]
    dataset_view = run.dataset_frame()
    assert dataset_view.info["source"] == "csv"
    assert dataset_view.frame.shape == (100, 2)


def test_run_benchmark_can_override_dataset_with_synthetic_notebook_dataset_spec(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "notebook_synthetic_run"
    spec = synthetic_dataset(
        "regime_switching_factor_sv",
        params={"n_assets": 3, "seed": 23},
        n_points=40,
        name="synthetic_override",
    )

    run = run_benchmark(
        _notebook_smoke_config(output_dir=output_dir),
        with_dataset=spec,
    )

    assert run.config()["benchmark"]["dataset"]["provider"]["kind"] == "synthetic"
    assert run.config()["benchmark"]["dataset"]["provider"]["config"]["generator"] == "regime_switching_factor_sv"
    assert run.config()["benchmark"]["dataset"]["provider"]["config"]["params"]["n_assets"] == 3
    assert run.summary()["dataset"]["resolved_source"] == "synthetic"
    dataset_view = run.dataset_frame()
    assert dataset_view.info["source"] == "synthetic"
    assert dataset_view.frame.shape == (100, 3)


def test_save_model_to_catalog_persists_notebook_model(tmp_path: Path) -> None:
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_entrypoint_model(external_model_path)
    model_spec = entrypoint_model(
        "catalog_willow_like_model",
        f"{external_model_path}:build_estimator",
        shift=0.5,
    )

    saved_path = save_model_to_catalog(model_spec, model_dir=tmp_path / "model_catalog")
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert payload["name"] == "catalog_willow_like_model"
    assert payload["reference"]["kind"] == "entrypoint"
    assert payload["reference"]["value"] == f"{external_model_path}:build_estimator"
    assert payload["params"]["shift"] == 0.5


def test_save_model_to_catalog_reuses_existing_reference(tmp_path: Path) -> None:
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_entrypoint_model(external_model_path)
    model_dir = tmp_path / "model_catalog"

    first = save_model_to_catalog(
        entrypoint_model(
            "first_name",
            f"{external_model_path}:build_estimator",
            shift=0.5,
        ),
        model_dir=model_dir,
    )
    second = save_model_to_catalog(
        entrypoint_model(
            "second_name",
            f"{external_model_path}:build_estimator",
            shift=0.5,
        ),
        model_dir=model_dir,
    )

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["name"] == "first_name"


def test_catalog_model_loads_reference_and_allows_pipeline_override(tmp_path: Path) -> None:
    external_model_path = tmp_path / "tiny_external_model.py"
    _write_external_entrypoint_model(external_model_path)
    model_dir = tmp_path / "model_catalog"

    save_model_to_catalog(
        entrypoint_model(
            "cataloged_model",
            f"{external_model_path}:build_estimator",
            shift=0.5,
            pipeline="minmax",
            steps=[
                {
                    "type": "min_max_scale",
                    "params": {"feature_min": 0.0, "feature_max": 1.0, "clip": False},
                }
            ],
            execution={"mode": "subprocess", "python": sys.executable},
        ),
        model_dir=model_dir,
    )

    loaded = catalog_model("cataloged_model", model_dir=model_dir)
    overridden = catalog_model(
        "cataloged_model",
        model_dir=model_dir,
        pipeline="raw",
        steps=[],
    )

    assert loaded.reference_kind == "entrypoint"
    assert loaded.reference_value == f"{external_model_path}:build_estimator"
    assert loaded.pipeline_name == "minmax"
    assert loaded.pipeline_steps[0]["type"] == "min_max_scale"
    assert loaded.execution.mode == "subprocess"
    assert loaded.execution.python == sys.executable
    assert loaded.params["shift"] == 0.5

    assert overridden.reference_value == f"{external_model_path}:build_estimator"
    assert overridden.pipeline_name == "raw"
    assert overridden.pipeline_steps == []


def test_catalog_model_uses_manifest_default_pipeline_when_none_saved(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model_catalog"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "plugin_model.json").write_text(
        json.dumps(
            {
                "name": "plugin_model",
                "reference": {"kind": "plugin", "value": "demo_plugin"},
                "params": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ts_benchmark.notebook.api.resolve_default_pipeline_config",
        lambda reference, default_name=None: PipelineConfig(
            name="minmax",
            steps=[{"type": "min_max_scale", "params": {"feature_min": 0.0, "feature_max": 1.0, "clip": False}}],
        ),
    )

    loaded = catalog_model("plugin_model", model_dir=model_dir)

    assert loaded.pipeline_name == "minmax"
    assert loaded.pipeline_steps[0]["type"] == "min_max_scale"


def test_save_dataset_definition_persists_notebook_dataset_spec(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    _write_returns_csv(csv_path)
    dataset_spec = csv_dataset(
        csv_path,
        name="Story 2 Dataset",
        description="Notebook-saved CSV dataset",
        time_column="date",
        target_columns=["asset_a", "asset_b"],
        frequency="B",
        semantics={"target_kind": "returns", "return_kind": "simple"},
    )

    saved_path = save_dataset_definition(dataset_spec, dataset_dir=tmp_path / "datasets")
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert payload["name"] == "Story 2 Dataset"
    assert payload["provider"]["kind"] == "csv"
    assert payload["provider"]["config"]["path"] == str(csv_path.resolve())
    assert payload["schema"]["target_columns"] == ["asset_a", "asset_b"]


def test_save_dataset_definition_persists_synthetic_notebook_dataset_spec(tmp_path: Path) -> None:
    dataset_spec = synthetic_dataset(
        "regime_switching_factor_sv",
        params={"n_assets": 4, "seed": 31},
        n_points=60,
        name="Synthetic Story Dataset",
        description="Notebook-saved synthetic dataset",
    )

    saved_path = save_dataset_definition(dataset_spec, dataset_dir=tmp_path / "datasets")
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert payload["name"] == "Synthetic Story Dataset"
    assert payload["provider"]["kind"] == "synthetic"
    assert payload["provider"]["config"]["generator"] == "regime_switching_factor_sv"
    assert payload["provider"]["config"]["params"]["n_assets"] == 4


def test_save_benchmark_definition_persists_notebook_run_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "saved_benchmark_run"
    run = run_benchmark(_notebook_smoke_config(output_dir=output_dir))

    saved_path = save_benchmark_definition(run, tmp_path / "generated_benchmark.json")
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert payload["benchmark"]["name"] == "notebook_smoke"
    assert payload["benchmark"]["models"][0]["name"] == "debug_model"
