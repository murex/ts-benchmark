"""Unit tests for preprocessing, metrics, windows, plugin resolution, and config validation."""

from __future__ import annotations

import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Preprocessing round-trip tests
# ---------------------------------------------------------------------------

from ts_benchmark.preprocessing.pipeline import PreprocessingPipeline, build_pipeline_from_config
from ts_benchmark.preprocessing.transforms import (
    ClipTransform,
    DemeanTransform,
    IdentityTransform,
    MinMaxScalerTransform,
    RobustScalerTransform,
    StandardScalerTransform,
    WinsorizeTransform,
)


class TestTransformRoundTrips:
    """inverse_transform(transform(x)) should recover x for invertible transforms."""

    @pytest.fixture()
    def data(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(0.01, 0.03, size=(100, 4))

    def test_identity_round_trip(self, data: np.ndarray) -> None:
        t = IdentityTransform()
        t.fit(data)
        assert np.allclose(t.inverse_transform(t.transform(data)), data)

    def test_demean_round_trip(self, data: np.ndarray) -> None:
        t = DemeanTransform()
        t.fit(data)
        transformed = t.transform(data)
        assert np.abs(transformed.mean(axis=0)).max() < 1e-12
        assert np.allclose(t.inverse_transform(transformed), data)

    def test_standard_scaler_round_trip(self, data: np.ndarray) -> None:
        t = StandardScalerTransform(with_mean=True, with_std=True)
        t.fit(data)
        transformed = t.transform(data)
        assert np.abs(transformed.mean(axis=0)).max() < 1e-10
        assert np.allclose(np.std(transformed, axis=0, ddof=1), 1.0, atol=1e-10)
        assert np.allclose(t.inverse_transform(transformed), data)

    def test_standard_scaler_mean_only_round_trip(self, data: np.ndarray) -> None:
        t = StandardScalerTransform(with_mean=True, with_std=False)
        t.fit(data)
        assert np.allclose(t.inverse_transform(t.transform(data)), data)

    def test_standard_scaler_std_only_round_trip(self, data: np.ndarray) -> None:
        t = StandardScalerTransform(with_mean=False, with_std=True)
        t.fit(data)
        assert np.allclose(t.inverse_transform(t.transform(data)), data)

    def test_robust_scaler_round_trip(self, data: np.ndarray) -> None:
        t = RobustScalerTransform()
        t.fit(data)
        assert np.allclose(t.inverse_transform(t.transform(data)), data)

    def test_minmax_scaler_round_trip(self, data: np.ndarray) -> None:
        t = MinMaxScalerTransform(feature_min=0.0, feature_max=1.0)
        t.fit(data)
        transformed = t.transform(data)
        assert np.allclose(transformed.min(axis=0), 0.0, atol=1e-10)
        assert np.allclose(transformed.max(axis=0), 1.0, atol=1e-10)
        assert np.allclose(t.inverse_transform(transformed), data)

    def test_minmax_scaler_constant_feature_is_safe(self) -> None:
        x = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ]
        )
        t = MinMaxScalerTransform(feature_min=-1.0, feature_max=1.0)
        t.fit(x)
        transformed = t.transform(x)
        assert np.allclose(transformed[:, 0], -1.0)
        assert np.allclose(t.inverse_transform(transformed), x)

    def test_transforms_work_on_3d_arrays(self, data: np.ndarray) -> None:
        data_3d = data.reshape(10, 10, 4)
        for Transform in [DemeanTransform, StandardScalerTransform, RobustScalerTransform, MinMaxScalerTransform]:
            t = Transform()
            t.fit(data)
            transformed = t.transform(data_3d)
            recovered = t.inverse_transform(transformed)
            assert np.allclose(recovered, data_3d, atol=1e-10)


class TestPipelineRoundTrips:
    def test_multi_step_pipeline_round_trip(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.normal(0.005, 0.02, size=(80, 3))
        pipeline = build_pipeline_from_config(
            "test_pipeline",
            [
                {"type": "demean", "params": {}},
                {"type": "standard_scale", "params": {"with_mean": True, "with_std": True}},
            ],
        )
        pipeline.fit(x)
        y = pipeline.transform(x)
        x_back = pipeline.inverse_transform(y)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_empty_pipeline_is_identity(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pipeline = build_pipeline_from_config("raw", [])
        pipeline.fit(x)
        assert np.allclose(pipeline.transform(x), x)
        assert np.allclose(pipeline.inverse_transform(x), x)

    def test_unknown_transform_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            build_pipeline_from_config("bad", [{"type": "nonexistent", "params": {}}])

    def test_pipeline_summary(self) -> None:
        pipeline = build_pipeline_from_config(
            "std",
            [{"type": "standard_scale", "params": {"with_mean": True, "with_std": True}}],
        )
        pipeline.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        summary = pipeline.summary()
        assert summary["name"] == "std"
        assert len(summary["steps"]) == 1
        assert summary["steps"][0]["type"] == "standard_scale"

    def test_minmax_pipeline_round_trip(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        pipeline = build_pipeline_from_config(
            "minmax",
            [{"type": "min_max_scale", "params": {"feature_min": -1.0, "feature_max": 1.0}}],
        )
        pipeline.fit(x)
        y = pipeline.transform(x)
        assert np.allclose(y.min(axis=0), -1.0)
        assert np.allclose(y.max(axis=0), 1.0)
        assert np.allclose(pipeline.inverse_transform(y), x)


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------

from ts_benchmark.metrics.scoring import (
    compute_sample_scoring_metrics,
    coverage_error,
    energy_score,
    predictive_mean_mse,
    sample_crps,
)
from ts_benchmark.metrics.distributional import (
    compute_distributional_metrics,
    correlation_matrix_error,
    moment_errors,
)
from ts_benchmark.metrics import MetricConfig
from ts_benchmark.results.types import BenchmarkResults, MetricResult, ModelResult
from ts_benchmark.utils import JsonObject


class TestScoringMetrics:
    @pytest.fixture()
    def perfect_forecast(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        realized = rng.normal(0, 0.02, size=(5, 3, 2))
        samples = np.tile(realized[:, None, :, :], (1, 32, 1, 1))
        samples += rng.normal(0, 1e-8, size=samples.shape)
        return samples, realized

    def test_crps_is_nonneg(self, perfect_forecast) -> None:
        samples, realized = perfect_forecast
        val = sample_crps(samples, realized)
        assert val >= 0.0

    def test_crps_near_zero_for_perfect_forecast(self, perfect_forecast) -> None:
        samples, realized = perfect_forecast
        val = sample_crps(samples, realized)
        assert val < 0.01

    def test_energy_score_is_nonneg(self, perfect_forecast) -> None:
        samples, realized = perfect_forecast
        val = energy_score(samples, realized)
        assert val >= 0.0

    def test_energy_score_near_zero_for_perfect_forecast(self, perfect_forecast) -> None:
        samples, realized = perfect_forecast
        val = energy_score(samples, realized)
        assert val < 0.01

    def test_predictive_mean_mse_is_nonneg(self, perfect_forecast) -> None:
        samples, realized = perfect_forecast
        val = predictive_mean_mse(samples, realized)
        assert val >= 0.0
        assert val < 1e-12

    def test_coverage_error_near_zero_for_wide_interval(self) -> None:
        rng = np.random.default_rng(11)
        realized = rng.normal(0, 0.01, size=(10, 3, 2))
        samples = rng.normal(0, 0.01, size=(10, 200, 3, 2))
        val = coverage_error(samples, realized, alpha=0.10)
        assert val < 0.15

    def test_compute_sample_scoring_metrics_returns_all_keys(self) -> None:
        rng = np.random.default_rng(22)
        samples = rng.normal(size=(5, 32, 3, 2))
        realized = rng.normal(size=(5, 3, 2))
        result = compute_sample_scoring_metrics(samples, realized)
        assert set(result) == {"crps", "energy_score", "predictive_mean_mse", "coverage_90_error"}
        for v in result.values():
            assert np.isfinite(v)


class TestDistributionalMetrics:
    @pytest.fixture()
    def matching_samples(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(55)
        ref = rng.normal(0, 0.02, size=(4, 64, 5, 3))
        model = rng.normal(0, 0.02, size=(4, 64, 5, 3))
        return model, ref

    def test_moment_errors_keys(self, matching_samples) -> None:
        model, ref = matching_samples
        result = moment_errors(model, ref)
        assert set(result) == {"mean_error", "volatility_error", "skew_error", "excess_kurtosis_error"}
        for v in result.values():
            assert np.isfinite(v)
            assert v >= 0.0

    def test_correlation_error_is_nonneg(self, matching_samples) -> None:
        model, ref = matching_samples
        val = correlation_matrix_error(model, ref)
        assert val >= 0.0
        assert np.isfinite(val)

    def test_identical_samples_have_zero_moments_error(self) -> None:
        rng = np.random.default_rng(33)
        data = rng.normal(0, 0.02, size=(4, 64, 5, 3))
        result = moment_errors(data, data)
        for v in result.values():
            assert v < 1e-10

    def test_compute_distributional_returns_all_keys(self, matching_samples) -> None:
        model, ref = matching_samples
        result = compute_distributional_metrics(model, ref)
        expected_keys = {
            "mean_error", "volatility_error", "skew_error", "excess_kurtosis_error",
            "cross_correlation_error", "autocorrelation_error", "squared_autocorrelation_error",
            "var_95_error", "es_95_error", "max_drawdown_error", "mmd_rbf",
        }
        assert set(result) >= expected_keys


class TestBenchmarkResultsRanking:
    def test_failed_model_without_metrics_is_preserved(self) -> None:
        results = BenchmarkResults.from_model_results(
            [
                ModelResult(
                    model_name="good_model",
                    metric_results=[
                        MetricResult(
                            model_name="good_model",
                            metric_name="crps",
                            value=0.1,
                            direction="minimize",
                        )
                    ],
                ),
                ModelResult(
                    model_name="failed_model",
                    metadata=JsonObject({"error": "ValueError: training exploded"}),
                ),
            ],
            metric_configs=[MetricConfig(name="crps", direction="minimize")],
        )

        assert [item.model_name for item in results.model_results] == ["good_model", "failed_model"]
        assert results.model_results[0].average_rank == 1.0
        assert results.model_results[1].average_rank is None
        assert results.model_results[1].metadata["error"] == "ValueError: training exploded"
        assert set(results.metrics_frame().index) == {"good_model", "failed_model"}


# ---------------------------------------------------------------------------
# Rolling window tests
# ---------------------------------------------------------------------------

from ts_benchmark.dataset.windows import (
    rolling_context_future_pairs,
    rolling_history_context_future_triplets,
)


class TestRollingWindows:
    def test_basic_window_extraction(self) -> None:
        x = np.arange(20).reshape(10, 2).astype(float)
        contexts, futures = rolling_context_future_pairs(x, context_length=3, horizon=2, stride=1)
        assert contexts.shape[1:] == (3, 2)
        assert futures.shape[1:] == (2, 2)
        assert contexts.shape[0] == futures.shape[0]
        assert contexts.shape[0] == 6  # positions 3..8

    def test_stride_reduces_windows(self) -> None:
        x = np.arange(40).reshape(20, 2).astype(float)
        c1, _ = rolling_context_future_pairs(x, context_length=5, horizon=3, stride=1)
        c2, _ = rolling_context_future_pairs(x, context_length=5, horizon=3, stride=4)
        assert c2.shape[0] < c1.shape[0]

    def test_context_future_alignment(self) -> None:
        x = np.arange(20).reshape(10, 2).astype(float)
        contexts, futures = rolling_context_future_pairs(x, context_length=3, horizon=2, stride=1)
        # first window: context=[0,1,2], future=[3,4]
        np.testing.assert_array_equal(contexts[0], x[0:3])
        np.testing.assert_array_equal(futures[0], x[3:5])

    def test_history_context_future_alignment(self) -> None:
        x = np.arange(20).reshape(10, 2).astype(float)
        histories, contexts, futures = rolling_history_context_future_triplets(
            x,
            context_length=3,
            horizon=2,
            stride=1,
        )
        assert len(histories) == contexts.shape[0] == futures.shape[0] == 6
        np.testing.assert_array_equal(histories[0], x[0:3])
        np.testing.assert_array_equal(contexts[0], x[0:3])
        np.testing.assert_array_equal(futures[0], x[3:5])
        np.testing.assert_array_equal(histories[-1], x[0:8])
        np.testing.assert_array_equal(contexts[-1], x[5:8])
        np.testing.assert_array_equal(futures[-1], x[8:10])

    def test_short_series_raises(self) -> None:
        x = np.arange(4).reshape(2, 2).astype(float)
        with pytest.raises(ValueError, match="too short"):
            rolling_context_future_pairs(x, context_length=2, horizon=2, stride=1)

    def test_1d_raises(self) -> None:
        x = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="\\[time, n_assets\\]"):
            rolling_context_future_pairs(x, context_length=3, horizon=2, stride=1)


# ---------------------------------------------------------------------------
# Plugin resolution tests
# ---------------------------------------------------------------------------

from ts_benchmark.model.resolution import (
    import_object,
    instantiate_model_target,
    resolve_model_builder,
)
from ts_benchmark.model.definition import ModelReferenceConfig


class TestPluginResolution:
    def test_builtin_resolution(self) -> None:
        builder = resolve_model_builder(
            reference=ModelReferenceConfig(kind="builtin", value="historical_bootstrap")
        )
        assert callable(builder)

    def test_unknown_builtin_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_model_builder(
                reference=ModelReferenceConfig(kind="builtin", value="nonexistent")
            )

    def test_entrypoint_resolution(self) -> None:
        builder = resolve_model_builder(
            reference=ModelReferenceConfig(
                kind="entrypoint",
                value="ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel"
            )
        )
        assert callable(builder)

    def test_file_entrypoint_resolution(self) -> None:
        builder = resolve_model_builder(
            reference=ModelReferenceConfig(
                kind="entrypoint",
                value=str(ROOT / "src" / "ts_benchmark" / "model" / "builtins" / "debug_smoke_model.py")
                + ":DebugSmokeModel",
            )
        )
        assert callable(builder)

    def test_file_entrypoint_resolution_supports_relative_imports(self, tmp_path: Path) -> None:
        package_dir = tmp_path / "demo_pkg"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (package_dir / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
        (package_dir / "adapter.py").write_text(
            "from .helper import VALUE\n"
            "\n"
            "class DemoAdapter:\n"
            "    scale = VALUE\n",
            encoding="utf-8",
        )

        builder = resolve_model_builder(
            reference=ModelReferenceConfig(
                kind="entrypoint",
                value=str(package_dir / "adapter.py") + ":DemoAdapter",
            )
        )

        assert callable(builder)
        assert getattr(builder, "scale") == 7

    def test_file_entrypoint_resolution_supports_repo_local_imports(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "demo_repo"
        repo_root.mkdir()
        (repo_root / "pyproject.toml").write_text("[project]\nname = 'demo-repo'\nversion = '0.0.1'\n", encoding="utf-8")
        (repo_root / "core.py").write_text("VALUE = 11\n", encoding="utf-8")
        adapter_dir = repo_root / "adapters"
        adapter_dir.mkdir()
        (adapter_dir / "adapter.py").write_text(
            "import core\n"
            "\n"
            "class DemoAdapter:\n"
            "    scale = core.VALUE\n",
            encoding="utf-8",
        )

        builder = resolve_model_builder(
            reference=ModelReferenceConfig(
                kind="entrypoint",
                value=str(adapter_dir / "adapter.py") + ":DemoAdapter",
            )
        )

        assert callable(builder)
        assert getattr(builder, "scale") == 11

    def test_file_entrypoint_resolution_supports_src_layout_repo_imports(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "src_repo"
        repo_root.mkdir()
        (repo_root / "pyproject.toml").write_text("[project]\nname = 'src-repo'\nversion = '0.0.1'\n", encoding="utf-8")
        src_root = repo_root / "src"
        package_dir = src_root / "demo_model"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (package_dir / "core.py").write_text("VALUE = 13\n", encoding="utf-8")
        (package_dir / "adapter.py").write_text(
            "from demo_model.core import VALUE\n"
            "\n"
            "class DemoAdapter:\n"
            "    scale = VALUE\n",
            encoding="utf-8",
        )

        builder = resolve_model_builder(
            reference=ModelReferenceConfig(
                kind="entrypoint",
                value=str(package_dir / "adapter.py") + ":DemoAdapter",
            )
        )

        assert callable(builder)
        assert getattr(builder, "scale") == 13

    def test_bad_entrypoint_format_raises(self) -> None:
        with pytest.raises(ValueError, match="module.submodule:QualifiedName"):
            import_object("no_colon_here")

    def test_unsupported_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            resolve_model_builder(
                reference=ModelReferenceConfig(kind="imaginary", value="foo")
            )

    def test_instantiate_builtin(self) -> None:
        model = instantiate_model_target(
            reference=ModelReferenceConfig(kind="builtin", value="historical_bootstrap"),
            params={"block_size": 3},
        )
        assert hasattr(model, "fit")
        assert hasattr(model, "sample")
        assert model.block_size == 3


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

from ts_benchmark.benchmark.io import load_benchmark_config, validate_benchmark_config


def _minimal_config(**overrides) -> dict:
    config = {
        "version": "1.0",
        "benchmark": {
            "name": "unit_test",
            "dataset": {
                "provider": {
                    "kind": "synthetic",
                    "config": {
                        "generator": "regime_switching_factor_sv",
                        "params": {"n_assets": 2, "seed": 1},
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
                    "name": "test_model",
                    "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                    "params": {"block_size": 2},
                    "pipeline": {"name": "raw", "steps": []},
                }
            ],
        },
        "run": {
            "execution": {"scheduler": "auto"},
            "output": {},
        },
    }
    config.update(overrides)
    return config


def _set_unconditional_windowed(protocol: dict[str, object], *, train_size: int = 80, test_size: int = 20) -> None:
    protocol.clear()
    protocol.update(
        {
            "kind": "unconditional_windowed",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_windowed": {
                "train_size": train_size,
                "test_size": test_size,
                "eval_stride": 5,
                "train_stride": 1,
            },
        }
    )


def _set_unconditional_path_dataset(
    protocol: dict[str, object],
    *,
    horizon: int = 3,
    n_train_paths: int = 5,
    n_realized_paths: int = 3,
) -> None:
    protocol.clear()
    protocol.update(
        {
            "kind": "unconditional_path_dataset",
            "horizon": horizon,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_path_dataset": {
                "n_train_paths": n_train_paths,
                "n_realized_paths": n_realized_paths,
            },
        }
    )


class TestConfigValidation:
    def test_valid_config_loads(self) -> None:
        config = load_benchmark_config(_minimal_config())
        assert config.name == "unit_test"
        assert config.version == "1.0"

    def test_unconditional_config_loads_with_zero_context_length(self) -> None:
        cfg = _minimal_config()
        _set_unconditional_windowed(cfg["benchmark"]["protocol"])
        config = load_benchmark_config(cfg)
        assert config.protocol.generation_mode == "unconditional"
        assert config.protocol.context_length == 0
        assert config.protocol.unconditional_train_data_mode == "windowed_path"
        assert config.protocol.unconditional_train_window_length == 3

    def test_unconditional_config_rejects_nonzero_context_length(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional_windowed",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_windowed": {
                "train_size": 80,
                "test_size": 20,
                "context_length": 1,
                "eval_stride": 5,
                "train_stride": 1,
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_config_requires_train_data_mode(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_windowed_path_config_requires_train_window_length(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional_windowed",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_windowed_path_requires_window_length_to_match_horizon(self) -> None:
        cfg = _minimal_config()
        _set_unconditional_windowed(cfg["benchmark"]["protocol"], train_size=2)
        with pytest.raises(ValueError, match="greater than or equal to horizon"):
            load_benchmark_config(cfg)

    def test_unconditional_path_dataset_config_loads(self) -> None:
        cfg = _minimal_config()
        _set_unconditional_path_dataset(cfg["benchmark"]["protocol"])
        config = load_benchmark_config(cfg)
        assert config.protocol.generation_mode == "unconditional"
        assert config.protocol.unconditional_train_data_mode == "path_dataset"
        assert config.protocol.unconditional_n_train_paths == 5
        assert config.protocol.unconditional_n_eval_paths == 3

    def test_unconditional_path_dataset_requires_n_train_paths(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional_path_dataset",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_path_dataset": {
                "n_realized_paths": 3,
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_path_dataset_requires_n_eval_paths(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional_path_dataset",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_path_dataset": {
                "n_train_paths": 5,
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_path_dataset_requires_positive_realized_paths(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["protocol"] = {
            "kind": "unconditional_path_dataset",
            "horizon": 3,
            "n_model_scenarios": 8,
            "n_reference_scenarios": 12,
            "unconditional_path_dataset": {
                "n_train_paths": 5,
                "n_realized_paths": 0,
            },
        }
        with pytest.raises(jsonschema.ValidationError):
            load_benchmark_config(cfg)

    def test_unconditional_path_dataset_is_restricted_to_synthetic_datasets(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "dummy.csv"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {
            "layout": "wide",
            "time_column": "date",
            "target_columns": ["asset_a", "asset_b"],
            "frequency": "B",
        }
        _set_unconditional_path_dataset(cfg["benchmark"]["protocol"])
        with pytest.raises(ValueError, match="currently supported only for synthetic datasets"):
            load_benchmark_config(cfg)

    def test_functional_smoke_defaults_are_applied_when_enabled(self) -> None:
        cfg = _minimal_config(
            diagnostics={
                "functional_smoke": {
                    "enabled": True,
                }
            }
        )
        config = load_benchmark_config(cfg)
        smoke = config.diagnostics.functional_smoke

        assert smoke.enabled is True
        assert smoke.finite_required is True
        assert smoke.mean_abs_error_max == pytest.approx(0.005)
        assert smoke.std_ratio_min == pytest.approx(0.5)
        assert smoke.std_ratio_max == pytest.approx(1.5)
        assert smoke.crps_max == pytest.approx(0.05)
        assert smoke.energy_score_max == pytest.approx(0.1)
        assert smoke.cross_correlation_error_max == pytest.approx(1.0)

    def test_functional_smoke_all_null_thresholds_fall_back_to_defaults(self) -> None:
        cfg = _minimal_config(
            diagnostics={
                "functional_smoke": {
                    "enabled": True,
                    "finite_required": True,
                    "mean_abs_error_max": None,
                    "std_ratio_min": None,
                    "std_ratio_max": None,
                    "crps_max": None,
                    "energy_score_max": None,
                    "cross_correlation_error_max": None,
                }
            }
        )
        config = load_benchmark_config(cfg)
        smoke = config.diagnostics.functional_smoke

        assert smoke.enabled is True
        assert smoke.mean_abs_error_max == pytest.approx(0.005)
        assert smoke.std_ratio_min == pytest.approx(0.5)
        assert smoke.std_ratio_max == pytest.approx(1.5)
        assert smoke.crps_max == pytest.approx(0.05)
        assert smoke.energy_score_max == pytest.approx(0.1)
        assert smoke.cross_correlation_error_max == pytest.approx(1.0)

    def test_duplicate_model_names_rejected(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["models"].append(cfg["benchmark"]["models"][0].copy())
        with pytest.raises(ValueError, match="unique"):
            validate_benchmark_config(cfg)

    def test_empty_models_rejected(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["models"] = []
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            validate_benchmark_config(cfg)

    def test_missing_generator_rejected(self) -> None:
        cfg = _minimal_config()
        del cfg["benchmark"]["dataset"]["provider"]["config"]["generator"]
        with pytest.raises(ValueError, match="generator"):
            validate_benchmark_config(cfg)

    def test_csv_without_path_rejected(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {"kind": "csv", "config": {}}
        with pytest.raises(ValueError, match="path"):
            validate_benchmark_config(cfg)

    def test_protocol_fields_in_params_rejected(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["models"][0]["params"]["horizon"] = 5
        with pytest.raises(ValueError, match="benchmark-owned protocol"):
            validate_benchmark_config(cfg)


# ---------------------------------------------------------------------------
# Metric applicability at config load time (Refactor 3)
# ---------------------------------------------------------------------------


class TestMetricApplicabilityAtLoadTime:
    def test_synthetic_only_metric_rejected_for_csv(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "/tmp/fake.csv"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {"layout": "wide"}
        cfg["benchmark"]["metrics"] = [{"name": "volatility_error"}]
        with pytest.raises(ValueError, match="not compatible"):
            validate_benchmark_config(cfg)

    def test_reference_scenario_metric_rejected_for_csv(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "/tmp/fake.csv"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {"layout": "wide"}
        cfg["benchmark"]["metrics"] = [{"name": "mmd_rbf"}]
        with pytest.raises(ValueError, match="not compatible"):
            validate_benchmark_config(cfg)

    def test_compatible_metric_accepted_for_csv(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "/tmp/fake.csv"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {"layout": "wide"}
        cfg["benchmark"]["metrics"] = [{"name": "crps"}, {"name": "energy_score"}]
        validate_benchmark_config(cfg)  # should not raise

    def test_long_layout_csv_accepted_with_required_fields(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "/tmp/fake.csv", "value_column": "price"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {
            "layout": "long",
            "time_column": "date",
            "series_id_columns": ["ticker"],
        }
        cfg["benchmark"]["metrics"] = [{"name": "crps"}]
        validate_benchmark_config(cfg)

    def test_long_layout_csv_requires_value_column(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["provider"] = {
            "kind": "csv",
            "config": {"path": "/tmp/fake.csv"},
        }
        cfg["benchmark"]["dataset"]["schema"] = {
            "layout": "long",
            "time_column": "date",
            "series_id_columns": ["ticker"],
        }
        with pytest.raises(ValueError, match="value_column"):
            validate_benchmark_config(cfg)

    def test_all_metrics_accepted_for_synthetic(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["metrics"] = [
            {"name": "crps"},
            {"name": "volatility_error"},
            {"name": "mmd_rbf"},
        ]
        validate_benchmark_config(cfg)  # should not raise

    def test_no_metrics_block_accepted(self) -> None:
        cfg = _minimal_config()
        validate_benchmark_config(cfg)  # should not raise


# ---------------------------------------------------------------------------
# Flattened config structure tests (Refactor 4)
# ---------------------------------------------------------------------------


class TestFlattenedConfig:
    def test_config_has_flat_fields(self) -> None:
        config = load_benchmark_config(_minimal_config())
        assert hasattr(config, "name")
        assert hasattr(config, "dataset")
        assert hasattr(config, "protocol")
        assert hasattr(config, "models")
        assert hasattr(config, "metrics")
        assert hasattr(config, "run")
        assert hasattr(config, "diagnostics")

    def test_run_config_has_flat_device(self) -> None:
        cfg = _minimal_config()
        cfg["run"]["execution"] = {"device": "cpu", "scheduler": "sequential"}
        config = load_benchmark_config(cfg)
        assert config.run.device == "cpu"
        assert config.run.scheduler == "sequential"

    def test_dataset_config_has_flat_schema_fields(self) -> None:
        cfg = _minimal_config()
        cfg["benchmark"]["dataset"]["schema"] = {
            "layout": "tensor",
            "frequency": "B",
            "time_column": "date",
        }
        config = load_benchmark_config(cfg)
        assert config.dataset.layout == "tensor"
        assert config.dataset.frequency == "B"
        assert config.dataset.time_column == "date"
