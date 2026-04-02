from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.dataset.providers.synthetic import RegimeSwitchingFactorSVGenerator
from ts_benchmark.metrics import MetricConfig, rank_metrics_table, select_metric_configs_for_run
from ts_benchmark.model import (
    ForecastWindowCollection,
    ForecastProtocol,
    GaussianCovarianceModel,
    HistoricalBootstrapModel,
    RuntimeContext,
    ScenarioModel,
    ScenarioRequest,
    ScenarioSamples,
    StochasticVolatilityBootstrapModel,
    TrainingData,
    UnconditionalPathDatasetProtocol,
    UnconditionalWindowedProtocol,
)
from ts_benchmark.model.catalog.plugins import extract_model_plugin_manifest, list_model_plugins
from ts_benchmark.run.evaluator import ScenarioBenchmark


class GaussianContractSmokeModel(ScenarioModel):
    name = "gaussian_contract_smoke"

    def fit(self, train_data: TrainingData) -> "GaussianContractSmokeModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        self.mean_ = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        self.cov_ = cov
        self.protocol_ = train_data.protocol
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        rng = np.random.default_rng(request.seed)
        draws = rng.multivariate_normal(
            mean=self.mean_,
            cov=self.cov_,
            size=(request.n_scenarios, request.horizon),
        )
        result = ScenarioSamples(samples=np.asarray(draws, dtype=float))
        result.validate(
            expected_horizon=request.horizon,
            expected_n_assets=request.n_assets,
        )
        return result


def test_historical_models_smoke() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=3, seed=1)
    protocol = ForecastProtocol(
        train_size=220,
        test_size=80,
        context_length=20,
        horizon=5,
        eval_stride=20,
        train_stride=1,
        n_model_scenarios=16,
        n_reference_scenarios=32,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=1,
    )

    models = {
        "historical_bootstrap": HistoricalBootstrapModel(block_size=3),
        "historical_sv_bootstrap": StochasticVolatilityBootstrapModel(
            ewma_lambda=0.97,
            block_size=3,
            vol_of_vol=0.08,
        ),
    }
    metric_configs = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "energy_score"}, {"name": "cross_correlation_error"}],
        has_reference_scenarios=True,
        n_assets=3,
        dataset_source="synthetic",
    )
    benchmark = ScenarioBenchmark(
        models=models,
        protocol=protocol,
        metric_configs=metric_configs,
        runtime=RuntimeContext(seed=1),
    )
    results = benchmark.run(dataset)
    metrics = results.metrics_frame()
    assert not metrics.empty
    assert "crps" in metrics.columns


def test_contract_model_smoke() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=3, seed=3)
    protocol = ForecastProtocol(
        train_size=180,
        test_size=60,
        context_length=12,
        horizon=4,
        eval_stride=20,
        train_stride=2,
        n_model_scenarios=12,
        n_reference_scenarios=24,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=3,
    )

    models = {
        "historical_bootstrap": HistoricalBootstrapModel(block_size=2),
        "gaussian_contract_smoke": GaussianContractSmokeModel(),
    }
    metric_configs = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "energy_score"}, {"name": "cross_correlation_error"}],
        has_reference_scenarios=True,
        n_assets=3,
        dataset_source="synthetic",
    )
    benchmark = ScenarioBenchmark(
        models=models,
        protocol=protocol,
        metric_configs=metric_configs,
        runtime=RuntimeContext(seed=3),
    )
    results = benchmark.run(dataset)
    metrics = results.metrics_frame()
    assert "gaussian_contract_smoke" in metrics.index
    assert np.isfinite(metrics.values).all()
    assert results.metadata["train_stride"] == protocol.train_stride


def test_gaussian_covariance_smoke() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=3, seed=11)
    protocol = ForecastProtocol(
        train_size=160,
        test_size=40,
        context_length=12,
        horizon=4,
        eval_stride=10,
        train_stride=2,
        n_model_scenarios=16,
        n_reference_scenarios=24,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=11,
    )

    benchmark = ScenarioBenchmark(
        models={"gaussian_covariance": GaussianCovarianceModel()},
        protocol=protocol,
        metric_configs=select_metric_configs_for_run(
            [{"name": "crps"}, {"name": "energy_score"}, {"name": "mean_error"}],
            has_reference_scenarios=True,
            n_assets=3,
            dataset_source="synthetic",
        ),
        runtime=RuntimeContext(seed=11),
    )
    results = benchmark.run(dataset)
    metrics = results.metrics_frame()
    assert "gaussian_covariance" in metrics.index
    assert np.isfinite(metrics.loc["gaussian_covariance", "crps"])
    assert np.isfinite(metrics.loc["gaussian_covariance", "energy_score"])


def test_unconditional_generation_mode_smoke() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=3, seed=5)
    protocol = UnconditionalWindowedProtocol(
        train_size=180,
        test_size=60,
        horizon=4,
        eval_stride=20,
        train_stride=1,
        n_model_scenarios=12,
        n_reference_scenarios=24,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=5,
    )
    assert dataset.contexts.shape[1] == 0

    metric_configs = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "mean_error"}],
        has_reference_scenarios=True,
        n_assets=3,
        dataset_source="synthetic",
    )
    benchmark = ScenarioBenchmark(
        models={"historical_bootstrap": HistoricalBootstrapModel(block_size=2)},
        protocol=protocol,
        metric_configs=metric_configs,
        runtime=RuntimeContext(seed=5),
    )
    results = benchmark.run(dataset)
    metrics = results.metrics_frame()
    assert "historical_bootstrap" in metrics.index
    assert "crps" in metrics.columns
    assert "mean_error" in metrics.columns
    assert results.metadata["generation_mode"] == "unconditional"
    assert results.metadata["path_construction"] == "windowed_path"
    assert results.metadata["train_size"] == 180


def test_unconditional_path_dataset_smoke() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=3, seed=6)
    protocol = UnconditionalPathDatasetProtocol(
        horizon=4,
        n_train_paths=5,
        n_realized_paths=3,
        n_model_scenarios=12,
        n_reference_scenarios=24,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=6,
    )
    assert dataset.train_paths is not None
    assert len(dataset.train_paths) == 5
    assert all(path.shape == (protocol.train_size, 3) for path in dataset.train_paths)
    assert dataset.realized_futures.shape == (3, protocol.horizon, 3)
    assert dataset.contexts.shape == (3, 0, 3)

    metric_configs = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "mean_error"}],
        has_reference_scenarios=True,
        n_assets=3,
        dataset_source="synthetic",
    )
    benchmark = ScenarioBenchmark(
        models={"historical_bootstrap": HistoricalBootstrapModel(block_size=2)},
        protocol=protocol,
        metric_configs=metric_configs,
        runtime=RuntimeContext(seed=6),
    )
    results = benchmark.run(dataset)
    metrics = results.metrics_frame()
    assert "historical_bootstrap" in metrics.index
    assert results.metadata["path_construction"] == "path_dataset"
    assert results.metadata["n_train_paths"] == 5
    assert results.metadata["n_realized_paths"] == 3
    assert np.isfinite(metrics.loc["historical_bootstrap", "crps"])


def test_builtin_plugin_listing_smoke() -> None:
    plugins = list_model_plugins()
    assert "historical_bootstrap" in plugins
    assert "timegrad" not in plugins
    assert plugins["historical_bootstrap"]["source"] == "builtin"
    assert plugins["historical_bootstrap"]["manifest"]["display_name"] == "Historical bootstrap"
    assert "cuda" in plugins["historical_bootstrap"]["manifest"]["runtime_device_hints"]
    assert plugins["historical_bootstrap"]["manifest"]["capabilities"]["uses_benchmark_device"] is True


def test_runtime_propagates_to_builtin_models() -> None:
    protocol = ForecastProtocol(
        train_size=40,
        test_size=8,
        context_length=6,
        horizon=3,
        eval_stride=2,
        train_stride=1,
        n_model_scenarios=5,
        n_reference_scenarios=8,
    )
    train = np.linspace(-0.02, 0.02, num=160, dtype=float).reshape(40, 4)
    context = train[-6:]

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a benchmark dependency
        torch = None

    requested_device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
    runtime = RuntimeContext(device=requested_device, seed=7)

    model = HistoricalBootstrapModel(block_size=2)
    contexts = np.stack([train[: protocol.context_length]], axis=0)
    targets = np.stack(
        [train[protocol.context_length : protocol.context_length + protocol.horizon]],
        axis=0,
    )
    model.fit(
        TrainingData(
            returns=train,
            protocol=protocol,
            forecast_windows=ForecastWindowCollection(
                histories=[train[: protocol.context_length]],
                contexts=contexts,
                targets=targets,
                source_kind="single_path",
                stride=1,
            ),
            runtime=runtime,
        )
    )
    samples = model.sample(
        ScenarioRequest(
            context=context,
            horizon=protocol.horizon,
            n_scenarios=5,
            protocol=protocol,
            seed=11,
            runtime=runtime,
        )
    )
    assert samples.samples.shape == (5, protocol.horizon, train.shape[1])
    info = model.model_info()
    assert info["runtime_device"] == requested_device
    assert isinstance(info["resolved_device"], str)


def test_manifest_extraction_from_object() -> None:
    class Dummy:
        name = "dummy_plugin"
        PLUGIN_MANIFEST = {
            "display_name": "Dummy plugin",
            "default_pipeline": "standardized",
            "required_pipeline": "standardized",
            "runtime_device_hints": ["cpu", "cuda"],
            "capabilities": {
                "multivariate": True,
                "probabilistic_sampling": True,
                "uses_benchmark_device": True,
            },
        }

    manifest = extract_model_plugin_manifest(Dummy(), default_name="dummy_plugin")
    assert manifest is not None
    assert manifest.name == "dummy_plugin"
    assert manifest.display_name == "Dummy plugin"
    assert manifest.default_pipeline == "standardized"
    assert manifest.required_pipeline == "standardized"
    assert manifest.runtime_device_hints == ("cpu", "cuda")
    assert manifest.capabilities.uses_benchmark_device is True


def test_metric_object_resolution_and_ranking_smoke() -> None:
    selected = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "energy_score"}],
        has_reference_scenarios=False,
        n_assets=3,
        dataset_source="csv",
    )
    assert [metric.name for metric in selected] == ["crps", "energy_score"]
    assert selected[0].direction == "minimize"

    metrics_table = pd.DataFrame(
        {
            "loss": [0.4, 0.2],
            "score": [0.7, 0.9],
            "target_metric": [0.45, 0.51],
        },
        index=["model_a", "model_b"],
    )
    metric_defs = [
        MetricConfig(name="loss", direction="minimize", granularity="global", aggregation="mean"),
        MetricConfig(name="score", direction="maximize", granularity="global", aggregation="mean"),
        MetricConfig(
            name="target_metric",
            direction="target",
            target_value=0.5,
            granularity="global",
            aggregation="mean",
        ),
    ]
    filtered, rank_table = rank_metrics_table(metrics_table, metric_defs)
    assert filtered.index.tolist()[0] == "model_b"
    assert float(rank_table.loc["model_b", "average_rank"]) < float(rank_table.loc["model_a", "average_rank"])
