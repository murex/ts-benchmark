
"""Example of plugging a custom model into the benchmark contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark import ForecastProtocol
from ts_benchmark.dataset import RegimeSwitchingFactorSVGenerator
from ts_benchmark.metrics import select_metric_configs_for_run
from ts_benchmark.model import (
    HistoricalBootstrapModel,
    RuntimeContext,
    ScenarioModel,
    ScenarioRequest,
    ScenarioSamples,
    TrainingData,
)
from ts_benchmark.run.evaluator import ScenarioBenchmark


class GaussianIIDContractModel(ScenarioModel):
    """Very small example model implementing the contract directly."""

    name = "gaussian_iid_contract"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = float(ridge)
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None

    def fit(self, train_data: TrainingData) -> "GaussianIIDContractModel":
        train_data.validate()
        x = np.asarray(train_data.concatenated_training_values(), dtype=float)
        self.mean_ = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        self.cov_ = cov
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        if self.mean_ is None or self.cov_ is None:
            raise RuntimeError("Model must be fit before sampling.")

        rng = np.random.default_rng(request.seed)
        draws = rng.multivariate_normal(
            mean=self.mean_,
            cov=self.cov_,
            size=(request.n_scenarios, request.horizon),
        )
        samples = ScenarioSamples(
            samples=np.asarray(draws, dtype=float),
            metadata={"distribution": "multivariate_normal"},
        )
        samples.validate(
            expected_horizon=request.horizon,
            expected_n_assets=request.n_assets,
        )
        return samples


def main() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=4, seed=31)
    protocol = ForecastProtocol(
        train_size=400,
        test_size=120,
        context_length=30,
        horizon=10,
        eval_stride=15,
        train_stride=1,
        n_model_scenarios=64,
        n_reference_scenarios=128,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=31,
    )

    models = {
        "historical_bootstrap": HistoricalBootstrapModel(block_size=5),
        "gaussian_iid_contract": GaussianIIDContractModel(),
    }
    metric_configs = select_metric_configs_for_run(
        [{"name": "crps"}, {"name": "energy_score"}, {"name": "cross_correlation_error"}],
        has_reference_scenarios=dataset.has_reference_scenarios(),
        n_assets=len(dataset.asset_names),
        dataset_source=dataset.source,
    )

    benchmark = ScenarioBenchmark(
        models=models,
        protocol=protocol,
        metric_configs=metric_configs,
        runtime=RuntimeContext(seed=31),
    )
    results = benchmark.run(dataset)
    print(results.metrics_frame().round(6).to_string())


if __name__ == "__main__":
    main()
