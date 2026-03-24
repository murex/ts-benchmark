"""Run the synthetic benchmark with the two historical baselines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark import Protocol
from ts_benchmark.dataset import RegimeSwitchingFactorSVGenerator
from ts_benchmark.metrics import select_metric_configs_for_run
from ts_benchmark.model import (
    HistoricalBootstrapModel,
    RuntimeContext,
    StochasticVolatilityBootstrapModel,
)
from ts_benchmark.run.evaluator import ScenarioBenchmark


def main() -> None:
    generator = RegimeSwitchingFactorSVGenerator(n_assets=6, seed=11)
    protocol = Protocol(
        train_size=1200,
        test_size=260,
        context_length=60,
        horizon=20,
        eval_stride=20,
        train_stride=1,
        n_model_scenarios=128,
        n_reference_scenarios=256,
    )
    dataset = generator.make_benchmark_dataset(
        protocol=protocol,
        seed=11,
    )

    models = {
        "historical_bootstrap": HistoricalBootstrapModel(block_size=5),
        "historical_sv_bootstrap": StochasticVolatilityBootstrapModel(
            ewma_lambda=0.97,
            block_size=5,
            vol_of_vol=0.10,
        ),
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
        runtime=RuntimeContext(seed=11),
    )
    results = benchmark.run(dataset)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    metrics_path = output_dir / "historical_baselines_metrics.csv"
    ranks_path = output_dir / "historical_baselines_ranks.csv"
    results.save_metrics_csv(str(metrics_path))
    results.save_ranks_csv(str(ranks_path))

    print("=== Historical baselines benchmark ===")
    print(results.metrics_frame().round(6).to_string())
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved ranks   to: {ranks_path}")


if __name__ == "__main__":
    main()
