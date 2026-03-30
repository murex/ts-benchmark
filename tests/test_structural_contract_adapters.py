from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark.protocol import Protocol
from ts_benchmark.model import (
    ForecastWindowCollection,
    RuntimeContext,
    ScenarioRequest,
    TrainPathCollection,
    TrainingData,
)
from ts_benchmark.model.model_contract import (
    FitReport,
    GenerationMode,
    GenerationResult,
    ModelCapabilities,
)
from ts_benchmark.model.wrappers.duck_typed import DuckTypedGeneratorScenarioModel


class _DummyGenerator:
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modes=frozenset({GenerationMode.FORECAST, GenerationMode.UNCONDITIONAL})
        )

    def sample(self, request) -> GenerationResult:
        return GenerationResult(
            samples=np.zeros((request.num_samples, request.task.horizon, 2), dtype=float),
            diagnostics={
                "mode": getattr(request.task, "mode", None),
                "context_length": getattr(request.task, "context_length", None),
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"dummy")


class _DummyEstimator:
    def fit(self, train, *, schema, task, valid=None, runtime=None):
        examples = list(getattr(train, "examples", []))
        n_context_examples = sum(example.context is not None for example in examples)
        train_kind = "forecast_examples" if n_context_examples else "unconditional_examples"
        del schema, valid, runtime
        return _DummyGenerator(), FitReport(
            diagnostics={
                "mode": getattr(task, "mode", None),
                "context_length": getattr(task, "context_length", None),
                "train_kind": train_kind,
                "n_train_examples": len(examples),
                "n_context_examples": n_context_examples,
            }
        )


def test_structural_estimators_expose_schema_and_task_fit_parameters() -> None:
    fit_signature = inspect.signature(_DummyEstimator.fit)
    assert "schema" in fit_signature.parameters
    assert "task" in fit_signature.parameters


def test_duck_typed_bridge_passes_context_length_to_structural_estimators() -> None:
    protocol = Protocol(
        train_size=24,
        test_size=6,
        context_length=5,
        horizon=2,
        eval_stride=2,
        train_stride=1,
        n_model_scenarios=4,
        n_reference_scenarios=8,
    )
    model = DuckTypedGeneratorScenarioModel(estimator=_DummyEstimator(), name="dummy")
    training_data = TrainingData(
        returns=np.ones((24, 2), dtype=float),
        protocol=protocol,
        forecast_windows=ForecastWindowCollection(
            contexts=np.ones((18, 5, 2), dtype=float),
            targets=np.ones((18, 2, 2), dtype=float),
            source_kind="single_path",
            stride=1,
        ),
        runtime=RuntimeContext(device="cpu", seed=11),
    )
    model.fit(training_data)

    info = model.model_info()
    assert info["fit_report"]["diagnostics"]["mode"] == "forecast"
    assert info["fit_report"]["diagnostics"]["context_length"] == protocol.context_length
    assert info["fit_report"]["diagnostics"]["train_kind"] == "forecast_examples"
    assert info["fit_report"]["diagnostics"]["n_train_examples"] == 18
    assert info["fit_report"]["diagnostics"]["n_context_examples"] == 18

    request = ScenarioRequest(
        context=np.ones((protocol.context_length, 2), dtype=float),
        horizon=protocol.horizon,
        n_scenarios=protocol.n_model_scenarios,
        protocol=protocol,
        runtime=RuntimeContext(device="cpu", seed=17),
    )
    result = model.sample(request)
    assert result.metadata["diagnostics"]["mode"] == "forecast"
    assert result.metadata["diagnostics"]["context_length"] == protocol.context_length


def test_duck_typed_bridge_passes_unconditional_mode_without_context_length() -> None:
    protocol = Protocol(
        train_size=24,
        test_size=6,
        context_length=0,
        horizon=2,
        generation_mode="unconditional",
        eval_stride=2,
        train_stride=1,
        unconditional_train_data_mode="windowed_path",
        unconditional_train_window_length=6,
        n_model_scenarios=4,
        n_reference_scenarios=8,
    )
    model = DuckTypedGeneratorScenarioModel(estimator=_DummyEstimator(), name="dummy")
    training_data = TrainingData(
        returns=np.ones((24, 2), dtype=float),
        protocol=protocol,
        path_collection=TrainPathCollection(
            paths=[np.ones((6, 2), dtype=float) for _ in range(19)],
            source_kind="windowed_path",
            window_length=6,
            stride=1,
        ),
        runtime=RuntimeContext(device="cpu", seed=11),
    )
    model.fit(training_data)

    info = model.model_info()
    assert info["fit_report"]["diagnostics"]["mode"] == "unconditional"
    assert info["fit_report"]["diagnostics"]["context_length"] is None
    assert info["fit_report"]["diagnostics"]["train_kind"] == "unconditional_examples"
    assert info["fit_report"]["diagnostics"]["n_train_examples"] == 19
    assert info["fit_report"]["diagnostics"]["n_context_examples"] == 0

    request = ScenarioRequest(
        context=np.zeros((0, 2), dtype=float),
        horizon=protocol.horizon,
        n_scenarios=protocol.n_model_scenarios,
        protocol=protocol,
        mode="unconditional",
        runtime=RuntimeContext(device="cpu", seed=17),
    )
    result = model.sample(request)
    assert result.metadata["diagnostics"]["mode"] == "unconditional"
    assert result.metadata["diagnostics"]["context_length"] is None
