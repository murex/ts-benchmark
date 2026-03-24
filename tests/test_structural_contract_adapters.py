from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.benchmark.protocol import Protocol
from ts_benchmark.model import RuntimeContext, ScenarioRequest, TrainingData
from ts_benchmark.model.catalog.plugins import extract_model_parameter_schema, extract_model_plugin_manifest
from ts_benchmark.model.model_contract import FitReport, GenerationMode, GenerationResult, ModelCapabilities
from ts_benchmark.model.wrappers.duck_typed import DuckTypedGeneratorScenarioModel
from ts_benchmark_official_adapters.plugin import (
    build_gluonts_deepvar,
    build_gluonts_gpvar,
    build_pytorchts_timegrad,
)


def test_official_adapter_builders_follow_structural_estimator_contract() -> None:
    builders = [
        (build_gluonts_deepvar, "gluonts_deepvar"),
        (build_gluonts_gpvar, "gluonts_gpvar"),
        (build_pytorchts_timegrad, "pytorchts_timegrad"),
    ]

    for builder, expected_name in builders:
        estimator = builder()
        fit_signature = inspect.signature(estimator.fit)
        assert "schema" in fit_signature.parameters
        assert "task" in fit_signature.parameters

        manifest = extract_model_plugin_manifest(builder, default_name=expected_name)
        assert manifest is not None
        assert manifest.name == expected_name
        assert manifest.capabilities.benchmark_protocol_contract is False

        schema = extract_model_parameter_schema(builder, default_name=expected_name)
        assert schema is not None
        assert len(schema.fields) > 0


class _DummyGenerator:
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modes=frozenset({GenerationMode.FORECAST, GenerationMode.UNCONDITIONAL})
        )

    def sample(self, request) -> GenerationResult:
        return GenerationResult(
            samples=np.zeros((1, request.num_samples, request.task.horizon, 2), dtype=float),
            diagnostics={
                "mode": getattr(request.task, "mode", None),
                "context_length": getattr(request.task, "context_length", None),
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"dummy")


class _DummyEstimator:
    def fit(self, train, *, schema, task, valid=None, runtime=None):
        del train, schema, valid, runtime
        return _DummyGenerator(), FitReport(
            diagnostics={
                "mode": getattr(task, "mode", None),
                "context_length": getattr(task, "context_length", None),
            }
        )


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
        runtime=RuntimeContext(device="cpu", seed=11),
    )
    model.fit(training_data)

    info = model.model_info()
    assert info["fit_report"]["diagnostics"]["mode"] == "forecast"
    assert info["fit_report"]["diagnostics"]["context_length"] == protocol.context_length

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
        n_model_scenarios=4,
        n_reference_scenarios=8,
    )
    model = DuckTypedGeneratorScenarioModel(estimator=_DummyEstimator(), name="dummy")
    training_data = TrainingData(
        returns=np.ones((24, 2), dtype=float),
        protocol=protocol,
        runtime=RuntimeContext(device="cpu", seed=11),
    )
    model.fit(training_data)

    info = model.model_info()
    assert info["fit_report"]["diagnostics"]["mode"] == "unconditional"
    assert info["fit_report"]["diagnostics"]["context_length"] is None

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
