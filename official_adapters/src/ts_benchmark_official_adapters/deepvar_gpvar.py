"""Structural-contract adapters for officially released GluonTS MXNet models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
from pathlib import Path
import time
from typing import Any

import numpy as np

from ts_benchmark.model.model_contract import (
    DataSchema,
    FitReport,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
    RuntimeContext,
    TaskSpec,
    TrainData,
)

from .contract_support import (
    coerce_forecast_training_series_collection,
    coerce_series_values,
    make_list_dataset_from_series,
    make_list_dataset_from_series_collection,
    normalize_forecast_samples,
    require_forecast_task,
    runtime_device,
    save_pickle_payload,
)
from .manifests import GLUONTS_DEEPVAR_MANIFEST, GLUONTS_GPVAR_MANIFEST

_GLUONTS_MX_EXTRA = "ts-benchmark-official-adapters[gluonts-mx]"


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _resolve_mx_context(device: str | None):
    import mxnet as mx  # type: ignore

    def _gpu_context(index: int = 0):
        ctx = mx.gpu(index)
        try:
            mx.nd.array([0], ctx=ctx)
        except Exception:
            return mx.cpu()
        return ctx

    if not device or device == "auto":
        try:
            return _gpu_context(0) if mx.context.num_gpus() > 0 else mx.cpu()
        except Exception:
            return mx.cpu()
    text = str(device).strip().lower()
    if text in {"cpu", "mps"}:
        return mx.cpu()
    if text == "cuda":
        return _gpu_context(0)
    if text.startswith("cuda"):
        index = 0
        if ":" in text:
            try:
                index = int(text.split(":", 1)[1])
            except ValueError:
                index = 0
        return _gpu_context(index)
    if text == "gpu":
        return _gpu_context(0)
    return mx.cpu()


@dataclass
class GluonMxMultivariateConfig:
    epochs: int = 5
    batch_size: int = 32
    num_batches_per_epoch: int = 50
    learning_rate: float = 1e-3
    num_layers: int = 2
    num_cells: int = 40
    cell_type: str = "lstm"
    dropout_rate: float = 0.1
    scaling: bool = True
    num_parallel_samples: int = 100
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepVARConfig(GluonMxMultivariateConfig):
    rank: int = 5


@dataclass
class GPVARConfig(GluonMxMultivariateConfig):
    target_dim_sample: int | None = None


class _GluonMxGenerator:
    def __init__(
        self,
        *,
        name: str,
        backend: str,
        config: GluonMxMultivariateConfig,
        predictor: object,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        freq: str,
        runtime_device_hint: str | None,
        resolved_device: str | None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.config = config
        self.predictor = predictor
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.freq = freq
        self.runtime_device_hint = runtime_device_hint
        self.resolved_device = resolved_device

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modes=frozenset({GenerationMode.FORECAST}),
            supports_multivariate_targets=True,
            supports_known_covariates=False,
            supports_observed_covariates=False,
            supports_static_covariates=False,
            supports_constraints=False,
        )

    def sample(self, request: GenerationRequest) -> GenerationResult:
        horizon = require_forecast_task(request.task)
        if horizon != self.prediction_length:
            raise ValueError(f"Expected horizon={self.prediction_length}, got {horizon}.")
        if request.constraints:
            raise ValueError("This adapter does not support generation constraints.")

        values = coerce_series_values(request.series)
        if values.shape[-1] != self.target_dim:
            raise ValueError(
                f"Request target_dim={values.shape[-1]} does not match trained target_dim={self.target_dim}."
            )

        dataset_test = make_list_dataset_from_series(
            values,
            freq=self.freq,
            install_extra=_GLUONTS_MX_EXTRA,
        )
        try:
            forecast_it = self.predictor.predict(dataset_test, num_samples=request.num_samples)
        except TypeError:
            forecast_it = self.predictor.predict(dataset_test)

        forecasts = list(forecast_it)
        if len(forecasts) != 1:
            raise ValueError(
                f"Backend returned {len(forecasts)} forecasts for a single request series."
            )

        samples = normalize_forecast_samples(
            np.asarray(forecasts[0].samples, dtype=float),
            num_samples=request.num_samples,
            horizon=horizon,
            target_dim=self.target_dim,
        )
        diagnostics = {
            "backend": self.backend,
            "forecast_classes": [forecast.__class__.__name__ for forecast in forecasts],
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "freq": self.freq,
            "runtime_device_hint": self.runtime_device_hint,
            "resolved_device": self.resolved_device,
        }
        return GenerationResult(samples=samples, diagnostics=diagnostics)

    def save(self, path: str | Path) -> None:
        save_pickle_payload(
            path,
            {
                "name": self.name,
                "backend": self.backend,
                "config": asdict(self.config),
                "predictor": self.predictor,
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "target_dim": self.target_dim,
                "freq": self.freq,
                "runtime_device_hint": self.runtime_device_hint,
                "resolved_device": self.resolved_device,
            },
        )


class _BaseGluonMxEstimator:
    name = "gluonts_multivariate"
    backend = "gluonts-mx"
    PLUGIN_MANIFEST = None
    CONFIG_CLS = GluonMxMultivariateConfig

    def __init__(self, config: GluonMxMultivariateConfig):
        self.config = config

    def _load_estimator_class(self):
        raise NotImplementedError

    def _estimator_overrides(self) -> dict[str, Any]:
        return {}

    def _trainer(self, runtime: RuntimeContext | None) -> tuple[object, str]:
        try:
            from gluonts.mx.trainer import Trainer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "This adapter requires the GluonTS MXNet training stack. Install the "
                f"`{_GLUONTS_MX_EXTRA}` extra."
            ) from exc

        kwargs = {
            "ctx": _resolve_mx_context(runtime_device(runtime)),
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "num_batches_per_epoch": self.config.num_batches_per_epoch,
        }
        kwargs.update(dict(self.config.trainer_kwargs))
        trainer = Trainer(**_filter_supported_kwargs(Trainer, kwargs))
        return trainer, str(kwargs["ctx"])

    def fit(
        self,
        train: TrainData,
        *,
        schema: DataSchema,
        task: TaskSpec,
        valid: TrainData | None = None,
        runtime: RuntimeContext | None = None,
    ) -> tuple[_GluonMxGenerator, FitReport]:
        del valid
        started_at = time.perf_counter()
        horizon = require_forecast_task(task)
        target_dim = int(getattr(schema, "target_dim"))
        freq = getattr(schema, "freq", None) or "B"
        train_series_collection, context_length = coerce_forecast_training_series_collection(
            train,
            target_dim=target_dim,
            horizon=horizon,
        )

        estimator_cls = self._load_estimator_class()
        dataset_train = make_list_dataset_from_series_collection(
            train_series_collection,
            freq=freq,
            install_extra=_GLUONTS_MX_EXTRA,
        )
        trainer, resolved_device = self._trainer(runtime)
        estimator_kwargs = {
            "freq": freq,
            "target_dim": target_dim,
            "prediction_length": horizon,
            "context_length": context_length,
            "batch_size": self.config.batch_size,
            "num_layers": self.config.num_layers,
            "num_cells": self.config.num_cells,
            "cell_type": self.config.cell_type,
            "dropout_rate": self.config.dropout_rate,
            "scaling": self.config.scaling,
            "num_parallel_samples": self.config.num_parallel_samples,
            "trainer": trainer,
        }
        estimator_kwargs.update(self._estimator_overrides())
        estimator_kwargs.update(dict(self.config.estimator_kwargs))

        estimator = estimator_cls(
            **_filter_supported_kwargs(estimator_cls.__init__, estimator_kwargs)
        )
        predictor = estimator.train(dataset_train)
        generator = _GluonMxGenerator(
            name=self.name,
            backend=self.backend,
            config=self.config,
            predictor=predictor,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=target_dim,
            freq=freq,
            runtime_device_hint=runtime_device(runtime),
            resolved_device=resolved_device,
        )
        report = FitReport(
            fit_time_sec=time.perf_counter() - started_at,
            diagnostics={
                "backend": self.backend,
                "config": asdict(self.config),
                "prediction_length": horizon,
                "context_length": context_length,
                "target_dim": target_dim,
                "freq": freq,
                "runtime_device_hint": runtime_device(runtime),
                "resolved_device": resolved_device,
                "training_series_count": len(train_series_collection),
            },
        )
        return generator, report


class DeepVARAdapter(_BaseGluonMxEstimator):
    name = "gluonts_deepvar"
    PLUGIN_MANIFEST = GLUONTS_DEEPVAR_MANIFEST
    CONFIG_CLS = DeepVARConfig

    def _load_estimator_class(self):
        try:
            from gluonts.mx.model.deepvar import DeepVAREstimator  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Could not import `DeepVAREstimator`. Install the "
                f"`{_GLUONTS_MX_EXTRA}` extra."
            ) from exc
        return DeepVAREstimator

    def _estimator_overrides(self) -> dict[str, Any]:
        assert isinstance(self.config, DeepVARConfig)
        return {"rank": self.config.rank}


class GPVARAdapter(_BaseGluonMxEstimator):
    name = "gluonts_gpvar"
    PLUGIN_MANIFEST = GLUONTS_GPVAR_MANIFEST
    CONFIG_CLS = GPVARConfig

    def _load_estimator_class(self):
        try:
            from gluonts.mx.model.gpvar import GPVAREstimator  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Could not import `GPVAREstimator`. Install the "
                f"`{_GLUONTS_MX_EXTRA}` extra."
            ) from exc
        return GPVAREstimator

    def _estimator_overrides(self) -> dict[str, Any]:
        assert isinstance(self.config, GPVARConfig)
        out: dict[str, Any] = {}
        if self.config.target_dim_sample is not None:
            out["target_dim_sample"] = self.config.target_dim_sample
        return out
