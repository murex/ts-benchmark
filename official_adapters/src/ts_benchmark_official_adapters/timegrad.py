"""Structural-contract adapter around the official `pytorch-ts` TimeGrad model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Callable

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
    coerce_forecast_history_series_collection,
    coerce_forecast_training_series_collection,
    coerce_series_values,
    make_list_dataset_from_series,
    make_list_dataset_from_series_collection,
    normalize_forecast_samples,
    require_forecast_task,
    runtime_device,
    save_pickle_payload,
)

_TIMEGRAD_EXTRA = "ts-benchmark-official-adapters[timegrad]"


@dataclass
class PytorchTsTimeGradConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1
    lags_seq: tuple[int, ...] = (1,)
    num_inference_steps: int = 100
    scaling: str = "mean"
    beta_end: float = 0.1
    trainer_kwargs: dict[str, Any] = field(default_factory=lambda: {"max_epochs": 5})
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_factory: Callable[["PytorchTsTimeGradConfig"], Any] | None = None


def _ensure_gluonts_distribution_output_compat() -> None:
    """Bridge GluonTS module renames expected by older `pytorchts` releases."""

    legacy_name = "gluonts.torch.modules.distribution_output"
    if legacy_name in sys.modules:
        return
    try:
        importlib.import_module(legacy_name)
        return
    except ModuleNotFoundError:
        pass
    compat_module = importlib.import_module("gluonts.torch.distributions.distribution_output")
    sys.modules[legacy_name] = compat_module


class _TimeGradGenerator:
    def __init__(
        self,
        *,
        config: PytorchTsTimeGradConfig,
        predictor: object,
        prediction_length: int,
        context_length: int,
        target_dim: int,
        freq: str,
        runtime_device_hint: str | None,
        resolved_device: str | None,
    ) -> None:
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
            install_extra=_TIMEGRAD_EXTRA,
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
            "backend": "pytorchts",
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


class PytorchTsTimeGradAdapter:
    """Estimator-like wrapper for `pts.model.time_grad.TimeGradEstimator`."""

    name = "pytorchts_timegrad"
    CONFIG_CLS = PytorchTsTimeGradConfig

    def __init__(self, config: PytorchTsTimeGradConfig):
        self.config = config

    @staticmethod
    def _resolve_scaling(value: str) -> bool:
        text = str(value).strip().lower()
        return text not in {"none", "false", "off", "no"}

    def _build_trainer(self, runtime: RuntimeContext | None) -> tuple[object, str]:
        try:
            from pts import Trainer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Could not import `pts.Trainer`. Install the "
                f"`{_TIMEGRAD_EXTRA}` extra."
            ) from exc

        trainer_kwargs = dict(self.config.trainer_kwargs)
        if "max_epochs" in trainer_kwargs and "epochs" not in trainer_kwargs:
            trainer_kwargs["epochs"] = trainer_kwargs.pop("max_epochs")

        runtime_hint = runtime_device(runtime)
        if runtime_hint in {None, "", "auto"}:
            try:
                import torch
            except Exception:  # pragma: no cover - torch is a hard dependency of pytorchts
                torch = None
            if torch is not None and torch.cuda.is_available():
                trainer_kwargs.setdefault("device", "cuda")
            else:
                trainer_kwargs.setdefault("device", "cpu")
        elif runtime_hint == "mps":
            trainer_kwargs.setdefault("device", "cpu")
        else:
            trainer_kwargs.setdefault("device", runtime_hint)
        resolved_device = str(trainer_kwargs.get("device", "cpu"))
        return Trainer(**trainer_kwargs), resolved_device

    def _resolve_input_size(self, target_dim: int, freq: str) -> int:
        try:
            from pts.feature import fourier_time_features_from_frequency  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Could not import `pts.feature`. Install the "
                f"`{_TIMEGRAD_EXTRA}` extra."
            ) from exc

        lags_seq = list(self.config.lags_seq)
        time_feature_dim = 2 * len(fourier_time_features_from_frequency(freq))
        embedding_dim = target_dim
        return target_dim * len(lags_seq) + embedding_dim + time_feature_dim

    def fit(
        self,
        train: TrainData,
        *,
        schema: DataSchema,
        task: TaskSpec,
        valid: TrainData | None = None,
        runtime: RuntimeContext | None = None,
    ) -> tuple[_TimeGradGenerator, FitReport]:
        del valid
        started_at = time.perf_counter()
        horizon = require_forecast_task(task)
        target_dim = int(getattr(schema, "target_dim"))
        freq = getattr(schema, "freq", None) or "B"
        _, context_length = coerce_forecast_training_series_collection(
            train,
            target_dim=target_dim,
            horizon=horizon,
        )
        history_series_collection, _ = coerce_forecast_history_series_collection(
            train,
            horizon=horizon,
            target_dim=target_dim,
        )
        _ensure_gluonts_distribution_output_compat()

        try:
            from pts.model.time_grad import TimeGradEstimator  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Could not import `TimeGradEstimator`. Install the "
                f"`{_TIMEGRAD_EXTRA}` extra."
            ) from exc

        dataset_train = make_list_dataset_from_series_collection(
            history_series_collection,
            freq=freq,
            install_extra=_TIMEGRAD_EXTRA,
        )
        trainer, resolved_device = self._build_trainer(runtime)
        estimator_kwargs = {
            "input_size": self._resolve_input_size(target_dim, freq),
            "target_dim": target_dim,
            "num_layers": self.config.num_layers,
            "num_cells": self.config.hidden_size,
            "dropout_rate": self.config.dropout_rate,
            "lags_seq": list(self.config.lags_seq),
            "diff_steps": self.config.num_inference_steps,
            "prediction_length": horizon,
            "context_length": context_length,
            "freq": freq,
            "scaling": self._resolve_scaling(self.config.scaling),
            "trainer": trainer,
            "beta_end": self.config.beta_end,
        }
        estimator_kwargs.update(dict(self.config.estimator_kwargs))
        estimator = TimeGradEstimator(**estimator_kwargs)
        if getattr(estimator, "train_sampler", None) is not None:
            estimator.train_sampler.min_instances = 1

        predictor = estimator.train(dataset_train, prefetch_factor=None)
        generator = _TimeGradGenerator(
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
                "backend": "pytorchts",
                "config": asdict(self.config),
                "prediction_length": horizon,
                "context_length": context_length,
                "target_dim": target_dim,
                "freq": freq,
                "runtime_device_hint": runtime_device(runtime),
                "resolved_device": resolved_device,
                "training_series_source": "forecast_history_examples",
                "training_series_count": len(history_series_collection),
            },
        )
        return generator, report
