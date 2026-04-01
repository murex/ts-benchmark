"""Plugin entry points for official multivariate benchmark adapters."""

from __future__ import annotations

from .deepvar_gpvar import DeepVARAdapter, DeepVARConfig, GPVARAdapter, GPVARConfig
from .timegrad import PytorchTsTimeGradAdapter, PytorchTsTimeGradConfig


def build_gluonts_deepvar(**params):
    return DeepVARAdapter(DeepVARConfig(**params))


def build_gluonts_gpvar(**params):
    return GPVARAdapter(GPVARConfig(**params))


def build_pytorchts_timegrad(**params):
    return PytorchTsTimeGradAdapter(PytorchTsTimeGradConfig(**params))

build_gluonts_deepvar.CONFIG_CLS = DeepVARConfig

build_gluonts_gpvar.CONFIG_CLS = GPVARConfig

build_pytorchts_timegrad.CONFIG_CLS = PytorchTsTimeGradConfig
