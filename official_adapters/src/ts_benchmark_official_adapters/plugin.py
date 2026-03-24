"""Plugin entry points for official multivariate benchmark adapters."""

from __future__ import annotations

from .deepvar_gpvar import DeepVARAdapter, DeepVARConfig, GPVARAdapter, GPVARConfig
from .manifests import (
    GLUONTS_DEEPVAR_MANIFEST,
    GLUONTS_GPVAR_MANIFEST,
    PYTORCHTS_TIMEGRAD_MANIFEST,
)
from .timegrad import PytorchTsTimeGradAdapter, PytorchTsTimeGradConfig


def build_gluonts_deepvar(**params):
    return DeepVARAdapter(DeepVARConfig(**params))


def build_gluonts_gpvar(**params):
    return GPVARAdapter(GPVARConfig(**params))


def build_pytorchts_timegrad(**params):
    return PytorchTsTimeGradAdapter(PytorchTsTimeGradConfig(**params))


def get_gluonts_deepvar_manifest():
    return GLUONTS_DEEPVAR_MANIFEST


def get_gluonts_gpvar_manifest():
    return GLUONTS_GPVAR_MANIFEST


def get_pytorchts_timegrad_manifest():
    return PYTORCHTS_TIMEGRAD_MANIFEST


build_gluonts_deepvar.PLUGIN_MANIFEST = GLUONTS_DEEPVAR_MANIFEST
build_gluonts_deepvar.get_plugin_manifest = get_gluonts_deepvar_manifest
build_gluonts_deepvar.CONFIG_CLS = DeepVARConfig

build_gluonts_gpvar.PLUGIN_MANIFEST = GLUONTS_GPVAR_MANIFEST
build_gluonts_gpvar.get_plugin_manifest = get_gluonts_gpvar_manifest
build_gluonts_gpvar.CONFIG_CLS = GPVARConfig

build_pytorchts_timegrad.PLUGIN_MANIFEST = PYTORCHTS_TIMEGRAD_MANIFEST
build_pytorchts_timegrad.get_plugin_manifest = get_pytorchts_timegrad_manifest
build_pytorchts_timegrad.CONFIG_CLS = PytorchTsTimeGradConfig
