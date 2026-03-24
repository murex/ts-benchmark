from __future__ import annotations

import importlib
import inspect
import sys
import types

import numpy as np


def _patch_numpy_bool_alias() -> None:
    if not hasattr(np, "bool"):
        np.bool = bool


def _install_gluonts_timegrad_shim() -> None:
    try:
        distribution_mod = importlib.import_module(
            "gluonts.torch.distributions.distribution_output"
        )
        lambda_mod = importlib.import_module("gluonts.torch.modules.lambda_layer")
        output_mod = importlib.import_module("gluonts.torch.distributions.output")
    except Exception:
        return

    shim = types.ModuleType("gluonts.torch.modules.distribution_output")
    shim.DistributionOutput = distribution_mod.DistributionOutput
    shim.LambdaLayer = lambda_mod.LambdaLayer
    shim.PtArgProj = output_mod.PtArgProj
    sys.modules.setdefault("gluonts.torch.modules.distribution_output", shim)


def _patch_pytorch_predictor_init() -> None:
    try:
        from gluonts.torch.model.predictor import PyTorchPredictor
    except Exception:
        return

    signature = inspect.signature(PyTorchPredictor.__init__)
    if "freq" in signature.parameters:
        return

    original_init = PyTorchPredictor.__init__

    def patched_init(self, *args, freq=None, **kwargs):
        return original_init(self, *args, **kwargs)

    PyTorchPredictor.__init__ = patched_init


def apply_runtime_compat() -> None:
    _patch_numpy_bool_alias()
    _install_gluonts_timegrad_shim()
    _patch_pytorch_predictor_init()

