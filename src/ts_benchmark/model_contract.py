"""Public structural model contract for external model authors.

This module is the stable import path for the model-side contract:

    from ts_benchmark.model_contract import ...

It intentionally re-exports the small, benchmark-independent contract
defined under ``ts_benchmark.model.model_contract``.
"""

from .model.model_contract import (
    Array,
    Constraint,
    DataSchema,
    FitReport,
    FittedTSGenerator,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ModelCapabilities,
    RuntimeContext,
    TrainData,
    TrainExample,
    TSSeries,
    TSGeneratorEstimator,
    TaskSpec,
)

__all__ = [
    "Array",
    "Constraint",
    "DataSchema",
    "FitReport",
    "FittedTSGenerator",
    "GenerationMode",
    "GenerationRequest",
    "GenerationResult",
    "ModelCapabilities",
    "RuntimeContext",
    "TrainData",
    "TrainExample",
    "TSSeries",
    "TSGeneratorEstimator",
    "TaskSpec",
]
