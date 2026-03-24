"""External multivariate plugin adapters for ts-benchmark."""

from .compat import apply_runtime_compat

apply_runtime_compat()

from .deepvar_gpvar import DeepVARAdapter, DeepVARConfig, GPVARAdapter, GPVARConfig
from .timegrad import PytorchTsTimeGradAdapter, PytorchTsTimeGradConfig

__all__ = [
    "DeepVARAdapter",
    "DeepVARConfig",
    "GPVARAdapter",
    "GPVARConfig",
    "PytorchTsTimeGradAdapter",
    "PytorchTsTimeGradConfig",
]
