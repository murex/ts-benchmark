"""Dataset providers: synthetic generators and tabular loaders."""

from .synthetic_config import RegimeSwitchingFactorSVConfig
from .synthetic import RegimeSwitchingFactorSVGenerator, SyntheticDatasetInstance, SyntheticSimulation
from .tabular import load_returns_frame, make_tabular_benchmark_dataset

__all__ = [
    "RegimeSwitchingFactorSVConfig",
    "RegimeSwitchingFactorSVGenerator",
    "SyntheticDatasetInstance",
    "SyntheticSimulation",
    "load_returns_frame",
    "make_tabular_benchmark_dataset",
]
