"""Dataset definitions, runtime objects, and providers."""

from __future__ import annotations

from importlib import import_module

_EXPORT_TO_MODULE = {
    "DatasetConfig": ".definition",
    "DatasetInstance": ".runtime",
    "DatasetProviderConfig": ".definition",
    "ForecastWindowDataset": ".windows",
    "RegimeSwitchingFactorSVGenerator": ".providers.synthetic",
    "SyntheticDatasetInstance": ".providers.synthetic",
    "SyntheticSimulation": ".providers.synthetic",
    "build_dataset": ".factory",
    "load_returns_frame": ".providers.tabular",
    "make_tabular_benchmark_dataset": ".providers.tabular",
    "rolling_context_future_pairs": ".windows",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
