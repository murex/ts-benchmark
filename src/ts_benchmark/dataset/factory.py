"""Dataset construction helpers."""

from __future__ import annotations

from pathlib import Path

from ..benchmark.protocol import Protocol
from ..serialization import to_jsonable
from ..utils import JsonObject
from .definition import (
    CsvDatasetProviderConfig,
    DatasetConfig,
    ParquetDatasetProviderConfig,
    SyntheticDatasetProviderConfig,
)
from .providers.synthetic import RegimeSwitchingFactorSVGenerator
from .providers.tabular import make_tabular_benchmark_dataset
from .runtime import DatasetInstance


GENERATOR_REGISTRY = {
    "regime_switching_factor_sv": RegimeSwitchingFactorSVGenerator,
}


def _resolve_data_path(source_path: Path | None, path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return str(resolved)
    if source_path is not None:
        candidate = (source_path.parent / resolved).resolve()
        if candidate.exists():
            return str(candidate)
    return str(resolved.resolve())


def _resolved_dataset_name(dataset_config: DatasetConfig) -> str:
    if dataset_config.name:
        return str(dataset_config.name)
    provider = dataset_config.provider
    if isinstance(provider, SyntheticDatasetProviderConfig):
        return f"synthetic::{provider.generator}"
    if isinstance(provider, (CsvDatasetProviderConfig, ParquetDatasetProviderConfig)) and provider.path:
        return Path(str(provider.path)).stem
    return f"{dataset_config.provider.kind}::dataset"


def build_dataset(
    dataset_config: DatasetConfig,
    protocol: Protocol,
    *,
    seed: int = 7,
    source_path: Path | None = None,
) -> DatasetInstance:
    provider = dataset_config.provider

    if isinstance(provider, SyntheticDatasetProviderConfig):
        generator_name = str(provider.generator)
        if generator_name not in GENERATOR_REGISTRY:
            raise KeyError(
                f"Unknown generator '{generator_name}'. Supported: {sorted(GENERATOR_REGISTRY)}"
            )
        generator_kwargs = dict(to_jsonable(provider.params))
        generator = GENERATOR_REGISTRY[str(generator_name)](**generator_kwargs)
        dataset = generator.make_benchmark_dataset(
            protocol=protocol,
            seed=seed,
        )
        dataset.name = _resolved_dataset_name(dataset_config)
        dataset.freq = dataset_config.freq
        dataset.metadata = JsonObject(
            {
                **dataset.metadata.to_builtin(),
                "config_dataset_name": dataset_config.name,
                "dataset_layout": dataset_config.layout,
                "dataset_semantics": dataset_config.semantics.to_builtin(),
            }
        )
        return dataset

    if isinstance(provider, (CsvDatasetProviderConfig, ParquetDatasetProviderConfig)):
        loader_params = provider.config_payload()
        resolved_path = _resolve_data_path(source_path, provider.path)
        loader_params["path"] = resolved_path
        loader_params["layout"] = dataset_config.layout
        if dataset_config.time_column:
            loader_params["date_column"] = dataset_config.time_column
        if dataset_config.layout == "wide" and dataset_config.target_columns:
            loader_params["asset_columns"] = list(dataset_config.target_columns)
        if dataset_config.layout == "long":
            if dataset_config.series_id_columns:
                loader_params["series_id_columns"] = list(dataset_config.series_id_columns)
            if loader_params.get("value_column") is None and dataset_config.target_columns:
                loader_params["value_column"] = dataset_config.target_columns[0]
        semantics = dataset_config.semantics.to_builtin()
        if semantics.get("target_kind") is not None and loader_params.get("value_type") is None:
            loader_params["value_type"] = semantics.get("target_kind")
        if semantics.get("return_kind") is not None and loader_params.get("return_kind") is None:
            loader_params["return_kind"] = semantics.get("return_kind")
        return make_tabular_benchmark_dataset(
            dataset_name=_resolved_dataset_name(dataset_config),
            source=provider.kind,
            path=str(resolved_path),
            freq=dataset_config.freq,
            protocol=protocol,
            params=loader_params,
        )

    raise NotImplementedError(f"Unsupported dataset provider: {provider.kind}")
