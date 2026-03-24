"""Static plugin manifests exposed through benchmark entry points."""

from __future__ import annotations

from ts_benchmark.model.catalog.plugins import ModelPluginManifest, PluginCapabilities


def _capabilities(*, uses_benchmark_device: bool) -> PluginCapabilities:
    return PluginCapabilities(
        multivariate=True,
        probabilistic_sampling=True,
        benchmark_protocol_contract=False,
        explicit_preprocessing=True,
        uses_benchmark_device=uses_benchmark_device,
    )


GLUONTS_DEEPVAR_MANIFEST = ModelPluginManifest(
    name="gluonts_deepvar",
    display_name="GluonTS DeepVAR",
    version="0.1.0",
    family="rnn",
    description="External adapter around GluonTS DeepVAR for multivariate probabilistic forecasting.",
    runtime_device_hints=("cpu", "cuda"),
    supported_dataset_sources=("synthetic", "csv", "parquet"),
    required_input="returns",
    default_pipeline="raw",
    tags=("official", "gluonts", "deepvar", "multivariate"),
    notes="Requires the optional GluonTS MXNet stack; adapter consumes forecast horizon and context length through the structural task contract.",
    capabilities=_capabilities(uses_benchmark_device=True),
    manifest_source="entry_point",
)

GLUONTS_GPVAR_MANIFEST = ModelPluginManifest(
    name="gluonts_gpvar",
    display_name="GluonTS GPVAR",
    version="0.1.0",
    family="gaussian-process",
    description="External adapter around GluonTS GPVAR for multivariate probabilistic forecasting.",
    runtime_device_hints=("cpu", "cuda"),
    supported_dataset_sources=("synthetic", "csv", "parquet"),
    required_input="returns",
    default_pipeline="raw",
    tags=("official", "gluonts", "gpvar", "multivariate"),
    notes="Requires the optional GluonTS MXNet stack; adapter consumes forecast horizon and context length through the structural task contract.",
    capabilities=_capabilities(uses_benchmark_device=True),
    manifest_source="entry_point",
)

PYTORCHTS_TIMEGRAD_MANIFEST = ModelPluginManifest(
    name="pytorchts_timegrad",
    display_name="pytorch-ts TimeGrad",
    version="0.1.0",
    family="diffusion",
    description="External adapter around pytorch-ts TimeGrad for multivariate probabilistic forecasting.",
    runtime_device_hints=("cpu", "cuda"),
    supported_dataset_sources=("synthetic", "csv", "parquet"),
    required_input="returns",
    default_pipeline="raw",
    tags=("official", "pytorchts", "timegrad", "multivariate", "diffusion"),
    notes="Requires the optional pytorch-ts TimeGrad dependencies; adapter consumes forecast horizon and context length through the structural task contract.",
    capabilities=_capabilities(uses_benchmark_device=True),
    manifest_source="entry_point",
)
