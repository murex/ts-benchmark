from __future__ import annotations

import importlib
from importlib import metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.model.catalog import plugins
from ts_benchmark.model.catalog.plugins import (
    MODEL_ENTRYPOINT_GROUP,
    MODEL_MANIFEST_ENTRYPOINT_GROUP,
    clear_plugin_caches,
    resolve_model_plugin_manifest,
    resolve_model_plugin_parameter_schema,
)


def _write_plugin_package(tmp_path: Path, package_name: str, *, plugin_source: str, toml_source: str | None) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(plugin_source, encoding="utf-8")
    if toml_source is not None:
        (package_dir / "ts_benchmark_plugin.toml").write_text(toml_source, encoding="utf-8")


def _patch_entry_points(
    monkeypatch,
    *,
    model_entry_points: list[metadata.EntryPoint],
    manifest_entry_points: list[metadata.EntryPoint] | None = None,
) -> None:
    manifest_entry_points = manifest_entry_points or []

    def fake_entry_points(*, group=None):
        if group == MODEL_ENTRYPOINT_GROUP:
            return list(model_entry_points)
        if group == MODEL_MANIFEST_ENTRYPOINT_GROUP:
            return list(manifest_entry_points)
        return []

    monkeypatch.setattr(plugins.metadata, "entry_points", fake_entry_points)


def test_plugin_resource_metadata_overrides_legacy_python_sources(monkeypatch, tmp_path: Path) -> None:
    _write_plugin_package(
        tmp_path,
        "demo_resource_plugin",
        plugin_source="""
def build_model(ridge: float = 0.1):
    return {"ridge": ridge}

def get_plugin_manifest():
    return {
        "display_name": "Legacy Python manifest",
        "default_pipeline": "raw",
    }

build_model.PARAMETER_SCHEMA = {
    "name": "demo_plugin",
    "fields": [
        {"name": "legacy_param", "value_type": "int", "default": 3},
    ],
}
""".strip(),
        toml_source="""
[plugins.demo_plugin.manifest]
display_name = "Resource file manifest"
default_pipeline = "minmax"
tags = ["resource"]

[plugins.demo_plugin.manifest.capabilities]
explicit_preprocessing = true

[plugins.demo_plugin.parameters]
name = "demo_plugin"

[[plugins.demo_plugin.parameters.fields]]
name = "ridge"
value_type = "float"
default = 0.25
description = "Ridge coefficient from packaged metadata."
""".strip(),
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _patch_entry_points(
        monkeypatch,
        model_entry_points=[
            metadata.EntryPoint(
                "demo_plugin",
                "demo_resource_plugin.plugin:build_model",
                MODEL_ENTRYPOINT_GROUP,
            )
        ],
        manifest_entry_points=[
            metadata.EntryPoint(
                "demo_plugin",
                "demo_resource_plugin.plugin:get_plugin_manifest",
                MODEL_MANIFEST_ENTRYPOINT_GROUP,
            )
        ],
    )

    clear_plugin_caches()
    manifest = resolve_model_plugin_manifest("demo_plugin")
    schema = resolve_model_plugin_parameter_schema("demo_plugin")

    assert manifest is not None
    assert manifest.display_name == "Resource file manifest"
    assert manifest.default_pipeline == "minmax"
    assert manifest.manifest_source == "resource_file"
    assert manifest.capabilities.explicit_preprocessing is True

    assert schema is not None
    assert schema.schema_source == "resource_file"
    assert [(field.name, field.value_type, field.default) for field in schema.fields] == [
        ("ridge", "float", 0.25)
    ]


def test_plugin_resource_metadata_ignores_broken_legacy_manifest_entrypoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_plugin_package(
        tmp_path,
        "broken_legacy_manifest_plugin",
        plugin_source="""
def build_model(beta: float = 0.75):
    return {"beta": beta}
""".strip(),
        toml_source="""
[manifest]
display_name = "Broken legacy manifest fallback"
default_pipeline = "raw"
""".strip(),
    )

    broken_manifest_package = tmp_path / "broken_manifest_provider"
    broken_manifest_package.mkdir()
    (broken_manifest_package / "__init__.py").write_text("", encoding="utf-8")
    (broken_manifest_package / "plugin.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _patch_entry_points(
        monkeypatch,
        model_entry_points=[
            metadata.EntryPoint(
                "broken_plugin",
                "broken_legacy_manifest_plugin.plugin:build_model",
                MODEL_ENTRYPOINT_GROUP,
            )
        ],
        manifest_entry_points=[
            metadata.EntryPoint(
                "broken_plugin",
                "broken_manifest_provider.plugin:get_plugin_manifest",
                MODEL_MANIFEST_ENTRYPOINT_GROUP,
            )
        ],
    )

    clear_plugin_caches()
    manifest = resolve_model_plugin_manifest("broken_plugin")

    assert manifest is not None
    assert manifest.display_name == "Broken legacy manifest fallback"
    assert manifest.manifest_source == "resource_file"


def test_plugin_resource_metadata_supports_single_plugin_top_level_format(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_plugin_package(
        tmp_path,
        "single_resource_plugin",
        plugin_source="""
def build_model(alpha: float = 1.5):
    return {"alpha": alpha}
""".strip(),
        toml_source="""
[manifest]
display_name = "Top level resource manifest"
default_pipeline = "standardized"

[parameters]
name = "single_plugin"

[[parameters.fields]]
name = "alpha"
value_type = "float"
default = 1.5
""".strip(),
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _patch_entry_points(
        monkeypatch,
        model_entry_points=[
            metadata.EntryPoint(
                "single_plugin",
                "single_resource_plugin.plugin:build_model",
                MODEL_ENTRYPOINT_GROUP,
            )
        ],
    )

    clear_plugin_caches()
    manifest = resolve_model_plugin_manifest("single_plugin")
    schema = resolve_model_plugin_parameter_schema("single_plugin")

    assert manifest is not None
    assert manifest.display_name == "Top level resource manifest"
    assert manifest.default_pipeline == "standardized"
    assert manifest.manifest_source == "resource_file"

    assert schema is not None
    assert schema.schema_source == "resource_file"
    assert [(field.name, field.value_type, field.default) for field in schema.fields] == [
        ("alpha", "float", 1.5)
    ]


def test_example_plugin_uses_packaged_toml_metadata(monkeypatch) -> None:
    example_src = (
        ROOT
        / "plugin_examples"
        / "eqbench_demo_gaussian_plugin"
        / "src"
    )
    monkeypatch.syspath_prepend(str(example_src))
    importlib.invalidate_caches()
    _patch_entry_points(
        monkeypatch,
        model_entry_points=[
            metadata.EntryPoint(
                "demo_gaussian_plugin",
                "eqbench_demo_gaussian_plugin.plugin:build_model",
                MODEL_ENTRYPOINT_GROUP,
            )
        ],
    )

    clear_plugin_caches()
    manifest = resolve_model_plugin_manifest("demo_gaussian_plugin")
    schema = resolve_model_plugin_parameter_schema("demo_gaussian_plugin")

    assert manifest is not None
    assert manifest.display_name == "Demo Gaussian plugin"
    assert manifest.default_pipeline == "raw"
    assert manifest.manifest_source == "resource_file"

    assert schema is not None
    assert schema.schema_source == "resource_file"
    assert [(field.name, field.value_type, field.default) for field in schema.fields] == [
        ("ridge", "float", 1e-06)
    ]


def test_official_adapters_use_packaged_toml_manifest(monkeypatch) -> None:
    adapters_src = ROOT / "official_adapters" / "src"
    monkeypatch.syspath_prepend(str(adapters_src))
    importlib.invalidate_caches()
    _patch_entry_points(
        monkeypatch,
        model_entry_points=[
            metadata.EntryPoint(
                "gluonts_deepvar",
                "ts_benchmark_official_adapters.plugin:build_gluonts_deepvar",
                MODEL_ENTRYPOINT_GROUP,
            ),
            metadata.EntryPoint(
                "gluonts_gpvar",
                "ts_benchmark_official_adapters.plugin:build_gluonts_gpvar",
                MODEL_ENTRYPOINT_GROUP,
            ),
            metadata.EntryPoint(
                "pytorchts_timegrad",
                "ts_benchmark_official_adapters.plugin:build_pytorchts_timegrad",
                MODEL_ENTRYPOINT_GROUP,
            ),
        ],
    )

    clear_plugin_caches()
    manifest = resolve_model_plugin_manifest("pytorchts_timegrad")
    schema = resolve_model_plugin_parameter_schema("pytorchts_timegrad")

    assert manifest is not None
    assert manifest.display_name == "pytorch-ts TimeGrad"
    assert manifest.default_pipeline == "raw"
    assert manifest.manifest_source == "resource_file"
    assert manifest.capabilities.probabilistic_sampling is True

    assert schema is not None
    assert schema.schema_source == "config_dataclass"
    assert any(field.name == "hidden_size" and field.value_type == "int" for field in schema.fields)
