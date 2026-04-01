"""Packaged resource helpers for plugin manifest and parameter schema metadata."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, resources
import tomllib
from typing import Any, Mapping

PLUGIN_METADATA_RESOURCE_NAME = "ts_benchmark_plugin.toml"


@dataclass(frozen=True)
class PluginResourceMetadata:
    manifest: Mapping[str, Any] | None = None
    parameter_schema: Mapping[str, Any] | None = None
    package: str | None = None
    resource_name: str | None = None


def _candidate_resource_packages(module_name: str) -> tuple[str, ...]:
    candidates: list[str] = []
    current = module_name.strip()
    while current:
        candidates.append(current)
        if "." not in current:
            break
        current = current.rsplit(".", 1)[0]
    return tuple(candidates)


def _load_resource_payload(package_name: str) -> Mapping[str, Any] | None:
    try:
        resource_root = resources.files(package_name)
    except (ModuleNotFoundError, TypeError, AttributeError):
        return None

    resource_path = resource_root.joinpath(PLUGIN_METADATA_RESOURCE_NAME)
    try:
        if not resource_path.is_file():
            return None
        return tomllib.loads(resource_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"Invalid TOML in plugin metadata resource "
            f"'{package_name}:{PLUGIN_METADATA_RESOURCE_NAME}'."
        ) from exc


def _extract_plugin_section(
    payload: Mapping[str, Any],
    *,
    entrypoint_name: str,
) -> Mapping[str, Any] | None:
    if "manifest" in payload or "parameters" in payload:
        return payload

    for container_key in ("plugins", "models"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        section = container.get(entrypoint_name)
        if isinstance(section, Mapping):
            return section
    return None


def load_plugin_resource_metadata(
    entry_point: metadata.EntryPoint,
) -> PluginResourceMetadata | None:
    module_name = getattr(entry_point, "module", None) or str(entry_point.value).split(":", 1)[0]
    if not module_name:
        return None

    for package_name in _candidate_resource_packages(module_name):
        payload = _load_resource_payload(package_name)
        if payload is None:
            continue
        section = _extract_plugin_section(payload, entrypoint_name=entry_point.name)
        if section is None:
            continue
        manifest = section.get("manifest")
        parameter_schema = section.get("parameters")
        if manifest is None and parameter_schema is None:
            continue
        return PluginResourceMetadata(
            manifest=manifest if isinstance(manifest, Mapping) else None,
            parameter_schema=parameter_schema if isinstance(parameter_schema, Mapping) else None,
            package=package_name,
            resource_name=PLUGIN_METADATA_RESOURCE_NAME,
        )
    return None
