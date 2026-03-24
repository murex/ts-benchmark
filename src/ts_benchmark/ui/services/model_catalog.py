"""Persistent model-catalog helpers for the Streamlit UI."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ts_benchmark.model.catalog.plugins import (
    BUILTIN_MODEL_MANIFESTS,
    extract_model_parameter_schema,
    extract_model_plugin_manifest,
    get_model_plugin_info,
    resolve_model_plugin,
)
from ts_benchmark.model.resolution import import_object
from ts_benchmark.serialization import to_jsonable

from .. import APP_ROOT, MODEL_ADAPTER_UPLOAD_DIR, MODEL_CATALOG_DIR

SUPPORTED_MODEL_REFERENCE_KINDS = ("builtin", "plugin", "entrypoint")
IGNORED_PYTHON_FILE_DIRS = {
    "__pycache__",
    ".git",
    ".streamlit_runtime",
    ".streamlit_uploads",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _normalize_plugin_source_label(source: Any) -> str:
    text = str(source or "").strip().lower()
    if text in {"entry_point", "entrypoint"}:
        return "plugin"
    if not text:
        return ""
    return text


def default_catalog_model_dict() -> dict[str, Any]:
    return {
        "name": "",
        "description": "",
        "reference": {
            "kind": "entrypoint",
            "value": "",
        },
        "params": {},
        "metadata": {},
    }


def normalize_catalog_model(model: dict[str, Any] | None) -> dict[str, Any]:
    payload = copy.deepcopy(default_catalog_model_dict())
    if model:
        payload.update(copy.deepcopy(model))

    payload["name"] = str(payload.get("name") or "").strip()
    payload["description"] = str(payload.get("description") or "").strip()
    reference = dict(payload.get("reference") or {})
    kind = str(reference.get("kind") or "entrypoint").strip().lower()
    if kind not in SUPPORTED_MODEL_REFERENCE_KINDS:
        kind = "entrypoint"
    payload["reference"] = {
        "kind": kind,
        "value": str(reference.get("value") or "").strip(),
    }
    payload["params"] = dict(payload.get("params") or {})
    payload["metadata"] = dict(payload.get("metadata") or {})
    return payload


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return slug or "model"


def _reference_key(reference: dict[str, Any]) -> tuple[str, str]:
    normalized = normalize_catalog_model({"reference": reference})
    ref = normalized["reference"]
    return str(ref["kind"]), str(ref["value"])


def _builtin_catalog_model(name: str) -> dict[str, Any]:
    manifest = BUILTIN_MODEL_MANIFESTS[name]
    return normalize_catalog_model(
        {
            "name": name,
            "description": manifest.description or "",
            "reference": {
                "kind": "builtin",
                "value": name,
            },
            "params": {},
            "metadata": {},
        }
    )


def builtin_catalog_models() -> list[dict[str, Any]]:
    return [_builtin_catalog_model(name) for name in sorted(BUILTIN_MODEL_MANIFESTS)]


def saved_model_paths(model_dir: Path = MODEL_CATALOG_DIR) -> dict[str, Path]:
    return {path.stem: path for path in sorted(model_dir.glob("*.json"))}


def _resolve_saved_model_path(name: str, model_dir: Path = MODEL_CATALOG_DIR) -> Path | None:
    target = str(name).strip()
    if not target:
        return None
    slug_match = model_dir / f"{_slugify_name(target)}.json"
    if slug_match.exists():
        return slug_match
    for path in saved_model_paths(model_dir).values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("name") or "").strip() == target:
            return path
    return None


def list_saved_catalog_models(model_dir: Path = MODEL_CATALOG_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in saved_model_paths(model_dir).values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(normalize_catalog_model(payload))
    return rows


def load_catalog_model(name: str, model_dir: Path = MODEL_CATALOG_DIR) -> dict[str, Any]:
    target = str(name).strip()
    if target in BUILTIN_MODEL_MANIFESTS:
        return _builtin_catalog_model(target)
    path = _resolve_saved_model_path(target, model_dir=model_dir)
    if path is None:
        raise FileNotFoundError(f"No catalog model named '{name}'.")
    return normalize_catalog_model(json.loads(path.read_text(encoding="utf-8")))


def _existing_catalog_models(model_dir: Path = MODEL_CATALOG_DIR) -> list[dict[str, Any]]:
    return builtin_catalog_models() + list_saved_catalog_models(model_dir=model_dir)


def save_catalog_model(model: dict[str, Any], model_dir: Path = MODEL_CATALOG_DIR) -> Path:
    payload = normalize_catalog_model(model)
    name = payload["name"]
    if not name:
        raise ValueError("Catalog models require a non-empty name.")

    reference = payload["reference"]
    if not reference["value"]:
        raise ValueError("Catalog models require a non-empty reference value.")
    if reference["kind"] == "builtin":
        raise ValueError("Built-in models are already part of the catalog.")
    if name in BUILTIN_MODEL_MANIFESTS:
        raise ValueError(f"'{name}' is reserved for a built-in catalog model.")

    reference_key = _reference_key(reference)
    for entry in _existing_catalog_models(model_dir=model_dir):
        if entry["name"] == name:
            continue
        if _reference_key(entry["reference"]) == reference_key:
            raise ValueError(
                f"Model reference '{reference['kind']}:{reference['value']}' is already cataloged as '{entry['name']}'."
            )

    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{_slugify_name(name)}.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return path


def delete_catalog_model(name: str, model_dir: Path = MODEL_CATALOG_DIR) -> Path:
    target = str(name).strip()
    if target in BUILTIN_MODEL_MANIFESTS:
        raise ValueError("Built-in catalog models cannot be removed.")
    path = _resolve_saved_model_path(target, model_dir=model_dir)
    if path is None:
        raise FileNotFoundError(f"No catalog model named '{name}'.")
    path.unlink()
    return path


def list_entrypoint_python_files(search_root: str | Path = APP_ROOT) -> list[Path]:
    root = Path(search_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Search root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Search root must be a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in IGNORED_PYTHON_FILE_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        files.append(path)
    return sorted(files)


def _entrypoint_symbol_sort_key(symbol: dict[str, str]) -> tuple[int, str]:
    name = str(symbol.get("name") or "")
    kind = str(symbol.get("kind") or "")
    lowered = name.lower()
    if bool(symbol.get("extends_scenario_model")):
        return (0, lowered)
    if kind == "function" and lowered.startswith(("build_", "make_", "create_")):
        return (1, lowered)
    if kind == "class" and lowered.endswith(("adapter", "model")):
        return (2, lowered)
    if kind == "function":
        return (3, lowered)
    return (4, lowered)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return str(node.id)
    if isinstance(node, ast.Attribute):
        return str(node.attr)
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    return ""


def _extends_scenario_model(node: ast.ClassDef) -> bool:
    return any(_base_name(base) == "ScenarioModel" for base in node.bases)


def inspect_entrypoint_python_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Entrypoint file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Entrypoint file must be a Python module: {file_path}")

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    symbols: list[dict[str, str]] = []
    scenario_model_classes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            symbols.append({"name": node.name, "kind": "function"})
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            extends_scenario_model = _extends_scenario_model(node)
            symbols.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "extends_scenario_model": extends_scenario_model,
                }
            )
            if extends_scenario_model:
                scenario_model_classes.append(node.name)

    symbols = sorted(symbols, key=_entrypoint_symbol_sort_key)
    return {
        "path": str(file_path),
        "symbols": symbols,
        "scenario_model_classes": scenario_model_classes,
    }


def find_repo_scenario_model_candidates(search_root: str | Path) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for path in list_entrypoint_python_files(search_root=search_root):
        try:
            inspection = inspect_entrypoint_python_file(path)
        except Exception:
            continue
        for symbol in inspection["symbols"]:
            if not bool(symbol.get("extends_scenario_model")):
                continue
            candidates.append(
                {
                    "path": inspection["path"],
                    "symbol": str(symbol["name"]),
                }
            )
    return candidates


def build_file_entrypoint_value(path: str | Path, symbol: str) -> str:
    file_path = Path(path).expanduser().resolve()
    if not str(symbol).strip():
        raise ValueError("Entrypoint symbols require a non-empty name.")
    return f"{file_path}:{str(symbol).strip()}"


def store_uploaded_entrypoint_file(
    *,
    filename: str,
    content: bytes,
    upload_dir: Path = MODEL_ADAPTER_UPLOAD_DIR,
) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix != ".py":
        raise ValueError("Adapter uploads must be Python files.")
    digest = hashlib.sha1(content).hexdigest()[:12]
    target_name = f"{_slugify_name(Path(filename).stem)}_{digest}{suffix}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / target_name
    if not target.exists():
        target.write_bytes(content)
    return target.resolve()


def describe_catalog_model_entry(model: dict[str, Any]) -> dict[str, Any]:
    entry = normalize_catalog_model(model)
    reference = entry["reference"]
    built_in = reference["kind"] == "builtin" and reference["value"] in BUILTIN_MODEL_MANIFESTS
    detail = {
        "entry": entry,
        "built_in": built_in,
        "removable": not built_in,
        "manifest": {},
        "resolution": {},
        "parameter_schema": {},
        "parameters": [],
    }

    try:
        if reference["kind"] in {"builtin", "plugin"}:
            info = get_model_plugin_info(reference["value"])
            builder = resolve_model_plugin(reference["value"])
            detail["manifest"] = {} if info.manifest is None else to_jsonable(info.manifest)
            detail["resolution"] = {
                "status": "available",
                "source": _normalize_plugin_source_label(info.source),
                "target": info.target,
                "package": info.package,
                "version": info.package_version,
            }
        elif reference["kind"] == "entrypoint":
            builder = import_object(reference["value"])
            manifest_obj = extract_model_plugin_manifest(builder, default_name=entry["name"])
            detail["manifest"] = {} if manifest_obj is None else to_jsonable(manifest_obj)
            detail["resolution"] = {
                "status": "importable",
                "target": reference["value"],
            }
        else:
            raise ValueError(f"Unsupported model reference kind '{reference['kind']}'.")
        parameter_schema = extract_model_parameter_schema(builder, default_name=entry["name"])
        detail["parameter_schema"] = {} if parameter_schema is None else to_jsonable(parameter_schema)
        detail["parameters"] = [] if parameter_schema is None else list(detail["parameter_schema"].get("fields") or [])
    except Exception as exc:
        detail["resolution"] = {
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    return detail


def describe_catalog_model(name: str, model_dir: Path = MODEL_CATALOG_DIR) -> dict[str, Any]:
    return describe_catalog_model_entry(load_catalog_model(name, model_dir=model_dir))


def list_model_catalog(model_dir: Path = MODEL_CATALOG_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in _existing_catalog_models(model_dir=model_dir):
        detail = describe_catalog_model_entry(entry)
        manifest = detail["manifest"]
        resolution = detail["resolution"]
        rows.append(
            {
                "name": entry["name"],
                "display_name": manifest.get("display_name") or entry["name"],
                "description": entry["description"] or manifest.get("description") or "",
                "origin": "Built-in" if detail["built_in"] else "Catalog",
                "reference_kind": entry["reference"]["kind"],
                "reference_value": entry["reference"]["value"],
                "family": manifest.get("family") or "",
                "version": manifest.get("version") or resolution.get("version") or "",
                "status": resolution.get("status", "error" if resolution.get("error") else ""),
                "removable": detail["removable"],
            }
        )
    return rows
