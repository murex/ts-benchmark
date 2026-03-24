"""Helpers for resolving model references from config declarations."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
from pathlib import Path
import sys
from typing import Any

from .catalog.plugins import BUILTIN_MODEL_FACTORIES, resolve_model_plugin
from .definition import ModelReferenceConfig

MODEL_REGISTRY = BUILTIN_MODEL_FACTORIES
PROJECT_ROOT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


def import_object(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise ValueError(
            "Entrypoints must use the format 'module.submodule:QualifiedName' or '/path/to/file.py:QualifiedName'."
        )
    module_name, attr_name = entrypoint.split(":", 1)
    module = _import_module_target(module_name)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def _import_module_target(module_target: str) -> Any:
    text = str(module_target).strip()
    path = Path(text).expanduser()
    if path.suffix == ".py" or path.exists():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Entrypoint file not found: {resolved}")
        if resolved.suffix != ".py":
            raise ValueError(f"Entrypoint file must point to a Python module: {resolved}")
        package_import = _derive_package_import(resolved)
        candidate_roots = _candidate_import_roots(resolved)
        if package_import is not None:
            import_root, module_name = package_import
            original_path = list(sys.path)
            try:
                for root in reversed([import_root, *candidate_roots]):
                    if str(root) not in sys.path:
                        sys.path.insert(0, str(root))
                importlib.invalidate_caches()
                if module_name in sys.modules:
                    return importlib.reload(sys.modules[module_name])
                return importlib.import_module(module_name)
            finally:
                sys.path[:] = original_path
        module_name = f"ts_benchmark.file_entrypoint_{hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from entrypoint file: {resolved}")
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            for root in reversed(candidate_roots):
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            sys.path[:] = original_path
    return importlib.import_module(text)


def _derive_package_import(file_path: Path) -> tuple[Path, str] | None:
    parts = [file_path.stem]
    parent = file_path.parent
    found_package = False
    while (parent / "__init__.py").exists():
        found_package = True
        parts.insert(0, parent.name)
        parent = parent.parent
    if not found_package:
        return None
    return parent, ".".join(parts)


def _candidate_import_roots(file_path: Path) -> list[Path]:
    roots: list[Path] = [file_path.parent]
    current = file_path.parent
    project_root: Path | None = None
    while current != current.parent:
        if any((current / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            project_root = current
            break
        current = current.parent
    if project_root is not None:
        roots.append(project_root)
        src_root = project_root / "src"
        if src_root.exists() and src_root.is_dir():
            roots.append(src_root)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def resolve_model_builder(
    *,
    reference: ModelReferenceConfig,
) -> Any:
    kind = str(reference.kind)
    value = str(reference.value)

    if kind == "builtin":
        if value not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown builtin model '{value}'. Supported: {sorted(MODEL_REGISTRY)}"
            )
        return MODEL_REGISTRY[value]

    if kind == "plugin":
        return resolve_model_plugin(value)

    if kind == "entrypoint":
        return import_object(value)

    raise ValueError(f"Unsupported model reference kind '{kind}'.")


def instantiate_model_target(
    *,
    reference: ModelReferenceConfig,
    params: dict[str, Any] | None = None,
) -> object:
    builder = resolve_model_builder(
        reference=reference,
    )
    payload = dict(params or {})
    if isinstance(builder, type):
        return builder(**payload)
    if callable(builder):
        return builder(**payload)
    raise TypeError(
        "Resolved model builder is neither a class nor a callable factory."
    )
