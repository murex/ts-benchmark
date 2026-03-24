"""Environment discovery helpers for the Streamlit UI."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pandas as pd

from ts_benchmark.model.catalog.plugins import clear_plugin_caches, list_model_plugins
from ts_benchmark.tracking import mlflow_available


def _normalize_plugin_source_label(source: Any) -> str:
    text = str(source or "").strip().lower()
    if text in {"entry_point", "entrypoint"}:
        return "plugin"
    if not text:
        return ""
    return text


def _run_text_command(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
    return output or None


def detect_cpu_count() -> int:
    for command in (["nproc"], ["getconf", "_NPROCESSORS_ONLN"]):
        output = _run_text_command(command)
        if output is None:
            continue
        try:
            count = int(output.splitlines()[0].strip())
        except Exception:
            continue
        if count > 0:
            return count
    fallback = os.cpu_count()
    return 0 if fallback is None else int(fallback)


def detect_gpu_inventory() -> list[dict[str, Any]]:
    output = _run_text_command(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"])
    if output is None:
        return []
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
        except Exception:
            continue
        rows.append({"index": index, "name": parts[1], "device": f"cuda:{index}"})
    return rows


def detect_gpu_count() -> int:
    return len(detect_gpu_inventory())


def detect_devices() -> list[str]:
    devices = ["auto", "cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.extend([f"cuda:{idx}" for idx in range(torch.cuda.device_count())])
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for device in devices:
        if device not in seen:
            seen.add(device)
            out.append(device)
    return out


def detect_python_executable() -> str:
    return sys.executable


def discover_plugins_df() -> pd.DataFrame:
    try:
        # Plugin packages can change during development. Clear cached entry-point
        # metadata so the UI reflects the current environment without requiring a
        # fresh Python process.
        clear_plugin_caches()
        plugins = list_model_plugins()
    except Exception as exc:
        frame = pd.DataFrame()
        frame.attrs["error"] = str(exc)
        return frame
    rows: list[dict[str, Any]] = []
    for name, info in sorted(plugins.items()):
        manifest = info.get("manifest") or {}
        capabilities = manifest.get("capabilities") or {}
        rows.append(
            {
                "plugin": name,
                "source": _normalize_plugin_source_label(info.get("source")),
                "display_name": manifest.get("display_name") or name,
                "family": manifest.get("family"),
                "version": manifest.get("version") or info.get("package_version"),
                "devices": ", ".join(manifest.get("runtime_device_hints") or []),
                "uses_benchmark_device": capabilities.get("uses_benchmark_device"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["error"] = None
    return frame


def check_mlflow_available() -> bool:
    return mlflow_available()


def check_streamlit_available() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        return False
    return True


def inspect_subprocess_envs(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    default_execution = dict(
        ((config.get("run") or {}).get("execution") or {}).get("model_execution") or {}
    )
    for model in config.get("benchmark", {}).get("models", []) or []:
        execution = dict(model.get("execution") or default_execution)
        rows.append(
            {
                "model": model.get("name"),
                "mode": execution.get("mode", "inprocess"),
                "venv": execution.get("venv"),
                "python": execution.get("python"),
                "cwd": execution.get("cwd"),
                "pythonpath": ", ".join(execution.get("pythonpath") or []),
                "env_keys": ", ".join(sorted((execution.get("env") or {}).keys())),
            }
        )
    return pd.DataFrame(rows)


def environment_summary() -> dict[str, Any]:
    gpu_inventory = detect_gpu_inventory()
    return {
        "python_executable": detect_python_executable(),
        "cwd": os.getcwd(),
        "cpu_count": detect_cpu_count(),
        "gpu_count": len(gpu_inventory),
        "gpu_inventory": gpu_inventory,
        "device_options": detect_devices(),
        "streamlit_available": check_streamlit_available(),
        "mlflow_available": check_mlflow_available(),
    }
