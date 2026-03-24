"""Run-execution helpers for benchmark orchestration."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from ..benchmark.definition import BenchmarkConfig
from ..results import ResolvedRunExecution, RunRecord, RunTrackingRecord


def resolve_execution_devices(requested: str | None) -> list[str]:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a benchmark dependency
        torch = None

    def _available_cuda_devices() -> list[str]:
        if torch is None or not torch.cuda.is_available():
            return []
        return [f"cuda:{index}" for index in range(int(torch.cuda.device_count()))]

    def _available_mps_devices() -> list[str]:
        if torch is None:
            return []
        backends = getattr(torch, "backends", None)
        mps_backend = None if backends is None else getattr(backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return ["mps"]
        return []

    def _normalize_token(token: str) -> list[str]:
        text = token.strip()
        if not text or text == "auto":
            return _available_cuda_devices() or _available_mps_devices() or ["cpu"]
        if text in {"cuda", "cuda:all", "all_cuda"}:
            return _available_cuda_devices() or ["cpu"]
        return [text]

    if requested is None:
        return _normalize_token("auto")

    devices: list[str] = []
    for token in str(requested).split(","):
        devices.extend(_normalize_token(token))

    unique: list[str] = []
    seen: set[str] = set()
    for device in devices:
        if device not in seen:
            seen.add(device)
            unique.append(device)
    return unique or ["cpu"]


def should_parallelize_models(config: BenchmarkConfig, devices: list[str]) -> bool:
    models = config.models
    scheduler = str(config.run.scheduler or "auto")
    if scheduler == "sequential":
        return False
    if len(models) <= 1 or len(devices) <= 1:
        return False
    if scheduler == "model_parallel":
        return True
    return any(device.startswith("cuda:") for device in devices)


def format_devices_for_metadata(devices: list[str]) -> str | None:
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    return ",".join(devices)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_record(
    *,
    config: BenchmarkConfig,
    output_dir,
    resolved_execution: ResolvedRunExecution,
    status: str,
    requested_at: str,
    started_at: str,
    finished_at: str | None,
    tracking_result: RunTrackingRecord | None = None,
) -> RunRecord:
    return RunRecord(
        name=config.run.name,
        description=config.run.description,
        seed=config.run.seed,
        device=config.run.device,
        scheduler=config.run.scheduler,
        output=copy.deepcopy(config.run.output),
        tracking=copy.deepcopy(config.run.tracking),
        metadata=config.run.metadata,
        status=status,
        requested_at=requested_at,
        started_at=started_at,
        finished_at=finished_at,
        resolved_output_dir=output_dir,
        resolved_execution=resolved_execution,
        tracking_result=tracking_result,
    )
