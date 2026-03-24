"""Persistent subprocess-backed scenario model proxy."""

from __future__ import annotations

import atexit
import json
import os
import pickle
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..contracts import ScenarioModel, ScenarioRequest, ScenarioSamples, TrainingData
from ..definition import ModelExecutionConfig, ModelReferenceConfig
from ...serialization import to_jsonable

_VENV_ROOT = Path(os.getenv("TS_BENCHMARK_VENV_ROOT", str(Path.home() / "venvs"))).expanduser()


def _resolve_path(raw: str | None, *, source_path: Path | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if source_path is not None:
        return (source_path.parent / path).resolve()
    return path.resolve()


def _resolve_venv_python(raw: str, *, source_path: Path | None) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        if any(sep in raw for sep in ("/", os.sep)):
            candidate = _resolve_path(raw, source_path=source_path) or candidate
        else:
            candidate = (_VENV_ROOT / raw).resolve()
    if candidate.is_file():
        return candidate
    python_path = candidate / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(
            f"Could not find a Python executable for execution.venv='{raw}' "
            f"(looked for '{python_path}')."
        )
    return python_path


class ExternalProcessScenarioModel(ScenarioModel):
    """Run a benchmark model in a dedicated Python interpreter."""

    def __init__(
        self,
        *,
        name: str,
        reference: ModelReferenceConfig,
        params: dict[str, Any],
        execution: ModelExecutionConfig,
        source_path: Path | None = None,
    ):
        self.name = name
        self.reference = reference
        self.params = dict(params)
        self.execution = execution
        self.source_path = source_path

        self._tmpdir = tempfile.TemporaryDirectory(prefix=f"tsbench-model-{name}-")
        self._stderr_path = Path(self._tmpdir.name) / "worker.stderr.log"
        self._stderr_handle: Any | None = None
        self._process: subprocess.Popen[str] | None = None
        self._cached_model_info: dict[str, Any] | None = None
        self._cached_debug_artifacts: dict[str, Any] | None = None
        self._rpc_counter = 0
        atexit.register(self.close)

    def _python_executable(self) -> Path:
        if self.execution.python:
            return _resolve_path(self.execution.python, source_path=self.source_path) or Path(sys.executable)
        if self.execution.venv:
            return _resolve_venv_python(self.execution.venv, source_path=self.source_path)
        return Path(sys.executable)

    def _cwd(self) -> Path | None:
        return _resolve_path(self.execution.cwd, source_path=self.source_path)

    def _pythonpath(self) -> list[str]:
        resolved: list[str] = []
        for item in self.execution.pythonpath:
            path = _resolve_path(item, source_path=self.source_path)
            resolved.append(str(path if path is not None else item))
        return resolved

    def _worker_command(self) -> list[str]:
        if self.execution.venv:
            venv_root = self._venv_root()
            if venv_root is None:
                raise RuntimeError(f"Could not resolve venv root for '{self.execution.venv}'.")
            activate = venv_root / "bin" / "activate"
            command = (
                f"source {shlex.quote(str(activate))} && "
                "exec python -m ts_benchmark.model.wrappers.worker"
            )
            return ["bash", "-lc", command]
        return [str(self._python_executable()), "-m", "ts_benchmark.model.wrappers.worker"]

    def _venv_root(self) -> Path | None:
        python_exe = self._python_executable()
        if python_exe.parent.name == "bin":
            return python_exe.parent.parent
        return None

    def _venv_site_packages(self) -> list[str]:
        root = self._venv_root()
        if root is None:
            return []
        lib_dir = root / "lib"
        out: list[str] = []
        if lib_dir.exists():
            for candidate in sorted(lib_dir.glob("python*/site-packages")):
                out.append(str(candidate))
        return out

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath = self._venv_site_packages() + self._pythonpath()
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath.append(existing)
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        backend = str(env.get("MPLBACKEND") or "").strip()
        if backend.startswith("module://matplotlib_inline"):
            env["MPLBACKEND"] = "Agg"
        venv_root = self._venv_root()
        if venv_root is not None:
            env.setdefault("VIRTUAL_ENV", str(venv_root))
        for key, value in dict(self.execution.env).items():
            env[str(key)] = str(value)
        return env

    def _stderr_tail(self, max_lines: int = 120) -> str:
        if not self._stderr_path.exists():
            return ""
        lines = self._stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def _failure(self, prefix: str, *, response: dict[str, Any] | None = None) -> RuntimeError:
        details: list[str] = [prefix]
        if response is not None:
            error_type = response.get("error_type")
            message = response.get("message")
            if error_type or message:
                details.append(f"{error_type or 'Error'}: {message}")
            traceback_text = response.get("traceback")
            if traceback_text:
                details.append(str(traceback_text))
        stderr_tail = self._stderr_tail()
        if stderr_tail:
            details.append("Worker stderr tail:")
            details.append(stderr_tail)
        return RuntimeError("\n".join(details))

    def _spec_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reference": to_jsonable(self.reference),
            "params": self.params,
        }

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._stderr_handle = self._stderr_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            self._worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            bufsize=1,
            cwd=None if self._cwd() is None else str(self._cwd()),
            env=self._worker_env(),
        )
        self._send_command({"action": "init", "model_spec": self._spec_payload()})

    def _send_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_started()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        if self._process.poll() is not None:
            raise self._failure("External model worker exited before handling the request.")

        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

        line = self._process.stdout.readline()
        if not line:
            raise self._failure("External model worker closed stdout unexpectedly.")

        response = json.loads(line)
        if response.get("status") != "ok":
            raise self._failure(
                f"External model worker failed during '{payload.get('action')}'.",
                response=response,
            )
        return response

    def _pickle_path(self, stem: str) -> Path:
        self._rpc_counter += 1
        return Path(self._tmpdir.name) / f"{self._rpc_counter:05d}-{stem}-{uuid.uuid4().hex}.pkl"

    def fit(self, train_data: TrainingData) -> "ExternalProcessScenarioModel":
        train_data.validate()
        input_path = self._pickle_path("fit-input")
        with input_path.open("wb") as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._send_command({"action": "fit", "input_path": str(input_path)})
        self._cached_model_info = self._send_command({"action": "model_info"}).get("payload") or {}
        return self

    def sample(self, request: ScenarioRequest) -> ScenarioSamples:
        request.validate()
        input_path = self._pickle_path("sample-input")
        output_path = self._pickle_path("sample-output")
        with input_path.open("wb") as f:
            pickle.dump(request, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._send_command(
            {
                "action": "sample",
                "input_path": str(input_path),
                "output_path": str(output_path),
            }
        )
        with output_path.open("rb") as f:
            result = pickle.load(f)
        if not isinstance(result, ScenarioSamples):
            raise TypeError(
                "External model worker returned an unexpected sample payload type: "
                f"{type(result)!r}"
            )
        return result

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name,
            "class": self.__class__.__name__,
            "execution_mode": "subprocess",
            "python_executable": str(self._python_executable()),
            "execution": {
                "mode": self.execution.mode,
                "venv": self.execution.venv,
                "python": self.execution.python,
                "cwd": self.execution.cwd,
                "pythonpath": list(self.execution.pythonpath),
                "env": dict(self.execution.env),
            },
        }
        if self._cached_model_info:
            info["child_model_info"] = dict(self._cached_model_info)
            for key, value in self._cached_model_info.items():
                info.setdefault(key, value)
        return info

    def debug_artifacts(self) -> dict[str, Any] | None:
        payload = self._send_command({"action": "debug_artifacts"}).get("payload")
        self._cached_debug_artifacts = None if payload is None else dict(payload)
        stderr_text = self._stderr_path.read_text(encoding="utf-8", errors="replace") if self._stderr_path.exists() else ""
        if self._cached_debug_artifacts is None and not stderr_text:
            return None
        return {
            "name": self.name,
            "execution_mode": "subprocess",
            "python_executable": str(self._python_executable()),
            "stderr_log_path": str(self._stderr_path),
            "stderr_text": stderr_text,
            "child_debug_artifacts": self._cached_debug_artifacts,
        }

    def close(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        assert process.stdin is not None
                        assert process.stdout is not None
                        process.stdin.write(json.dumps({"action": "close"}) + "\n")
                        process.stdin.flush()
                        process.stdout.readline()
                    except Exception:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.stdout is not None:
                    process.stdout.close()
        self._process = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
