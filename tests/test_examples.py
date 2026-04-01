from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.model import ScenarioModel


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_public_api_is_small_and_model_contract_is_importable() -> None:
    import ts_benchmark
    import ts_benchmark.model_contract as model_contract

    assert sorted(ts_benchmark.__all__) == [
        "BenchmarkConfig",
        "BenchmarkDiagnostics",
        "BenchmarkResults",
        "BenchmarkRunArtifacts",
        "Protocol",
        "dump_benchmark_config",
        "list_benchmark_summaries",
        "load_benchmark_config",
        "run_benchmark_from_config",
        "summarize_benchmark",
        "validate_benchmark_config",
    ]
    assert hasattr(model_contract, "TSGeneratorEstimator")
    assert hasattr(model_contract, "TSSeries")
    assert hasattr(model_contract, "TrainExample")
    assert hasattr(model_contract, "TrainData")
    assert not hasattr(model_contract, "TSBatch")
    assert hasattr(model_contract, "GenerationMode")


def test_example_scripts_import_smoke() -> None:
    example_paths = [
        ROOT / "examples" / "run_config_benchmark.py",
        ROOT / "examples" / "run_csv_example.py",
        ROOT / "examples" / "run_custom_contract_model.py",
        ROOT / "examples" / "run_historical_baselines.py",
    ]
    for path in example_paths:
        module = _load_module(path, f"example_{path.stem}")
        assert hasattr(module, "ROOT")


def test_plugin_example_import_smoke() -> None:
    module = _load_module(
        ROOT
        / "plugin_examples"
        / "eqbench_demo_gaussian_plugin"
        / "src"
        / "eqbench_demo_gaussian_plugin"
        / "plugin.py",
        "plugin_example_demo_gaussian",
    )

    model = module.build_model()

    assert isinstance(model, ScenarioModel)
    assert module.build_model.__name__ == "build_model"


def test_examples_do_not_reference_removed_api_names() -> None:
    patterns = [
        "from ts_benchmark.config import",
        "import ts_benchmark.config",
        "from ts_benchmark.data import",
        "import ts_benchmark.data",
        "from ts_benchmark.evaluation import",
        "import ts_benchmark.evaluation",
        "from ts_benchmark.models import",
        "import ts_benchmark.models",
        "from ts_benchmark.plugins import",
        "import ts_benchmark.plugins",
        "metrics_table",
        "run_metadata",
    ]
    text_paths = [ROOT / "README.md"]
    text_paths.extend(sorted((ROOT / "examples").glob("*.py")))
    text_paths.extend(
        path
        for path in sorted((ROOT / "plugin_examples").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".toml"}
    )

    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern not in text, f"Found removed API reference {pattern!r} in {os.fspath(path)}"
