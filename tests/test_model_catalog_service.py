from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui.services import model_catalog


def test_list_model_catalog_includes_builtins(tmp_path: Path) -> None:
    rows = model_catalog.list_model_catalog(model_dir=tmp_path)

    ewma = next(row for row in rows if row["name"] == "ewma_gaussian")
    fhs = next(row for row in rows if row["name"] == "filtered_historical_simulation")
    historical = next(row for row in rows if row["name"] == "historical_bootstrap")
    student_t = next(row for row in rows if row["name"] == "student_t_covariance")

    assert ewma["origin"] == "Built-in"
    assert ewma["reference_kind"] == "builtin"
    assert ewma["removable"] is False
    assert fhs["origin"] == "Built-in"
    assert fhs["reference_kind"] == "builtin"
    assert fhs["removable"] is False
    assert student_t["origin"] == "Built-in"
    assert student_t["reference_kind"] == "builtin"
    assert student_t["removable"] is False
    assert historical["origin"] == "Built-in"
    assert historical["reference_kind"] == "builtin"
    assert historical["removable"] is False


def test_save_load_and_delete_saved_catalog_model(tmp_path: Path) -> None:
    path = model_catalog.save_catalog_model(
        {
            "name": "debug_entrypoint",
            "description": "Debug smoke entrypoint",
            "reference": {
                "kind": "entrypoint",
                "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
            },
        },
        model_dir=tmp_path,
    )

    loaded = model_catalog.load_catalog_model("debug_entrypoint", model_dir=tmp_path)
    listed = model_catalog.list_model_catalog(model_dir=tmp_path)

    assert path.exists()
    assert loaded["reference"]["kind"] == "entrypoint"
    assert loaded["reference"]["value"].endswith(":DebugSmokeModel")
    assert any(row["name"] == "debug_entrypoint" and row["origin"] == "Catalog" for row in listed)

    deleted_path = model_catalog.delete_catalog_model("debug_entrypoint", model_dir=tmp_path)

    assert deleted_path == path
    assert not path.exists()


def test_save_catalog_model_rejects_builtin_and_duplicate_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="already part of the catalog"):
        model_catalog.save_catalog_model(
            {
                "name": "historical_bootstrap",
                "reference": {
                    "kind": "builtin",
                    "value": "historical_bootstrap",
                },
            },
            model_dir=tmp_path,
        )

    model_catalog.save_catalog_model(
        {
            "name": "debug_entrypoint",
            "reference": {
                "kind": "entrypoint",
                "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
            },
        },
        model_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="already cataloged as 'debug_entrypoint'"):
        model_catalog.save_catalog_model(
            {
                "name": "duplicate_debug",
                "reference": {
                    "kind": "entrypoint",
                    "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
                },
            },
            model_dir=tmp_path,
        )


def test_describe_catalog_model_reports_builtin_parameters(tmp_path: Path) -> None:
    detail = model_catalog.describe_catalog_model("historical_bootstrap", model_dir=tmp_path)

    assert detail["resolution"]["status"] == "available"
    assert detail["parameter_schema"]["schema_source"] == "call_signature"
    assert any(parameter["name"] == "block_size" for parameter in detail["parameters"])
    assert all(parameter["parameter_type"] == "explicit" for parameter in detail["parameters"])


def test_describe_catalog_model_reports_entrypoint_parameters(tmp_path: Path) -> None:
    model_catalog.save_catalog_model(
        {
            "name": "debug_entrypoint",
            "reference": {
                "kind": "entrypoint",
                "value": "ts_benchmark.model.builtins.debug_smoke_model:DebugSmokeModel",
            },
        },
        model_dir=tmp_path,
    )

    detail = model_catalog.describe_catalog_model("debug_entrypoint", model_dir=tmp_path)

    assert detail["resolution"]["status"] == "importable"
    assert detail["parameter_schema"]["schema_source"] == "call_signature"
    assert any(parameter["name"] == "scale" for parameter in detail["parameters"])


def test_inspect_entrypoint_python_file_finds_public_symbols() -> None:
    inspection = model_catalog.inspect_entrypoint_python_file(
        ROOT / "plugin_examples" / "eqbench_demo_gaussian_plugin" / "src" / "eqbench_demo_gaussian_plugin" / "plugin.py"
    )

    assert inspection["path"].endswith("plugin.py")
    assert inspection["scenario_model_classes"] == ["GaussianPluginModel"]
    assert any(symbol["name"] == "build_model" and symbol["kind"] == "function" for symbol in inspection["symbols"])
    assert any(
        symbol["name"] == "GaussianPluginModel"
        and symbol["kind"] == "class"
        and symbol["extends_scenario_model"] is True
        for symbol in inspection["symbols"]
    )


def test_store_uploaded_entrypoint_file_persists_python_upload(tmp_path: Path) -> None:
    stored = model_catalog.store_uploaded_entrypoint_file(
        filename="My Adapter.py",
        content=b"class DemoAdapter:\n    pass\n",
        upload_dir=tmp_path,
    )

    assert stored.parent == tmp_path
    assert stored.name.startswith("my_adapter_")
    assert stored.suffix == ".py"
    assert stored.read_text(encoding="utf-8") == "class DemoAdapter:\n    pass\n"


def test_list_entrypoint_python_files_accepts_custom_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "my_model_repo"
    repo_root.mkdir()
    (repo_root / "adapter.py").write_text("class DemoAdapter:\n    pass\n", encoding="utf-8")
    ignored_dir = repo_root / ".pytest_cache"
    ignored_dir.mkdir()
    (ignored_dir / "ignored.py").write_text("x = 1\n", encoding="utf-8")

    files = model_catalog.list_entrypoint_python_files(search_root=str(repo_root))

    assert files == [repo_root / "adapter.py"]


def test_inspect_entrypoint_python_file_marks_scenario_model_subclasses(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "from ts_benchmark.model import ScenarioModel\n"
        "\n"
        "class DemoScenarioModel(ScenarioModel):\n"
        "    pass\n",
        encoding="utf-8",
    )

    inspection = model_catalog.inspect_entrypoint_python_file(adapter)

    assert inspection["scenario_model_classes"] == ["DemoScenarioModel"]
    assert inspection["symbols"][0]["name"] == "DemoScenarioModel"
    assert inspection["symbols"][0]["extends_scenario_model"] is True


def test_find_repo_scenario_model_candidates_returns_detected_classes(tmp_path: Path) -> None:
    repo_root = tmp_path / "demo_repo"
    repo_root.mkdir()
    (repo_root / "alpha.py").write_text("class NotAModel:\n    pass\n", encoding="utf-8")
    (repo_root / "beta.py").write_text(
        "from ts_benchmark.model import ScenarioModel\n"
        "\n"
        "class RepoModel(ScenarioModel):\n"
        "    pass\n",
        encoding="utf-8",
    )

    candidates = model_catalog.find_repo_scenario_model_candidates(repo_root)

    assert candidates == [
        {
            "path": str(repo_root / "beta.py"),
            "symbol": "RepoModel",
        }
    ]
