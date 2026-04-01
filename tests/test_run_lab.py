from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui.pages import run_lab


def test_available_benchmark_rows_include_local_saved_catalog(monkeypatch, tmp_path: Path) -> None:
    official_path = tmp_path / "official.json"
    local_path = tmp_path / "local.json"

    monkeypatch.setattr(
        run_lab,
        "list_saved_benchmarks",
        lambda benchmark_dir=None: (
            [{"name": "Official Smoke", "path": official_path, "origin": "official", "read_only": True}]
            if benchmark_dir is None
            else [{"name": "Local Draft", "path": local_path}]
        ),
    )
    monkeypatch.setattr(run_lab, "BENCHMARK_CATALOG_DIR", tmp_path / "catalog")

    rows = run_lab._available_benchmark_rows()

    assert [row["name"] for row in rows] == ["Official Smoke", "Local Draft"]
    assert rows[0]["origin"] == "official"
    assert rows[1]["origin"] == "saved"
    assert rows[1]["read_only"] is False


def test_load_selected_benchmark_uses_row_path(monkeypatch, tmp_path: Path) -> None:
    selected_path = tmp_path / "saved.json"
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_lab, "get_current_config_path", lambda: None)
    monkeypatch.setattr(run_lab, "get_current_config", lambda: {})
    monkeypatch.setattr(run_lab, "_reset_run_lab_state", lambda preserve=None: captured.__setitem__("preserve", preserve))
    monkeypatch.setattr(run_lab, "load_config_dict", lambda path: {"benchmark": {"name": "Saved"}, "_path": str(path)})
    monkeypatch.setattr(run_lab, "set_current_config", lambda value: captured.__setitem__("config", value))
    monkeypatch.setattr(run_lab, "set_current_config_path", lambda value: captured.__setitem__("path", value))
    monkeypatch.setattr(run_lab, "set_current_run", lambda value: captured.__setitem__("run", value))
    monkeypatch.setattr(run_lab, "set_current_run_artifacts", lambda value: captured.__setitem__("artifacts", value))
    monkeypatch.setattr(run_lab, "set_validation_result", lambda value: captured.__setitem__("validation", value))
    monkeypatch.setattr(run_lab.st, "rerun", lambda: captured.__setitem__("rerun", True))

    current = run_lab._load_selected_benchmark({"name": "Saved", "path": selected_path})

    assert current["benchmark"]["name"] == "Saved"
    assert captured["path"] == selected_path.resolve()
    assert captured["rerun"] is True


def test_attach_run_results_updates_local_saved_benchmark_metadata(monkeypatch, tmp_path: Path) -> None:
    benchmark_path = tmp_path / "catalog" / "saved.json"
    previous_results_dir = tmp_path / "previous"
    source_run_dir = tmp_path / "run"
    merged_results_dir = tmp_path / "merged"
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_lab, "BENCHMARK_CATALOG_DIR", benchmark_path.parent)
    monkeypatch.setattr(run_lab, "materialize_benchmark_results", lambda **kwargs: merged_results_dir)
    monkeypatch.setattr(run_lab, "load_run_artifacts", lambda path: {"summary": {"models": ["sbts"]}, "run_dir": str(path)})
    monkeypatch.setattr(
        run_lab,
        "update_saved_benchmark_results",
        lambda name_or_path, run_dir, *, summary=None, benchmark_dir=None: captured.update(
            {
                "name_or_path": name_or_path,
                "run_dir": run_dir,
                "summary": summary,
                "benchmark_dir": benchmark_dir,
            }
        ),
    )

    run_info, artifacts = run_lab._attach_run_results(
        {
            "benchmark_path": str(benchmark_path),
            "output_dir": str(source_run_dir),
        },
        benchmark_config={"benchmark": {"name": "Saved"}},
        previous_results_dir=previous_results_dir,
    )

    assert artifacts["summary"] == {"models": ["sbts"]}
    assert run_info["latest_results_dir"] == str(merged_results_dir)
    assert captured["name_or_path"] == benchmark_path
    assert captured["run_dir"] == merged_results_dir
    assert captured["summary"] == {"models": ["sbts"]}
    assert captured["benchmark_dir"] == benchmark_path.parent
