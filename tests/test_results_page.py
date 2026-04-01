from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui.pages import results


def test_available_benchmark_rows_include_local_saved_catalog(monkeypatch, tmp_path: Path) -> None:
    official_path = tmp_path / "official.json"
    local_path = tmp_path / "local.json"

    monkeypatch.setattr(
        results,
        "list_saved_benchmarks",
        lambda benchmark_dir=None: (
            [{"name": "Official Smoke", "path": official_path, "origin": "official", "read_only": True}]
            if benchmark_dir is None
            else [{"name": "Local Draft", "path": local_path}]
        ),
    )
    monkeypatch.setattr(results, "BENCHMARK_CATALOG_DIR", tmp_path / "catalog")

    rows = results._available_benchmark_rows()

    assert [row["name"] for row in rows] == ["Official Smoke", "Local Draft"]
    assert rows[0]["origin"] == "official"
    assert rows[1]["origin"] == "saved"
    assert rows[1]["read_only"] is False


def test_benchmark_option_map_labels_saved_and_official_rows() -> None:
    labels, options = results._benchmark_option_map(
        [
            {"name": "Smoke", "path": Path("/tmp/official.json"), "origin": "official"},
            {"name": "Smoke", "path": Path("/tmp/local.json"), "origin": "saved"},
        ]
    )

    assert labels[0] == "Smoke (official)"
    assert labels[1] == "Smoke (saved)"
    assert options["Smoke (official)"]["origin"] == "official"
    assert options["Smoke (saved)"]["origin"] == "saved"
