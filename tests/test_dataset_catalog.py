from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui.services import datasets


def test_save_and_load_saved_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    payload = {
        "name": "demo_returns",
        "description": "Saved dataset smoke test",
        "provider": {
            "kind": "csv",
            "config": {
                "path": "/tmp/demo.csv",
                "dropna": "any",
            },
        },
        "schema": {
            "layout": "wide",
            "time_column": "date",
            "target_columns": ["SPX", "SX5E"],
            "frequency": "B",
        },
        "semantics": {"target_kind": "returns"},
        "metadata": {"team": "ui-test"},
    }

    path = datasets.save_dataset_definition(payload, dataset_dir=dataset_dir)
    loaded = datasets.load_saved_dataset("demo_returns", dataset_dir=dataset_dir)
    listed = datasets.list_saved_datasets(dataset_dir=dataset_dir)

    assert path.exists()
    assert loaded["name"] == "demo_returns"
    assert loaded["provider"]["kind"] == "csv"
    assert loaded["provider"]["config"]["path"] == "/tmp/demo.csv"
    assert listed[0]["name"] == "demo_returns"
    assert listed[0]["description"] == "Saved dataset smoke test"


def test_delete_saved_dataset_is_blocked_when_used_by_benchmark(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    datasets.save_dataset_definition(
        {
            "name": "shared_dataset",
            "description": "Used by a benchmark",
            "provider": {"kind": "synthetic", "config": {"generator": "regime_switching_factor_sv", "params": {}}},
            "schema": {"layout": "tensor", "frequency": "B"},
            "semantics": {},
            "metadata": {},
        },
        dataset_dir=dataset_dir,
    )
    (config_dir / "benchmark_a.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "benchmark": {
                    "name": "benchmark_a",
                    "dataset": {"name": "shared_dataset"},
                    "protocol": {"train_size": 10, "test_size": 2, "context_length": 2, "horizon": 1},
                    "models": [
                        {
                            "name": "historical",
                            "reference": {"kind": "builtin", "value": "historical_bootstrap"},
                            "pipeline": {"name": "raw", "steps": []},
                        }
                    ],
                },
                "run": {"execution": {"scheduler": "auto"}, "output": {}},
            }
        ),
        encoding="utf-8",
    )

    deleted, usages = datasets.delete_saved_dataset(
        "shared_dataset",
        dataset_dir=dataset_dir,
        config_dir=config_dir,
    )

    assert deleted is False
    assert [path.name for path in usages] == ["benchmark_a.json"]
    assert (dataset_dir / "shared_dataset.json").exists()


def test_delete_saved_dataset_succeeds_when_unused(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    path = datasets.save_dataset_definition(
        {
            "name": "unused_dataset",
            "description": "",
            "provider": {"kind": "synthetic", "config": {"generator": "regime_switching_factor_sv", "params": {}}},
            "schema": {"layout": "tensor", "frequency": "B"},
            "semantics": {},
            "metadata": {},
        },
        dataset_dir=dataset_dir,
    )

    deleted, usages = datasets.delete_saved_dataset(
        "unused_dataset",
        dataset_dir=dataset_dir,
        config_dir=config_dir,
    )

    assert deleted is True
    assert usages == []
    assert not path.exists()


def test_switch_dataset_source_updates_source_specific_shape() -> None:
    synthetic = datasets.normalize_dataset_dict(
        {
            "name": "switch_me",
            "provider": {"kind": "synthetic", "config": {"generator": "regime_switching_factor_sv", "params": {"n_assets": 4}}},
            "schema": {"layout": "tensor", "frequency": "B"},
            "semantics": {},
            "metadata": {},
        }
    )

    file_backed = datasets.switch_dataset_source(synthetic, "csv")
    restored = datasets.switch_dataset_source(file_backed, "synthetic")

    assert file_backed["provider"]["kind"] == "csv"
    assert file_backed["schema"]["layout"] == "wide"
    assert file_backed["provider"]["config"]["path"] == ""
    assert restored["provider"]["kind"] == "synthetic"
    assert restored["schema"]["layout"] == "tensor"


def test_normalize_dataset_dict_preserves_long_layout_and_value_column() -> None:
    payload = datasets.normalize_dataset_dict(
        {
            "name": "long_prices",
            "provider": {
                "kind": "csv",
                "config": {
                    "path": "/tmp/long.csv",
                    "value_column": "price",
                },
            },
            "schema": {
                "layout": "long",
                "time_column": "date",
                "series_id_columns": ["ticker"],
                "frequency": "B",
            },
        }
    )

    assert payload["schema"]["layout"] == "long"
    assert payload["schema"]["series_id_columns"] == ["ticker"]
    assert payload["provider"]["config"]["value_column"] == "price"


def test_store_uploaded_dataset_file_preserves_content_and_extension(tmp_path: Path) -> None:
    payload = b"date,asset_a\n2024-01-01,0.1\n"

    path = datasets.store_uploaded_dataset_file(
        filename="Demo Returns.CSV",
        content=payload,
        upload_dir=tmp_path,
    )

    assert path.name == "demo_returns.csv"
    assert path.read_bytes() == payload


def test_inspect_tabular_source_reports_columns_numeric_columns_and_preview(tmp_path: Path) -> None:
    path = tmp_path / "returns.csv"
    path.write_text(
        "date,asset_a,asset_b,label\n"
        "2024-01-01,0.1,0.2,train\n"
        "2024-01-02,0.3,0.4,test\n",
        encoding="utf-8",
    )

    inspection = datasets.inspect_tabular_source(
        path=path,
        source="csv",
        max_rows=1,
    )

    assert inspection["path"] == str(path.resolve())
    assert inspection["columns"] == ["date", "asset_a", "asset_b", "label"]
    assert inspection["numeric_columns"] == ["asset_a", "asset_b"]
    assert inspection["preview"].shape == (1, 4)
