from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.dataset.providers.tabular import load_returns_frame


def test_load_returns_frame_supports_explicit_price_and_log_price_modes(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,asset_a,asset_b\n"
        "2024-01-01,100.0,200.0\n"
        "2024-01-02,110.0,210.0\n"
        "2024-01-03,121.0,220.5\n",
        encoding="utf-8",
    )

    price_returns, _, price_meta = load_returns_frame(
        path=path,
        source="csv",
        params={
            "date_column": "date",
            "asset_columns": ["asset_a", "asset_b"],
            "value_type": "price",
            "dropna": "any",
        },
    )
    log_price_returns, _, log_price_meta = load_returns_frame(
        path=path,
        source="csv",
        params={
            "date_column": "date",
            "asset_columns": ["asset_a", "asset_b"],
            "value_type": "log_price",
            "dropna": "any",
        },
    )

    assert np.allclose(price_returns["asset_a"].to_numpy(), [0.1, 0.1])
    assert np.allclose(price_returns["asset_b"].to_numpy(), [0.05, 0.05])
    assert np.allclose(log_price_returns["asset_a"].to_numpy(), [10.0, 11.0])
    assert np.allclose(log_price_returns["asset_b"].to_numpy(), [10.0, 10.5])
    assert price_meta["value_mode"] == "price"
    assert log_price_meta["value_mode"] == "log_price"


def test_load_returns_frame_supports_explicit_return_and_log_return_modes(tmp_path: Path) -> None:
    path = tmp_path / "returns.csv"
    path.write_text(
        "date,asset_a,asset_b\n"
        "2024-01-01,0.10,0.05\n"
        "2024-01-02,0.08,0.03\n",
        encoding="utf-8",
    )

    returns_frame, _, returns_meta = load_returns_frame(
        path=path,
        source="csv",
        params={
            "date_column": "date",
            "asset_columns": ["asset_a", "asset_b"],
            "value_type": "return",
            "dropna": "any",
        },
    )
    log_returns_frame, _, log_returns_meta = load_returns_frame(
        path=path,
        source="csv",
        params={
            "date_column": "date",
            "asset_columns": ["asset_a", "asset_b"],
            "value_type": "log_return",
            "dropna": "any",
        },
    )

    assert np.allclose(returns_frame.to_numpy(), [[0.10, 0.05], [0.08, 0.03]])
    assert np.allclose(log_returns_frame.to_numpy(), [[0.10, 0.05], [0.08, 0.03]])
    assert returns_meta["value_mode"] == "return"
    assert log_returns_meta["value_mode"] == "log_return"


def test_load_returns_frame_supports_long_layout_with_series_id_column(tmp_path: Path) -> None:
    path = tmp_path / "long_prices.csv"
    path.write_text(
        "date,ticker,price\n"
        "2024-01-01,AAPL,100.0\n"
        "2024-01-01,MSFT,200.0\n"
        "2024-01-02,AAPL,110.0\n"
        "2024-01-02,MSFT,210.0\n"
        "2024-01-03,AAPL,121.0\n"
        "2024-01-03,MSFT,220.5\n",
        encoding="utf-8",
    )

    returns_frame, timestamps, metadata = load_returns_frame(
        path=path,
        source="csv",
        params={
            "layout": "long",
            "date_column": "date",
            "series_id_columns": ["ticker"],
            "value_column": "price",
            "value_type": "price",
            "dropna": "any",
        },
    )

    assert list(returns_frame.columns) == ["AAPL", "MSFT"]
    assert np.allclose(returns_frame.to_numpy(), [[0.1, 0.05], [0.1, 0.05]])
    assert timestamps is not None
    assert list(timestamps.astype(str)) == ["2024-01-02", "2024-01-03"]
    assert metadata["layout"] == "long"
