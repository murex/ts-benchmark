"""Reporting helpers for benchmark outputs."""

from __future__ import annotations

import pandas as pd


def metrics_table_to_markdown(df: pd.DataFrame, digits: int = 6) -> str:
    """Render benchmark results as a markdown table."""
    return df.round(digits).to_markdown()
