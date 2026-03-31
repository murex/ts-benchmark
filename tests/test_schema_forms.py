from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_benchmark.ui.schema_forms import (
    _count_sliding_windows,
    _default_region_lengths,
    _protocol_summary_lines,
)


def test_count_sliding_windows_matches_forecast_examples() -> None:
    assert _count_sliding_windows(120, 12, 1) == 109
    assert _count_sliding_windows(24, 2, 2) == 12
    assert _count_sliding_windows(24, 2, 1) == 23


def test_protocol_summary_lines_report_forecast_counts() -> None:
    lines = _protocol_summary_lines(
        {
            "kind": "forecast",
            "horizon": 2,
            "n_model_scenarios": 64,
            "n_reference_scenarios": 128,
            "forecast": {
                "train_size": 120,
                "test_size": 24,
                "context_length": 10,
                "train_stride": 1,
                "eval_stride": 2,
            },
        },
        dataset_supports_reference_scenarios=True,
    )

    assert lines == [
        "Fit on 109 forecast examples: benchmark-owned past ending in context 10 with target 2.",
        "Evaluate on 12 forecast origins in the held-out future region.",
        "Request 64 model scenarios per evaluation window.",
        "Compare against 128 reference scenarios per evaluation window.",
    ]


def test_protocol_summary_lines_hide_reference_for_non_synthetic() -> None:
    lines = _protocol_summary_lines(
        {
            "kind": "forecast",
            "horizon": 2,
            "n_model_scenarios": 64,
            "n_reference_scenarios": 128,
            "forecast": {
                "train_size": 120,
                "test_size": 24,
                "context_length": 10,
                "train_stride": 1,
                "eval_stride": 2,
            },
        },
        dataset_supports_reference_scenarios=False,
    )

    assert lines == [
        "Fit on 109 forecast examples: benchmark-owned past ending in context 10 with target 2.",
        "Evaluate on 12 forecast origins in the held-out future region.",
        "Request 64 model scenarios per evaluation window.",
    ]


def test_protocol_summary_lines_report_unconditional_path_dataset_counts() -> None:
    lines = _protocol_summary_lines(
        {
            "kind": "unconditional_path_dataset",
            "horizon": 2,
            "n_model_scenarios": 64,
            "n_reference_scenarios": 128,
            "unconditional_path_dataset": {
                "n_train_paths": 5,
                "n_realized_paths": 3,
            },
        },
        dataset_supports_reference_scenarios=True,
    )

    assert lines == [
        "Fit on 5 benchmark-provided unconditional training paths of length 2.",
        "Evaluate on 3 held-out unconditional realized paths of length 2.",
        "Request 64 model scenarios per realized path.",
        "Compare against 128 reference scenarios per realized path.",
    ]


def test_default_region_lengths_ignore_path_dataset_fixed_lengths() -> None:
    assert _default_region_lengths(
        {
            "kind": "unconditional_path_dataset",
            "horizon": 2,
        }
    ) == (120, 40)

    assert _default_region_lengths(
        {
            "kind": "forecast",
            "forecast": {
                "train_size": 96,
                "test_size": 24,
            },
        }
    ) == (96, 24)
