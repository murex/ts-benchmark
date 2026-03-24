"""Per-window metric computation for diagnostic analysis."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..dataset.runtime import DatasetInstance
from ..metrics.distributional import compute_distributional_metrics
from ..metrics.scoring import compute_sample_scoring_metrics


def build_per_window_metrics(
    *,
    dataset: DatasetInstance,
    generated_scenarios: Mapping[str, np.ndarray],
    reference_scenarios: np.ndarray | None,
    include: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamps = dataset.evaluation_timestamps or [None] * int(dataset.contexts.shape[0])
    for model_name, samples in generated_scenarios.items():
        for context_index in range(int(samples.shape[0])):
            row: dict[str, object] = {
                "model": model_name,
                "context_index": context_index,
                "evaluation_timestamp": timestamps[context_index],
            }
            scoring = compute_sample_scoring_metrics(
                samples[context_index : context_index + 1],
                dataset.realized_futures[context_index : context_index + 1],
            )
            row.update(scoring)
            if reference_scenarios is not None:
                row.update(
                    compute_distributional_metrics(
                        samples[context_index : context_index + 1],
                        reference_scenarios[context_index : context_index + 1],
                    )
                )
            if include:
                row = {
                    key: value
                    for key, value in row.items()
                    if key in {"model", "context_index", "evaluation_timestamp"} or key in include
                }
            rows.append(row)
    return pd.DataFrame(rows)
