"""Window extraction helpers for conditional scenario models."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is optional for the core package
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class when torch is unavailable."""

        pass


def rolling_context_future_pairs(
    returns: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract rolling context/future pairs.

    Parameters
    ----------
    returns:
        Array of shape `[time, n_assets]`.
    context_length:
        Number of historical steps in the conditioning context.
    horizon:
        Forecast horizon.
    stride:
        Step between consecutive windows.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be shaped [time, n_assets].")
    if context_length <= 0 or horizon <= 0:
        raise ValueError("context_length and horizon must be positive.")
    if len(x) < context_length + horizon:
        raise ValueError("returns are too short for the requested window sizes.")

    contexts = []
    futures = []
    for end in range(context_length, len(x) - horizon + 1, stride):
        contexts.append(x[end - context_length : end])
        futures.append(x[end : end + horizon])

    return np.stack(contexts, axis=0), np.stack(futures, axis=0)


class ForecastWindowDataset(Dataset):
    """PyTorch dataset over precomputed context/future windows."""

    def __init__(self, contexts: np.ndarray, futures: np.ndarray):
        if torch is None:
            raise ImportError(
                "ForecastWindowDataset requires the optional 'torch' extra. "
                "Install ts-benchmark[torch] to use this helper."
            )
        if contexts.shape[0] != futures.shape[0]:
            raise ValueError("contexts and futures must have the same first dimension.")
        self.contexts = torch.as_tensor(contexts, dtype=torch.float32)
        self.futures = torch.as_tensor(futures, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.contexts.shape[0])

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self.contexts[idx], self.futures[idx]
