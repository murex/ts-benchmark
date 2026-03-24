"""Randomness helpers."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch import can fail in minimal environments
    torch = None


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
