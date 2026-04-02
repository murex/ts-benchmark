"""Typed config objects for synthetic dataset providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RegimeSwitchingFactorSVConfig:
    """Typed parameter object for the regime-switching factor SV generator."""

    n_assets: int = 6
    factor_loadings: Sequence[float] | None = None
    idio_scales: Sequence[float] | None = None
    regime_drifts: Sequence[float] | None = None
    transition_matrix: Sequence[Sequence[float]] | None = None
    market_log_var_means: Sequence[float] | None = None
    idio_log_var_means: Sequence[Sequence[float]] | None = None
    market_phi: float = 0.985
    idio_phi: float = 0.965
    market_vol_of_vol: float = 0.12
    idio_vol_of_vol: float = 0.10
    market_leverage: float = 0.06
    idio_leverage: float = 0.03
    student_df: float = 7.0
    seed: int | None = None
