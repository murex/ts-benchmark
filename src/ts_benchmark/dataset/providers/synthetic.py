"""Synthetic benchmark generator for multivariate time-series returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...benchmark.protocol import Protocol
from ..runtime import DatasetInstance


@dataclass
class SyntheticState:
    """Latent state used to generate the next return observation."""

    regime: int
    market_log_var: float
    idio_log_var: np.ndarray

    def copy(self) -> "SyntheticState":
        return SyntheticState(
            regime=int(self.regime),
            market_log_var=float(self.market_log_var),
            idio_log_var=np.array(self.idio_log_var, dtype=float, copy=True),
        )


@dataclass
class SyntheticSimulation:
    """Full synthetic simulation output."""

    returns: np.ndarray
    states: List[SyntheticState]
    asset_names: List[str]
    metadata: Dict[str, Any]


SyntheticDatasetInstance = DatasetInstance


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


class RegimeSwitchingFactorSVGenerator:
    """Regime-switching factor stochastic-volatility generator.

    The process is designed to reproduce:
    - heavy tails,
    - market-wide dependence,
    - idiosyncratic heteroskedasticity,
    - volatility clustering,
    - regime changes between calm and stress.

    Returns are generated as:

        r_{t,i} = mu_{z_t} + beta_i * sigma^m_t * e^m_t
                          + s_i * sigma^{id}_{t,i} * e^{id}_{t,i}

    where the latent regime z_t follows a Markov chain and
    log-volatilities mean revert with Gaussian innovations.
    """

    def __init__(
        self,
        n_assets: int = 6,
        factor_loadings: Optional[np.ndarray] = None,
        idio_scales: Optional[np.ndarray] = None,
        regime_drifts: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
        market_log_var_means: Optional[np.ndarray] = None,
        idio_log_var_means: Optional[np.ndarray] = None,
        market_phi: float = 0.985,
        idio_phi: float = 0.965,
        market_vol_of_vol: float = 0.12,
        idio_vol_of_vol: float = 0.10,
        market_leverage: float = 0.06,
        idio_leverage: float = 0.03,
        student_df: float = 7.0,
        seed: Optional[int] = None,
    ):
        if n_assets <= 0:
            raise ValueError("n_assets must be positive.")
        self.n_assets = int(n_assets)
        self.n_regimes = 2

        self.factor_loadings = (
            np.asarray(factor_loadings, dtype=float)
            if factor_loadings is not None
            else np.linspace(0.7, 1.2, self.n_assets)
        )
        self.idio_scales = (
            np.asarray(idio_scales, dtype=float)
            if idio_scales is not None
            else np.linspace(0.55, 0.95, self.n_assets)
        )
        self.regime_drifts = (
            np.asarray(regime_drifts, dtype=float)
            if regime_drifts is not None
            else np.array([0.0004, -0.0008], dtype=float)
        )
        self.transition_matrix = (
            np.asarray(transition_matrix, dtype=float)
            if transition_matrix is not None
            else np.array([[0.985, 0.015], [0.08, 0.92]], dtype=float)
        )
        self.market_log_var_means = (
            np.asarray(market_log_var_means, dtype=float)
            if market_log_var_means is not None
            else np.array([-9.1, -7.2], dtype=float)
        )

        if idio_log_var_means is None:
            base = np.linspace(-9.7, -8.9, self.n_assets)
            idio_log_var_means = np.vstack([base, base + 0.85])
        self.idio_log_var_means = np.asarray(idio_log_var_means, dtype=float)

        self.market_phi = float(market_phi)
        self.idio_phi = float(idio_phi)
        self.market_vol_of_vol = float(market_vol_of_vol)
        self.idio_vol_of_vol = float(idio_vol_of_vol)
        self.market_leverage = float(market_leverage)
        self.idio_leverage = float(idio_leverage)
        self.student_df = float(student_df)
        self.seed = seed

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.factor_loadings.shape != (self.n_assets,):
            raise ValueError("factor_loadings must have shape [n_assets].")
        if self.idio_scales.shape != (self.n_assets,):
            raise ValueError("idio_scales must have shape [n_assets].")
        if self.regime_drifts.shape != (self.n_regimes,):
            raise ValueError("regime_drifts must have shape [2].")
        if self.transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("transition_matrix must have shape [2, 2].")
        if self.market_log_var_means.shape != (self.n_regimes,):
            raise ValueError("market_log_var_means must have shape [2].")
        if self.idio_log_var_means.shape != (self.n_regimes, self.n_assets):
            raise ValueError("idio_log_var_means must have shape [2, n_assets].")
        if np.any(self.transition_matrix < 0.0):
            raise ValueError("transition probabilities must be non-negative.")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("rows of transition_matrix must sum to 1.")
        if self.student_df <= 2.0:
            raise ValueError("student_df must be > 2 to have finite variance.")

    def _rng(self, seed: Optional[int] = None) -> np.random.Generator:
        base_seed = self.seed if seed is None else seed
        return np.random.default_rng(base_seed)

    def _standardized_student_t(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        x = rng.standard_t(self.student_df, size=size)
        scale = np.sqrt(self.student_df / (self.student_df - 2.0))
        return x / scale

    def initial_state(self) -> SyntheticState:
        return SyntheticState(
            regime=0,
            market_log_var=float(self.market_log_var_means[0]),
            idio_log_var=np.array(self.idio_log_var_means[0], dtype=float, copy=True),
        )

    def _transition_regimes(
        self,
        regimes: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        probs = self.transition_matrix[regimes]
        cum = np.cumsum(probs, axis=1)
        u = rng.random(len(regimes))[:, None]
        next_regimes = (u > cum).sum(axis=1)
        return next_regimes.astype(int)

    def _clip_log_var(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -13.5, -4.5)

    def _step_single(
        self,
        state: SyntheticState,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, SyntheticState]:
        market_sigma = np.exp(0.5 * state.market_log_var)
        idio_sigma = np.exp(0.5 * state.idio_log_var)

        market_shock = float(self._standardized_student_t(rng, 1)[0])
        idio_shocks = self._standardized_student_t(rng, self.n_assets)

        returns = (
            self.regime_drifts[state.regime]
            + self.factor_loadings * market_sigma * market_shock
            + self.idio_scales * idio_sigma * idio_shocks
        )

        next_regime = int(self._transition_regimes(np.array([state.regime]), rng)[0])
        next_market_mean = self.market_log_var_means[next_regime]
        next_idio_mean = self.idio_log_var_means[next_regime]

        next_market_log_var = (
            next_market_mean
            + self.market_phi * (state.market_log_var - next_market_mean)
            + self.market_vol_of_vol * rng.normal()
            - self.market_leverage * market_shock
        )
        next_idio_log_var = (
            next_idio_mean
            + self.idio_phi * (state.idio_log_var - next_idio_mean)
            + self.idio_vol_of_vol * rng.normal(size=self.n_assets)
            - self.idio_leverage * idio_shocks
        )

        next_state = SyntheticState(
            regime=next_regime,
            market_log_var=float(self._clip_log_var(np.array(next_market_log_var))[()]),
            idio_log_var=self._clip_log_var(np.asarray(next_idio_log_var, dtype=float)),
        )
        return np.asarray(returns, dtype=float), next_state

    def simulate(
        self,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> SyntheticSimulation:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        rng = self._rng(seed)
        state = self.initial_state()
        returns = np.zeros((n_steps, self.n_assets), dtype=float)
        states: List[SyntheticState] = []

        for t in range(n_steps):
            states.append(state.copy())
            returns[t], state = self._step_single(state, rng)

        asset_names = [f"Asset_{i + 1:02d}" for i in range(self.n_assets)]
        metadata = {
            "n_assets": self.n_assets,
            "transition_matrix": self.transition_matrix.copy(),
            "student_df": self.student_df,
        }
        return SyntheticSimulation(
            returns=returns,
            states=states,
            asset_names=asset_names,
            metadata=metadata,
        )

    def sample_paths_from_state(
        self,
        state: SyntheticState,
        horizon: int,
        n_scenarios: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample future paths conditional on a known latent state."""
        if horizon <= 0 or n_scenarios <= 0:
            raise ValueError("horizon and n_scenarios must be positive.")

        rng = np.random.default_rng() if rng is None else rng

        regimes = np.full(n_scenarios, int(state.regime), dtype=int)
        market_log_var = np.full(n_scenarios, float(state.market_log_var), dtype=float)
        idio_log_var = np.repeat(
            np.asarray(state.idio_log_var, dtype=float)[None, :],
            repeats=n_scenarios,
            axis=0,
        )
        paths = np.zeros((n_scenarios, horizon, self.n_assets), dtype=float)

        for t in range(horizon):
            market_sigma = np.exp(0.5 * market_log_var)[:, None]
            idio_sigma = np.exp(0.5 * idio_log_var)

            market_shocks = self._standardized_student_t(rng, n_scenarios)[:, None]
            idio_shocks = self._standardized_student_t(rng, (n_scenarios, self.n_assets))

            paths[:, t, :] = (
                self.regime_drifts[regimes][:, None]
                + self.factor_loadings[None, :] * market_sigma * market_shocks
                + self.idio_scales[None, :] * idio_sigma * idio_shocks
            )

            next_regimes = self._transition_regimes(regimes, rng)
            next_market_mean = self.market_log_var_means[next_regimes]
            next_idio_mean = self.idio_log_var_means[next_regimes]

            market_log_var = self._clip_log_var(
                next_market_mean
                + self.market_phi * (market_log_var - next_market_mean)
                + self.market_vol_of_vol * rng.normal(size=n_scenarios)
                - self.market_leverage * market_shocks[:, 0]
            )
            idio_log_var = self._clip_log_var(
                next_idio_mean
                + self.idio_phi * (idio_log_var - next_idio_mean)
                + self.idio_vol_of_vol * rng.normal(size=(n_scenarios, self.n_assets))
                - self.idio_leverage * idio_shocks
            )
            regimes = next_regimes

        return paths

    def make_benchmark_dataset(
        self,
        protocol: Protocol,
        seed: Optional[int] = None,
    ) -> SyntheticDatasetInstance:
        """Create a full benchmark dataset with evaluation contexts.

        Evaluation windows begin at the start of the test segment. The context
        window may therefore include the tail of the training segment.
        """
        protocol.validate()
        train_size = int(protocol.train_size)
        test_size = int(protocol.test_size)
        generation_mode = str(protocol.generation_mode)
        context_length = int(protocol.context_length)
        horizon = int(protocol.horizon)
        eval_stride = int(protocol.eval_stride)
        unconditional_train_data_mode = protocol.unconditional_train_data_mode
        unconditional_n_train_paths = protocol.unconditional_n_train_paths
        unconditional_n_eval_paths = protocol.unconditional_n_eval_paths
        if generation_mode == "forecast" and train_size <= context_length:
            raise ValueError("train_size must be larger than context_length.")
        if (
            test_size < horizon
            and not (
                generation_mode == "unconditional"
                and unconditional_train_data_mode == "path_dataset"
            )
        ):
            raise ValueError("test_size must be at least horizon.")
        if eval_stride <= 0:
            raise ValueError("eval_stride must be positive.")

        total_steps = train_size + test_size
        train_paths: list[np.ndarray] | None = None
        eval_paths: list[np.ndarray] | None = None
        if generation_mode == "unconditional" and unconditional_train_data_mode == "path_dataset":
            if unconditional_n_train_paths is None:
                raise ValueError(
                    "unconditional_n_train_paths is required for synthetic path-dataset training."
                )
            if unconditional_n_eval_paths is None:
                raise ValueError(
                    "unconditional_n_eval_paths is required for synthetic path-dataset evaluation."
                )
            seed_rng = self._rng(seed)
            child_seeds = seed_rng.integers(
                0,
                np.iinfo(np.int64).max,
                size=int(unconditional_n_train_paths) + int(unconditional_n_eval_paths) + 1,
                dtype=np.int64,
            )
            train_paths = [
                self.simulate(train_size, seed=int(child_seed)).returns
                for child_seed in child_seeds[: int(unconditional_n_train_paths)]
            ]
            eval_paths = [
                self.simulate(horizon, seed=int(child_seed)).returns
                for child_seed in child_seeds[
                    int(unconditional_n_train_paths) : int(unconditional_n_train_paths) + int(unconditional_n_eval_paths)
                ]
            ]
            simulation = self.simulate(max(train_size, horizon), seed=int(child_seeds[-1]))
            returns = None
            train_returns = np.concatenate(train_paths, axis=0)
            test_returns = np.concatenate(eval_paths, axis=0)
            full_returns = np.concatenate([train_returns, test_returns], axis=0)
        else:
            simulation = self.simulate(total_steps, seed=seed)
            returns = simulation.returns
            train_returns = returns[:train_size]
            test_returns = returns[train_size:]
            full_returns = returns

        contexts = []
        realized_futures = []
        reference_states = []

        if generation_mode == "unconditional" and unconditional_train_data_mode == "path_dataset":
            if eval_paths is None:
                raise ValueError("Expected independent evaluation paths for unconditional path-dataset mode.")
            contexts = [np.zeros((0, self.n_assets), dtype=float) for _ in eval_paths]
            realized_futures = [np.asarray(path, dtype=float) for path in eval_paths]
        else:
            for start in range(train_size, total_steps - horizon + 1, eval_stride):
                if generation_mode == "forecast":
                    contexts.append(returns[start - context_length : start])
                    reference_states.append(simulation.states[start].copy())
                else:
                    contexts.append(np.zeros((0, self.n_assets), dtype=float))
                realized_futures.append(returns[start : start + horizon])

        if not contexts:
            raise ValueError("No evaluation windows were generated. Adjust test_size / horizon / eval_stride.")

        def reference_sampler(n_scenarios: int, sampler_seed: int | None = None) -> np.ndarray:
            rng = np.random.default_rng(sampler_seed)
            scenarios = []
            if generation_mode == "forecast":
                for state in reference_states:
                    scenarios.append(
                        self.sample_paths_from_state(
                            state=state,
                            horizon=horizon,
                            n_scenarios=n_scenarios,
                            rng=rng,
                        )
                    )
            else:
                child_seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(contexts), dtype=np.int64)
                for child_seed in child_seeds:
                    window_scenarios = []
                    for scenario_index in range(int(n_scenarios)):
                        path_seed = int((int(child_seed) + scenario_index) % np.iinfo(np.int64).max)
                        window_scenarios.append(self.simulate(horizon, seed=path_seed).returns)
                    scenarios.append(np.stack(window_scenarios, axis=0))
            return np.stack(scenarios, axis=0)

        return DatasetInstance(
            name="synthetic::regime_switching_factor_sv",
            source="synthetic",
            full_returns=full_returns,
            train_returns=train_returns,
            test_returns=test_returns,
            contexts=np.stack(contexts, axis=0),
            realized_futures=np.stack(realized_futures, axis=0),
            asset_names=simulation.asset_names,
            protocol=protocol,
            freq="B",
            train_paths=train_paths,
            metadata={
                **simulation.metadata,
                "generator": "regime_switching_factor_sv",
                "generator_class": self.__class__.__name__,
                "reference_state_count": len(reference_states),
                "reference_mode": generation_mode,
                "path_construction": unconditional_train_data_mode,
                "n_train_paths": unconditional_n_train_paths,
                "n_realized_paths": unconditional_n_eval_paths,
                "train_path_count": None if train_paths is None else len(train_paths),
                "train_path_length": None if train_paths is None else train_size,
                "eval_path_count": None if eval_paths is None else len(eval_paths),
                "eval_path_length": None if eval_paths is None else horizon,
                "is_synthetic": True,
            },
            evaluation_timestamps=None,
            reference_sampler=reference_sampler,
        )
