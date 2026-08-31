"""
Multi-Year Stochastic Trajectory Monte Carlo Forecasting Engine.
Simulates long-term portfolio wealth accumulation across 1-5 years using
Correlated Multi-Asset Geometric Brownian Motion (Cholesky decomposition) and
Historical Block Bootstrapping with 5%, 25%, 50%, 75%, and 95% probability cones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


@dataclass
class TrajectorySimulationResult:
    """Dataclass holding multi-year stochastic trajectory simulation outputs."""

    days: np.ndarray            # Shape: (total_days + 1,) Day indices [0, 1, ..., M]
    years: np.ndarray           # Shape: (total_days + 1,) Year fractions [0, 1/252, ..., T]
    percentile_5: np.ndarray    # Shape: (total_days + 1,) 5th percentile wealth path (95% VaR lower bound)
    percentile_25: np.ndarray   # Shape: (total_days + 1,) 25th percentile wealth path (1st quartile)
    percentile_50: np.ndarray   # Shape: (total_days + 1,) 50th percentile wealth path (Median)
    percentile_75: np.ndarray   # Shape: (total_days + 1,) 75th percentile wealth path (3rd quartile)
    percentile_95: np.ndarray   # Shape: (total_days + 1,) 95th percentile wealth path (Optimistic)
    mean_trajectory: np.ndarray # Shape: (total_days + 1,) Expected (mean) wealth path
    sample_paths: np.ndarray    # Shape: (min(num_simulations, 50), total_days + 1) Background paths
    initial_wealth: float       # Initial starting wealth (e.g. $10,000 USD)
    final_wealth_stats: Dict[str, float]  # Summary statistics dictionary

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "days": self.days.tolist(),
            "years": self.years.tolist(),
            "percentile_5": self.percentile_5.tolist(),
            "percentile_25": self.percentile_25.tolist(),
            "percentile_50": self.percentile_50.tolist(),
            "percentile_75": self.percentile_75.tolist(),
            "percentile_95": self.percentile_95.tolist(),
            "mean_trajectory": self.mean_trajectory.tolist(),
            "sample_paths": self.sample_paths.tolist(),
            "initial_wealth": float(self.initial_wealth),
            "final_wealth_stats": dict(self.final_wealth_stats),
        }


def run_trajectory_monte_carlo(
    expected_returns: Union[np.ndarray, Sequence[float], Any],
    cov_matrix: Union[np.ndarray, Any],
    weights: Union[np.ndarray, Sequence[float], Any],
    initial_capital: float = 10000.0,
    years: Union[int, float] = 3,
    num_simulations: int = 2000,
    model: str = "gbm",
    historical_returns: Optional[Union[np.ndarray, Any]] = None,
    block_size: int = 10,
    seed: Optional[int] = None,
) -> TrajectorySimulationResult:
    """
    Simulate future stochastic portfolio wealth trajectories and construct probability cones.

    Models:
    1. 'gbm': Correlated Multi-Asset Geometric Brownian Motion via Cholesky decomposition.
       dW_t = L * dZ_t where Sigma = L * L^T.
    2. 'bootstrap': Historical Block Bootstrapping preserving non-normal distributions
       and cross-asset autocorrelation.

    Parameters
    ----------
    expected_returns : np.ndarray or pd.Series
        Asset annualized expected returns vector (k,).
    cov_matrix : np.ndarray or pd.DataFrame
        Asset annualized covariance matrix (k x k).
    weights : np.ndarray or pd.Series
        Portfolio asset weights (k,).
    initial_capital : float, default 10000.0
        Starting portfolio wealth in USD.
    years : int or float, default 3
        Simulation time horizon in years.
    num_simulations : int, default 2000
        Number of stochastic simulation paths.
    model : str, default 'gbm'
        Stochastic model: 'gbm' or 'bootstrap'.
    historical_returns : np.ndarray or pd.DataFrame, optional
        Historical daily returns (T_hist x k), required if model='bootstrap'.
    block_size : int, default 10
        Block length in trading days for block bootstrapping.
    seed : int, optional
        Random seed for deterministic simulation.

    Returns
    -------
    TrajectorySimulationResult
        Simulated trajectory quantile cones, mean path, sample paths, and terminal wealth statistics.
    """
    mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    w = np.asarray(getattr(weights, "values", weights), dtype=np.float64).flatten()

    k = len(w)
    total_days = int(round(252 * float(years)))
    if total_days < 1:
        total_days = 1

    dt = 1.0 / 252.0
    days = np.arange(total_days + 1, dtype=np.int64)
    years_grid = days / 252.0

    rng = np.random.default_rng(seed)

    if model.lower() in ("bootstrap", "historical_bootstrapping", "block_bootstrap"):
        if historical_returns is None:
            raise ValueError("historical_returns must be provided when model='bootstrap'")

        H = np.asarray(getattr(historical_returns, "values", historical_returns), dtype=np.float64)
        if H.ndim == 1:
            H = H.reshape((-1, 1))

        t_hist = H.shape[0]
        if t_hist < 5:
            raise ValueError(f"historical_returns has too few observations ({t_hist}) for bootstrapping")

        # Historical portfolio daily returns
        h_p = H @ w  # Shape: (t_hist,)

        b = max(2, min(block_size, t_hist // 2))
        num_blocks = int(np.ceil(total_days / b))

        start_indices = rng.integers(0, t_hist - b + 1, size=(num_simulations, num_blocks))
        offsets = np.arange(b)
        indices = (start_indices[:, :, None] + offsets[None, None, :]).reshape(num_simulations, -1)[:, :total_days]

        sim_port_returns = h_p[indices]  # Shape: (num_simulations, total_days)
        wealth_steps = float(initial_capital) * np.cumprod(1.0 + sim_port_returns, axis=1)
        wealth_paths = np.hstack([
            np.full((num_simulations, 1), float(initial_capital), dtype=np.float64),
            wealth_steps,
        ])

    else:
        # Default: Correlated Multi-Asset Geometric Brownian Motion (GBM)
        cov_sym = 0.5 * (cov + cov.T)

        # Robust Cholesky decomposition with positive-definite eigenvalue floor
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_psd = (eigvecs * eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_psd)  # Shape: (k, k)

        # Asset drift vector: (mu - 0.5 * diag(Sigma)) * dt
        asset_drift = (mu - 0.5 * np.diag(cov_psd)) * dt  # Shape: (k,)

        # Generate standard normal innovations Z: Shape (num_simulations, total_days, k)
        Z = rng.standard_normal(size=(num_simulations, total_days, k))
        # Correlated Wiener innovations: Z @ L^T
        correlated_innovations = Z @ L.T  # Shape: (num_simulations, total_days, k)

        # Asset daily percentage returns
        asset_returns = np.exp(asset_drift + np.sqrt(dt) * correlated_innovations) - 1.0

        # Portfolio daily percentage returns: asset_returns @ w -> Shape (num_simulations, total_days)
        port_returns = asset_returns @ w

        # Wealth accumulation trajectory: V_t = V_0 * prod(1 + r_port)
        wealth_steps = float(initial_capital) * np.cumprod(1.0 + port_returns, axis=1)
        wealth_paths = np.hstack([
            np.full((num_simulations, 1), float(initial_capital), dtype=np.float64),
            wealth_steps,
        ])

    # Extract probability cones across all simulation paths at each time step
    percentile_5 = np.percentile(wealth_paths, 5, axis=0)
    percentile_25 = np.percentile(wealth_paths, 25, axis=0)
    percentile_50 = np.percentile(wealth_paths, 50, axis=0)
    percentile_75 = np.percentile(wealth_paths, 75, axis=0)
    percentile_95 = np.percentile(wealth_paths, 95, axis=0)
    mean_trajectory = np.mean(wealth_paths, axis=0)

    # Background sample paths (up to 50 paths)
    n_sample = min(num_simulations, 50)
    sample_paths = wealth_paths[:n_sample, :].copy()

    # Terminal Wealth Summary Statistics
    final_wealths = wealth_paths[:, -1]
    exp_final = float(np.mean(final_wealths))
    med_final = float(np.median(final_wealths))
    min_final = float(np.min(final_wealths))
    max_final = float(np.max(final_wealths))

    prob_loss = float(np.mean(final_wealths < float(initial_capital)))
    prob_double = float(np.mean(final_wealths >= 2.0 * float(initial_capital)))

    horizon_years = max(float(years), 1e-4)
    exp_cagr = float((exp_final / float(initial_capital)) ** (1.0 / horizon_years) - 1.0) if exp_final > 0 else -1.0
    med_cagr = float((med_final / float(initial_capital)) ** (1.0 / horizon_years) - 1.0) if med_final > 0 else -1.0

    p5_terminal = float(np.percentile(final_wealths, 5))
    var_95 = float(max(0.0, float(initial_capital) - p5_terminal))

    tail_losses = final_wealths[final_wealths <= p5_terminal]
    cvar_95 = float(max(0.0, float(initial_capital) - np.mean(tail_losses))) if len(tail_losses) > 0 else var_95

    # Simulated Max Drawdown
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (running_max - wealth_paths) / np.maximum(running_max, 1e-12)
    max_sim_dd = float(np.max(drawdowns))
    mean_sim_dd = float(np.mean(np.max(drawdowns, axis=1)))

    final_stats: Dict[str, float] = {
        "initial_capital": float(initial_capital),
        "years": float(years),
        "expected_final_wealth": exp_final,
        "median_final_wealth": med_final,
        "min_final_wealth": min_final,
        "max_final_wealth": max_final,
        "probability_of_loss": prob_loss,
        "probability_of_doubling": prob_double,
        "expected_cagr": exp_cagr,
        "median_cagr": med_cagr,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_simulated_drawdown": max_sim_dd,
        "mean_simulated_drawdown": mean_sim_dd,
    }

    return TrajectorySimulationResult(
        days=days,
        years=years_grid,
        percentile_5=percentile_5,
        percentile_25=percentile_25,
        percentile_50=percentile_50,
        percentile_75=percentile_75,
        percentile_95=percentile_95,
        mean_trajectory=mean_trajectory,
        sample_paths=sample_paths,
        initial_wealth=float(initial_capital),
        final_wealth_stats=final_stats,
    )
