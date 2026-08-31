"""
Weight Space Monte Carlo Simulation Engine.
Performs vectorized Dirichlet uniform simplex sampling on Delta^(k-1) to map the
continuous risk-return-Sharpe opportunity set across thousands of random portfolios in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np


@dataclass
class WeightMonteCarloResult:
    """Dataclass holding weight space Monte Carlo simulation outputs."""

    weights: np.ndarray         # Shape: (num_portfolios, k) Portfolio weight vectors
    returns: np.ndarray         # Shape: (num_portfolios,) Annualized expected returns
    volatilities: np.ndarray    # Shape: (num_portfolios,) Annualized volatilities
    sharpe_ratios: np.ndarray   # Shape: (num_portfolios,) Sharpe ratios relative to rf
    max_sharpe_idx: int         # Index of sampled portfolio with highest Sharpe ratio
    min_vol_idx: int            # Index of sampled portfolio with minimum volatility

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "weights": self.weights.tolist(),
            "returns": self.returns.tolist(),
            "volatilities": self.volatilities.tolist(),
            "sharpe_ratios": self.sharpe_ratios.tolist(),
            "max_sharpe_idx": int(self.max_sharpe_idx),
            "min_vol_idx": int(self.min_vol_idx),
        }


def run_weight_space_monte_carlo(
    expected_returns: Union[np.ndarray, Sequence[float], Any],
    cov_matrix: Union[np.ndarray, Any],
    rf: float = 0.04,
    num_portfolios: int = 10000,
    seed: Optional[int] = None,
) -> WeightMonteCarloResult:
    """
    Run vectorized Dirichlet uniform simplex Monte Carlo simulation over portfolio weight space.

    Generates N random allocations sampled uniformly on the standard simplex:
        W ~ Dirichlet(alpha = (1, 1, ..., 1))

    and computes portfolio expected return, volatility, and Sharpe ratio for each allocation.

    Parameters
    ----------
    expected_returns : np.ndarray or pd.Series
        Asset annualized expected returns vector (k,).
    cov_matrix : np.ndarray or pd.DataFrame
        Asset annualized covariance matrix (k x k).
    rf : float, default 0.04
        Risk-free rate.
    num_portfolios : int, default 10000
        Number of portfolio weight vectors to sample.
    seed : int, optional
        Random number generator seed for deterministic reproducibility.

    Returns
    -------
    WeightMonteCarloResult
        Simulated weights, returns, volatilities, Sharpe ratios, and extreme point indices.
    """
    mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    k = len(mu)

    # Corner case: Single asset universe
    if k == 1:
        weights = np.ones((num_portfolios, 1), dtype=np.float64)
        rets = np.full(num_portfolios, float(mu[0]), dtype=np.float64)
        vol = float(np.sqrt(max(cov[0, 0], 0.0)))
        vols = np.full(num_portfolios, vol, dtype=np.float64)
        sharpes = (rets - rf) / max(vol, 1e-12)
        return WeightMonteCarloResult(
            weights=weights,
            returns=rets,
            volatilities=vols,
            sharpe_ratios=sharpes,
            max_sharpe_idx=0,
            min_vol_idx=0,
        )

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # 1. Uniform simplex Dirichlet sampling via standard exponential distribution:
    # If E_1, ..., E_k ~ Exp(1), then E / sum(E) ~ Dirichlet(1, ..., 1) uniformly on Delta^(k-1).
    exp_variates = rng.standard_exponential(size=(num_portfolios, k), dtype=np.float64)
    weights = exp_variates / np.sum(exp_variates, axis=1, keepdims=True)

    # 2. Vectorized Portfolio Returns: Shape (num_portfolios,)
    returns = weights @ mu

    # 3. Vectorized Portfolio Variances & Volatilities (Optimized quadratic form)
    cov_sym = 0.5 * (cov + cov.T)
    w_cov = weights @ cov_sym
    variances = np.einsum("ij,ij->i", w_cov, weights)
    volatilities = np.sqrt(np.maximum(variances, 1e-14))

    # 4. Vectorized Sharpe Ratios
    sharpe_ratios = (returns - rf) / np.maximum(volatilities, 1e-12)

    # 5. Locate extreme points
    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    min_vol_idx = int(np.argmin(volatilities))

    return WeightMonteCarloResult(
        weights=weights,
        returns=returns,
        volatilities=volatilities,
        sharpe_ratios=sharpe_ratios,
        max_sharpe_idx=max_sharpe_idx,
        min_vol_idx=min_vol_idx,
    )
