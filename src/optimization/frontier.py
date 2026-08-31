"""
Markowitz Efficient Frontier & Capital Allocation Line (CAL) Engine.
Computes the continuous upper Pareto efficient frontier curve using warm-started
quadratic optimization sweeps and generates the tangent Capital Allocation Line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize as sco

from src.optimization.optimizer import (
    OptimizationResult,
    normalize_and_clamp_weights,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe,
    optimize_target_return,
    parse_and_validate_bounds,
)


@dataclass
class EfficientFrontierResult:
    """Dataclass holding continuous Markowitz Efficient Frontier calculation outputs."""

    returns: np.ndarray         # Shape: (num_points,) Expected returns along frontier
    volatilities: np.ndarray    # Shape: (num_points,) Volatilities along frontier
    weights: np.ndarray         # Shape: (num_points, k) Optimal asset weights along frontier
    sharpe_ratios: np.ndarray   # Shape: (num_points,) Sharpe ratios along frontier
    gmv_portfolio: OptimizationResult
    max_sharpe_portfolio: OptimizationResult
    cal_line: Tuple[np.ndarray, np.ndarray]  # (cal_volatilities, cal_returns)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "returns": self.returns.tolist(),
            "volatilities": self.volatilities.tolist(),
            "weights": self.weights.tolist(),
            "sharpe_ratios": self.sharpe_ratios.tolist(),
            "gmv_portfolio": self.gmv_portfolio.to_dict(),
            "max_sharpe_portfolio": self.max_sharpe_portfolio.to_dict(),
            "cal_line": (self.cal_line[0].tolist(), self.cal_line[1].tolist()),
        }


def compute_capital_allocation_line(
    max_sharpe_portfolio: OptimizationResult,
    rf: float = 0.04,
    max_vol: Optional[float] = None,
    max_volatility: Optional[float] = None,
    num_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Capital Allocation Line (CAL) tangent from (0, Rf) through the Max Sharpe portfolio.

    Formula:
        E[R_c] = Rf + Sharpe_max * sigma_c

    Parameters
    ----------
    max_sharpe_portfolio : OptimizationResult
        Optimized Maximum Sharpe (Tangency) portfolio.
    rf : float, default 0.04
        Risk-free rate.
    max_vol : float, optional
        Maximum volatility on the line.
    max_volatility : float, optional
        Alias for max_vol.
    num_points : int, default 50
        Number of points on the line.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (cal_volatilities, cal_returns)
    """
    end_vol: float
    if max_vol is not None:
        end_vol = float(max_vol)
    elif max_volatility is not None:
        end_vol = float(max_volatility)
    else:
        end_vol = max(0.50, float(1.3 * max_sharpe_portfolio.volatility))

    cal_volatilities = np.linspace(0.0, end_vol, num_points, dtype=np.float64)
    slope = float(max_sharpe_portfolio.sharpe_ratio)
    cal_returns = rf + slope * cal_volatilities

    return (cal_volatilities, cal_returns)


def compute_efficient_frontier(
    expected_returns: Union[np.ndarray, Sequence[float], Any],
    cov_matrix: Union[np.ndarray, Any],
    rf: float = 0.04,
    num_points: int = 100,
    bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
    custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> EfficientFrontierResult:
    """
    Compute the continuous upper Pareto efficient frontier curve.

    Performs warm-start chained quadratic optimizations across target returns
    from GMV portfolio return (mu_min) to maximum feasible return (mu_max).

    Parameters
    ----------
    expected_returns : np.ndarray or pd.Series
        Asset expected returns (k,).
    cov_matrix : np.ndarray or pd.DataFrame
        Asset covariance matrix (k x k).
    rf : float, default 0.04
        Risk-free rate.
    num_points : int, default 100
        Number of discrete points on the frontier.
    bounds : tuple or list of tuples, default (0.0, 1.0)
        Asset bounds.
    custom_bounds : list of tuples, optional
        Per-asset custom bounds.

    Returns
    -------
    EfficientFrontierResult
        Calculated efficient frontier containing curve arrays, GMV, Max Sharpe, and CAL.
    """
    mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    n = len(mu)

    parsed_bounds = parse_and_validate_bounds(n, bounds=bounds, custom_bounds=custom_bounds)

    # 1. Solve GMV portfolio (lower end of upper efficient frontier)
    gmv_res = optimize_global_minimum_variance(
        cov, expected_returns=mu, rf=rf, bounds=parsed_bounds
    )

    # 2. Solve Max Sharpe portfolio
    ms_res = optimize_maximum_sharpe(
        mu, cov, rf=rf, bounds=parsed_bounds
    )

    # Handle N=1 corner case
    if n == 1:
        rets = np.full(num_points, gmv_res.expected_return, dtype=np.float64)
        vols = np.full(num_points, gmv_res.volatility, dtype=np.float64)
        weights = np.ones((num_points, 1), dtype=np.float64)
        sharpes = np.full(num_points, gmv_res.sharpe_ratio, dtype=np.float64)
        cal_line = compute_capital_allocation_line(ms_res, rf=rf, num_points=num_points)
        return EfficientFrontierResult(
            returns=rets,
            volatilities=vols,
            weights=weights,
            sharpe_ratios=sharpes,
            gmv_portfolio=gmv_res,
            max_sharpe_portfolio=ms_res,
            cal_line=cal_line,
        )

    # 3. Determine maximum feasible return under boundary constraints
    # Maximize w^T mu subject to sum(w) = 1, w_min <= w <= w_max (via linprog)
    mu_min = float(gmv_res.expected_return)
    mu_max = float(np.max(mu))

    try:
        lp_res = sco.linprog(
            c=-mu,
            A_eq=np.ones((1, n)),
            b_eq=[1.0],
            bounds=parsed_bounds,
            method="highs",
        )
        if lp_res.success:
            mu_max = float(-lp_res.fun)
    except Exception:
        pass

    # Ensure mu_max >= mu_min
    if mu_max < mu_min:
        mu_max = mu_min

    # If mu_max == mu_min, all assets have identical returns
    if abs(mu_max - mu_min) < 1e-8:
        target_returns = np.full(num_points, mu_min, dtype=np.float64)
    else:
        target_returns = np.linspace(mu_min, mu_max, num_points, dtype=np.float64)

    # 4. Warm-start chained optimization along the target returns grid
    frontier_returns = np.zeros(num_points, dtype=np.float64)
    frontier_vols = np.zeros(num_points, dtype=np.float64)
    frontier_weights = np.zeros((num_points, n), dtype=np.float64)
    frontier_sharpes = np.zeros(num_points, dtype=np.float64)

    w_curr = gmv_res.weights.copy()

    for i in range(num_points):
        target_r = target_returns[i]

        if i == 0:
            frontier_weights[i] = gmv_res.weights
            frontier_returns[i] = gmv_res.expected_return
            frontier_vol = gmv_res.volatility
            frontier_vols[i] = frontier_vol
            frontier_sharpes[i] = gmv_res.sharpe_ratio
            w_curr = gmv_res.weights.copy()
            continue

        target_opt = optimize_target_return(
            expected_returns=mu,
            cov_matrix=cov,
            target_return=target_r,
            rf=rf,
            bounds=parsed_bounds,
            initial_weights=w_curr,
        )

        if target_opt.success:
            w_curr = target_opt.weights.copy()
            frontier_weights[i] = target_opt.weights
            frontier_returns[i] = target_opt.expected_return
            frontier_vols[i] = target_opt.volatility
            frontier_sharpes[i] = target_opt.sharpe_ratio
        else:
            # Fallback interpolation using previous point and max asset weight
            alpha = float(i) / max(float(num_points - 1), 1.0)
            max_asset_idx = int(np.argmax(mu))
            w_target_max = np.zeros(n, dtype=np.float64)
            w_target_max[max_asset_idx] = 1.0
            w_target_max = normalize_and_clamp_weights(w_target_max, parsed_bounds)
            w_interp = (1.0 - alpha) * gmv_res.weights + alpha * w_target_max
            w_curr = normalize_and_clamp_weights(w_interp, parsed_bounds)

            vol = float(np.sqrt(max(w_curr.T @ cov @ w_curr, 0.0)))
            exp_ret = float(w_curr @ mu)
            sr = float((exp_ret - rf) / max(vol, 1e-12))

            frontier_weights[i] = w_curr
            frontier_returns[i] = exp_ret
            frontier_vols[i] = vol
            frontier_sharpes[i] = sr

    # Ensure strictly monotonic target returns on the upper frontier
    for i in range(1, num_points):
        if frontier_returns[i] < frontier_returns[i - 1]:
            frontier_returns[i] = frontier_returns[i - 1]

    # 5. Generate Capital Allocation Line (CAL)
    max_frontier_vol = float(np.max(frontier_vols))
    asset_max_vol = float(np.max(np.sqrt(np.diag(cov))))
    cal_end_vol = max(0.50, float(1.3 * max(max_frontier_vol, asset_max_vol, ms_res.volatility)))

    cal_line = compute_capital_allocation_line(
        ms_res, rf=rf, max_vol=cal_end_vol, num_points=num_points
    )

    return EfficientFrontierResult(
        returns=frontier_returns,
        volatilities=frontier_vols,
        weights=frontier_weights,
        sharpe_ratios=frontier_sharpes,
        gmv_portfolio=gmv_res,
        max_sharpe_portfolio=ms_res,
        cal_line=cal_line,
    )
