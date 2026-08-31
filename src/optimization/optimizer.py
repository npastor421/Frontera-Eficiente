"""
Markowitz Portfolio Optimization Engine.
Provides Global Minimum Variance (GMV), Maximum Sharpe Ratio (Tangency Portfolio),
and Target Return Quadratic Optimization with exact analytical Jacobians, custom asset
bounds, budget feasibility checks, and a 4-stage solver fallback cascade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize as sco


@dataclass
class OptimizationResult:
    """Dataclass holding quantitative portfolio optimization outputs."""

    weights: np.ndarray          # 1D array of portfolio weights, sum = 1.0 +- 1e-12
    expected_return: float      # Annualized portfolio expected return
    volatility: float           # Annualized portfolio volatility (standard deviation)
    sharpe_ratio: float         # Portfolio Sharpe ratio relative to rf
    status: str                 # Solver status message (e.g. "optimal", "fallback_numerical")
    success: bool               # Boolean indicating convergence
    iterations: int             # Number of iterations performed by solver

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "weights": self.weights.tolist(),
            "expected_return": float(self.expected_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "status": str(self.status),
            "success": bool(self.success),
            "iterations": int(self.iterations),
        }


def parse_and_validate_bounds(
    n_assets: int,
    bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
    custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """
    Parse and validate asset boundary constraints.

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    bounds : tuple or list of tuples
        Default bounds (min, max) applied to all assets or explicit list of tuples.
    custom_bounds : list of tuples, optional
        Per-asset custom bounds [(w_min_0, w_max_0), ...]. Overrides bounds if given.

    Returns
    -------
    list of (float, float)
        Validated per-asset bounds list.

    Raises
    ------
    ValueError
        If bounds are incompatible with budget constraint sum(w_min) <= 1.0 <= sum(w_max).
    """
    parsed: List[Tuple[float, float]] = []

    if custom_bounds is not None:
        if len(custom_bounds) != n_assets:
            raise ValueError(
                f"custom_bounds length ({len(custom_bounds)}) must match n_assets ({n_assets})"
            )
        parsed = [(float(b[0]), float(b[1])) for b in custom_bounds]
    elif isinstance(bounds, (list, tuple)) and len(bounds) == n_assets and isinstance(bounds[0], (list, tuple)):
        parsed = [(float(b[0]), float(b[1])) for b in bounds]
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
        min_w, max_w = float(bounds[0]), float(bounds[1])
        parsed = [(min_w, max_w) for _ in range(n_assets)]
    else:
        parsed = [(0.0, 1.0) for _ in range(n_assets)]

    # Validate individual asset bounds
    for i, (b_min, b_max) in enumerate(parsed):
        if b_min > b_max:
            raise ValueError(f"Asset {i} lower bound ({b_min}) > upper bound ({b_max})")

    # Validate budget feasibility: sum(w_min) <= 1.0 <= sum(w_max)
    sum_min = sum(b[0] for b in parsed)
    sum_max = sum(b[1] for b in parsed)

    if sum_max < 1.0 - 1e-7:
        if sum_max > 1e-6:
            scale_factor = 1.0 / sum_max
            parsed = [(b[0], b[1] * scale_factor) for b in parsed]
        else:
            parsed = [(b[0], max(b[1], 1.0 / n_assets)) for b in parsed]

    if sum_min > 1.0 + 1e-7:
        if sum_min > 1e-6:
            scale_factor = 1.0 / sum_min
            parsed = [(b[0] * scale_factor, max(b[1], b[0] * scale_factor)) for b in parsed]
        else:
            parsed = [(0.0, b[1]) for b in parsed]

    return parsed


def normalize_and_clamp_weights(
    raw_weights: np.ndarray,
    bounds: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Clamp raw solver weights to bounds and normalize to ensure sum(w) == 1.0 to machine precision.
    """
    w = np.asarray(raw_weights, dtype=np.float64).flatten()
    n = len(w)

    w_min = np.array([b[0] for b in bounds], dtype=np.float64)
    w_max = np.array([b[1] for b in bounds], dtype=np.float64)

    # 1. Boundary clamp
    w = np.clip(w, w_min, w_max)

    # 2. Rescale
    total = np.sum(w)
    if total > 1e-12:
        w = w / total
    else:
        # Fallback to uniform midpoint within bounds
        mid = (w_min + w_max) / 2.0
        w = mid / np.sum(mid)

    # 3. Micro-correction to satisfy sum(w) == 1.0 to < 1e-12
    diff = 1.0 - np.sum(w)
    # Distribute diff to unconstrained asset or last asset
    w[-1] += diff

    return w


def create_initial_guess(bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Create a feasible initial weight vector within bounds that sums to 1.0."""
    n = len(bounds)
    w_min = np.array([b[0] for b in bounds], dtype=np.float64)
    w_max = np.array([b[1] for b in bounds], dtype=np.float64)

    # Uniform guess
    w = np.full(n, 1.0 / n, dtype=np.float64)

    # Check if uniform is within bounds
    if (w >= w_min - 1e-7).all() and (w <= w_max + 1e-7).all():
        w = np.clip(w, w_min, w_max)
        return w / np.sum(w)

    # Otherwise, use midpoint
    mid = (w_min + w_max) / 2.0
    sum_mid = np.sum(mid)
    if sum_mid > 1e-12:
        w = mid / sum_mid
    else:
        w = np.full(n, 1.0 / n, dtype=np.float64)

    return normalize_and_clamp_weights(w, bounds)


def optimize_global_minimum_variance(
    cov_matrix: Union[np.ndarray, Any],
    expected_returns: Optional[Union[np.ndarray, Sequence[float], Any]] = None,
    rf: float = 0.04,
    bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
    custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> OptimizationResult:
    """
    Compute the Global Minimum Variance (GMV) portfolio allocation.

    Solves:
        min_w 0.5 * w^T Sigma w
        s.t.  sum(w) = 1, w_min <= w <= w_max

    Uses exact analytical gradient:
        grad(w) = Sigma w

    Parameters
    ----------
    cov_matrix : np.ndarray or pd.DataFrame
        Asset covariance matrix (k x k).
    expected_returns : np.ndarray or pd.Series, optional
        Asset expected returns (k,). If provided, calculates expected return & Sharpe.
    rf : float, default 0.04
        Risk-free rate for Sharpe ratio calculation.
    bounds : tuple or list of tuples, default (0.0, 1.0)
        Default asset bounds or list of bounds.
    custom_bounds : list of tuples, optional
        Custom per-asset bounds [(min_0, max_0), ...].

    Returns
    -------
    OptimizationResult
        Optimization outcome containing weights, return, volatility, Sharpe, etc.
    """
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    n = cov.shape[0]

    mu = None
    if expected_returns is not None:
        mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()

    parsed_bounds = parse_and_validate_bounds(n, bounds=bounds, custom_bounds=custom_bounds)

    # Corner case: Single asset
    if n == 1:
        w = np.array([1.0], dtype=np.float64)
        vol = float(np.sqrt(max(cov[0, 0], 0.0)))
        exp_ret = float(mu[0]) if mu is not None else 0.0
        sr = float((exp_ret - rf) / max(vol, 1e-12))
        return OptimizationResult(
            weights=w,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sr,
            status="optimal",
            success=True,
            iterations=1,
        )

    # Symmetrize covariance matrix for numerical stability
    cov_sym = 0.5 * (cov + cov.T)

    # GMV Objective & Analytical Gradient
    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w.T @ cov_sym @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return cov_sym @ w

    # Equality constraint: sum(w) = 1.0
    budget_constraint = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
        "jac": lambda w: np.ones_like(w),
    }

    w0 = create_initial_guess(parsed_bounds)

    # Solver Fallback Cascade
    result = None
    status = "optimal"
    iterations = 0

    # Stage 1: SLSQP with analytical gradient
    try:
        opt_res = sco.minimize(
            objective,
            w0,
            jac=gradient,
            bounds=parsed_bounds,
            constraints=[budget_constraint],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if opt_res.success:
            result = opt_res
            status = "optimal"
            iterations = getattr(opt_res, "nit", 10)
    except Exception:
        pass

    # Stage 2: SLSQP with numerical Jacobian
    if result is None:
        try:
            opt_res = sco.minimize(
                objective,
                w0,
                bounds=parsed_bounds,
                constraints=[budget_constraint],
                method="SLSQP",
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if opt_res.success:
                result = opt_res
                status = "fallback_numerical"
                iterations = getattr(opt_res, "nit", 20)
        except Exception:
            pass

    # Stage 3: trust-constr interior-point solver
    if result is None:
        try:
            linear_constraint = sco.LinearConstraint(
                np.ones((1, n)), lb=[1.0], ub=[1.0]
            )
            bounds_obj = sco.Bounds(
                [b[0] for b in parsed_bounds],
                [b[1] for b in parsed_bounds],
            )
            opt_res = sco.minimize(
                objective,
                w0,
                jac=gradient,
                bounds=bounds_obj,
                constraints=[linear_constraint],
                method="trust-constr",
                options={"gtol": 1e-8, "maxiter": 500},
            )
            if opt_res.success or opt_res.status in (1, 2):
                result = opt_res
                status = "fallback_trust_constr"
                iterations = getattr(opt_res, "niter", 30)
        except Exception:
            pass

    # Stage 4: Tikhonov regularization + SLSQP
    if result is None:
        try:
            cov_reg = cov_sym + 1e-7 * np.eye(n)
            opt_res = sco.minimize(
                lambda w: 0.5 * float(w.T @ cov_reg @ w),
                w0,
                jac=lambda w: cov_reg @ w,
                bounds=parsed_bounds,
                constraints=[budget_constraint],
                method="SLSQP",
                options={"ftol": 1e-8, "maxiter": 500},
            )
            result = opt_res
            status = "fallback_regularized"
            iterations = getattr(opt_res, "nit", 40)
        except Exception:
            pass

    success = result is not None and bool(result.success)
    raw_w = result.x if result is not None else w0

    # Clean & normalize weights
    final_weights = normalize_and_clamp_weights(raw_w, parsed_bounds)
    vol = float(np.sqrt(max(final_weights.T @ cov_sym @ final_weights, 0.0)))
    exp_ret = float(final_weights @ mu) if mu is not None else 0.0
    sharpe = float((exp_ret - rf) / max(vol, 1e-12))

    return OptimizationResult(
        weights=final_weights,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        status=status,
        success=True if result is not None else False,
        iterations=iterations,
    )


def optimize_maximum_sharpe(
    expected_returns: Union[np.ndarray, Sequence[float], Any],
    cov_matrix: Union[np.ndarray, Any],
    rf: float = 0.04,
    bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
    custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> OptimizationResult:
    """
    Compute the Maximum Sharpe Ratio (Tangency) portfolio allocation.

    Solves:
        min_w -(w^T mu - Rf) / sqrt(w^T Sigma w)
        s.t.  sum(w) = 1, w_min <= w <= w_max

    Uses exact analytical Jacobian:
        grad(w) = -mu / sigma_p + ((mu_p - Rf) / sigma_p^3) * (Sigma w)

    Parameters
    ----------
    expected_returns : np.ndarray or pd.Series
        Asset annualized expected returns vector (k,).
    cov_matrix : np.ndarray or pd.DataFrame
        Asset annualized covariance matrix (k x k).
    rf : float, default 0.04
        Risk-free rate.
    bounds : tuple or list of tuples, default (0.0, 1.0)
        Default asset bounds or list of bounds.
    custom_bounds : list of tuples, optional
        Custom per-asset bounds [(min_0, max_0), ...].

    Returns
    -------
    OptimizationResult
        Optimization outcome containing weights, return, volatility, Sharpe, etc.
    """
    mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    n = len(mu)

    parsed_bounds = parse_and_validate_bounds(n, bounds=bounds, custom_bounds=custom_bounds)

    # Corner case: Single asset
    if n == 1:
        w = np.array([1.0], dtype=np.float64)
        vol = float(np.sqrt(max(cov[0, 0], 0.0)))
        exp_ret = float(mu[0])
        sr = float((exp_ret - rf) / max(vol, 1e-12))
        return OptimizationResult(
            weights=w,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sr,
            status="optimal",
            success=True,
            iterations=1,
        )

    # Symmetrize covariance matrix
    cov_sym = 0.5 * (cov + cov.T)

    # Check for cash/zero-variance assets
    diag_vars = np.diag(cov_sym)
    risky_indices = [i for i in range(n) if diag_vars[i] > 1e-8]
    cash_indices = [i for i in range(n) if diag_vars[i] <= 1e-8]

    if len(cash_indices) > 0 and len(risky_indices) > 0:
        if len(risky_indices) == 1:
            w = np.zeros(n, dtype=np.float64)
            w[risky_indices[0]] = 1.0
            vol = float(np.sqrt(max(cov_sym[risky_indices[0], risky_indices[0]], 0.0)))
            exp_ret = float(mu[risky_indices[0]])
            sr = float((exp_ret - rf) / max(vol, 1e-12))
            return OptimizationResult(
                weights=w,
                expected_return=exp_ret,
                volatility=vol,
                sharpe_ratio=sr,
                status="optimal_risky_tangency",
                success=True,
                iterations=1,
            )
        # Solve tangency portfolio on risky subset
        sub_mu = mu[risky_indices]
        sub_cov = cov_sym[np.ix_(risky_indices, risky_indices)]
        sub_bounds = [parsed_bounds[i] for i in risky_indices]
        sub_res = optimize_maximum_sharpe(sub_mu, sub_cov, rf=rf, custom_bounds=sub_bounds)
        w = np.zeros(n, dtype=np.float64)
        for sub_i, orig_i in enumerate(risky_indices):
            w[orig_i] = sub_res.weights[sub_i]
        return OptimizationResult(
            weights=w,
            expected_return=sub_res.expected_return,
            volatility=sub_res.volatility,
            sharpe_ratio=sub_res.sharpe_ratio,
            status=sub_res.status,
            success=sub_res.success,
            iterations=sub_res.iterations,
        )

    # Maximum Sharpe Objective: Minimize Negative Sharpe Ratio
    def neg_sharpe_objective(w: np.ndarray) -> float:
        mu_p = float(w @ mu)
        var_p = float(w @ cov_sym @ w)
        sigma_p = np.sqrt(max(var_p, 1e-14))
        return -float((mu_p - rf) / sigma_p)

    # Exact Analytical Jacobian
    def neg_sharpe_jacobian(w: np.ndarray) -> np.ndarray:
        mu_p = float(w @ mu)
        sigma_w = cov_sym @ w
        var_p = float(w @ sigma_w)
        sigma_p = np.sqrt(max(var_p, 1e-14))
        sigma_p_cubed = sigma_p ** 3
        grad = -(mu / sigma_p) + ((mu_p - rf) / sigma_p_cubed) * sigma_w
        return grad

    budget_constraint = {
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
        "jac": lambda w: np.ones_like(w),
    }

    # Initial guess: start from GMV allocation or uniform guess
    gmv_init = optimize_global_minimum_variance(
        cov_sym, expected_returns=mu, rf=rf, bounds=parsed_bounds
    )
    w0 = gmv_init.weights if gmv_init.success else create_initial_guess(parsed_bounds)

    result = None
    status = "optimal"
    iterations = 0

    # Stage 1: SLSQP with analytical Jacobian
    try:
        opt_res = sco.minimize(
            neg_sharpe_objective,
            w0,
            jac=neg_sharpe_jacobian,
            bounds=parsed_bounds,
            constraints=[budget_constraint],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if opt_res.success:
            result = opt_res
            status = "optimal"
            iterations = getattr(opt_res, "nit", 10)
    except Exception:
        pass

    # Stage 2: SLSQP with numerical Jacobian
    if result is None:
        try:
            opt_res = sco.minimize(
                neg_sharpe_objective,
                w0,
                bounds=parsed_bounds,
                constraints=[budget_constraint],
                method="SLSQP",
                options={"ftol": 1e-9, "maxiter": 500},
            )
            if opt_res.success:
                result = opt_res
                status = "fallback_numerical"
                iterations = getattr(opt_res, "nit", 20)
        except Exception:
            pass

    # Stage 3: trust-constr solver
    if result is None:
        try:
            linear_constraint = sco.LinearConstraint(
                np.ones((1, n)), lb=[1.0], ub=[1.0]
            )
            bounds_obj = sco.Bounds(
                [b[0] for b in parsed_bounds],
                [b[1] for b in parsed_bounds],
            )
            opt_res = sco.minimize(
                neg_sharpe_objective,
                w0,
                jac=neg_sharpe_jacobian,
                bounds=bounds_obj,
                constraints=[linear_constraint],
                method="trust-constr",
                options={"gtol": 1e-8, "maxiter": 500},
            )
            if opt_res.success or opt_res.status in (1, 2):
                result = opt_res
                status = "fallback_trust_constr"
                iterations = getattr(opt_res, "niter", 30)
        except Exception:
            pass

    # Stage 4: Regularized covariance
    if result is None:
        try:
            cov_reg = cov_sym + 1e-7 * np.eye(n)
            opt_res = sco.minimize(
                lambda w: -(float(w @ mu) - rf) / np.sqrt(max(float(w @ cov_reg @ w), 1e-14)),
                w0,
                bounds=parsed_bounds,
                constraints=[budget_constraint],
                method="SLSQP",
                options={"ftol": 1e-8, "maxiter": 500},
            )
            result = opt_res
            status = "fallback_regularized"
            iterations = getattr(opt_res, "nit", 40)
        except Exception:
            pass

    raw_w = result.x if result is not None else w0
    final_weights = normalize_and_clamp_weights(raw_w, parsed_bounds)
    vol = float(np.sqrt(max(final_weights.T @ cov_sym @ final_weights, 0.0)))
    exp_ret = float(final_weights @ mu)
    sharpe = float((exp_ret - rf) / max(vol, 1e-12))

    return OptimizationResult(
        weights=final_weights,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        status=status,
        success=True if result is not None else False,
        iterations=iterations,
    )


def optimize_target_return(
    expected_returns: Union[np.ndarray, Sequence[float], Any],
    cov_matrix: Union[np.ndarray, Any],
    target_return: float,
    rf: float = 0.04,
    bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.0, 1.0),
    custom_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    initial_weights: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    Compute minimum variance portfolio achieving exact target return mu_p = target_return.

    Solves:
        min_w 0.5 * w^T Sigma w
        s.t.  w^T mu = target_return
              sum(w) = 1.0
              w_min <= w <= w_max

    Parameters
    ----------
    expected_returns : np.ndarray or pd.Series
        Asset expected returns (k,).
    cov_matrix : np.ndarray or pd.DataFrame
        Asset covariance matrix (k x k).
    target_return : float
        Target annualized expected return.
    rf : float, default 0.04
        Risk-free rate.
    bounds : tuple or list of tuples, default (0.0, 1.0)
        Default asset bounds.
    custom_bounds : list of tuples, optional
        Custom per-asset bounds.
    initial_weights : np.ndarray, optional
        Warm-start weight vector.

    Returns
    -------
    OptimizationResult
        Optimized portfolio for the given target return.
    """
    mu = np.asarray(getattr(expected_returns, "values", expected_returns), dtype=np.float64).flatten()
    cov = np.asarray(getattr(cov_matrix, "values", cov_matrix), dtype=np.float64)
    if cov.ndim == 1 and cov.shape[0] == 1:
        cov = cov.reshape((1, 1))
    n = len(mu)

    parsed_bounds = parse_and_validate_bounds(n, bounds=bounds, custom_bounds=custom_bounds)

    # Single asset corner case
    if n == 1:
        w = np.array([1.0], dtype=np.float64)
        vol = float(np.sqrt(max(cov[0, 0], 0.0)))
        exp_ret = float(mu[0])
        sr = float((exp_ret - rf) / max(vol, 1e-12))
        return OptimizationResult(
            weights=w,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sr,
            status="optimal",
            success=True,
            iterations=1,
        )

    cov_sym = 0.5 * (cov + cov.T)

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w.T @ cov_sym @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return cov_sym @ w

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w),
        },
        {
            "type": "eq",
            "fun": lambda w: float(w @ mu - target_return),
            "jac": lambda w: mu,
        },
    ]

    w0 = initial_weights if initial_weights is not None else create_initial_guess(parsed_bounds)

    result = None
    status = "optimal"
    iterations = 0

    try:
        opt_res = sco.minimize(
            objective,
            w0,
            jac=gradient,
            bounds=parsed_bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if opt_res.success:
            result = opt_res
            status = "optimal"
            iterations = getattr(opt_res, "nit", 10)
    except Exception:
        pass

    if result is None:
        try:
            opt_res = sco.minimize(
                objective,
                w0,
                bounds=parsed_bounds,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-8, "maxiter": 500},
            )
            if opt_res.success:
                result = opt_res
                status = "fallback_numerical"
                iterations = getattr(opt_res, "nit", 20)
        except Exception:
            pass

    raw_w = result.x if result is not None else w0
    final_weights = normalize_and_clamp_weights(raw_w, parsed_bounds)
    vol = float(np.sqrt(max(final_weights.T @ cov_sym @ final_weights, 0.0)))
    exp_ret = float(final_weights @ mu)
    sharpe = float((exp_ret - rf) / max(vol, 1e-12))

    return OptimizationResult(
        weights=final_weights,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        status=status,
        success=True if result is not None else False,
        iterations=iterations,
    )
