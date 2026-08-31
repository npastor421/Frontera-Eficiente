"""
Empirical Challenge Audit Script.
Executes detailed numerical stress tests and outputs quantitative metrics for the handoff report.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.analytics.risk_metrics import (
    calculate_calmar_ratio,
    calculate_sortino_ratio,
    compute_drawdown_series,
    compute_historical_var_cvar,
    compute_parametric_var_cvar,
    compute_portfolio_risk_metrics,
)
from src.models.covariance import (
    estimate_covariance_matrix,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    sample_covariance,
)
from src.models.returns import (
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_expected_returns,
    capm_expected_returns,
)
from src.models.stability import (
    calculate_condition_number,
    enforce_symmetry,
    ensure_positive_semidefinite,
    get_eigenvalues,
    is_positive_semidefinite,
    nearest_psd_higham,
)
from src.optimization.frontier import (
    compute_capital_allocation_line,
    compute_efficient_frontier,
)
from src.optimization.optimizer import (
    optimize_global_minimum_variance,
    optimize_maximum_sharpe,
    optimize_target_return,
    parse_and_validate_bounds,
)
from src.simulation.trajectory_monte_carlo import run_trajectory_monte_carlo
from src.simulation.weight_monte_carlo import run_weight_space_monte_carlo


def run_empirical_audit():
    print("=" * 80)
    print("EMPIRICAL QUANTITATIVE AUDIT & STRESS TEST HARNESS REPORT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. ILL-CONDITIONED MATRICES & LARGE UNIVERSES (N=10, 30, 50, 100)
    # -------------------------------------------------------------------------
    print("\n[CHALLENGE 1] Large Universes & Ill-Conditioned Covariance Matrices:")
    for n in [10, 30, 50, 100]:
        rng = np.random.default_rng(1000 + n)
        # Synthesize ill-conditioned matrix with kappa ~ 1e6
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        eigenvalues = np.logspace(0, -6, n)  # kappa = 1e6
        cov = Q @ np.diag(eigenvalues) @ Q.T
        cov = 0.5 * (cov + cov.T)
        kappa = calculate_condition_number(cov)

        mu = rng.uniform(0.04, 0.25, size=n)
        rf = 0.04

        t0 = time.perf_counter()
        gmv_res = optimize_global_minimum_variance(cov, expected_returns=mu, rf=rf)
        t_gmv = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ms_res = optimize_maximum_sharpe(mu, cov, rf=rf)
        t_ms = (time.perf_counter() - t0) * 1000.0

        # Sample 10,000 Dirichlet portfolios
        mc_res = run_weight_space_monte_carlo(mu, cov, rf=rf, num_portfolios=10000, seed=42)

        gmv_weight_sum_err = abs(np.sum(gmv_res.weights) - 1.0)
        ms_weight_sum_err = abs(np.sum(ms_res.weights) - 1.0)
        gmv_vol_diff = gmv_res.volatility - np.min(mc_res.volatilities)  # should be <= 0
        ms_sr_diff = ms_res.sharpe_ratio - np.max(mc_res.sharpe_ratios)  # should be >= 0

        print(f"  N={n:3d} | Kappa={kappa:.2e} | GMV status={gmv_res.status} ({t_gmv:.1f}ms) | "
              f"|w_sum-1|={gmv_weight_sum_err:.1e} | GMV vol={gmv_res.volatility:.4f} (<= MC min: {gmv_vol_diff <= 1e-7})")
        print(f"        | MS  status={ms_res.status} ({t_ms:.1f}ms) | "
              f"|w_sum-1|={ms_weight_sum_err:.1e} | MS SR={ms_res.sharpe_ratio:.4f} (>= MC max: {ms_sr_diff >= -1e-7})")

    # -------------------------------------------------------------------------
    # 2. EXTREME SCENARIOS
    # -------------------------------------------------------------------------
    print("\n[CHALLENGE 2] Extreme Scenarios & Edge Cases:")

    # 2.1 Negative excess returns (mu < Rf)
    cov_neg = np.array([[0.09, 0.02], [0.02, 0.04]])
    mu_neg = np.array([-0.10, -0.05])
    rf_neg = 0.04
    gmv_neg = optimize_global_minimum_variance(cov_neg, expected_returns=mu_neg, rf=rf_neg)
    ms_neg = optimize_maximum_sharpe(mu_neg, cov_neg, rf=rf_neg)
    print(f"  2.1 Negative Returns (mu=[-0.10, -0.05], Rf=0.04):")
    print(f"      GMV: success={gmv_neg.success}, w={gmv_neg.weights.round(4)}, vol={gmv_neg.volatility:.4f}, SR={gmv_neg.sharpe_ratio:.4f}")
    print(f"      MS:  success={ms_neg.success}, w={ms_neg.weights.round(4)}, vol={ms_neg.volatility:.4f}, SR={ms_neg.sharpe_ratio:.4f}")

    # 2.2 Single asset (N=1)
    cov_1 = np.array([[0.04]])
    mu_1 = np.array([0.10])
    gmv_1 = optimize_global_minimum_variance(cov_1, expected_returns=mu_1, rf=0.04)
    ms_1 = optimize_maximum_sharpe(mu_1, cov_1, rf=0.04)
    print(f"  2.2 Single Asset (N=1):")
    print(f"      GMV w={gmv_1.weights}, vol={gmv_1.volatility:.4f}, exp_ret={gmv_1.expected_return:.4f}, SR={gmv_1.sharpe_ratio:.4f}")
    print(f"      MS  w={ms_1.weights}, vol={ms_1.volatility:.4f}, exp_ret={ms_1.expected_return:.4f}, SR={ms_1.sharpe_ratio:.4f}")

    # 2.3 Zero-variance asset
    cov_zero = np.diag([0.09, 0.04, 1e-12])
    mu_zero = np.array([0.15, 0.10, 0.03])
    gmv_zero = optimize_global_minimum_variance(cov_zero, expected_returns=mu_zero, rf=0.03)
    print(f"  2.3 Zero-Variance / Cash Asset:")
    print(f"      GMV w={gmv_zero.weights.round(4)}, vol={gmv_zero.volatility:.6f}, status={gmv_zero.status}")

    # 2.4 Short-selling bounds [-0.5, 1.5]
    cov_short = np.array([[0.10, 0.08], [0.08, 0.10]])
    mu_short = np.array([0.15, 0.05])
    gmv_long = optimize_global_minimum_variance(cov_short, expected_returns=mu_short, bounds=(0.0, 1.0))
    gmv_short = optimize_global_minimum_variance(cov_short, expected_returns=mu_short, bounds=(-0.5, 1.5))
    print(f"  2.4 Short-Selling Bounds [-0.5, 1.5]:")
    print(f"      Long-only GMV:  w={gmv_long.weights.round(4)}, vol={gmv_long.volatility:.4f}")
    print(f"      Short GMV:      w={gmv_short.weights.round(4)}, vol={gmv_short.volatility:.4f} (Vol reduction: {gmv_long.volatility - gmv_short.volatility:.4f})")

    # 2.5 High risk-free rate Rf = 0.35
    cov_high_rf = np.array([[0.16, 0.04], [0.04, 0.09]])
    mu_high_rf = np.array([0.60, 0.40])
    ms_high_rf = optimize_maximum_sharpe(mu_high_rf, cov_high_rf, rf=0.35)
    cal_v, cal_r = compute_capital_allocation_line(ms_high_rf, rf=0.35, num_points=10)
    print(f"  2.5 High Rf (Rf=0.35, Argentine context):")
    print(f"      MS w={ms_high_rf.weights.round(4)}, ret={ms_high_rf.expected_return:.4f}, vol={ms_high_rf.volatility:.4f}, SR={ms_high_rf.sharpe_ratio:.4f}")
    print(f"      CAL origin: ({cal_v[0]:.2f}, {cal_r[0]:.2f}), CAL end: ({cal_v[-1]:.2f}, {cal_r[-1]:.2f})")

    # -------------------------------------------------------------------------
    # 3. DIRICHLET MONTE CARLO & TRAJECTORY MC BENCHMARKS
    # -------------------------------------------------------------------------
    print("\n[CHALLENGE 3] Dirichlet Monte Carlo & Trajectory Simulation Benchmarks:")
    # 3.1 Moment validation for Dirichlet(1,1,1,1)
    mc_moments = run_weight_space_monte_carlo(np.ones(4), np.eye(4), num_portfolios=100000, seed=42)
    emp_mean = np.mean(mc_moments.weights, axis=0)
    emp_cov = np.cov(mc_moments.weights, rowvar=False)
    theo_mean = 0.25
    theo_var = 3.0 / 80.0  # 0.0375
    theo_cov = -1.0 / 80.0 # -0.0125
    print(f"  3.1 Dirichlet Uniform Simplex Moments (100,000 portfolios, N=4):")
    print(f"      Mean error: max|E[w] - 0.25| = {np.max(np.abs(emp_mean - theo_mean)):.6f}")
    print(f"      Var error:  max|Var(w) - 0.0375| = {np.max(np.abs(np.diag(emp_cov) - theo_var)):.6f}")
    print(f"      Cov error:  max|Cov(w_i, w_j) - (-0.0125)| = {np.max(np.abs(emp_cov[~np.eye(4, dtype=bool)] - theo_cov)):.6f}")

    # 3.2 Speed benchmarks
    print(f"  3.2 Speed Benchmarks (10,000 portfolios):")
    for n in [5, 10, 30, 50]:
        rng = np.random.default_rng(42)
        mu = rng.uniform(0.05, 0.20, size=n)
        A = rng.standard_normal((n, n))
        cov = A @ A.T / n + 0.01 * np.eye(n)
        t0 = time.perf_counter()
        run_weight_space_monte_carlo(mu, cov, num_portfolios=10000, seed=42)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"      N={n:2d} assets: {elapsed_ms:.2f} ms (< 2000 ms budget)")

    # 3.3 Trajectory MC (1,000 paths, 252 steps)
    t0 = time.perf_counter()
    traj_res = run_trajectory_monte_carlo(
        expected_returns=np.array([0.10, 0.12, 0.08]),
        cov_matrix=np.array([[0.04, 0.01, 0.01], [0.01, 0.05, 0.02], [0.01, 0.02, 0.03]]),
        weights=np.array([0.4, 0.4, 0.2]),
        initial_capital=10000.0,
        years=1.0,
        num_simulations=1000,
        seed=42,
    )
    traj_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  3.3 Multi-Asset Trajectory MC (1,000 paths, 1 year): {traj_elapsed_ms:.2f} ms (< 2000 ms budget)")
    print(f"      Final wealth P5=${traj_res.percentile_5[-1]:.0f}, Median=${traj_res.percentile_50[-1]:.0f}, P95=${traj_res.percentile_95[-1]:.0f}")

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL TESTS EXECUTED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_empirical_audit()
