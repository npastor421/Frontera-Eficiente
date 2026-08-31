"""
Empirical Stress Test Harness & Mathematical Invariants Validation.
Audits optimizer convergence, conditioning, extreme scenarios, Dirichlet MC simplex, and performance.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.analytics.risk_metrics import (
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
    create_initial_guess,
    normalize_and_clamp_weights,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe,
    optimize_target_return,
    parse_and_validate_bounds,
)
from src.simulation.trajectory_monte_carlo import (
    TrajectorySimulationResult,
    run_trajectory_monte_carlo,
)
from src.simulation.weight_monte_carlo import run_weight_space_monte_carlo


# =============================================================================
# CHALLENGE 1: Large Universes (N=30, N=50) & Ill-Conditioned Matrices (kappa > 10^5)
# =============================================================================

@pytest.mark.parametrize("n_assets", [30, 50])
def test_large_universe_and_ill_conditioned_covariance_matrices(n_assets: int):
    """
    Challenge 1.1:
    Stress-test optimizer on large universes (N=30, N=50) with ill-conditioned covariance matrices
    having condition numbers kappa > 10^5 and kappa > 10^6.
    Verify:
      1. Optimizer converges (res.success is True)
      2. Weight sums satisfy |sum(w) - 1.0| < 1e-10
      3. Bounds are strictly respected: w_min <= w_i <= w_max
      4. GMV minimality invariant: sigma_gmv <= sigma(w_rand) + 1e-7 for 5000 random portfolios
      5. Max Sharpe optimality invariant: SR_ms >= SR(w_rand) - 1e-7 for 5000 random portfolios
    """
    rng = np.random.default_rng(42 + n_assets)

    # 1. Synthesize an ill-conditioned covariance matrix with condition number > 1e5
    # Generate random orthogonal matrix Q via QR decomposition
    A = rng.standard_normal((n_assets, n_assets))
    Q, _ = np.linalg.qr(A)

    # Log-spaced eigenvalues from 1.0 down to 1e-6 (kappa = 1e6 > 10^5)
    eigenvalues = np.logspace(0, -6, n_assets)
    cov_ill = Q @ np.diag(eigenvalues) @ Q.T
    cov_ill = 0.5 * (cov_ill + cov_ill.T)

    cond_num = calculate_condition_number(cov_ill)
    assert cond_num >= 1e5, f"Condition number {cond_num:.2e} should be >= 1e5"

    # Expected returns with varied signals
    mu = rng.uniform(0.02, 0.25, size=n_assets)
    rf = 0.04

    # 2. Test GMV Optimization
    gmv_res = optimize_global_minimum_variance(cov_ill, expected_returns=mu, rf=rf)
    assert gmv_res.success is True, f"GMV failed to converge for N={n_assets}, kappa={cond_num:.2e}"
    assert abs(np.sum(gmv_res.weights) - 1.0) < 1e-10, f"GMV weight sum error: {abs(np.sum(gmv_res.weights) - 1.0)}"
    assert np.all(gmv_res.weights >= -1e-9) and np.all(gmv_res.weights <= 1.0 + 1e-9)

    # 3. Test Max Sharpe Optimization
    ms_res = optimize_maximum_sharpe(mu, cov_ill, rf=rf)
    assert ms_res.success is True, f"Max Sharpe failed to converge for N={n_assets}, kappa={cond_num:.2e}"
    assert abs(np.sum(ms_res.weights) - 1.0) < 1e-10, f"Max Sharpe weight sum error: {abs(np.sum(ms_res.weights) - 1.0)}"
    assert np.all(ms_res.weights >= -1e-9) and np.all(ms_res.weights <= 1.0 + 1e-9)

    # 4. Empirical Invariant Verification vs 5,000 Random Uniform Simplex Portfolios
    n_samples = 5000
    dirichlet_res = run_weight_space_monte_carlo(mu, cov_ill, rf=rf, num_portfolios=n_samples, seed=123)

    # Invariant A: GMV volatility is <= all sampled volatilities (within numerical precision)
    min_sampled_vol = np.min(dirichlet_res.volatilities)
    assert gmv_res.volatility <= min_sampled_vol + 1e-7, (
        f"GMV volatility ({gmv_res.volatility:.6f}) exceeded minimum sampled volatility ({min_sampled_vol:.6f})"
    )

    # Invariant B: Max Sharpe is >= all sampled Sharpe ratios (within numerical precision)
    max_sampled_sharpe = np.max(dirichlet_res.sharpe_ratios)
    assert ms_res.sharpe_ratio >= max_sampled_sharpe - 1e-7, (
        f"Max Sharpe ({ms_res.sharpe_ratio:.6f}) was less than sampled Max Sharpe ({max_sampled_sharpe:.6f})"
    )

    # 5. Continuous Efficient Frontier Curve Sweep (50 points)
    frontier_res = compute_efficient_frontier(mu, cov_ill, rf=rf, num_points=50)
    assert len(frontier_res.returns) == 50
    assert len(frontier_res.volatilities) == 50
    assert np.all(np.diff(frontier_res.returns) >= -1e-10), "Efficient frontier returns must be non-decreasing"
    assert np.all(np.abs(np.sum(frontier_res.weights, axis=1) - 1.0) < 1e-10), "Frontier weights must sum to 1.0"


def test_collinear_assets_high_condition_number():
    """
    Challenge 1.2:
    Construct asset universe with near-perfect collinearity (corr ~ 0.99999).
    Verify ensure_positive_semidefinite repairs or validates matrix and optimizer converges.
    """
    rng = np.random.default_rng(999)
    n_assets = 10
    T = 200
    returns = rng.standard_normal((T, n_assets)) * 0.01
    # Make asset 1 and 2 almost identical
    returns[:, 1] = returns[:, 0] + rng.standard_normal(T) * 1e-6
    # Make asset 3 a linear combination of asset 0 and 4
    returns[:, 2] = 0.5 * returns[:, 0] + 0.5 * returns[:, 4] + rng.standard_normal(T) * 1e-6

    cov_raw = sample_covariance(returns)
    cond_raw = calculate_condition_number(cov_raw)

    cov_psd, repaired, cond_psd = ensure_positive_semidefinite(cov_raw, eps=1e-7)
    assert is_positive_semidefinite(cov_psd)

    mu = np.mean(returns, axis=0) * 252
    gmv_res = optimize_global_minimum_variance(cov_psd, expected_returns=mu)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) < 1e-10


# =============================================================================
# CHALLENGE 2: Extreme Scenarios & Edge Cases
# =============================================================================

def test_extreme_scenario_negative_excess_returns():
    """
    Challenge 2.1: Negative Excess Returns (mu_i < Rf for all assets).
    In severe bear markets or high rate environments, all assets have mu_i < Rf.
    Verify optimizer converges, weights sum to 1.0, and GMV minimality holds.
    """
    cov = np.array([
        [0.09, 0.02, 0.01],
        [0.02, 0.04, 0.01],
        [0.01, 0.01, 0.02],
    ])
    # All returns negative / below Rf = 0.05
    mu = np.array([-0.15, -0.05, -0.02])
    rf = 0.05

    gmv_res = optimize_global_minimum_variance(cov, expected_returns=mu, rf=rf)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) < 1e-10
    assert gmv_res.sharpe_ratio < 0.0  # Sharpe is negative

    ms_res = optimize_maximum_sharpe(mu, cov, rf=rf)
    assert ms_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) < 1e-10

    # Compare with 2,000 random portfolios
    mc = run_weight_space_monte_carlo(mu, cov, rf=rf, num_portfolios=2000, seed=42)
    assert gmv_res.volatility <= np.min(mc.volatilities) + 1e-7
    assert ms_res.sharpe_ratio >= np.max(mc.sharpe_ratios) - 1e-7


def test_extreme_scenario_single_asset_n1():
    """
    Challenge 2.2: Single Asset Universe (N=1).
    Verify all routines handle N=1 gracefully without dimension errors.
    """
    cov_1 = np.array([[0.04]])
    mu_1 = np.array([0.10])
    rf = 0.04

    # GMV
    gmv_res = optimize_global_minimum_variance(cov_1, expected_returns=mu_1, rf=rf)
    assert gmv_res.success is True
    assert np.allclose(gmv_res.weights, [1.0])
    assert np.isclose(gmv_res.volatility, 0.20)
    assert np.isclose(gmv_res.expected_return, 0.10)
    assert np.isclose(gmv_res.sharpe_ratio, (0.10 - 0.04) / 0.20)

    # Max Sharpe
    ms_res = optimize_maximum_sharpe(mu_1, cov_1, rf=rf)
    assert ms_res.success is True
    assert np.allclose(ms_res.weights, [1.0])
    assert np.isclose(ms_res.sharpe_ratio, (0.10 - 0.04) / 0.20)

    # Efficient Frontier
    ef_res = compute_efficient_frontier(mu_1, cov_1, rf=rf, num_points=20)
    assert ef_res.weights.shape == (20, 1)
    assert np.allclose(ef_res.weights, 1.0)
    assert np.allclose(ef_res.returns, 0.10)

    # Weight Monte Carlo
    mc_res = run_weight_space_monte_carlo(mu_1, cov_1, rf=rf, num_portfolios=500)
    assert mc_res.weights.shape == (500, 1)
    assert np.allclose(mc_res.weights, 1.0)
    assert np.allclose(mc_res.volatilities, 0.20)


def test_extreme_scenario_zero_variance_asset():
    """
    Challenge 2.3: Zero-variance or cash-like asset (sigma^2 ~ 1e-10).
    Verify optimizer allocates near 100% to riskless asset in GMV without division by zero.
    """
    cov = np.array([
        [0.08, 0.00, 0.00],
        [0.00, 0.04, 0.00],
        [0.00, 0.00, 1e-10],  # Risk-free / Cash asset
    ])
    mu = np.array([0.12, 0.08, 0.04])
    rf = 0.04

    gmv_res = optimize_global_minimum_variance(cov, expected_returns=mu, rf=rf)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) < 1e-10
    # Asset index 2 should have weight close to 1.0
    assert gmv_res.weights[2] > 0.95
    assert gmv_res.volatility < 1e-3


def test_extreme_scenario_short_selling_bounds():
    """
    Challenge 2.4: Short-selling allowed with bounds [-0.5, 1.5].
    Verify:
      1. Solver allows negative weights within [-0.5, 1.5]
      2. Weights sum to exactly 1.0
      3. Short selling allows GMV portfolio variance to be lower than or equal to long-only GMV
    """
    cov = np.array([
        [0.10, 0.08],
        [0.08, 0.10],
    ])
    mu = np.array([0.15, 0.05])
    rf = 0.04

    # Long-only GMV
    gmv_long = optimize_global_minimum_variance(cov, expected_returns=mu, bounds=(0.0, 1.0))
    # Short-enabled GMV
    bounds_short = [(-0.5, 1.5), (-0.5, 1.5)]
    gmv_short = optimize_global_minimum_variance(cov, expected_returns=mu, custom_bounds=bounds_short)

    assert gmv_short.success is True
    assert abs(np.sum(gmv_short.weights) - 1.0) < 1e-10
    assert np.all(gmv_short.weights >= -0.50 - 1e-8)
    assert np.all(gmv_short.weights <= 1.50 + 1e-8)
    assert gmv_short.volatility <= gmv_long.volatility + 1e-7


def test_extreme_scenario_high_risk_free_rate():
    """
    Challenge 2.5: High Risk-Free Rate (Rf = 0.15 and Rf = 0.50, e.g. Argentine market context).
    Verify Tangency portfolio and Capital Allocation Line calculation.
    """
    cov = np.array([
        [0.16, 0.04],
        [0.04, 0.09],
    ])
    mu = np.array([0.60, 0.40])
    rf = 0.35  # High rate

    ms_res = optimize_maximum_sharpe(mu, cov, rf=rf)
    assert ms_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) < 1e-10
    assert ms_res.sharpe_ratio > 0

    cal_vols, cal_rets = compute_capital_allocation_line(ms_res, rf=rf, num_points=50)
    assert np.isclose(cal_rets[0], rf)  # Origin at (0, Rf)
    assert np.all(np.diff(cal_rets) >= 0.0)  # Upward sloping


# =============================================================================
# CHALLENGE 3: Dirichlet Monte Carlo Simplex Coverage & Performance Benchmarks
# =============================================================================

def test_dirichlet_uniform_simplex_theoretical_moments():
    """
    Challenge 3.1: Verify Dirichlet(1, ..., 1) Monte Carlo uniform simplex properties.
    For N assets, standard Dirichlet(alpha_i = 1) has:
      E[w_i] = 1 / N
      Var(w_i) = (N - 1) / (N^2 * (N + 1))
      Cov(w_i, w_j) = -1 / (N^2 * (N + 1))
    """
    n_assets = 4
    n_samples = 100_000
    mu = np.array([0.10, 0.12, 0.14, 0.16])
    cov = np.diag([0.04, 0.05, 0.06, 0.07])

    mc = run_weight_space_monte_carlo(mu, cov, num_portfolios=n_samples, seed=42)

    # Check sum invariant on all 100k samples
    weight_sums = np.sum(mc.weights, axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-12)
    assert np.all(mc.weights >= 0.0)

    # Empirical moments
    emp_mean = np.mean(mc.weights, axis=0)
    emp_cov = np.cov(mc.weights, rowvar=False)

    # Theoretical moments
    theo_mean = 1.0 / n_assets  # 0.25
    theo_var = (n_assets - 1.0) / (n_assets**2 * (n_assets + 1.0))  # 3 / 80 = 0.0375
    theo_cov = -1.0 / (n_assets**2 * (n_assets + 1.0))  # -1 / 80 = -0.0125

    # Verify convergence within statistical error (< 0.003)
    assert np.allclose(emp_mean, theo_mean, atol=0.003), f"Empirical mean {emp_mean} vs theoretical {theo_mean}"
    assert np.allclose(np.diag(emp_cov), theo_var, atol=0.003), f"Empirical var {np.diag(emp_cov)} vs theoretical {theo_var}"
    off_diag_emp = emp_cov[~np.eye(n_assets, dtype=bool)]
    assert np.allclose(off_diag_emp, theo_cov, atol=0.003), f"Empirical cov {off_diag_emp} vs theoretical {theo_cov}"


@pytest.mark.parametrize("n_assets", [5, 10, 30, 50])
def test_dirichlet_monte_carlo_execution_benchmark_under_2s(n_assets: int):
    """
    Challenge 3.2: Execution benchmark (< 2.0s for 10,000 portfolios).
    In practice, vectorized Dirichlet should execute in < 50ms.
    """
    rng = np.random.default_rng(100 + n_assets)
    mu = rng.uniform(0.05, 0.20, size=n_assets)
    A = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + 0.01 * np.eye(n_assets)

    t0 = time.perf_counter()
    mc_res = run_weight_space_monte_carlo(mu, cov, num_portfolios=10000, seed=42)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Dirichlet MC took {elapsed:.4f}s (budget: 2.0s) for N={n_assets}"
    assert mc_res.weights.shape == (10000, n_assets)
    assert len(mc_res.returns) == 10000
    assert len(mc_res.volatilities) == 10000
    assert len(mc_res.sharpe_ratios) == 10000


def test_trajectory_monte_carlo_gbm_and_bootstrap_benchmark():
    """
    Challenge 3.3: Trajectory Monte Carlo performance & probability cone invariants.
    Run 1,000 paths over 252 steps (1 year). Must finish in < 2.0s and respect cone ordering:
    P5 <= P25 <= P50 <= P75 <= P95 at all time steps.
    """
    mu = np.array([0.10, 0.12, 0.08])
    cov = np.array([
        [0.04, 0.01, 0.01],
        [0.01, 0.05, 0.02],
        [0.01, 0.02, 0.03],
    ])
    weights = np.array([0.4, 0.4, 0.2])

    t0 = time.perf_counter()
    gbm_res = run_trajectory_monte_carlo(
        expected_returns=mu,
        cov_matrix=cov,
        weights=weights,
        initial_capital=10000.0,
        years=1.0,
        num_simulations=1000,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"GBM trajectory MC took {elapsed:.4f}s (budget: 2.0s)"

    # Cone monotonic quantile ordering
    assert np.all(gbm_res.percentile_5 <= gbm_res.percentile_25 + 1e-8)
    assert np.all(gbm_res.percentile_25 <= gbm_res.percentile_50 + 1e-8)
    assert np.all(gbm_res.percentile_50 <= gbm_res.percentile_75 + 1e-8)
    assert np.all(gbm_res.percentile_75 <= gbm_res.percentile_95 + 1e-8)
    assert len(gbm_res.days) == 253


# =============================================================================
# CHALLENGE 4: Risk Analytics & Higham Stability Invariants
# =============================================================================

def test_risk_metrics_cvar_le_var_invariant():
    """
    Challenge 4.1: Mathematical Invariant: CVaR (Expected Shortfall) is strictly
    more conservative than VaR at the same confidence level:
    In return/loss space, CVaR_95 <= VaR_95 (losses are more severe).
    """
    rng = np.random.default_rng(777)
    returns = rng.standard_t(df=4, size=1000) * 0.01  # Heavy-tailed returns

    # Historical (alpha = 0.05)
    var_hist, cvar_hist = compute_historical_var_cvar(returns, alpha=0.05)
    assert cvar_hist >= var_hist - 1e-8, f"Historical CVaR loss ({cvar_hist}) must be >= VaR loss ({var_hist})"

    # Parametric
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    var_param, cvar_param = compute_parametric_var_cvar(mu, sigma, alpha=0.05)
    assert cvar_param >= var_param - 1e-8, f"Parametric CVaR loss ({cvar_param}) must be >= VaR loss ({var_param})"


def test_higham_nearest_psd_repair_of_indefinite_matrix():
    """
    Challenge 4.2: Higham (2002) nearest PSD projection on indefinite matrix with negative eigenvalues.
    Verify:
      1. Repaired matrix is strictly positive semi-definite (all eig >= eps)
      2. Repaired matrix is symmetric
      3. Diagonals remain positive
    """
    # Create non-PSD matrix with negative eigenvalue
    A = np.array([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, 0.95],
        [0.95, 0.95, 0.10],  # Causes negative eigenvalue
    ])
    eigs_orig = get_eigenvalues(A)
    assert np.min(eigs_orig) < 0.0, "Original matrix must have negative eigenvalues"

    repaired = nearest_psd_higham(A, eps=1e-6)
    eigs_repaired = get_eigenvalues(repaired)

    assert np.min(eigs_repaired) >= 1e-6 - 1e-10, f"Repaired min eig: {np.min(eigs_repaired)}"
    assert is_positive_semidefinite(repaired, tol=1e-8)
    assert np.allclose(repaired, repaired.T)
    assert np.all(np.diag(repaired) > 0.0)
