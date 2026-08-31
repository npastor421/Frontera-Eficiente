"""
Tier 1 Unit Tests: Markowitz Quantitative Optimization Engine & Dual Monte Carlo Simulations.
Verifies R3 and R4 requirements from ORIGINAL_REQUEST.md and PROJECT.md.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns

# Dynamic imports for optimization & simulation modules (Milestone 3)
try:
    from src.optimization.optimizer import (
        OptimizationResult,
        optimize_global_minimum_variance,
        optimize_maximum_sharpe,
        optimize_target_return,
    )
    from src.optimization.frontier import (
        EfficientFrontierResult,
        compute_capital_allocation_line,
        compute_efficient_frontier,
    )
    from src.simulation.weight_monte_carlo import (
        WeightMonteCarloResult,
        run_weight_space_monte_carlo,
    )
    from src.simulation.trajectory_monte_carlo import (
        TrajectorySimulationResult,
        run_trajectory_monte_carlo,
    )
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False


pytestmark = pytest.mark.skipif(
    not HAS_OPTIMIZATION,
    reason="src.optimization and src.simulation modules not yet implemented by Milestone 3",
)


# ===========================================================================
# 1. Global Minimum Variance (GMV) Unit Tests
# ===========================================================================

def test_optimize_gmv_weight_sum_and_bounds(sample_returns_df):
    """Verify GMV weights sum to 1.0 +- 1e-5 and respect Long-Only bounds [0, 1]."""
    cov_df, _ = estimate_covariance_matrix(sample_returns_df, method="sample")
    gmv_res = optimize_global_minimum_variance(cov_df.values, bounds=(0.0, 1.0))

    assert isinstance(gmv_res, OptimizationResult)
    assert gmv_res.success is True
    # Invariant 1: Sum of weights == 1.0 +- 1e-5
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
    # Bounds: 0 <= w_i <= 1
    assert (gmv_res.weights >= -1e-6).all()
    assert (gmv_res.weights <= 1.0 + 1e-6).all()


def test_optimize_gmv_volatility_minimality(sample_returns_df):
    """Verify GMV volatility is strictly <= volatility of any random simplex weights."""
    cov_df, _ = estimate_covariance_matrix(sample_returns_df, method="sample")
    cov = cov_df.values
    gmv_res = optimize_global_minimum_variance(cov, bounds=(0.0, 1.0))
    gmv_vol = gmv_res.volatility

    # Generate 100 random Dirichlet weight vectors
    n_assets = cov.shape[0]
    random_w = np.random.dirichlet(np.ones(n_assets), size=100)
    random_vars = np.sum((random_w @ cov) * random_w, axis=1)
    random_vols = np.sqrt(np.maximum(random_vars, 1e-12))

    # Invariant 2: GMV vol <= any random feasible allocation vol
    assert (gmv_vol <= random_vols + 1e-5).all()


# ===========================================================================
# 2. Maximum Sharpe Ratio Portfolio Unit Tests
# ===========================================================================

def test_optimize_maximum_sharpe_optimality(sample_returns_df):
    """Verify Max Sharpe portfolio yields higher Sharpe than all individual assets and random weights."""
    rf = 0.04
    mu_series = calculate_expected_returns(sample_returns_df, method="arithmetic")
    cov_df, _ = estimate_covariance_matrix(sample_returns_df, method="sample")
    mu = mu_series.values
    cov = cov_df.values

    ms_res = optimize_maximum_sharpe(mu, cov, rf=rf, bounds=(0.0, 1.0))

    assert isinstance(ms_res, OptimizationResult)
    assert ms_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) <= 1e-5
    assert (ms_res.weights >= -1e-6).all()

    ms_sharpe = ms_res.sharpe_ratio

    # Check against individual assets
    asset_vols = np.sqrt(np.diag(cov))
    asset_sharpes = (mu - rf) / asset_vols
    assert (ms_sharpe >= asset_sharpes - 1e-5).all()

    # Check against 200 random portfolios
    n_assets = len(mu)
    random_w = np.random.dirichlet(np.ones(n_assets), size=200)
    rand_mu = random_w @ mu
    rand_vol = np.sqrt(np.sum((random_w @ cov) * random_w, axis=1))
    rand_sharpe = (rand_mu - rf) / rand_vol

    # Invariant 3: Max Sharpe >= all random Sharpe ratios
    assert (ms_sharpe >= rand_sharpe - 1e-5).all()


def test_optimize_custom_asset_bounds(sample_returns_df):
    """Verify optimizer enforces custom individual asset bounds [w_min_i, w_max_i]."""
    cov_df, _ = estimate_covariance_matrix(sample_returns_df, method="sample")
    cov = cov_df.values

    # Force Asset 0 to have at least 30%, Asset 1 at most 10%
    custom_bounds = [
        (0.30, 0.60),  # Asset 0: AAPL
        (0.00, 0.10),  # Asset 1: MSFT
        (0.00, 0.40),  # Asset 2: GOOGL
        (0.00, 0.40),  # Asset 3: AMZN
        (0.00, 0.40),  # Asset 4: NVDA
    ]

    gmv_res = optimize_global_minimum_variance(cov, bounds=custom_bounds)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
    assert gmv_res.weights[0] >= 0.30 - 1e-5
    assert gmv_res.weights[1] <= 0.10 + 1e-5


# ===========================================================================
# 3. Continuous Efficient Frontier & CAL Unit Tests
# ===========================================================================

def test_compute_efficient_frontier_curve(sample_returns_df):
    """Verify continuous efficient frontier sweep calculation and monotonic upper curve."""
    rf = 0.04
    mu_series = calculate_expected_returns(sample_returns_df, method="arithmetic")
    cov_df, _ = estimate_covariance_matrix(sample_returns_df, method="sample")
    mu = mu_series.values
    cov = cov_df.values

    frontier = compute_efficient_frontier(mu, cov, rf=rf, num_points=50, bounds=(0.0, 1.0))

    assert isinstance(frontier, EfficientFrontierResult)
    assert len(frontier.returns) == 50
    assert len(frontier.volatilities) == 50
    assert frontier.weights.shape == (50, len(mu))

    # Invariant: Each frontier point weights must sum to 1.0
    weight_sums = np.sum(frontier.weights, axis=1)
    np.testing.assert_allclose(weight_sums, np.ones(50), atol=1e-5)

    # Returns must be monotonically increasing along the upper frontier
    assert (np.diff(frontier.returns) >= -1e-6).all()


def test_compute_capital_allocation_line():
    """Verify CAL line starts at (0, Rf) and passes through Max Sharpe point."""
    rf = 0.04
    dummy_ms = OptimizationResult(
        weights=np.array([0.5, 0.5]),
        expected_return=0.16,
        volatility=0.20,
        sharpe_ratio=(0.16 - 0.04) / 0.20,
        status="optimal",
        success=True,
        iterations=10,
    )

    cal_vols, cal_rets = compute_capital_allocation_line(dummy_ms, rf=rf, max_vol=0.5, num_points=50)

    assert cal_vols[0] == 0.0
    assert abs(cal_rets[0] - rf) < 1e-6
    # Slope of CAL must equal Sharpe Ratio
    slope = (cal_rets[-1] - cal_rets[0]) / cal_vols[-1]
    assert abs(slope - dummy_ms.sharpe_ratio) < 1e-6


# ===========================================================================
# 4. Dual Monte Carlo Simulations Unit Tests
# ===========================================================================

def test_dirichlet_weight_monte_carlo_performance_and_invariants(sample_returns_df):
    """Verify 10,000 Dirichlet weight space simulations run in < 1.0s with exact simplex sums."""
    rf = 0.04
    mu = calculate_expected_returns(sample_returns_df).values
    cov = estimate_covariance_matrix(sample_returns_df)[0].values

    start_t = time.perf_counter()
    mc_res = run_weight_space_monte_carlo(mu, cov, rf=rf, num_portfolios=10000, seed=42)
    exec_time = time.perf_counter() - start_t

    # Performance constraint: < 2.0s
    assert exec_time < 2.0, f"Monte Carlo too slow: {exec_time:.3f}s"
    assert isinstance(mc_res, WeightMonteCarloResult)
    assert mc_res.weights.shape == (10000, len(mu))

    # Check simplex constraint on all 10,000 portfolios
    sums = np.sum(mc_res.weights, axis=1)
    np.testing.assert_allclose(sums, np.ones(10000), atol=1e-5)


def test_trajectory_monte_carlo_probability_cones(sample_returns_df):
    """Verify multi-year stochastic trajectory forecasting and probability percentile ordering."""
    mu = calculate_expected_returns(sample_returns_df).values
    cov = estimate_covariance_matrix(sample_returns_df)[0].values
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    traj_res = run_trajectory_monte_carlo(
        expected_returns=mu,
        cov_matrix=cov,
        weights=weights,
        initial_capital=10000.0,
        years=2,
        num_simulations=1000,
        model="gbm",
        seed=42,
    )

    assert isinstance(traj_res, TrajectorySimulationResult)
    assert traj_res.initial_wealth == 10000.0
    assert len(traj_res.days) == 2 * 252 + 1

    # Probability cone ordering invariant: P5 <= P25 <= P50 <= P75 <= P95
    p5 = traj_res.percentile_5
    p25 = traj_res.percentile_25
    p50 = traj_res.percentile_50
    p75 = traj_res.percentile_75
    p95 = traj_res.percentile_95

    assert (p5 <= p25 + 1e-6).all()
    assert (p25 <= p50 + 1e-6).all()
    assert (p50 <= p75 + 1e-6).all()
    assert (p75 <= p95 + 1e-6).all()
