"""
Tier 2 Boundary & Corner Cases: Sparse Allocation & Zero Weights (w_i = 0.0).
Verifies that assigning 0.0% weight to a subset of assets does not break analytics, quadratic forms, or visualizers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns


def test_zero_weights_quadratic_form(sample_returns_df):
    """Verify portfolio variance with sparse weights w=[1, 0, 0, 0, 0] matches asset 0 variance."""
    cov_df, _ = estimate_covariance_matrix(sample_returns_df)
    cov = cov_df.values

    w_sparse = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    port_var = float(w_sparse.T @ cov @ w_sparse)
    asset_0_var = float(cov[0, 0])

    assert abs(port_var - asset_0_var) < 1e-10


def test_zero_weights_in_risk_metrics(sample_returns_df):
    """Verify compute_portfolio_risk_metrics handles sparse zero-weight vectors."""
    try:
        from src.analytics.risk_metrics import compute_portfolio_risk_metrics
    except ImportError:
        pytest.skip("src.analytics module not yet implemented")

    mu_series = calculate_expected_returns(sample_returns_df)
    cov_df, _ = estimate_covariance_matrix(sample_returns_df)

    # 2 active assets, 3 zero-weighted assets
    w_sparse = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

    metrics = compute_portfolio_risk_metrics(
        weights=w_sparse,
        daily_returns=sample_returns_df,
        expected_returns=mu_series,
        cov_matrix=cov_df,
        rf=0.04,
    )

    assert not np.isnan(metrics["annualized_volatility"])
    assert not np.isnan(metrics["sharpe_ratio"])
    assert metrics["annualized_volatility"] > 0.0


def test_optimizer_enforcing_zero_weight_bounds(sample_returns_df):
    """Verify setting upper bound to 0.0 forces asset allocation to exactly 0.0."""
    try:
        from src.optimization.optimizer import optimize_global_minimum_variance
    except ImportError:
        pytest.skip("src.optimization module not yet implemented")

    cov_df, _ = estimate_covariance_matrix(sample_returns_df)
    custom_bounds = [
        (0.0, 0.0),  # Asset 0 forced to 0
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]

    gmv_res = optimize_global_minimum_variance(cov_df.values, bounds=custom_bounds)
    assert gmv_res.success is True
    assert abs(gmv_res.weights[0]) < 1e-6
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
