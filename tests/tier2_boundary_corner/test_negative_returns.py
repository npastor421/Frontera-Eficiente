"""
Tier 2 Boundary & Corner Cases: Negative & Bear Market Regimes (mu <= Rf).
Verifies return estimation, negative Sharpe handling, and optimizer behavior under distress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.covariance import estimate_covariance_matrix
from src.models.returns import (
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_expected_returns,
    ewma_returns,
)


def test_negative_return_estimators(negative_returns_df):
    """Verify return estimators compute negative expected returns cleanly."""
    arith = annualized_arithmetic_returns(negative_returns_df)
    geom = annualized_geometric_returns(negative_returns_df)
    ewma = ewma_returns(negative_returns_df)

    assert (arith.values < 0.0).all()
    assert (geom.values < 0.0).all()
    assert (ewma.values < 0.0).all()


def test_negative_sharpe_ratio_invariants(negative_returns_df):
    """Verify negative Sharpe ratio is calculated correctly without NaN."""
    rf = 0.04
    mu = calculate_expected_returns(negative_returns_df).values
    cov = estimate_covariance_matrix(negative_returns_df)[0].values

    asset_vols = np.sqrt(np.diag(cov))
    sharpes = (mu - rf) / asset_vols

    assert (sharpes < 0.0).all()
    assert not np.isnan(sharpes).any()


def test_gmv_optimization_in_bear_market(negative_returns_df):
    """Verify GMV portfolio minimizes variance regardless of negative returns."""
    try:
        from src.optimization.optimizer import optimize_global_minimum_variance
    except ImportError:
        pytest.skip("src.optimization module not yet implemented")

    cov_df, _ = estimate_covariance_matrix(negative_returns_df)
    gmv_res = optimize_global_minimum_variance(cov_df.values)

    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
    assert gmv_res.volatility > 0.0
