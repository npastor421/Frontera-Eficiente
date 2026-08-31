"""
Tier 2 Boundary & Corner Cases: Single Asset Universe (N=1).
Verifies system stability, estimator behavior, and mathematical sanity when N=1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import (
    estimate_covariance_matrix,
    ledoit_wolf_constant_correlation,
    sample_covariance,
)
from src.models.returns import (
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_expected_returns,
)
from src.models.stability import ensure_positive_semidefinite, is_positive_semidefinite


def test_single_asset_clean_and_returns(single_asset_prices):
    """Verify single asset price series cleans and produces valid 1D return series."""
    clean_p, daily_r = clean_and_align_prices(single_asset_prices)
    assert clean_p.shape[1] == 1
    assert daily_r.shape[1] == 1
    assert not daily_r.isna().any().any()


def test_single_asset_expected_returns(single_asset_prices):
    """Verify return estimators on N=1 series."""
    _, daily_r = clean_and_align_prices(single_asset_prices)
    arith = annualized_arithmetic_returns(daily_r)
    geom = annualized_geometric_returns(daily_r)

    assert isinstance(arith, pd.Series)
    assert len(arith) == 1
    assert geom.iloc[0] <= arith.iloc[0] + 1e-6


def test_single_asset_covariance_matrix(single_asset_prices):
    """Verify 1x1 covariance matrix estimation and PSD property."""
    _, daily_r = clean_and_align_prices(single_asset_prices)
    cov_s = sample_covariance(daily_r)
    assert cov_s.shape == (1, 1)
    assert cov_s.iloc[0, 0] > 0.0
    assert is_positive_semidefinite(cov_s)

    cov_lw, delta = ledoit_wolf_constant_correlation(daily_r)
    assert cov_lw.shape == (1, 1)
    assert is_positive_semidefinite(cov_lw)


def test_single_asset_optimizer_and_metrics(single_asset_prices):
    """Verify optimization with N=1 trivially yields w=[1.0]."""
    try:
        from src.optimization.optimizer import optimize_global_minimum_variance
        from src.analytics.risk_metrics import compute_portfolio_risk_metrics
    except ImportError:
        pytest.skip("Optimization or Analytics modules not yet implemented")

    _, daily_r = clean_and_align_prices(single_asset_prices)
    cov_df, _ = estimate_covariance_matrix(daily_r)
    mu_series = calculate_expected_returns(daily_r)

    gmv_res = optimize_global_minimum_variance(cov_df.values)
    assert gmv_res.success is True
    np.testing.assert_allclose(gmv_res.weights, [1.0], atol=1e-5)

    metrics = compute_portfolio_risk_metrics(
        weights=np.array([1.0]),
        daily_returns=daily_r,
        expected_returns=mu_series,
        cov_matrix=cov_df,
        rf=0.04,
    )
    assert "annualized_volatility" in metrics
    assert abs(metrics["annualized_volatility"] - np.sqrt(cov_df.iloc[0, 0])) < 1e-4
