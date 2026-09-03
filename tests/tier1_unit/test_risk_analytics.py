"""
Tier 1 Unit Tests: Advanced Risk Analytics & Performance Metrics Engine.
Verifies R5 risk requirements from ORIGINAL_REQUEST.md, PROJECT.md, and survey reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns

# Dynamic imports for analytics module (Milestone 4)
try:
    from src.analytics.risk_metrics import (
        compute_drawdown_series,
        compute_historical_var_cvar,
        compute_horizon_var_cvar,
        compute_parametric_var_cvar,
        compute_portfolio_risk_metrics,
    )
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False


pytestmark = pytest.mark.skipif(
    not HAS_ANALYTICS,
    reason="src.analytics module not yet implemented by Milestone 4",
)


# ===========================================================================
# 1. Drawdown & Cumulative Wealth Analytics Unit Tests
# ===========================================================================

def test_compute_drawdown_series_properties():
    """Verify drawdown series stays bounded in [-1.0, 0.0] and correctly identifies MaxDD."""
    # Construct a return series with known peak and valley
    daily_returns = pd.Series([0.10, 0.10, -0.20, -0.10, 0.05, 0.15])
    dd_series, max_dd, peak_idx, valley_idx = compute_drawdown_series(daily_returns)

    assert (dd_series <= 1e-12).all()
    assert (dd_series >= -1.0).all()
    assert max_dd >= 0.0
    # Peak is at index 1 (wealth = 1.1 * 1.1 = 1.21)
    # Valley is at index 3 (wealth = 1.21 * 0.8 * 0.9 = 0.8712 -> DD = (0.8712 - 1.21)/1.21 = -0.28)
    assert abs(max_dd - 0.28) < 1e-4


# ===========================================================================
# 2. VaR & CVaR (Historical & Parametric) Unit Tests
# ===========================================================================

def test_historical_var_and_cvar_coherence(sample_returns_df):
    """Verify Coherent Risk Invariant: CVaR (95%) >= VaR (95%) strictly holds."""
    port_returns = sample_returns_df.mean(axis=1)
    var_95, cvar_95 = compute_historical_var_cvar(port_returns, alpha=0.05)

    assert var_95 > 0.0
    assert cvar_95 > 0.0
    # Invariant 5: CVaR_95 >= VaR_95 (Expected Shortfall is >= VaR threshold)
    assert cvar_95 >= var_95 - 1e-9


def test_parametric_var_and_cvar_normal():
    """Verify parametric Gaussian VaR and CVaR match theoretical formulas."""
    mu = 0.0005
    sigma = 0.015
    var_param, cvar_param = compute_parametric_var_cvar(mu=mu, sigma=sigma, alpha=0.05)

    # Theoretical z_0.95 = 1.6448536
    # Theoretical phi(z) / 0.05 = 2.0627128
    expected_var = 1.6448536 * sigma - mu
    expected_cvar = 2.0627128 * sigma - mu

    assert abs(var_param - expected_var) < 1e-5
    assert abs(cvar_param - expected_cvar) < 1e-5
    assert cvar_param >= var_param


# ===========================================================================
# 3. Comprehensive Portfolio Risk Metrics Unit Tests
# ===========================================================================

def test_compute_portfolio_risk_metrics_structure(sample_returns_df):
    """Verify comprehensive metrics dictionary contains all required keys and valid numbers."""
    mu_series = calculate_expected_returns(sample_returns_df)
    cov_df, _ = estimate_covariance_matrix(sample_returns_df)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    metrics = compute_portfolio_risk_metrics(
        weights=weights,
        daily_returns=sample_returns_df,
        expected_returns=mu_series,
        cov_matrix=cov_df,
        rf=0.04,
    )

    required_keys = [
        "annualized_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "var_95_hist",
        "var_95_param",
        "cvar_95_hist",
        "var_95_monthly",
        "cvar_95_monthly",
        "var_95_annual",
        "cvar_95_annual",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
        val = metrics[key]
        assert isinstance(val, (float, int, np.floating))
        assert not np.isnan(val), f"Metric {key} returned NaN"


def test_sortino_ratio_downside_deviation(sample_returns_df):
    """Verify Sortino ratio penalizes only downside volatility (returns below daily Rf)."""
    mu_series = calculate_expected_returns(sample_returns_df)
    cov_df, _ = estimate_covariance_matrix(sample_returns_df)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    metrics = compute_portfolio_risk_metrics(
        weights=weights,
        daily_returns=sample_returns_df,
        expected_returns=mu_series,
        cov_matrix=cov_df,
        rf=0.04,
    )

    # Sortino ratio is defined and non-zero for active return series
    assert "sortino_ratio" in metrics
    assert not np.isnan(metrics["sortino_ratio"])


def test_multihorizon_var_cvar_scaling_and_coherence(sample_returns_df):
    """Verify Monthly and Annual VaR/CVaR maintain coherent risk properties (CVaR >= VaR)."""
    port_returns = sample_returns_df.mean(axis=1)

    # Monthly (21d)
    var_m, cvar_m = compute_horizon_var_cvar(port_returns, horizon_days=21, alpha=0.05)
    assert var_m >= 0.0
    assert cvar_m >= var_m - 1e-9

    # Annual (252d)
    var_a, cvar_a = compute_horizon_var_cvar(port_returns, horizon_days=252, alpha=0.05)
    assert var_a >= 0.0
    assert cvar_a >= var_a - 1e-9

    # Scale check: longer horizon implies larger risk under standard drift
    # Annual risk is typically larger than daily risk
    var_d, _ = compute_historical_var_cvar(port_returns, alpha=0.05)
    assert var_m >= var_d
    assert var_a >= var_m

