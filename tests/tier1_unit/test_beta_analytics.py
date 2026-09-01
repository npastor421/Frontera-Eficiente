"""
Unit Tests for Beta (β), Jensen's Alpha (α) and Benchmark Risk Analytics.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk_metrics import (
    calculate_beta,
    calculate_jensen_alpha,
    calculate_asset_betas,
    compute_portfolio_risk_metrics,
)


def test_calculate_beta_identical():
    """Beta of market with itself must be exactly 1.0."""
    rng = np.random.default_rng(42)
    m = pd.Series(rng.normal(0.001, 0.01, 100))
    beta = calculate_beta(m, m)
    assert abs(beta - 1.0) < 1e-6


def test_calculate_beta_scaled():
    """Beta of a 2x leveraged market series must be 2.0."""
    rng = np.random.default_rng(42)
    m = pd.Series(rng.normal(0.001, 0.01, 100))
    p = 2.0 * m
    beta = calculate_beta(p, m)
    assert abs(beta - 2.0) < 1e-6


def test_calculate_beta_cash_zero_variance():
    """Beta of cash with 0 variance must be 0.0."""
    rng = np.random.default_rng(42)
    m = pd.Series(rng.normal(0.001, 0.01, 100))
    cash = pd.Series(np.zeros(100))
    beta = calculate_beta(cash, m)
    assert abs(beta - 0.0) < 1e-6


def test_calculate_jensen_alpha():
    """Verify Jensen's alpha formula: Rp - (Rf + beta * (Rm - Rf))."""
    r_p = 0.15
    r_m = 0.10
    rf = 0.04
    beta = 1.2
    # Expected CAPM = 0.04 + 1.2 * (0.10 - 0.04) = 0.04 + 0.072 = 0.112
    # Alpha = 0.15 - 0.112 = 0.038
    alpha = calculate_jensen_alpha(portfolio_return=r_p, benchmark_return=r_m, beta=beta, rf=rf)
    assert abs(alpha - 0.038) < 1e-6


def test_calculate_asset_betas_and_container():
    """Verify compute_portfolio_risk_metrics populates beta and alpha fields."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    m_returns = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates, name="SPY")
    
    asset_a = m_returns * 1.5 + rng.normal(0, 0.002, 100)
    asset_b = m_returns * 0.5 + rng.normal(0, 0.002, 100)
    
    returns_df = pd.DataFrame({"AAPL": asset_a, "JNJ": asset_b}, index=dates)
    
    betas = calculate_asset_betas(returns_df, m_returns)
    assert abs(betas["AAPL"] - 1.5) < 0.2
    assert abs(betas["JNJ"] - 0.5) < 0.2
    
    metrics = compute_portfolio_risk_metrics(
        weights={"AAPL": 0.5, "JNJ": 0.5},
        daily_returns=returns_df,
        rf=0.04,
        benchmark_returns=m_returns,
        benchmark_ticker="SPY",
    )
    assert metrics.beta is not None
    assert abs(metrics.beta - 1.0) < 0.2
    assert metrics.alpha_jensen is not None
    assert metrics.r_squared is not None
    assert metrics.benchmark_ticker == "SPY"
