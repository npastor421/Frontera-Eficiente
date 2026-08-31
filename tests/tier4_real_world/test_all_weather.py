"""
Tier 4 Real-World Workflows: Ray Dalio All-Weather Multi-Asset Risk Parity Universe.
Validates multi-asset class interaction across Equities (SPY), Long Bonds (TLT), Intermediate Bonds (IEF), Gold (GLD), and Commodities (DBC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns
from src.models.stability import is_positive_semidefinite


def test_all_weather_covariance_and_allocation(all_weather_prices):
    """Verify All-Weather asset class diversification and risk reduction."""
    clean_p, daily_r = clean_and_align_prices(all_weather_prices)
    assert len(clean_p.columns) == 5

    cov_df, meta = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")
    assert is_positive_semidefinite(cov_df)

    # Dalio All-Weather weights: SPY=30%, TLT=40%, IEF=15%, GLD=7.5%, DBC=7.5%
    w_aw = np.array([0.30, 0.40, 0.15, 0.075, 0.075])
    assert abs(np.sum(w_aw) - 1.0) < 1e-6

    var_aw = float(w_aw.T @ cov_df.values @ w_aw)
    vol_aw = np.sqrt(var_aw)
    vol_spy = np.sqrt(cov_df.loc["SPY", "SPY"])

    # All-Weather must achieve lower annualized volatility than 100% SPY
    assert vol_aw < vol_spy

    try:
        from src.optimization.optimizer import optimize_maximum_sharpe
        from src.analytics.risk_metrics import compute_portfolio_risk_metrics
    except ImportError:
        pytest.skip("src.optimization or src.analytics not yet implemented")

    mu_series = calculate_expected_returns(daily_r)
    ms_res = optimize_maximum_sharpe(mu_series.values, cov_df.values, rf=0.03)
    assert ms_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) <= 1e-5
