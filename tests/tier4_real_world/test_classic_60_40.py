"""
Tier 4 Real-World Workflows: Classic 60/40 Equity-Bond Portfolio (SPY / TLT).
Validates end-to-end multi-asset portfolio workflow on institutional benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns


def test_classic_60_40_full_workflow(classic_60_40_prices):
    """Verify classic 60/40 pipeline: data sanitization -> risk estimation -> allocation."""
    clean_p, daily_r = clean_and_align_prices(classic_60_40_prices)
    assert set(clean_p.columns) == {"SPY", "TLT"}

    mu_series = calculate_expected_returns(daily_r, method="arithmetic")
    cov_df, _ = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")

    # 60/40 weights
    w_6040 = np.array([0.60, 0.40])
    var_6040 = float(w_6040.T @ cov_df.values @ w_6040)
    vol_6040 = np.sqrt(var_6040)

    # Volatility of 60/40 should be lower than 100% equity (SPY) due to diversification
    vol_spy = np.sqrt(cov_df.loc["SPY", "SPY"])
    assert vol_6040 < vol_spy, "60/40 portfolio volatility must be lower than pure equity"

    try:
        from src.analytics.risk_metrics import compute_portfolio_risk_metrics
    except ImportError:
        pytest.skip("src.analytics module not yet implemented")

    metrics = compute_portfolio_risk_metrics(
        weights=w_6040,
        daily_returns=daily_r,
        expected_returns=mu_series,
        cov_matrix=cov_df,
        rf=0.03,
    )
    assert metrics["sharpe_ratio"] > 0.0
    assert metrics["max_drawdown"] < 0.50
