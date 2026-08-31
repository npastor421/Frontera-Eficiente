"""
Tier 2 Boundary & Corner Cases: Flash Crashes, Outliers & Tail Shocks.
Verifies numerical stability under extreme market dislocations (-90% crashes, +300% spikes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import calculate_daily_returns, clean_and_align_prices
from src.models.returns import (
    annualized_arithmetic_returns,
    annualized_geometric_returns,
)


def test_flash_crash_log_returns_clipping(extreme_outlier_prices):
    """Verify log return calculation handles near-total wipeout without raising invalid log domain."""
    log_ret = calculate_daily_returns(extreme_outlier_prices, method="log")
    assert isinstance(log_ret, pd.DataFrame)
    assert not log_ret.isna().any().any()
    assert not np.isinf(log_ret.values).any()


def test_extreme_cagr_calculation(extreme_outlier_prices):
    """Verify annualized CAGR computation clips extreme losses and returns valid numbers."""
    clean_p, daily_r = clean_and_align_prices(extreme_outlier_prices)
    geom_ret = annualized_geometric_returns(daily_r)

    assert isinstance(geom_ret, pd.Series)
    assert not geom_ret.isna().any()
    assert geom_ret["CRASH_ASSET"] < 0.0  # Crash asset has severe negative CAGR


def test_drawdown_on_flash_crash_asset(extreme_outlier_prices):
    """Verify drawdown accurately captures >= 75% peak-to-trough drop."""
    try:
        from src.analytics.risk_metrics import compute_drawdown_series
    except ImportError:
        pytest.skip("src.analytics module not yet implemented")

    clean_p, daily_r = clean_and_align_prices(extreme_outlier_prices)
    crash_returns = daily_r["CRASH_ASSET"]

    dd_series, max_dd, _, _ = compute_drawdown_series(crash_returns)
    assert max_dd >= 0.75, f"Max drawdown should be >= 0.75, got {max_dd}"
