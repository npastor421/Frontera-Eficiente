"""
Tier 4 Real-World Workflows: Hybrid Crypto (24/7/365) + TradFi (252-Day) Portfolios.
Validates asynchronous trading calendar alignment, forward fill holiday propagation, and mixed-volatility risk modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import align_to_calendar, clean_and_align_prices, trim_common_inception
from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns
from src.models.stability import is_positive_semidefinite


def test_crypto_tradfi_asynchronous_calendar_alignment(crypto_tradfi_raw_prices):
    """Verify merging 7-day crypto with 5-day TradFi aligns cleanly to business days without NaNs."""
    crypto_df, tradfi_df = crypto_tradfi_raw_prices

    # Merge on outer index
    merged_raw = pd.concat([crypto_df, tradfi_df], axis=1)
    assert merged_raw["SPY"].isna().any(), "TradFi should have NaNs on weekend rows before alignment"

    # Clean and align to Business Days ('B')
    clean_p, daily_r = clean_and_align_prices(merged_raw, freq="B")

    assert set(clean_p.columns) == {"BTC-USD", "ETH-USD", "SPY", "QQQ"}
    assert not clean_p.isna().any().any()
    assert not daily_r.isna().any().any()


def test_crypto_tradfi_mixed_volatility_modeling(crypto_tradfi_raw_prices):
    """Verify covariance estimation and Ledoit-Wolf shrinkage across disparate volatility regimes."""
    crypto_df, tradfi_df = crypto_tradfi_raw_prices
    merged_raw = pd.concat([crypto_df, tradfi_df], axis=1)
    clean_p, daily_r = clean_and_align_prices(merged_raw, freq="B")

    cov_df, _ = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")
    assert is_positive_semidefinite(cov_df)

    vol_btc = np.sqrt(cov_df.loc["BTC-USD", "BTC-USD"])
    vol_spy = np.sqrt(cov_df.loc["SPY", "SPY"])

    # Crypto volatility should be significantly higher than US equity index
    assert vol_btc > vol_spy * 1.5

    try:
        from src.optimization.optimizer import optimize_maximum_sharpe
    except ImportError:
        pytest.skip("src.optimization not yet implemented")

    mu_series = calculate_expected_returns(daily_r)
    ms_res = optimize_maximum_sharpe(mu_series.values, cov_df.values, rf=0.04)
    assert ms_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) <= 1e-5
