"""
Unit Tests for Data Caching (src/data/cache.py).
"""

import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from src.data.cache import (
    clear_data_cache,
    get_cached_asset_data,
    get_cached_raw_prices,
)


@mock.patch("src.data.loader.yf.download")
def test_get_cached_raw_prices_and_defensive_copy(mock_yf):
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = pd.MultiIndex.from_tuples(
        [("Adj Close", "AAPL"), ("Adj Close", "MSFT")],
        names=["Price", "Ticker"],
    )
    data = np.array([
        [100.0, 200.0],
        [102.0, 204.0],
        [101.0, 202.0],
        [103.0, 206.0],
        [105.0, 210.0],
    ])
    mock_yf.return_value = pd.DataFrame(data, index=dates, columns=cols)

    df1 = get_cached_raw_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-08")
    assert df1.loc[pd.Timestamp("2024-01-01"), "AAPL"] == 100.0

    # Mutate df1
    df1.loc[pd.Timestamp("2024-01-01"), "AAPL"] = 9999.0

    # Fetch again - should not be mutated due to defensive copy
    df2 = get_cached_raw_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-08")
    assert df2.loc[pd.Timestamp("2024-01-01"), "AAPL"] == 100.0


@mock.patch("src.data.loader.yf.download")
def test_get_cached_asset_data_pipeline(mock_yf):
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    cols = pd.MultiIndex.from_tuples(
        [("Adj Close", "SPY"), ("Adj Close", "QQQ")],
        names=["Price", "Ticker"],
    )
    data = np.array([
        [400.0, 300.0],
        [404.0, 303.0],
        [402.0, 300.0],
        [406.0, 306.0],
        [408.0, 309.0],
        [410.0, 312.0],
    ])
    mock_yf.return_value = pd.DataFrame(data, index=dates, columns=cols)

    clean_prices, daily_returns = get_cached_asset_data(
        ("SPY", "QQQ"),
        "2024-01-01",
        "2024-01-10",
        freq="B",
    )

    assert isinstance(clean_prices, pd.DataFrame)
    assert isinstance(daily_returns, pd.DataFrame)
    assert len(clean_prices) == 6
    assert len(daily_returns) == 5
    assert not clean_prices.isna().any().any()
    assert not daily_returns.isna().any().any()


def test_clear_data_cache():
    # Should run cleanly without raising exception
    clear_data_cache()
