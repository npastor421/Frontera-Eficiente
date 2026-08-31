"""
Tier 1 Unit Tests: Data Loader, File Parser, Calendar Harmonization & Cache Module.
Verifies R1 requirements from ORIGINAL_REQUEST.md and PROJECT.md.
"""

from __future__ import annotations

import datetime
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.cache import (
    clear_data_cache,
    get_cached_asset_data,
    get_cached_raw_prices,
)
from src.data.cleaner import (
    align_to_calendar,
    calculate_daily_returns,
    clean_and_align_prices,
    normalize_datetime_index,
    trim_common_inception,
    validate_price_data,
)
from src.data.loader import (
    _detect_and_clean_numeric_col,
    _detect_csv_delimiter,
    fetch_asset_data,
    parse_manual_data,
    validate_tickers,
)


# ===========================================================================
# 1. Ticker Validation Unit Tests
# ===========================================================================

def test_validate_tickers_string_input():
    """Verify comma-separated ticker strings are parsed, trimmed, and deduplicated."""
    res = validate_tickers("AAPL, MSFT, AAPL, GOOGL ")
    assert res == ["AAPL", "MSFT", "GOOGL"]


def test_validate_tickers_sequence_input():
    """Verify list and array inputs preserve order and strip whitespace."""
    res = validate_tickers([" SPY ", "TLT", "SPY", "GLD "])
    assert res == ["SPY", "TLT", "GLD"]


def test_validate_tickers_empty_raises_error():
    """Verify empty string or empty sequence raises ValueError."""
    with pytest.raises(ValueError, match="Ticker list cannot be empty"):
        validate_tickers("")
    with pytest.raises(ValueError, match="Ticker list cannot be empty"):
        validate_tickers([])


def test_validate_tickers_invalid_type():
    """Verify non-sequence type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported ticker format"):
        validate_tickers(12345)  # type: ignore


# ===========================================================================
# 2. yfinance Fetching & Multi-Index Handling Unit Tests
# ===========================================================================

def test_fetch_asset_data_multi_ticker_mocked(mock_yfinance_download):
    """Verify multi-ticker yfinance MultiIndex output is cleanly sliced and flattened."""
    tickers = ["AAPL", "MSFT", "NVDA"]
    mock_df = mock_yfinance_download(tickers, start_date="2023-01-01", periods=50)

    with patch("yfinance.download", return_value=mock_df) as mock_yf:
        result = fetch_asset_data(tickers, start_date="2023-01-01", end_date="2023-03-15")
        mock_yf.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == tickers
        assert len(result) == 50
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is None
        assert not result.isna().any().any()


def test_fetch_asset_data_single_ticker_mocked(mock_yfinance_download):
    """Verify single ticker download returns a 1-column DataFrame with ticker name."""
    mock_df = mock_yfinance_download(["SPY"], start_date="2023-01-01", periods=30)

    with patch("yfinance.download", return_value=mock_df):
        result = fetch_asset_data(["SPY"], start_date="2023-01-01", end_date="2023-02-15")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["SPY"]
        assert len(result) == 30


def test_fetch_asset_data_empty_response_raises():
    """Verify empty yfinance response raises a descriptive ValueError."""
    empty_df = pd.DataFrame()
    with patch("yfinance.download", return_value=empty_df):
        with pytest.raises(ValueError, match="No price data found"):
            fetch_asset_data(["INVALID_TICKER_XYZ"], "2023-01-01", "2023-01-10")


# ===========================================================================
# 3. Manual File Parsing Unit Tests (CSV, Excel, Formats)
# ===========================================================================

def test_parse_manual_data_wide_prices_csv(wide_prices_csv_text):
    """Verify standard Wide Prices CSV parsing with auto date detection."""
    df = parse_manual_data(wide_prices_csv_text)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"AAPL", "MSFT", "GOOGL"}
    assert len(df) == 5
    assert df.loc["2023-01-03", "AAPL"] == 125.07
    assert df.dtypes["AAPL"] == np.float64


def test_parse_manual_data_european_comma_decimal(european_comma_decimal_csv_text):
    """Verify semicolon delimiter with Latin/European comma decimals is cleanly converted to float."""
    df = parse_manual_data(european_comma_decimal_csv_text)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"AAPL", "MSFT", "GOOGL"}
    assert df.loc["2023-01-03", "AAPL"] == 125.07
    assert df.loc["2023-01-04", "MSFT"] == 229.10


def test_parse_manual_data_long_tidy_format(long_tidy_csv_text):
    """Verify Long / Tidy format (Date, Ticker, Price) is pivoted into wide matrix."""
    df = parse_manual_data(long_tidy_csv_text)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"AAPL", "MSFT"}
    assert len(df) == 3
    assert df.loc["2023-01-03", "AAPL"] == 125.07
    assert df.loc["2023-01-03", "MSFT"] == 239.58


def test_parse_manual_data_excel_bytes(tmp_path, wide_excel_bytes):
    """Verify in-memory Excel workbook (.xlsx) bytes parsing via file path."""
    excel_file = tmp_path / "sample_data.xlsx"
    excel_file.write_bytes(wide_excel_bytes)
    df = parse_manual_data(excel_file)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["AAPL", "MSFT"]
    assert len(df) == 10


def test_parse_manual_data_invalid_text_raises():
    """Verify unparseable text raises ValueError."""
    with pytest.raises(ValueError):
        parse_manual_data("not_a_valid_file_and_no_dates_here")


# ===========================================================================
# 4. Data Cleaner & Calendar Harmonization Unit Tests
# ===========================================================================

def test_normalize_datetime_index():
    """Verify DatetimeIndex timezone stripping, normalization, and deduplication."""
    dates = pd.date_range("2023-01-01 09:30:00", periods=5, freq="D", tz="America/New_York")
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, index=dates)
    norm = normalize_datetime_index(df)

    assert norm.index.tz is None
    assert norm.index[0] == pd.Timestamp("2023-01-01 00:00:00")
    assert norm.index.name == "Date"


def test_align_to_calendar_ffill():
    """Verify business day reindexing fills gaps (e.g. weekends/holidays)."""
    dates = [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-05")]  # Missing Jan 4
    df = pd.DataFrame({"Asset": [100.0, 102.0]}, index=dates)
    aligned = align_to_calendar(df, freq="B", method="ffill")

    assert len(aligned) == 3
    assert pd.Timestamp("2023-01-04") in aligned.index
    assert aligned.loc["2023-01-04", "Asset"] == 100.0  # Forward-filled from Jan 3


def test_trim_common_inception():
    """Verify leading NaNs are trimmed to common starting date."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Asset1": [10.0, 11.0, 12.0, 13.0, 14.0],
        "Asset2": [np.nan, np.nan, 100.0, 105.0, 110.0],
    }, index=dates)
    trimmed = trim_common_inception(df)

    assert len(trimmed) == 3
    assert trimmed.index[0] == pd.Timestamp("2023-01-03")
    assert not trimmed.isna().any().any()


def test_validate_price_data_flat_series_raises():
    """Verify flat price series (zero return variance) raises ValueError."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    df = pd.DataFrame({"STALE": [100.0] * 10}, index=dates)
    with pytest.raises(ValueError, match="near-zero return variance"):
        validate_price_data(df)


def test_calculate_daily_returns_simple_and_log():
    """Verify simple and log daily return calculations."""
    dates = pd.date_range("2023-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)

    simple_ret = calculate_daily_returns(prices, method="simple")
    assert len(simple_ret) == 2
    np.testing.assert_allclose(simple_ret["A"].values, [0.10, 0.10], rtol=1e-5)

    log_ret = calculate_daily_returns(prices, method="log")
    assert len(log_ret) == 2
    np.testing.assert_allclose(log_ret["A"].values, [np.log(1.1), np.log(1.1)], rtol=1e-5)


def test_clean_and_align_prices_end_to_end(sample_prices_df):
    """Verify full sanitization pipeline returns clean prices and daily returns."""
    clean_p, daily_r = clean_and_align_prices(sample_prices_df, freq="B")
    assert isinstance(clean_p, pd.DataFrame)
    assert isinstance(daily_r, pd.DataFrame)
    assert len(daily_r) == len(clean_p) - 1
    assert not clean_p.isna().any().any()
    assert not daily_r.isna().any().any()


# ===========================================================================
# 5. Data Cache Unit Tests
# ===========================================================================

def test_cache_defensive_copies(sample_prices_df):
    """Verify cached return objects return defensive copies preventing state corruption."""
    with patch("src.data.cache.fetch_asset_data", return_value=sample_prices_df.copy()):
        df1 = get_cached_raw_prices(["AAPL", "MSFT"], "2023-01-01", "2023-06-01")
        df1.iloc[0, 0] = 999999.0  # Mutate df1

        df2 = get_cached_raw_prices(["AAPL", "MSFT"], "2023-01-01", "2023-06-01")
        # df2 should not be affected by mutation of df1
        assert df2.iloc[0, 0] != 999999.0

        clear_data_cache()
