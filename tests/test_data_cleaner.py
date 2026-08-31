"""
Unit and Edge Case Tests for Data Cleaner (src/data/cleaner.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    align_to_calendar,
    calculate_daily_returns,
    clean_and_align_prices,
    normalize_datetime_index,
    trim_common_inception,
    validate_price_data,
)


class TestNormalizeDatetimeIndex:
    def test_string_index_normalization(self):
        df = pd.DataFrame(
            {"A": [10.0, 11.0], "B": [20.0, 22.0]},
            index=["2024-01-02 15:30:00", "2024-01-01 09:00:00"],
        )
        res = normalize_datetime_index(df)
        assert isinstance(res.index, pd.DatetimeIndex)
        assert res.index.name == "Date"
        # Should be sorted chronologically
        assert list(res.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        assert res.iloc[0]["A"] == 11.0

    def test_timezone_removal(self):
        tz_idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="America/New_York")
        df = pd.DataFrame({"A": [1, 2, 3]}, index=tz_idx)
        res = normalize_datetime_index(df)
        assert res.index.tz is None
        assert res.index.name == "Date"

    def test_deduplication(self):
        idx = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        df = pd.DataFrame({"A": [100.0, 999.0, 105.0]}, index=idx)
        res = normalize_datetime_index(df)
        assert len(res) == 2
        assert res.iloc[0]["A"] == 100.0


class TestAlignToCalendar:
    def test_business_day_reindexing_and_ffill(self):
        # Create series with a gap (e.g. Wednesday missing)
        idx = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
        df = pd.DataFrame({"A": [10.0, 12.0, 14.0, 15.0]}, index=idx)
        aligned = align_to_calendar(df, freq="B", method="ffill")
        # 2024-01-03 (Wednesday) should be present and forward-filled to 12.0
        assert pd.Timestamp("2024-01-03") in aligned.index
        assert aligned.loc[pd.Timestamp("2024-01-03"), "A"] == 12.0
        assert len(aligned) == 5  # Mon through Fri


class TestTrimCommonInception:
    def test_common_inception_trimming(self):
        # Asset A starts on 2024-01-01, Asset B starts on 2024-01-03 (e.g. IPO)
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "AssetA": [100.0, 101.0, 102.0, 103.0, 104.0],
                "AssetB": [np.nan, np.nan, 50.0, 52.0, 53.0],
            },
            index=idx,
        )
        trimmed = trim_common_inception(df, drop_incomplete=True)
        assert len(trimmed) == 3
        assert trimmed.index.min() == pd.Timestamp("2024-01-03")
        assert not trimmed.isna().any().any()

    def test_no_overlap_raises_error(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "AssetA": [100.0, 101.0, np.nan],
                "AssetB": [np.nan, np.nan, 50.0],
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="No common overlapping date range"):
            trim_common_inception(df)


class TestValidatePriceData:
    def test_valid_dataframe_passes(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": np.linspace(100, 110, 10),
                "B": np.linspace(50, 45, 10),
            },
            index=idx,
        )
        # Should not raise
        validate_price_data(df, min_obs=5)

    def test_negative_or_zero_price_raises(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"A": [10.0, 12.0, 0.0, 14.0, 15.0]}, index=idx)
        with pytest.raises(ValueError, match="strictly positive"):
            validate_price_data(df, min_obs=5)

    def test_flat_series_raises_zero_variance_error(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": [100.0] * 10,  # Constant price
                "B": np.linspace(50, 55, 10),
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="near-zero return variance"):
            validate_price_data(df, min_obs=5)

    def test_insufficient_observations_raises(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=idx)
        with pytest.raises(ValueError, match="Insufficient historical observations"):
            validate_price_data(df, min_obs=5)


class TestCalculateDailyReturns:
    def test_simple_returns_calculation(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 110.0, 121.0, 121.0],
                "MSFT": [200.0, 180.0, 180.0, 198.0],
            },
            index=idx,
        )
        returns = calculate_daily_returns(prices, method="simple")
        assert len(returns) == 3
        # Row 1: 110/100 - 1 = +0.10, 180/200 - 1 = -0.10
        assert pytest.approx(returns.iloc[0]["AAPL"], rel=1e-5) == 0.10
        assert pytest.approx(returns.iloc[0]["MSFT"], rel=1e-5) == -0.10
        # Row 2: 121/110 - 1 = +0.10, 180/180 - 1 = 0.0
        assert pytest.approx(returns.iloc[1]["AAPL"], rel=1e-5) == 0.10
        assert pytest.approx(returns.iloc[1]["MSFT"], rel=1e-5) == 0.0

    def test_log_returns_calculation(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"AAPL": [100.0, 110.0, 121.0]}, index=idx)
        returns = calculate_daily_returns(prices, method="log")
        assert len(returns) == 2
        # log(110/100) = ln(1.1)
        assert pytest.approx(returns.iloc[0]["AAPL"], rel=1e-5) == np.log(1.1)


class TestCleanAndAlignPricesPipeline:
    def test_end_to_end_cleaning_pipeline(self):
        # Create unaligned data with missing weekend/holidays
        dates = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08"]
        raw_df = pd.DataFrame(
            {
                "Asset1": [100.0, 102.0, 104.0, 103.0, 105.0],
                "Asset2": [50.0, 51.0, 52.0, 52.5, 53.0],
            },
            index=dates,
        )
        clean_prices, daily_returns = clean_and_align_prices(raw_df, freq="B", min_obs=4)

        assert len(clean_prices) == 6  # 2024-01-01 to 2024-01-08 on B freq has 6 days
        assert len(daily_returns) == 5
        assert not clean_prices.isna().any().any()
        assert not daily_returns.isna().any().any()
        assert list(clean_prices.columns) == ["Asset1", "Asset2"]
        assert list(daily_returns.columns) == ["Asset1", "Asset2"]

    def test_leap_year_alignment(self):
        # 2024 is a leap year; Feb 29 2024 is a Thursday
        idx = [pd.Timestamp("2024-02-28"), pd.Timestamp("2024-03-01")]
        df = pd.DataFrame({"A": [100.0, 102.0]}, index=idx)
        aligned = align_to_calendar(df, freq="B", method="ffill")
        assert pd.Timestamp("2024-02-29") in aligned.index
        assert aligned.loc[pd.Timestamp("2024-02-29"), "A"] == 100.0

    def test_return_mathematical_consistency(self):
        # For small returns, log return and simple return are asymptotically close
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"A": [100.0, 100.1, 100.2, 100.15, 100.3, 100.25, 100.4, 100.35, 100.5, 100.6]}, index=idx)
        simple_r = calculate_daily_returns(prices, method="simple")
        log_r = calculate_daily_returns(prices, method="log")
        diff = np.abs(simple_r - log_r)
        assert (diff < 1e-4).all().all()

