"""
Unit and Edge Case Tests for Data Loader (src/data/loader.py).
"""

import io
from pathlib import Path
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    fetch_asset_data,
    load_manual_file,
    parse_manual_data,
    validate_tickers,
)


class TestValidateTickers:
    def test_single_ticker_str(self):
        assert validate_tickers("AAPL") == ["AAPL"]
        assert validate_tickers("  BTC-USD  ") == ["BTC-USD"]

    def test_comma_separated_str(self):
        assert validate_tickers("AAPL, MSFT, BTC-USD") == ["AAPL", "MSFT", "BTC-USD"]
        assert validate_tickers("AAPL,  AAPL.BA, SPY , ") == ["AAPL", "AAPL.BA", "SPY"]

    def test_sequence_input_and_deduplication(self):
        assert validate_tickers(["AAPL", "MSFT", "AAPL", "NVDA"]) == ["AAPL", "MSFT", "NVDA"]
        assert validate_tickers(("SPY", "QQQ", "SPY")) == ["SPY", "QQQ"]

    def test_empty_input_raises_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tickers("")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tickers([])
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_tickers("   ,   ")

    def test_unsupported_type_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported ticker format"):
            validate_tickers(12345)


class TestFetchAssetData:
    @mock.patch("yfinance.download")
    def test_fetch_multi_ticker_multiindex(self, mock_yf):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Build MultiIndex columns (Price, Ticker)
        cols = pd.MultiIndex.from_tuples(
            [("Adj Close", "AAPL"), ("Adj Close", "MSFT"), ("Close", "AAPL"), ("Close", "MSFT")],
            names=["Price", "Ticker"],
        )
        data = np.array([
            [150.0, 300.0, 150.0, 300.0],
            [152.0, 305.0, 152.0, 305.0],
            [151.0, 302.0, 151.0, 302.0],
            [153.0, 308.0, 153.0, 308.0],
            [155.0, 310.0, 155.0, 310.0],
        ])
        mock_yf.return_value = pd.DataFrame(data, index=dates, columns=cols)

        df = fetch_asset_data(["AAPL", "MSFT"], "2024-01-01", "2024-01-08")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["AAPL", "MSFT"]
        assert len(df) == 5
        assert df["AAPL"].iloc[0] == 150.0
        assert df["MSFT"].iloc[-1] == 310.0
        assert df.index.name == "Date"

    @mock.patch("yfinance.download")
    def test_fetch_single_ticker_flat(self, mock_yf):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_yf.return_value = pd.DataFrame(
            {"Adj Close": [100.0, 102.0, 101.0], "Volume": [1000, 1200, 1100]},
            index=dates,
        )

        df = fetch_asset_data("SPY", "2024-01-01", "2024-01-04")
        assert list(df.columns) == ["SPY"]
        assert df["SPY"].iloc[0] == 100.0
        assert len(df) == 3

    @mock.patch("yfinance.download")
    def test_fetch_crypto_and_cedear(self, mock_yf):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_tuples(
            [("Close", "BTC-USD"), ("Close", "AAPL.BA")],
            names=["Price", "Ticker"],
        )
        mock_yf.return_value = pd.DataFrame(
            [[40000.0, 15000.0], [41000.0, 15200.0], [42000.0, 15100.0], [43000.0, 15300.0]],
            index=dates,
            columns=cols,
        )

        df = fetch_asset_data(["BTC-USD", "AAPL.BA"], "2024-01-01", "2024-01-05")
        assert "BTC-USD" in df.columns
        assert "AAPL.BA" in df.columns
        assert df["BTC-USD"].iloc[0] == 40000.0

    @mock.patch("yfinance.download")
    def test_empty_download_raises_value_error(self, mock_yf):
        mock_yf.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No price data found"):
            fetch_asset_data(["INVALID123"], "2024-01-01", "2024-01-10")


class TestParseManualData:
    def test_wide_prices_csv_comma_decimal_dot(self):
        csv_content = (
            "Date,AAPL,MSFT,GOOG\n"
            "2024-01-02,185.50,370.20,140.10\n"
            "2024-01-03,184.20,368.50,139.80\n"
            "2024-01-04,186.10,372.00,141.20\n"
        )
        df = parse_manual_data(csv_content)
        assert list(df.columns) == ["AAPL", "MSFT", "GOOG"]
        assert len(df) == 3
        assert df.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 185.50
        assert df.index.name == "Date"

    def test_wide_prices_csv_semicolon_and_comma_decimal(self):
        # European / Latin format
        csv_content = (
            "Fecha;SPY;GLD;BTC\n"
            "2024-01-02;475,25;190,50;42500,75\n"
            "2024-01-03;473,10;191,20;43100,00\n"
            "2024-01-04;476,80;189,90;44200,50\n"
        )
        df = parse_manual_data(csv_content)
        assert list(df.columns) == ["SPY", "GLD", "BTC"]
        assert df.loc[pd.Timestamp("2024-01-02"), "SPY"] == 475.25
        assert df.loc[pd.Timestamp("2024-01-04"), "BTC"] == 44200.50

    def test_long_format_csv(self):
        csv_content = (
            "Date,Ticker,Price\n"
            "2024-01-02,AAPL,180.0\n"
            "2024-01-02,MSFT,370.0\n"
            "2024-01-03,AAPL,182.0\n"
            "2024-01-03,MSFT,375.0\n"
        )
        df = parse_manual_data(csv_content)
        assert set(df.columns) == {"AAPL", "MSFT"}
        assert len(df) == 2
        assert df.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 180.0
        assert df.loc[pd.Timestamp("2024-01-03"), "MSFT"] == 375.0

    def test_bytes_io_input(self):
        csv_bytes = b"Date,ASSET_A,ASSET_B\n2024-01-01,10.0,20.0\n2024-01-02,11.0,21.0\n"
        buf = io.BytesIO(csv_bytes)
        df = parse_manual_data(buf)
        assert list(df.columns) == ["ASSET_A", "ASSET_B"]
        assert len(df) == 2

    def test_excel_file_parsing(self, tmp_path):
        excel_path = tmp_path / "test_data.xlsx"
        raw_df = pd.DataFrame(
            {
                "Fecha": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
                "StockA": [100.5, 102.3],
                "StockB": [50.0, 51.2],
            }
        )
        raw_df.to_excel(excel_path, index=False)

        df = load_manual_file(excel_path)
        assert list(df.columns) == ["StockA", "StockB"]
        assert len(df) == 2
        assert df.iloc[0, 0] == 100.5

    def test_empty_or_invalid_file_raises_error(self):
        with pytest.raises(ValueError, match="empty|not found"):
            parse_manual_data("")

        no_date_csv = "ColA,ColB\n10,20\n30,40\n"
        with pytest.raises(ValueError, match="valid Date|recognizable dates|valid dates"):
            parse_manual_data(no_date_csv)

    def test_thousands_separator_and_currency_symbols(self):
        csv_text = (
            "Date;PriceA;PriceB\n"
            "2024-01-02;$1.250,50;€500,25\n"
            "2024-01-03;$1.260,75;€505,80\n"
        )
        df = parse_manual_data(csv_text)
        assert list(df.columns) == ["PriceA", "PriceB"]
        assert pytest.approx(df.loc[pd.Timestamp("2024-01-02"), "PriceA"], rel=1e-5) == 1250.50
        assert pytest.approx(df.loc[pd.Timestamp("2024-01-02"), "PriceB"], rel=1e-5) == 500.25

    def test_long_format_returns_parsing(self):
        csv_text = (
            "Date,Symbol,Return\n"
            "2024-01-02,AAPL,0.015\n"
            "2024-01-02,GOOG,-0.005\n"
            "2024-01-03,AAPL,-0.002\n"
            "2024-01-03,GOOG,0.020\n"
        )
        df = parse_manual_data(csv_text, is_returns=True)
        assert set(df.columns) == {"AAPL", "GOOG"}
        assert len(df) == 2
        assert pytest.approx(df.loc[pd.Timestamp("2024-01-02"), "AAPL"], rel=1e-5) == 0.015

    def test_explicit_date_col_override(self):
        csv_text = (
            "CustomPeriod,ValA,ValB\n"
            "2024-02-01,10.0,20.0\n"
            "2024-02-02,11.0,21.0\n"
        )
        df = parse_manual_data(csv_text, date_col="CustomPeriod")
        assert list(df.columns) == ["ValA", "ValB"]
        assert df.index.min() == pd.Timestamp("2024-02-01")

