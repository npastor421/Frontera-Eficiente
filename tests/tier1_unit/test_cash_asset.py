"""
Unit Tests for Cash / Liquidity Asset Integration across the Quantitative Pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import fetch_asset_data, is_cash_ticker
from src.data.cleaner import clean_and_align_prices
from src.models.returns import calculate_expected_returns
from src.models.covariance import estimate_covariance_matrix, corr_from_covariance
from src.analytics.risk_metrics import compute_portfolio_risk_metrics
from src.optimization.optimizer import optimize_maximum_sharpe


def test_is_cash_ticker():
    assert is_cash_ticker("CASH") is True
    assert is_cash_ticker("cash") is True
    assert is_cash_ticker("USD") is True
    assert is_cash_ticker("usd_cash") is True
    assert is_cash_ticker("LIQUIDEZ") is True
    assert is_cash_ticker("AAPL") is False
    assert is_cash_ticker("SPY") is False


def test_cash_data_loading_and_cleaning(monkeypatch):
    # Mock yfinance return
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_spy_prices = pd.DataFrame({"SPY": np.linspace(100, 110, 100)}, index=dates)

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda **kwargs: mock_spy_prices)

    df_prices = fetch_asset_data(tickers=["SPY", "CASH"], start_date="2023-01-01", end_date="2023-05-31")
    assert "SPY" in df_prices.columns
    assert "CASH" in df_prices.columns
    assert len(df_prices) == 100

    clean_p, daily_r = clean_and_align_prices(df_prices)
    assert "SPY" in daily_r.columns
    assert "CASH" in daily_r.columns

    # Expected returns with CASH
    rf = 0.05
    exp_rets = calculate_expected_returns(daily_r, method="arithmetic", rf=rf)
    assert abs(exp_rets["CASH"] - rf) < 1e-6

    # Covariance with CASH
    cov_df, meta = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")
    assert abs(cov_df.loc["CASH", "CASH"]) < 1e-12
    assert abs(cov_df.loc["CASH", "SPY"]) < 1e-12
    assert abs(cov_df.loc["SPY", "CASH"]) < 1e-12

    # Correlation matrix
    corr_df = corr_from_covariance(cov_df)
    assert corr_df.loc["CASH", "CASH"] == 1.0
    assert corr_df.loc["CASH", "SPY"] == 0.0

    # Risk metrics: 20% CASH, 80% SPY
    weights = {"SPY": 0.80, "CASH": 0.20}
    metrics = compute_portfolio_risk_metrics(
        weights=weights,
        daily_returns=daily_r,
        expected_returns=exp_rets,
        cov_matrix=cov_df,
        rf=rf,
    )
    expected_p_return = 0.80 * exp_rets["SPY"] + 0.20 * rf
    assert abs(metrics.annualized_return - expected_p_return) < 1e-5


def test_optimize_maximum_sharpe_with_cash():
    # 2 risky assets + 1 cash asset
    mu = pd.Series({"AAPL": 0.15, "MSFT": 0.12, "CASH": 0.04})
    cov = pd.DataFrame(
        [
            [0.04, 0.01, 0.0],
            [0.01, 0.03, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=["AAPL", "MSFT", "CASH"],
        columns=["AAPL", "MSFT", "CASH"],
    )

    opt = optimize_maximum_sharpe(mu, cov, rf=0.04)
    assert opt.success is True
    # Pure tangency allocates to risky assets
    assert abs(opt.weights[2]) < 1e-6
    assert abs(sum(opt.weights) - 1.0) < 1e-4
