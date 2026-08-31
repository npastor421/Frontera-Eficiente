"""
Shared Pytest Fixtures, Mock Data Generators, and Synthetic Market Environments
for Frontera Eficiente Markowitz Quantitative Portfolio Optimization Platform.
"""

from __future__ import annotations

import datetime
import io
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Global Random Seed & Environment Control Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure consistent random seed for all deterministic test executions."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Synthetic Data Generators (Pure Mathematical / Parametric)
# ---------------------------------------------------------------------------

def generate_gbm_prices(
    tickers: List[str],
    start_date: str = "2023-01-01",
    periods: int = 252,
    freq: str = "B",
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    corr_matrix: Optional[np.ndarray] = None,
    initial_prices: Optional[List[float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic correlated Geometric Brownian Motion (GBM) daily price paths.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq, name="Date")

    if mu is None:
        mu = np.linspace(0.06, 0.18, n)  # Annualized drift
    if sigma is None:
        sigma = np.linspace(0.15, 0.35, n)  # Annualized vol
    if corr_matrix is None:
        # Generate valid positive-definite correlation matrix
        raw_mat = rng.uniform(0.2, 0.6, size=(n, n))
        corr_matrix = (raw_mat + raw_mat.T) / 2.0
        np.fill_diagonal(corr_matrix, 1.0)
        # Ensure PSD
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-4)
        corr_matrix = (eigvecs * eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 1.0)

    if initial_prices is None:
        initial_prices = [100.0 * (1.0 + 0.5 * i) for i in range(n)]

    dt = 1.0 / 252.0
    cov = np.outer(sigma, sigma) * corr_matrix
    l_factor = np.linalg.cholesky(cov)

    # Standard normal innovations
    z = rng.standard_normal(size=(periods - 1, n))
    correlated_innovations = z @ l_factor.T

    # Asset daily drift
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    daily_log_returns = drift + np.sqrt(dt) * correlated_innovations

    prices = np.zeros((periods, n))
    prices[0] = initial_prices
    for t in range(1, periods):
        prices[t] = prices[t - 1] * np.exp(daily_log_returns[t - 1])

    return pd.DataFrame(prices, index=dates, columns=tickers, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixtures for Standard Asset Universes
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tickers() -> List[str]:
    """Standard 5-asset US equity/ETF ticker list."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


@pytest.fixture
def sample_prices_df(sample_tickers) -> pd.DataFrame:
    """Deterministic 252-day synthetic price DataFrame for 5 assets."""
    return generate_gbm_prices(
        tickers=sample_tickers,
        start_date="2023-01-03",
        periods=252,
        seed=101,
    )


@pytest.fixture
def sample_returns_df(sample_prices_df) -> pd.DataFrame:
    """Daily percentage returns derived from sample_prices_df."""
    return sample_prices_df.pct_change().dropna()


@pytest.fixture
def classic_60_40_prices() -> pd.DataFrame:
    """Deterministic price series for classic 60/40 universe: SPY (equity) and TLT (bonds)."""
    return generate_gbm_prices(
        tickers=["SPY", "TLT"],
        start_date="2022-01-03",
        periods=504,
        mu=np.array([0.10, 0.03]),
        sigma=np.array([0.18, 0.14]),
        corr_matrix=np.array([[1.0, -0.25], [-0.25, 1.0]]),
        initial_prices=[400.0, 140.0],
        seed=203,
    )


@pytest.fixture
def all_weather_prices() -> pd.DataFrame:
    """Deterministic price series for Ray Dalio All-Weather: SPY, TLT, IEF, GLD, DBC."""
    tickers = ["SPY", "TLT", "IEF", "GLD", "DBC"]
    mu = np.array([0.10, 0.04, 0.03, 0.07, 0.05])
    sigma = np.array([0.18, 0.15, 0.08, 0.16, 0.22])
    corr = np.array([
        [1.00, -0.20, -0.15,  0.05,  0.30],
        [-0.20, 1.00,  0.85,  0.20, -0.25],
        [-0.15, 0.85,  1.00,  0.15, -0.20],
        [0.05,  0.20,  0.15,  1.00,  0.25],
        [0.30, -0.25, -0.20,  0.25,  1.00],
    ])
    return generate_gbm_prices(
        tickers=tickers,
        start_date="2022-01-03",
        periods=504,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr,
        seed=303,
    )


@pytest.fixture
def cedears_prices() -> pd.DataFrame:
    """Deterministic price series for Argentine CEDEARs (.BA) trading in ARS."""
    tickers = ["AAPL.BA", "MSFT.BA", "GOOGL.BA", "MELI.BA", "SPY.BA", "KO.BA"]
    # Higher expected drift and volatility due to local ARS FX depreciation (CCL)
    mu = np.array([0.85, 0.80, 0.75, 0.90, 0.70, 0.65])
    sigma = np.array([0.55, 0.52, 0.50, 0.60, 0.48, 0.45])
    return generate_gbm_prices(
        tickers=tickers,
        start_date="2023-01-02",
        periods=248,
        mu=mu,
        sigma=sigma,
        seed=404,
    )


@pytest.fixture
def crypto_tradfi_raw_prices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (crypto_prices_7d, tradfi_prices_5d) to test asynchronous calendar alignment.
    BTC-USD trades 365 days; SPY and QQQ trade 252 business days.
    """
    # 90 calendar days
    cal_dates = pd.date_range(start="2023-01-01", periods=90, freq="D", name="Date")
    rng = np.random.default_rng(505)

    # Crypto (7-day calendar)
    btc_ret = rng.normal(0.001, 0.035, size=90)
    eth_ret = rng.normal(0.0012, 0.045, size=90)
    btc_p = 20000.0 * np.cumprod(1.0 + btc_ret)
    eth_p = 1500.0 * np.cumprod(1.0 + eth_ret)
    crypto_df = pd.DataFrame({"BTC-USD": btc_p, "ETH-USD": eth_p}, index=cal_dates)

    # TradFi (Business days only: Mon-Fri)
    b_dates = pd.date_range(start="2023-01-01", periods=90, freq="B", name="Date")
    spy_ret = rng.normal(0.0004, 0.012, size=len(b_dates))
    qqq_ret = rng.normal(0.0006, 0.016, size=len(b_dates))
    spy_p = 380.0 * np.cumprod(1.0 + spy_ret)
    qqq_p = 270.0 * np.cumprod(1.0 + qqq_ret)
    tradfi_df = pd.DataFrame({"SPY": spy_p, "QQQ": qqq_p}, index=b_dates)

    return crypto_df, tradfi_df


# ---------------------------------------------------------------------------
# Fixtures for Edge Cases & Boundary Conditions (Tier 2)
# ---------------------------------------------------------------------------

@pytest.fixture
def single_asset_prices() -> pd.DataFrame:
    """N=1 single asset price DataFrame."""
    return generate_gbm_prices(
        tickers=["SPY"],
        start_date="2023-01-03",
        periods=252,
        seed=606,
    )


@pytest.fixture
def collinear_prices() -> pd.DataFrame:
    """
    Asset universe with perfectly/near-perfectly collinear assets (rho >= 0.999).
    Useful for testing singularity, high condition number, and Ledoit-Wolf regularization.
    """
    base_df = generate_gbm_prices(
        tickers=["SPY"],
        start_date="2023-01-03",
        periods=252,
        seed=707,
    )
    p = base_df["SPY"].values
    # IVV is identical to SPY + 1e-6 noise
    rng = np.random.default_rng(708)
    ivv_p = p * (1.0 + rng.normal(0.0, 1e-5, size=len(p)))
    voo_p = p * 1.0001
    out = pd.DataFrame(
        {"SPY": p, "IVV": ivv_p, "VOO": voo_p},
        index=base_df.index,
    )
    return out


@pytest.fixture
def negative_returns_df() -> pd.DataFrame:
    """
    Asset returns universe where all expected returns are negative or lower than Rf.
    Tests optimizer behavior under bear-market regimes.
    """
    rng = np.random.default_rng(808)
    dates = pd.date_range(start="2022-01-03", periods=252, freq="B", name="Date")
    # Strictly negative daily returns with mean -15% to -25% annualized
    r1 = -0.0008 + rng.normal(0.0, 0.005, size=252)
    r2 = -0.0010 + rng.normal(0.0, 0.006, size=252)
    r3 = -0.0006 + rng.normal(0.0, 0.004, size=252)
    # Ensure sample means are strictly negative
    r1 -= max(0.0, float(np.mean(r1)) + 0.0005)
    r2 -= max(0.0, float(np.mean(r2)) + 0.0005)
    r3 -= max(0.0, float(np.mean(r3)) + 0.0005)
    return pd.DataFrame({"BEAR_1": r1, "BEAR_2": r2, "BEAR_3": r3}, index=dates)


@pytest.fixture
def non_psd_covariance_matrix() -> pd.DataFrame:
    """
    Symmetric matrix with an intentional negative eigenvalue.
    Tests Higham (2002) nearest PSD repair.
    """
    mat = np.array([
        [ 0.040,  0.035,  0.038],
        [ 0.035,  0.030,  0.036],
        [ 0.038,  0.036,  0.025],
    ], dtype=np.float64)
    # Check that minimum eigenvalue is negative
    eigvals = np.linalg.eigvalsh(mat)
    assert np.min(eigvals) < 0, f"Matrix expected to have negative eigenvalue, got {eigvals}"
    cols = ["Asset_A", "Asset_B", "Asset_C"]
    return pd.DataFrame(mat, index=cols, columns=cols)


@pytest.fixture
def extreme_outlier_prices() -> pd.DataFrame:
    """
    Price series with flash crashes (-90% drop) and extreme spikes (+300%).
    Tests numerical stability of log returns, CAGR, VaR, and Sortino calculations.
    """
    dates = pd.date_range(start="2023-01-03", periods=252, freq="B", name="Date")
    rng = np.random.default_rng(909)
    p1 = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=252)))
    p2 = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=252)))

    # Inject flash crash at day 100 on asset 1 (drops 80%)
    p1[100:] = p1[100:] * 0.20
    # Inject massive spike at day 150 on asset 2 (quadruples)
    p2[150:] = p2[150:] * 4.0

    return pd.DataFrame({"CRASH_ASSET": p1, "SPIKE_ASSET": p2}, index=dates)


# ---------------------------------------------------------------------------
# Fixtures for Manual CSV/Excel Parser Testing
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_prices_csv_text() -> str:
    """Standard comma-delimited Wide Prices CSV string."""
    return (
        "Date,AAPL,MSFT,GOOGL\n"
        "2023-01-03,125.07,239.58,89.70\n"
        "2023-01-04,126.36,229.10,88.71\n"
        "2023-01-05,125.02,222.31,86.77\n"
        "2023-01-06,129.62,224.93,88.16\n"
        "2023-01-09,130.15,227.12,88.80\n"
    )


@pytest.fixture
def european_comma_decimal_csv_text() -> str:
    """Semicolon delimiter with Latin/European comma decimals (e.g. '125,07')."""
    return (
        "Fecha;AAPL;MSFT;GOOGL\n"
        "2023-01-03;125,07;239,58;89,70\n"
        "2023-01-04;126,36;229,10;88,71\n"
        "2023-01-05;125,02;222,31;86,77\n"
        "2023-01-06;129,62;224,93;88,16\n"
        "2023-01-09;130,15;227,12;88,80\n"
    )


@pytest.fixture
def long_tidy_csv_text() -> str:
    """Long / Tidy format CSV string: Date, Ticker, Price."""
    return (
        "Date,Ticker,Price\n"
        "2023-01-03,AAPL,125.07\n"
        "2023-01-03,MSFT,239.58\n"
        "2023-01-04,AAPL,126.36\n"
        "2023-01-04,MSFT,229.10\n"
        "2023-01-05,AAPL,125.02\n"
        "2023-01-05,MSFT,222.31\n"
    )


@pytest.fixture
def wide_excel_bytes() -> bytes:
    """Generates an in-memory .xlsx file bytes containing wide price data."""
    dates = pd.date_range("2023-01-03", periods=10, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "AAPL": np.linspace(130, 145, 10),
        "MSFT": np.linspace(230, 250, 10),
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prices")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Mock yfinance Helper
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_yfinance_download():
    """
    Context manager / fixture to mock `yfinance.download` with realistic multi-index DataFrame.
    """
    def _create_mock_yf_response(tickers: List[str], start_date="2023-01-01", periods=100):
        dates = pd.date_range(start=start_date, periods=periods, freq="B")
        rng = np.random.default_rng(42)
        n = len(tickers)

        if n == 1:
            p = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=periods))
            df = pd.DataFrame({
                "Open": p,
                "High": p * 1.01,
                "Low": p * 0.99,
                "Close": p,
                "Adj Close": p,
                "Volume": 1000000,
            }, index=dates)
            return df

        # Multi-ticker: create MultiIndex
        metrics = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        tuples = [(m, t) for m in metrics for t in tickers]
        mindex = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])

        data = {}
        for m in metrics:
            for t in tickers:
                base = rng.uniform(50, 300)
                p = base * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=periods))
                data[(m, t)] = p

        df = pd.DataFrame(data, index=dates, columns=mindex)
        return df

    return _create_mock_yf_response
