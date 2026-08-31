"""
Unit tests for Expected Returns Estimators (src/models/returns.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.returns import (
    ReturnMethod,
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_capm_betas,
    calculate_expected_returns,
    capm_expected_returns,
    ewma_returns,
)


@pytest.fixture
def sample_returns_df():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    data = {
        "AAPL": np.random.normal(0.0008, 0.015, size=252),
        "MSFT": np.random.normal(0.0006, 0.012, size=252),
        "SPY": np.random.normal(0.0004, 0.010, size=252),
    }
    return pd.DataFrame(data, index=dates)


def test_annualized_arithmetic_returns(sample_returns_df):
    mu = annualized_arithmetic_returns(sample_returns_df, ann_factor=252)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == ["AAPL", "MSFT", "SPY"]
    # Check exact equality with numpy mean
    for col in sample_returns_df.columns:
        expected = sample_returns_df[col].mean() * 252
        assert np.isclose(mu[col], expected)


def test_annualized_geometric_returns(sample_returns_df):
    mu_geom = annualized_geometric_returns(sample_returns_df, ann_factor=252)
    assert isinstance(mu_geom, pd.Series)
    assert list(mu_geom.index) == ["AAPL", "MSFT", "SPY"]
    # Due to volatility drag (Jensen's inequality), (1 + r_mean)^252 - 1 >= CAGR
    for col in sample_returns_df.columns:
        daily_mean = sample_returns_df[col].mean()
        compounded_arith = (1.0 + daily_mean) ** 252 - 1.0
        assert compounded_arith >= mu_geom[col]


def test_geometric_returns_compound_property():
    # Price doubles in 252 days: P0 = 100, P_252 = 200
    # CAGR should be exactly 100% (1.0)
    prices = np.linspace(100, 200, 253)
    daily_rets = prices[1:] / prices[:-1] - 1.0
    cagr = annualized_geometric_returns(daily_rets, ann_factor=252)
    # Using log sum compounding: exp(sum(ln(1+r))) = P_T / P_0 = 2.0
    assert np.isclose(cagr, 1.0, atol=1e-3)


def test_ewma_returns_weighting(sample_returns_df):
    mu_ewma = ewma_returns(sample_returns_df, decay=0.94, ann_factor=252)
    assert isinstance(mu_ewma, pd.Series)
    assert len(mu_ewma) == 3

    # Check regime shift sensitivity: create series with 0 return first half, +5% daily second half
    shifted_rets = np.zeros(252)
    shifted_rets[200:] = 0.05
    df_shift = pd.DataFrame({"SHIFT": shifted_rets})
    mu_ewma_shift = ewma_returns(df_shift, decay=0.94, ann_factor=252)["SHIFT"]
    mu_arith_shift = annualized_arithmetic_returns(df_shift, ann_factor=252)["SHIFT"]
    # EWMA should place much higher weight on recent high returns than arithmetic mean
    assert mu_ewma_shift > mu_arith_shift


def test_capm_betas_and_returns(sample_returns_df):
    spy = sample_returns_df["SPY"]
    betas = calculate_capm_betas(sample_returns_df, spy)
    assert isinstance(betas, pd.Series)
    # Beta of SPY with itself must be exactly 1.0
    assert np.isclose(betas["SPY"], 1.0, atol=1e-7)

    # CAPM expected returns
    mu_capm, b = capm_expected_returns(
        sample_returns_df, benchmark_returns=spy, rf=0.03, market_return=0.10
    )
    assert np.isclose(b["SPY"], 1.0, atol=1e-7)
    # Expected return of SPY under CAPM must equal market_return (0.10)
    assert np.isclose(mu_capm["SPY"], 0.10, atol=1e-7)


def test_calculate_expected_returns_dispatcher(sample_returns_df):
    for method in ["arithmetic", "geometric", "cagr", "ewma", "capm"]:
        res = calculate_expected_returns(
            sample_returns_df,
            method=method,
            rf=0.04,
            benchmark_returns=sample_returns_df["SPY"],
        )
        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert not res.isna().any()


def test_returns_error_handling(sample_returns_df):
    # Empty DataFrame
    with pytest.raises(ValueError):
        annualized_arithmetic_returns(pd.DataFrame())

    # NaN handling
    nan_df = sample_returns_df.copy()
    nan_df.iloc[5, 0] = np.nan
    with pytest.raises(ValueError):
        annualized_arithmetic_returns(nan_df)

    # Invalid decay
    with pytest.raises(ValueError):
        ewma_returns(sample_returns_df, decay=1.5)

    # Invalid method
    with pytest.raises(ValueError):
        calculate_expected_returns(sample_returns_df, method="unknown_method")
