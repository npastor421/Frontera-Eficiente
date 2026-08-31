"""
Unit tests for Robust Covariance Estimators (src/models/covariance.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.covariance import (
    CovarianceMethod,
    covariance_to_correlation,
    estimate_covariance_matrix,
    ewma_covariance,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    sample_covariance,
)


@pytest.fixture
def correlated_returns_df():
    np.random.seed(42)
    t_obs, n_assets = 300, 5
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    corr_true = np.array([
        [1.0, 0.6, 0.5, 0.4, 0.3],
        [0.6, 1.0, 0.7, 0.5, 0.4],
        [0.5, 0.7, 1.0, 0.6, 0.5],
        [0.4, 0.5, 0.6, 1.0, 0.6],
        [0.3, 0.4, 0.5, 0.6, 1.0],
    ])
    l_chol = np.linalg.cholesky(corr_true)
    z = np.random.randn(t_obs, n_assets)
    r = (z @ l_chol.T) * 0.015
    dates = pd.date_range("2023-01-01", periods=t_obs, freq="B")
    return pd.DataFrame(r, index=dates, columns=tickers)


def test_sample_covariance_properties(correlated_returns_df):
    cov = sample_covariance(correlated_returns_df, ann_factor=252)
    assert isinstance(cov, pd.DataFrame)
    assert cov.shape == (5, 5)
    # Check exact symmetry
    assert np.allclose(cov.values, cov.values.T)
    # Check positive eigenvalues
    vals = np.linalg.eigvalsh(cov.values)
    assert np.all(vals > 0)


def test_ledoit_wolf_constant_correlation(correlated_returns_df):
    cov_lw, delta = ledoit_wolf_constant_correlation(correlated_returns_df, ann_factor=252)
    assert isinstance(cov_lw, pd.DataFrame)
    assert cov_lw.shape == (5, 5)
    assert 0.0 <= delta <= 1.0
    # Symmetry check
    assert np.allclose(cov_lw.values, cov_lw.values.T)
    # Positive definiteness
    vals = np.linalg.eigvalsh(cov_lw.values)
    assert np.all(vals > 0)


def test_ledoit_wolf_small_sample_t_less_than_n():
    # 6 assets, only 4 observations (T < N)
    # Sample covariance is singular, but Ledoit-Wolf must produce well-conditioned invertible matrix
    np.random.seed(42)
    x = np.random.randn(4, 6) * 0.02
    cov_lw, delta = ledoit_wolf_constant_correlation(x, ann_factor=252)
    assert delta > 0.0
    vals = np.linalg.eigvalsh(cov_lw)
    # Shrunk matrix should have positive eigenvalues
    assert np.min(vals) > 0


def test_ledoit_wolf_diagonal(correlated_returns_df):
    cov_diag, delta = ledoit_wolf_diagonal(correlated_returns_df, ann_factor=252)
    assert isinstance(cov_diag, pd.DataFrame)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(cov_diag.values, cov_diag.values.T)
    vals = np.linalg.eigvalsh(cov_diag.values)
    assert np.all(vals > 0)


def test_ewma_covariance(correlated_returns_df):
    cov_ewma = ewma_covariance(correlated_returns_df, decay=0.94, ann_factor=252)
    assert isinstance(cov_ewma, pd.DataFrame)
    assert np.allclose(cov_ewma.values, cov_ewma.values.T)
    vals = np.linalg.eigvalsh(cov_ewma.values)
    assert np.all(vals > 0)

    # Spike in volatility in recent days should noticeably increase EWMA variance
    df_spike = correlated_returns_df.copy()
    df_spike.iloc[-5:, :] *= 10.0
    cov_spike = ewma_covariance(df_spike, decay=0.94, ann_factor=252)
    assert np.trace(cov_spike.values) > np.trace(cov_ewma.values) * 2.0


def test_covariance_to_correlation(correlated_returns_df):
    cov = sample_covariance(correlated_returns_df)
    corr = covariance_to_correlation(cov)
    assert isinstance(corr, pd.DataFrame)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.all(corr.values >= -1.0)
    assert np.all(corr.values <= 1.0)
    assert np.allclose(corr.values, corr.values.T)


def test_estimate_covariance_matrix_dispatcher(correlated_returns_df):
    methods = [
        "sample",
        "ledoit_wolf_cc",
        "ledoit_wolf_diag",
        "ewma",
    ]
    for m in methods:
        cov, meta = estimate_covariance_matrix(correlated_returns_df, method=m)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
        assert "method" in meta
        assert meta["n_assets"] == 5
        assert np.allclose(cov.values, cov.values.T)


def test_single_asset_covariance():
    np.random.seed(42)
    df_single = pd.DataFrame({"AAPL": np.random.normal(0, 0.01, size=100)})
    cov, delta = ledoit_wolf_constant_correlation(df_single)
    assert cov.shape == (1, 1)
    assert delta == 0.0
    assert cov.iloc[0, 0] > 0
