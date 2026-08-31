"""
Tier 1 Unit Tests: Expected Returns Estimators (src/models/returns.py).
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
def standard_returns():
    np.random.seed(101)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.DataFrame(
        {
            "ASSET_A": np.random.normal(0.0005, 0.01, size=252),
            "ASSET_B": np.random.normal(0.0010, 0.02, size=252),
            "BENCHMARK": np.random.normal(0.0004, 0.008, size=252),
        },
        index=dates,
    )


def test_t1_arithmetic_returns_exact_mean(standard_returns):
    mu = annualized_arithmetic_returns(standard_returns, ann_factor=252)
    assert len(mu) == 3
    for col in standard_returns.columns:
        assert np.isclose(mu[col], standard_returns[col].mean() * 252)


def test_t1_geometric_returns_compound(standard_returns):
    cagr = annualized_geometric_returns(standard_returns, ann_factor=252)
    assert len(cagr) == 3
    for col in standard_returns.columns:
        # Check against manual log-sum
        log_ret = np.log1p(standard_returns[col])
        expected = np.expm1(log_ret.mean() * 252)
        assert np.isclose(cagr[col], expected)


def test_t1_ewma_decay_behavior(standard_returns):
    ewma_94 = ewma_returns(standard_returns, decay=0.94, ann_factor=252)
    ewma_97 = ewma_returns(standard_returns, decay=0.97, ann_factor=252)
    assert len(ewma_94) == 3
    assert len(ewma_97) == 3
    assert not ewma_94.equals(ewma_97)


def test_t1_capm_beta_identity(standard_returns):
    bm = standard_returns["BENCHMARK"]
    betas = calculate_capm_betas(standard_returns, bm)
    assert np.isclose(betas["BENCHMARK"], 1.0, atol=1e-6)


def test_t1_capm_expected_returns_formula(standard_returns):
    bm = standard_returns["BENCHMARK"]
    rf = 0.05
    mu_m = 0.12
    capm_ret, betas = capm_expected_returns(
        standard_returns, benchmark_returns=bm, rf=rf, market_return=mu_m
    )
    for col in standard_returns.columns:
        expected = rf + betas[col] * (mu_m - rf)
        assert np.isclose(capm_ret[col], expected)


def test_t1_calculate_expected_returns_enum_and_string(standard_returns):
    for m in [ReturnMethod.ARITHMETIC, ReturnMethod.GEOMETRIC, ReturnMethod.EWMA, ReturnMethod.CAPM]:
        res1 = calculate_expected_returns(
            standard_returns, method=m, benchmark_returns=standard_returns["BENCHMARK"]
        )
        res2 = calculate_expected_returns(
            standard_returns, method=m.value, benchmark_returns=standard_returns["BENCHMARK"]
        )
        assert np.allclose(res1.values, res2.values)
