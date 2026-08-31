"""
Tier 2 Boundary & Corner Case Tests for Models Layer.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_capm_betas,
    calculate_condition_number,
    calculate_expected_returns,
    covariance_to_correlation,
    ensure_positive_semidefinite,
    estimate_covariance_matrix,
    ewma_covariance,
    ewma_returns,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    nearest_psd_higham,
    sample_covariance,
)


def test_bva_single_asset():
    np.random.seed(42)
    single_df = pd.DataFrame({"SOLO": np.random.normal(0.001, 0.02, size=100)})
    mu = calculate_expected_returns(single_df, method="arithmetic")
    assert len(mu) == 1
    assert "SOLO" in mu.index

    cov, meta = estimate_covariance_matrix(single_df, method="ledoit_wolf_cc")
    assert cov.shape == (1, 1)
    assert meta["shrinkage_delta"] == 0.0

    corr = covariance_to_correlation(cov)
    assert np.isclose(corr.iloc[0, 0], 1.0)


def test_bva_high_collinearity_assets():
    # 2 assets that are 99.999% collinear
    np.random.seed(42)
    base = np.random.randn(200) * 0.01
    noise = np.random.randn(200) * 1e-6
    df_collinear = pd.DataFrame({"A1": base, "A2": base + noise})

    cov_sample = sample_covariance(df_collinear)
    cond_sample = calculate_condition_number(cov_sample)
    # Collinear sample covariance has huge condition number
    assert cond_sample > 1e4

    # Ledoit-Wolf diagonal shrinks towards identity, reducing condition number
    cov_lw, delta = ledoit_wolf_diagonal(df_collinear)
    cond_lw = calculate_condition_number(cov_lw)
    assert cond_lw < cond_sample


def test_bva_severely_underdetermined_t_much_less_than_n():
    # T=5 observations, N=20 assets
    np.random.seed(42)
    x = np.random.randn(5, 20) * 0.015
    df_fat = pd.DataFrame(x, columns=[f"A_{i}" for i in range(20)])

    cov_lw, delta = ledoit_wolf_constant_correlation(df_fat)
    assert delta > 0.0
    # Must be positive semi-definite
    eigs = np.linalg.eigvalsh(cov_lw.values)
    assert np.all(eigs > 0)


def test_bva_extreme_market_crash_returns():
    # -50% market crash in one day
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.01, size=100)
    rets[50] = -0.50  # -50% flash crash
    df_crash = pd.DataFrame({"CRASH_ASSET": rets})

    # Geometric return must handle -50% gracefully without NaN or complex numbers
    cagr = annualized_geometric_returns(df_crash, ann_factor=252)
    assert not np.isnan(cagr["CRASH_ASSET"])
    assert cagr["CRASH_ASSET"] < 0  # Severe loss reflected in CAGR


def test_bva_negative_definite_matrix_higham_repair():
    # Matrix with all negative eigenvalues
    neg_def = np.array([
        [-2.0, -0.5, -0.3],
        [-0.5, -1.5, -0.2],
        [-0.3, -0.2, -1.0]
    ])
    repaired, was_repaired, cond = ensure_positive_semidefinite(neg_def, eps=1e-7)
    assert was_repaired is True
    eigs = np.linalg.eigvalsh(repaired)
    assert np.all(eigs >= 1e-7 - 1e-12)


def test_bva_annualization_factors():
    np.random.seed(42)
    df = pd.DataFrame({"A": np.random.normal(0.001, 0.01, size=100)})
    for ann in [52, 252, 365]:
        mu = annualized_arithmetic_returns(df, ann_factor=ann)
        cov = sample_covariance(df, ann_factor=ann)
        assert np.isclose(mu["A"], df["A"].mean() * ann)
        assert np.isclose(cov.iloc[0, 0], df["A"].var() * ann)
