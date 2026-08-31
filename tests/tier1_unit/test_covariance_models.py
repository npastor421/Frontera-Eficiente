"""
Tier 1 Unit Tests: Expected Returns, Robust Covariance Models & Matrix Stability.
Verifies R2 requirements from ORIGINAL_REQUEST.md and PROJECT.md.
"""

from __future__ import annotations

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
from src.models.returns import (
    ReturnMethod,
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_capm_betas,
    calculate_expected_returns,
    capm_expected_returns,
    ewma_returns,
)
from src.models.stability import (
    calculate_condition_number,
    enforce_symmetry,
    ensure_positive_semidefinite,
    get_eigenvalues,
    is_positive_semidefinite,
    nearest_psd_higham,
)


# ===========================================================================
# 1. Expected Return Estimators Unit Tests
# ===========================================================================

def test_annualized_arithmetic_returns(sample_returns_df):
    """Verify arithmetic return annualized scaling equals daily mean * 252."""
    res = annualized_arithmetic_returns(sample_returns_df, ann_factor=252)
    assert isinstance(res, pd.Series)
    assert len(res) == 5
    expected = sample_returns_df.mean(axis=0) * 252
    np.testing.assert_allclose(res.values, expected.values, rtol=1e-6)


def test_annualized_geometric_returns_cagr(sample_returns_df):
    """Verify CAGR is strictly less than arithmetic mean for volatile assets (volatility drag)."""
    geom = annualized_geometric_returns(sample_returns_df, ann_factor=252)
    assert isinstance(geom, pd.Series)
    # Mathematical invariant: Compounded Arithmetic Mean >= CAGR (Jensen's inequality)
    compounded_arith = (1.0 + sample_returns_df.mean(axis=0)) ** 252 - 1.0
    assert (geom.values <= compounded_arith.values + 1e-12).all()


def test_ewma_returns_decay(sample_returns_df):
    """Verify EWMA expected returns with decay lambda=0.94."""
    ewma_res = ewma_returns(sample_returns_df, decay=0.94, ann_factor=252)
    assert isinstance(ewma_res, pd.Series)
    assert len(ewma_res) == 5
    assert not ewma_res.isna().any()


def test_capm_expected_returns_and_betas(sample_returns_df):
    """Verify CAPM returns: mu_i = Rf + beta_i * (mu_M - Rf)."""
    # Use first asset as benchmark
    bm_returns = sample_returns_df["AAPL"]
    betas = calculate_capm_betas(sample_returns_df, bm_returns)
    assert isinstance(betas, pd.Series)
    # Benchmark beta against itself must be exactly 1.0
    assert abs(betas["AAPL"] - 1.0) < 1e-6

    capm_mu, _ = capm_expected_returns(sample_returns_df, bm_returns, rf=0.04, ann_factor=252)
    assert isinstance(capm_mu, pd.Series)
    assert not capm_mu.isna().any()


def test_calculate_expected_returns_dispatcher(sample_returns_df):
    """Verify unified calculate_expected_returns entry point with all method strings."""
    for method_name in ["arithmetic", "geometric", "cagr", "ewma", "capm"]:
        res = calculate_expected_returns(sample_returns_df, method=method_name, rf=0.04)
        assert isinstance(res, pd.Series)
        assert len(res) == len(sample_returns_df.columns)
        assert not res.isna().any()


# ===========================================================================
# 2. Covariance Matrix Estimators Unit Tests
# ===========================================================================

def test_sample_covariance_symmetry_and_psd(sample_returns_df):
    """Verify sample covariance is symmetric and PSD."""
    cov = sample_covariance(sample_returns_df, ann_factor=252)
    assert isinstance(cov, pd.DataFrame)
    # Symmetry check: Cov == Cov.T
    np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)
    # PSD check: all eigenvalues >= -1e-8
    assert is_positive_semidefinite(cov)


def test_ledoit_wolf_constant_correlation(sample_returns_df):
    """Verify Ledoit-Wolf constant correlation shrinkage produces valid delta in [0, 1] and PSD cov."""
    cov_lw, delta = ledoit_wolf_constant_correlation(sample_returns_df, ann_factor=252)
    assert isinstance(cov_lw, pd.DataFrame)
    assert 0.0 <= delta <= 1.0
    # Symmetry and PSD invariants
    np.testing.assert_allclose(cov_lw.values, cov_lw.values.T, atol=1e-12)
    assert is_positive_semidefinite(cov_lw)


def test_ledoit_wolf_diagonal(sample_returns_df):
    """Verify Ledoit-Wolf diagonal target shrinkage (scikit-learn wrapper)."""
    cov_diag, delta = ledoit_wolf_diagonal(sample_returns_df, ann_factor=252)
    assert isinstance(cov_diag, pd.DataFrame)
    assert 0.0 <= delta <= 1.0
    np.testing.assert_allclose(cov_diag.values, cov_diag.values.T, atol=1e-12)
    assert is_positive_semidefinite(cov_diag)


def test_ewma_covariance(sample_returns_df):
    """Verify RiskMetrics EWMA covariance with decay lambda=0.94."""
    cov_ewma = ewma_covariance(sample_returns_df, decay=0.94, ann_factor=252)
    assert isinstance(cov_ewma, pd.DataFrame)
    np.testing.assert_allclose(cov_ewma.values, cov_ewma.values.T, atol=1e-12)
    assert is_positive_semidefinite(cov_ewma)


def test_covariance_to_correlation(sample_returns_df):
    """Verify covariance to correlation transformation yields diag=1.0 and bounds [-1, 1]."""
    cov = sample_covariance(sample_returns_df)
    corr = covariance_to_correlation(cov)
    assert isinstance(corr, pd.DataFrame)
    # Diagonal elements must be 1.0
    np.testing.assert_allclose(np.diag(corr.values), np.ones(5), atol=1e-10)
    # Off-diagonals bounded in [-1.0, 1.0]
    assert (corr.values >= -1.0 - 1e-10).all()
    assert (corr.values <= 1.0 + 1e-10).all()


def test_estimate_covariance_matrix_dispatcher(sample_returns_df):
    """Verify unified estimate_covariance_matrix returns covariance matrix and metadata."""
    for method in ["sample", "ledoit_wolf_cc", "ledoit_wolf_diag", "ewma"]:
        cov_df, meta = estimate_covariance_matrix(sample_returns_df, method=method)
        assert isinstance(cov_df, pd.DataFrame)
        assert isinstance(meta, dict)
        assert meta["n_assets"] == 5
        assert is_positive_semidefinite(cov_df)


# ===========================================================================
# 3. Matrix Stability & Higham (2002) PSD Repair Unit Tests
# ===========================================================================

def test_enforce_symmetry():
    """Verify enforce_symmetry creates exact symmetric copy."""
    asym = np.array([[1.0, 2.0], [3.0, 4.0]])
    sym = enforce_symmetry(asym)
    expected = np.array([[1.0, 2.5], [2.5, 4.0]])
    np.testing.assert_allclose(sym, expected, atol=1e-12)


def test_get_eigenvalues_and_psd_check():
    """Verify eigenvalue extraction and PSD validation."""
    psd_mat = np.array([[2.0, 0.5], [0.5, 2.0]])
    assert is_positive_semidefinite(psd_mat)

    non_psd_mat = np.array([[1.0, 2.0], [2.0, 1.0]])  # det = -3, has negative eigval
    assert not is_positive_semidefinite(non_psd_mat)


def test_higham_nearest_psd_repair(non_psd_covariance_matrix):
    """Verify Higham (2002) projection transforms non-PSD matrix into strictly PSD matrix."""
    assert not is_positive_semidefinite(non_psd_covariance_matrix)

    psd_repaired = nearest_psd_higham(non_psd_covariance_matrix, eps=1e-7)
    assert is_positive_semidefinite(psd_repaired, tol=1e-8)
    min_eig = np.min(get_eigenvalues(psd_repaired))
    assert min_eig >= 1e-7 - 1e-10


def test_ensure_positive_semidefinite_workflow(non_psd_covariance_matrix):
    """Verify ensure_positive_semidefinite diagnoses and repairs non-PSD matrix with condition number."""
    repaired_df, was_repaired, cond_num = ensure_positive_semidefinite(
        non_psd_covariance_matrix, eps=1e-7
    )
    assert was_repaired is True
    assert isinstance(repaired_df, pd.DataFrame)
    assert is_positive_semidefinite(repaired_df)
    assert cond_num > 0.0
    assert not np.isinf(cond_num)
