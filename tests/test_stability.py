"""
Unit tests for Matrix Stability and Higham (2002) PSD Repair (src/models/stability.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    RiskModelConfig,
    build_risk_model,
    calculate_condition_number,
    enforce_symmetry,
    ensure_positive_semidefinite,
    get_eigenvalues,
    is_positive_semidefinite,
    nearest_psd_higham,
)


@pytest.fixture
def non_psd_matrix():
    # Matrix with a clear negative eigenvalue
    mat = np.array([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, 0.95],
        [0.95, 0.95, 0.1],
    ])
    cols = ["A", "B", "C"]
    return pd.DataFrame(mat, index=cols, columns=cols)


@pytest.fixture
def clean_returns_df():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    data = {
        "AAPL": np.random.normal(0.0008, 0.015, size=252),
        "MSFT": np.random.normal(0.0006, 0.012, size=252),
        "NVDA": np.random.normal(0.0012, 0.025, size=252),
    }
    return pd.DataFrame(data, index=dates)


def test_enforce_symmetry():
    asym = np.array([[1.0, 2.0], [0.0, 1.0]])
    sym = enforce_symmetry(asym)
    assert np.isclose(sym[0, 1], 1.0)
    assert np.isclose(sym[1, 0], 1.0)
    assert np.allclose(sym, sym.T)


def test_eigenvalue_and_psd_check(non_psd_matrix):
    eigs = get_eigenvalues(non_psd_matrix)
    assert len(eigs) == 3
    assert np.min(eigs) < 0.0
    assert not is_positive_semidefinite(non_psd_matrix)


def test_condition_number(non_psd_matrix):
    # Non-PSD matrix should return inf condition number
    cond_bad = calculate_condition_number(non_psd_matrix)
    assert np.isinf(cond_bad)

    # Identity matrix should have condition number 1.0
    eye = np.eye(4)
    assert np.isclose(calculate_condition_number(eye), 1.0)


def test_higham_nearest_psd_repair(non_psd_matrix):
    eps = 1e-6
    repaired = nearest_psd_higham(non_psd_matrix, eps=eps)
    assert isinstance(repaired, pd.DataFrame)
    assert list(repaired.columns) == list(non_psd_matrix.columns)
    assert list(repaired.index) == list(non_psd_matrix.index)

    # Invariant: matrix is symmetric
    assert np.allclose(repaired.values, repaired.values.T)

    # Invariant: all eigenvalues >= eps
    repaired_eigs = get_eigenvalues(repaired)
    assert np.all(repaired_eigs >= eps - 1e-10)
    assert is_positive_semidefinite(repaired, tol=1e-8)


def test_ensure_positive_semidefinite(non_psd_matrix):
    # Test on non-PSD matrix
    psd_cov, was_repaired, cond_num = ensure_positive_semidefinite(non_psd_matrix, eps=1e-7)
    assert was_repaired is True
    assert is_positive_semidefinite(psd_cov)
    assert not np.isinf(cond_num)
    assert cond_num > 0

    # Test on already PSD matrix
    clean_cov = np.eye(3) * 0.04
    psd_clean, repaired_clean, cond_clean = ensure_positive_semidefinite(clean_cov)
    assert repaired_clean is False
    assert np.isclose(cond_clean, 1.0)


def test_build_risk_model_pipeline(clean_returns_df):
    config = RiskModelConfig(
        return_method="arithmetic",
        covariance_method="ledoit_wolf_cc",
        annualization_factor=252,
    )
    output = build_risk_model(clean_returns_df, config=config)

    assert output.expected_returns is not None
    assert len(output.expected_returns) == 3
    assert output.covariance_matrix.shape == (3, 3)
    assert output.correlation_matrix.shape == (3, 3)
    assert np.allclose(np.diag(output.correlation_matrix), 1.0)
    assert output.is_psd is True
    assert output.condition_number > 0
    assert len(output.annual_volatilities) == 3
    assert np.all(output.annual_volatilities > 0)
