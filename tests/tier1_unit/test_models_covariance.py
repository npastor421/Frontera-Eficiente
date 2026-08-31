"""
Tier 1 Unit Tests: Robust Covariance Estimators (src/models/covariance.py).
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
def synthetic_rets():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    data = {
        "EQ1": np.random.normal(0.0005, 0.015, size=200),
        "EQ2": np.random.normal(0.0007, 0.018, size=200),
        "EQ3": np.random.normal(0.0003, 0.010, size=200),
        "EQ4": np.random.normal(0.0006, 0.012, size=200),
    }
    return pd.DataFrame(data, index=dates)


def test_t1_sample_covariance_symmetry_and_psd(synthetic_rets):
    cov = sample_covariance(synthetic_rets, ann_factor=252)
    assert np.allclose(cov.values, cov.values.T)
    eigs = np.linalg.eigvalsh(cov.values)
    assert np.all(eigs > 0)


def test_t1_ledoit_wolf_cc_analytical(synthetic_rets):
    cov_lw, delta = ledoit_wolf_constant_correlation(synthetic_rets, ann_factor=252)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(cov_lw.values, cov_lw.values.T)
    eigs = np.linalg.eigvalsh(cov_lw.values)
    assert np.all(eigs > 0)


def test_t1_ledoit_wolf_diag_sklearn(synthetic_rets):
    cov_diag, delta = ledoit_wolf_diagonal(synthetic_rets, ann_factor=252)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(cov_diag.values, cov_diag.values.T)
    eigs = np.linalg.eigvalsh(cov_diag.values)
    assert np.all(eigs > 0)


def test_t1_ewma_covariance_properties(synthetic_rets):
    cov_ewma = ewma_covariance(synthetic_rets, decay=0.94, ann_factor=252)
    assert np.allclose(cov_ewma.values, cov_ewma.values.T)
    eigs = np.linalg.eigvalsh(cov_ewma.values)
    assert np.all(eigs > 0)


def test_t1_covariance_to_correlation_properties(synthetic_rets):
    cov = sample_covariance(synthetic_rets)
    corr = covariance_to_correlation(cov)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.all(corr.values >= -1.0)
    assert np.all(corr.values <= 1.0)
    assert np.allclose(corr.values, corr.values.T)


def test_t1_estimate_covariance_matrix_metadata(synthetic_rets):
    for m in [CovarianceMethod.SAMPLE, CovarianceMethod.LEDOIT_WOLF_CC, CovarianceMethod.LEDOIT_WOLF_DIAG, CovarianceMethod.EWMA]:
        cov, meta = estimate_covariance_matrix(synthetic_rets, method=m)
        assert cov.shape == (4, 4)
        assert meta["n_assets"] == 4
        assert meta["t_samples"] == 200
