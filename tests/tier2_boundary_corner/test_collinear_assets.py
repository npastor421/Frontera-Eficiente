"""
Tier 2 Boundary & Corner Cases: Collinear & Degenerate Asset Universes (rho -> 1.0).
Verifies condition number diagnostics, Ledoit-Wolf shrinkage stabilization, and solver resilience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import (
    estimate_covariance_matrix,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    sample_covariance,
)
from src.models.returns import calculate_expected_returns
from src.models.stability import calculate_condition_number, ensure_positive_semidefinite


def test_collinear_assets_high_condition_number(collinear_prices):
    """Verify collinear assets generate high sample covariance condition number."""
    _, daily_r = clean_and_align_prices(collinear_prices)
    cov_sample = sample_covariance(daily_r)
    cond_sample = calculate_condition_number(cov_sample)

    # Condition number should be elevated due to near-identical returns
    assert cond_sample > 1e4, f"Condition number expected > 10^4, got {cond_sample}"


def test_ledoit_wolf_shrinkage_reduces_ill_conditioning(collinear_prices):
    """Verify Ledoit-Wolf shrinkage stabilizes condition number relative to sample covariance."""
    _, daily_r = clean_and_align_prices(collinear_prices)
    cov_sample = sample_covariance(daily_r)
    cond_sample = calculate_condition_number(cov_sample)

    cov_lw_diag, _ = ledoit_wolf_diagonal(daily_r)
    cond_lw = calculate_condition_number(cov_lw_diag)

    # Shrinkage regularizes spectrum -> lower or comparable condition number
    assert cond_lw < cond_sample or cond_lw < 1e8


def test_ensure_positive_semidefinite_on_collinear_matrix(collinear_prices):
    """Verify ensure_positive_semidefinite guarantees numerical solvability."""
    _, daily_r = clean_and_align_prices(collinear_prices)
    cov_df = sample_covariance(daily_r)

    psd_cov, was_repaired, cond_num = ensure_positive_semidefinite(cov_df, eps=1e-6)
    assert not np.isinf(cond_num)
    assert cond_num > 0.0


def test_optimizer_stability_under_collinearity(collinear_prices):
    """Verify optimizer converges and satisfies sum(w)=1.0 even with collinear assets."""
    try:
        from src.optimization.optimizer import optimize_global_minimum_variance
    except ImportError:
        pytest.skip("src.optimization module not yet implemented")

    _, daily_r = clean_and_align_prices(collinear_prices)
    cov_df, _ = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")

    gmv_res = optimize_global_minimum_variance(cov_df.values)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
