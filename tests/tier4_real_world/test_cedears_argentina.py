"""
Tier 4 Real-World Workflows: Argentine Market CEDEARs (.BA) in ARS.
Validates high-inflation, high-drift asset modeling, BYMA calendar alignment, and Ledoit-Wolf shrinkage stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns
from src.models.stability import is_positive_semidefinite


def test_cedears_argentina_pipeline(cedears_prices):
    """Verify CEDEARs (.BA) asset universe handling with higher volatility and drift."""
    clean_p, daily_r = clean_and_align_prices(cedears_prices, freq="B")
    assert len(clean_p.columns) == 6
    assert all(".BA" in str(col) for col in clean_p.columns)

    # Risk models
    mu_series = calculate_expected_returns(daily_r, method="arithmetic", ann_factor=248)
    cov_df, _ = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc", ann_factor=248)

    assert is_positive_semidefinite(cov_df)
    # High drift expected (> 5% individual, > 30% average annualized in local currency)
    assert (mu_series.values > 0.05).all()
    assert mu_series.mean() > 0.30

    try:
        from src.optimization.optimizer import optimize_global_minimum_variance
    except ImportError:
        pytest.skip("src.optimization not yet implemented")

    gmv_res = optimize_global_minimum_variance(cov_df.values)
    assert gmv_res.success is True
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5
