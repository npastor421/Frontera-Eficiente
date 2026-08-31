"""
Tier 3 Integration Tests: End-to-End Computational Pipeline Flow.
Validates sequential dataflow: Ingestion -> Cleaning -> Covariance & PSD -> Optimization -> Analytics -> Export.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_and_align_prices
from src.models.covariance import estimate_covariance_matrix
from src.models.returns import calculate_expected_returns
from src.models.stability import ensure_positive_semidefinite


def test_full_pipeline_data_to_analytics_to_export(sample_prices_df):
    """
    Test full end-to-end dataflow pipeline:
    1. Raw prices -> clean_and_align_prices -> daily returns
    2. Daily returns -> calculate_expected_returns & estimate_covariance_matrix
    3. ensure_positive_semidefinite on covariance
    4. optimize_maximum_sharpe & optimize_global_minimum_variance
    5. compute_portfolio_risk_metrics
    6. export_full_excel workbook generation
    """
    # 1. Cleaning
    clean_p, daily_r = clean_and_align_prices(sample_prices_df)
    assert not clean_p.empty
    assert not daily_r.empty

    # 2. Risk Modeling
    mu_series = calculate_expected_returns(daily_r, method="arithmetic")
    cov_df, _ = estimate_covariance_matrix(daily_r, method="ledoit_wolf_cc")

    # 3. Stability check
    psd_cov, was_repaired, cond_num = ensure_positive_semidefinite(cov_df)
    assert not np.isinf(cond_num)

    # 4. Optimization (if Milestone 3 implemented)
    try:
        from src.optimization.optimizer import (
            optimize_global_minimum_variance,
            optimize_maximum_sharpe,
        )
        from src.analytics.risk_metrics import compute_portfolio_risk_metrics
        from src.export.exporter import export_full_excel
    except ImportError:
        pytest.skip("Downstream optimization, analytics, or export modules not yet implemented")

    rf = 0.04
    ms_res = optimize_maximum_sharpe(mu_series.values, psd_cov.values, rf=rf)
    gmv_res = optimize_global_minimum_variance(psd_cov.values)

    assert ms_res.success is True
    assert gmv_res.success is True
    assert abs(np.sum(ms_res.weights) - 1.0) <= 1e-5
    assert abs(np.sum(gmv_res.weights) - 1.0) <= 1e-5

    # 5. Risk Analytics
    ms_metrics = compute_portfolio_risk_metrics(
        weights=ms_res.weights,
        daily_returns=daily_r,
        expected_returns=mu_series,
        cov_matrix=psd_cov,
        rf=rf,
    )
    gmv_metrics = compute_portfolio_risk_metrics(
        weights=gmv_res.weights,
        daily_returns=daily_r,
        expected_returns=mu_series,
        cov_matrix=psd_cov,
        rf=rf,
    )

    assert ms_metrics["sharpe_ratio"] >= gmv_metrics["sharpe_ratio"] - 1e-5
    assert gmv_metrics["annualized_volatility"] <= ms_metrics["annualized_volatility"] + 1e-5

    # 6. Multi-Sheet Export
    excel_bytes = export_full_excel(
        metrics_dict={"Max Sharpe": ms_metrics, "GMV": gmv_metrics},
        weights_dict={
            ticker: {"Max Sharpe": float(ms_res.weights[i]), "GMV": float(gmv_res.weights[i])}
            for i, ticker in enumerate(clean_p.columns)
        },
        corr_matrix=cov_df,
        cov_matrix=psd_cov,
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 500
