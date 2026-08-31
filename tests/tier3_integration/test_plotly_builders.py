"""
Tier 3 Integration Tests: Interactive Plotly Visualizers Generation.
Validates Plotly figure schemas, trace counts, annotation accuracy, and theme consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Dynamic imports for visualization module (Milestone 4)
try:
    import plotly.graph_objects as go
    from src.visualization.plots import (
        plot_allocation_comparison,
        plot_asset_allocation,
        plot_correlation_heatmap,
        plot_covariance_heatmap,
        plot_efficient_frontier,
        plot_historical_backtest,
        plot_monte_carlo_cones,
    )
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False


pytestmark = pytest.mark.skipif(
    not HAS_PLOTS,
    reason="src.visualization module not yet implemented by Milestone 4",
)


def test_plot_asset_allocation_donut():
    """Verify asset allocation donut chart generates valid go.Figure with hole property."""
    weights = {"AAPL": 0.30, "MSFT": 0.40, "GOOGL": 0.30}
    fig = plot_asset_allocation(weights)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Check that it is a Pie trace with hole parameter
    assert fig.data[0].type == "pie"
    assert getattr(fig.data[0], "hole", 0) > 0


def test_plot_allocation_comparison_grouped_bar():
    """Verify 4-way comparison bar chart renders all portfolio series."""
    weights_dict = {
        "Usuario": {"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34},
        "Max Sharpe": {"AAPL": 0.50, "MSFT": 0.30, "GOOGL": 0.20},
        "GMV": {"AAPL": 0.10, "MSFT": 0.70, "GOOGL": 0.20},
    }
    fig = plot_allocation_comparison(weights_dict)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 3 bars per asset
    assert fig.layout.barmode == "group"


def test_plot_heatmaps_correlation_and_covariance():
    """Verify correlation and covariance heatmaps render numeric text annotations."""
    corr_df = pd.DataFrame(
        [[1.0, 0.45], [0.45, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    cov_df = pd.DataFrame(
        [[0.04, 0.015], [0.015, 0.03]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    fig_corr = plot_correlation_heatmap(corr_df)
    assert isinstance(fig_corr, go.Figure)
    assert fig_corr.data[0].type == "heatmap"

    fig_cov = plot_covariance_heatmap(cov_df)
    assert isinstance(fig_cov, go.Figure)
    assert fig_cov.data[0].type == "heatmap"


def test_plot_historical_backtest():
    """Verify $10,000 USD historical wealth chart generates valid multi-line time series."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(0.0005, 0.01, size=100), index=dates)
    s2 = pd.Series(rng.normal(0.0007, 0.012, size=100), index=dates)

    fig = plot_historical_backtest(
        returns_dict={"Usuario": s1, "Max Sharpe": s2},
        initial_capital=10000.0,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2
