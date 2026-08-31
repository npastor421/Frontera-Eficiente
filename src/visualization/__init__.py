"""
Interactive Plotly Visualizers Package for Frontera Eficiente.
"""

from src.visualization.plots import (
    plot_allocation_comparison,
    plot_allocation_comparison_bar,
    plot_asset_allocation,
    plot_asset_allocation_donut,
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_efficient_frontier,
    plot_historical_backtest,
    plot_monte_carlo_cones,
    plot_monte_carlo_projection_cones,
    plot_portfolio_comparison_scatter,
)

__all__ = [
    "plot_efficient_frontier",
    "plot_asset_allocation",
    "plot_asset_allocation_donut",
    "plot_allocation_comparison",
    "plot_allocation_comparison_bar",
    "plot_correlation_heatmap",
    "plot_covariance_heatmap",
    "plot_historical_backtest",
    "plot_monte_carlo_cones",
    "plot_monte_carlo_projection_cones",
    "plot_portfolio_comparison_scatter",
]
