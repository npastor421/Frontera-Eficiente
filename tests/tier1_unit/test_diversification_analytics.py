"""
Tier 1 Unit Tests: Diversification Analytics & Visualization Engine.
Verifies quantitative properties:
- Herfindahl-Hirschman Index (HHI)
- Effective Number of Constituents (ENC)
- Choueifaty Diversification Ratio (DR >= 1.0)
- Shannon Entropy & Dimension Aggregation
- Asset & Market Cap Categorization
- Treemap & Donut Chart Generation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.diversification import (
    DiversificationSummary,
    aggregate_dimension_weights,
    compute_choueifaty_diversification_ratio,
    compute_diversification_summary,
    compute_effective_number_of_constituents,
    compute_herfindahl_hirschman_index,
    compute_shannon_entropy,
)
from src.data.asset_metadata import (
    categorize_asset_class,
    categorize_market_cap,
    get_single_asset_metadata,
)
from src.visualization.diversification_plots import (
    plot_dimension_donut,
    plot_diversification_treemap,
)


# ===========================================================================
# 1. HHI & ENC Properties
# ===========================================================================

def test_hhi_bounds_and_known_values():
    """Verify HHI equals 10,000 / N for equal weights and 10,000 for single asset concentration."""
    # 1 asset: 100%
    assert abs(compute_herfindahl_hirschman_index([1.0]) - 10000.0) < 1e-4

    # 4 equal assets: 25% each -> 4 * (25^2) = 4 * 625 = 2500
    assert abs(compute_herfindahl_hirschman_index([0.25, 0.25, 0.25, 0.25]) - 2500.0) < 1e-4

    # 10 equal assets: 1000
    assert abs(compute_herfindahl_hirschman_index([0.10] * 10) - 1000.0) < 1e-4

    # Handles zero or empty
    assert compute_herfindahl_hirschman_index([]) == 0.0
    assert compute_herfindahl_hirschman_index([0.0, 0.0]) == 0.0


def test_effective_number_of_constituents():
    """Verify ENC returns exact N for equal weights and 1.0 for single concentrated asset."""
    assert abs(compute_effective_number_of_constituents([1.0]) - 1.0) < 1e-4
    assert abs(compute_effective_number_of_constituents([0.2, 0.2, 0.2, 0.2, 0.2]) - 5.0) < 1e-4

    # Concentrated 90% / 10%
    enc = compute_effective_number_of_constituents([0.9, 0.1])
    assert 1.0 < enc < 2.0


# ===========================================================================
# 2. Choueifaty Diversification Ratio (DR)
# ===========================================================================

def test_choueifaty_diversification_ratio():
    """Verify DR >= 1.0 for standard portfolios, and DR == 1.0 for identical/collinear assets."""
    weights = [0.5, 0.5]
    asset_vols = [0.20, 0.20]

    # Perfect correlation (sigma_p = weighted sum = 0.20)
    dr_collinear = compute_choueifaty_diversification_ratio(weights, asset_vols, portfolio_vol=0.20)
    assert abs(dr_collinear - 1.0) < 1e-4

    # Diversified portfolio (e.g. sigma_p = 0.15 due to low correlation)
    dr_diversified = compute_choueifaty_diversification_ratio(weights, asset_vols, portfolio_vol=0.15)
    assert dr_diversified > 1.0
    assert abs(dr_diversified - (0.20 / 0.15)) < 1e-4


# ===========================================================================
# 3. Shannon Entropy
# ===========================================================================

def test_shannon_entropy():
    """Verify Shannon entropy is maximized for uniform distribution and 0 for concentrated."""
    # 1 asset -> 0.0
    assert abs(compute_shannon_entropy([1.0])) < 1e-6

    # 4 equal assets -> ln(4)
    h_4 = compute_shannon_entropy([0.25, 0.25, 0.25, 0.25])
    assert abs(h_4 - np.log(4)) < 1e-4


# ===========================================================================
# 4. Classification & Metadata
# ===========================================================================

def test_market_cap_categorization():
    """Verify market cap scaling categories."""
    assert categorize_market_cap(300e9) == "Mega Cap (>$200B)"
    assert categorize_market_cap(50e9) == "Large Cap ($10B-$200B)"
    assert categorize_market_cap(5e9) == "Mid Cap ($2B-$10B)"
    assert categorize_market_cap(500e6) == "Small/Micro Cap (<$2B)"
    assert categorize_market_cap(None) == "N/A"
    assert categorize_market_cap(0.0) == "N/A"


def test_asset_class_categorization():
    """Verify asset class mapping."""
    assert categorize_asset_class("", "CASH") == "Liquidez (CASH)"
    assert categorize_asset_class("", "USD") == "Liquidez (CASH)"
    assert categorize_asset_class("CRYPTOCURRENCY", "BTC-USD") == "Criptoactivos"
    assert categorize_asset_class("ETF", "GLD") == "Materias Primas (Commodities)"
    assert categorize_asset_class("ETF", "TLT") == "Renta Fija (Bonos)"
    assert categorize_asset_class("ETF", "SPY") == "ETF / Fondos"
    assert categorize_asset_class("EQUITY", "AAPL") == "Renta Variable (Acciones)"


def test_cash_metadata_retrieval():
    """Verify synthetic cash assets get instantaneous structured metadata without API calls."""
    meta = get_single_asset_metadata("CASH")
    assert meta["ticker"] == "CASH"
    assert meta["asset_class"] == "Liquidez (CASH)"
    assert meta["sector"] == "Liquidez / Monetario"


# ===========================================================================
# 5. Dimension Aggregation & Diversification Summary
# ===========================================================================

def test_aggregate_dimension_weights():
    """Verify dimension aggregation groups and normalizes percentages properly."""
    tickers = ["AAPL", "MSFT", "BND", "CASH"]
    weights = [0.30, 0.20, 0.30, 0.20]
    metadata = {
        "AAPL": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable (Acciones)"},
        "MSFT": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable (Acciones)"},
        "BND": {"sector": "Renta Fija", "country": "Estados Unidos", "asset_class": "Renta Fija (Bonos)"},
        "CASH": {"sector": "Liquidez", "country": "Global (USD)", "asset_class": "Liquidez (CASH)"},
    }

    sectors = aggregate_dimension_weights(tickers, weights, metadata, dimension="sector")
    assert sectors["Tecnología"] == 50.0
    assert sectors["Renta Fija"] == 30.0
    assert sectors["Liquidez"] == 20.0

    classes = aggregate_dimension_weights(tickers, weights, metadata, dimension="asset_class")
    assert classes["Renta Variable (Acciones)"] == 50.0
    assert classes["Renta Fija (Bonos)"] == 30.0


def test_compute_diversification_summary():
    """Verify comprehensive summary returns valid DiversificationSummary dataclass."""
    tickers = ["AAPL", "GOOGL"]
    weights = [0.60, 0.40]
    asset_vols = [0.25, 0.22]
    portfolio_vol = 0.20
    metadata = {
        "AAPL": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable"},
        "GOOGL": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable"},
    }

    summary = compute_diversification_summary(tickers, weights, asset_vols, portfolio_vol, metadata)
    assert isinstance(summary, DiversificationSummary)
    assert summary.choueifaty_ratio > 1.0
    assert summary.effective_n_assets > 1.0
    assert summary.hhi_assets > 0.0
    assert summary.hhi_sectors == 10000.0  # 100% in one sector (Tecnología)
    assert summary.top_3_assets_pct == 100.0


# ===========================================================================
# 6. Plotly Figures Smoke Tests
# ===========================================================================

def test_treemap_and_donut_render():
    """Verify Treemap and Donut return valid Plotly figures without throwing exceptions."""
    tickers = ["AAPL", "MSFT", "GLD"]
    weights = [0.4, 0.4, 0.2]
    metadata = {
        "AAPL": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable", "short_name": "Apple"},
        "MSFT": {"sector": "Tecnología", "country": "Estados Unidos", "asset_class": "Renta Variable", "short_name": "Microsoft"},
        "GLD": {"sector": "Metales", "country": "Estados Unidos", "asset_class": "Commodities", "short_name": "Gold Trust"},
    }

    fig_tree = plot_diversification_treemap(tickers, weights, metadata)
    assert fig_tree is not None
    assert len(fig_tree.data) > 0

    dim_weights = {"Tecnología": 80.0, "Metales": 20.0}
    fig_donut = plot_dimension_donut(dim_weights, title="Sectores")
    assert fig_donut is not None
    assert len(fig_donut.data) > 0

    # Edge cases: empty figures
    empty_tree = plot_diversification_treemap([], [], {})
    assert empty_tree is not None
    empty_donut = plot_dimension_donut({}, title="Vacío")
    assert empty_donut is not None
