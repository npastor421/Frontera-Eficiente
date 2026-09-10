"""
Portfolio Diversification Analytics Module.

Computes mathematical metrics of genuine portfolio diversification:
- Choueifaty Diversification Ratio (DR)
- Herfindahl-Hirschman Concentration Index (HHI) for assets and sectors
- Effective Number of Constituents / Sectors (ENC)
- Shannon Entropy of Allocation
- Aggregation across sector, country, asset class, and market cap dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class DiversificationSummary:
    """Dataclass holding quantitative diversification indicators for a portfolio."""
    choueifaty_ratio: float
    effective_n_assets: float
    effective_n_sectors: float
    hhi_assets: float
    hhi_sectors: float
    hhi_countries: float
    top_3_assets_pct: float
    top_3_sectors_pct: float
    top_3_countries_pct: float
    shannon_entropy: float
    max_shannon_entropy: float
    entropy_ratio: float  # Shannon / Max Shannon (0 to 1)


def compute_herfindahl_hirschman_index(weights: Sequence[float]) -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI).

    HHI = sum((w_i * 100)^2)
    Ranges from ~0 (infinite equal diversification) to 10,000 (100% single concentration).

    US DOJ / FTC Antitrust benchmark:
    - HHI < 1500: Unconcentrated / Highly Diversified
    - 1500 <= HHI <= 2500: Moderately Concentrated
    - HHI > 2500: Highly Concentrated

    Parameters
    ----------
    weights : sequence of float
        Portfolio weights (summing to 1.0 or normalized).

    Returns
    -------
    float
        HHI index in [0, 10000].
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(np.abs(w))
    if w_sum <= 1e-12:
        return 0.0
    w_norm = w / w_sum
    pct = w_norm * 100.0
    return float(np.sum(pct ** 2))


def compute_effective_number_of_constituents(weights: Sequence[float]) -> float:
    """
    Compute the Effective Number of Constituents (ENC), also known as Inverse Simpson Index.

    ENC = 1 / sum(w_i^2)

    For N equally-weighted assets (w_i = 1/N), ENC = N.
    For 1 asset with 100%, ENC = 1.0.

    Parameters
    ----------
    weights : sequence of float
        Portfolio weights.

    Returns
    -------
    float
        Effective number of assets or sectors (>= 1.0).
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(np.abs(w))
    if w_sum <= 1e-12:
        return 0.0
    w_norm = w / w_sum
    denom = float(np.sum(w_norm ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 / denom)


def compute_choueifaty_diversification_ratio(
    weights: Sequence[float],
    asset_vols: Sequence[float],
    portfolio_vol: float,
) -> float:
    """
    Compute Choueifaty's Diversification Ratio (DR).

    DR = sum(w_i * sigma_i) / sigma_p

    A ratio of 1.0 implies perfect correlation (zero diversification benefits).
    Higher DR indicates superior diversification from non-correlated asset co-movements.

    Parameters
    ----------
    weights : sequence of float
        Asset weights (summing to 1.0).
    asset_vols : sequence of float
        Individual asset annualized volatilities.
    portfolio_vol : float
        Actual portfolio annualized volatility.

    Returns
    -------
    float
        Diversification Ratio (DR >= 1.0 under standard long-only portfolios).
    """
    w = np.asarray(weights, dtype=np.float64)
    sigmas = np.asarray(asset_vols, dtype=np.float64)
    weighted_avg_vol = float(np.sum(np.abs(w) * sigmas))
    p_vol = max(float(portfolio_vol), 1e-12)
    return float(weighted_avg_vol / p_vol)


def compute_shannon_entropy(weights: Sequence[float]) -> float:
    """
    Compute Shannon Information Entropy of portfolio weights.

    H = - sum(w_i * ln(w_i)) for w_i > 0.
    Maximum entropy for N assets is ln(N).
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(np.abs(w))
    if w_sum <= 1e-12:
        return 0.0
    w_norm = w / w_sum
    pos_w = w_norm[w_norm > 1e-12]
    if len(pos_w) == 0:
        return 0.0
    return float(-np.sum(pos_w * np.log(pos_w)))


def aggregate_dimension_weights(
    tickers: Sequence[str],
    weights: Sequence[float],
    metadata: Dict[str, Dict[str, Any]],
    dimension: str = "sector",
) -> Dict[str, float]:
    """
    Aggregate portfolio weights by a specified classification dimension.

    Parameters
    ----------
    tickers : sequence of str
        List of tickers in the portfolio.
    weights : sequence of float
        Weights corresponding to tickers (sum to 1.0 or percentage).
    metadata : dict
        Mapping ticker -> metadata dict (from fetch_asset_classification).
    dimension : str
        Dimension key in metadata ('sector', 'country', 'asset_class', 'market_cap_category').

    Returns
    -------
    dict
        Dictionary of category -> percentage weight (0.0 to 100.0), sorted descending by weight.
    """
    aggregated: Dict[str, float] = {}
    w_arr = np.asarray(weights, dtype=np.float64)
    # If weights sum to ~1.0, convert to percentages (0-100)
    multiplier = 100.0 if np.sum(np.abs(w_arr)) <= 2.0 else 1.0

    for i, t in enumerate(tickers):
        clean_t = str(t).strip().upper()
        w_val = float(w_arr[i]) * multiplier if i < len(w_arr) else 0.0
        if abs(w_val) < 1e-6:
            continue
        meta = metadata.get(clean_t, {})
        category = str(meta.get(dimension, "Otros / No Clasificado") or "Otros / No Clasificado")
        aggregated[category] = aggregated.get(category, 0.0) + w_val

    # Round to 2 decimal places and sort descending
    sorted_items = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return {k: round(v, 2) for k, v in sorted_items}


def compute_diversification_summary(
    tickers: Sequence[str],
    weights: Sequence[float],
    asset_vols: Sequence[float],
    portfolio_vol: float,
    metadata: Dict[str, Dict[str, Any]],
) -> DiversificationSummary:
    """
    Compute comprehensive diversification metrics for a portfolio.
    """
    w_arr = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(np.abs(w_arr))
    w_norm = w_arr / w_sum if w_sum > 1e-12 else np.ones_like(w_arr) / max(len(w_arr), 1)

    # 1. Choueifaty Diversification Ratio
    dr = compute_choueifaty_diversification_ratio(w_norm, asset_vols, portfolio_vol)

    # 2. Asset-level HHI and ENC
    hhi_assets = compute_herfindahl_hirschman_index(w_norm)
    enc_assets = compute_effective_number_of_constituents(w_norm)

    # Top 3 asset weight
    sorted_w_pct = sorted((w * 100.0 for w in w_norm), reverse=True)
    top_3_assets = sum(sorted_w_pct[:3])

    # 3. Sector-level aggregation & metrics
    sector_weights_dict = aggregate_dimension_weights(tickers, w_norm, metadata, dimension="sector")
    sector_w_norm = [v / 100.0 for v in sector_weights_dict.values()]
    hhi_sectors = compute_herfindahl_hirschman_index(sector_w_norm)
    enc_sectors = compute_effective_number_of_constituents(sector_w_norm)
    top_3_sectors = sum(list(sector_weights_dict.values())[:3])

    # 4. Country-level aggregation & metrics
    country_weights_dict = aggregate_dimension_weights(tickers, w_norm, metadata, dimension="country")
    country_w_norm = [v / 100.0 for v in country_weights_dict.values()]
    hhi_countries = compute_herfindahl_hirschman_index(country_w_norm)
    top_3_countries = sum(list(country_weights_dict.values())[:3])

    # 5. Shannon Entropy
    shannon = compute_shannon_entropy(w_norm)
    n = len(w_norm)
    max_shannon = float(np.log(n)) if n > 1 else 1.0
    entropy_ratio = min(max(shannon / max(max_shannon, 1e-12), 0.0), 1.0)

    return DiversificationSummary(
        choueifaty_ratio=dr,
        effective_n_assets=enc_assets,
        effective_n_sectors=enc_sectors,
        hhi_assets=hhi_assets,
        hhi_sectors=hhi_sectors,
        hhi_countries=hhi_countries,
        top_3_assets_pct=round(top_3_assets, 2),
        top_3_sectors_pct=round(top_3_sectors, 2),
        top_3_countries_pct=round(top_3_countries, 2),
        shannon_entropy=shannon,
        max_shannon_entropy=max_shannon,
        entropy_ratio=entropy_ratio,
    )
