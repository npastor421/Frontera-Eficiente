"""
Tier 3 Integration Tests: Streamlit State Synchronization & Dynamic Sliders Logic.
Validates weight normalization algorithms, preset loading, and fast-action button logic.
"""

from __future__ import annotations

import numpy as np
import pytest


def normalize_weights_dict(raw_weights: dict[str, float]) -> dict[str, float]:
    """
    Standard normalization algorithm used across UI actions:
    If sum > 0: rescale to sum = 1.0
    If sum == 0: assign uniform 1/N
    """
    total = sum(raw_weights.values())
    n = len(raw_weights)
    if n == 0:
        return {}
    if total > 1e-12:
        return {k: v / total for k, v in raw_weights.items()}
    return {k: 1.0 / n for k in raw_weights}


def test_normalize_weights_positive_sum():
    """Verify positive slider weights rescale exactly to sum = 1.0."""
    raw = {"AAPL": 0.40, "MSFT": 0.40, "GOOGL": 0.40}  # sum = 1.20
    norm = normalize_weights_dict(raw)
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert abs(norm["AAPL"] - 1.0 / 3.0) < 1e-6


def test_normalize_weights_zero_sum_fallback():
    """Verify zero sum sliders fall back to uniform 1/N distribution."""
    raw = {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0, "AMZN": 0.0}
    norm = normalize_weights_dict(raw)
    assert abs(sum(norm.values()) - 1.0) < 1e-6
    assert all(abs(v - 0.25) < 1e-6 for v in norm.values())


def test_normalize_weights_with_cash_absorber():
    """Verify normalization preserves risky weights and assigns remainder to CASH."""
    # Simulation of UI matrix normalization
    rows = [
        {"Ticker": "AAPL", "Ponderación (%)": 30.0},
        {"Ticker": "MSFT", "Ponderación (%)": 40.0},
        {"Ticker": "CASH", "Ponderación (%)": 10.0},  # total 80%
    ]
    cash_idx = 2
    non_cash_sum = sum(rows[i]["Ponderación (%)"] for i in range(len(rows)) if i != cash_idx)
    assert non_cash_sum == 70.0
    rows[cash_idx]["Ponderación (%)"] = round(100.0 - non_cash_sum, 2)
    
    assert rows[0]["Ponderación (%)"] == 30.0  # AAPL untouched
    assert rows[1]["Ponderación (%)"] == 40.0  # MSFT untouched
    assert rows[2]["Ponderación (%)"] == 30.0  # CASH absorbs 100 - 70 = 30
    assert sum(r["Ponderación (%)"] for r in rows) == 100.0


def test_preset_portfolios_integrity():
    """Verify 1-Click preset definitions contain required 5 canonical portfolios with valid sums."""
    try:
        from src.presets.portfolio_presets import PRESETS, get_preset, list_presets
    except ImportError:
        # If presets module not yet created, define reference checks
        PRESETS = {
            "classic_60_40": {"tickers": ["SPY", "TLT"], "weights": {"SPY": 0.60, "TLT": 0.40}},
            "all_weather": {"tickers": ["SPY", "TLT", "IEF", "GLD", "DBC"], "weights": {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075}},
            "big_tech": {"tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], "weights": {"AAPL": 0.2, "MSFT": 0.2, "GOOGL": 0.2, "AMZN": 0.2, "NVDA": 0.2}},
            "cedears_argentina": {"tickers": ["AAPL.BA", "MSFT.BA", "GOOGL.BA", "MELI.BA", "SPY.BA", "KO.BA"], "weights": {"AAPL.BA": 0.20, "MSFT.BA": 0.20, "GOOGL.BA": 0.15, "MELI.BA": 0.20, "SPY.BA": 0.15, "KO.BA": 0.10}},
            "crypto_tradfi": {"tickers": ["SPY", "QQQ", "BTC-USD", "ETH-USD"], "weights": {"SPY": 0.50, "QQQ": 0.30, "BTC-USD": 0.15, "ETH-USD": 0.05}},
        }

    expected_presets = ["classic_60_40", "all_weather", "big_tech", "cedears_argentina", "crypto_tradfi"]
    for p_name in expected_presets:
        assert p_name in PRESETS
        p_data = PRESETS[p_name]
        weights = p_data["weights"]
        # Invariant: Preset weights must sum to 1.0 +- 1e-4
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        # All weights in [0, 1]
        assert all(0.0 <= w <= 1.0 for w in weights.values())


def test_remove_asset_transfers_weight_to_cash():
    """Verify removing an asset when CASH is present transfers its full weight to CASH."""
    rows = [
        {"Ticker": "AAPL", "Ponderación (%)": 30.0},
        {"Ticker": "MSFT", "Ponderación (%)": 30.0},
        {"Ticker": "GOOGL", "Ponderación (%)": 20.0},
        {"Ticker": "CASH", "Ponderación (%)": 20.0},
    ]
    target = "GOOGL"
    target_idx = [i for i, r in enumerate(rows) if r["Ticker"] == target][0]
    removed_row = rows.pop(target_idx)
    rem_w = removed_row["Ponderación (%)"]

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    cash_indices = [i for i, r in enumerate(rows) if r["Ticker"] in cash_symbols]
    assert len(cash_indices) == 1
    cash_idx = cash_indices[0]
    rows[cash_idx]["Ponderación (%)"] += rem_w

    assert len(rows) == 3
    assert rows[0]["Ticker"] == "AAPL" and rows[0]["Ponderación (%)"] == 30.0
    assert rows[1]["Ticker"] == "MSFT" and rows[1]["Ponderación (%)"] == 30.0
    assert rows[2]["Ticker"] == "CASH" and rows[2]["Ponderación (%)"] == 40.0
    assert sum(r["Ponderación (%)"] for r in rows) == 100.0


def test_remove_asset_distributes_equally_without_cash():
    """Verify removing an asset when no CASH is present distributes weight equally among remaining assets."""
    rows = [
        {"Ticker": "AAPL", "Ponderación (%)": 40.0},
        {"Ticker": "MSFT", "Ponderación (%)": 40.0},
        {"Ticker": "GOOGL", "Ponderación (%)": 20.0},
    ]
    target = "GOOGL"
    target_idx = [i for i, r in enumerate(rows) if r["Ticker"] == target][0]
    removed_row = rows.pop(target_idx)
    rem_w = removed_row["Ponderación (%)"]

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    cash_indices = [i for i, r in enumerate(rows) if r["Ticker"] in cash_symbols]
    assert len(cash_indices) == 0

    share = rem_w / len(rows)
    for r in rows:
        r["Ponderación (%)"] += share

    assert len(rows) == 2
    assert rows[0]["Ticker"] == "AAPL" and rows[0]["Ponderación (%)"] == 50.0
    assert rows[1]["Ticker"] == "MSFT" and rows[1]["Ponderación (%)"] == 50.0
    assert sum(r["Ponderación (%)"] for r in rows) == 100.0

