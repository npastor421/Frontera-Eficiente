"""
Unit Tests for Custom Portfolio Persistence Storage and Comparison Visualizer.
"""

import json
import pytest
from pathlib import Path
from src.presets.storage import (
    load_saved_portfolios,
    save_custom_portfolio,
    delete_custom_portfolio,
    export_portfolios_json,
    import_portfolios_json,
)
from src.visualization.plots import plot_portfolio_comparison_scatter


def test_save_load_delete_custom_portfolio(tmp_path, monkeypatch):
    test_storage_file = tmp_path / "test_user_portfolios.json"
    import src.presets.storage as storage_mod
    monkeypatch.setattr(storage_mod, "STORAGE_FILE", test_storage_file)

    # 1. Initial state is empty
    assert load_saved_portfolios() == {}

    # 2. Save a portfolio
    p_id = save_custom_portfolio(
        name="Mi Cartera Tech",
        tickers=["AAPL", "MSFT", "NVDA"],
        weights={"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2},
        description="Prueba unitaria de persistencia",
    )

    assert p_id is not None
    saved = load_saved_portfolios()
    assert p_id in saved
    assert saved[p_id]["name"] == "Mi Cartera Tech"
    assert saved[p_id]["tickers"] == ["AAPL", "MSFT", "NVDA"]
    assert abs(sum(saved[p_id]["weights"].values()) - 1.0) < 1e-4

    # 3. Export JSON
    json_export = export_portfolios_json()
    assert "Mi Cartera Tech" in json_export

    # 4. Delete portfolio
    del_ok = delete_custom_portfolio(p_id)
    assert del_ok is True
    assert load_saved_portfolios() == {}

    # 5. Import JSON
    imported_count = import_portfolios_json(json_export)
    assert imported_count == 1
    assert p_id in load_saved_portfolios()


def test_plot_portfolio_comparison_scatter():
    mock_data = {
        "Cartera A": {"return": 0.15, "volatility": 0.18, "sharpe": 0.61},
        "Cartera B": {"return": 0.10, "volatility": 0.08, "sharpe": 0.75},
    }
    fig = plot_portfolio_comparison_scatter(mock_data, rf=0.04)
    assert fig is not None
    assert len(fig.data) == 2
