"""
Unit Tests for Multi-Tenant Isolated Portfolio Storage.
"""

import json
from pathlib import Path
import pytest

from src.presets.storage import (
    load_saved_portfolios,
    save_custom_portfolio,
    delete_custom_portfolio,
    export_portfolios_json,
    import_portfolios_json,
    get_storage_path,
    _sanitize_user_id,
)


def test_sanitize_user_id():
    """Verify email sanitization produces valid filenames."""
    assert _sanitize_user_id("user@gmail.com") == "user_at_gmail.com"
    assert _sanitize_user_id(None) == "guest"
    assert _sanitize_user_id("") == "guest"
    assert _sanitize_user_id("guest") == "guest"
    assert _sanitize_user_id("John.Doe+crypto@firm.co.uk") == "john.doe_crypto_at_firm.co.uk"


def test_multi_user_isolation(tmp_path, monkeypatch):
    """Verify User A and User B have separate storage and cannot see each other's portfolios."""
    monkeypatch.setattr("src.presets.storage.PORTFOLIOS_DIR", tmp_path)
    monkeypatch.setattr("src.presets.storage.LEGACY_STORAGE_FILE", tmp_path / "non_existent_legacy.json")

    # User A creates portfolio
    p_id_a = save_custom_portfolio(
        name="Cartera Alfa",
        tickers=["SPY", "TLT"],
        weights={"SPY": 0.6, "TLT": 0.4},
        user_id="user_a@test.com",
    )

    # User B creates portfolio
    p_id_b = save_custom_portfolio(
        name="Cartera Beta",
        tickers=["AAPL", "MSFT", "NVDA"],
        weights={"AAPL": 0.33, "MSFT": 0.33, "NVDA": 0.34},
        user_id="user_b@test.com",
    )

    # User A loads portfolios
    user_a_ports = load_saved_portfolios("user_a@test.com")
    assert p_id_a in user_a_ports
    assert p_id_b not in user_a_ports
    assert user_a_ports[p_id_a]["name"] == "Cartera Alfa"

    # User B loads portfolios
    user_b_ports = load_saved_portfolios("user_b@test.com")
    assert p_id_b in user_b_ports
    assert p_id_a not in user_b_ports
    assert user_b_ports[p_id_b]["name"] == "Cartera Beta"

    # Guest loads portfolios -> empty when no legacy file
    guest_ports = load_saved_portfolios("guest")
    assert len(guest_ports) == 0

    # User A deletes their portfolio
    assert delete_custom_portfolio(p_id_a, user_id="user_a@test.com") is True
    assert len(load_saved_portfolios("user_a@test.com")) == 0
    # User B's portfolio remains untouched
    assert len(load_saved_portfolios("user_b@test.com")) == 1
