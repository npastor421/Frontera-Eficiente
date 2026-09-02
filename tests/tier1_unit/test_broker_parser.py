"""
Unit Tests for Broker Holdings Report Parser and Ticker Normalizer.
"""

import io
from pathlib import Path
import pandas as pd
import pytest

from src.data.broker_parser import (
    parse_broker_holdings,
    normalize_broker_ticker,
    BrokerHoldingsReport,
)


def test_normalize_broker_ticker_rules():
    """Verify ticker normalization for CEDEARs, local stocks, funds, and cash."""
    # Global USD mode
    assert normalize_broker_ticker("YPFD", instrument_type="Acciones", mode="global_usd") == "YPF"
    assert normalize_broker_ticker("GGAL", instrument_type="Acciones", mode="global_usd") == "GGAL"
    assert normalize_broker_ticker("ITUB3", instrument_type="Cedears", mode="global_usd") == "ITUB"
    assert normalize_broker_ticker("GOOGL", instrument_type="Cedears", mode="global_usd") == "GOOGL"
    assert normalize_broker_ticker("PESOS DISPONIBLES", instrument_type="Caja") == "CASH"
    assert normalize_broker_ticker("BALMM", instrument_type="Fondos Comunes de Inversión") == "CASH"
    assert normalize_broker_ticker("AL30", instrument_type="Bonos") == "AL30.BA"

    # BYMA ARS mode
    assert normalize_broker_ticker("YPFD", instrument_type="Acciones", mode="byma_ars") == "YPFD.BA"
    assert normalize_broker_ticker("GOOGL", instrument_type="Cedears", mode="byma_ars") == "GOOGL.BA"
    assert normalize_broker_ticker("ITUB3", instrument_type="Cedears", mode="byma_ars") == "ITUB.BA"


def test_parse_real_broker_excel_file():
    """Verify parsing of the actual uploaded broker holdings report."""
    test_files = list(Path("Cartera prueba").glob("*.xlsx"))
    if not test_files:
        pytest.skip("No broker test file in Cartera prueba directory")

    excel_path = test_files[0]
    report = parse_broker_holdings(excel_path, mode="global_usd")

    assert isinstance(report, BrokerHoldingsReport)
    assert report.instruments_count == 14
    assert len(report.tickers) == 14

    # Verify expected normalized tickers in global mode
    expected_tickers = {
        "YPF", "GOOGL", "ITUB", "JPM", "MELI", "META", "MSFT",
        "MU", "NBIS", "NU", "NVDA", "RACE", "TSM", "V"
    }
    assert set(report.tickers) == expected_tickers

    # Verify weights sum exactly to 100.0% and 1.0
    assert abs(sum(report.weights_pct.values()) - 100.0) < 1e-4
    assert abs(sum(report.weights.values()) - 1.0) < 1e-4

    # Verify breakdown
    assert "Cedears" in report.by_type_breakdown
    assert report.by_type_breakdown["Cedears"] == 13
    assert report.by_type_breakdown.get("Acciones", 0) == 1


def test_parse_broker_with_cash_and_funds():
    """Verify parsing synthetic report with mixed assets including funds and cash."""
    data = {
        "Ticker": ["AAPL", "GGAL", "FCI_AHORRO", "SALDO DISPONIBLE"],
        "Tipo de Instrumento": ["Cedears", "Acciones", "Fondos", "Caja"],
        "Porcentaje de tenencia": [40.0, 30.0, 20.0, 10.0],
        "Valor actual": [400000, 300000, 200000, 100000],
    }
    df = pd.DataFrame(data)
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False)
    excel_buf.seek(0)

    report = parse_broker_holdings(excel_buf, filename="holdings.xlsx", mode="global_usd")

    # AAPL, GGAL, CASH (funds + cash consolidated)
    assert "AAPL" in report.tickers
    assert "GGAL" in report.tickers
    assert "CASH" in report.tickers
    assert report.weights_pct["CASH"] == 30.0
    assert abs(sum(report.weights_pct.values()) - 100.0) < 1e-4
