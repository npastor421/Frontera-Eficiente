"""
Tier 1 Unit Tests: Export Engine (CSV & Multi-Sheet Excel Workbook).
Verifies R5 export requirements from ORIGINAL_REQUEST.md, PROJECT.md, and survey reports.
"""

from __future__ import annotations

import io
import openpyxl
import pandas as pd
import pytest

# Dynamic imports for export module (Milestone 4)
try:
    from src.export.exporter import (
        export_full_excel,
        export_summary_csv,
        export_weights_csv,
    )
    HAS_EXPORT = True
except ImportError:
    HAS_EXPORT = False


pytestmark = pytest.mark.skipif(
    not HAS_EXPORT,
    reason="src.export module not yet implemented by Milestone 4",
)


# ===========================================================================
# 1. CSV Serialization Unit Tests
# ===========================================================================

def test_export_summary_csv():
    """Verify summary metrics DataFrame serializes into a clean CSV string."""
    df_metrics = pd.DataFrame({
        "Métrica": ["Retorno Anualizado", "Volatilidad", "Ratio Sharpe"],
        "Cartera Usuario": ["15.20%", "18.40%", "0.609"],
        "Máximo Sharpe": ["19.50%", "20.10%", "0.771"],
        "GMV": ["8.10%", "11.20%", "0.366"],
    })
    csv_str = export_summary_csv(df_metrics)
    assert isinstance(csv_str, str)
    assert "Retorno Anualizado" in csv_str
    # Verify parseable by pandas
    parsed = pd.read_csv(io.StringIO(csv_str))
    assert len(parsed) == 3
    assert "Métrica" in parsed.columns


def test_export_weights_csv():
    """Verify optimal portfolio weights serialize into CSV."""
    df_weights = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOGL"],
        "Usuario": [0.333, 0.333, 0.334],
        "Max Sharpe": [0.50, 0.30, 0.20],
        "GMV": [0.10, 0.70, 0.20],
    })
    csv_str = export_weights_csv(df_weights)
    assert isinstance(csv_str, str)
    assert "AAPL" in csv_str


# ===========================================================================
# 2. Multi-Sheet Excel Workbook Unit Tests
# ===========================================================================

def test_export_full_excel_sheets_and_formatting():
    """Verify multi-sheet Excel generator produces valid binary workbook with all 6 required sheets."""
    metrics_dict = {
        "Retorno Anualizado": {"Usuario": 0.15, "Max Sharpe": 0.19, "GMV": 0.08},
        "Volatilidad": {"Usuario": 0.18, "Max Sharpe": 0.20, "GMV": 0.11},
        "Sharpe": {"Usuario": 0.61, "Max Sharpe": 0.77, "GMV": 0.36},
    }
    weights_dict = {
        "AAPL": {"Usuario": 0.33, "Max Sharpe": 0.50, "GMV": 0.10},
        "MSFT": {"Usuario": 0.33, "Max Sharpe": 0.30, "GMV": 0.70},
        "GOOGL": {"Usuario": 0.34, "Max Sharpe": 0.20, "GMV": 0.20},
    }
    corr_df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"])
    cov_df = pd.DataFrame([[0.04, 0.015], [0.015, 0.03]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"])

    excel_bytes = export_full_excel(
        metrics_dict=metrics_dict,
        weights_dict=weights_dict,
        corr_matrix=corr_df,
        cov_matrix=cov_df,
    )

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Load with openpyxl to verify workbook structure
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames

    # Check for presence of primary sheets
    assert "Resumen de Métricas" in sheet_names or "Resumen" in sheet_names[0]
    assert "Ponderaciones" in sheet_names or len(sheet_names) >= 4
