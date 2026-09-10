"""
Tier 1 Unit Tests: Export Engine (CSV & Multi-Sheet Excel Workbook).
Verifies R5 export requirements, multi-sheet workbook generation, and comprehensive reporting.
"""

from __future__ import annotations

import io
import openpyxl
import pandas as pd
import pytest

# Dynamic imports for export module (Milestone 4)
try:
    from src.export.exporter import (
        export_correlation_csv,
        export_full_excel,
        export_summary_csv,
        export_wealth_series_csv,
        export_weights_csv,
        generate_excel_workbook,
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


def test_export_correlation_and_wealth_csv():
    """Verify correlation matrix and wealth series serialize to CSV."""
    corr_df = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"])
    assert "AAPL" in export_correlation_csv(corr_df)

    wealth_series = pd.Series([10000.0, 10200.0, 10150.0], name="Wealth")
    assert "Wealth" in export_wealth_series_csv(wealth_series)


# ===========================================================================
# 2. Multi-Sheet Excel Workbook Unit Tests
# ===========================================================================

def test_export_full_excel_sheets_and_formatting():
    """Verify multi-sheet Excel generator produces valid binary workbook with core sheets."""
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

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames

    assert "Configuración y Modelos" in sheet_names
    assert "Resumen de Métricas" in sheet_names
    assert "Ponderaciones" in sheet_names
    assert "Matriz de Correlación" in sheet_names
    assert "Matriz de Covarianza" in sheet_names


def test_export_full_excel_all_ten_sheets():
    """Verify comprehensive 10-sheet workbook generation when all data structures are supplied."""
    model_metadata = {
        "universo_tickers": ["AAPL", "MSFT", "GLD", "CASH"],
        "total_activos": 4,
        "fuente_datos": "Yahoo Finance (En Vivo)",
        "fecha_inicio": "2023-01-01",
        "fecha_fin": "2026-01-01",
        "dias_habiles_observados": 750,
        "tasa_libre_riesgo": 0.04,
        "benchmark": "SPY",
        "modelo_retornos": "Media Histórica",
        "modelo_retornos_desc": "Media histórica anualizada (252 días)",
        "modelo_covarianza": "Shrinkage Ledoit-Wolf",
        "modelo_covarianza_desc": "Shrinkage lineal con target correlación constante",
        "shrinkage_delta": 0.1420,
        "numero_condicion": 15.4,
        "autovalor_minimo": 0.0025,
        "reparacion_higham_psd": False,
        "tipo_posiciones": "Solo Posiciones Largas (Long-Only, w_i >= 0)",
        "peso_min_activo": 0.0,
        "peso_max_activo": 1.0,
        "peso_min_cash": 0.05,
        "peso_max_cash": 0.40,
    }

    frontier_data = {
        "optimal_points": [
            {"Punto": "Máximo Sharpe", "Volatilidad": 0.18, "Retorno": 0.22, "Sharpe": 1.00},
            {"Punto": "GMV", "Volatilidad": 0.10, "Retorno": 0.12, "Sharpe": 0.80},
        ],
        "individual_assets": [
            {"Activo": "AAPL", "Volatilidad": 0.25, "Retorno": 0.24, "Sharpe": 0.80},
        ],
        "frontier_curve": pd.DataFrame({
            "Nivel Volatilidad": [0.10, 0.15, 0.20],
            "Retorno Objetivo": [0.12, 0.18, 0.23],
            "Ratio Sharpe": [0.80, 0.93, 0.95],
        }),
        "cal_line": pd.DataFrame({
            "Nivel Volatilidad": [0.0, 0.18, 0.36],
            "Retorno CAL": [0.04, 0.22, 0.40],
        }),
    }

    diversification_data = {
        "summary_metrics": pd.DataFrame({
            "Indicador": ["Ratio Choueifaty (DR)", "ENC Sectores", "HHI Sectores"],
            "Cartera Usuario": [1.35, 3.2, 1850.0],
            "Máximo Sharpe": [1.42, 2.8, 2100.0],
        }),
        "sector_breakdown": pd.DataFrame({
            "Sector": ["Tecnología", "Metales", "Liquidez"],
            "Cartera Usuario": [0.50, 0.30, 0.20],
        }),
        "country_breakdown": pd.DataFrame({
            "País": ["Estados Unidos", "Global (USD)"],
            "Cartera Usuario": [0.80, 0.20],
        }),
        "asset_class_breakdown": pd.DataFrame({
            "Clase de Activo": ["Renta Variable", "Commodities", "Liquidez"],
            "Cartera Usuario": [0.50, 0.30, 0.20],
        }),
        "market_cap_breakdown": pd.DataFrame({
            "Escala Capitalización": ["Mega Cap", "N/A"],
            "Cartera Usuario": [0.80, 0.20],
        }),
        "asset_classification_table": pd.DataFrame({
            "Ticker": ["AAPL", "GLD", "CASH"],
            "Nombre": ["Apple Inc.", "SPDR Gold Trust", "Liquidez USD"],
            "Sector": ["Tecnología", "Metales", "Liquidez"],
        }),
    }

    wealth_df = pd.DataFrame({
        "Fecha": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Usuario ($)": [10000.0, 10100.0, 10200.0, 10150.0, 10300.0],
        "Benchmark ($)": [10000.0, 10050.0, 10080.0, 10020.0, 10100.0],
    })

    drawdown_df = pd.DataFrame({
        "Fecha": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Drawdown Usuario": [0.0, 0.0, 0.0, -0.005, 0.0],
    })

    monte_carlo_data = {
        "horizon_years": 3,
        "num_simulations": 2000,
        "model": "Movimiento Browniano Geométrico (GBM)",
        "terminal_scenarios": pd.DataFrame({
            "Escenario": ["P5 (Adverso)", "P50 (Mediana)", "P95 (Favorable)"],
            "Capital Proyectado": [9500.0, 15200.0, 24000.0],
        }),
    }

    comparator_data = {
        "summary_table": pd.DataFrame({
            "Portafolio": ["Cartera Usuario", "Máximo Sharpe", "Clásico 60/40"],
            "Retorno Anual": ["16.20%", "21.40%", "11.50%"],
            "Volatilidad": ["18.10%", "19.50%", "12.00%"],
            "Ratio Sharpe": ["0.674", "0.892", "0.625"],
        }),
    }

    excel_bytes = export_full_excel(
        metrics_dict={"Retorno Anualizado": {"Usuario": 0.16, "Max Sharpe": 0.21}},
        weights_dict={"AAPL": {"Usuario": 0.50, "Max Sharpe": 0.70}},
        corr_matrix=pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["AAPL", "GLD"], columns=["AAPL", "GLD"]),
        cov_matrix=pd.DataFrame([[0.06, 0.01], [0.01, 0.04]], index=["AAPL", "GLD"], columns=["AAPL", "GLD"]),
        wealth_df=wealth_df,
        drawdown_df=drawdown_df,
        model_metadata=model_metadata,
        diversification_data=diversification_data,
        frontier_data=frontier_data,
        monte_carlo_data=monte_carlo_data,
        comparator_data=comparator_data,
    )

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames

    # Check that all 10 specialized sheets exist
    expected_sheets = [
        "Configuración y Modelos",
        "Resumen de Métricas",
        "Ponderaciones",
        "Frontera Eficiente y CAL",
        "Diversificación Real",
        "Evolución Histórica",
        "Simulación Monte Carlo",
        "Matriz de Correlación",
        "Matriz de Covarianza",
        "Comparador Multi-Portafolio",
    ]

    for expected in expected_sheets:
        assert expected in sheet_names, f"Missing sheet: {expected}"
