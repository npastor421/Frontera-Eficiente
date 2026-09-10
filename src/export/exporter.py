"""
Export Engine Module for Frontera Eficiente.

Provides comprehensive serialization and institutional-grade multi-sheet Excel reporting:
1. CSV Exporters:
   - Summary Metrics (`export_summary_csv`, `export_metrics_csv`)
   - Asset Allocations (`export_weights_csv`)
   - Correlation Matrix (`export_correlation_csv`)
   - Wealth Time Series (`export_wealth_series_csv`)
2. Multi-Sheet Styled Excel Workbook (`export_full_excel`, `generate_excel_workbook`):
   - Sheet 1: ⚙️ Configuración y Modelos (Parámetros, Metodología, Restricciones y Diagnósticos)
   - Sheet 2: 📑 Resumen y Métricas (Rendimiento, Riesgo, Ratios, MDD, VaR/CVaR multihorizonte)
   - Sheet 3: 🍩 Ponderaciones y Activos (Pesos por cartera, estadísticas individuales por activo, fórmula =SUM())
   - Sheet 4: 📈 Frontera Eficiente y CAL (Puntos óptimos, sweep continuo de Markowitz, CAL)
   - Sheet 5: 🌐 Diversificación Real (Choueifaty DR, ENC, HHI, desgloses sectorial, geográfico, clase, cap)
   - Sheet 6: 💰 Backtest y Drawdown (Serie temporal histórica $10,000 USD y Drawdown día a día)
   - Sheet 7: 🔮 Proyección Monte Carlo (Parámetros estocásticos, conos de percentiles y escenarios futuros)
   - Sheet 8: 🧊 Matriz de Correlación (Correlación lineal de Pearson N x N)
   - Sheet 9: 🧊 Matriz de Covarianza (Covarianza anualizada estimada según el modelo)
   - Sheet 10: ⚖️ Comparador Multi-Portafolio (Comparativa de KPIs y pesos de múltiples carteras)
   - Formato institucional: Navy Header (`#1F4E79`), Zebra striping (`#FFFFFF` y `#F8F9FA`), bordes finos,
     formatos numéricos explícitos (0.00%, 0.000, $#,##0.00) y auto-ajuste de columnas.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd


# ===========================================================================
# 1. CSV Serialization Helpers
# ===========================================================================

def export_summary_csv(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> str:
    """Export summary metrics table to CSV formatted string."""
    if isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict
    elif isinstance(df_or_dict, dict):
        df = _convert_metrics_dict_to_df(df_or_dict)
    else:
        raise TypeError(f"Expected pd.DataFrame or dict, got {type(df_or_dict)}")

    return df.to_csv(index=False)


def export_metrics_csv(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> str:
    """Alias for export_summary_csv."""
    return export_summary_csv(df_or_dict)


def export_weights_csv(df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> str:
    """Export asset weights allocation table to CSV formatted string."""
    if isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict
    elif isinstance(df_or_dict, dict):
        df = _convert_weights_dict_to_df(df_or_dict)
    else:
        raise TypeError(f"Expected pd.DataFrame or dict, got {type(df_or_dict)}")

    return df.to_csv(index=False)


def export_correlation_csv(corr_df: Union[pd.DataFrame, np.ndarray], tickers: Optional[List[str]] = None) -> str:
    """Export correlation matrix to CSV formatted string."""
    if isinstance(corr_df, np.ndarray):
        cols = tickers if tickers else [f"Asset_{i+1}" for i in range(corr_df.shape[0])]
        df = pd.DataFrame(corr_df, index=cols, columns=cols)
    else:
        df = corr_df

    return df.to_csv(index=True)


def export_wealth_series_csv(wealth_df: Union[pd.DataFrame, pd.Series]) -> str:
    """Export historical wealth series to CSV formatted string."""
    if isinstance(wealth_df, pd.Series):
        df = wealth_df.to_frame()
    else:
        df = wealth_df

    return df.to_csv(index=True)


# ===========================================================================
# 2. DataFrame Transformation Helpers
# ===========================================================================

_METRIC_LABELS_ES = {
    "annualized_return": "Retorno Anualizado (Aritmético)",
    "cagr": "Retorno Compuesto Anual (CAGR)",
    "annualized_volatility": "Volatilidad Anualizada (Riesgo)",
    "sharpe_ratio": "Ratio de Sharpe",
    "sortino_ratio": "Ratio de Sortino",
    "calmar_ratio": "Ratio de Calmar",
    "beta": "Beta (β vs Benchmark)",
    "alpha_jensen": "Alfa de Jensen Anualizada (α)",
    "r_squared": "Coeficiente de Determinación (R²)",
    "max_drawdown": "Máximo Drawdown (MDD)",
    "var_95_hist": "VaR 95% Histórico (1 Día)",
    "cvar_95_hist": "CVaR 95% Histórico (1 Día)",
    "var_95_monthly": "VaR 95% Mensual (21 Días Hábiles)",
    "cvar_95_monthly": "CVaR 95% Mensual (21 Días Hábiles)",
    "var_95_annual": "VaR 95% Anual (252 Días Hábiles)",
    "cvar_95_annual": "CVaR 95% Anual (252 Días Hábiles)",
    "var_95_param": "VaR 95% Paramétrico (1 Día)",
    "cvar_95_param": "CVaR 95% Paramétrico (1 Día)",
    "recovery_days": "Días de Recuperación de Drawdown",
}


def _convert_metrics_dict_to_df(metrics_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Convert arbitrary nested metrics dictionary or DataFrame into standard summary DataFrame."""
    if isinstance(metrics_input, pd.DataFrame):
        return metrics_input.copy()

    if not metrics_input:
        return pd.DataFrame(columns=["Métrica"])

    first_key = next(iter(metrics_input.keys()))
    first_val = metrics_input[first_key]

    # Structure 1: Metric -> {Portfolio: Value} e.g. {"Retorno": {"Usuario": 0.15, "Max Sharpe": 0.19}}
    if isinstance(first_val, dict) and not any(k in _METRIC_LABELS_ES for k in first_val.keys()):
        rows = []
        for metric_name, port_map in metrics_input.items():
            row = {"Métrica": metric_name}
            row.update(port_map)
            rows.append(row)
        return pd.DataFrame(rows)

    # Structure 2: Portfolio -> MetricsObj or {metric_name: val} e.g. {"Max Sharpe": {...}, "GMV": {...}}
    portfolios = list(metrics_input.keys())
    all_metric_keys = []
    for port in portfolios:
        val_obj = metrics_input[port]
        keys = val_obj.keys() if hasattr(val_obj, "keys") else (val_obj.to_dict().keys() if hasattr(val_obj, "to_dict") else [])
        for k in keys:
            if k not in ("daily_returns", "cumulative_wealth", "drawdown_series") and k not in all_metric_keys:
                all_metric_keys.append(k)

    if not all_metric_keys:
        all_metric_keys = list(_METRIC_LABELS_ES.keys())

    rows = []
    for m_key in all_metric_keys:
        label = _METRIC_LABELS_ES.get(m_key, str(m_key))
        row = {"Métrica": label}
        for port in portfolios:
            val_obj = metrics_input[port]
            if hasattr(val_obj, "__getitem__") and m_key in val_obj:
                row[port] = val_obj[m_key]
            elif hasattr(val_obj, m_key):
                row[port] = getattr(val_obj, m_key)
            else:
                row[port] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _convert_weights_dict_to_df(weights_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """Convert arbitrary weights dictionary or DataFrame into standard weights DataFrame."""
    if isinstance(weights_input, pd.DataFrame):
        if "Ticker" not in weights_input.columns and weights_input.index.name != "Ticker":
            df = weights_input.reset_index()
            if "index" in df.columns:
                df = df.rename(columns={"index": "Ticker"})
            return df
        return weights_input.copy()

    if not weights_input:
        return pd.DataFrame(columns=["Ticker"])

    first_key = next(iter(weights_input.keys()))
    first_val = weights_input[first_key]

    # Structure 1: Ticker -> {Portfolio: weight} e.g. {"AAPL": {"Usuario": 0.33, "Max Sharpe": 0.50}}
    if isinstance(first_val, dict):
        rows = []
        for ticker, port_map in weights_input.items():
            row = {"Ticker": ticker}
            row.update(port_map)
            rows.append(row)
        return pd.DataFrame(rows)

    # Structure 2: Portfolio -> {Ticker: weight} e.g. {"Usuario": {"AAPL": 0.33, "MSFT": 0.33}}
    if isinstance(first_val, (dict, pd.Series)):
        all_tickers = []
        for port, w_dict in weights_input.items():
            for t in w_dict.keys():
                if t not in all_tickers:
                    all_tickers.append(t)

        rows = []
        for ticker in all_tickers:
            row = {"Ticker": ticker}
            for port, w_dict in weights_input.items():
                row[port] = w_dict.get(ticker, 0.0) if isinstance(w_dict, dict) else (w_dict[ticker] if ticker in w_dict else 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    # Flat dictionary: Ticker -> weight
    rows = [{"Ticker": k, "Peso": v} for k, v in weights_input.items()]
    return pd.DataFrame(rows)


# ===========================================================================
# 3. Multi-Sheet Styled Excel Workbook Generator
# ===========================================================================

def export_full_excel(
    metrics_dict: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    weights_dict: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
    wealth_df: Optional[Union[pd.DataFrame, pd.Series]] = None,
    mc_samples_df: Optional[pd.DataFrame] = None,
    model_metadata: Optional[Dict[str, Any]] = None,
    diversification_data: Optional[Dict[str, Any]] = None,
    frontier_data: Optional[Dict[str, Any]] = None,
    drawdown_df: Optional[pd.DataFrame] = None,
    monte_carlo_data: Optional[Dict[str, Any]] = None,
    comparator_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate an institutional-grade, multi-sheet formatted Excel workbook (.xlsx)
    encompassing 100% of data, quantitative models, constraints, and visualizations
    from all dashboard tabs.

    Sheets:
    1. ⚙️ Configuración y Modelos (Metodología, estimadores, diagnósticos y restricciones)
    2. 📑 Resumen y Métricas (Rendimiento, riesgo y VaR/CVaR multihorizonte)
    3. 🍩 Ponderaciones y Activos (Ponderaciones + estadísticas de activos individuales + =SUM())
    4. 📈 Frontera Eficiente y CAL (Puntos cardinales óptimos, curva continua de Markowitz y CAL)
    5. 🌐 Diversificación Real (Choueifaty DR, ENC, HHI, entropía y desgloses sectorial, geográfico, clase y cap)
    6. 💰 Backtest y Drawdown (Series temporales históricas de riqueza $10,000 USD y Drawdown)
    7. 🔮 Proyección Monte Carlo (Parámetros, conos de probabilidad y percentiles de capital futuro)
    8. 🧊 Matriz de Correlación (Correlación lineal de Pearson)
    9. 🧊 Matriz de Covarianza (Covarianza anualizada)
    10. ⚖️ Comparador Multi-Portafolio (Comparativa integral entre carteras, presets y portafolios guardados)

    Returns
    -------
    bytes
        In-memory Excel binary workbook.
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    # Visual Theme Styles
    navy_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    section_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    zebra_fill_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    zebra_fill_gray = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
    total_fill = PatternFill(start_color="E9EEF4", end_color="E9EEF4", fill_type="solid")
    banner_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    banner_font = Font(name="Calibri", size=11, bold=True, color="1F4E79")
    bold_font = Font(name="Calibri", size=10, bold=True, color="000000")
    regular_font = Font(name="Calibri", size=10, color="000000")
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E79")
    subtitle_font = Font(name="Calibri", size=9, italic=True, color="595959")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    thick_bottom_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="double", color="1F4E79"),
    )

    styles = {
        "navy_fill": navy_fill,
        "section_fill": section_fill,
        "zebra_fill_white": zebra_fill_white,
        "zebra_fill_gray": zebra_fill_gray,
        "total_fill": total_fill,
        "banner_fill": banner_fill,
        "header_font": header_font,
        "banner_font": banner_font,
        "bold_font": bold_font,
        "regular_font": regular_font,
        "title_font": title_font,
        "subtitle_font": subtitle_font,
        "thin_border": thin_border,
        "thick_bottom_border": thick_bottom_border,
    }

    # -----------------------------------------------------------------------
    # Sheet 1: ⚙️ Configuración y Modelos
    # -----------------------------------------------------------------------
    ws_config = wb.create_sheet(title="Configuración y Modelos")
    _write_config_sheet(ws=ws_config, metadata=model_metadata, styles=styles)

    # -----------------------------------------------------------------------
    # Sheet 2: 📑 Resumen de Métricas
    # -----------------------------------------------------------------------
    ws_metrics = wb.create_sheet(title="Resumen de Métricas")
    if metrics_dict is not None:
        df_m = _convert_metrics_dict_to_df(metrics_dict)
    else:
        df_m = pd.DataFrame({"Métrica": ["Sin datos"]})

    _write_styled_dataframe(
        ws=ws_metrics,
        df=df_m,
        navy_fill=navy_fill,
        header_font=header_font,
        zebra_fill_white=zebra_fill_white,
        zebra_fill_gray=zebra_fill_gray,
        thin_border=thin_border,
        regular_font=regular_font,
        is_metrics_sheet=True,
    )

    # -----------------------------------------------------------------------
    # Sheet 3: 🍩 Ponderaciones y Activos
    # -----------------------------------------------------------------------
    ws_weights = wb.create_sheet(title="Ponderaciones")
    if weights_dict is not None:
        df_w = _convert_weights_dict_to_df(weights_dict)
    else:
        df_w = pd.DataFrame({"Ticker": ["Sin datos"], "Peso": [0.0]})

    _write_styled_dataframe(
        ws=ws_weights,
        df=df_w,
        navy_fill=navy_fill,
        header_font=header_font,
        zebra_fill_white=zebra_fill_white,
        zebra_fill_gray=zebra_fill_gray,
        thin_border=thin_border,
        regular_font=regular_font,
        is_weights_sheet=True,
        total_fill=total_fill,
        bold_font=bold_font,
        thick_bottom_border=thick_bottom_border,
    )

    # -----------------------------------------------------------------------
    # Sheet 4: 📈 Frontera Eficiente y CAL
    # -----------------------------------------------------------------------
    ws_frontier = wb.create_sheet(title="Frontera Eficiente y CAL")
    _write_frontier_sheet(ws=ws_frontier, frontier_data=frontier_data, styles=styles)

    # -----------------------------------------------------------------------
    # Sheet 5: 🌐 Diversificación Real
    # -----------------------------------------------------------------------
    ws_div = wb.create_sheet(title="Diversificación Real")
    _write_diversification_sheet(ws=ws_div, div_data=diversification_data, styles=styles)

    # -----------------------------------------------------------------------
    # Sheet 6: 💰 Backtest y Drawdown
    # -----------------------------------------------------------------------
    ws_hist = wb.create_sheet(title="Evolución Histórica")
    _write_backtest_sheet(ws=ws_hist, wealth_df=wealth_df, drawdown_df=drawdown_df, styles=styles)

    # -----------------------------------------------------------------------
    # Sheet 7: 🔮 Proyección Monte Carlo
    # -----------------------------------------------------------------------
    ws_mc = wb.create_sheet(title="Simulación Monte Carlo")
    _write_monte_carlo_sheet(ws=ws_mc, mc_data=monte_carlo_data, mc_samples_df=mc_samples_df, styles=styles)

    # -----------------------------------------------------------------------
    # Sheet 8: 🧊 Matriz de Correlación
    # -----------------------------------------------------------------------
    ws_corr = wb.create_sheet(title="Matriz de Correlación")
    if corr_matrix is not None:
        df_corr = corr_matrix.reset_index().rename(columns={"index": "Activo"}) if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame(corr_matrix)
    else:
        df_corr = pd.DataFrame({"Activo": ["N/A"]})

    _write_styled_dataframe(
        ws=ws_corr,
        df=df_corr,
        navy_fill=navy_fill,
        header_font=header_font,
        zebra_fill_white=zebra_fill_white,
        zebra_fill_gray=zebra_fill_gray,
        thin_border=thin_border,
        regular_font=regular_font,
        number_format="0.0000",
    )

    # -----------------------------------------------------------------------
    # Sheet 9: 🧊 Matriz de Covarianza
    # -----------------------------------------------------------------------
    ws_cov = wb.create_sheet(title="Matriz de Covarianza")
    if cov_matrix is not None:
        df_cov = cov_matrix.reset_index().rename(columns={"index": "Activo"}) if isinstance(cov_matrix, pd.DataFrame) else pd.DataFrame(cov_matrix)
    else:
        df_cov = pd.DataFrame({"Activo": ["N/A"]})

    _write_styled_dataframe(
        ws=ws_cov,
        df=df_cov,
        navy_fill=navy_fill,
        header_font=header_font,
        zebra_fill_white=zebra_fill_white,
        zebra_fill_gray=zebra_fill_gray,
        thin_border=thin_border,
        regular_font=regular_font,
        number_format="0.000000",
    )

    # -----------------------------------------------------------------------
    # Sheet 10: ⚖️ Comparador Multi-Portafolio
    # -----------------------------------------------------------------------
    if comparator_data is not None:
        ws_comp = wb.create_sheet(title="Comparador Multi-Portafolio")
        _write_comparator_sheet(ws=ws_comp, comparator_data=comparator_data, styles=styles)

    # Serialize to memory buffer
    output_buf = io.BytesIO()
    wb.save(output_buf)
    return output_buf.getvalue()


def generate_excel_workbook(
    metrics_df: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    weights_df: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    corr_df: Optional[pd.DataFrame] = None,
    cov_df: Optional[pd.DataFrame] = None,
    wealth_df: Optional[Union[pd.DataFrame, pd.Series]] = None,
    mc_samples_df: Optional[pd.DataFrame] = None,
    model_metadata: Optional[Dict[str, Any]] = None,
    diversification_data: Optional[Dict[str, Any]] = None,
    frontier_data: Optional[Dict[str, Any]] = None,
    drawdown_df: Optional[pd.DataFrame] = None,
    monte_carlo_data: Optional[Dict[str, Any]] = None,
    comparator_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Alias matching canonical parameter names for multi-sheet Excel generator."""
    return export_full_excel(
        metrics_dict=metrics_df,
        weights_dict=weights_df,
        corr_matrix=corr_df,
        cov_matrix=cov_df,
        wealth_df=wealth_df,
        mc_samples_df=mc_samples_df,
        model_metadata=model_metadata,
        diversification_data=diversification_data,
        frontier_data=frontier_data,
        drawdown_df=drawdown_df,
        monte_carlo_data=monte_carlo_data,
        comparator_data=comparator_data,
    )


# ===========================================================================
# 4. Sheet Builders for Specialized Sections
# ===========================================================================

def _write_config_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, metadata: Optional[Dict[str, Any]], styles: Dict[str, Any]) -> None:
    """Build Sheet 1: Configuration, Statistical Models, Numerical Diagnostics & Constraints."""
    ws.views.sheetView[0].showGridLines = True
    meta = metadata or {}

    # Header title banner
    ws.cell(row=1, column=1, value="INFORME CUANTITATIVO DE OPTIMIZACIÓN — FRONTERA EFICIENTE").font = styles["title_font"]
    ws.cell(row=2, column=1, value="Especificación Metodológica de Modelos Estadísticos, Restricciones de Inversión y Parámetros").font = styles["subtitle_font"]

    row_cursor = 4

    # 1. General Portfolio Parameters
    df_gen = pd.DataFrame([
        {"Parámetro General": "Fecha de Generación del Informe", "Valor": meta.get("fecha_generacion", "Automática")},
        {"Parámetro General": "Universo de Inversión (Tickers)", "Valor": ", ".join(meta.get("universo_tickers", [])) if meta.get("universo_tickers") else "N/A"},
        {"Parámetro General": "Número Total de Activos (N)", "Valor": meta.get("total_activos", len(meta.get("universo_tickers", [])))},
        {"Parámetro General": "Rango Temporal Histórico", "Valor": f"{meta.get('fecha_inicio', 'N/A')} a {meta.get('fecha_fin', 'N/A')}"},
        {"Parámetro General": "Observaciones Diarias (Días Hábiles)", "Valor": meta.get("dias_habiles_observados", "N/A")},
        {"Parámetro General": "Fuente de Ingesta de Datos", "Valor": meta.get("fuente_datos", "Yahoo Finance (En Vivo)")},
        {"Parámetro General": "Tasa Libre de Riesgo (Rf)", "Valor": f"{meta.get('tasa_libre_riesgo', 0.04):.2%}"},
        {"Parámetro General": "Activo Benchmark de Mercado", "Valor": meta.get("benchmark", "SPY")},
        {"Parámetro General": "Frecuencia de Negociación", "Valor": "Diaria (252 días bursátiles anualizados)"},
    ])
    row_cursor = _write_section_table(ws, df_gen, "1. PARÁMETROS GENERALES Y UNIVERSO DE ACTIVOS", row_cursor, styles)

    # 2. Statistical Models and Estimators
    df_models = pd.DataFrame([
        {"Modelo / Estimador": "Estimador de Retornos Esperados (μ)", "Especificación / Parámetro": meta.get("modelo_retornos", "Media Histórica"), "Detalle Metodológico": meta.get("modelo_retornos_desc", "Rendimientos medios históricos anualizados por 252 días bursátiles")},
        {"Modelo / Estimador": "Estimador de Matriz de Covarianza (Σ)", "Especificación / Parámetro": meta.get("modelo_covarianza", "Ledoit-Wolf Shrinkage"), "Detalle Metodológico": meta.get("modelo_covarianza_desc", "Contracción lineal hacia target de correlación constante (Ledoit & Wolf 2004) para minimizar error de muestreo fuera de muestra")},
        {"Modelo / Estimador": "Intensidad de Contracción Shrinkage (δ*)", "Especificación / Parámetro": f"{meta.get('shrinkage_delta', 0.0):.4f}" if isinstance(meta.get('shrinkage_delta'), (int, float)) else str(meta.get('shrinkage_delta', 'N/A')), "Detalle Metodológico": "Ponderación óptima de la matriz target frente a la muestral; δ* ∈ [0, 1]"},
        {"Modelo / Estimador": "Número de Condición Matricial κ(Σ)", "Especificación / Parámetro": f"{meta.get('numero_condicion', 1.0):.2f}" if isinstance(meta.get('numero_condicion'), (int, float)) else str(meta.get('numero_condicion', 'N/A')), "Detalle Metodológico": "Razón λ_max / λ_min. Mide la estabilidad numérica ante perturbaciones (κ < 100 indica alta estabilidad)"},
        {"Modelo / Estimador": "Autovalor Mínimo (λ_min)", "Especificación / Parámetro": f"{meta.get('autovalor_minimo', 0.0):.6f}" if isinstance(meta.get('autovalor_minimo'), (int, float)) else str(meta.get('autovalor_minimo', 'N/A')), "Detalle Metodológico": "Garantiza condición semidefinida positiva estricta (PSD, λ_min > 0)"},
        {"Modelo / Estimador": "Reparación Higham PSD (1988)", "Especificación / Parámetro": "Aplicada ✅" if meta.get("reparacion_higham_psd") else "No requerida", "Detalle Metodológico": "Proyección por alternancia espectral a la matriz semi-definida positiva más cercana"},
    ])
    row_cursor = _write_section_table(ws, df_models, "2. MODELOS MATEMÁTICOS Y DIAGNÓSTICOS NUMÉRICOS", row_cursor, styles)

    # 3. Investment Constraints
    df_constraints = pd.DataFrame([
        {"Restricción Cuantitativa": "Régimen de Inversión", "Valor Aplicado": meta.get("tipo_posiciones", "Solo Posiciones Largas (Long-Only, w_i ≥ 0)")},
        {"Restricción Cuantitativa": "Restricción Presupuestaria de Capital", "Valor Aplicado": "100.00% (Suma exacta de ponderaciones Σ w_i = 1.0)"},
        {"Restricción Cuantitativa": "Ponderación Mínima por Activo (w_min)", "Valor Aplicado": f"{meta.get('peso_min_activo', 0.0):.2%}" if isinstance(meta.get('peso_min_activo'), (int, float)) else str(meta.get('peso_min_activo', '0.00%'))},
        {"Restricción Cuantitativa": "Ponderación Máxima por Activo (w_max)", "Valor Aplicado": f"{meta.get('peso_max_activo', 1.0):.2%}" if isinstance(meta.get('peso_max_activo'), (int, float)) else str(meta.get('peso_max_activo', '100.00%'))},
        {"Restricción Cuantitativa": "Piso Mínimo para CASH / Liquidez", "Valor Aplicado": f"{meta.get('peso_min_cash', 0.05):.2%}" if isinstance(meta.get('peso_min_cash'), (int, float)) else str(meta.get('peso_min_cash', '5.00%'))},
        {"Restricción Cuantitativa": "Techo Máximo para CASH / Liquidez", "Valor Aplicado": f"{meta.get('peso_max_cash', 0.40):.2%}" if isinstance(meta.get('peso_max_cash'), (int, float)) else str(meta.get('peso_max_cash', '40.00%'))},
    ])
    _write_section_table(ws, df_constraints, "3. RESTRICCIONES DE OPTIMIZACIÓN Y ASIGNACIÓN DE CAPITAL", row_cursor, styles)

    _autofit_columns(ws)


def _write_frontier_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, frontier_data: Optional[Dict[str, Any]], styles: Dict[str, Any]) -> None:
    """Build Sheet 4: Efficient Frontier, Capital Allocation Line (CAL) and Optimal Points."""
    ws.views.sheetView[0].showGridLines = True
    fdata = frontier_data or {}

    ws.cell(row=1, column=1, value="FRONTERA EFICIENTE DE MARKOWITZ Y LÍNEA DE ASIGNACIÓN DE CAPITAL (CAL)").font = styles["title_font"]
    ws.cell(row=2, column=1, value="Coordenadas óptimas de tangencia, mínima varianza y barrido de la curva continua").font = styles["subtitle_font"]

    row_cursor = 4

    # 1. Optimal Portfolios Coordinates
    opt_points = fdata.get("optimal_points")
    if opt_points:
        df_opt = pd.DataFrame(opt_points)
        row_cursor = _write_section_table(ws, df_opt, "1. CARTERAS ÓPTIMAS Y COORDENADAS RIESGO-RETORNO", row_cursor, styles)

    # 2. Individual Assets Coordinates
    ind_assets = fdata.get("individual_assets")
    if ind_assets:
        df_ind = pd.DataFrame(ind_assets)
        row_cursor = _write_section_table(ws, df_ind, "2. COORDENADAS DE ACTIVOS INDIVIDUALES", row_cursor, styles)

    # 3. Continuous Frontier Curve Sweep
    curve_df = fdata.get("frontier_curve")
    if curve_df is not None and isinstance(curve_df, pd.DataFrame) and len(curve_df) > 0:
        row_cursor = _write_section_table(ws, curve_df.head(100), "3. BARRIDO DISCRETO DE LA FRONTERA EFICIENTE DE MARKOWITZ", row_cursor, styles)

    # 4. CAL Line Points
    cal_df = fdata.get("cal_line")
    if cal_df is not None and isinstance(cal_df, pd.DataFrame) and len(cal_df) > 0:
        _write_section_table(ws, cal_df.head(50), "4. PUNTOS DE LA LÍNEA DE ASIGNACIÓN DE CAPITAL (CAL)", row_cursor, styles)

    _autofit_columns(ws)


def _write_diversification_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, div_data: Optional[Dict[str, Any]], styles: Dict[str, Any]) -> None:
    """Build Sheet 5: Comprehensive Diversification Analytics (Choueifaty, ENC, HHI and Multi-Dimension Breakdowns)."""
    ws.views.sheetView[0].showGridLines = True
    ddata = div_data or {}

    ws.cell(row=1, column=1, value="DIAGNÓSTICO INTEGRAL DE DIVERSIFICACIÓN REAL").font = styles["title_font"]
    ws.cell(row=2, column=1, value="Métricas Cuantitativas de Concentración, Desgloses Multidimensionales y Clasificación de Activos").font = styles["subtitle_font"]

    row_cursor = 4

    # 1. Summary Indicators Table
    summary_df = ddata.get("summary_metrics")
    if summary_df is not None and isinstance(summary_df, pd.DataFrame):
        row_cursor = _write_section_table(ws, summary_df, "1. INDICADORES MATEMÁTICOS DE DIVERSIFICACIÓN Y CONCENTRACIÓN", row_cursor, styles)

    # 2. Breakdown Dimensions (Sector, Country, Class, Cap)
    for key, title in [
        ("sector_breakdown", "2. DISTRIBUCIÓN POR SECTOR ECONÓMICO (GICS)"),
        ("country_breakdown", "3. DISTRIBUCIÓN POR PAÍS / GEOGRAFÍA"),
        ("asset_class_breakdown", "4. DISTRIBUCIÓN POR CLASE DE ACTIVO"),
        ("market_cap_breakdown", "5. DISTRIBUCIÓN POR CAPITALIZACIÓN DE MERCADO"),
    ]:
        b_df = ddata.get(key)
        if b_df is not None and isinstance(b_df, pd.DataFrame) and len(b_df) > 0:
            row_cursor = _write_section_table(ws, b_df, title, row_cursor, styles)

    # 3. Asset Details Matrix
    details_df = ddata.get("asset_classification_table")
    if details_df is not None and isinstance(details_df, pd.DataFrame) and len(details_df) > 0:
        _write_section_table(ws, details_df, "6. MATRIZ DETALLADA DE CLASIFICACIÓN POR ACTIVO", row_cursor, styles)

    _autofit_columns(ws)


def _write_backtest_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    wealth_df: Optional[Union[pd.DataFrame, pd.Series]],
    drawdown_df: Optional[pd.DataFrame],
    styles: Dict[str, Any],
) -> None:
    """Build Sheet 6: Historical Backtest ($10,000 USD Wealth Growth) & Underwater Drawdowns."""
    ws.views.sheetView[0].showGridLines = True

    # Combine wealth and drawdown into a consolidated historical sheet
    combined_df = None
    if wealth_df is not None:
        if isinstance(wealth_df, pd.Series):
            w_df = wealth_df.to_frame(name="Cartera ($)").copy()
            if not any(c in w_df.columns for c in ["Fecha", "Date", "date"]):
                w_df = w_df.reset_index()
        elif isinstance(wealth_df, pd.DataFrame):
            w_df = wealth_df.copy()
            if not any(c in w_df.columns for c in ["Fecha", "Date", "date"]):
                w_df = w_df.reset_index()
        else:
            w_df = pd.DataFrame()

        for c in ["index", "Date", "date"]:
            if c in w_df.columns and "Fecha" not in w_df.columns:
                w_df = w_df.rename(columns={c: "Fecha"})

        combined_df = w_df

    if drawdown_df is not None and isinstance(drawdown_df, pd.DataFrame) and len(drawdown_df) > 0:
        dd_clean = drawdown_df.copy()
        if not any(c in dd_clean.columns for c in ["Fecha", "Date", "date"]):
            dd_clean = dd_clean.reset_index()
        for c in ["index", "Date", "date"]:
            if c in dd_clean.columns and "Fecha" not in dd_clean.columns:
                dd_clean = dd_clean.rename(columns={c: "Fecha"})

        if combined_df is not None and "Fecha" in combined_df.columns and "Fecha" in dd_clean.columns:
            # Merge on Fecha
            combined_df["Fecha_str"] = pd.to_datetime(combined_df["Fecha"]).dt.strftime("%Y-%m-%d")
            dd_clean["Fecha_str"] = pd.to_datetime(dd_clean["Fecha"]).dt.strftime("%Y-%m-%d")
            dd_cols_to_merge = [c for c in dd_clean.columns if c != "Fecha"]
            merged = pd.merge(combined_df, dd_clean[dd_cols_to_merge], on="Fecha_str", how="left").drop(columns=["Fecha_str"])
            combined_df = merged
        elif combined_df is None:
            combined_df = dd_clean

    if combined_df is not None and len(combined_df) > 0:
        _write_styled_dataframe(
            ws=ws,
            df=combined_df,
            navy_fill=styles["navy_fill"],
            header_font=styles["header_font"],
            zebra_fill_white=styles["zebra_fill_white"],
            zebra_fill_gray=styles["zebra_fill_gray"],
            thin_border=styles["thin_border"],
            regular_font=styles["regular_font"],
            is_wealth_sheet=True,
        )
    else:
        ws.cell(row=1, column=1, value="Sin datos históricos suficientes").font = styles["regular_font"]


def _write_monte_carlo_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    mc_data: Optional[Dict[str, Any]],
    mc_samples_df: Optional[pd.DataFrame],
    styles: Dict[str, Any],
) -> None:
    """Build Sheet 7: Stochastic Future Wealth Cones & Scenarios."""
    ws.views.sheetView[0].showGridLines = True
    mdata = mc_data or {}

    ws.cell(row=1, column=1, value="PROYECCIÓN ESTOCÁSTICA MONTE CARLO (CONOS DE PROBABILIDAD)").font = styles["title_font"]
    ws.cell(row=2, column=1, value="Simulación multi-activo de trayectorias de riqueza futura a 1-5 años").font = styles["subtitle_font"]

    row_cursor = 4

    # 1. Simulation Parameters
    df_params = pd.DataFrame([
        {"Parámetro de Simulación": "Horizonte Temporal", "Valor": f"{mdata.get('horizon_years', 3)} años"},
        {"Parámetro de Simulación": "Número de Trayectorias", "Valor": f"{mdata.get('num_simulations', 2000):,}"},
        {"Parámetro de Simulación": "Modelo Estocástico", "Valor": mdata.get("model", "Movimiento Browniano Geométrico (GBM)")},
        {"Parámetro de Simulación": "Capital Inicial", "Valor": "$10,000 USD"},
    ])
    row_cursor = _write_section_table(ws, df_params, "1. PARÁMETROS DE LA SIMULACIÓN ESTOCÁSTICA", row_cursor, styles)

    # 2. Terminal Wealth Scenarios
    scenarios_df = mdata.get("terminal_scenarios")
    if scenarios_df is not None and isinstance(scenarios_df, pd.DataFrame):
        row_cursor = _write_section_table(ws, scenarios_df, "2. ESCENARIOS DE CAPITAL FINAL PROYECTADO", row_cursor, styles)

    # 3. Future Cones Evolution Table
    cones_df = mdata.get("cones_df")
    if cones_df is not None and isinstance(cones_df, pd.DataFrame):
        row_cursor = _write_section_table(ws, cones_df, "3. EVOLUCIÓN TEMPORAL DE PERCENTILES (CONOS)", row_cursor, styles)
    elif mc_samples_df is not None:
        _write_section_table(ws, mc_samples_df.head(500), "3. MUESTRA DE CARTERAS ALEATORIAS SIMULADAS", row_cursor, styles)

    _autofit_columns(ws)


def _write_comparator_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, comparator_data: Dict[str, Any], styles: Dict[str, Any]) -> None:
    """Build Sheet 10: Multi-Portfolio Comparator Analysis."""
    ws.views.sheetView[0].showGridLines = True

    ws.cell(row=1, column=1, value="COMPARADOR MULTI-PORTAFOLIO AVANZADO").font = styles["title_font"]
    ws.cell(row=2, column=1, value="Análisis comparativo de indicadores clave y asignación entre carteras").font = styles["subtitle_font"]

    row_cursor = 4

    comp_table = comparator_data.get("summary_table")
    if comp_table is not None and isinstance(comp_table, pd.DataFrame):
        row_cursor = _write_section_table(ws, comp_table, "1. TABLA COMPARATIVA DE INDICADORES CLAVE (KPIs)", row_cursor, styles)

    comp_weights = comparator_data.get("weights_table")
    if comp_weights is not None and isinstance(comp_weights, pd.DataFrame):
        _write_section_table(ws, comp_weights, "2. COMPARATIVA DE ASIGNACIÓN DE ACTIVOS (%)", row_cursor, styles)

    _autofit_columns(ws)


# ===========================================================================
# 5. Internal Table Layout and Formatting Utilities
# ===========================================================================

def _write_section_table(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    df: pd.DataFrame,
    title: str,
    start_row: int,
    styles: Dict[str, Any],
) -> int:
    """Write a titled section banner followed by a styled table, returning the next available row."""
    num_cols = max(len(df.columns), 1)

    # Section Banner
    banner_cell = ws.cell(row=start_row, column=1, value=title)
    banner_cell.fill = styles["section_fill"]
    banner_cell.font = styles["header_font"]
    banner_cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[start_row].height = 24

    for c in range(2, num_cols + 1):
        cell_empty = ws.cell(row=start_row, column=c)
        cell_empty.fill = styles["section_fill"]
        cell_empty.border = styles["thin_border"]
    banner_cell.border = styles["thin_border"]

    # Table Content
    next_row = _write_styled_dataframe(
        ws=ws,
        df=df,
        navy_fill=styles["navy_fill"],
        header_font=styles["header_font"],
        zebra_fill_white=styles["zebra_fill_white"],
        zebra_fill_gray=styles["zebra_fill_gray"],
        thin_border=styles["thin_border"],
        regular_font=styles["regular_font"],
        start_row=start_row + 1,
        auto_fit_columns=False,
    )

    # Return next available row leaving 2 empty rows as visual breathing room
    return next_row + 2


def _write_styled_dataframe(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    df: pd.DataFrame,
    navy_fill: PatternFill,
    header_font: Font,
    zebra_fill_white: PatternFill,
    zebra_fill_gray: PatternFill,
    thin_border: Border,
    regular_font: Font,
    number_format: Optional[str] = None,
    is_metrics_sheet: bool = False,
    is_weights_sheet: bool = False,
    is_wealth_sheet: bool = False,
    total_fill: Optional[PatternFill] = None,
    bold_font: Optional[Font] = None,
    thick_bottom_border: Optional[Border] = None,
    start_row: int = 1,
    auto_fit_columns: bool = True,
) -> int:
    """Helper to write and style a DataFrame onto an OpenPyXL Worksheet at a given row."""
    ws.views.sheetView[0].showGridLines = True

    # 1. Header Row
    headers = list(df.columns)
    ws.row_dimensions[start_row].height = 26
    for col_idx, col_name in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=col_idx, value=str(col_name))
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    # 2. Data Rows
    num_rows = len(df)
    for r_idx in range(num_rows):
        row_num = start_row + 1 + r_idx
        ws.row_dimensions[row_num].height = 20
        fill = zebra_fill_white if r_idx % 2 == 0 else zebra_fill_gray
        metric_name = str(df.iloc[r_idx, 0]).lower() if (is_metrics_sheet and len(headers) > 0) else ""

        for c_idx, col_name in enumerate(headers, start=1):
            val = df.iloc[r_idx, c_idx - 1]
            cell = ws.cell(row=row_num, column=c_idx)
            cell.fill = fill
            cell.font = regular_font
            cell.border = thin_border

            # Check for Datetime / Timestamp objects
            if isinstance(val, (pd.Timestamp, np.datetime64)):
                cell.value = str(pd.to_datetime(val).strftime("%Y-%m-%d"))
                cell.alignment = Alignment(horizontal="center", vertical="center")
                continue

            # String column or missing
            if c_idx == 1 or isinstance(val, str) or pd.isna(val):
                cell.value = "" if pd.isna(val) else str(val)
                cell.alignment = Alignment(horizontal="left", vertical="center")
                continue

            # Numeric Value
            if not isinstance(val, (int, float, np.integer, np.floating)):
                cell.value = str(val)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                continue

            num_val = float(val)
            cell.value = num_val
            cell.alignment = Alignment(horizontal="right", vertical="center")

            # Determine Number Format
            if number_format:
                cell.number_format = number_format
            elif is_metrics_sheet:
                col_header = str(col_name).lower()
                if any(kw in metric_name for kw in ["retorno", "cagr", "volatilidad", "drawdown", "var", "cvar", "rendimiento", "%", "alfa", "r²"]):
                    cell.number_format = "0.00%"
                elif any(kw in metric_name for kw in ["sharpe", "sortino", "calmar", "beta", "ratio", "entropía"]):
                    cell.number_format = "0.000"
                elif "días" in metric_name or "days" in metric_name or "hhi" in metric_name or "enc" in metric_name:
                    cell.number_format = "#,##0"
                else:
                    cell.number_format = "0.0000"
            elif is_weights_sheet:
                # If column is Retorno or Volatilidad -> %, if Sharpe or Beta -> 0.000, else weight %
                c_name = str(col_name).lower()
                if "sharpe" in c_name or "beta" in c_name:
                    cell.number_format = "0.000"
                elif "retorno" in c_name or "volatilidad" in c_name or "peso" in c_name or "%" in c_name or c_idx > 1:
                    cell.number_format = "0.00%"
                else:
                    cell.number_format = "0.0000"
            elif is_wealth_sheet:
                c_name = str(col_name).lower()
                if "drawdown" in c_name or "dd" in c_name or "%" in c_name:
                    cell.number_format = "0.00%"
                else:
                    cell.number_format = "$#,##0.00"
            else:
                c_name = str(col_name).lower()
                if "%" in c_name or "retorno" in c_name or "volatilidad" in c_name or "peso" in c_name:
                    cell.number_format = "0.00%"
                elif "sharpe" in c_name or "ratio" in c_name or "beta" in c_name:
                    cell.number_format = "0.000"
                elif "capital" in c_name or "$" in c_name:
                    cell.number_format = "$#,##0.00"
                elif abs(num_val) < 0.001 and num_val != 0.0:
                    cell.number_format = "0.000000"
                elif abs(num_val) < 1.0:
                    cell.number_format = "0.0000"
                else:
                    cell.number_format = "#,##0.00"

    current_end_row = start_row + num_rows

    # 3. Total Check Row for Weights Sheet
    if is_weights_sheet and num_rows > 0:
        tot_row = current_end_row + 1
        ws.row_dimensions[tot_row].height = 22
        cell_label = ws.cell(row=tot_row, column=1, value="TOTAL")
        cell_label.font = bold_font if bold_font else regular_font
        cell_label.fill = total_fill if total_fill else zebra_fill_gray
        cell_label.alignment = Alignment(horizontal="left", vertical="center")
        cell_label.border = thick_bottom_border if thick_bottom_border else thin_border

        for c_idx in range(2, len(headers) + 1):
            col_letter = get_column_letter(c_idx)
            c_name = str(headers[c_idx - 1]).lower()
            cell_sum = ws.cell(row=tot_row, column=c_idx)
            # Only put SUM on weight columns
            if not any(kw in c_name for kw in ["beta", "sharpe", "retorno", "volatilidad"]):
                cell_sum.value = f"=SUM({col_letter}{start_row + 1}:{col_letter}{tot_row - 1})"
                cell_sum.number_format = "0.00%"
            else:
                cell_sum.value = "-"
                cell_sum.alignment = Alignment(horizontal="center", vertical="center")
            cell_sum.font = bold_font if bold_font else regular_font
            cell_sum.fill = total_fill if total_fill else zebra_fill_gray
            cell_sum.border = thick_bottom_border if thick_bottom_border else thin_border

        current_end_row = tot_row

    # 4. Auto-fit column widths
    if auto_fit_columns:
        _autofit_columns(ws)

    return current_end_row


def _autofit_columns(ws: openpyxl.worksheet.worksheet.Worksheet) -> None:
    """Adjust worksheet column widths to fit content comfortably with padding."""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            # Truncate extremely long strings to prevent absurdly wide columns
            if len(val_str) > 60:
                val_str = val_str[:60]
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)
