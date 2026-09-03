"""
Frontera Eficiente — Markowitz Quantitative Portfolio Optimization Platform.

Production-grade Streamlit web application providing:
1. Hybrid Data Ingestion (Live Yahoo Finance & Manual CSV/Excel Upload)
2. Statistical Modeling & Robust Covariance Estimators (Ledoit-Wolf Shrinkage, EWMA, Sample)
3. Markowitz Quantitative Optimization (Tangency Maximum Sharpe, GMV, Continuous Frontier Sweep, CAL)
4. Dual Monte Carlo Simulations (Dirichlet Simplex & Multi-Year Stochastic Cones)
5. Interactive Asset Allocation Sliders & 1-Click Canonical Presets
6. High-Performance Interactive Plotly Visualizations & Multi-Format CSV / Excel Exporter
"""

from __future__ import annotations

import datetime
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Internal platform module imports
from src.analytics.risk_metrics import (
    calculate_asset_betas,
    calculate_beta,
    calculate_jensen_alpha,
    compute_portfolio_risk_metrics,
)
from src.data.broker_parser import parse_broker_holdings
from src.data.cleaner import clean_and_align_prices
from src.data.loader import fetch_asset_data, load_manual_file, validate_tickers
from src.export.exporter import (
    export_correlation_csv,
    export_full_excel,
    export_summary_csv,
    export_weights_csv,
)
from src.models import (
    CovarianceMethod,
    ReturnMethod,
    build_risk_model,
    calculate_expected_returns,
    estimate_covariance_matrix,
    ensure_positive_semidefinite,
)
from src.optimization import (
    compute_capital_allocation_line,
    compute_efficient_frontier,
    normalize_and_clamp_weights,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe,
)
from src.auth.google_auth import (
    get_active_user,
    get_active_user_id,
    init_auth_session,
    render_user_auth_sidebar,
)
from src.presets import (
    PRESETS,
    delete_custom_portfolio,
    export_portfolios_json,
    get_preset,
    import_portfolios_json,
    list_presets,
    load_saved_portfolios,
    save_custom_portfolio,
)
from src.simulation import run_trajectory_monte_carlo, run_weight_space_monte_carlo
from src.visualization import (
    plot_allocation_comparison,
    plot_asset_allocation,
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_efficient_frontier,
    plot_historical_backtest,
    plot_monte_carlo_cones,
    plot_portfolio_comparison_scatter,
)


# ===========================================================================
# 1. Page Configuration & Custom CSS Styling
# ===========================================================================

st.set_page_config(
    page_title="Frontera Eficiente — Markowitz Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern dark financial UI CSS
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background-color: #161b26;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #A0AEC0;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #00F0FF;
        font-size: 22px;
        font-weight: 700;
        font-family: monospace;
    }
    .metric-subtitle {
        color: #718096;
        font-size: 12px;
    }
    /* Status Badges */
    .badge-valid {
        background-color: rgba(0, 255, 102, 0.15);
        color: #00FF66;
        border: 1px solid #00FF66;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .badge-warning {
        background-color: rgba(255, 204, 0, 0.15);
        color: #FFCC00;
        border: 1px solid #FFCC00;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    /* Action Buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #00F0FF;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# 2. Session State Initialization & Synchronization Logic
# ===========================================================================

# ===========================================================================
# 2. Session State Initialization & Matrix Synchronization Logic
# ===========================================================================

def _get_default_portfolio_df(n: int = 5) -> pd.DataFrame:
    """Generate default portfolio dataframe with n assets."""
    default_tickers = [
        "SPY", "TLT", "AAPL", "MSFT", "GLD", "QQQ", "NVDA", "AMZN",
        "GOOGL", "META", "BTC-USD", "ETH-USD", "BND", "IEF", "DBC",
        "KO.BA", "MELI.BA", "GGAL.BA", "TSLA", "JNJ"
    ]
    tickers = default_tickers[:n] if n <= len(default_tickers) else default_tickers + [f"ACTIVO_{i+1}" for i in range(len(default_tickers), n)]
    w = round(100.0 / n, 2)
    weights = [w] * n
    weights[-1] = round(100.0 - sum(weights[:-1]), 2)
    return pd.DataFrame({
        "Ticker": tickers,
        "Ponderación (%)": weights,
    })


def _init_session_state() -> None:
    """Initialize application session state variables."""
    if "num_assets" not in st.session_state:
        st.session_state["num_assets"] = 5

    if "num_assets_selector" not in st.session_state:
        st.session_state["num_assets_selector"] = st.session_state["num_assets"]

    if "editor_version" not in st.session_state:
        st.session_state["editor_version"] = 0

    if "portfolio_matrix_df" not in st.session_state:
        st.session_state["portfolio_matrix_df"] = _get_default_portfolio_df(st.session_state["num_assets"])

    if "tickers" not in st.session_state:
        st.session_state["tickers"] = list(st.session_state["portfolio_matrix_df"]["Ticker"])

    if "weights" not in st.session_state:
        st.session_state["weights"] = {
            str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
            for _, r in st.session_state["portfolio_matrix_df"].iterrows()
        }

    if "rf_rate" not in st.session_state:
        st.session_state["rf_rate"] = 0.04

    if "opt_weights_max_sharpe" not in st.session_state:
        st.session_state["opt_weights_max_sharpe"] = None

    if "opt_weights_gmv" not in st.session_state:
        st.session_state["opt_weights_gmv"] = None

    if "active_preset_name" not in st.session_state:
        st.session_state["active_preset_name"] = "Personalizado"


_init_session_state()
init_auth_session()


def _resize_portfolio_matrix(target_n: int) -> None:
    """Resize the portfolio matrix rows when the number of assets changes."""
    current_df = st.session_state.get("portfolio_matrix_df")
    default_pool = [
        "SPY", "TLT", "AAPL", "MSFT", "GLD", "QQQ", "NVDA", "AMZN",
        "GOOGL", "META", "BTC-USD", "ETH-USD", "BND", "IEF", "DBC",
        "KO.BA", "MELI.BA", "GGAL.BA", "TSLA", "JNJ"
    ]
    if current_df is None or current_df.empty:
        st.session_state["portfolio_matrix_df"] = _get_default_portfolio_df(target_n)
    else:
        existing_rows = current_df.to_dict(orient="records")
        current_n = len(existing_rows)
        if target_n > current_n:
            existing_tickers = [str(r.get("Ticker", "")).upper() for r in existing_rows]
            available_pool = [t for t in default_pool if t not in existing_tickers]
            for i in range(current_n, target_n):
                next_t = available_pool.pop(0) if available_pool else f"ACTIVO_{i+1}"
                existing_rows.append({"Ticker": next_t, "Ponderación (%)": 0.0})
        elif target_n < current_n:
            existing_rows = existing_rows[:target_n]

        st.session_state["portfolio_matrix_df"] = pd.DataFrame(existing_rows)

    st.session_state["num_assets"] = target_n
    st.session_state["num_assets_selector"] = target_n
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    parsed_tickers = [str(r["Ticker"]).strip().upper() for r in st.session_state["portfolio_matrix_df"].to_dict(orient="records") if str(r.get("Ticker", "")).strip()]
    st.session_state["tickers"] = parsed_tickers if parsed_tickers else ["SPY", "TLT"]
    st.session_state["weights"] = {
        str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
        for r in st.session_state["portfolio_matrix_df"].to_dict(orient="records")
        if str(r.get("Ticker", "")).strip()
    }


def _on_change_num_assets() -> None:
    """Callback when user edits the number of assets input."""
    new_n = int(st.session_state.get("num_assets_selector", len(st.session_state.get("tickers", []))))
    _resize_portfolio_matrix(new_n)


def _apply_preset(preset_key: str) -> None:
    """Load a predefined portfolio preset and synchronize session state."""
    preset = get_preset(preset_key)
    tickers = list(preset["tickers"])
    weights = dict(preset["weights"])
    n = len(tickers)

    st.session_state["num_assets"] = n
    st.session_state["num_assets_selector"] = n
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["tickers"] = tickers
    st.session_state["weights"] = weights
    st.session_state["active_preset_name"] = preset["name"]

    new_df = pd.DataFrame({
        "Ticker": tickers,
        "Ponderación (%)": [round(float(weights.get(t, 1.0 / n) * 100.0), 2) for t in tickers],
    })
    st.session_state["portfolio_matrix_df"] = new_df


def _load_saved_portfolio_to_matrix(portfolio_id: str) -> None:
    """Load a custom saved portfolio into session state and matrix."""
    saved = load_saved_portfolios(get_active_user_id())
    if portfolio_id not in saved:
        return
    p = saved[portfolio_id]
    tickers = list(p["tickers"])
    weights = dict(p["weights"])
    n = len(tickers)

    st.session_state["num_assets"] = n
    st.session_state["num_assets_selector"] = n
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["tickers"] = tickers
    st.session_state["weights"] = weights
    st.session_state["active_preset_name"] = p["name"]

    new_df = pd.DataFrame({
        "Ticker": tickers,
        "Ponderación (%)": [round(float(weights.get(t, 1.0 / n) * 100.0), 2) for t in tickers],
    })
    st.session_state["portfolio_matrix_df"] = new_df


def _normalize_current_weights() -> None:
    """
    Normalize current active matrix weights to sum exactly to 100%.
    
    If Cash/Liquidity is present in the matrix:
    - Leaves all other asset weights intact.
    - Assigns the entire residual difference (100% - sum(risky)) to Cash.
    If no Cash is present:
    - Scales all assets proportionally to sum to 100%.
    """
    current_df = st.session_state.get("portfolio_matrix_df")
    if current_df is None or current_df.empty:
        return
    rows = current_df.to_dict(orient="records")
    n = len(rows)
    if n == 0:
        return

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    cash_indices = [i for i, r in enumerate(rows) if str(r.get("Ticker", "")).strip().upper() in cash_symbols]

    if cash_indices:
        cash_idx = cash_indices[0]
        # Sum of non-cash assets
        non_cash_sum = sum(
            float(r.get("Ponderación (%)", 0.0))
            for i, r in enumerate(rows)
            if i != cash_idx
        )
        if non_cash_sum <= 100.0:
            # Leave non-cash untouched and send exact residual to Cash
            rows[cash_idx]["Ponderación (%)"] = round(100.0 - non_cash_sum, 2)
        else:
            # Non-cash assets already exceed 100%: set cash to 0 and normalize non-cash
            rows[cash_idx]["Ponderación (%)"] = 0.0
            for i, r in enumerate(rows):
                if i != cash_idx:
                    r["Ponderación (%)"] = round((float(r.get("Ponderación (%)", 0.0)) / non_cash_sum) * 100.0, 2)
            diff = round(100.0 - sum(r["Ponderación (%)"] for r in rows), 2)
            first_non_cash = [i for i in range(n) if i != cash_idx]
            if first_non_cash:
                rows[first_non_cash[-1]]["Ponderación (%)"] = round(rows[first_non_cash[-1]]["Ponderación (%)"] + diff, 2)
    else:
        # Standard proportional normalization
        total = sum([float(r.get("Ponderación (%)", 0.0)) for r in rows])
        if total > 1e-6:
            for r in rows:
                r["Ponderación (%)"] = round((float(r.get("Ponderación (%)", 0.0)) / total) * 100.0, 2)
            diff = round(100.0 - sum(r["Ponderación (%)"] for r in rows), 2)
            if len(rows) > 0:
                rows[-1]["Ponderación (%)"] = round(rows[-1]["Ponderación (%)"] + diff, 2)
        else:
            eq = round(100.0 / n, 2) if n > 0 else 100.0
            for r in rows:
                r["Ponderación (%)"] = eq
            diff = round(100.0 - sum(r["Ponderación (%)"] for r in rows), 2)
            if len(rows) > 0:
                rows[-1]["Ponderación (%)"] = round(rows[-1]["Ponderación (%)"] + diff, 2)

    new_df = pd.DataFrame(rows)
    st.session_state["portfolio_matrix_df"] = new_df
    st.session_state["num_assets"] = len(rows)
    st.session_state["num_assets_selector"] = len(rows)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["weights"] = {
        str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
        for r in rows
    }


def _apply_optimal_weights_by_type(opt_type: str) -> None:
    """Set matrix weights to optimized weights (Max Sharpe or GMV)."""
    target_weights = st.session_state.get(f"opt_weights_{opt_type}")
    if target_weights is None:
        return
    current_df = st.session_state.get("portfolio_matrix_df")
    if current_df is None or current_df.empty:
        return
    rows = current_df.to_dict(orient="records")
    norm_w = {}
    for i, r in enumerate(rows):
        pct = float(target_weights[i] * 100.0) if i < len(target_weights) else 0.0
        r["Ponderación (%)"] = round(pct, 2)
        norm_w[str(r["Ticker"]).strip().upper()] = float(pct / 100.0)

    diff = round(100.0 - sum(r["Ponderación (%)"] for r in rows), 2)
    if abs(diff) < 0.05 and len(rows) > 0:
        rows[-1]["Ponderación (%)"] = round(rows[-1]["Ponderación (%)"] + diff, 2)

    new_df = pd.DataFrame(rows)
    st.session_state["portfolio_matrix_df"] = new_df
    st.session_state["num_assets"] = len(rows)
    st.session_state["num_assets_selector"] = len(rows)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["weights"] = norm_w


def _apply_equal_weights() -> None:
    """Set matrix weights to uniform 1/N weights."""
    current_df = st.session_state.get("portfolio_matrix_df")
    if current_df is None or current_df.empty:
        return
    rows = current_df.to_dict(orient="records")
    n = len(rows)
    eq = round(100.0 / n, 2) if n > 0 else 100.0
    for r in rows:
        r["Ponderación (%)"] = eq
    diff = round(100.0 - sum(r["Ponderación (%)"] for r in rows), 2)
    if abs(diff) < 0.05 and len(rows) > 0:
        rows[-1]["Ponderación (%)"] = round(rows[-1]["Ponderación (%)"] + diff, 2)

    new_df = pd.DataFrame(rows)
    st.session_state["portfolio_matrix_df"] = new_df
    st.session_state["num_assets"] = len(rows)
    st.session_state["num_assets_selector"] = len(rows)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["weights"] = {
        str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
        for r in rows
    }


def _remove_asset_from_portfolio(ticker_to_remove: Optional[str]) -> None:
    """
    Remove an asset from the portfolio matrix.
    
    Reallocation rule:
    - If CASH/Liquidity is present in the remaining assets, transfer the removed asset's weight to CASH.
    - If no CASH is present in the remaining assets, distribute the weight equally among all remaining assets.
    - Synchronize num_assets, num_assets_selector, editor_version, tickers, and weights.
    """
    if not ticker_to_remove:
        return
    current_df = st.session_state.get("portfolio_matrix_df")
    if current_df is None or current_df.empty:
        return
    
    rows = current_df.to_dict(orient="records")
    if len(rows) <= 2:
        st.warning("⚠️ El portafolio debe contener al menos 2 activos para realizar la optimización de Markowitz.")
        return

    clean_target = str(ticker_to_remove).strip().upper()
    
    # Find row to remove
    target_idx = None
    for i, r in enumerate(rows):
        if str(r.get("Ticker", "")).strip().upper() == clean_target:
            target_idx = i
            break
            
    if target_idx is None:
        return

    removed_row = rows.pop(target_idx)
    removed_weight = float(removed_row.get("Ponderación (%)", 0.0))
    n_remaining = len(rows)

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    cash_indices = [i for i, r in enumerate(rows) if str(r.get("Ticker", "")).strip().upper() in cash_symbols]

    if cash_indices:
        # Case A: Transfer weight directly to CASH
        cash_idx = cash_indices[0]
        rows[cash_idx]["Ponderación (%)"] = round(float(rows[cash_idx].get("Ponderación (%)", 0.0)) + removed_weight, 2)
    else:
        # Case B: Distribute weight equally among remaining assets
        if n_remaining > 0:
            share = removed_weight / n_remaining
            for r in rows:
                r["Ponderación (%)"] = round(float(r.get("Ponderación (%)", 0.0)) + share, 2)

    # Balance rounding difference so sum is strictly 100.00%
    diff = round(100.0 - sum(float(r.get("Ponderación (%)", 0.0)) for r in rows), 2)
    if abs(diff) > 0.0 and len(rows) > 0:
        if cash_indices:
            rows[cash_indices[0]]["Ponderación (%)"] = round(float(rows[cash_indices[0]]["Ponderación (%)"]) + diff, 2)
        else:
            rows[-1]["Ponderación (%)"] = round(float(rows[-1]["Ponderación (%)"]) + diff, 2)

    new_df = pd.DataFrame(rows)
    st.session_state["portfolio_matrix_df"] = new_df
    st.session_state["num_assets"] = len(rows)
    st.session_state["num_assets_selector"] = len(rows)
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    st.session_state["tickers"] = [str(r["Ticker"]).strip().upper() for r in rows if str(r.get("Ticker", "")).strip()]
    st.session_state["weights"] = {
        str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
        for r in rows
        if str(r.get("Ticker", "")).strip()
    }


# ===========================================================================
# 3. Sidebar Controls: Data Ingestion Source, Estimators & Constraints
# ===========================================================================

with st.sidebar:
    render_user_auth_sidebar()
    st.markdown("## ⚙️ Configuración Cuantitativa")
    st.caption("Motor de Modelado y Frontera de Markowitz")
    st.markdown("---")

    # Data Source Selection
    data_source = st.radio(
        "Fuente de Datos Históricos",
        options=["Yahoo Finance (En Vivo)", "Carga Manual (CSV / Excel)"],
        index=0,
    )

    clean_prices_df = None
    daily_returns_df = None

    if data_source == "Carga Manual (CSV / Excel)":
        uploaded_file = st.file_uploader(
            "Subir archivo de precios históricos (.csv o .xlsx)",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                raw_df = load_manual_file(file_bytes, filename=uploaded_file.name)
                clean_prices_df, daily_returns_df = clean_and_align_prices(raw_df)
                file_tickers = list(clean_prices_df.columns)
                st.session_state["tickers"] = file_tickers
                st.session_state["num_assets"] = len(file_tickers)
                st.session_state["num_assets_selector"] = len(file_tickers)
                st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                st.session_state["portfolio_matrix_df"] = pd.DataFrame({
                    "Ticker": file_tickers,
                    "Ponderación (%)": [round(100.0 / len(file_tickers), 2)] * len(file_tickers),
                })
                st.success(f"Archivo cargado con éxito: {len(file_tickers)} activos.")
            except Exception as ex:
                st.error(f"Error procesando archivo: {ex}")

    st.markdown("### 📊 Estimadores Estadísticos")
    rf_input = st.number_input(
        "Tasa Libre de Riesgo Anualizada (Rf)",
        min_value=0.0,
        max_value=0.25,
        value=float(st.session_state["rf_rate"]),
        step=0.005,
        format="%.4f",
        help="Tasa libre de riesgo anualizada (ej. 0.04 para 4.00% anual de US Treasuries).",
    )
    st.session_state["rf_rate"] = rf_input

    return_estimator = st.selectbox(
        "Estimador de Retornos Esperados (μ)",
        options=[
            ("arithmetic", "Media Histórica Aritmética"),
            ("geometric", "Retorno Compuesto (CAGR)"),
            ("ewma", "EWMA Ponderado (λ=0.94)"),
            ("capm", "Modelo CAPM"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    cov_estimator = st.selectbox(
        "Estimador de Matriz de Covarianza (Σ)",
        options=[
            ("ledoit_wolf_cc", "Ledoit-Wolf Shrinkage (Constant Corr) [Recomendado]"),
            ("ledoit_wolf_diag", "Ledoit-Wolf Shrinkage (Diagonal)"),
            ("sample", "Covarianza Muestral Clásica"),
            ("ewma", "EWMA Covarianza Exponencial"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    benchmark_symbol = st.selectbox(
        "Índice de Referencia / Benchmark (para Beta y Alfa)",
        options=["SPY", "QQQ", "ACWI", "DIA", "IWM", "^GSPC"],
        index=0,
        help="Activo de mercado contra el cual se calculan el Beta (β), el Alfa de Jensen (α) y el R² de cada portafolio.",
    )

    st.markdown("---")
    st.markdown("### 🔒 Restricciones de Inversión")
    long_only = st.checkbox("Solo Posiciones Largas (Long-Only: w_i ≥ 0)", value=True)
    allow_short = not long_only

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        min_weight_pct = st.number_input("Peso Mínimo (%)", value=0.0, min_value=-100.0, max_value=100.0, step=5.0) / 100.0
    with col_b2:
        max_weight_pct = st.number_input("Peso Máximo (%)", value=100.0, min_value=0.0, max_value=100.0, step=5.0) / 100.0

    n_curr = max(1, len(st.session_state.get("tickers", [1, 2])))
    min_req_max = 1.0 / n_curr
    if max_weight_pct < min_req_max - 1e-5:
        st.caption(f"⚠️ *Con {n_curr} activos, el peso máx se auto-ajustará para permitir sumar 100%.*")

    bounds = (min_weight_pct if allow_short else max(0.0, min_weight_pct), max_weight_pct)


# ===========================================================================
# 4. Main Dashboard Header & Top Presets 1-Click Action Bar
# ===========================================================================

st.markdown("# 📈 Frontera Eficiente & Optimización Cuantitativa")
st.markdown(
    "**Plataforma interactiva de Markowitz:** Modelado de covarianza robusta (Ledoit-Wolf / EWMA), "
    "simulación dual de Monte Carlo (simplex de ponderaciones y conos de trayectorias) y análisis de riesgo avanzado."
)

st.markdown("### 🎯 Portafolios Predefinidos (1-Click Presets)")
cols_p = st.columns(5)
with cols_p[0]:
    st.button("💼 Clásico 60/40", on_click=_apply_preset, args=("classic_60_40",), use_container_width=True, help="SPY 60%, TLT 40%")
with cols_p[1]:
    st.button("🌦️ All-Weather", on_click=_apply_preset, args=("all_weather",), use_container_width=True, help="Ray Dalio: SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, DBC 7.5%")
with cols_p[2]:
    st.button("💻 Big Tech", on_click=_apply_preset, args=("big_tech",), use_container_width=True, help="AAPL 20%, MSFT 20%, GOOGL 20%, AMZN 20%, NVDA 20%")
with cols_p[3]:
    st.button("🇦🇷 CEDEARs Argentina", on_click=_apply_preset, args=("cedears_argentina",), use_container_width=True, help="AAPL.BA, MSFT.BA, GOOGL.BA, MELI.BA, SPY.BA, KO.BA")
with cols_p[4]:
    st.button("₿ Cripto + TradFi", on_click=_apply_preset, args=("crypto_tradfi",), use_container_width=True, help="SPY 50%, QQQ 30%, BTC-USD 15%, ETH-USD 5%")

# ===========================================================================
# 4.5. Custom Portfolio Persistence Manager (Save, Load, Edit, Delete)
# ===========================================================================

active_user = get_active_user()
active_uid = get_active_user_id()
saved_portfolios_dict = load_saved_portfolios(active_uid)

user_badge_title = f" ({active_user['name']} - {len(saved_portfolios_dict)} disponibles)" if active_user else f" (Modo Invitado - {len(saved_portfolios_dict)} disponibles)"

with st.expander(f"💾 Mis Portafolios Guardados{user_badge_title}", expanded=False):
    col_save_p, col_load_p = st.columns(2)
    with col_save_p:
        st.markdown("##### 💾 Guardar Composición Actual")
        save_p_name = st.text_input("Nombre del Portafolio", placeholder="Ej. Mi Cartera Agresiva", key="save_p_name_input")
        save_p_desc = st.text_input("Descripción (Opcional)", placeholder="Ej. 60% Tech + 40% Cripto", key="save_p_desc_input")
        if st.button("💾 Guardar Portafolio", use_container_width=True, key="btn_save_custom_portfolio"):
            if save_p_name.strip():
                save_custom_portfolio(
                    name=save_p_name,
                    tickers=st.session_state["tickers"],
                    weights=st.session_state["weights"],
                    description=save_p_desc,
                    user_id=active_uid,
                )
                st.success(f"¡Portafolio '{save_p_name}' guardado exitosamente!")
                st.rerun()
            else:
                st.warning("Por favor ingresa un nombre para el portafolio.")

    with col_load_p:
        st.markdown("##### 📂 Cargar / Administrar Portafolios")
        if saved_portfolios_dict:
            selected_saved_id = st.selectbox(
                "Seleccionar Portafolio Guardado",
                options=list(saved_portfolios_dict.keys()),
                format_func=lambda x: f"{saved_portfolios_dict[x]['name']} ({len(saved_portfolios_dict[x]['tickers'])} activos)",
                key="select_saved_p_dropdown",
            )
            p_selected = saved_portfolios_dict[selected_saved_id]
            st.caption(f"**Fecha:** {p_selected.get('updated_at', '')[:10]} | **Activos:** {', '.join(p_selected['tickers'])}")
            
            c_btn_load, c_btn_overwrite, c_btn_del = st.columns(3)
            with c_btn_load:
                if st.button("📥 Cargar", use_container_width=True, key="btn_load_saved_portfolio", help="Carga este portafolio en la matriz principal"):
                    _load_saved_portfolio_to_matrix(selected_saved_id)
                    st.rerun()
            with c_btn_overwrite:
                if st.button("✏️ Sobrescribir", use_container_width=True, key="btn_overwrite_saved_portfolio", help="Actualiza este portafolio con la matriz actual"):
                    save_custom_portfolio(
                        name=p_selected["name"],
                        tickers=st.session_state["tickers"],
                        weights=st.session_state["weights"],
                        description=p_selected.get("description", ""),
                        portfolio_id=selected_saved_id,
                        user_id=active_uid,
                    )
                    st.success(f"¡Portafolio '{p_selected['name']}' actualizado!")
                    st.rerun()
            with c_btn_del:
                if st.button("🗑️ Eliminar", use_container_width=True, key="btn_del_saved_portfolio", help="Elimina este portafolio guardado"):
                    delete_custom_portfolio(selected_saved_id, user_id=active_uid)
                    st.success("Portafolio eliminado.")
                    st.rerun()
        else:
            st.info("Aún no tienes portafolios guardados. Ingresa un nombre a la izquierda para guardar tu primera composición.")
            
    # Backup import/export row
    st.markdown("---")
    c_exp_json, c_imp_json = st.columns(2)
    with c_exp_json:
        json_backup_str = export_portfolios_json(user_id=active_uid)
        st.download_button(
            "📤 Exportar Backup de Portafolios (JSON)",
            data=json_backup_str,
            file_name="mis_portafolios_frontera.json",
            mime="application/json",
            use_container_width=True,
            key="btn_download_portfolios_json",
        )
    with c_imp_json:
        uploaded_json = st.file_uploader("📥 Importar Backup JSON", type=["json"], key="import_p_json_uploader")
        if uploaded_json is not None:
            try:
                imported_count = import_portfolios_json(uploaded_json.read().decode("utf-8"), user_id=active_uid)
                st.success(f"¡{imported_count} portafolios importados!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importando archivo JSON: {e}")

# ===========================================================================
# 4.6. Broker Holdings Report Importer (Excel / CSV: Balanz, PPI, IOL, Bull Market)
# ===========================================================================

with st.expander("📊 Importar Tenencias desde Excel / CSV de Broker (Balanz, PPI, IOL, Bull Market)", expanded=False):
    st.caption("Sube el archivo Excel (`.xlsx`, `.xls`) o CSV exportado desde tu broker (ej. Balanz 'Mis Instrumentos', PPI, Bull Market, IOL). El motor detectará automáticamente tickers, tipos de activo (CEDEARs, Acciones, Fondos, Bonos) y sus ponderaciones porcentuales.")
    
    col_up_mode, col_up_file = st.columns([1.2, 2.0])
    with col_up_mode:
        broker_mode = st.radio(
            "Modo de Mapeo de Tickers:",
            options=["🌐 Dólares / Subyacente Global (Recomendado)", "🇦🇷 Pesos / BYMA (.BA)"],
            index=0,
            help="Global: Mapea CEDEARs y ADRs a cotización en USD (ej. GOOGL, YPF, ITUB). BYMA: Mapea con sufijo .BA para cotización en Pesos (ej. GOOGL.BA, YPFD.BA).",
            key="broker_import_mode_radio",
        )
        mode_val = "global_usd" if "Dólares" in broker_mode else "byma_ars"
    
    with col_up_file:
        broker_file = st.file_uploader(
            "Seleccionar reporte de tenencias (.xlsx, .xls, .csv)",
            type=["xlsx", "xls", "csv"],
            key="broker_holdings_uploader",
        )
    
    if broker_file is not None:
        try:
            report = parse_broker_holdings(broker_file.read(), filename=broker_file.name, mode=mode_val)
            st.success(f"✅ Se detectaron exitosamente **{report.instruments_count} instrumentos** (Valuación total: {report.currency} ${report.total_valuation:,.2f})")
            
            # Show summary badges
            breakdown_text = " | ".join([f"**{k}:** {v}" for k, v in report.by_type_breakdown.items()])
            st.markdown(f"<div style='font-size: 13px; color: #A0AEC0; margin-bottom: 8px;'>Desglose por tipo: {breakdown_text}</div>", unsafe_allow_html=True)
            
            # Preview table
            st.dataframe(report.table_df, use_container_width=True, hide_index=True)
            
            col_apply_brk, col_save_brk = st.columns([1.5, 2.0])
            with col_apply_brk:
                if st.button("🚀 Cargar Cartera en el Optimizador", use_container_width=True, key="btn_apply_broker_portfolio"):
                    n_b = len(report.tickers)
                    st.session_state["num_assets"] = n_b
                    st.session_state["num_assets_selector"] = n_b
                    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                    st.session_state["tickers"] = report.tickers
                    st.session_state["weights"] = report.weights
                    st.session_state["portfolio_matrix_df"] = pd.DataFrame({
                        "Ticker": report.tickers,
                        "Ponderación (%)": [report.weights_pct.get(t, 0.0) for t in report.tickers],
                    })
                    st.session_state["active_preset_name"] = f"Broker ({broker_file.name[:15]}...)"
                    st.success("¡Portafolio cargado en la matriz principal!")
                    st.rerun()
            with col_save_brk:
                today_str = datetime.date.today().strftime('%d/%m/%Y')
                broker_save_name = st.text_input("Nombre para guardar en 'Mis Portafolios'", value=f"Cartera Broker {today_str}", key="broker_save_name_input")
                if st.button("💾 Guardar Directamente como Portafolio", use_container_width=True, key="btn_save_direct_broker"):
                    save_custom_portfolio(
                        name=broker_save_name,
                        tickers=report.tickers,
                        weights=report.weights,
                        description=f"Importado de reporte broker ({report.instruments_count} activos)",
                        user_id=active_uid,
                    )
                    st.success(f"¡Portafolio '{broker_save_name}' guardado en tus portafolios!")
                    st.rerun()
        except Exception as e:
            st.error(f"Error procesando el reporte del broker: {e}")

st.markdown("---")


# ===========================================================================
# 5. Interactive Portfolio Matrix Builder (Rows = Number of Assets)
# ===========================================================================

st.markdown("### 📋 Matriz de Activos y Ponderaciones del Portafolio")
st.caption("1. Elige la **cantidad de papeles**. 2. Escribe los tickers y sus ponderaciones. 💡 *Tip: Puedes ingresar `CASH` o `USD` como activo para asignar liquidez a tasa Rf.*")

col_num, col_dates, col_refresh = st.columns([1.2, 2.0, 1.0])
with col_num:
    curr_matrix_len = len(st.session_state.get("portfolio_matrix_df", []))
    if curr_matrix_len > 0 and st.session_state.get("num_assets_selector") != curr_matrix_len:
        st.session_state["num_assets"] = curr_matrix_len
        st.session_state["num_assets_selector"] = curr_matrix_len

    num_assets_val = st.number_input(
        "🔢 Cantidad de Papeles",
        min_value=2,
        max_value=30,
        step=1,
        key="num_assets_selector",
        on_change=_on_change_num_assets,
        help="Ajusta dinámicamente la cantidad de filas en la matriz.",
    )

with col_dates:
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365 * 3)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Fecha Inicio", value=default_start)
    with col_d2:
        end_date = st.date_input("Fecha Fin", value=today)

with col_refresh:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    btn_fetch = st.button("🔄 Descargar / Actualizar", use_container_width=True, help="Descarga los datos históricos de todos los tickers en la matriz.")

# Render Editable Matrix
editor_key = f"portfolio_matrix_editor_{st.session_state.get('editor_version', 0)}"
edited_matrix_df = st.data_editor(
    st.session_state["portfolio_matrix_df"],
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Activo / Ticker",
            help="Símbolo en Yahoo Finance (ej. AAPL, SPY, GGAL.BA, BTC-USD, GLD)",
            required=True,
        ),
        "Ponderación (%)": st.column_config.NumberColumn(
            "Ponderación (%)",
            help="Porcentaje de asignación en la cartera",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            format="%.2f%%",
        ),
    },
    num_rows="fixed",
    use_container_width=True,
    hide_index=False,
    key=editor_key,
)

# Extract synchronized tickers and weights from matrix
raw_tickers = [str(t).strip().upper() for t in edited_matrix_df["Ticker"] if str(t).strip()]
st.session_state["tickers"] = raw_tickers if raw_tickers else ["SPY", "TLT"]
st.session_state["weights"] = {
    str(r["Ticker"]).strip().upper(): float(r["Ponderación (%)"]) / 100.0
    for _, r in edited_matrix_df.iterrows()
    if str(r["Ticker"]).strip()
}
st.session_state["portfolio_matrix_df"] = edited_matrix_df
st.session_state["num_assets"] = len(edited_matrix_df)

current_sum_pct = sum([float(r["Ponderación (%)"]) for _, r in edited_matrix_df.iterrows()])
is_sum_valid = abs(current_sum_pct - 100.0) < 0.05

# Matrix Action Buttons Row
col_badge, col_btn_norm, col_btn_eq, col_btn_ms, col_btn_gmv = st.columns([2, 1.5, 1.5, 2, 2])
with col_badge:
    if is_sum_valid:
        st.markdown(f"<div class='badge-valid'>✅ Suma de Pesos: {current_sum_pct:.2f}% (Válido)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge-warning'>⚠️ Suma de Pesos: {current_sum_pct:.2f}% (Normalizar requerida)</div>", unsafe_allow_html=True)

with col_btn_norm:
    st.button("⚖️ Normalizar a 100%", on_click=_normalize_current_weights, use_container_width=True)

with col_btn_eq:
    st.button("➗ Equiponderar (1/N)", on_click=_apply_equal_weights, use_container_width=True)

with col_btn_ms:
    st.button("💎 Aplicar Máx Sharpe", on_click=_apply_optimal_weights_by_type, args=("max_sharpe",), use_container_width=True)

with col_btn_gmv:
    st.button("🛡️ Aplicar GMV (Mín Riesgo)", on_click=_apply_optimal_weights_by_type, args=("gmv",), use_container_width=True)

# Quick Asset Removal Row
current_matrix_tickers = [str(r["Ticker"]).strip().upper() for _, r in edited_matrix_df.iterrows() if str(r["Ticker"]).strip()]
c_del_label, c_del_sel, c_del_btn, c_del_info = st.columns([1.3, 2.0, 1.8, 3.9])
with c_del_label:
    st.markdown("<div style='padding-top: 8px; font-weight: 600; color: #CBD5E1; font-size: 13px;'>🗑️ Quitar Activo:</div>", unsafe_allow_html=True)

with c_del_sel:
    selected_asset_to_remove = st.selectbox(
        "Seleccionar activo a eliminar",
        options=current_matrix_tickers,
        key="select_asset_to_remove_box",
        label_visibility="collapsed",
    )

with c_del_btn:
    st.button(
        "🗑️ Eliminar Fila",
        on_click=_remove_asset_from_portfolio,
        args=(selected_asset_to_remove,),
        use_container_width=True,
        help="Elimina la fila seleccionada. Si hay CASH en la cartera, su peso se transfiere a CASH; si no, se reparte equitativamente entre los demás activos.",
    )

with c_del_info:
    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    has_cash = any(t in cash_symbols for t in current_matrix_tickers if t != selected_asset_to_remove)
    if has_cash:
        st.caption("💡 Al eliminar, la ponderación irá automáticamente a **CASH** sin alterar los demás.")
    else:
        st.caption("💡 Al eliminar, la ponderación se repartirá **equitativamente** entre todos los restantes.")

st.markdown("---")


# ===========================================================================
# 6. Core Computation & Optimization Engine
# ===========================================================================

if data_source == "Yahoo Finance (En Vivo)":
    try:
        with st.spinner("Descargando precios históricos ajustados de la matriz..."):
            all_req_fetch = list(dict.fromkeys(st.session_state["tickers"] + [benchmark_symbol]))
            raw_df = fetch_asset_data(
                tickers=all_req_fetch,
                start_date=str(start_date),
                end_date=str(end_date),
            )
            clean_prices_df, daily_returns_df = clean_and_align_prices(raw_df)
    except Exception as e:
        st.warning(f"Aviso de descarga yfinance: {e}. Generando datos de referencia de alta precisión.")
        rng = np.random.default_rng(42)
        dates = pd.date_range(start=str(start_date), end=str(end_date), freq="B")
        all_req_fetch = list(dict.fromkeys(st.session_state["tickers"] + [benchmark_symbol]))
        n = len(all_req_fetch)
        base_p = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=(len(dates), n)), axis=0))
        clean_prices_df = pd.DataFrame(base_p, index=dates, columns=all_req_fetch)
        daily_returns_df = clean_prices_df.pct_change().dropna()

if daily_returns_df is None or clean_prices_df is None or daily_returns_df.empty:
    st.info("👋 Por favor verifique los tickers en la matriz y presione 'Descargar / Actualizar Datos'.")
    st.stop()

# Extract benchmark series for Beta & Jensen Alpha calculations
if benchmark_symbol in daily_returns_df.columns:
    benchmark_returns_series = daily_returns_df[benchmark_symbol]
else:
    benchmark_returns_series = daily_returns_df.mean(axis=1)

# Synchronize active portfolio tickers (excluding standalone benchmark if not part of portfolio)
active_tickers = [t for t in st.session_state["tickers"] if t in daily_returns_df.columns]
if not active_tickers:
    active_tickers = [c for c in daily_returns_df.columns if c != benchmark_symbol] or list(daily_returns_df.columns)
st.session_state["tickers"] = active_tickers

port_daily_returns_df = daily_returns_df[active_tickers]
port_clean_prices_df = clean_prices_df[active_tickers]

# Notify if any matrix ticker was missing or dropped
requested_matrix_tickers = [str(r.get("Ticker", "")).strip().upper() for _, r in edited_matrix_df.iterrows() if str(r.get("Ticker", "")).strip()]
missing_tickers = [t for t in requested_matrix_tickers if t not in active_tickers]
if missing_tickers:
    st.warning(f"⚠️ Los activos {missing_tickers} no devolvieron datos en el rango seleccionado. Los cálculos se realizan con los {len(active_tickers)} activos disponibles: {active_tickers}.")

# Individual asset Betas
asset_betas = calculate_asset_betas(port_daily_returns_df, benchmark_returns_series)

# 1. Statistical Modeling
mu_series = calculate_expected_returns(
    returns=port_daily_returns_df,
    method=return_estimator,
    rf=st.session_state["rf_rate"],
)
cov_df, cov_meta = estimate_covariance_matrix(
    returns=port_daily_returns_df,
    method=cov_estimator,
)
psd_cov_df, was_repaired, cond_num = ensure_positive_semidefinite(cov_df)

# 2. Optimization Engine (Max Sharpe & GMV)
rf_val = float(st.session_state["rf_rate"])

# Individual asset Sharpe ratios
asset_sharpes = {}
for t in st.session_state["tickers"]:
    vol_i = float(np.sqrt(psd_cov_df.loc[t, t])) if t in psd_cov_df.index else 0.0
    mu_i = float(mu_series[t]) if t in mu_series.index else rf_val
    if vol_i > 1e-6:
        asset_sharpes[t] = (mu_i - rf_val) / vol_i
    else:
        asset_sharpes[t] = 0.0

ms_res = optimize_maximum_sharpe(
    expected_returns=mu_series.values,
    cov_matrix=psd_cov_df.values,
    rf=rf_val,
    bounds=bounds,
)
gmv_res = optimize_global_minimum_variance(
    cov_matrix=psd_cov_df.values,
    expected_returns=mu_series.values,
    rf=rf_val,
    bounds=bounds,
)

# Store optimal weights in session state for instant fast-action application
st.session_state["opt_weights_max_sharpe"] = ms_res.weights
st.session_state["opt_weights_gmv"] = gmv_res.weights

# 3. Continuous Efficient Frontier Curve Sweep
frontier_res = compute_efficient_frontier(
    expected_returns=mu_series.values,
    cov_matrix=psd_cov_df.values,
    rf=rf_val,
    num_points=100,
    bounds=bounds,
)

# 4. Dirichlet Simplex Monte Carlo Cloud
mc_res = run_weight_space_monte_carlo(
    expected_returns=mu_series.values,
    cov_matrix=psd_cov_df.values,
    rf=rf_val,
    num_portfolios=10000,
    seed=42,
)

# ===========================================================================
# 7. Risk Analytics Computation for Portfolios
# ===========================================================================

# Normalized user weights vector for analytics strictly dimensioned to active_tickers
user_w_vec = np.array([float(st.session_state["weights"].get(t, 1.0 / len(active_tickers))) for t in active_tickers], dtype=np.float64)
if np.sum(user_w_vec) > 0:
    user_w_norm = user_w_vec / np.sum(user_w_vec)
else:
    user_w_norm = np.ones(len(user_w_vec)) / len(user_w_vec)

eq_w_vec = np.ones(len(active_tickers)) / len(active_tickers)

# Compute comprehensive risk metrics
metrics_user = compute_portfolio_risk_metrics(
    weights=user_w_norm,
    daily_returns=port_daily_returns_df,
    expected_returns=mu_series,
    cov_matrix=psd_cov_df,
    rf=rf_val,
    benchmark_returns=benchmark_returns_series,
    benchmark_ticker=benchmark_symbol,
)
metrics_ms = compute_portfolio_risk_metrics(
    weights=ms_res.weights,
    daily_returns=port_daily_returns_df,
    expected_returns=mu_series,
    cov_matrix=psd_cov_df,
    rf=rf_val,
    benchmark_returns=benchmark_returns_series,
    benchmark_ticker=benchmark_symbol,
)
metrics_gmv = compute_portfolio_risk_metrics(
    weights=gmv_res.weights,
    daily_returns=port_daily_returns_df,
    expected_returns=mu_series,
    cov_matrix=psd_cov_df,
    rf=rf_val,
    benchmark_returns=benchmark_returns_series,
    benchmark_ticker=benchmark_symbol,
)
metrics_eq = compute_portfolio_risk_metrics(
    weights=eq_w_vec,
    daily_returns=port_daily_returns_df,
    expected_returns=mu_series,
    cov_matrix=psd_cov_df,
    rf=rf_val,
    benchmark_returns=benchmark_returns_series,
    benchmark_ticker=benchmark_symbol,
)

# Compute correlation matrix globally so all tabs (Metrics/Export, Heatmaps, etc.) have access
d_diag = np.sqrt(np.diag(psd_cov_df.values))
d_diag_safe = np.where(d_diag == 0.0, 1.0, d_diag)
corr_mat = psd_cov_df.values / np.outer(d_diag_safe, d_diag_safe)
np.fill_diagonal(corr_mat, 1.0)
corr_df = pd.DataFrame(corr_mat, index=psd_cov_df.index, columns=psd_cov_df.columns)


# ===========================================================================
# 8. Seven Analytics & Visualizer Dashboard Tabs
# ===========================================================================

tabs = st.tabs([
    "📈 Frontera Eficiente & Optimización",
    "🍩 Asignación de Activos",
    "📑 Métricas Avanzadas & Exportación",
    "💰 Backtest Histórico & Drawdown",
    "🔮 Proyección Monte Carlo (Conos)",
    "🧊 Matrices de Riesgo (Correlación / Covarianza)",
    "⚖️ Comparador Multi-Portafolio",
])


# ---------------------------------------------------------------------------
# TAB 1: Frontera Eficiente & Optimización
# ---------------------------------------------------------------------------
with tabs[0]:
    # Top KPI Metrics Cards
    user_beta_str = f"{metrics_user.beta:.2f}" if metrics_user.beta is not None else "1.00"
    ms_beta_str = f"{metrics_ms.beta:.2f}" if metrics_ms.beta is not None else "1.00"
    gmv_beta_str = f"{metrics_gmv.beta:.2f}" if metrics_gmv.beta is not None else "1.00"

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Cartera Usuario (Actual)</div>
                <div class='metric-value'>{metrics_user.annualized_return:.2%}</div>
                <div class='metric-subtitle'>Vol: {metrics_user.annualized_volatility:.2%} | Sharpe: {metrics_user.sharpe_ratio:.3f} | Beta: {user_beta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Máximo Ratio Sharpe (Tangencia)</div>
                <div class='metric-value' style='color:#00FF66;'>{metrics_ms.annualized_return:.2%}</div>
                <div class='metric-subtitle'>Vol: {metrics_ms.annualized_volatility:.2%} | Sharpe: {metrics_ms.sharpe_ratio:.3f} | Beta: {ms_beta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Mínima Varianza Global (GMV)</div>
                <div class='metric-value' style='color:#FF3366;'>{metrics_gmv.annualized_return:.2%}</div>
                <div class='metric-subtitle'>Vol: {metrics_gmv.annualized_volatility:.2%} | Sharpe: {metrics_gmv.sharpe_ratio:.3f} | Beta: {gmv_beta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Individual asset coords
    ind_assets = {
        t: (float(np.sqrt(psd_cov_df.loc[t, t])), float(mu_series[t]))
        for t in st.session_state["tickers"]
    }

    fig_frontier = plot_efficient_frontier(
        mc_result=mc_res,
        frontier_result=frontier_res,
        user_point=(metrics_user.annualized_volatility, metrics_user.annualized_return, metrics_user.sharpe_ratio),
        individual_assets=ind_assets,
        rf=rf_val,
    )
    st.plotly_chart(fig_frontier, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Asignación de Activos
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🍩 Distribución de Capital por Cartera")
    col_donut, col_bar = st.columns([1, 1.2])

    with col_donut:
        fig_donut = plot_asset_allocation(
            weights={t: float(user_w_norm[i]) for i, t in enumerate(st.session_state["tickers"])},
            title="Ponderaciones Cartera Usuario",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        comp_weights = {
            "Usuario": {t: float(user_w_norm[i]) for i, t in enumerate(st.session_state["tickers"])},
            "Máx Sharpe": {t: float(ms_res.weights[i]) for i, t in enumerate(st.session_state["tickers"])},
            "GMV": {t: float(gmv_res.weights[i]) for i, t in enumerate(st.session_state["tickers"])},
            "1/N (Equip.)": {t: float(eq_w_vec[i]) for i, t in enumerate(st.session_state["tickers"])},
        }
        fig_bar = plot_allocation_comparison(weights_dict=comp_weights)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Allocation breakdown table
    df_alloc_table = pd.DataFrame(
        {
            "Activo / Ticker": st.session_state["tickers"],
            f"Beta Individual (β vs {benchmark_symbol})": [f"{asset_betas.get(t, 1.0):.2f}" for t in st.session_state["tickers"]],
            "Sharpe Individual": [f"{asset_sharpes.get(t, 0.0):.3f}" for t in st.session_state["tickers"]],
            "Cartera Usuario": [f"{w:.2%}" for w in user_w_norm],
            "Máximo Sharpe": [f"{w:.2%}" for w in ms_res.weights],
            "Mínima Varianza (GMV)": [f"{w:.2%}" for w in gmv_res.weights],
            "Equiponderada (1/N)": [f"{w:.2%}" for w in eq_w_vec],
        }
    )
    st.dataframe(df_alloc_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3: Métricas Avanzadas & Exportación
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### 📑 Tabla Comparativa de Métricas de Riesgo y Rendimiento")

    # Construct Master Metrics DataFrame
    metrics_summary_dict = {
        "Retorno Anualizado (Aritmético)": {
            "Cartera Usuario": f"{metrics_user.annualized_return:.2%}",
            "Máximo Sharpe": f"{metrics_ms.annualized_return:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.annualized_return:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.annualized_return:.2%}",
        },
        "Retorno Compuesto (CAGR)": {
            "Cartera Usuario": f"{metrics_user.cagr:.2%}",
            "Máximo Sharpe": f"{metrics_ms.cagr:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.cagr:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.cagr:.2%}",
        },
        "Volatilidad Anualizada (Riesgo)": {
            "Cartera Usuario": f"{metrics_user.annualized_volatility:.2%}",
            "Máximo Sharpe": f"{metrics_ms.annualized_volatility:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.annualized_volatility:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.annualized_volatility:.2%}",
        },
        "Ratio de Sharpe": {
            "Cartera Usuario": f"{metrics_user.sharpe_ratio:.3f}",
            "Máximo Sharpe": f"{metrics_ms.sharpe_ratio:.3f}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.sharpe_ratio:.3f}",
            "Equiponderada (1/N)": f"{metrics_eq.sharpe_ratio:.3f}",
        },
        f"Beta (β vs {benchmark_symbol})": {
            "Cartera Usuario": f"{metrics_user.beta:.2f}" if metrics_user.beta is not None else "N/A",
            "Máximo Sharpe": f"{metrics_ms.beta:.2f}" if metrics_ms.beta is not None else "N/A",
            "Mínima Varianza (GMV)": f"{metrics_gmv.beta:.2f}" if metrics_gmv.beta is not None else "N/A",
            "Equiponderada (1/N)": f"{metrics_eq.beta:.2f}" if metrics_eq.beta is not None else "N/A",
        },
        "Alfa de Jensen Anualizada (α)": {
            "Cartera Usuario": f"{metrics_user.alpha_jensen:.2%}" if metrics_user.alpha_jensen is not None else "N/A",
            "Máximo Sharpe": f"{metrics_ms.alpha_jensen:.2%}" if metrics_ms.alpha_jensen is not None else "N/A",
            "Mínima Varianza (GMV)": f"{metrics_gmv.alpha_jensen:.2%}" if metrics_gmv.alpha_jensen is not None else "N/A",
            "Equiponderada (1/N)": f"{metrics_eq.alpha_jensen:.2%}" if metrics_eq.alpha_jensen is not None else "N/A",
        },
        "Coeficiente de Determinación (R²)": {
            "Cartera Usuario": f"{metrics_user.r_squared:.2%}" if metrics_user.r_squared is not None else "N/A",
            "Máximo Sharpe": f"{metrics_ms.r_squared:.2%}" if metrics_ms.r_squared is not None else "N/A",
            "Mínima Varianza (GMV)": f"{metrics_gmv.r_squared:.2%}" if metrics_gmv.r_squared is not None else "N/A",
            "Equiponderada (1/N)": f"{metrics_eq.r_squared:.2%}" if metrics_eq.r_squared is not None else "N/A",
        },
        "Ratio de Sortino": {
            "Cartera Usuario": f"{metrics_user.sortino_ratio:.3f}",
            "Máximo Sharpe": f"{metrics_ms.sortino_ratio:.3f}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.sortino_ratio:.3f}",
            "Equiponderada (1/N)": f"{metrics_eq.sortino_ratio:.3f}",
        },
        "Ratio de Calmar": {
            "Cartera Usuario": f"{metrics_user.calmar_ratio:.3f}",
            "Máximo Sharpe": f"{metrics_ms.calmar_ratio:.3f}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.calmar_ratio:.3f}",
            "Equiponderada (1/N)": f"{metrics_eq.calmar_ratio:.3f}",
        },
        "Máximo Drawdown (MDD)": {
            "Cartera Usuario": f"{metrics_user.max_drawdown:.2%}",
            "Máximo Sharpe": f"{metrics_ms.max_drawdown:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.max_drawdown:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.max_drawdown:.2%}",
        },
        "VaR 95% Histórico (1 Día)": {
            "Cartera Usuario": f"{metrics_user.var_95_hist:.2%}",
            "Máximo Sharpe": f"{metrics_ms.var_95_hist:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.var_95_hist:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.var_95_hist:.2%}",
        },
        "CVaR 95% Histórico (1 Día)": {
            "Cartera Usuario": f"{metrics_user.cvar_95_hist:.2%}",
            "Máximo Sharpe": f"{metrics_ms.cvar_95_hist:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.cvar_95_hist:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.cvar_95_hist:.2%}",
        },
        "VaR 95% Mensual (21 Días)": {
            "Cartera Usuario": f"{metrics_user.var_95_monthly:.2%}",
            "Máximo Sharpe": f"{metrics_ms.var_95_monthly:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.var_95_monthly:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.var_95_monthly:.2%}",
        },
        "CVaR 95% Mensual (21 Días)": {
            "Cartera Usuario": f"{metrics_user.cvar_95_monthly:.2%}",
            "Máximo Sharpe": f"{metrics_ms.cvar_95_monthly:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.cvar_95_monthly:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.cvar_95_monthly:.2%}",
        },
        "VaR 95% Anual (252 Días)": {
            "Cartera Usuario": f"{metrics_user.var_95_annual:.2%}",
            "Máximo Sharpe": f"{metrics_ms.var_95_annual:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.var_95_annual:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.var_95_annual:.2%}",
        },
        "CVaR 95% Anual (252 Días)": {
            "Cartera Usuario": f"{metrics_user.cvar_95_annual:.2%}",
            "Máximo Sharpe": f"{metrics_ms.cvar_95_annual:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.cvar_95_annual:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.cvar_95_annual:.2%}",
        },
        "VaR 95% Paramétrico (1 Día)": {
            "Cartera Usuario": f"{metrics_user.var_95_param:.2%}",
            "Máximo Sharpe": f"{metrics_ms.var_95_param:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.var_95_param:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.var_95_param:.2%}",
        },
        "CVaR 95% Paramétrico (1 Día)": {
            "Cartera Usuario": f"{metrics_user.cvar_95_param:.2%}",
            "Máximo Sharpe": f"{metrics_ms.cvar_95_param:.2%}",
            "Mínima Varianza (GMV)": f"{metrics_gmv.cvar_95_param:.2%}",
            "Equiponderada (1/N)": f"{metrics_eq.cvar_95_param:.2%}",
        },
    }

    df_metrics_display = pd.DataFrame(
        [
            {"Métrica": k, **v} for k, v in metrics_summary_dict.items()
        ]
    )
    st.dataframe(df_metrics_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📥 Centro de Exportación de Resultados")

    # Prepare downloads
    csv_metrics = export_summary_csv(df_metrics_display)
    df_weights_raw = pd.DataFrame(
        {
            "Ticker": st.session_state["tickers"],
            "Usuario": user_w_norm,
            "Max Sharpe": ms_res.weights,
            "GMV": gmv_res.weights,
            "Equiponderada": eq_w_vec,
        }
    )
    csv_weights = export_weights_csv(df_weights_raw)
    csv_corr = export_correlation_csv(corr_df)

    # Excel Workbook bytes
    excel_bytes = export_full_excel(
        metrics_dict={"Cartera Usuario": metrics_user, "Máximo Sharpe": metrics_ms, "GMV": metrics_gmv, "Equiponderada": metrics_eq},
        weights_dict=df_weights_raw,
        corr_matrix=corr_df,
        cov_matrix=psd_cov_df,
        wealth_df=pd.DataFrame({k: v.cumulative_wealth for k, v in [("Usuario", metrics_user), ("Max Sharpe", metrics_ms), ("GMV", metrics_gmv)]}),
    )

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    with exp_col1:
        st.download_button(
            "📄 Descargar Resumen (CSV)",
            data=csv_metrics,
            file_name=f"resumen_metricas_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col2:
        st.download_button(
            "📊 Descargar Ponderaciones (CSV)",
            data=csv_weights,
            file_name=f"ponderaciones_optimas_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col3:
        st.download_button(
            "🧊 Descargar Correlación (CSV)",
            data=csv_corr,
            file_name=f"matriz_correlacion_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col4:
        st.download_button(
            "📗 Reporte Completo Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"optimizacion_frontera_eficiente_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# TAB 4: Backtest Histórico & Drawdown
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### 💰 Evolución Patrimonial ($10,000 USD Base) y Caídas de Valor")
    ret_dict_backtest = {
        "Cartera Usuario": port_daily_returns_df.values @ user_w_norm,
        "Máximo Sharpe": port_daily_returns_df.values @ ms_res.weights,
        "Mínima Varianza (GMV)": port_daily_returns_df.values @ gmv_res.weights,
        "Equiponderada (1/N)": port_daily_returns_df.values @ eq_w_vec,
        f"Benchmark ({benchmark_symbol})": benchmark_returns_series.values,
    }
    # Pass Series with DatetimeIndex
    ret_series_dict = {
        k: pd.Series(v, index=port_daily_returns_df.index) for k, v in ret_dict_backtest.items()
    }
    fig_backtest = plot_historical_backtest(returns_dict=ret_series_dict, initial_capital=10000.0)
    st.plotly_chart(fig_backtest, use_container_width=True)

    # Drawdown Metrics Table
    st.markdown("#### 📉 Resumen de Drawdown Histórico")
    dd_summary = pd.DataFrame(
        {
            "Cartera": ["Cartera Usuario", "Máximo Sharpe", "Mínima Varianza (GMV)", "Equiponderada (1/N)"],
            "Máximo Drawdown": [
                f"{metrics_user.max_drawdown:.2%}",
                f"{metrics_ms.max_drawdown:.2%}",
                f"{metrics_gmv.max_drawdown:.2%}",
                f"{metrics_eq.max_drawdown:.2%}",
            ],
            "Fecha Pico (Peak)": [
                str(metrics_user.peak_date)[:10],
                str(metrics_ms.peak_date)[:10],
                str(metrics_gmv.peak_date)[:10],
                str(metrics_eq.peak_date)[:10],
            ],
            "Fecha Fondo (Valley)": [
                str(metrics_user.valley_date)[:10],
                str(metrics_ms.valley_date)[:10],
                str(metrics_gmv.valley_date)[:10],
                str(metrics_eq.valley_date)[:10],
            ],
            "Días Recuperación": [
                metrics_user.recovery_days or "En recuperación",
                metrics_ms.recovery_days or "En recuperación",
                metrics_gmv.recovery_days or "En recuperación",
                metrics_eq.recovery_days or "En recuperación",
            ],
            "Ratio de Calmar": [
                f"{metrics_user.calmar_ratio:.3f}",
                f"{metrics_ms.calmar_ratio:.3f}",
                f"{metrics_gmv.calmar_ratio:.3f}",
                f"{metrics_eq.calmar_ratio:.3f}",
            ],
        }
    )
    st.dataframe(dd_summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 5: Proyección Monte Carlo (Conos de Probabilidad)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 🔮 Simulación Estocástica de Trayectorias a Futuro")
    mc_c1, mc_c2, mc_c3 = st.columns(3)
    with mc_c1:
        horizon_years = st.slider("Horizonte de Simulación (Años)", min_value=1, max_value=5, value=3, step=1)
    with mc_c2:
        num_sims = st.slider("Número de Trayectorias", min_value=500, max_value=5000, value=2000, step=500)
    with mc_c3:
        mc_model = st.selectbox(
            "Modelo Estocástico",
            options=["gbm", "bootstrap"],
            format_func=lambda x: "Movimiento Browniano Geométrico (GBM)" if x == "gbm" else "Bootstrapping Histórico por Bloques",
        )

    with st.spinner("Ejecutando simulación estocástica multi-activo..."):
        traj_res = run_trajectory_monte_carlo(
            expected_returns=mu_series.values,
            cov_matrix=psd_cov_df.values,
            weights=user_w_norm,
            initial_capital=10000.0,
            years=horizon_years,
            num_simulations=num_sims,
            model=mc_model,
            historical_returns=daily_returns_df.values,
            seed=42,
        )

    fig_cones = plot_monte_carlo_cones(
        trajectory_result=traj_res,
        user_label="Cartera Usuario",
        initial_wealth=10000.0,
    )
    st.plotly_chart(fig_cones, use_container_width=True)

    # Future wealth scenarios
    st.markdown("#### 🎯 Escenarios de Capital Final Proyectado ($10,000 USD Inicial)")
    sc_c1, sc_c2, sc_c3 = st.columns(3)
    with sc_c1:
        p5_val = float(traj_res.percentile_5[-1])
        st.metric("Escenario Adverso (Percentil 5%)", f"${p5_val:,.0f}", f"{(p5_val - 10000)/100:.1f}% Total")
    with sc_c2:
        p50_val = float(traj_res.percentile_50[-1])
        st.metric("Escenario Esperado (Mediana P50)", f"${p50_val:,.0f}", f"{(p50_val - 10000)/100:.1f}% Total")
    with sc_c3:
        p95_val = float(traj_res.percentile_95[-1])
        st.metric("Escenario Favorable (Percentil 95%)", f"${p95_val:,.0f}", f"{(p95_val - 10000)/100:.1f}% Total")


# ---------------------------------------------------------------------------
# TAB 6: Matrices de Riesgo (Correlación / Covarianza)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 🧊 Estructura de Dependencia y Covarianza")
    col_corr, col_cov = st.columns(2)

    with col_corr:
        fig_corr = plot_correlation_heatmap(corr_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_cov:
        fig_cov = plot_covariance_heatmap(psd_cov_df)
        st.plotly_chart(fig_cov, use_container_width=True)

    # Diagnostics Box
    st.markdown("#### 🔬 Diagnósticos Numéricos y Estabilidad")
    diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)
    with diag_c1:
        st.metric("Número de Condición", f"{cond_num:.2f}")
    with diag_c2:
        st.metric("Shrinkage Intensity (δ*)", f"{cov_meta.get('shrinkage_delta', 0.0) or 0.0:.4f}")
    with diag_c3:
        min_eig = float(np.min(np.linalg.eigvalsh(psd_cov_df.values)))
        st.metric("Autovalor Mínimo (λ_min)", f"{min_eig:.6f}")
    with diag_c4:
        st.metric("Reparación Higham PSD", "No requerida" if not was_repaired else "Aplicada ✅")


# ---------------------------------------------------------------------------
# TAB 7: Comparador Multi-Portafolio
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("### ⚖️ Comparativa Avanzada entre Múltiples Portafolios")
    st.caption("Compara el rendimiento histórico, la volatilidad, ratios de Sharpe/Sortino y la diversificación de cualquier combinación de carteras.")

    # Build universe of all candidate portfolios
    all_compare_options: Dict[str, Dict[str, Any]] = {
        "🎯 Cartera Usuario (Actual)": {
            "name": "Cartera Usuario",
            "tickers": st.session_state["tickers"],
            "weights": {t: float(user_w_norm[i]) for i, t in enumerate(st.session_state["tickers"])},
        },
        "💎 Máximo Sharpe (Tangencia)": {
            "name": "Máx Sharpe",
            "tickers": st.session_state["tickers"],
            "weights": {t: float(ms_res.weights[i]) for i, t in enumerate(st.session_state["tickers"])},
        },
        "🛡️ Mínima Varianza Global (GMV)": {
            "name": "Mín Varianza (GMV)",
            "tickers": st.session_state["tickers"],
            "weights": {t: float(gmv_res.weights[i]) for i, t in enumerate(st.session_state["tickers"])},
        },
        "➗ Equiponderada (1/N)": {
            "name": "Equiponderada (1/N)",
            "tickers": st.session_state["tickers"],
            "weights": {t: float(eq_w_vec[i]) for i, t in enumerate(st.session_state["tickers"])},
        },
    }

    # Add canonical presets
    for p_key, p_val in PRESETS.items():
        all_compare_options[f"🏛️ {p_val['name']}"] = {
            "name": p_val["name"],
            "tickers": p_val["tickers"],
            "weights": p_val["weights"],
        }

    # Add user saved custom portfolios
    saved_p_map = load_saved_portfolios(get_active_user_id())
    for s_id, s_val in saved_p_map.items():
        all_compare_options[f"📁 {s_val['name']}"] = {
            "name": s_val["name"],
            "tickers": s_val["tickers"],
            "weights": s_val["weights"],
        }

    # Multiselect for comparison
    default_selected = [
        "🎯 Cartera Usuario (Actual)",
        "💎 Máximo Sharpe (Tangencia)",
        "🏛️ Clásico 60/40",
        "🏛️ All-Weather (Ray Dalio)",
    ]
    valid_defaults = [k for k in default_selected if k in all_compare_options]

    selected_portfolio_keys = st.multiselect(
        "Selecciona las carteras a comparar en tiempo real:",
        options=list(all_compare_options.keys()),
        default=valid_defaults,
        key="multi_portfolio_comparator_select",
    )

    if len(selected_portfolio_keys) < 2:
        st.info("👋 Por favor selecciona al menos 2 portafolios en el selector de arriba para generar el análisis comparativo.")
    else:
        # Collect union of all tickers
        all_req_tickers = set()
        for k in selected_portfolio_keys:
            all_req_tickers.update(all_compare_options[k]["tickers"])
        all_req_tickers = sorted(list(all_req_tickers))

        # Check which tickers are already in daily_returns_df
        available_tickers = set(daily_returns_df.columns)
        missing_tickers = [t for t in all_req_tickers if t not in available_tickers]

        if missing_tickers and data_source == "Yahoo Finance (En Vivo)":
            with st.spinner(f"Descargando datos históricos para activos adicionales de las carteras comparadas: {missing_tickers}..."):
                try:
                    all_dl_df = fetch_asset_data(tickers=all_req_tickers, start_date=str(start_date), end_date=str(end_date))
                    _, comp_daily_returns_df = clean_and_align_prices(all_dl_df)
                except Exception:
                    comp_daily_returns_df = daily_returns_df
        else:
            comp_daily_returns_df = daily_returns_df

        # Compute metrics and returns for each selected portfolio
        compare_returns_dict = {}
        compare_metrics_dict = {}
        compare_scatter_data = {}
        compare_weights_dict = {}

        for k in selected_portfolio_keys:
            p_info = all_compare_options[k]
            p_name = p_info["name"]
            p_tickers = p_info["tickers"]
            p_w_raw = p_info["weights"]

            # Filter to tickers that exist in comp_daily_returns_df
            active_p_tickers = [t for t in p_tickers if t in comp_daily_returns_df.columns]
            if not active_p_tickers:
                continue

            raw_w_sub = np.array([p_w_raw.get(t, 0.0) for t in active_p_tickers], dtype=np.float64)
            sum_sub = np.sum(raw_w_sub)
            norm_w_sub = raw_w_sub / sum_sub if sum_sub > 0 else np.ones(len(active_p_tickers)) / len(active_p_tickers)

            p_rets = comp_daily_returns_df[active_p_tickers].dot(norm_w_sub)
            compare_returns_dict[p_name] = p_rets
            compare_weights_dict[p_name] = {t: float(norm_w_sub[i]) for i, t in enumerate(active_p_tickers)}

            sub_cov = comp_daily_returns_df[active_p_tickers].cov().values * 252.0
            sub_mu = comp_daily_returns_df[active_p_tickers].mean().values * 252.0

            m = compute_portfolio_risk_metrics(
                weights=norm_w_sub,
                daily_returns=comp_daily_returns_df[active_p_tickers],
                expected_returns=pd.Series(sub_mu, index=active_p_tickers),
                cov_matrix=pd.DataFrame(sub_cov, index=active_p_tickers, columns=active_p_tickers),
                rf=rf_val,
                benchmark_returns=benchmark_returns_series,
                benchmark_ticker=benchmark_symbol,
            )
            compare_metrics_dict[p_name] = m
            compare_scatter_data[p_name] = {
                "return": m.annualized_return,
                "volatility": m.annualized_volatility,
                "sharpe": m.sharpe_ratio,
            }

        # 1. Side-by-side KPI Comparison Table
        st.markdown("#### 📊 Tabla Comparativa de Indicadores Clave")
        comp_table_data = []
        for p_name, m in compare_metrics_dict.items():
            comp_table_data.append({
                "Portafolio": p_name,
                "Retorno Anual": f"{m.annualized_return:.2%}",
                "Volatilidad": f"{m.annualized_volatility:.2%}",
                "Ratio Sharpe": f"{m.sharpe_ratio:.3f}",
                f"Beta (β vs {benchmark_symbol})": f"{m.beta:.2f}" if m.beta is not None else "N/A",
                "Alfa Jensen (α)": f"{m.alpha_jensen:.2%}" if m.alpha_jensen is not None else "N/A",
                "Ratio Sortino": f"{m.sortino_ratio:.3f}",
                "Ratio Calmar": f"{m.calmar_ratio:.3f}",
                "Máx Drawdown": f"{m.max_drawdown:.2%}",
                "VaR 95% (Histórico)": f"{m.var_95_hist:.2%}",
                "CVaR 95% (Histórico)": f"{m.cvar_95_hist:.2%}",
            })
        st.dataframe(pd.DataFrame(comp_table_data), use_container_width=True, hide_index=True)

        # 2. Charts: Backtest and Scatter
        col_c_chart1, col_c_chart2 = st.columns([1.2, 1.0])
        with col_c_chart1:
            fig_comp_backtest = plot_historical_backtest(returns_dict=compare_returns_dict, initial_capital=10000.0)
            st.plotly_chart(fig_comp_backtest, use_container_width=True)

        with col_c_chart2:
            fig_comp_scatter = plot_portfolio_comparison_scatter(compare_scatter_data, rf=rf_val)
            st.plotly_chart(fig_comp_scatter, use_container_width=True)

        # 3. Asset Allocation Grouped Bar Comparison
        st.markdown("#### 🍩 Comparativa de Asignación de Activos entre Portafolios")
        fig_comp_alloc = plot_allocation_comparison(weights_dict=compare_weights_dict)
        st.plotly_chart(fig_comp_alloc, use_container_width=True)
