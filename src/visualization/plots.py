"""
Interactive Plotly Visualizers Module for Frontera Eficiente.

Provides interactive charts:
1. Markowitz Efficient Frontier + Capital Allocation Line (CAL) + Monte Carlo Cloud + Optimal Points
2. Asset Allocation Donut Chart & Grouped Comparative Bar Chart
3. Interactive Correlation & Annualized Covariance Heatmaps with Cell Annotations
4. Historical Wealth Index Growth ($10,000 USD) with Underwater Drawdown Subplot
5. Monte Carlo Future Wealth Stochastic Cones (5%, 25%, 50%, 75%, 95% Percentile Fan Chart)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Standard UI color palette for dark financial dashboards
THEME_DARK = {
    "paper_bgcolor": "#0e1117",
    "plot_bgcolor": "#161b26",
    "font_color": "#FAFAFA",
    "grid_color": "#2d3748",
    "accent_cyan": "#00F0FF",
    "accent_green": "#00FF66",
    "accent_magenta": "#FF3366",
    "accent_gold": "#FFCC00",
    "accent_orange": "#FFA500",
    "accent_purple": "#9D4EDD",
}

PALETTE_SERIES = [
    "#00F0FF",  # Cyan
    "#00FF66",  # Green
    "#FF3366",  # Pink/Red
    "#FFCC00",  # Yellow
    "#9D4EDD",  # Purple
    "#3A86FF",  # Blue
    "#FF8800",  # Orange
    "#06D6A0",  # Mint
    "#118AB2",  # Ocean
    "#EF476F",  # Rose
]


# ===========================================================================
# 1. Markowitz Efficient Frontier Plot
# ===========================================================================

def plot_efficient_frontier(
    mc_result: Optional[Any] = None,
    frontier_result: Optional[Any] = None,
    user_point: Optional[Union[Tuple[float, float, float], Dict[str, float]]] = None,
    individual_assets: Optional[Union[Dict[str, Tuple[float, float]], pd.DataFrame]] = None,
    rf: float = 0.04,
) -> go.Figure:
    """
    Generate interactive Markowitz Efficient Frontier Plotly Figure.

    Layers:
    1. Monte Carlo Random Portfolio Cloud (go.Scattergl colored by Sharpe ratio)
    2. Continuous Markowitz Efficient Frontier curve (go.Scatter line)
    3. Capital Allocation Line (CAL) (go.Scatter dashed line)
    4. Global Minimum Variance (GMV) portfolio star
    5. Maximum Sharpe tangency portfolio diamond
    6. User custom portfolio marker
    7. Individual asset markers with text labels

    Parameters
    ----------
    mc_result : Optional[Any], optional
        Monte Carlo simulation result (WeightMonteCarloResult or tuple (vols, rets, sharpes)).
    frontier_result : Optional[Any], optional
        Frontier sweep result (EfficientFrontierResult or tuple (vols, rets)).
    user_point : Optional[tuple | dict], optional
        User portfolio (volatility, expected_return, sharpe_ratio).
    individual_assets : Optional[dict | pd.DataFrame], optional
        Individual asset coordinates {ticker: (vol, ret)}.
    rf : float, default 0.04
        Risk-free rate.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure.
    """
    fig = go.Figure()

    # 1. Monte Carlo Cloud
    if mc_result is not None:
        mc_vols, mc_rets, mc_sharpes = _extract_mc_data(mc_result)
        if len(mc_vols) > 0:
            fig.add_trace(
                go.Scattergl(
                    x=mc_vols,
                    y=mc_rets,
                    mode="markers",
                    name="Carteras Simuladas",
                    marker=dict(
                        size=3.5,
                        opacity=0.45,
                        color=mc_sharpes,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Ratio Sharpe", font=dict(color=THEME_DARK["font_color"], size=12)),
                            tickfont=dict(color=THEME_DARK["font_color"]),
                            thickness=14,
                            len=0.75,
                        ),
                    ),
                    hovertemplate=(
                        "<b>Cartera Simulada</b><br>"
                        "Retorno Esperado: %{y:.2%}<br>"
                        "Volatilidad Anual: %{x:.2%}<br>"
                        "Ratio Sharpe: %{marker.color:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

    # 2. Continuous Efficient Frontier Curve
    f_vols, f_rets = _extract_frontier_data(frontier_result)
    if len(f_vols) > 0:
        fig.add_trace(
            go.Scatter(
                x=f_vols,
                y=f_rets,
                mode="lines",
                name="Frontera Eficiente (Markowitz)",
                line=dict(color=THEME_DARK["accent_cyan"], width=3.5),
                hovertemplate=(
                    "<b>Frontera Eficiente</b><br>"
                    "Retorno Objetivo: %{y:.2%}<br>"
                    "Volatilidad Mínima: %{x:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    # 3. Maximum Sharpe & GMV Points from frontier_result
    ms_vol, ms_ret, ms_sr = _extract_optimal_point(frontier_result, "max_sharpe")
    gmv_vol, gmv_ret, gmv_sr = _extract_optimal_point(frontier_result, "gmv")

    # Capital Allocation Line (CAL)
    if ms_vol is not None and ms_ret is not None and ms_vol > 0:
        max_x = max(float(np.max(f_vols)) if len(f_vols) > 0 else 0.40, ms_vol * 1.5, 0.40)
        cal_x = np.linspace(0.0, max_x, 50)
        cal_slope = (ms_ret - rf) / ms_vol
        cal_y = rf + cal_slope * cal_x

        fig.add_trace(
            go.Scatter(
                x=cal_x,
                y=cal_y,
                mode="lines",
                name="Línea Asignación Capital (CAL)",
                line=dict(color=THEME_DARK["accent_orange"], width=2.0, dash="dash"),
                hovertemplate=(
                    "<b>Línea CAL (Rf=%.1f%%)</b><br>"
                    "Retorno: %%{y:.2%%}<br>"
                    "Volatilidad: %%{x:.2%%}"
                    "<extra></extra>"
                ) % (rf * 100),
            )
        )

    # GMV Marker
    if gmv_vol is not None and gmv_ret is not None:
        fig.add_trace(
            go.Scatter(
                x=[gmv_vol],
                y=[gmv_ret],
                mode="markers",
                name="Mínima Varianza Global (GMV)",
                marker=dict(
                    symbol="star",
                    size=16,
                    color=THEME_DARK["accent_magenta"],
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=(
                    "<b>Cartera GMV</b><br>"
                    "Retorno: %{y:.2%}<br>"
                    "Volatilidad: %{x:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    # Max Sharpe Marker
    if ms_vol is not None and ms_ret is not None:
        fig.add_trace(
            go.Scatter(
                x=[ms_vol],
                y=[ms_ret],
                mode="markers",
                name="Máximo Ratio Sharpe",
                marker=dict(
                    symbol="diamond",
                    size=16,
                    color=THEME_DARK["accent_green"],
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=(
                    "<b>Cartera Máx Sharpe</b><br>"
                    "Retorno: %{y:.2%}<br>"
                    "Volatilidad: %{x:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    # 4. User Portfolio Point
    if user_point is not None:
        u_vol, u_ret, u_sr = _extract_user_point(user_point)
        if u_vol is not None and u_ret is not None:
            fig.add_trace(
                go.Scatter(
                    x=[u_vol],
                    y=[u_ret],
                    mode="markers",
                    name="Cartera Usuario (Actual)",
                    marker=dict(
                        symbol="circle-dot",
                        size=18,
                        color=THEME_DARK["accent_gold"],
                        line=dict(color="white", width=2.0),
                    ),
                    hovertemplate=(
                        "<b>Cartera Actual Usuario</b><br>"
                        "Retorno: %{y:.2%}<br>"
                        "Volatilidad: %{x:.2%}<br>"
                        "Ratio Sharpe: " + (f"{u_sr:.3f}" if u_sr is not None else "N/A") +
                        "<extra></extra>"
                    ),
                )
            )

    # 5. Individual Assets
    if individual_assets is not None:
        _add_individual_assets_traces(fig, individual_assets, rf)

    # Layout styling
    fig.update_layout(
        title=dict(
            text="<b>Frontera Eficiente de Markowitz & Espacio Riesgo-Retorno</b>",
            font=dict(size=18, color=THEME_DARK["font_color"]),
            x=0.02,
        ),
        xaxis=dict(
            title="Volatilidad Anualizada (Riesgo σ)",
            tickformat=".1%",
            gridcolor=THEME_DARK["grid_color"],
            zerolinecolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
        ),
        yaxis=dict(
            title="Retorno Esperado Anualizado (μ)",
            tickformat=".1%",
            gridcolor=THEME_DARK["grid_color"],
            zerolinecolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        legend=dict(
            font=dict(color=THEME_DARK["font_color"], size=11),
            bgcolor="rgba(22, 27, 38, 0.8)",
            bordercolor=THEME_DARK["grid_color"],
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        hovermode="closest",
        margin=dict(l=60, r=40, t=60, b=50),
        height=580,
    )

    return fig


# ===========================================================================
# 2. Asset Allocation Visualizers (Donut & Comparative Bar)
# ===========================================================================

def plot_asset_allocation(
    weights: Union[Dict[str, float], pd.Series, np.ndarray],
    tickers: Optional[List[str]] = None,
    title: str = "Asignación de Activos (Ponderación %)",
) -> go.Figure:
    """
    Generate modern interactive Donut chart for asset allocation.

    Parameters
    ----------
    weights : dict | pd.Series | np.ndarray
        Asset weights.
    tickers : Optional[List[str]], optional
        Ticker labels if weights is array.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly Figure with donut Pie trace (hole=0.45).
    """
    labels, vals = _extract_weights_labels_vals(weights, tickers)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=vals,
                hole=0.45,
                textinfo="label+percent",
                insidetextorientation="radial",
                marker=dict(
                    colors=PALETTE_SERIES[: len(labels)],
                    line=dict(color="#0e1117", width=2),
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Ponderación: %{percent:.2%}<br>"
                    "Peso Exacto: %{value:.4f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color=THEME_DARK["font_color"], size=16), x=0.02),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"]),
        legend=dict(
            font=dict(color=THEME_DARK["font_color"], size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=30, r=30, t=50, b=50),
        height=380,
    )
    return fig


def plot_asset_allocation_donut(
    weights: Union[Dict[str, float], pd.Series, np.ndarray],
    tickers: Optional[List[str]] = None,
    title: str = "Asignación de Activos (Ponderación %)",
) -> go.Figure:
    """Alias for plot_asset_allocation."""
    return plot_asset_allocation(weights=weights, tickers=tickers, title=title)


def plot_allocation_comparison(
    weights_dict: Optional[Union[Dict[str, Dict[str, float]], pd.DataFrame]] = None,
    user_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    gmv_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    max_sharpe_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    eq_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    tickers: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate grouped comparative bar chart comparing multiple portfolio allocations.

    Parameters
    ----------
    weights_dict : Optional[dict | pd.DataFrame], optional
        Mapping of {Portfolio_Name: {Ticker: weight}} or DataFrame.
    user_weights, gmv_weights, max_sharpe_weights, eq_weights : Optional[dict | array], optional
        Explicit portfolio weights.
    tickers : Optional[List[str]], optional
        Ordered asset tickers list.

    Returns
    -------
    go.Figure
        Plotly Figure with barmode='group'.
    """
    portfolios_data: Dict[str, Dict[str, float]] = {}

    if weights_dict is not None:
        if isinstance(weights_dict, pd.DataFrame):
            for col in weights_dict.columns:
                portfolios_data[str(col)] = dict(weights_dict[col])
        elif isinstance(weights_dict, dict):
            for p_name, w_map in weights_dict.items():
                if isinstance(w_map, dict):
                    portfolios_data[p_name] = {str(k): float(v) for k, v in w_map.items()}
                else:
                    labels, vals = _extract_weights_labels_vals(w_map, tickers)
                    portfolios_data[p_name] = dict(zip(labels, vals))

    if user_weights is not None:
        labels, vals = _extract_weights_labels_vals(user_weights, tickers)
        portfolios_data["Usuario"] = dict(zip(labels, vals))
    if max_sharpe_weights is not None:
        labels, vals = _extract_weights_labels_vals(max_sharpe_weights, tickers)
        portfolios_data["Máximo Sharpe"] = dict(zip(labels, vals))
    if gmv_weights is not None:
        labels, vals = _extract_weights_labels_vals(gmv_weights, tickers)
        portfolios_data["GMV"] = dict(zip(labels, vals))
    if eq_weights is not None:
        labels, vals = _extract_weights_labels_vals(eq_weights, tickers)
        portfolios_data["Equiponderada (1/N)"] = dict(zip(labels, vals))

    if not portfolios_data:
        portfolios_data = {"Sin datos": {"Asset_1": 1.0}}

    # Union of all tickers in order
    all_tickers: List[str] = []
    if tickers:
        all_tickers = list(tickers)
    for p_map in portfolios_data.values():
        for t in p_map.keys():
            if t not in all_tickers:
                all_tickers.append(t)

    fig = go.Figure()
    for idx, (p_name, p_map) in enumerate(portfolios_data.items()):
        y_vals = [p_map.get(t, 0.0) for t in all_tickers]
        color = PALETTE_SERIES[idx % len(PALETTE_SERIES)]
        fig.add_trace(
            go.Bar(
                name=p_name,
                x=all_tickers,
                y=y_vals,
                marker_color=color,
                hovertemplate=(
                    f"<b>{p_name}</b><br>"
                    "Activo: %{x}<br>"
                    "Ponderación: %{y:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Comparativa de Ponderaciones por Cartera</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.02,
        ),
        barmode="group",
        xaxis=dict(
            title="Activo / Ticker",
            gridcolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
        ),
        yaxis=dict(
            title="Ponderación",
            tickformat=".1%",
            gridcolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
            range=[0.0, max(1.0, max(max(p.values(), default=0.0) for p in portfolios_data.values()) * 1.15)],
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        legend=dict(
            font=dict(color=THEME_DARK["font_color"], size=11),
            bgcolor="rgba(22, 27, 38, 0.8)",
            bordercolor=THEME_DARK["grid_color"],
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        height=380,
    )
    return fig


def plot_allocation_comparison_bar(
    user_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    gmv_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    max_sharpe_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    eq_weights: Optional[Union[Dict[str, float], pd.Series, np.ndarray]] = None,
    weights_dict: Optional[Union[Dict[str, Dict[str, float]], pd.DataFrame]] = None,
    tickers: Optional[List[str]] = None,
) -> go.Figure:
    """Alias matching explicit parameter signature."""
    return plot_allocation_comparison(
        weights_dict=weights_dict,
        user_weights=user_weights,
        gmv_weights=gmv_weights,
        max_sharpe_weights=max_sharpe_weights,
        eq_weights=eq_weights,
        tickers=tickers,
    )


# ===========================================================================
# 3. Risk Heatmaps (Correlation & Covariance)
# ===========================================================================

def plot_correlation_heatmap(
    corr_matrix: Union[pd.DataFrame, np.ndarray],
    tickers: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate interactive Correlation Matrix Heatmap with numeric text annotations.

    Parameters
    ----------
    corr_matrix : pd.DataFrame | np.ndarray
        Correlation matrix in [-1.0, 1.0].
    tickers : Optional[List[str]], optional
        Ticker names if corr_matrix is np.ndarray.

    Returns
    -------
    go.Figure
        Plotly Heatmap Figure.
    """
    labels, mat = _extract_matrix_and_labels(corr_matrix, tickers)

    # Format annotations
    text_annotations = [[f"{val:.2f}" for val in row] for row in mat]

    # High-contrast dark diverging scale: blue (-1) -> dark slate (0) -> crimson (+1)
    # Ensures white text (#FFFFFF) has ultra-clear visibility across the entire range
    corr_colorscale = [
        [0.0, "#0d3b66"],   # Deep Blue (-1.0)
        [0.25, "#1f4e79"],  # Muted Blue (-0.5)
        [0.5, "#161b26"],   # Dark Slate (0.0 - High Contrast with White Text)
        [0.75, "#8b2635"],  # Muted Crimson (+0.5)
        [1.0, "#d90429"],   # Vivid Crimson (+1.0)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale=corr_colorscale,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#FFFFFF", family="Arial, sans-serif"),
            colorbar=dict(
                title=dict(text="Correlación (ρ)", font=dict(color=THEME_DARK["font_color"])),
                tickfont=dict(color=THEME_DARK["font_color"]),
                thickness=14,
            ),
            hovertemplate=(
                "<b>Correlación</b><br>"
                "Activo X: %{x}<br>"
                "Activo Y: %{y}<br>"
                "Coeficiente ρ: %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Matriz de Correlación entre Activos</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.02,
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        xaxis=dict(color=THEME_DARK["font_color"]),
        yaxis=dict(color=THEME_DARK["font_color"], autorange="reversed"),
        margin=dict(l=50, r=40, t=50, b=50),
        height=450,
    )
    return fig


def plot_covariance_heatmap(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    tickers: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate interactive Annualized Covariance Matrix Heatmap with numeric text annotations.

    Parameters
    ----------
    cov_matrix : pd.DataFrame | np.ndarray
        Covariance matrix.
    tickers : Optional[List[str]], optional
        Ticker names if cov_matrix is np.ndarray.

    Returns
    -------
    go.Figure
        Plotly Heatmap Figure.
    """
    labels, mat = _extract_matrix_and_labels(cov_matrix, tickers)

    text_annotations = [[f"{val:.4f}" for val in row] for row in mat]

    cov_colorscale = [
        [0.0, "#0e1117"],
        [0.3, "#102a43"],
        [0.6, "#1f4e79"],
        [1.0, "#00875a"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale=cov_colorscale,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#FFFFFF", family="Arial, sans-serif"),
            colorbar=dict(
                title=dict(text="Covarianza Anual", font=dict(color=THEME_DARK["font_color"])),
                tickfont=dict(color=THEME_DARK["font_color"]),
                thickness=14,
            ),
            hovertemplate=(
                "<b>Covarianza Anualizada</b><br>"
                "Activo X: %{x}<br>"
                "Activo Y: %{y}<br>"
                "Covarianza: %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Matriz de Covarianza Anualizada</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.02,
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        xaxis=dict(color=THEME_DARK["font_color"]),
        yaxis=dict(color=THEME_DARK["font_color"], autorange="reversed"),
        margin=dict(l=50, r=40, t=50, b=50),
        height=450,
    )
    return fig


# ===========================================================================
# 4. Historical Backtest & Underwater Drawdown Visualizers
# ===========================================================================

def plot_historical_backtest(
    returns_dict: Optional[Dict[str, Union[pd.Series, np.ndarray]]] = None,
    wealth_df: Optional[pd.DataFrame] = None,
    drawdown_series: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
    initial_capital: float = 10000.0,
) -> go.Figure:
    """
    Generate 2-Row Subplot: Cumulative Wealth Growth ($10,000 USD) and Underwater Drawdown.

    Parameters
    ----------
    returns_dict : Optional[dict], optional
        Dictionary of {Portfolio_Name: daily_returns_series}.
    wealth_df : Optional[pd.DataFrame], optional
        DataFrame of pre-calculated wealth trajectories.
    drawdown_series : Optional[pd.Series | dict], optional
        Pre-calculated drawdown series.
    benchmark_df : Optional[pd.DataFrame], optional
        Benchmark price/wealth DataFrame.
    initial_capital : float, default 10000.0
        Starting capital in USD.

    Returns
    -------
    go.Figure
        Plotly 2-Row Subplot Figure.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"<b>Evolución Patrimonial (${initial_capital:,.0f} USD)</b>",
            "<b>Drawdown Histórico (Caída desde Máximos)</b>",
        ],
    )

    series_map: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    if returns_dict is not None:
        for p_name, r_series in returns_dict.items():
            if isinstance(r_series, (pd.Series, pd.DataFrame)):
                s = r_series if isinstance(r_series, pd.Series) else r_series.iloc[:, 0]
            else:
                s = pd.Series(np.asarray(r_series, dtype=np.float64))
            cum_w = initial_capital * np.cumprod(1.0 + s.values)
            wealth_s = pd.Series(cum_w, index=s.index)
            r_max = np.maximum.accumulate(wealth_s.values)
            dd = (wealth_s.values - r_max) / np.maximum(r_max, 1e-12)
            dd_s = pd.Series(np.minimum(dd, 0.0), index=s.index)
            series_map[p_name] = (wealth_s, dd_s)

    elif wealth_df is not None:
        for col in wealth_df.columns:
            w_col = wealth_df[col]
            r_max = np.maximum.accumulate(w_col.values)
            dd = (w_col.values - r_max) / np.maximum(r_max, 1e-12)
            dd_s = pd.Series(np.minimum(dd, 0.0), index=w_col.index)
            series_map[str(col)] = (w_col, dd_s)

    if not series_map:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        dummy_w = pd.Series(np.linspace(initial_capital, initial_capital * 1.1, 10), index=dates)
        dummy_dd = pd.Series(np.zeros(10), index=dates)
        series_map["Cartera"] = (dummy_w, dummy_dd)

    for idx, (p_name, (w_series, dd_series_p)) in enumerate(series_map.items()):
        color = PALETTE_SERIES[idx % len(PALETTE_SERIES)]
        # Upper Subplot: Wealth
        fig.add_trace(
            go.Scatter(
                x=w_series.index,
                y=w_series.values,
                mode="lines",
                name=p_name,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{p_name}</b><br>"
                    "Fecha: %{x|%Y-%m-%d}<br>"
                    "Capital: $%{y:,.2f}<br>"
                    "Rendimiento: %{customdata:.2%}"
                    "<extra></extra>"
                ),
                customdata=(w_series.values - initial_capital) / initial_capital,
            ),
            row=1,
            col=1,
        )

        # Lower Subplot: Drawdown
        is_primary = idx == 0
        fig.add_trace(
            go.Scatter(
                x=dd_series_p.index,
                y=dd_series_p.values,
                mode="lines",
                name=f"DD {p_name}",
                line=dict(color=color, width=1.5),
                fill="tozeroy" if is_primary else "none",
                fillcolor="rgba(255, 65, 54, 0.2)" if is_primary else None,
                showlegend=False,
                hovertemplate=(
                    f"<b>Drawdown ({p_name})</b><br>"
                    "Fecha: %{x|%Y-%m-%d}<br>"
                    "Caída: %{y:.2%}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"]),
        legend=dict(
            font=dict(color=THEME_DARK["font_color"], size=11),
            bgcolor="rgba(22, 27, 38, 0.8)",
            bordercolor=THEME_DARK["grid_color"],
            x=0.01,
            y=0.99,
        ),
        margin=dict(l=60, r=40, t=50, b=40),
        height=580,
    )

    fig.update_yaxes(
        title_text="Capital (USD)",
        tickprefix="$",
        tickformat=",.0f",
        gridcolor=THEME_DARK["grid_color"],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Drawdown",
        tickformat=".1%",
        gridcolor=THEME_DARK["grid_color"],
        range=[-1.0, 0.05],
        row=2,
        col=1,
    )
    fig.update_xaxes(gridcolor=THEME_DARK["grid_color"], row=1, col=1)
    fig.update_xaxes(gridcolor=THEME_DARK["grid_color"], row=2, col=1)

    return fig


# ===========================================================================
# 5. Monte Carlo Stochastic Cones Visualizer
# ===========================================================================

def plot_monte_carlo_cones(
    trajectory_result: Optional[Any] = None,
    user_label: str = "Cartera",
    days: Optional[np.ndarray] = None,
    percentiles: Optional[Dict[str, np.ndarray]] = None,
    initial_wealth: float = 10000.0,
) -> go.Figure:
    """
    Generate Monte Carlo Stochastic Wealth Projection Fan Chart (5%, 25%, 50%, 75%, 95% Cones).

    Parameters
    ----------
    trajectory_result : Optional[Any], optional
        TrajectorySimulationResult object from simulation module.
    user_label : str, default 'Cartera'
        Name label for chart.
    days : Optional[np.ndarray], optional
        Trading days array (0 to T*252).
    percentiles : Optional[dict], optional
        Mapping of percentile arrays.
    initial_wealth : float, default 10000.0
        Starting capital in USD.

    Returns
    -------
    go.Figure
        Plotly Figure with shaded quantile bands.
    """
    fig = go.Figure()

    # Extract days and percentiles
    if trajectory_result is not None:
        p_days = getattr(trajectory_result, "days", np.arange(253))
        p5 = getattr(trajectory_result, "percentile_5", None)
        p25 = getattr(trajectory_result, "percentile_25", None)
        p50 = getattr(trajectory_result, "percentile_50", None)
        p75 = getattr(trajectory_result, "percentile_75", None)
        p95 = getattr(trajectory_result, "percentile_95", None)
        mean_traj = getattr(trajectory_result, "mean_trajectory", None)
        init_w = getattr(trajectory_result, "initial_wealth", initial_wealth)
    elif percentiles is not None:
        p_days = days if days is not None else np.arange(len(next(iter(percentiles.values()))))
        p5 = percentiles.get("p5")
        p25 = percentiles.get("p25")
        p50 = percentiles.get("p50")
        p75 = percentiles.get("p75")
        p95 = percentiles.get("p95")
        mean_traj = percentiles.get("mean")
        init_w = initial_wealth
    else:
        # Generate synthetic reference cone
        p_days = np.arange(252 * 3 + 1)
        t_years = p_days / 252.0
        init_w = initial_wealth
        p50 = init_w * np.exp(0.08 * t_years)
        p95 = init_w * np.exp((0.08 + 1.645 * 0.15) * t_years)
        p75 = init_w * np.exp((0.08 + 0.674 * 0.15) * t_years)
        p25 = init_w * np.exp((0.08 - 0.674 * 0.15) * t_years)
        p5 = init_w * np.exp((0.08 - 1.645 * 0.15) * t_years)
        mean_traj = init_w * np.exp(0.09 * t_years)

    x_years = p_days / 252.0

    # 1. 95th Percentile Upper Bound (Hidden line)
    if p95 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=p95,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 2. 5th Percentile Lower Bound with Fill to 95th
    if p5 is not None and p95 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=p5,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 240, 255, 0.12)",
                line=dict(width=0),
                name="Intervalo 5% - 95%",
                hovertemplate="<b>Percentil 5%:</b> $%{y:,.0f}<extra></extra>",
            )
        )

    # 3. 75th Percentile Upper Bound (Hidden line)
    if p75 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=p75,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 4. 25th Percentile Lower Bound with Fill to 75th
    if p25 is not None and p75 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=p25,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 240, 255, 0.28)",
                line=dict(width=0),
                name="Intervalo Intercuartil 25% - 75%",
                hovertemplate="<b>Percentil 25%:</b> $%{y:,.0f}<extra></extra>",
            )
        )

    # 5. Median Path (P50)
    if p50 is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=p50,
                mode="lines",
                name="Mediana Proyectada (P50)",
                line=dict(color=THEME_DARK["accent_cyan"], width=3.0),
                hovertemplate=(
                    "<b>Mediana (P50)</b><br>"
                    "Año: %{x:.1f}<br>"
                    "Capital: $%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    # 6. Expected Mean Trajectory
    if mean_traj is not None:
        fig.add_trace(
            go.Scatter(
                x=x_years,
                y=mean_traj,
                mode="lines",
                name="Media Teórica (E[W])",
                line=dict(color=THEME_DARK["accent_orange"], width=2.0, dash="dash"),
                hovertemplate=(
                    "<b>Media Teórica</b><br>"
                    "Año: %{x:.1f}<br>"
                    "Capital: $%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    # 7. Initial Capital Line
    fig.add_hline(
        y=init_w,
        line_dash="dot",
        line_color=THEME_DARK["grid_color"],
        annotation_text=f"Capital Inicial: ${init_w:,.0f}",
        annotation_position="bottom right",
        annotation_font_color=THEME_DARK["font_color"],
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Proyección Estocástica Monte Carlo — {user_label}</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.02,
        ),
        xaxis=dict(
            title="Horizonte Temporal (Años)",
            tickformat=".1f",
            gridcolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
        ),
        yaxis=dict(
            title="Valor Patrimonial Estimado (USD)",
            tickprefix="$",
            tickformat=",.0f",
            gridcolor=THEME_DARK["grid_color"],
            color=THEME_DARK["font_color"],
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        legend=dict(
            font=dict(color=THEME_DARK["font_color"], size=11),
            bgcolor="rgba(22, 27, 38, 0.8)",
            bordercolor=THEME_DARK["grid_color"],
            x=0.01,
            y=0.99,
        ),
        margin=dict(l=60, r=40, t=50, b=50),
        height=480,
    )
    return fig


def plot_monte_carlo_projection_cones(
    trajectory_result: Optional[Any] = None,
    user_label: str = "Cartera",
    days: Optional[np.ndarray] = None,
    percentiles: Optional[Dict[str, np.ndarray]] = None,
    initial_wealth: float = 10000.0,
) -> go.Figure:
    """Alias for plot_monte_carlo_cones."""
    return plot_monte_carlo_cones(
        trajectory_result=trajectory_result,
        user_label=user_label,
        days=days,
        percentiles=percentiles,
        initial_wealth=initial_wealth,
    )


# ===========================================================================
# 6. Internal Data Extraction & Parsing Helpers
# ===========================================================================

def _extract_mc_data(mc_result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract volatilities, returns, and sharpes from Monte Carlo result."""
    if hasattr(mc_result, "volatilities") and hasattr(mc_result, "returns") and hasattr(mc_result, "sharpe_ratios"):
        return (
            np.asarray(mc_result.volatilities, dtype=np.float64),
            np.asarray(mc_result.returns, dtype=np.float64),
            np.asarray(mc_result.sharpe_ratios, dtype=np.float64),
        )
    if isinstance(mc_result, (tuple, list)) and len(mc_result) >= 3:
        return (
            np.asarray(mc_result[0], dtype=np.float64),
            np.asarray(mc_result[1], dtype=np.float64),
            np.asarray(mc_result[2], dtype=np.float64),
        )
    return np.array([]), np.array([]), np.array([])


def _extract_frontier_data(frontier_result: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract volatilities and returns from Efficient Frontier result."""
    if hasattr(frontier_result, "volatilities") and hasattr(frontier_result, "returns"):
        return (
            np.asarray(frontier_result.volatilities, dtype=np.float64),
            np.asarray(frontier_result.returns, dtype=np.float64),
        )
    if isinstance(frontier_result, (tuple, list)) and len(frontier_result) >= 2:
        return (
            np.asarray(frontier_result[0], dtype=np.float64),
            np.asarray(frontier_result[1], dtype=np.float64),
        )
    return np.array([]), np.array([])


def _extract_optimal_point(frontier_result: Any, kind: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract optimal point (vol, ret, sharpe) from frontier result."""
    if frontier_result is None:
        return None, None, None

    attr_name = f"{kind}_portfolio"
    if hasattr(frontier_result, attr_name):
        port = getattr(frontier_result, attr_name)
        if port is not None:
            v = getattr(port, "volatility", None)
            r = getattr(port, "expected_return", None)
            s = getattr(port, "sharpe_ratio", None)
            return v, r, s

    return None, None, None


def _extract_user_point(user_point: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (vol, ret, sharpe) from user point representation."""
    if isinstance(user_point, (tuple, list)):
        v = float(user_point[0])
        r = float(user_point[1])
        s = float(user_point[2]) if len(user_point) > 2 else None
        return v, r, s
    if isinstance(user_point, dict):
        v = float(user_point.get("volatility", user_point.get("annualized_volatility", 0.0)))
        r = float(user_point.get("expected_return", user_point.get("annualized_return", 0.0)))
        s = user_point.get("sharpe_ratio")
        s_val = float(s) if s is not None else None
        return v, r, s_val
    if hasattr(user_point, "annualized_volatility") and hasattr(user_point, "annualized_return"):
        return (
            float(user_point.annualized_volatility),
            float(user_point.annualized_return),
            float(getattr(user_point, "sharpe_ratio", 0.0)),
        )
    return None, None, None


def _extract_weights_labels_vals(
    weights: Any, tickers: Optional[List[str]] = None
) -> Tuple[List[str], List[float]]:
    """Extract labels and numeric weights from diverse weight inputs."""
    if isinstance(weights, dict):
        labels = list(weights.keys())
        vals = [float(v) for v in weights.values()]
        return labels, vals
    if isinstance(weights, pd.Series):
        labels = [str(k) for k in weights.index]
        vals = [float(v) for v in weights.values]
        return labels, vals
    arr = np.asarray(weights, dtype=np.float64).ravel()
    labels = tickers if tickers and len(tickers) == len(arr) else [f"Asset_{i+1}" for i in range(len(arr))]
    return labels, [float(v) for v in arr]


def _extract_matrix_and_labels(
    mat_input: Any, tickers: Optional[List[str]] = None
) -> Tuple[List[str], np.ndarray]:
    """Extract labels and 2D array from matrix input."""
    if isinstance(mat_input, pd.DataFrame):
        labels = [str(c) for c in mat_input.columns]
        mat = mat_input.values.astype(np.float64)
        return labels, mat
    mat = np.asarray(mat_input, dtype=np.float64)
    n = mat.shape[0] if mat.ndim > 0 else 1
    labels = tickers if tickers and len(tickers) == n else [f"Activo_{i+1}" for i in range(n)]
    return labels, mat


def _add_individual_assets_traces(fig: go.Figure, individual_assets: Any, rf: float) -> None:
    """Helper to add individual assets scatter trace to Efficient Frontier plot."""
    t_names = []
    t_vols = []
    t_rets = []

    if isinstance(individual_assets, dict):
        for ticker, coords in individual_assets.items():
            t_names.append(ticker)
            t_vols.append(float(coords[0]))
            t_rets.append(float(coords[1]))
    elif isinstance(individual_assets, pd.DataFrame):
        for idx in individual_assets.index:
            t_names.append(str(idx))
            t_vols.append(float(individual_assets.loc[idx, "volatility"]))
            t_rets.append(float(individual_assets.loc[idx, "return"]))

    if t_names:
        sharpes = [(r - rf) / max(v, 1e-12) for v, r in zip(t_vols, t_rets)]
        fig.add_trace(
            go.Scatter(
                x=t_vols,
                y=t_rets,
                mode="markers+text",
                text=t_names,
                textposition="top center",
                textfont=dict(color="#E0E6ED", size=11),
                name="Activos Individuales",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color="#B0C4DE",
                    line=dict(color="white", width=1.0),
                ),
                hovertemplate=(
                    "<b>Activo: %{text}</b><br>"
                    "Retorno Esperado: %{y:.2%}<br>"
                    "Volatilidad Anual: %{x:.2%}<br>"
                    "Ratio Sharpe: %{customdata:.3f}"
                    "<extra></extra>"
                ),
                customdata=sharpes,
            )
        )


def plot_portfolio_comparison_scatter(
    portfolios_data: Dict[str, Dict[str, float]],
    rf: float = 0.04,
) -> go.Figure:
    """
    Generate an interactive Risk-Return Scatter Plot comparing multiple portfolios.

    Parameters
    ----------
    portfolios_data : dict
        {portfolio_name: {"return": annualized_ret, "volatility": annualized_vol, "sharpe": sharpe_ratio}}
    rf : float
        Risk-free rate.
    """
    fig = go.Figure()
    names = list(portfolios_data.keys())
    vols = [portfolios_data[n]["volatility"] for n in names]
    rets = [portfolios_data[n]["return"] for n in names]
    sharpes = [portfolios_data[n].get("sharpe", (r - rf) / max(v, 1e-12)) for n, v, r in zip(names, vols, rets)]

    for i, name in enumerate(names):
        color = PALETTE_SERIES[i % len(PALETTE_SERIES)]
        fig.add_trace(
            go.Scatter(
                x=[vols[i]],
                y=[rets[i]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                textfont=dict(color=color, size=12),
                name=name,
                marker=dict(
                    size=16,
                    color=color,
                    symbol="diamond",
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Retorno Anual: %{y:.2%}<br>"
                    "Volatilidad: %{x:.2%}<br>"
                    f"Ratio Sharpe: {sharpes[i]:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Comparativa Riesgo vs Retorno de Portafolios</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.02,
        ),
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        xaxis=dict(
            title=dict(text="Volatilidad Anualizada (Riesgo σ)", font=dict(color=THEME_DARK["font_color"])),
            color=THEME_DARK["font_color"],
            gridcolor=THEME_DARK["grid_color"],
            tickformat=".1%",
        ),
        yaxis=dict(
            title=dict(text="Retorno Anualizado Esperado (μ)", font=dict(color=THEME_DARK["font_color"])),
            color=THEME_DARK["font_color"],
            gridcolor=THEME_DARK["grid_color"],
            tickformat=".1%",
        ),
        legend=dict(
            font=dict(color=THEME_DARK["font_color"]),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=THEME_DARK["grid_color"],
        ),
        margin=dict(l=60, r=40, t=50, b=50),
        height=480,
    )
    return fig
