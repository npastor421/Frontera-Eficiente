"""
Portfolio Diversification Interactive Visualizers Module.

Provides Plotly visualizers:
1. Hierarchical Multi-Level Treemap: Asset Class -> Country -> Sector -> Ticker.
2. Classification Dimension Donut Charts: Sector, Country, Asset Class, Market Cap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

# Dark theme palette consistent with main visualizer
THEME_DARK = {
    "paper_bgcolor": "#0e1117",
    "plot_bgcolor": "#161b26",
    "font_color": "#FAFAFA",
    "grid_color": "#2d3748",
}

PALETTE_SERIES = [
    "#00F0FF",  # Cyan
    "#00FF66",  # Green
    "#FF3366",  # Magenta/Pink
    "#FFCC00",  # Gold
    "#9D4EDD",  # Purple
    "#3A86FF",  # Electric Blue
    "#FF8800",  # Warm Orange
    "#06D6A0",  # Mint Green
    "#118AB2",  # Deep Teal
    "#EF476F",  # Coral
    "#8338EC",  # Violet
    "#FB5607",  # Red-Orange
    "#48CAE4",  # Sky Blue
]


def plot_diversification_treemap(
    tickers: Sequence[str],
    weights: Sequence[float],
    metadata: Dict[str, Dict[str, Any]],
    title: str = "Mapa Jerárquico de Diversificación (Clase → País → Sector → Activo)",
) -> go.Figure:
    """
    Generate an interactive hierarchical Treemap for portfolio allocation.

    Hierarchy Levels:
    Portafolio -> Clase de Activo -> País -> Sector -> Ticker

    Parameters
    ----------
    tickers : sequence of str
        List of asset ticker symbols.
    weights : sequence of float
        Asset weights (normalized or percentages).
    metadata : dict
        Mapping ticker -> metadata dict (from fetch_asset_classification).
    title : str
        Figure title.

    Returns
    -------
    go.Figure
        Interactive Plotly Treemap figure.
    """
    w_arr = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(np.abs(w_arr))
    if w_sum <= 1e-12 or len(tickers) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=THEME_DARK["paper_bgcolor"],
            font=dict(color=THEME_DARK["font_color"]),
            title=dict(text="Sin datos suficientes para el Treemap", font=dict(size=14)),
        )
        return fig

    # Normalize weights to percentages (0 to 100)
    w_pct = (w_arr / w_sum) * 100.0

    # Build unique nodes bottom-up
    # Tree structures:
    # id -> (label, parent_id, value, hovertext)
    node_values: Dict[str, float] = {}
    node_labels: Dict[str, str] = {}
    node_parents: Dict[str, str] = {}
    node_hovers: Dict[str, str] = {}

    root_id = "PORTAFOLIO"
    node_labels[root_id] = "Portafolio Total"
    node_parents[root_id] = ""
    node_values[root_id] = 0.0
    node_hovers[root_id] = "Total: 100.00%"

    for i, t in enumerate(tickers):
        clean_t = str(t).strip().upper()
        weight = float(w_pct[i]) if i < len(w_pct) else 0.0
        if weight <= 0.001:
            continue

        meta = metadata.get(clean_t, {})
        asset_class = str(meta.get("asset_class") or "Renta Variable")
        country = str(meta.get("country") or "Global")
        sector = str(meta.get("sector") or "Otros / Fondos")
        short_name = str(meta.get("short_name") or clean_t)

        class_id = f"CLASS::{asset_class}"
        country_id = f"{class_id}/COUNTRY::{country}"
        sector_id = f"{country_id}/SEC::{sector}"
        leaf_id = f"{sector_id}/ASSET::{clean_t}"

        # Initialize parents if not present
        if class_id not in node_labels:
            node_labels[class_id] = asset_class
            node_parents[class_id] = root_id
            node_values[class_id] = 0.0

        if country_id not in node_labels:
            node_labels[country_id] = country
            node_parents[country_id] = class_id
            node_values[country_id] = 0.0

        if sector_id not in node_labels:
            node_labels[sector_id] = sector
            node_parents[sector_id] = country_id
            node_values[sector_id] = 0.0

        # Leaf node
        node_labels[leaf_id] = clean_t
        node_parents[leaf_id] = sector_id
        node_values[leaf_id] = weight
        node_hovers[leaf_id] = (
            f"<b>{clean_t}</b> ({short_name})<br>"
            f"Ponderación: <b>{weight:.2f}%</b><br>"
            f"Sector: {sector}<br>"
            f"País: {country}<br>"
            f"Clase: {asset_class}"
        )

        # Accumulate up the tree
        node_values[sector_id] += weight
        node_values[country_id] += weight
        node_values[class_id] += weight
        node_values[root_id] += weight

    # Set parent hovertexts
    for cid in node_labels:
        if cid == root_id:
            continue
        if cid not in node_hovers:
            val = node_values[cid]
            lbl = node_labels[cid]
            node_hovers[cid] = f"<b>{lbl}</b><br>Ponderación Total: <b>{val:.2f}%</b>"

    ids = list(node_labels.keys())
    labels = [node_labels[k] for k in ids]
    parents = [node_parents[k] for k in ids]
    values = [round(node_values[k], 4) for k in ids]
    hovers = [node_hovers[k] for k in ids]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hoverinfo="text",
            hovertext=hovers,
            texttemplate="<b>%{label}</b><br>%{value:.1f}%",
            marker=dict(
                colorscale="Tealgrn",
                line=dict(color="#161b26", width=2),
            ),
            pathbar=dict(visible=True),
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter, system-ui, sans-serif"),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=THEME_DARK["font_color"], size=16),
            x=0.01,
            y=0.98,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=520,
    )

    return fig


def plot_dimension_donut(
    dim_weights: Dict[str, float],
    title: str = "Distribución",
    hole: float = 0.55,
) -> go.Figure:
    """
    Generate an interactive Plotly Donut chart for any classification dimension.

    Parameters
    ----------
    dim_weights : dict
        Mapping category name -> weight percentage (0.0 to 100.0).
    title : str
        Chart header title.
    hole : float, default 0.55
        Center hole diameter ratio for donut appearance.

    Returns
    -------
    go.Figure
        Interactive Plotly Figure.
    """
    fig = go.Figure()

    if not dim_weights or sum(dim_weights.values()) <= 1e-12:
        fig.update_layout(
            paper_bgcolor=THEME_DARK["paper_bgcolor"],
            font=dict(color=THEME_DARK["font_color"]),
            title=dict(text=title, font=dict(size=14)),
        )
        return fig

    labels = list(dim_weights.keys())
    values = list(dim_weights.values())

    colors = [PALETTE_SERIES[i % len(PALETTE_SERIES)] for i in range(len(labels))]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Participación: <b>%{value:.2f}%</b> (%{percent})<extra></extra>",
            marker=dict(
                colors=colors,
                line=dict(color="#0e1117", width=2),
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME_DARK["paper_bgcolor"],
        plot_bgcolor=THEME_DARK["plot_bgcolor"],
        font=dict(color=THEME_DARK["font_color"], family="Inter, system-ui, sans-serif", size=12),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=THEME_DARK["font_color"], size=14),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=30),
        height=340,
    )

    return fig
