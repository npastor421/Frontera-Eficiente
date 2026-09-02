"""
Broker Holdings Report Parser and Ticker Normalizer.

Parses portfolio holdings reports exported from major Argentine and global brokers
(Balanz, PPI, InvertirOnline, Bull Market, Cohen, etc.) in Excel (.xlsx, .xls)
and CSV formats. Automatically classifies and normalizes tickers for CEDEARs,
Local Stocks (BYMA/ADR), Mutual Funds (FCI), Bonds, and Cash.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Comprehensive mapping of Argentine local tickers to US ADRs and normalized symbols
ARGENTINE_TO_US_ADR_MAP: Dict[str, str] = {
    "YPFD": "YPF",
    "YPF": "YPF",
    "GGAL": "GGAL",
    "BMA": "BMA",
    "BBAR": "BBAR",
    "PAMP": "PAM",
    "PAM": "PAM",
    "CRES": "CRESY",
    "CRESY": "CRESY",
    "EDN": "EDN",
    "TGSU2": "TGS",
    "TGS": "TGS",
    "CEPU": "CEPU",
    "LOMA": "LOMA",
    "SUPV": "SUPV",
    "IRSA": "IRS",
    "IRS": "IRS",
    "TECO2": "TEO",
    "TEO": "TEO",
    "TGLT": "TGLT",
    "BIOX": "BIOX",
    "DESP": "DESP",
    "GLOB": "GLOB",
    "MELI": "MELI",
    "VIST": "VIST",
}

# CEDEAR suffix normalizations (e.g. ITUB3 -> ITUB, BBD -> BBD)
CEDEAR_NORMALIZATION_MAP: Dict[str, str] = {
    "ITUB3": "ITUB",
    "ITUB4": "ITUB",
    "BBD3": "BBD",
    "BBD4": "BBD",
    "VALE3": "VALE",
    "PETR3": "PBR",
    "PETR4": "PBR",
    "GOOG": "GOOGL",
    "BRKB": "BRK-B",
    "BRK.B": "BRK-B",
    "BRKB.BA": "BRK-B",
    "BF.B": "BF-B",
}

# Known Money Market and Cash keywords
CASH_KEYWORDS = {
    "CASH", "USD", "ARS", "PESOS", "DOLARES", "LIQUIDEZ", "DISPONIBLE",
    "SALDO DISPONIBLE", "CAJA", "MONEY MARKET", "FCI LIQUIDEZ",
    "PESOS DISPONIBLES", "DOLARES DISPONIBLES", "USD DISPONIBLE"
}


@dataclass
class BrokerParsedItem:
    """Represents a single parsed instrument row from a broker holdings report."""
    raw_ticker: str
    normalized_ticker: str
    instrument_type: str
    description: str
    weight_pct: float
    current_value: float
    currency: str
    raw_row: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerHoldingsReport:
    """Represents the complete parsed broker holdings report."""
    items: List[BrokerParsedItem]
    tickers: List[str]
    weights: Dict[str, float]  # Normalized to sum to 1.0
    weights_pct: Dict[str, float]  # Sums to 100.0%
    total_valuation: float
    currency: str
    instruments_count: int
    by_type_breakdown: Dict[str, int]
    table_df: pd.DataFrame


def normalize_broker_ticker(
    raw_ticker: str,
    instrument_type: str = "",
    mode: str = "global_usd"
) -> str:
    """
    Normalize a raw broker ticker to a Yahoo Finance compatible symbol.

    Args:
        raw_ticker: Ticker symbol from the broker report (e.g. 'YPFD', 'ITUB3', 'GOOGL').
        instrument_type: Instrument classification (e.g. 'Cedears', 'Acciones', 'Fondos').
        mode: 'global_usd' (maps to Wall Street underlying/ADR) or 'byma_ars' (appends .BA).

    Returns:
        Normalized ticker string for Yahoo Finance (e.g. 'YPF', 'ITUB', 'GOOGL.BA', 'CASH').
    """
    clean = str(raw_ticker).strip().upper()
    itype = str(instrument_type).strip().lower()

    # 1. Cash / Liquidity detection
    if clean in CASH_KEYWORDS or any(kw in clean for kw in ["DISPONIBLE", "CAJA", "LIQUIDEZ"]):
        return "CASH"

    # 2. Money Market / Liquidity Mutual Funds
    if "fondo" in itype or "fci" in itype:
        if any(kw in clean for kw in ["MM", "LIQ", "PESOS", "AHORRO", "MONEY", "T0", "T+0"]):
            return "CASH"
        return "CASH"

    # 3. Cedear Normalization
    if "cedear" in itype:
        # Check explicit CEDEAR mappings
        base_symbol = CEDEAR_NORMALIZATION_MAP.get(clean, clean)
        # Strip any existing .BA suffix for base processing
        base_symbol = base_symbol.replace(".BA", "")

        if mode == "byma_ars":
            return f"{base_symbol}.BA" if not base_symbol.endswith(".BA") else base_symbol
        return base_symbol

    # 4. Local Stocks (Acciones)
    if "accion" in itype or "acciones" in itype or "renta variable" in itype:
        base_symbol = clean.replace(".BA", "")
        if mode == "global_usd":
            # Map to ADR if available, otherwise append .BA for local Argentine stocks
            if base_symbol in ARGENTINE_TO_US_ADR_MAP:
                return ARGENTINE_TO_US_ADR_MAP[base_symbol]
            return f"{base_symbol}.BA"
        else:
            return f"{base_symbol}.BA"

    # 5. Bonds / Government Debt (Bonos / Títulos Públicos)
    if "bono" in itype or "titulo" in itype or "título" in itype or "on" in itype or "obligacion" in itype:
        base_symbol = clean.replace(".BA", "")
        return f"{base_symbol}.BA"

    # 6. Default fallback
    base_symbol = CEDEAR_NORMALIZATION_MAP.get(clean, clean)
    if base_symbol in ARGENTINE_TO_US_ADR_MAP and mode == "global_usd":
        return ARGENTINE_TO_US_ADR_MAP[base_symbol]
    if mode == "byma_ars" and not base_symbol.endswith(".BA") and base_symbol != "CASH":
        return f"{base_symbol}.BA"
    return base_symbol


def parse_broker_holdings(
    file_input: Union[str, Path, bytes, io.BytesIO],
    filename: str = "",
    mode: str = "global_usd"
) -> BrokerHoldingsReport:
    """
    Parse an Excel (.xlsx, .xls) or CSV broker holdings report into a structured portfolio.

    Args:
        file_input: File path or raw bytes of the broker report.
        filename: Optional filename to identify extension format.
        mode: 'global_usd' or 'byma_ars'.

    Returns:
        BrokerHoldingsReport with normalized tickers, weights, and overview metadata.
    """
    # Load DataFrame
    df_raw: Optional[pd.DataFrame] = None
    
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        ext = p.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(p)
        else:
            df_raw = pd.read_csv(p)
    elif isinstance(file_input, bytes):
        b_io = io.BytesIO(file_input)
        is_excel = filename.lower().endswith((".xlsx", ".xls")) or file_input[:4] in (b"PK\x03\x04", b"\xd0\xcf\x11\xe0")
        if is_excel:
            df_raw = pd.read_excel(b_io)
        else:
            df_raw = pd.read_csv(b_io)
    elif isinstance(file_input, io.BytesIO):
        try:
            df_raw = pd.read_excel(file_input)
        except Exception:
            file_input.seek(0)
            df_raw = pd.read_csv(file_input)

    if df_raw is None or df_raw.empty:
        raise ValueError("El archivo del broker está vacío o no pudo ser leído.")

    # Identify columns flexibly using case-insensitive regex matching
    col_map: Dict[str, Optional[str]] = {
        "ticker": None,
        "type": None,
        "description": None,
        "weight_pct": None,
        "value": None,
        "currency": None,
    }

    for col in df_raw.columns:
        c_clean = str(col).strip()
        c_lower = c_clean.lower()

        if not col_map["ticker"] and re.search(r"^(ticker|s[ií]mbolo|especie|activo|codigo|c[oó]digo)$", c_lower):
            col_map["ticker"] = col
        elif not col_map["type"] and re.search(r"(tipo|instrumento|clase|categor[ií]a)", c_lower):
            col_map["type"] = col
        elif not col_map["description"] and re.search(r"(descripci[oó]n|nombre|denominaci[oó]n)", c_lower):
            col_map["description"] = col
        elif not col_map["weight_pct"] and re.search(r"(porcentaje.*tenencia|participaci[oó]n|%\s*tenencia|ponderaci[oó]n|%\s*cartera|peso)", c_lower):
            col_map["weight_pct"] = col
        elif not col_map["value"] and re.search(r"(valor\s*actual|monto|valuaci[oó]n|total|importe|saldo)", c_lower):
            col_map["value"] = col
        elif not col_map["currency"] and re.search(r"(moneda|divisa)", c_lower):
            col_map["currency"] = col

    # Fallbacks if regex didn't catch specific names
    if not col_map["ticker"]:
        # Use first string column
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                col_map["ticker"] = col
                break

    if not col_map["ticker"]:
        raise ValueError("No se encontró la columna de Ticker / Especie en el reporte del broker.")

    parsed_items: List[BrokerParsedItem] = []
    type_counts: Dict[str, int] = {}
    total_val = 0.0
    detected_currency = "ARS"

    # Helper function to parse numeric values safely
    def _clean_float(val: Any) -> float:
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).replace("$", "").replace("USD", "").replace("%", "").strip()
        # Handle European vs US number formats
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    raw_weights: List[float] = []
    raw_values: List[float] = []

    for _, row in df_raw.iterrows():
        raw_t = str(row.get(col_map["ticker"], "")).strip().upper()
        if not raw_t or raw_t in ["NAN", "NONE", "TOTAL", "TOTALES"]:
            continue

        itype = str(row.get(col_map["type"], "Cedears")).strip() if col_map["type"] else "Cedears"
        desc = str(row.get(col_map["description"], "")).strip() if col_map["description"] else raw_t
        curr = str(row.get(col_map["currency"], "Pesos")).strip() if col_map["currency"] else "Pesos"
        detected_currency = "USD" if "dolar" in curr.lower() or "usd" in curr.lower() else "ARS"

        norm_t = normalize_broker_ticker(raw_t, instrument_type=itype, mode=mode)

        w_val = _clean_float(row.get(col_map["weight_pct"], 0.0)) if col_map["weight_pct"] else 0.0
        v_val = _clean_float(row.get(col_map["value"], 0.0)) if col_map["value"] else 0.0

        raw_weights.append(w_val)
        raw_values.append(v_val)

        type_counts[itype] = type_counts.get(itype, 0) + 1
        total_val += v_val

        parsed_items.append(
            BrokerParsedItem(
                raw_ticker=raw_t,
                normalized_ticker=norm_t,
                instrument_type=itype,
                description=desc,
                weight_pct=w_val,
                current_value=v_val,
                currency=curr,
                raw_row=dict(row),
            )
        )

    if not parsed_items:
        raise ValueError("No se pudieron extraer activos válidos del reporte del broker.")

    # Calculate exact weights (consolidating duplicates if any, e.g. multiple CASH items)
    sum_w = sum(raw_weights)
    sum_v = sum(raw_values)

    consolidated_pcts: Dict[str, float] = {}
    for item in parsed_items:
        t = item.normalized_ticker
        if sum_w > 1.0:
            # Use percentage column
            pct = (item.weight_pct / sum_w) * 100.0
        elif sum_v > 0.0:
            # Use monetary valuation column
            pct = (item.current_value / sum_v) * 100.0
        else:
            # Uniform fallback
            pct = 100.0 / len(parsed_items)

        item.weight_pct = round(pct, 2)
        consolidated_pcts[t] = consolidated_pcts.get(t, 0.0) + pct

    # Round and balance to exactly 100.00%
    norm_tickers = list(consolidated_pcts.keys())
    rounded_pcts = {t: round(consolidated_pcts[t], 2) for t in norm_tickers}
    diff = round(100.0 - sum(rounded_pcts.values()), 2)
    if abs(diff) > 0.0 and len(norm_tickers) > 0:
        rounded_pcts[norm_tickers[0]] = round(rounded_pcts[norm_tickers[0]] + diff, 2)

    weights_1_0 = {t: round(rounded_pcts[t] / 100.0, 6) for t in norm_tickers}

    # Build clean presentation DataFrame
    table_rows = []
    for item in parsed_items:
        table_rows.append({
            "Ticker Original": item.raw_ticker,
            "Ticker Optimización": item.normalized_ticker,
            "Tipo": item.instrument_type,
            "Descripción": item.description,
            "Ponderación (%)": rounded_pcts.get(item.normalized_ticker, item.weight_pct),
            "Valor Actual": f"${item.current_value:,.2f}" if item.current_value > 0 else "N/A",
        })

    table_df = pd.DataFrame(table_rows)

    return BrokerHoldingsReport(
        items=parsed_items,
        tickers=norm_tickers,
        weights=weights_1_0,
        weights_pct=rounded_pcts,
        total_valuation=round(total_val, 2),
        currency=detected_currency,
        instruments_count=len(parsed_items),
        by_type_breakdown=type_counts,
        table_df=table_df,
    )
