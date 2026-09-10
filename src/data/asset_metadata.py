"""
Asset Metadata and Classification Module.

Retrieves and categorizes financial asset information (sector, country, asset class,
and market capitalization) using yfinance in real time with caching, plus built-in
rules for synthetic cash and liquidity assets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

CASH_SYMBOLS = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}

DEFAULT_METADATA: Dict[str, Any] = {
    "sector": "Otros / No Clasificado",
    "industry": "No Clasificado",
    "country": "Global",
    "asset_class": "Renta Variable (Acciones)",
    "market_cap": 0.0,
    "market_cap_category": "N/A",
    "short_name": "Desconocido",
}


def categorize_market_cap(market_cap: Optional[float]) -> str:
    """Categorize market capitalization into financial scale tiers."""
    if market_cap is None or market_cap <= 0:
        return "N/A"
    if market_cap >= 200e9:
        return "Mega Cap (>$200B)"
    if market_cap >= 10e9:
        return "Large Cap ($10B-$200B)"
    if market_cap >= 2e9:
        return "Mid Cap ($2B-$10B)"
    return "Small/Micro Cap (<$2B)"


def categorize_asset_class(quote_type: str, ticker: str, category: Optional[str] = None) -> str:
    """Classify the broad asset class from yfinance quoteType and symbol characteristics."""
    t_upper = str(ticker).strip().upper()
    if t_upper in CASH_SYMBOLS:
        return "Liquidez (CASH)"

    q_type = str(quote_type).strip().upper() if quote_type else ""

    if q_type == "CRYPTOCURRENCY" or "BTC" in t_upper or "ETH" in t_upper or "-USD" in t_upper:
        return "Criptoactivos"

    if q_type == "ETF":
        # Check commodity or bond cues in ticker or category
        if t_upper in {"GLD", "SLV", "DBC", "USO", "IAU", "GSG"}:
            return "Materias Primas (Commodities)"
        if t_upper in {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "TIP", "EMB"}:
            return "Renta Fija (Bonos)"
        return "ETF / Fondos"

    if q_type == "MUTUALFUND":
        return "Fondos Comunes"

    if q_type in {"CURRENCY", "MONEYMARKET"}:
        return "Liquidez (CASH)"

    if q_type in {"FUTURE", "COMMODITY"}:
        return "Materias Primas (Commodities)"

    return "Renta Variable (Acciones)"


def _translate_sector(sector: str) -> str:
    """Translate standard GICS English sector names to Spanish for consistent UI presentation."""
    translations = {
        "Technology": "Tecnología",
        "Financial Services": "Servicios Financieros",
        "Healthcare": "Salud",
        "Consumer Cyclical": "Consumo Discrecional / Cíclico",
        "Consumer Defensive": "Consumo Defensivo / Básico",
        "Energy": "Energía (Petróleo y Gas)",
        "Industrials": "Industrial",
        "Communication Services": "Servicios de Comunicación",
        "Utilities": "Servicios Públicos (Utilities)",
        "Real Estate": "Bienes Raíces (Real Estate)",
        "Basic Materials": "Materiales Básicos",
    }
    return translations.get(sector, sector if sector else "Otros / Diversificado")


def _translate_country(country: str) -> str:
    """Translate standard country names to Spanish."""
    translations = {
        "United States": "Estados Unidos",
        "Argentina": "Argentina",
        "Brazil": "Brasil",
        "United Kingdom": "Reino Unido",
        "Germany": "Alemania",
        "China": "China",
        "Japan": "Japón",
        "Canada": "Canadá",
        "France": "Francia",
        "Netherlands": "Países Bajos",
        "Switzerland": "Suiza",
        "Taiwan": "Taiwán",
        "South Korea": "Corea del Sur",
        "India": "India",
        "Australia": "Australia",
        "Israel": "Israel",
        "Mexico": "México",
        "Chile": "Chile",
    }
    return translations.get(country, country if country else "Global")


def get_single_asset_metadata(ticker: str) -> Dict[str, Any]:
    """
    Fetch metadata for a single asset via yfinance with fallbacks and normalizations.
    """
    clean_ticker = str(ticker).strip().upper()

    # Case 1: Cash / Liquidity Asset
    if clean_ticker in CASH_SYMBOLS:
        return {
            "ticker": clean_ticker,
            "short_name": "Liquidez en USD",
            "sector": "Liquidez / Monetario",
            "industry": "Efectivo / Caja",
            "country": "Global (USD)",
            "asset_class": "Liquidez (CASH)",
            "market_cap": 0.0,
            "market_cap_category": "N/A",
        }

    # Case 2: yfinance lookup
    try:
        yf_ticker = yf.Ticker(clean_ticker)
        # Fast info preferred when available, full info as fallback
        info = {}
        try:
            info = yf_ticker.info or {}
        except Exception:
            info = {}

        raw_sector = info.get("sector")
        raw_country = info.get("country")
        quote_type = info.get("quoteType", "EQUITY")
        market_cap = float(info.get("marketCap", 0.0) or 0.0)
        short_name = info.get("shortName") or info.get("longName") or clean_ticker

        # Handle Argentine CEDEARs or BYMA stocks where country might be missing or Argentina
        if clean_ticker.endswith(".BA"):
            if not raw_country:
                raw_country = "Argentina"

        sector = _translate_sector(raw_sector) if raw_sector else "Otros / Fondos"
        country = _translate_country(raw_country) if raw_country else "Estados Unidos"
        asset_class = categorize_asset_class(quote_type, clean_ticker, info.get("category"))
        market_cap_cat = categorize_market_cap(market_cap)

        return {
            "ticker": clean_ticker,
            "short_name": str(short_name),
            "sector": sector,
            "industry": str(info.get("industry") or sector),
            "country": country,
            "asset_class": asset_class,
            "market_cap": market_cap,
            "market_cap_category": market_cap_cat,
        }
    except Exception as ex:
        logger.warning(f"Error fetching metadata for {clean_ticker}: {ex}")
        return {
            "ticker": clean_ticker,
            "short_name": clean_ticker,
            "sector": "Otros / No Clasificado",
            "industry": "No Clasificado",
            "country": "Estados Unidos" if not clean_ticker.endswith(".BA") else "Argentina",
            "asset_class": "Renta Variable (Acciones)",
            "market_cap": 0.0,
            "market_cap_category": "N/A",
        }


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def fetch_asset_classification(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch comprehensive classification metadata for a list of portfolio tickers.
    Cached for 12 hours to ensure fast and responsive UI reruns.

    Parameters
    ----------
    tickers : list of str
        List of asset ticker symbols.

    Returns
    -------
    dict
        Dictionary mapping ticker -> classification dictionary.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for t in tickers:
        clean_t = str(t).strip().upper()
        if clean_t:
            result[clean_t] = get_single_asset_metadata(clean_t)
    return result
