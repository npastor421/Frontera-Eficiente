"""
Portfolio Presets Module.

Defines canonical reference investment portfolios for 1-click loading:
1. Clásico 60/40 (SPY 60%, TLT 40%)
2. All-Weather Ray Dalio (SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, DBC 7.5%)
3. Big Tech (AAPL 20%, MSFT 20%, GOOGL 20%, AMZN 20%, NVDA 20%)
4. CEDEARs Argentina (AAPL.BA 20%, MSFT.BA 20%, GOOGL.BA 15%, MELI.BA 20%, SPY.BA 15%, KO.BA 10%)
5. Cripto + TradFi (SPY 50%, QQQ 30%, BTC-USD 15%, ETH-USD 5%)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


PRESETS: Dict[str, Dict[str, Any]] = {
    "classic_60_40": {
        "name": "Clásico 60/40",
        "description": "Cartera institucional clásica equilibrando 60% Renta Variable (SPY) y 40% Bonos del Tesoro a Largo Plazo (TLT).",
        "tickers": ["SPY", "TLT"],
        "weights": {
            "SPY": 0.60,
            "TLT": 0.40,
        },
        "asset_classes": {
            "SPY": "US Equities (S&P 500)",
            "TLT": "US 20+ Year Treasury Bonds",
        },
    },
    "all_weather": {
        "name": "All-Weather (Ray Dalio)",
        "description": "Estrategia Paridad de Riesgo de Ray Dalio diseñada para rendir consistentemente en los 4 regímenes macroeconómicos.",
        "tickers": ["SPY", "TLT", "IEF", "GLD", "DBC"],
        "weights": {
            "SPY": 0.30,
            "TLT": 0.40,
            "IEF": 0.15,
            "GLD": 0.075,
            "DBC": 0.075,
        },
        "asset_classes": {
            "SPY": "US Equities (30%)",
            "TLT": "Long-Term Treasuries (40%)",
            "IEF": "Intermediate Treasuries (15%)",
            "GLD": "Gold / Inflation Hedge (7.5%)",
            "DBC": "Commodities Index (7.5%)",
        },
    },
    "big_tech": {
        "name": "Big Tech",
        "description": "Top 5 gigantes tecnológicos de alta capitalización, crecimiento continuo y liderazgo global.",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "weights": {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "GOOGL": 0.20,
            "AMZN": 0.20,
            "NVDA": 0.20,
        },
        "asset_classes": {
            "AAPL": "Apple Inc. (Consumer Tech)",
            "MSFT": "Microsoft Corp. (Cloud & Software)",
            "GOOGL": "Alphabet Inc. (Digital Advertising & AI)",
            "AMZN": "Amazon.com Inc. (E-Commerce & AWS)",
            "NVDA": "NVIDIA Corp. (Semiconductors & AI Hardware)",
        },
    },
    "cedears_argentina": {
        "name": "CEDEARs Argentina",
        "description": "Portafolio diversificado en Certificados de Depósito Argentinos (CEDEARs) operables en BYMA con cobertura implícita CCL.",
        "tickers": ["AAPL.BA", "MSFT.BA", "GOOGL.BA", "MELI.BA", "SPY.BA", "KO.BA"],
        "weights": {
            "AAPL.BA": 0.20,
            "MSFT.BA": 0.20,
            "GOOGL.BA": 0.15,
            "MELI.BA": 0.20,
            "SPY.BA": 0.15,
            "KO.BA": 0.10,
        },
        "asset_classes": {
            "AAPL.BA": "Apple Inc. (CEDEAR)",
            "MSFT.BA": "Microsoft Corp. (CEDEAR)",
            "GOOGL.BA": "Alphabet Inc. (CEDEAR)",
            "MELI.BA": "MercadoLibre Inc. (CEDEAR)",
            "SPY.BA": "SPDR S&P 500 ETF (CEDEAR)",
            "KO.BA": "The Coca-Cola Co. (CEDEAR)",
        },
    },
    "crypto_tradfi": {
        "name": "Cripto + TradFi",
        "description": "Portafolio híbrido moderno combinando índices tradicionales diversificados (SPY, QQQ) con asignación convexa a criptoactivos.",
        "tickers": ["SPY", "QQQ", "BTC-USD", "ETH-USD"],
        "weights": {
            "SPY": 0.50,
            "QQQ": 0.30,
            "BTC-USD": 0.15,
            "ETH-USD": 0.05,
        },
        "asset_classes": {
            "SPY": "S&P 500 ETF (50%)",
            "QQQ": "Nasdaq 100 ETF (30%)",
            "BTC-USD": "Bitcoin (15%)",
            "ETH-USD": "Ethereum (5%)",
        },
    },
}

# Lookup aliases mapping common user inputs to canonical preset keys
_PRESET_ALIASES: Dict[str, str] = {
    "classic_60_40": "classic_60_40",
    "clasico_60_40": "classic_60_40",
    "clásico 60/40": "classic_60_40",
    "clasico 60/40": "classic_60_40",
    "60/40": "classic_60_40",
    "60_40": "classic_60_40",
    "all_weather": "all_weather",
    "all-weather": "all_weather",
    "all-weather (ray dalio)": "all_weather",
    "all weather": "all_weather",
    "ray dalio": "all_weather",
    "big_tech": "big_tech",
    "big tech": "big_tech",
    "bigtech": "big_tech",
    "cedears_argentina": "cedears_argentina",
    "cedears argentina": "cedears_argentina",
    "cedears": "cedears_argentina",
    "cedear": "cedears_argentina",
    "crypto_tradfi": "crypto_tradfi",
    "crypto + tradfi": "crypto_tradfi",
    "cripto + tradfi": "crypto_tradfi",
    "cripto_tradfi": "crypto_tradfi",
    "crypto": "crypto_tradfi",
}


def _resolve_preset_key(name: str) -> str:
    """Normalize preset name string to canonical dictionary key."""
    norm = name.strip().lower()
    if norm in _PRESET_ALIASES:
        return _PRESET_ALIASES[norm]
    # Check if exact key in PRESETS
    if name in PRESETS:
        return name
    raise KeyError(
        f"Unknown preset portfolio '{name}'. Available presets: {list(PRESETS.keys())}"
    )


def list_presets() -> List[str]:
    """Return list of canonical preset portfolio identifiers."""
    return list(PRESETS.keys())


def get_preset_list() -> List[str]:
    """Alias for list_presets()."""
    return list_presets()


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get preset portfolio data dictionary by name or alias.

    Parameters
    ----------
    name : str
        Preset name (e.g. 'classic_60_40', 'Clásico 60/40', 'all_weather', etc.).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'name', 'description', 'tickers', 'weights', and 'asset_classes'.
    """
    canonical_key = _resolve_preset_key(name)
    return PRESETS[canonical_key]


def get_preset_tickers(name: str) -> List[str]:
    """Get list of asset tickers for a preset portfolio."""
    preset = get_preset(name)
    return list(preset["tickers"])


def get_preset_weights(name: str) -> Dict[str, float]:
    """Get mapping of ticker to target weight for a preset portfolio."""
    preset = get_preset(name)
    return dict(preset["weights"])


def get_preset_description(name: str) -> str:
    """Get description text for a preset portfolio."""
    preset = get_preset(name)
    return str(preset["description"])
