"""
Data Ingestion, Cleaning, Harmonization, and Caching module for Frontera Eficiente.
"""

from src.data.cache import (
    clear_data_cache,
    get_cached_asset_data,
    get_cached_raw_prices,
)
from src.data.cleaner import (
    align_to_calendar,
    calculate_daily_returns,
    clean_and_align_prices,
    normalize_datetime_index,
    trim_common_inception,
    validate_price_data,
)
from src.data.loader import (
    fetch_asset_data,
    load_manual_file,
    parse_manual_data,
    validate_tickers,
)

from src.data.asset_metadata import (
    categorize_asset_class,
    categorize_market_cap,
    fetch_asset_classification,
)

__all__ = [
    "fetch_asset_data",
    "parse_manual_data",
    "load_manual_file",
    "validate_tickers",
    "clean_and_align_prices",
    "normalize_datetime_index",
    "align_to_calendar",
    "trim_common_inception",
    "calculate_daily_returns",
    "validate_price_data",
    "get_cached_asset_data",
    "get_cached_raw_prices",
    "clear_data_cache",
    "fetch_asset_classification",
    "categorize_market_cap",
    "categorize_asset_class",
]
