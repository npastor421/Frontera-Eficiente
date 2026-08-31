"""
Data Caching Module for Frontera Eficiente.

Provides Streamlit caching wrappers (@st.cache_data) with configurable TTL,
defensive copy guarantees, fallback execution when running outside of Streamlit,
and explicit cache clearing mechanics.
"""

from __future__ import annotations

import datetime
from typing import Sequence, Tuple, Union

import pandas as pd

from src.data.cleaner import clean_and_align_prices
from src.data.loader import fetch_asset_data, validate_tickers

# Conditional Streamlit caching integration
try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False


def _streamlit_cache_wrapper(func):
    """Applies @st.cache_data if Streamlit is present, otherwise returns the function unchanged."""
    if HAS_STREAMLIT and hasattr(st, "cache_data"):
        return st.cache_data(ttl=3600, show_spinner=False, max_entries=128)(func)
    return func


@_streamlit_cache_wrapper
def _fetch_cached_raw_prices_impl(
    tickers_tuple: Tuple[str, ...],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Internal cached raw price fetcher taking hashable tuple of tickers."""
    return fetch_asset_data(
        tickers=list(tickers_tuple),
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


@_streamlit_cache_wrapper
def _fetch_and_clean_cached_impl(
    tickers_tuple: Tuple[str, ...],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    freq: str = "B",
    return_method: str = "simple",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Internal cached pipeline fetching and cleaning prices."""
    raw_prices = fetch_asset_data(
        tickers=list(tickers_tuple),
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )
    clean_prices, daily_returns = clean_and_align_prices(
        df_raw=raw_prices,
        freq=freq,
        drop_incomplete=True,
        return_method=return_method,
    )
    return clean_prices, daily_returns


def get_cached_raw_prices(
    tickers: Union[str, Sequence[str]],
    start_date: Union[str, datetime.date, datetime.datetime],
    end_date: Union[str, datetime.date, datetime.datetime],
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetches raw historical adjusted prices with caching and defensive copy.

    Parameters
    ----------
    tickers : str or Sequence[str]
        List or comma-separated string of ticker symbols.
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    interval : str, default '1d'
        Data frequency.

    Returns
    -------
    pd.DataFrame
        Defensive copy of cached price DataFrame.
    """
    valid_tickers = validate_tickers(tickers)
    start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, (datetime.date, datetime.datetime)) else str(start_date).strip()
    end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, (datetime.date, datetime.datetime)) else str(end_date).strip()

    tickers_key = tuple(sorted(valid_tickers))
    df = _fetch_cached_raw_prices_impl(
        tickers_tuple=tickers_key,
        start_date=start_str,
        end_date=end_str,
        interval=interval,
    )
    # Return defensive copy so caller mutations do not alter cache
    return df.copy()


def get_cached_asset_data(
    tickers: Union[str, Sequence[str]],
    start_date: Union[str, datetime.date, datetime.datetime],
    end_date: Union[str, datetime.date, datetime.datetime],
    interval: str = "1d",
    freq: str = "B",
    return_method: str = "simple",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches and cleans historical asset prices and returns with caching.

    Parameters
    ----------
    tickers : str or Sequence[str]
        List or string of ticker symbols.
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    interval : str, default '1d'
        Data frequency ('1d', '1wk', '1mo').
    freq : str, default 'B'
        Calendar alignment frequency ('B' for business days, 'D' for daily).
    return_method : str, default 'simple'
        Return calculation method ('simple' or 'log').

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of `(clean_prices_df, daily_returns_df)` (defensive copies).
    """
    valid_tickers = validate_tickers(tickers)
    start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, (datetime.date, datetime.datetime)) else str(start_date).strip()
    end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, (datetime.date, datetime.datetime)) else str(end_date).strip()

    tickers_key = tuple(sorted(valid_tickers))
    clean_prices, daily_returns = _fetch_and_clean_cached_impl(
        tickers_tuple=tickers_key,
        start_date=start_str,
        end_date=end_str,
        interval=interval,
        freq=freq,
        return_method=return_method,
    )
    return clean_prices.copy(), daily_returns.copy()


def clear_data_cache() -> None:
    """
    Clears the Streamlit cache for data ingestion functions.
    Safe to call in any context (no-op if Streamlit is not active or cached).
    """
    if HAS_STREAMLIT and hasattr(st, "cache_data"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
    if hasattr(_fetch_cached_raw_prices_impl, "clear"):
        try:
            _fetch_cached_raw_prices_impl.clear()
        except Exception:
            pass
    if hasattr(_fetch_and_clean_cached_impl, "clear"):
        try:
            _fetch_and_clean_cached_impl.clear()
        except Exception:
            pass
