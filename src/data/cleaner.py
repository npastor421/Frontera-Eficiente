"""
Data Cleaner and Harmonization Module for Frontera Eficiente.

Provides master trading calendar alignment (Business Days freq='B', Daily freq='D'),
forward-fill holiday handling, common inception trimming, zero-variance diagnostics,
and daily simple/log return calculation.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pandas as pd


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the DataFrame index to a timezone-naive, ascending DatetimeIndex
    normalized to 00:00:00 without duplicate timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with date-like index or DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized, deduplicated, sorted DatetimeIndex named 'Date'.

    Raises
    ------
    ValueError
        If the index cannot be converted to a valid DatetimeIndex.
    """
    if df.empty:
        return df.copy()

    df_out = df.copy()
    try:
        df_out.index = pd.to_datetime(df_out.index)
    except Exception as exc:
        raise ValueError(f"Could not convert index to DatetimeIndex: {exc}") from exc

    # Remove timezone if present
    if getattr(df_out.index, "tz", None) is not None:
        df_out.index = df_out.index.tz_convert(None)
    df_out.index = df_out.index.tz_localize(None).normalize()
    df_out.index.name = "Date"

    # Sort chronologically
    df_out = df_out.sort_index()

    # Deduplicate index keeping the first observation
    df_out = df_out[~df_out.index.duplicated(keep="first")]

    return df_out


def align_to_calendar(
    df: pd.DataFrame,
    freq: str = "B",
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Reindexes the price series to a continuous master calendar (e.g. Business Days 'B'
    or Calendar Days 'D') and applies forward-filling to handle market holidays and closures.

    Parameters
    ----------
    df : pd.DataFrame
        Input price DataFrame with DatetimeIndex.
    freq : str, default 'B'
        Pandas frequency string ('B' for business days, 'D' for calendar days, 'W-FRI' for weekly).
    method : str, default 'ffill'
        Interpolation/filling method ('ffill', 'bfill', or None).

    Returns
    -------
    pd.DataFrame
        Calendar-aligned DataFrame with contiguous frequency index.
    """
    if df.empty:
        return df.copy()

    df_norm = normalize_datetime_index(df)

    start_date = df_norm.index.min()
    end_date = df_norm.index.max()

    full_idx = pd.date_range(start=start_date, end=end_date, freq=freq, name="Date")
    aligned = df_norm.reindex(full_idx)

    if method == "ffill":
        aligned = aligned.ffill()
    elif method == "bfill":
        aligned = aligned.bfill()

    return aligned


def trim_common_inception(
    df: pd.DataFrame,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Trims the dataset to the common inception date where all assets have valid active price observations.
    Optionally drops any remaining rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame.
    drop_incomplete : bool, default True
        If True, drops any remaining interior/trailing NaN rows across assets.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame sharing a common inception date.

    Raises
    ------
    ValueError
        If no common overlapping date exists across all asset columns.
    """
    if df.empty:
        raise ValueError("Cannot trim an empty DataFrame.")

    # Find the earliest date where ALL columns have non-null data
    valid_rows_mask = df.notna().all(axis=1)

    if not valid_rows_mask.any():
        # Identify columns with excessive NaNs for detailed error diagnostics
        nan_summary = df.isna().sum()
        raise ValueError(
            f"No common overlapping date range found across all assets. "
            f"Missing count per asset:\n{nan_summary.to_dict()}"
        )

    first_valid_date = valid_rows_mask.idxmax()
    trimmed = df.loc[first_valid_date:].copy()

    if drop_incomplete:
        # Forward fill any isolated market holidays then drop any residual NaNs
        trimmed = trimmed.ffill().dropna(how="any")

    if trimmed.empty:
        raise ValueError("Dataset is empty after trimming to common inception date.")

    return trimmed


def validate_price_data(
    df: pd.DataFrame,
    min_obs: int = 5,
    variance_tol: float = 1e-12,
) -> None:
    """
    Validates structural integrity, positivity, and variance of a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame.
    min_obs : int, default 5
        Minimum required number of historical date rows.
    variance_tol : float, default 1e-12
        Tolerance threshold for near-zero variance detection (flat series).

    Raises
    ------
    ValueError
        If DataFrame is empty, contains non-positive prices, has too few rows,
        or contains flat/stale series with near-zero variance.
    """
    if df.empty:
        raise ValueError("Price DataFrame is empty.")

    if len(df.columns) == 0:
        raise ValueError("Price DataFrame contains no asset columns.")

    if len(df) < min_obs:
        raise ValueError(
            f"Insufficient historical observations. Found {len(df)} rows, minimum required is {min_obs}."
        )

    # Check for non-numeric types
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric (dtype: {df[col].dtype}).")

    # Check for non-positive prices
    min_val = df.min().min()
    if min_val <= 0:
        bad_cols = [c for c in df.columns if (df[c] <= 0).any()]
        raise ValueError(
            f"Prices must be strictly positive (> 0). Non-positive values detected in: {bad_cols}."
        )

    # Check for flat series (zero or near-zero variance)
    # Using relative variance (std / mean) or simple return variance
    pct_changes = df.pct_change().dropna()
    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    for col in df.columns:
        if str(col).upper() in cash_symbols:
            continue
        col_var = float(pct_changes[col].var()) if len(pct_changes) > 1 else 0.0
        if col_var < variance_tol:
            raise ValueError(
                f"Asset '{col}' has near-zero return variance ({col_var:.2e}), "
                "indicating stale, constant, or uninformative prices."
            )


def calculate_daily_returns(
    prices_df: pd.DataFrame,
    method: Union[Literal["simple", "arithmetic", "discrete", "log", "geometric", "continuous"], str] = "simple",
) -> pd.DataFrame:
    """
    Calculates daily simple (arithmetic) or log (continuously compounded) returns from prices.

    Simple Return:
        R_{i,t} = (P_{i,t} - P_{i,t-1}) / P_{i,t-1} = P_{i,t} / P_{i,t-1} - 1

    Log Return:
        r_{i,t} = ln(P_{i,t} / P_{i,t-1}) = ln(P_{i,t}) - ln(P_{i,t-1})

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame of historical asset prices.
    method : str, default 'simple'
        Return calculation method: 'simple' / 'arithmetic' or 'log' / 'continuous'.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily returns with the first row (NaN) dropped and DatetimeIndex preserved.

    Raises
    ------
    ValueError
        If prices contain non-positive values or method is unrecognized.
    """
    if prices_df.empty:
        raise ValueError("Prices DataFrame cannot be empty.")

    method_clean = method.lower().strip()

    if method_clean in ("simple", "arithmetic", "discrete"):
        returns = prices_df.pct_change()
    elif method_clean in ("log", "geometric", "continuous"):
        if (prices_df <= 0).any().any():
            raise ValueError("Prices must be strictly positive for log return calculation.")
        returns = np.log(prices_df / prices_df.shift(1))
    else:
        raise ValueError(
            f"Unrecognized return calculation method: '{method}'. "
            "Supported methods are 'simple' (or 'arithmetic') and 'log' (or 'continuous')."
        )

    # Drop the first row (which is NaN due to 1-period shift)
    returns = returns.iloc[1:].copy()

    # Clean any inf or -inf values
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # Ensure float64
    returns = returns.astype(np.float64)

    return returns


def clean_and_align_prices(
    df_raw: pd.DataFrame,
    freq: str = "B",
    drop_incomplete: bool = True,
    return_method: str = "simple",
    min_obs: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full data sanitization and harmonization pipeline.

    1. Normalizes DatetimeIndex (tz-naive, ascending, normalized).
    2. Harmonizes trading calendars (resamples to Business Days 'B' or Daily 'D').
    3. Propagates holiday prices via forward-fill ('ffill').
    4. Trims to common inception date across all assets.
    5. Validates positivity and variance.
    6. Calculates daily returns (simple or log).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw price DataFrame from yfinance or manual file parser.
    freq : str, default 'B'
        Target calendar frequency ('B' for business days, 'D' for calendar days).
    drop_incomplete : bool, default True
        Whether to drop any remaining incomplete rows.
    return_method : str, default 'simple'
        Return calculation method ('simple' or 'log').
    min_obs : int, default 5
        Minimum required observations after cleaning.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of `(clean_prices_df, daily_returns_df)`.

    Raises
    ------
    ValueError
        If data cleaning fails or data does not satisfy validity requirements.
    """
    if df_raw.empty:
        raise ValueError("Input DataFrame is empty.")

    # 1. Normalize datetime index
    df_norm = normalize_datetime_index(df_raw)

    # 2. Harmonize calendar with ffill
    df_aligned = align_to_calendar(df_norm, freq=freq, method="ffill")

    # 3. Trim to common inception date
    df_clean = trim_common_inception(df_aligned, drop_incomplete=drop_incomplete)

    # 4. Validate clean prices
    validate_price_data(df_clean, min_obs=min_obs)

    # 5. Compute daily returns
    df_returns = calculate_daily_returns(df_clean, method=return_method)

    return df_clean, df_returns
