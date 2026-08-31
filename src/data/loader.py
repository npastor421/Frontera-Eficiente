"""
Data Loader Module for Frontera Eficiente.

Provides historical price data fetching via yfinance and robust manual file
ingestion supporting CSV, TSV, and Excel formats (Wide Prices, Wide Returns,
Long/Tidy format, European/Latin comma decimals, and automatic delimiter detection).
"""

from __future__ import annotations

import csv
import datetime
import io
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf


def validate_tickers(tickers: Union[str, Sequence[str]]) -> list[str]:
    """
    Validates and standardizes ticker symbol inputs.

    Parameters
    ----------
    tickers : str or Sequence[str]
        Single ticker string or sequence of ticker strings.

    Returns
    -------
    list[str]
        Deduplicated, stripped, valid list of ticker symbols preserving input order.

    Raises
    ------
    ValueError
        If no valid ticker symbols are provided.
    """
    if isinstance(tickers, str):
        raw_list = [t.strip() for t in tickers.split(",") if t.strip()]
    elif isinstance(tickers, (list, tuple, set, pd.Index, np.ndarray)):
        raw_list = [str(t).strip() for t in tickers if str(t).strip()]
    else:
        raise ValueError(f"Unsupported ticker format: {type(tickers)}. Expected str or Sequence[str].")

    # Deduplicate while preserving order
    seen = set()
    cleaned_tickers: list[str] = []
    for t in raw_list:
        if t not in seen:
            seen.add(t)
            cleaned_tickers.append(t)

    if not cleaned_tickers:
        raise ValueError("Ticker list cannot be empty. Please provide at least one valid ticker symbol.")

    return cleaned_tickers


CASH_TICKERS = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}


def is_cash_ticker(ticker: str) -> bool:
    """Check if a ticker symbol represents a cash / liquidity asset."""
    return str(ticker).strip().upper() in CASH_TICKERS


def fetch_asset_data(
    tickers: Union[str, Sequence[str]],
    start_date: Union[str, datetime.date, datetime.datetime],
    end_date: Union[str, datetime.date, datetime.datetime],
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Downloads historical adjusted prices from yfinance and returns a clean DataFrame.

    Handles yfinance MultiIndex structures, single vs multi-ticker downloads,
    US equities, ETFs, crypto pairs (e.g. 'BTC-USD'), Argentine CEDEARs (e.g. 'AAPL.BA'),
    benchmark indices (e.g. '^GSPC', 'SPY'), and cash/liquidity assets ('CASH', 'USD').

    Parameters
    ----------
    tickers : str or Sequence[str]
        List or comma-separated string of ticker symbols.
    start_date : str, datetime.date, or datetime.datetime
        Start date (inclusive). Format 'YYYY-MM-DD' or datetime object.
    end_date : str, datetime.date, or datetime.datetime
        End date (inclusive/exclusive depending on interval). Format 'YYYY-MM-DD' or datetime object.
    interval : str, default '1d'
        Data frequency ('1d', '1wk', '1mo').
    auto_adjust : bool, default True
        Whether to download auto-adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Index: pd.DatetimeIndex (tz-naive, ascending, normalized to 00:00:00, named 'Date')
        Columns: Ticker symbols (str)
        Values: float64 adjusted close prices

    Raises
    ------
    ValueError
        If inputs are invalid or no price data could be retrieved for any requested ticker.
    """
    valid_tickers = validate_tickers(tickers)
    risky_tickers = [t for t in valid_tickers if not is_cash_ticker(t)]
    cash_tickers = [t for t in valid_tickers if is_cash_ticker(t)]

    # Format dates
    if isinstance(start_date, (datetime.date, datetime.datetime)):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date).strip()

    if isinstance(end_date, (datetime.date, datetime.datetime)):
        end_str = end_date.strftime("%Y-%m-%d")
    else:
        end_str = str(end_date).strip()

    prices = pd.DataFrame()

    if risky_tickers:
        # Download from yfinance
        try:
            raw_df = yf.download(
                tickers=risky_tickers,
                start=start_str,
                end=end_str,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
            )
        except Exception as exc:
            raise ValueError(f"Error fetching data from yfinance for {risky_tickers}: {exc}") from exc

        if raw_df is None or raw_df.empty:
            raise ValueError(
                f"No price data found for tickers {risky_tickers} between {start_str} and {end_str}."
            )

        # Extract price slice from MultiIndex or Flat Index
        if isinstance(raw_df.columns, pd.MultiIndex):
            level_0 = [str(x) for x in raw_df.columns.get_level_values(0)]
            level_1 = [str(x) for x in raw_df.columns.get_level_values(1)]

            if "Adj Close" in level_0:
                prices = raw_df["Adj Close"].copy()
            elif "Close" in level_0:
                prices = raw_df["Close"].copy()
            elif "Adj Close" in level_1:
                prices = raw_df.xs("Adj Close", axis=1, level=1).copy()
            elif "Close" in level_1:
                prices = raw_df.xs("Close", axis=1, level=1).copy()
            else:
                first_metric = raw_df.columns.levels[0][0]
                prices = raw_df[first_metric].copy()
        else:
            flat_cols = [str(c) for c in raw_df.columns]
            if "Adj Close" in flat_cols:
                prices = raw_df[["Adj Close"]].copy()
                if len(risky_tickers) == 1:
                    prices.columns = [risky_tickers[0]]
            elif "Close" in flat_cols:
                prices = raw_df[["Close"]].copy()
                if len(risky_tickers) == 1:
                    prices.columns = [risky_tickers[0]]
            else:
                prices = raw_df.copy()

        if isinstance(prices, pd.Series):
            ticker_name = risky_tickers[0] if len(risky_tickers) == 1 else str(prices.name)
            prices = prices.to_frame(name=ticker_name)

        prices.columns = [str(col).strip() for col in prices.columns]
        matched_cols = [t for t in risky_tickers if t in prices.columns]
        if matched_cols:
            prices = prices[matched_cols]
    else:
        # Only cash tickers were provided
        date_idx = pd.date_range(start_str, end_str, freq="B", name="Date")
        prices = pd.DataFrame(index=date_idx)

    # Normalize DatetimeIndex
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_convert(None)
    prices.index = prices.index.tz_localize(None).normalize()
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    # Add synthetic cash price series compounding at baseline rate
    if cash_tickers:
        n_obs = len(prices)
        if n_obs == 0:
            prices = pd.DataFrame(index=pd.date_range(start_str, end_str, freq="B", name="Date"))
            n_obs = len(prices)
        t_steps = np.arange(n_obs, dtype=np.float64)
        for c in cash_tickers:
            prices[c] = 100.0 * np.exp((0.04 / 252.0) * t_steps)

    # Reorder columns to preserve input order
    available_ordered = [t for t in valid_tickers if t in prices.columns]
    prices = prices[available_ordered]

    prices = prices.dropna(how="all", axis=1)
    if prices.empty or len(prices.columns) == 0:
        raise ValueError(
            f"All downloaded price series for {valid_tickers} were empty or NaN between {start_str} and {end_str}."
        )

    return prices.astype(np.float64)


def _detect_csv_delimiter(sample_text: str) -> str:
    """Auto-detects CSV delimiter from sample text."""
    delimiters = [",", ";", "\t", "|"]
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        # Fallback heuristic: count occurrences in first few lines
        lines = [line for line in sample_text.splitlines() if line.strip()][:5]
        if not lines:
            return ","
        counts = {d: sum(line.count(d) for line in lines) for d in delimiters}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def _detect_and_clean_numeric_col(series: pd.Series) -> pd.Series:
    """Converts a Series with potential comma decimals, currency symbols, or percent signs to float64."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(np.float64)

    s = series.astype(str).str.strip()
    # Strip currency signs, whitespace, and letters (except scientific notation e/E)
    s = s.str.replace(r"[^\d,\.\-\+eE]", "", regex=True)

    # Handle comma decimals e.g. '1.234,56' or '12,50'
    # If string contains both '.' and ',', determine thousands vs decimal separator
    sample = s.dropna().head(50)
    has_comma_dec = sample.str.contains(r"^\s*-?\d+,\d+\s*$", regex=True).any()
    has_mixed_dots_commas = sample.str.contains(r"\d+\.\d+,\d+", regex=True).any()

    if has_mixed_dots_commas:
        # Format 1.234,56 -> remove dot, replace comma with dot
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif has_comma_dec:
        s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce").astype(np.float64)


def parse_manual_data(
    file_or_path: Any,
    is_returns: bool = False,
    date_col: Optional[str] = None,
    decimal: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parses manually uploaded or provided market data files (CSV, TSV, TXT, XLSX, XLS).

    Supports:
    - Wide format (Prices or Returns): First column (or named date col) as Date, remaining columns as assets.
    - Long / Tidy format: Columns ['Date', 'Ticker', 'Price' | 'Return'].
    - European/Latin comma decimals ('12,50' or '1.234,56').
    - Auto-detection of CSV delimiters (',', ';', '\t', '|').

    Parameters
    ----------
    file_or_path : str, Path, bytes, io.BytesIO, io.StringIO, or Streamlit UploadedFile
        The file or data stream to parse.
    is_returns : bool, default False
        Whether the file contains return series (True) or raw asset prices (False).
    date_col : Optional[str], default None
        Explicit name of the date column if auto-detection is not desired.
    decimal : Optional[str], default None
        Decimal separator (e.g. '.' or ','). Auto-detected if None.
    delimiter : Optional[str], default None
        CSV delimiter character. Auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with tz-naive DatetimeIndex (named 'Date') and numeric float64 columns.

    Raises
    ------
    ValueError
        If the file cannot be parsed, has no valid date column, or contains no numeric asset series.
    """
    df_raw: pd.DataFrame
    filename = ""

    # Determine source type
    if isinstance(file_or_path, str):
        if not file_or_path.strip():
            raise ValueError("Uploaded file or input string is empty.")
        if "\n" in file_or_path or (not Path(file_or_path).exists() and ("," in file_or_path or ";" in file_or_path)):
            # String contains raw CSV text
            text_data = file_or_path
            delim = delimiter or _detect_csv_delimiter(text_data[:2000])
            df_raw = pd.read_csv(io.StringIO(text_data), sep=delim, decimal=decimal or ".")
        else:
            p = Path(file_or_path)
            filename = p.name.lower()
            if not p.exists():
                raise ValueError(f"File not found: {file_or_path}")
            if p.is_dir():
                raise ValueError(f"Expected a file path, but received a directory: {file_or_path}")
            if p.suffix.lower() in [".xlsx", ".xls"]:
                df_raw = pd.read_excel(p)
            else:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    sample_text = f.read(4096)
                delim = delimiter or _detect_csv_delimiter(sample_text)
                df_raw = pd.read_csv(p, sep=delim, decimal=decimal or ".")
    elif isinstance(file_or_path, Path):
        p = file_or_path
        filename = p.name.lower()
        if not p.exists():
            raise ValueError(f"File not found: {file_or_path}")
        if p.is_dir():
            raise ValueError(f"Expected a file path, but received a directory: {file_or_path}")
        if p.suffix.lower() in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(p)
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                sample_text = f.read(4096)
            delim = delimiter or _detect_csv_delimiter(sample_text)
            df_raw = pd.read_csv(p, sep=delim, decimal=decimal or ".")
    elif hasattr(file_or_path, "read"):
        # Streamlit UploadedFile or BytesIO / StringIO
        filename = getattr(file_or_path, "name", "").lower()
        content_bytes = file_or_path.read()
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)

        if isinstance(content_bytes, str):
            text_data = content_bytes
            delim = delimiter or _detect_csv_delimiter(text_data[:2000])
            df_raw = pd.read_csv(io.StringIO(text_data), sep=delim, decimal=decimal or ".")
        elif filename.endswith((".xlsx", ".xls")) or content_bytes.startswith(b"PK\x03\x04") or content_bytes.startswith(b"\xd0\xcf\x11\xe0"):
            df_raw = pd.read_excel(io.BytesIO(content_bytes))
        else:
            text_data = content_bytes.decode("utf-8", errors="replace")
            delim = delimiter or _detect_csv_delimiter(text_data[:2000])
            df_raw = pd.read_csv(io.StringIO(text_data), sep=delim, decimal=decimal or ".")
    elif isinstance(file_or_path, bytes):
        if filename.endswith((".xlsx", ".xls")) or file_or_path.startswith(b"PK\x03\x04") or file_or_path.startswith(b"\xd0\xcf\x11\xe0"):
            df_raw = pd.read_excel(io.BytesIO(file_or_path))
        else:
            text_data = file_or_path.decode("utf-8", errors="replace")
            delim = delimiter or _detect_csv_delimiter(text_data[:2000])
            df_raw = pd.read_csv(io.StringIO(text_data), sep=delim, decimal=decimal or ".")
    else:
        raise ValueError(f"Unsupported input type: {type(file_or_path)}")

    if df_raw.empty:
        raise ValueError("Uploaded file is empty.")

    # Identify Date Column
    candidate_date_names = [
        "date", "fecha", "timestamp", "time", "dia", "dates", "fechas",
        "datetime", "period", "periodo", "day"
    ]
    detected_date_col: Optional[str] = None

    if date_col and date_col in df_raw.columns:
        detected_date_col = date_col
    else:
        # Check standard names (case insensitive)
        for col in df_raw.columns:
            if str(col).strip().lower() in candidate_date_names:
                detected_date_col = col
                break

    # If not found by name, inspect columns to find string/date-like columns
    if detected_date_col is None:
        for col in df_raw.columns:
            sample_s = df_raw[col].dropna().head(20)
            if sample_s.empty:
                continue

            # If column is string/object, attempt datetime parsing
            if sample_s.dtype == "object" or isinstance(sample_s.iloc[0], str):
                try:
                    parsed_sample = pd.to_datetime(sample_s, format="mixed", errors="coerce")
                    valid_years = parsed_sample.dt.year.between(1950, 2100)
                    if valid_years.sum() >= max(2, int(len(sample_s) * 0.7)):
                        detected_date_col = col
                        break
                except Exception:
                    continue
            elif pd.api.types.is_integer_dtype(sample_s):
                # Check for YYYYMMDD integers (e.g. 20240102)
                if (sample_s >= 19500101).all() and (sample_s <= 21001231).all():
                    try:
                        parsed_sample = pd.to_datetime(sample_s.astype(str), format="%Y%m%d", errors="coerce")
                        if parsed_sample.notna().sum() >= max(2, int(len(sample_s) * 0.7)):
                            detected_date_col = col
                            break
                    except Exception:
                        continue

    if detected_date_col is None:
        raise ValueError("Could not identify a valid Date column with recognizable dates in the uploaded file.")

    # Parse and set DatetimeIndex
    if pd.api.types.is_integer_dtype(df_raw[detected_date_col]) and (df_raw[detected_date_col] >= 19500101).all():
        dates = pd.to_datetime(df_raw[detected_date_col].astype(str), format="%Y%m%d", errors="coerce")
    else:
        dates = pd.to_datetime(df_raw[detected_date_col], format="mixed", errors="coerce")

    # Ensure valid years
    valid_date_mask = dates.notna() & dates.dt.year.between(1950, 2100)
    if valid_date_mask.sum() < 2:
        raise ValueError(
            f"Column '{detected_date_col}' does not contain sufficient valid dates (found {valid_date_mask.sum()})."
        )

    df_working = df_raw.loc[valid_date_mask].copy()
    dates = dates.loc[valid_date_mask]

    # Check for Long / Tidy format:
    # Looks for a ticker/asset column and a price/return value column
    ticker_col_candidates = ["ticker", "asset", "symbol", "activo", "instrumento", "name", "nombre"]
    value_col_candidates = [
        "price", "close", "adj close", "adj_close", "precio", "return",
        "retorno", "rendimiento", "value", "valor", "nav"
    ]

    other_cols = [c for c in df_working.columns if c != detected_date_col]
    found_ticker_col = None
    found_val_col = None

    for col in other_cols:
        col_lower = str(col).strip().lower()
        if col_lower in ticker_col_candidates:
            found_ticker_col = col
        elif col_lower in value_col_candidates:
            found_val_col = col

    if found_ticker_col is not None and found_val_col is not None:
        # Long format detected -> Pivot
        df_working["_clean_val"] = _detect_and_clean_numeric_col(df_working[found_val_col])
        df_working["_clean_date"] = dates
        pivoted = df_working.pivot(index="_clean_date", columns=found_ticker_col, values="_clean_val")
        result_df = pivoted.copy()
    else:
        # Wide format
        df_working.index = dates
        feature_cols = [c for c in df_working.columns if c != detected_date_col]
        if not feature_cols:
            raise ValueError("No asset columns found in the uploaded file.")

        clean_cols = {}
        for col in feature_cols:
            cleaned_series = _detect_and_clean_numeric_col(df_working[col])
            # Only keep if column has valid numeric data
            if cleaned_series.notna().sum() >= 2:
                clean_cols[str(col).strip()] = cleaned_series

        if not clean_cols:
            raise ValueError("No valid numeric asset columns could be extracted from the uploaded file.")

        result_df = pd.DataFrame(clean_cols, index=df_working.index)

    # Normalize index
    result_df.index = pd.to_datetime(result_df.index)
    if getattr(result_df.index, "tz", None) is not None:
        result_df.index = result_df.index.tz_convert(None)
    result_df.index = result_df.index.tz_localize(None).normalize()
    result_df.index.name = "Date"
    result_df = result_df.sort_index()
    result_df = result_df[~result_df.index.duplicated(keep="first")]

    # Drop columns that are completely NaN
    result_df = result_df.dropna(how="all", axis=1)

    if result_df.empty or len(result_df.columns) == 0:
        raise ValueError("Parsed dataset resulted in an empty price/return matrix.")

    # Cast to float64
    result_df = result_df.astype(np.float64)

    return result_df


# Alias for flexible naming convention
load_manual_file = parse_manual_data
