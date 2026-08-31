"""
Expected Returns Estimators Module for Portfolio Optimization.

Provides mathematical estimators for asset expected returns:
1. Annualized Historical Arithmetic Mean Return
2. Annualized Compound / Geometric Mean Return (CAGR)
3. Exponentially Weighted Moving Average (EWMA) Return (decay lambda=0.94)
4. Capital Asset Pricing Model (CAPM) Expected Return with benchmark regression
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd


class ReturnMethod(str, Enum):
    """Supported expected return estimation methodologies."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    CAGR = "cagr"
    EWMA = "ewma"
    CAPM = "capm"

    @classmethod
    def from_str(cls, value: str) -> ReturnMethod:
        """Parse string to ReturnMethod enum, supporting common aliases."""
        val = value.strip().lower()
        if val in ("arithmetic", "arith", "mean", "hist", "historical"):
            return cls.ARITHMETIC
        if val in ("geometric", "geom", "cagr", "compound"):
            return cls.GEOMETRIC
        if val in ("ewma", "exponential", "ema"):
            return cls.EWMA
        if val in ("capm", "beta", "market_model"):
            return cls.CAPM
        raise ValueError(
            f"Unsupported return estimation method: '{value}'. "
            f"Allowed methods: {[m.value for m in cls]}"
        )


def annualized_arithmetic_returns(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    ann_factor: int = 252,
) -> pd.Series | float | np.ndarray:
    """
    Calculate annualized historical arithmetic mean return.

    Formula:
        mu_i = mean(R_i) * ann_factor

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Historical period returns (e.g. daily returns).
    ann_factor : int, default 252
        Annualization factor (252 for TradFi/hybrid, 365 for crypto).

    Returns
    -------
    pd.Series | float | np.ndarray
        Annualized arithmetic expected returns.
    """
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    if isinstance(returns, (pd.DataFrame, pd.Series)):
        if returns.empty:
            raise ValueError("Input returns DataFrame/Series is empty.")
        if returns.isna().any().any() if isinstance(returns, pd.DataFrame) else returns.isna().any():
            raise ValueError("Input returns contain NaN values. Clean data before modeling.")
        mean_ret = returns.mean(axis=0)
        return mean_ret * ann_factor

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Input returns array is empty.")
    if np.isnan(arr).any():
        raise ValueError("Input returns array contains NaN values.")

    mean_ret = np.mean(arr, axis=0)
    return mean_ret * ann_factor


def annualized_geometric_returns(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    ann_factor: int = 252,
    eps: float = 1e-12,
) -> pd.Series | float | np.ndarray:
    """
    Calculate annualized compound / geometric mean return (CAGR).

    Formula:
        mu_i = exp( mean( ln( max(1 + R_i, eps) ) ) * ann_factor ) - 1

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Historical period returns.
    ann_factor : int, default 252
        Annualization factor.
    eps : float, default 1e-12
        Lower bound clipping to prevent invalid log operations if R <= -1.

    Returns
    -------
    pd.Series | float | np.ndarray
        Annualized compound expected returns.
    """
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty.")
        if returns.isna().any().any():
            raise ValueError("Input returns contain NaN values.")
        clipped = returns.clip(lower=-1.0 + eps)
        log_returns = np.log1p(clipped)
        mean_log = log_returns.mean(axis=0)
        return np.expm1(mean_log * ann_factor)

    if isinstance(returns, pd.Series):
        if returns.empty:
            raise ValueError("Input returns Series is empty.")
        if returns.isna().any():
            raise ValueError("Input returns contain NaN values.")
        clipped = returns.clip(lower=-1.0 + eps)
        log_returns = np.log1p(clipped)
        mean_log = log_returns.mean()
        return float(np.expm1(mean_log * ann_factor))

    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Input returns array is empty.")
    if np.isnan(arr).any():
        raise ValueError("Input returns array contains NaN values.")

    clipped = np.maximum(arr, -1.0 + eps)
    log_returns = np.log1p(clipped)
    mean_log = np.mean(log_returns, axis=0)
    return np.expm1(mean_log * ann_factor)


def ewma_returns(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    decay: float = 0.94,
    ann_factor: int = 252,
) -> pd.Series | float | np.ndarray:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) expected returns.

    Weights decay exponentially into the past:
        w_t = (1 - lambda) * lambda^(T - t),  t = 1, ..., T
        w_tilde_t = w_t / sum(w_t) = (1 - lambda) * lambda^(T - t) / (1 - lambda^T)
        mu_i = sum_t (w_tilde_t * R_{i, t}) * ann_factor

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Historical period returns (ordered chronologically, oldest to newest).
    decay : float, default 0.94
        Smoothing parameter lambda in (0, 1).
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    pd.Series | float | np.ndarray
        Annualized EWMA expected returns.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay parameter lambda must be in (0, 1), got {decay}")
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    if isinstance(returns, (pd.DataFrame, pd.Series)):
        if returns.empty:
            raise ValueError("Input returns DataFrame/Series is empty.")
        if returns.isna().any().any() if isinstance(returns, pd.DataFrame) else returns.isna().any():
            raise ValueError("Input returns contain NaN values.")
        values = returns.values
    else:
        values = np.asarray(returns, dtype=np.float64)
        if values.size == 0:
            raise ValueError("Input returns array is empty.")
        if np.isnan(values).any():
            raise ValueError("Input returns array contains NaN values.")

    t_obs = values.shape[0] if values.ndim > 1 else len(values)
    # Weights for t=1..T (where t=T is newest, exponent 0):
    powers = np.arange(t_obs - 1, -1, -1, dtype=np.float64)
    weights = (1.0 - decay) * (decay ** powers)
    weights /= np.sum(weights)

    if values.ndim == 1:
        weighted_mean = np.sum(weights * values)
        result = float(weighted_mean * ann_factor)
        if isinstance(returns, pd.Series):
            return result
        return result

    # 2D case: values is (T, N)
    weighted_mean = weights @ values
    result_arr = weighted_mean * ann_factor

    if isinstance(returns, pd.DataFrame):
        return pd.Series(result_arr, index=returns.columns, name="ewma_returns")
    return result_arr


def calculate_capm_betas(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
) -> pd.Series | float | np.ndarray:
    """
    Calculate asset systematic risk beta coefficients relative to a market benchmark.

    Formula:
        beta_i = Cov(R_i, R_M) / Var(R_M)

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Asset return series or matrix.
    benchmark_returns : pd.Series | np.ndarray
        Market benchmark return series (e.g. SPY daily returns).

    Returns
    -------
    pd.Series | float | np.ndarray
        Asset CAPM beta coefficients.
    """
    bm_arr = np.asarray(benchmark_returns, dtype=np.float64).ravel()
    if bm_arr.size == 0:
        raise ValueError("benchmark_returns cannot be empty.")
    if np.isnan(bm_arr).any():
        raise ValueError("benchmark_returns contains NaN values.")

    var_bm = np.var(bm_arr, ddof=1)
    if var_bm < 1e-14:
        raise ValueError("Benchmark returns have zero or near-zero variance; cannot compute beta.")

    bm_demeaned = bm_arr - np.mean(bm_arr)
    t_bm = len(bm_arr)

    if isinstance(returns, pd.DataFrame):
        if len(returns) != t_bm:
            raise ValueError(
                f"Sample length mismatch: returns has {len(returns)} rows, "
                f"benchmark has {t_bm} rows."
            )
        if returns.isna().any().any():
            raise ValueError("returns contains NaN values.")
        y_demeaned = returns.values - np.mean(returns.values, axis=0)
        cov_im = (y_demeaned.T @ bm_demeaned) / (t_bm - 1)
        betas = cov_im / var_bm
        return pd.Series(betas, index=returns.columns, name="capm_beta")

    if isinstance(returns, pd.Series):
        if len(returns) != t_bm:
            raise ValueError(
                f"Sample length mismatch: returns has {len(returns)} rows, "
                f"benchmark has {t_bm} rows."
            )
        if returns.isna().any():
            raise ValueError("returns contains NaN values.")
        y_demeaned = returns.values - np.mean(returns.values)
        cov_im = np.dot(y_demeaned, bm_demeaned) / (t_bm - 1)
        return float(cov_im / var_bm)

    ret_arr = np.asarray(returns, dtype=np.float64)
    if ret_arr.shape[0] != t_bm:
        raise ValueError(
            f"Sample length mismatch: returns has {ret_arr.shape[0]} rows, "
            f"benchmark has {t_bm} rows."
        )
    if np.isnan(ret_arr).any():
        raise ValueError("returns contains NaN values.")

    if ret_arr.ndim == 1:
        y_demeaned = ret_arr - np.mean(ret_arr)
        cov_im = np.dot(y_demeaned, bm_demeaned) / (t_bm - 1)
        return float(cov_im / var_bm)

    y_demeaned = ret_arr - np.mean(ret_arr, axis=0)
    cov_im = (y_demeaned.T @ bm_demeaned) / (t_bm - 1)
    return cov_im / var_bm


def capm_expected_returns(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    benchmark_returns: Optional[pd.Series | np.ndarray] = None,
    rf: float = 0.04,
    market_return: Optional[float] = None,
    ann_factor: int = 252,
) -> tuple[pd.Series | float | np.ndarray, pd.Series | float | np.ndarray]:
    """
    Calculate expected returns via Capital Asset Pricing Model (CAPM).

    Formula:
        mu_i = R_f + beta_i * (mu_M - R_f)

    Parameters
    ----------
    returns : pd.DataFrame | pd.Series | np.ndarray
        Asset return matrix or series.
    benchmark_returns : Optional[pd.Series | np.ndarray], optional
        Benchmark market returns. If None, uses the equal-weighted portfolio of `returns`.
    rf : float, default 0.04
        Annual risk-free rate.
    market_return : Optional[float], optional
        Annualized expected market return (mu_M). If None, calculated from
        benchmark_returns arithmetic mean * ann_factor.
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    tuple[pd.Series | float | np.ndarray, pd.Series | float | np.ndarray]
        (expected_returns, betas)
    """
    if benchmark_returns is None:
        if isinstance(returns, pd.DataFrame):
            benchmark_returns = returns.mean(axis=1)
        elif isinstance(returns, pd.Series):
            benchmark_returns = returns
        else:
            arr = np.asarray(returns, dtype=np.float64)
            benchmark_returns = np.mean(arr, axis=1) if arr.ndim > 1 else arr

    betas = calculate_capm_betas(returns, benchmark_returns)

    if market_return is None:
        bm_arr = np.asarray(benchmark_returns, dtype=np.float64).ravel()
        mu_m = float(np.mean(bm_arr) * ann_factor)
    else:
        mu_m = float(market_return)

    if isinstance(betas, pd.Series):
        capm_mu = rf + betas * (mu_m - rf)
        capm_mu.name = "capm_returns"
        return capm_mu, betas
    elif isinstance(betas, float):
        capm_mu = rf + betas * (mu_m - rf)
        return capm_mu, betas
    else:
        capm_mu = rf + betas * (mu_m - rf)
        return capm_mu, betas


def calculate_expected_returns(
    returns: pd.DataFrame,
    method: str | ReturnMethod = "arithmetic",
    rf: float = 0.04,
    benchmark_returns: Optional[pd.Series] = None,
    market_return: Optional[float] = None,
    decay: float = 0.94,
    ann_factor: int = 252,
) -> pd.Series:
    """
    Unified entry point for computing annualized expected returns across all methods.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series (DatetimeIndex, asset columns).
    method : str | ReturnMethod, default 'arithmetic'
        Estimation method ('arithmetic', 'geometric'/'cagr', 'ewma', 'capm').
    rf : float, default 0.04
        Annual risk-free rate (used in CAPM).
    benchmark_returns : Optional[pd.Series], optional
        Benchmark market return series for CAPM.
    market_return : Optional[float], optional
        Expected annual market return for CAPM.
    decay : float, default 0.94
        EWMA exponential decay factor lambda in (0, 1).
    ann_factor : int, default 252
        Annualization factor (252 for TradFi, 365 for crypto).

    Returns
    -------
    pd.Series
        Annualized expected returns indexed by asset column names.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns)}")

    if isinstance(method, str):
        parsed_method = ReturnMethod.from_str(method)
    elif isinstance(method, ReturnMethod):
        parsed_method = method
    else:
        raise TypeError(f"method must be str or ReturnMethod enum, got {type(method)}")

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}

    if parsed_method == ReturnMethod.ARITHMETIC:
        res = annualized_arithmetic_returns(returns, ann_factor=ann_factor)
        out_series = pd.Series(res, index=returns.columns, name="expected_returns")

    elif parsed_method in (ReturnMethod.GEOMETRIC, ReturnMethod.CAGR):
        res = annualized_geometric_returns(returns, ann_factor=ann_factor)
        out_series = pd.Series(res, index=returns.columns, name="expected_returns")

    elif parsed_method == ReturnMethod.EWMA:
        res = ewma_returns(returns, decay=decay, ann_factor=ann_factor)
        out_series = pd.Series(res, index=returns.columns, name="expected_returns")

    elif parsed_method == ReturnMethod.CAPM:
        mu_capm, _ = capm_expected_returns(
            returns=returns,
            benchmark_returns=benchmark_returns,
            rf=rf,
            market_return=market_return,
            ann_factor=ann_factor,
        )
        out_series = pd.Series(mu_capm, index=returns.columns, name="expected_returns")
    else:
        raise ValueError(f"Unhandled method: {parsed_method}")

    # Set cash tickers explicitly to Rf
    for col in out_series.index:
        if str(col).upper() in cash_symbols:
            out_series[col] = float(rf)

    return out_series
