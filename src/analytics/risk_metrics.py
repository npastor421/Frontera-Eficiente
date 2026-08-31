"""
Advanced Risk Analytics & Performance Metrics Engine.

Calculates comprehensive portfolio risk and return statistics:
1. Annualized Return (Arithmetic expected return and CAGR)
2. Annualized Volatility (Square root of portfolio variance)
3. Sharpe Ratio (Annualized excess return over risk-free rate)
4. Sortino Ratio (Excess return over downside semideviation below MAR)
5. Calmar Ratio (CAGR over Maximum Drawdown)
6. Maximum Drawdown (Peak-to-trough series, peak date, valley date, recovery duration)
7. Value at Risk (VaR 95% Historical and Parametric Normal)
8. Conditional Value at Risk (CVaR 95% / Expected Shortfall)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PortfolioRiskMetrics:
    """
    Comprehensive Portfolio Risk and Performance Metrics Container.

    Supports both attribute access (e.g. `metrics.sharpe_ratio`) and
    dictionary-style subscription (e.g. `metrics['sharpe_ratio']`).
    """

    annualized_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95_hist: float
    var_95_param: float
    cvar_95_hist: float
    cvar_95_param: float
    peak_date: Optional[Any] = None
    valley_date: Optional[Any] = None
    recovery_days: Optional[int] = None
    daily_returns: Optional[pd.Series] = None
    cumulative_wealth: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None

    @property
    def var_95(self) -> float:
        """Alias for historical 95% Value at Risk."""
        return self.var_95_hist

    @property
    def cvar_95(self) -> float:
        """Alias for historical 95% Conditional Value at Risk."""
        return self.cvar_95_hist

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard Python dictionary."""
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like subscription access."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Metric '{key}' not found in PortfolioRiskMetrics.")

    def __contains__(self, key: str) -> bool:
        """Check if metric key exists in container."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric by key with optional default fallback."""
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        """Return list of metric field names."""
        return list(asdict(self).keys())

    def values(self) -> List[Any]:
        """Return list of metric values."""
        return list(asdict(self).values())

    def items(self) -> List[Tuple[str, Any]]:
        """Return key-value pairs."""
        return list(asdict(self).items())


# ===========================================================================
# 1. Portfolio Return Series Calculation
# ===========================================================================

def calculate_portfolio_returns(
    weights: Union[np.ndarray, List[float], pd.Series, Dict[str, float]],
    daily_returns: pd.DataFrame,
) -> pd.Series:
    """
    Calculate daily portfolio return series from asset weights and asset returns.

    Formula:
        R_{p, t} = sum_{i=1}^N w_i * R_{i, t}

    Parameters
    ----------
    weights : np.ndarray | list | pd.Series | dict
        Asset allocation weights.
    daily_returns : pd.DataFrame
        Historical asset daily returns (T observations, N assets).

    Returns
    -------
    pd.Series
        Portfolio daily return series with matching DatetimeIndex.
    """
    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError(f"daily_returns must be a pd.DataFrame, got {type(daily_returns)}")
    if daily_returns.empty:
        raise ValueError("daily_returns DataFrame cannot be empty.")

    tickers = list(daily_returns.columns)
    n_assets = len(tickers)

    if isinstance(weights, dict):
        w_vec = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=np.float64)
    elif isinstance(weights, pd.Series):
        if set(weights.index) == set(tickers):
            w_vec = np.array([float(weights[t]) for t in tickers], dtype=np.float64)
        else:
            w_vec = np.asarray(weights.values, dtype=np.float64)
    else:
        w_vec = np.asarray(weights, dtype=np.float64).ravel()

    if len(w_vec) != n_assets:
        raise ValueError(
            f"Weights vector length ({len(w_vec)}) does not match returns columns ({n_assets})."
        )

    port_ret_values = daily_returns.values @ w_vec
    return pd.Series(port_ret_values, index=daily_returns.index, name="portfolio_returns")


# ===========================================================================
# 2. Drawdown Series & Maximum Drawdown Analytics
# ===========================================================================

def compute_drawdown_series(
    returns_or_wealth: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> Tuple[pd.Series, float, Any, Any]:
    """
    Compute underwater drawdown series, maximum drawdown, peak date/index, and valley date/index.

    Formula:
        W_t = prod_{tau=1}^t (1 + R_{p, tau})
        M_t = max_{tau <= t} W_tau
        DD_t = (W_t - M_t) / M_t
        MDD = |min_t DD_t|

    Parameters
    ----------
    returns_or_wealth : pd.Series | pd.DataFrame | np.ndarray
        Either periodic return series (daily returns) or cumulative wealth series.

    Returns
    -------
    Tuple[pd.Series, float, Any, Any]
        (drawdown_series, max_drawdown_positive_float, peak_index, valley_index)
    """
    if isinstance(returns_or_wealth, pd.DataFrame):
        if returns_or_wealth.shape[1] == 1:
            series = returns_or_wealth.iloc[:, 0]
        else:
            raise ValueError("Input DataFrame must have exactly one column.")
    elif isinstance(returns_or_wealth, pd.Series):
        series = returns_or_wealth
    else:
        arr = np.asarray(returns_or_wealth, dtype=np.float64).ravel()
        series = pd.Series(arr)

    if series.empty:
        raise ValueError("Input series cannot be empty.")

    values = series.values

    # Determine if input is returns or cumulative wealth:
    # If all values are > 0 and mean > 1.0 (or first value is 1.0/10000.0)
    # Typically returns have values near 0 (e.g. -0.1 to 0.1).
    if np.all(values > 0) and (np.mean(values) > 2.0 or values[0] >= 1.0 and np.all(values >= 0.01)):
        wealth = values
    else:
        # Input is returns: compute compounding wealth starting at 1.0
        wealth = np.cumprod(1.0 + values)

    # Compute running maximum
    running_max = np.maximum.accumulate(wealth)
    # Drawdown in [-1.0, 0.0]
    drawdown = (wealth - running_max) / np.maximum(running_max, 1e-12)
    # Clip any positive floating inaccuracies to 0.0
    drawdown = np.minimum(drawdown, 0.0)

    dd_series = pd.Series(drawdown, index=series.index, name="drawdown")

    # Find valley (minimum drawdown)
    valley_pos = int(np.argmin(drawdown))
    max_dd = float(abs(drawdown[valley_pos]))

    # Find the peak prior to or at valley
    if valley_pos == 0:
        peak_pos = 0
    else:
        peak_pos = int(np.argmax(wealth[: valley_pos + 1]))

    peak_idx = series.index[peak_pos]
    valley_idx = series.index[valley_pos]

    return dd_series, max_dd, peak_idx, valley_idx


def calculate_drawdown_series(
    returns_or_wealth: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> pd.Series:
    """
    Calculate underwater drawdown series bounded in [-1.0, 0.0].

    Parameters
    ----------
    returns_or_wealth : pd.Series | pd.DataFrame | np.ndarray
        Returns or cumulative wealth series.

    Returns
    -------
    pd.Series
        Drawdown series.
    """
    dd_series, _, _, _ = compute_drawdown_series(returns_or_wealth)
    return dd_series


def calculate_max_drawdown(
    returns_or_wealth: Union[pd.Series, pd.DataFrame, np.ndarray],
) -> float:
    """
    Calculate Maximum Drawdown as a non-negative float.

    Parameters
    ----------
    returns_or_wealth : pd.Series | pd.DataFrame | np.ndarray
        Returns or cumulative wealth series.

    Returns
    -------
    float
        Maximum drawdown (e.g. 0.25 for 25% drawdown).
    """
    _, max_dd, _, _ = compute_drawdown_series(returns_or_wealth)
    return max_dd


# ===========================================================================
# 3. Sortino & Calmar Ratios
# ===========================================================================

def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    rf: float = 0.04,
    target_return: float = 0.0,
    ann_factor: int = 252,
) -> float:
    """
    Calculate annualized Sortino Ratio measuring excess return per unit of downside risk.

    Formula:
        Sortino = (mu_{ann} - R_f) / sigma_D
        sigma_D = sqrt( 1/T * sum_{t=1}^T min(0, R_{p, t} - tau)^2 ) * sqrt(ann_factor)

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Daily portfolio return series.
    rf : float, default 0.04
        Annualized risk-free rate.
    target_return : float, default 0.0
        Minimum Acceptable Return (MAR) annualized. Default 0.0.
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    float
        Annualized Sortino ratio. Returns float('inf') or np.nan if no downside risk.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0

    daily_rf = rf / ann_factor
    daily_mar = target_return / ann_factor

    # Downside deviations below MAR
    downside_diff = np.minimum(0.0, arr - daily_mar)
    downside_var = np.mean(downside_diff ** 2)
    downside_dev = np.sqrt(downside_var) * np.sqrt(ann_factor)

    ann_return = float(np.mean(arr) * ann_factor)
    excess_return = ann_return - rf

    if downside_dev < 1e-12:
        return float("inf") if excess_return > 0 else 0.0

    return float(excess_return / downside_dev)


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar Ratio measuring compound return relative to maximum drawdown.

    Formula:
        Calmar = CAGR / |MDD|

    Parameters
    ----------
    cagr : float
        Compound Annual Growth Rate.
    max_drawdown : float
        Maximum Drawdown (positive float).

    Returns
    -------
    float
        Calmar ratio. Returns np.nan or float('inf') if max_drawdown is 0.
    """
    if max_drawdown < 1e-12:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / max_drawdown)


# ===========================================================================
# 4. VaR & CVaR (Historical and Parametric Normal)
# ===========================================================================

def compute_historical_var_cvar(
    returns: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Calculate Historical Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Historical periodic return series.
    alpha : float, default 0.05
        Significance level (e.g. 0.05 for 95% confidence).

    Returns
    -------
    Tuple[float, float]
        (historical_var, historical_cvar) as positive risk values.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, 0.0

    # Quantile at alpha (e.g. 5th percentile)
    q_alpha = float(np.percentile(arr, alpha * 100.0))
    var_hist = -q_alpha

    # Tail returns at or below the quantile threshold
    tail_returns = arr[arr <= q_alpha]
    if len(tail_returns) > 0:
        cvar_hist = -float(np.mean(tail_returns))
    else:
        cvar_hist = var_hist

    # Enforce coherent risk invariant CVaR >= VaR
    cvar_hist = max(cvar_hist, var_hist)
    return float(var_hist), float(cvar_hist)


def compute_parametric_var_cvar(
    mu: float,
    sigma: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Calculate Parametric Gaussian Value at Risk (VaR) and CVaR (Expected Shortfall).

    Formula:
        z = norm.ppf(1 - alpha)  (z_0.95 = 1.6448536)
        VaR = z * sigma - mu
        CVaR = sigma * (phi(z) / alpha) - mu

    Parameters
    ----------
    mu : float
        Mean return (daily or annualized matching sigma).
    sigma : float
        Standard deviation (daily or annualized matching mu).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    Tuple[float, float]
        (parametric_var, parametric_cvar) as positive risk values.
    """
    z = float(stats.norm.ppf(1.0 - alpha))
    phi_z = float(stats.norm.pdf(z))

    var_param = z * sigma - mu
    cvar_param = (phi_z / alpha) * sigma - mu

    return float(var_param), float(cvar_param)


def calculate_var_95(
    returns: Union[pd.Series, np.ndarray],
    method: str = "historical",
    confidence: float = 0.95,
    ann_factor: int = 252,
) -> float:
    """
    Calculate 95% Value at Risk via Historical or Parametric method.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Historical daily returns.
    method : str, default 'historical'
        'historical' or 'parametric'.
    confidence : float, default 0.95
        Confidence level.
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    float
        VaR 95% value.
    """
    alpha = 1.0 - confidence
    if method.lower().startswith("param"):
        arr = np.asarray(returns, dtype=np.float64).ravel()
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        var_val, _ = compute_parametric_var_cvar(mu, sigma, alpha=alpha)
        return var_val
    else:
        var_val, _ = compute_historical_var_cvar(returns, alpha=alpha)
        return var_val


def calculate_cvar_95(
    returns: Union[pd.Series, np.ndarray],
    method: str = "historical",
    confidence: float = 0.95,
    ann_factor: int = 252,
) -> float:
    """
    Calculate 95% Conditional Value at Risk (Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Historical daily returns.
    method : str, default 'historical'
        'historical' or 'parametric'.
    confidence : float, default 0.95
        Confidence level.
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    float
        CVaR 95% value.
    """
    alpha = 1.0 - confidence
    if method.lower().startswith("param"):
        arr = np.asarray(returns, dtype=np.float64).ravel()
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        _, cvar_val = compute_parametric_var_cvar(mu, sigma, alpha=alpha)
        return cvar_val
    else:
        _, cvar_val = compute_historical_var_cvar(returns, alpha=alpha)
        return cvar_val


# ===========================================================================
# 5. Master Portfolio Risk & Performance Metrics Engine
# ===========================================================================

def compute_portfolio_risk_metrics(
    weights: Union[np.ndarray, List[float], pd.Series, Dict[str, float]],
    daily_returns: pd.DataFrame,
    expected_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    rf: float = 0.04,
    ann_factor: int = 252,
) -> PortfolioRiskMetrics:
    """
    Compute comprehensive suite of risk, return, and performance metrics for a portfolio.

    Parameters
    ----------
    weights : np.ndarray | list | pd.Series | dict
        Asset allocation weights.
    daily_returns : pd.DataFrame
        Asset daily returns.
    expected_returns : Optional[pd.Series | np.ndarray], optional
        Pre-calculated annualized expected returns. If None, uses arithmetic mean.
    cov_matrix : Optional[pd.DataFrame | np.ndarray], optional
        Pre-calculated annualized covariance matrix. If None, uses sample covariance.
    rf : float, default 0.04
        Annualized risk-free rate.
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    PortfolioRiskMetrics
        Dataclass container with all computed metrics, supporting dictionary access.
    """
    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError(f"daily_returns must be a pd.DataFrame, got {type(daily_returns)}")

    # 1. Compute portfolio daily returns series
    port_returns = calculate_portfolio_returns(weights, daily_returns)
    ret_arr = port_returns.values
    t_days = len(ret_arr)

    # 2. Extract weights vector
    tickers = list(daily_returns.columns)
    if isinstance(weights, dict):
        w_vec = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=np.float64)
    elif isinstance(weights, pd.Series):
        if set(weights.index) == set(tickers):
            w_vec = np.array([float(weights[t]) for t in tickers], dtype=np.float64)
        else:
            w_vec = np.asarray(weights.values, dtype=np.float64)
    else:
        w_vec = np.asarray(weights, dtype=np.float64).ravel()

    # 3. Annualized Return (Arithmetic)
    if expected_returns is not None:
        if isinstance(expected_returns, (pd.Series, pd.DataFrame)):
            mu_arr = expected_returns.values.ravel()
        else:
            mu_arr = np.asarray(expected_returns, dtype=np.float64).ravel()
        ann_return = float(np.dot(w_vec, mu_arr))
    else:
        ann_return = float(np.mean(ret_arr) * ann_factor)

    # 4. Compound Annual Growth Rate (CAGR)
    cum_wealth = pd.Series(np.cumprod(1.0 + ret_arr), index=port_returns.index, name="wealth")
    final_wealth = float(cum_wealth.iloc[-1]) if t_days > 0 else 1.0
    if t_days > 0 and final_wealth > 0:
        cagr = float((final_wealth ** (ann_factor / t_days)) - 1.0)
    else:
        cagr = -1.0 if final_wealth <= 0 else 0.0

    # 5. Annualized Volatility
    if cov_matrix is not None:
        if isinstance(cov_matrix, (pd.DataFrame, pd.Series)):
            cov_mat = cov_matrix.values
        else:
            cov_mat = np.asarray(cov_matrix, dtype=np.float64)
        var_p = float(w_vec.T @ cov_mat @ w_vec)
        ann_vol = float(np.sqrt(max(var_p, 0.0)))
    else:
        ann_vol = float(np.std(ret_arr, ddof=1) * np.sqrt(ann_factor)) if t_days > 1 else 0.0

    # 6. Sharpe Ratio
    sharpe_ratio = float((ann_return - rf) / ann_vol) if ann_vol > 1e-12 else 0.0

    # 7. Sortino Ratio
    sortino_ratio = calculate_sortino_ratio(ret_arr, rf=rf, target_return=0.0, ann_factor=ann_factor)

    # 8. Drawdown Analytics & Max Drawdown
    dd_series, max_dd, peak_date, valley_date = compute_drawdown_series(cum_wealth)

    # Recovery days calculation
    valley_pos = list(cum_wealth.index).index(valley_date) if valley_date in cum_wealth.index else 0
    peak_pos = list(cum_wealth.index).index(peak_date) if peak_date in cum_wealth.index else 0
    peak_val = float(cum_wealth.iloc[peak_pos])
    recovery_days = None
    if valley_pos < t_days - 1:
        post_valley_wealth = cum_wealth.iloc[valley_pos + 1 :]
        recovered = post_valley_wealth[post_valley_wealth >= peak_val]
        if not recovered.empty:
            rec_pos = list(cum_wealth.index).index(recovered.index[0])
            recovery_days = int(rec_pos - peak_pos)

    # 9. Calmar Ratio
    calmar_ratio = calculate_calmar_ratio(cagr, max_dd)

    # 10. VaR & CVaR (Historical and Parametric)
    var_95_hist, cvar_95_hist = compute_historical_var_cvar(ret_arr, alpha=0.05)

    # Parametric: using daily mean & daily vol
    daily_mu = float(np.mean(ret_arr))
    daily_sigma = float(np.std(ret_arr, ddof=1)) if t_days > 1 else 0.0
    var_95_param, cvar_95_param = compute_parametric_var_cvar(daily_mu, daily_sigma, alpha=0.05)

    return PortfolioRiskMetrics(
        annualized_return=ann_return,
        cagr=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_dd,
        var_95_hist=var_95_hist,
        var_95_param=var_95_param,
        cvar_95_hist=cvar_95_hist,
        cvar_95_param=cvar_95_param,
        peak_date=peak_date,
        valley_date=valley_date,
        recovery_days=recovery_days,
        daily_returns=port_returns,
        cumulative_wealth=cum_wealth,
        drawdown_series=dd_series,
    )
