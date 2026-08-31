"""
Robust Covariance Matrix Estimators Module for Portfolio Optimization.

Provides mathematical estimators for asset covariance matrices:
1. Classical Unbiased Sample Covariance Matrix
2. Ledoit-Wolf Analytical Shrinkage (Constant Correlation Target, Ledoit & Wolf 2004)
3. Ledoit-Wolf Analytical Shrinkage (Diagonal / Scaled Identity Target via scikit-learn)
4. Exponentially Weighted Moving Average (EWMA) Covariance Matrix (RiskMetrics decay=0.94)
5. Covariance to Correlation Matrix transformation helper
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class CovarianceMethod(str, Enum):
    """Supported covariance estimation methodologies."""

    SAMPLE = "sample"
    LEDOIT_WOLF_CC = "ledoit_wolf_constant_correlation"
    LEDOIT_WOLF_DIAG = "ledoit_wolf_diagonal"
    EWMA = "ewma"

    @classmethod
    def from_str(cls, value: str) -> CovarianceMethod:
        """Parse string to CovarianceMethod enum, supporting common aliases."""
        val = value.strip().lower()
        if val in ("sample", "empirical", "hist", "historical", "classical"):
            return cls.SAMPLE
        if val in (
            "ledoit_wolf_cc",
            "ledoit_wolf_constant_correlation",
            "lw_cc",
            "constant_correlation",
            "cc",
        ):
            return cls.LEDOIT_WOLF_CC
        if val in (
            "ledoit_wolf_diag",
            "ledoit_wolf_diagonal",
            "lw_diag",
            "lw_sklearn",
            "ledoit_wolf",
            "diagonal",
        ):
            return cls.LEDOIT_WOLF_DIAG
        if val in ("ewma", "exponential", "riskmetrics", "ema"):
            return cls.EWMA
        raise ValueError(
            f"Unsupported covariance estimation method: '{value}'. "
            f"Allowed methods: {[m.value for m in cls]}"
        )


def sample_covariance(
    returns: pd.DataFrame | np.ndarray,
    ann_factor: int = 252,
    ddof: int = 1,
) -> pd.DataFrame | np.ndarray:
    """
    Calculate unbiased annualized sample covariance matrix.

    Formula:
        S = 1/(T - ddof) * Y^T * Y * ann_factor
        where Y = R - mean(R)

    Parameters
    ----------
    returns : pd.DataFrame | np.ndarray
        Historical period returns matrix (T observations, N assets).
    ann_factor : int, default 252
        Annualization factor.
    ddof : int, default 1
        Delta degrees of freedom (1 for unbiased estimator).

    Returns
    -------
    pd.DataFrame | np.ndarray
        Annualized sample covariance matrix.
    """
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty.")
        if returns.isna().any().any():
            raise ValueError("Input returns contain NaN values.")
        cov_df = returns.cov(ddof=ddof) * ann_factor
        # Enforce exact symmetry
        cov_sym = (cov_df + cov_df.T) / 2.0
        return cov_sym

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] <= ddof:
        raise ValueError(
            f"Sample size T={arr.shape[0]} must be greater than ddof={ddof}."
        )
    if np.isnan(arr).any():
        raise ValueError("Input returns array contains NaN values.")

    cov_mat = np.cov(arr, rowvar=False, ddof=ddof) * ann_factor
    if arr.shape[1] == 1:
        return np.array([[float(cov_mat)]], dtype=np.float64)
    cov_sym = (cov_mat + cov_mat.T) / 2.0
    return cov_sym


def ledoit_wolf_constant_correlation(
    returns: pd.DataFrame | np.ndarray,
    ann_factor: int = 252,
) -> tuple[pd.DataFrame | np.ndarray, float]:
    """
    Calculate analytical Ledoit-Wolf shrinkage covariance with Constant Correlation target.

    Reference:
        Olivier Ledoit and Michael Wolf (2004).
        "Honey, I Shrunk the Sample Covariance Matrix",
        The Journal of Portfolio Management, 30(4), 110-119.

    Formula:
        Sigma_LW = (delta* * F + (1 - delta*) * S) * ann_factor
        where F_ii = S_ii, F_ij = r_bar * sqrt(S_ii * S_jj)

    Parameters
    ----------
    returns : pd.DataFrame | np.ndarray
        Historical period returns matrix (T observations, N assets).
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    tuple[pd.DataFrame | np.ndarray, float]
        (shrunk_covariance_matrix, optimal_shrinkage_intensity_delta)
    """
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    is_df = isinstance(returns, pd.DataFrame)
    if is_df:
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty.")
        if returns.isna().any().any():
            raise ValueError("Input returns contain NaN values.")
        cols = returns.columns
        x = returns.values
    else:
        x = np.asarray(returns, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.size == 0:
            raise ValueError("Input returns array is empty.")
        if np.isnan(x).any():
            raise ValueError("Input returns array contains NaN values.")
        cols = None

    t_obs, n_assets = x.shape
    if t_obs < 2:
        raise ValueError(f"Sample size T={t_obs} must be at least 2.")

    # 1D single asset case
    if n_assets == 1:
        s_single = np.var(x, ddof=1) * ann_factor
        mat = np.array([[s_single]], dtype=np.float64)
        if is_df:
            return pd.DataFrame(mat, index=cols, columns=cols), 0.0
        return mat, 0.0

    # Demean returns
    mean_x = np.mean(x, axis=0)
    y = x - mean_x  # shape: (T, N)

    # Sample covariance S (using 1/T normalization as in Ledoit-Wolf 2004 derivation)
    s = (y.T @ y) / t_obs  # shape: (N, N)

    # Asset variances and standard deviations
    var = np.diag(s)
    # Avoid zero division if asset is completely constant
    std = np.sqrt(np.maximum(var, 1e-16))

    # Sample correlation matrix
    outer_std = np.outer(std, std)
    corr = s / np.maximum(outer_std, 1e-16)
    np.fill_diagonal(corr, 1.0)

    # Average correlation r_bar across all distinct pairs (i != j)
    # Total sum of off-diagonals / (N * (N - 1))
    sum_off_diag = np.sum(corr) - float(n_assets)
    num_pairs = n_assets * (n_assets - 1)
    r_bar = sum_off_diag / float(num_pairs)
    # Clip r_bar to valid range [-1, 1]
    r_bar = max(-1.0, min(1.0, float(r_bar)))

    # Target matrix F
    f = r_bar * outer_std
    np.fill_diagonal(f, var)

    # Pi-hat calculation: asymptotic variance of sample covariance
    # z_{t, i, j} = y_{t, i} * y_{t, j} - s_{i, j}
    y_outer = y[:, :, None] * y[:, None, :]  # shape: (T, N, N)
    z = y_outer - s[None, :, :]  # shape: (T, N, N)
    pi_mat = np.mean(z ** 2, axis=0)  # shape: (N, N)
    pi_hat = float(np.sum(pi_mat))

    # Rho-hat calculation: asymptotic covariance between sample covariance and target F
    # dev_var_{t, i} = y_{t, i}^2 - s_{i, i}
    dev_var = (y ** 2) - var[None, :]  # shape: (T, N)

    # theta_{ii, ij} = 1/T sum_t dev_var_{t, i} * z_{t, i, j}
    theta_ii_ij = np.mean(dev_var[:, :, None] * z, axis=0)  # shape: (N, N)
    # theta_{jj, ij} = 1/T sum_t dev_var_{t, j} * z_{t, i, j}
    theta_jj_ij = np.mean(dev_var[:, None, :] * z, axis=0)  # shape: (N, N)

    # sqrt_ratio[i, j] = std[j] / std[i]
    sqrt_ratio = np.outer(1.0 / std, std)
    rho_mat = 0.5 * r_bar * (sqrt_ratio * theta_ii_ij + (1.0 / np.maximum(sqrt_ratio, 1e-16)) * theta_jj_ij)
    np.fill_diagonal(rho_mat, np.diag(pi_mat))
    rho_hat = float(np.sum(rho_mat))

    # Gamma-hat: squared Frobenius distance between target F and sample S
    gamma_hat = float(np.sum((f - s) ** 2))

    # Optimal analytical shrinkage intensity delta*
    if gamma_hat <= 1e-16 or np.isnan(gamma_hat):
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, float(kappa / t_obs)))

    # Compute shrunk covariance matrix
    sigma = delta * f + (1.0 - delta) * s
    # Annualize
    sigma_ann = sigma * ann_factor
    # Enforce exact numerical symmetry
    sigma_ann = (sigma_ann + sigma_ann.T) / 2.0

    if is_df:
        return pd.DataFrame(sigma_ann, index=cols, columns=cols), delta
    return sigma_ann, delta


def ledoit_wolf_diagonal(
    returns: pd.DataFrame | np.ndarray,
    ann_factor: int = 252,
) -> tuple[pd.DataFrame | np.ndarray, float]:
    """
    Calculate Ledoit-Wolf shrinkage covariance with Diagonal / Scaled Identity target.

    Uses `sklearn.covariance.LedoitWolf` under the hood.

    Parameters
    ----------
    returns : pd.DataFrame | np.ndarray
        Historical period returns matrix (T observations, N assets).
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    tuple[pd.DataFrame | np.ndarray, float]
        (shrunk_covariance_matrix, shrinkage_intensity_delta)
    """
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    is_df = isinstance(returns, pd.DataFrame)
    if is_df:
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty.")
        if returns.isna().any().any():
            raise ValueError("Input returns contain NaN values.")
        cols = returns.columns
        x = returns.values
    else:
        x = np.asarray(returns, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.size == 0:
            raise ValueError("Input returns array is empty.")
        if np.isnan(x).any():
            raise ValueError("Input returns array contains NaN values.")
        cols = None

    t_obs, n_assets = x.shape
    if t_obs < 2:
        raise ValueError(f"Sample size T={t_obs} must be at least 2.")

    if n_assets == 1:
        s_single = np.var(x, ddof=1) * ann_factor
        mat = np.array([[s_single]], dtype=np.float64)
        if is_df:
            return pd.DataFrame(mat, index=cols, columns=cols), 0.0
        return mat, 0.0

    lw = LedoitWolf(assume_centered=False)
    lw.fit(x)

    sigma_ann = lw.covariance_ * ann_factor
    sigma_ann = (sigma_ann + sigma_ann.T) / 2.0
    delta = float(lw.shrinkage_)

    if is_df:
        return pd.DataFrame(sigma_ann, index=cols, columns=cols), delta
    return sigma_ann, delta


def ewma_covariance(
    returns: pd.DataFrame | np.ndarray,
    decay: float = 0.94,
    ann_factor: int = 252,
) -> pd.DataFrame | np.ndarray:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) covariance matrix.

    Weights:
        w_t = (1 - lambda) * lambda^(T - t)
        w_tilde_t = w_t / sum(w_t)
        mu_ewma = sum_t (w_tilde_t * R_t)
        Sigma_EWMA = sum_t w_tilde_t * (R_t - mu_ewma)(R_t - mu_ewma)^T * ann_factor

    Parameters
    ----------
    returns : pd.DataFrame | np.ndarray
        Historical period returns matrix (T observations, N assets).
    decay : float, default 0.94
        Smoothing parameter lambda in (0, 1).
    ann_factor : int, default 252
        Annualization factor.

    Returns
    -------
    pd.DataFrame | np.ndarray
        Annualized EWMA covariance matrix.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay parameter lambda must be in (0, 1), got {decay}")
    if ann_factor <= 0:
        raise ValueError(f"ann_factor must be positive, got {ann_factor}")

    is_df = isinstance(returns, pd.DataFrame)
    if is_df:
        if returns.empty:
            raise ValueError("Input returns DataFrame is empty.")
        if returns.isna().any().any():
            raise ValueError("Input returns contain NaN values.")
        cols = returns.columns
        x = returns.values
    else:
        x = np.asarray(returns, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.size == 0:
            raise ValueError("Input returns array is empty.")
        if np.isnan(x).any():
            raise ValueError("Input returns array contains NaN values.")
        cols = None

    t_obs, n_assets = x.shape
    if t_obs < 1:
        raise ValueError("Returns matrix has 0 observations.")

    # Chronological decay weights (t=T has power 0, t=1 has power T-1)
    powers = np.arange(t_obs - 1, -1, -1, dtype=np.float64)
    w = (1.0 - decay) * (decay ** powers)
    w_tilde = w / np.sum(w)  # Shape (T,)

    # Weighted mean
    mu_ewma = w_tilde @ x  # Shape (N,)

    # Demeaned returns
    y_tilde = x - mu_ewma  # Shape (T, N)

    # Vectorized weighted outer product: (sqrt(w_tilde) * y_tilde)^T @ (sqrt(w_tilde) * y_tilde)
    y_weighted = np.sqrt(w_tilde)[:, None] * y_tilde
    cov_ewma = (y_weighted.T @ y_weighted) * ann_factor

    # Enforce exact numerical symmetry
    cov_sym = (cov_ewma + cov_ewma.T) / 2.0

    if is_df:
        return pd.DataFrame(cov_sym, index=cols, columns=cols)
    return cov_sym


def covariance_to_correlation(
    cov_matrix: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    Formula:
        Corr_ij = Cov_ij / (sigma_i * sigma_j)

    Parameters
    ----------
    cov_matrix : pd.DataFrame | np.ndarray
        Covariance matrix (N, N).

    Returns
    -------
    pd.DataFrame | np.ndarray
        Correlation matrix with 1.0 on diagonal and values bounded in [-1.0, 1.0].
    """
    is_df = isinstance(cov_matrix, pd.DataFrame)
    if is_df:
        cols = cov_matrix.columns
        idx = cov_matrix.index
        mat = cov_matrix.values
    else:
        mat = np.asarray(cov_matrix, dtype=np.float64)
        cols = None
        idx = None

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"cov_matrix must be square 2D, got shape {mat.shape}")

    diag = np.diag(mat)
    # Defensively avoid sqrt of negative or zero
    std = np.sqrt(np.maximum(diag, 0.0))
    outer_std = np.outer(std, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_std > 1e-12, mat / outer_std, 0.0)

    # Symmetrize, set diagonal to 1.0 and clip bounds
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    if is_df:
        return pd.DataFrame(corr, index=idx, columns=cols)
    return corr


# Backward compatibility alias
corr_from_covariance = covariance_to_correlation


def estimate_covariance_matrix(
    returns: pd.DataFrame,
    method: str | CovarianceMethod = "ledoit_wolf_cc",
    ann_factor: int = 252,
    decay: float = 0.94,
) -> tuple[pd.DataFrame, dict]:
    """
    Unified entry point for estimating annualized covariance matrices.

    Parameters
    ----------
    returns : pd.DataFrame
        Clean historical daily returns (DatetimeIndex, asset columns).
    method : str | CovarianceMethod, default 'ledoit_wolf_cc'
        Covariance method ('sample', 'ledoit_wolf_cc', 'ledoit_wolf_diag', 'ewma').
    ann_factor : int, default 252
        Annualization factor.
    decay : float, default 0.94
        Smoothing decay factor lambda for EWMA.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (annualized_covariance_matrix, metadata_dict)
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns)}")

    if isinstance(method, str):
        parsed_method = CovarianceMethod.from_str(method)
    elif isinstance(method, CovarianceMethod):
        parsed_method = method
    else:
        raise TypeError(f"method must be str or CovarianceMethod enum, got {type(method)}")

    cash_symbols = {"CASH", "USD", "USD_CASH", "LIQUIDEZ", "EFECTIVO", "MONEY", "CASH.USD"}
    cash_cols = [c for c in returns.columns if str(c).upper() in cash_symbols]
    risky_cols = [c for c in returns.columns if str(c).upper() not in cash_symbols]

    if cash_cols and risky_cols:
        # Estimate on risky subset
        risky_returns = returns[risky_cols]
        cov_risky, metadata = estimate_covariance_matrix(
            risky_returns, method=method, ann_factor=ann_factor, decay=decay
        )
        all_cols = list(returns.columns)
        full_cov = pd.DataFrame(0.0, index=all_cols, columns=all_cols)
        for r_col in risky_cols:
            for c_col in risky_cols:
                full_cov.loc[r_col, c_col] = cov_risky.loc[r_col, c_col]
        metadata["n_assets"] = len(all_cols)
        metadata["cash_assets"] = cash_cols
        return full_cov, metadata
    elif cash_cols and not risky_cols:
        all_cols = list(returns.columns)
        full_cov = pd.DataFrame(0.0, index=all_cols, columns=all_cols)
        metadata = {
            "method": parsed_method.value,
            "ann_factor": ann_factor,
            "decay": decay,
            "t_samples": len(returns),
            "n_assets": len(all_cols),
            "shrinkage_delta": 0.0,
            "cash_assets": cash_cols,
        }
        return full_cov, metadata

    t_obs, n_assets = returns.shape
    metadata = {
        "method": parsed_method.value,
        "ann_factor": ann_factor,
        "decay": decay,
        "t_samples": t_obs,
        "n_assets": n_assets,
        "shrinkage_delta": None,
    }

    if parsed_method == CovarianceMethod.SAMPLE:
        cov_df = sample_covariance(returns, ann_factor=ann_factor)
        return cov_df, metadata

    elif parsed_method == CovarianceMethod.LEDOIT_WOLF_CC:
        cov_df, delta = ledoit_wolf_constant_correlation(returns, ann_factor=ann_factor)
        metadata["shrinkage_delta"] = delta
        return cov_df, metadata

    elif parsed_method == CovarianceMethod.LEDOIT_WOLF_DIAG:
        cov_df, delta = ledoit_wolf_diagonal(returns, ann_factor=ann_factor)
        metadata["shrinkage_delta"] = delta
        return cov_df, metadata

    elif parsed_method == CovarianceMethod.EWMA:
        cov_df = ewma_covariance(returns, decay=decay, ann_factor=ann_factor)
        return cov_df, metadata

    raise ValueError(f"Unhandled covariance method: {parsed_method}")
