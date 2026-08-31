"""
Risk Modeling, Expected Returns & Robust Covariance Package.

Milestone 2 Module for Frontera Eficiente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.covariance import (
    CovarianceMethod,
    covariance_to_correlation,
    estimate_covariance_matrix,
    ewma_covariance,
    ledoit_wolf_constant_correlation,
    ledoit_wolf_diagonal,
    sample_covariance,
)
from src.models.returns import (
    ReturnMethod,
    annualized_arithmetic_returns,
    annualized_geometric_returns,
    calculate_capm_betas,
    calculate_expected_returns,
    capm_expected_returns,
    ewma_returns,
)
from src.models.stability import (
    calculate_condition_number,
    enforce_symmetry,
    ensure_positive_semidefinite,
    get_eigenvalues,
    is_positive_semidefinite,
    nearest_psd_higham,
)


@dataclass(frozen=True)
class RiskModelConfig:
    """Configuration container for statistical risk and return modeling."""

    return_method: ReturnMethod = ReturnMethod.ARITHMETIC
    covariance_method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF_CC
    risk_free_rate: float = 0.04
    market_benchmark: str = "SPY"
    ewma_decay: float = 0.94
    annualization_factor: int = 252


@dataclass
class RiskModelOutput:
    """Container holding calculated return and risk matrix structures."""

    expected_returns: pd.Series
    covariance_matrix: pd.DataFrame
    correlation_matrix: pd.DataFrame
    annual_volatilities: pd.Series
    shrinkage_delta: Optional[float]
    condition_number: float
    is_psd: bool
    eigenvalues: np.ndarray
    was_repaired: bool = False


def build_risk_model(
    returns: pd.DataFrame,
    config: Optional[RiskModelConfig] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> RiskModelOutput:
    """
    Execute full statistical risk and expected return estimation pipeline.

    Parameters
    ----------
    returns : pd.DataFrame
        Clean historical daily returns (DatetimeIndex, asset columns).
    config : Optional[RiskModelConfig], optional
        Model parameters and methods configuration.
    benchmark_returns : Optional[pd.Series], optional
        Benchmark market returns series (used if CAPM return method selected).

    Returns
    -------
    RiskModelOutput
        Structured dataclass containing expected returns, covariance, correlation,
        volatilities, eigenvalues, condition number, and stability flags.
    """
    if config is None:
        config = RiskModelConfig()

    # 1. Expected Returns
    expected_ret = calculate_expected_returns(
        returns=returns,
        method=config.return_method,
        rf=config.risk_free_rate,
        benchmark_returns=benchmark_returns,
        decay=config.ewma_decay,
        ann_factor=config.annualization_factor,
    )

    # 2. Covariance Matrix
    cov_matrix, meta = estimate_covariance_matrix(
        returns=returns,
        method=config.covariance_method,
        ann_factor=config.annualization_factor,
        decay=config.ewma_decay,
    )

    # 3. Numerical Stability & PSD Repair
    psd_cov, was_repaired, cond_num = ensure_positive_semidefinite(cov_matrix)
    eigenvalues = get_eigenvalues(psd_cov)
    is_psd_flag = is_positive_semidefinite(psd_cov)

    # 4. Correlation Matrix and Annualized Volatilities
    corr_matrix = covariance_to_correlation(psd_cov)
    volatilities = pd.Series(
        np.sqrt(np.diag(psd_cov)), index=psd_cov.columns, name="annual_volatility"
    )

    return RiskModelOutput(
        expected_returns=expected_ret,
        covariance_matrix=psd_cov if isinstance(psd_cov, pd.DataFrame) else pd.DataFrame(psd_cov, index=returns.columns, columns=returns.columns),
        correlation_matrix=corr_matrix if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame(corr_matrix, index=returns.columns, columns=returns.columns),
        annual_volatilities=volatilities,
        shrinkage_delta=meta.get("shrinkage_delta"),
        condition_number=cond_num,
        is_psd=is_psd_flag,
        eigenvalues=eigenvalues,
        was_repaired=was_repaired,
    )


__all__ = [
    # Enums & Config
    "ReturnMethod",
    "CovarianceMethod",
    "RiskModelConfig",
    "RiskModelOutput",
    "build_risk_model",
    # Returns
    "calculate_expected_returns",
    "annualized_arithmetic_returns",
    "annualized_geometric_returns",
    "ewma_returns",
    "calculate_capm_betas",
    "capm_expected_returns",
    # Covariance
    "estimate_covariance_matrix",
    "sample_covariance",
    "ledoit_wolf_constant_correlation",
    "ledoit_wolf_diagonal",
    "ewma_covariance",
    "covariance_to_correlation",
    # Stability
    "ensure_positive_semidefinite",
    "enforce_symmetry",
    "get_eigenvalues",
    "is_positive_semidefinite",
    "calculate_condition_number",
    "nearest_psd_higham",
]
