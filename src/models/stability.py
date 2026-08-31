"""
Numerical Stability & Positive Semi-Definite (PSD) Diagnostics Module.

Provides numerical validation and matrix repair routines for covariance matrices:
1. Matrix symmetry enforcement
2. Eigenvalue spectrum decomposition and inspection
3. Positive semi-definiteness (PSD) validation
4. Matrix condition number diagnostics
5. Higham (2002) nearest positive semi-definite matrix projection algorithm
6. Unified ensure_positive_semidefinite dispatcher
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd


def enforce_symmetry(
    matrix: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """
    Enforce exact mathematical symmetry on a square matrix.

    Formula:
        A_sym = (A + A^T) / 2

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Square matrix (N, N).

    Returns
    -------
    pd.DataFrame | np.ndarray
        Exactly symmetric matrix preserving index/columns if DataFrame.
    """
    is_df = isinstance(matrix, pd.DataFrame)
    if is_df:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        sym_mat = (matrix.values + matrix.values.T) / 2.0
        return pd.DataFrame(sym_mat, index=matrix.index, columns=matrix.columns)

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Matrix must be square 2D, got shape {arr.shape}")
    return (arr + arr.T) / 2.0


def get_eigenvalues(
    matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Compute sorted real eigenvalues of a symmetric or Hermitian matrix.

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Square matrix (N, N).

    Returns
    -------
    np.ndarray
        Sorted 1D array of eigenvalues in ascending order.
    """
    if isinstance(matrix, pd.DataFrame):
        arr = matrix.values
    else:
        arr = np.asarray(matrix, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Matrix must be square 2D, got shape {arr.shape}")

    arr_sym = (arr + arr.T) / 2.0
    return np.linalg.eigvalsh(arr_sym)


def is_positive_semidefinite(
    matrix: pd.DataFrame | np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """
    Check if a matrix is symmetric positive semi-definite within numerical tolerance.

    Condition:
        min(eigvalsh((A + A^T) / 2)) >= -tol

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Square matrix.
    tol : float, default 1e-8
        Tolerance for negative eigenvalues due to floating point noise.

    Returns
    -------
    bool
        True if all eigenvalues >= -tol, False otherwise.
    """
    eigenvalues = get_eigenvalues(matrix)
    return bool(np.min(eigenvalues) >= -tol)


def calculate_condition_number(
    matrix: pd.DataFrame | np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Calculate the 2-norm condition number of a symmetric positive semi-definite matrix.

    Formula:
        kappa = lambda_max / max(lambda_min, eps)

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Square matrix.
    eps : float, default 1e-15
        Small floor to prevent ZeroDivisionError.

    Returns
    -------
    float
        Matrix condition number kappa >= 1.0 (or np.inf if singular).
    """
    eigenvalues = get_eigenvalues(matrix)
    lambda_max = float(np.max(eigenvalues))
    lambda_min = float(np.min(eigenvalues))

    if lambda_min <= 0.0:
        return float(np.inf)
    return float(lambda_max / max(lambda_min, eps))


def nearest_psd_higham(
    matrix: pd.DataFrame | np.ndarray,
    eps: float = 1e-7,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> pd.DataFrame | np.ndarray:
    """
    Project a matrix to the nearest symmetric Positive Semi-Definite (PSD) matrix.

    Implements Nicholas Higham's (2002) alternating projection algorithm with
    Dykstra's correction term for spectral clipping onto the PSD cone S+(eps).

    Reference:
        Nicholas J. Higham (2002).
        "Computing the nearest correlation matrix—a problem from finance",
        IMA Journal of Numerical Analysis, 22(3), 329-343.

    Parameters
    ----------
    matrix : pd.DataFrame | np.ndarray
        Square matrix (N, N) to project.
    eps : float, default 1e-7
        Minimum eigenvalue floor for the projected matrix.
    max_iter : int, default 100
        Maximum number of alternating projection iterations.
    tol : float, default 1e-6
        Convergence tolerance on Frobenius norm relative change.

    Returns
    -------
    pd.DataFrame | np.ndarray
        Nearest positive semi-definite matrix with all eigenvalues >= eps.
    """
    is_df = isinstance(matrix, pd.DataFrame)
    if is_df:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        cols = matrix.columns
        idx = matrix.index
        a_mat = np.asarray(matrix.values, dtype=np.float64)
    else:
        a_mat = np.asarray(matrix, dtype=np.float64)
        if a_mat.ndim != 2 or a_mat.shape[0] != a_mat.shape[1]:
            raise ValueError(f"Matrix must be square 2D, got shape {a_mat.shape}")
        cols = None
        idx = None

    n = a_mat.shape[0]
    if n == 1:
        val = max(float(a_mat[0, 0]), eps)
        res = np.array([[val]], dtype=np.float64)
        if is_df:
            return pd.DataFrame(res, index=idx, columns=cols)
        return res

    # Dykstra's alternating projection algorithm
    y = (a_mat + a_mat.T) / 2.0
    ds = np.zeros_like(y)
    prev_y = y.copy()

    for i in range(max_iter):
        r = y - ds
        # Symmetrize
        r_sym = (r + r.T) / 2.0
        # Spectral decomposition
        vals, vecs = np.linalg.eigh(r_sym)
        # Project onto PSD cone: clip eigenvalues to >= eps
        vals_clipped = np.maximum(vals, eps)
        x = (vecs * vals_clipped) @ vecs.T
        ds = x - r
        y = (x + x.T) / 2.0

        if i > 0:
            norm_y = np.linalg.norm(y, ord="fro")
            rel_change = np.linalg.norm(y - prev_y, ord="fro") / max(norm_y, 1e-12)
            if rel_change < tol:
                break
        prev_y = y.copy()

    # Final symmetry pass
    y = (y + y.T) / 2.0

    if is_df:
        return pd.DataFrame(y, index=idx, columns=cols)
    return y


def ensure_positive_semidefinite(
    cov_matrix: pd.DataFrame | np.ndarray,
    eps: float = 1e-7,
    max_iter: int = 100,
) -> tuple[pd.DataFrame | np.ndarray, bool, float]:
    """
    Ensure covariance matrix is symmetric and strictly positive semi-definite.

    If the matrix has eigenvalues below eps, applies Higham's nearest PSD projection.
    Computes condition number diagnostics.

    Parameters
    ----------
    cov_matrix : pd.DataFrame | np.ndarray
        Input covariance matrix.
    eps : float, default 1e-7
        Minimum eigenvalue threshold.
    max_iter : int, default 100
        Maximum iterations for Higham projection.

    Returns
    -------
    tuple[pd.DataFrame | np.ndarray, bool, float]
        (psd_cov_matrix, was_repaired, condition_number)
    """
    # Enforce symmetry
    sym_cov = enforce_symmetry(cov_matrix)
    eigenvalues = get_eigenvalues(sym_cov)
    min_eig = float(np.min(eigenvalues))

    was_repaired = False
    if min_eig < eps:
        # Repair non-PSD / ill-conditioned matrix
        psd_cov = nearest_psd_higham(sym_cov, eps=eps, max_iter=max_iter)
        was_repaired = True
    else:
        psd_cov = sym_cov

    cond_num = calculate_condition_number(psd_cov)
    return psd_cov, was_repaired, cond_num
