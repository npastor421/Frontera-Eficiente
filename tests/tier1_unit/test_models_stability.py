"""
Tier 1 Unit Tests: Matrix Stability & Higham PSD Repair (src/models/stability.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.stability import (
    calculate_condition_number,
    enforce_symmetry,
    ensure_positive_semidefinite,
    get_eigenvalues,
    is_positive_semidefinite,
    nearest_psd_higham,
)


def test_t1_enforce_symmetry_random():
    np.random.seed(42)
    mat = np.random.randn(5, 5)
    sym = enforce_symmetry(mat)
    assert np.allclose(sym, sym.T)


def test_t1_eigenvalues_order():
    diag_mat = np.diag([3.0, 1.0, 2.0])
    eigs = get_eigenvalues(diag_mat)
    assert np.allclose(eigs, [1.0, 2.0, 3.0])


def test_t1_is_positive_semidefinite_behavior():
    psd_mat = np.diag([2.0, 1.0, 0.0])
    assert is_positive_semidefinite(psd_mat, tol=1e-8)

    neg_mat = np.diag([2.0, 1.0, -0.05])
    assert not is_positive_semidefinite(neg_mat, tol=1e-8)


def test_t1_condition_number_values():
    mat = np.diag([100.0, 10.0, 1.0])
    cond = calculate_condition_number(mat)
    assert np.isclose(cond, 100.0)


def test_t1_higham_projection_numerical_guarantees():
    non_psd = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.9],
        [0.9, 0.9, 0.2]
    ])
    repaired = nearest_psd_higham(non_psd, eps=1e-7)
    eigs = get_eigenvalues(repaired)
    assert np.min(eigs) >= 1e-7 - 1e-12
    assert np.allclose(repaired, repaired.T)


def test_t1_ensure_psd_full_contract():
    asym_non_psd = np.array([
        [1.0, 0.92, 0.88],
        [0.90, 1.0, 0.85],
        [0.86, 0.85, 0.1]
    ])
    psd, repaired, cond = ensure_positive_semidefinite(asym_non_psd, eps=1e-7)
    assert repaired is True
    assert np.allclose(psd, psd.T)
    assert is_positive_semidefinite(psd)
    assert cond > 0
