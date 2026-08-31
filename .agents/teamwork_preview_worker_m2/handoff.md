# Milestone 2 Handoff Report: Risk Modeling, Robust Covariance & PSD Stability

**Author**: `teamwork_preview_worker_m2` (Risk Modeling & Covariance Worker)  
**Working Directory**: `c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m2/`  
**Date**: 2026-08-31  
**Target Milestone**: Milestone 2 (`src/models/`)  

---

## 1. Observation

1. **Source Code Implementation**:
   - `src/models/returns.py` (290 lines):
     - `ReturnMethod` Enum with robust `.from_str()` parser for aliases.
     - `annualized_arithmetic_returns(returns, ann_factor=252)`: Unbiased historical arithmetic mean.
     - `annualized_geometric_returns(returns, ann_factor=252)`: Exact CAGR $\exp\left(\frac{N_{\text{ann}}}{T}\sum \ln(1+R_t)\right) - 1$ with lower bound clipping.
     - `ewma_returns(returns, decay=0.94, ann_factor=252)`: RiskMetrics exponential decay $\tilde{w}_t = \frac{(1-\lambda)\lambda^{T-t}}{1-\lambda^T}$.
     - `calculate_capm_betas(returns, benchmark_returns)`: Vectorized OLS beta $\beta_i = \text{Cov}(R_i, R_M)/\text{Var}(R_M)$.
     - `capm_expected_returns(returns, benchmark_returns, rf=0.04, market_return=None, ann_factor=252)`: Returns $\mu_i = R_f + \beta_i(\mu_M - R_f)$.
     - `calculate_expected_returns(...)`: Unified contract dispatcher.
   - `src/models/covariance.py` (336 lines):
     - `CovarianceMethod` Enum with `.from_str()` parser.
     - `sample_covariance(returns, ann_factor=252, ddof=1)`: Unbiased sample covariance matrix.
     - `ledoit_wolf_constant_correlation(returns, ann_factor=252)`: Closed-form Ledoit & Wolf (2004) Constant Correlation target $F$ with optimal analytical shrinkage intensity $\delta^* \in [0, 1]$ using 3D vectorized tensors.
     - `ledoit_wolf_diagonal(returns, ann_factor=252)`: Scikit-learn `LedoitWolf` wrapper with annualization scaling.
     - `ewma_covariance(returns, decay=0.94, ann_factor=252)`: Gram-matrix vectorized EWMA covariance.
     - `covariance_to_correlation(cov_matrix)`: Normalizes $\Sigma$ to correlation matrix $C_{ij} \in [-1, 1]$ with $C_{ii}=1.0$.
     - `estimate_covariance_matrix(...)`: Unified contract dispatcher returning `(cov_df, metadata_dict)`.
   - `src/models/stability.py` (215 lines):
     - `enforce_symmetry(matrix)`: Exact numerical symmetry $\frac{A + A^T}{2}$.
     - `get_eigenvalues(matrix)`: Sorted real eigenvalues via `np.linalg.eigvalsh`.
     - `is_positive_semidefinite(matrix, tol=1e-8)`: Validates $\lambda_{\min} \ge -10^{-8}$.
     - `calculate_condition_number(matrix, eps=1e-15)`: 2-norm condition number $\kappa = \lambda_{\max}/\max(\lambda_{\min}, \text{eps})$.
     - `nearest_psd_higham(matrix, eps=1e-7, max_iter=100, tol=1e-6)`: Nicholas Higham (2002) alternating projection algorithm with Dykstra's correction.
     - `ensure_positive_semidefinite(...)`: Unified contract dispatcher returning `(psd_cov_df, was_repaired, condition_number)`.
   - `src/models/__init__.py` (143 lines):
     - Exports all model functions, enums, dataclasses (`RiskModelConfig`, `RiskModelOutput`), and provides `build_risk_model(returns, config, benchmark_returns)`.

2. **Test Execution Evidence**:
   Command:
   ```bash
   python -m pytest tests/test_returns.py tests/test_covariance.py tests/test_stability.py tests/tier1_unit/test_covariance_models.py tests/tier1_unit/test_models_returns.py tests/tier1_unit/test_models_covariance.py tests/tier1_unit/test_models_stability.py tests/tier2_boundary_corner/test_models_boundary.py -v
   ```
   Verbatim output:
   ```
   ============================= test session starts =============================
   platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
   rootdir: C:\Nico\Antigravity\Frontera Eficiente
   configfile: pytest.ini
   plugins: anyio-4.13.0, langsmith-0.8.7
   collected 60 items

   tests/test_returns.py::test_annualized_arithmetic_returns PASSED         [  1%]
   tests/test_returns.py::test_annualized_geometric_returns PASSED          [  3%]
   tests/test_returns.py::test_geometric_returns_compound_property PASSED   [  5%]
   tests/test_returns.py::test_ewma_returns_weighting PASSED                [  6%]
   tests/test_returns.py::test_capm_betas_and_returns PASSED                [  8%]
   tests/test_returns.py::test_calculate_expected_returns_dispatcher PASSED [ 10%]
   tests/test_returns.py::test_returns_error_handling PASSED                [ 11%]
   tests/test_covariance.py::test_sample_covariance_properties PASSED       [ 13%]
   tests/test_covariance.py::test_ledoit_wolf_constant_correlation PASSED   [ 15%]
   tests/test_covariance.py::test_ledoit_wolf_small_sample_t_less_than_n PASSED [ 16%]
   tests/test_covariance.py::test_ledoit_wolf_diagonal PASSED               [ 18%]
   tests/test_covariance.py::test_ewma_covariance PASSED                    [ 20%]
   tests/test_covariance.py::test_covariance_to_correlation PASSED          [ 21%]
   tests/test_covariance.py::test_estimate_covariance_matrix_dispatcher PASSED [ 23%]
   tests/test_covariance.py::test_single_asset_covariance PASSED            [ 25%]
   tests/test_stability.py::test_enforce_symmetry PASSED                    [ 26%]
   tests/test_stability.py::test_eigenvalue_and_psd_check PASSED            [ 28%]
   tests/test_stability.py::test_condition_number PASSED                    [ 30%]
   tests/test_stability.py::test_higham_nearest_psd_repair PASSED           [ 31%]
   tests/test_stability.py::test_ensure_positive_semidefinite PASSED        [ 33%]
   tests/test_stability.py::test_build_risk_model_pipeline PASSED           [ 35%]
   tests/tier1_unit/test_covariance_models.py::test_annualized_arithmetic_returns PASSED [ 36%]
   tests/tier1_unit/test_covariance_models.py::test_annualized_geometric_returns_cagr PASSED [ 38%]
   tests/tier1_unit/test_covariance_models.py::test_ewma_returns_decay PASSED [ 40%]
   tests/tier1_unit/test_covariance_models.py::test_capm_expected_returns_and_betas PASSED [ 41%]
   tests/tier1_unit/test_covariance_models.py::test_calculate_expected_returns_dispatcher PASSED [ 43%]
   tests/tier1_unit/test_covariance_models.py::test_sample_covariance_symmetry_and_psd PASSED [ 45%]
   tests/tier1_unit/test_covariance_models.py::test_ledoit_wolf_constant_correlation PASSED [ 46%]
   tests/tier1_unit/test_covariance_models.py::test_ledoit_wolf_diagonal PASSED [ 48%]
   tests/tier1_unit/test_covariance_models.py::test_ewma_covariance PASSED  [ 50%]
   tests/tier1_unit/test_covariance_models.py::test_covariance_to_correlation PASSED [ 51%]
   tests/tier1_unit/test_covariance_models.py::test_estimate_covariance_matrix_dispatcher PASSED [ 53%]
   tests/tier1_unit/test_covariance_models.py::test_enforce_symmetry PASSED [ 55%]
   tests/tier1_unit/test_covariance_models.py::test_get_eigenvalues_and_psd_check PASSED [ 56%]
   tests/tier1_unit/test_covariance_models.py::test_higham_nearest_psd_repair PASSED [ 58%]
   tests/tier1_unit/test_covariance_models.py::test_ensure_positive_semidefinite_workflow PASSED [ 60%]
   tests/tier1_unit/test_models_returns.py::test_t1_arithmetic_returns_exact_mean PASSED [ 61%]
   tests/tier1_unit/test_models_returns.py::test_t1_geometric_returns_compound PASSED [ 63%]
   tests/tier1_unit/test_models_returns.py::test_t1_ewma_decay_behavior PASSED [ 65%]
   tests/tier1_unit/test_models_returns.py::test_t1_capm_beta_identity PASSED [ 66%]
   tests/tier1_unit/test_models_returns.py::test_t1_capm_expected_returns_formula PASSED [ 68%]
   tests/tier1_unit/test_models_returns.py::test_t1_calculate_expected_returns_enum_and_string PASSED [ 70%]
   tests/tier1_unit/test_models_covariance.py::test_t1_sample_covariance_symmetry_and_psd PASSED [ 71%]
   tests/tier1_unit/test_models_covariance.py::test_t1_ledoit_wolf_cc_analytical PASSED [ 73%]
   tests/tier1_unit/test_models_covariance.py::test_t1_ledoit_wolf_diag_sklearn PASSED [ 75%]
   tests/tier1_unit/test_models_covariance.py::test_t1_ewma_covariance_properties PASSED [ 76%]
   tests/tier1_unit/test_models_covariance.py::test_t1_covariance_to_correlation_properties PASSED [ 78%]
   tests/tier1_unit/test_models_covariance.py::test_t1_estimate_covariance_matrix_metadata PASSED [ 80%]
   tests/tier1_unit/test_models_stability.py::test_t1_enforce_symmetry_random PASSED [ 81%]
   tests/tier1_unit/test_models_stability.py::test_t1_eigenvalues_order PASSED [ 83%]
   tests/tier1_unit/test_models_stability.py::test_t1_is_positive_semidefinite_behavior PASSED [ 85%]
   tests/tier1_unit/test_models_stability.py::test_t1_condition_number_values PASSED [ 86%]
   tests/tier1_unit/test_models_stability.py::test_t1_higham_projection_numerical_guarantees PASSED [ 88%]
   tests/tier1_unit/test_models_stability.py::test_t1_ensure_psd_full_contract PASSED [ 90%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_single_asset PASSED [ 91%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_high_collinearity_assets PASSED [ 93%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_severely_underdetermined_t_much_less_than_n PASSED [ 95%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_extreme_market_crash_returns PASSED [ 96%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_negative_definite_matrix_higham_repair PASSED [ 98%]
   tests/tier2_boundary_corner/test_models_boundary.py::test_bva_annualization_factors PASSED [100%]

   ============================= 60 passed in 3.11s ==============================
   ```

---

## 2. Logic Chain

1. **Analytical Fidelity**:
   - `ledoit_wolf_constant_correlation` calculates the asymptotic variance of sample covariance ($\hat{\pi}$), asymptotic covariance with target ($\hat{\rho}$), and Frobenius norm deviation ($\hat{\gamma}$) per Ledoit & Wolf (2004). Shrinkage parameter $\delta^* = \max(0, \min(1, \frac{\hat{\pi}-\hat{\rho}}{\hat{\gamma} T}))$ is guaranteed in $[0, 1]$.
   - `ewma_covariance` forms a Gram matrix $W^{1/2} \tilde{Y}$ which guarantees non-negative eigenvalues by construction.
2. **Mathematical Invariants Guarantee**:
   - All returned covariance matrices are symmetrized via $(A + A^T) / 2.0$.
   - Any non-PSD or ill-conditioned matrix is automatically restored to positive semi-definiteness with minimum eigenvalue $\ge \epsilon$ ($10^{-7}$) via Higham's alternating projection algorithm with Dykstra's correction.
   - All expected returns functions validate inputs (no NaNs, positive annualization factors, $0 < \lambda < 1$).
3. **Interface Contract Compliance**:
   - `calculate_expected_returns(returns, method, rf, benchmark_returns, ann_factor)` matches `PROJECT.md § Interface Contracts`.
   - `estimate_covariance_matrix(returns, method, ann_factor, decay)` matches `PROJECT.md § Interface Contracts`.
   - `ensure_positive_semidefinite(cov_matrix, eps)` matches `PROJECT.md § Interface Contracts`.

---

## 3. Caveats

1. **CAPM Benchmark Inception**: If `benchmark_returns` is not provided to CAPM, the system defaults to the equal-weighted portfolio mean return of the provided assets. In production UI, user-selected benchmark ticker (e.g. `SPY`) should be passed.
2. **Sample Size for Single Asset**: When $N=1$, Constant Correlation shrinkage target $F = S$ and $\delta^* = 0$, correctly returning the sample variance.

---

## 4. Conclusion

Milestone 2 (`src/models/`) is 100% complete, fully tested, and mathematically verified. All interface contracts for Milestone 3 (Markowitz Optimization & Dual Monte Carlo) and Milestone 4 (Streamlit UI, Visualizers, Metrics) are satisfied.

---

## 5. Verification Method

To verify the implementation:

```bash
python -m pytest tests/test_returns.py tests/test_covariance.py tests/test_stability.py tests/tier1_unit/test_covariance_models.py tests/tier1_unit/test_models_returns.py tests/tier1_unit/test_models_covariance.py tests/tier1_unit/test_models_stability.py tests/tier2_boundary_corner/test_models_boundary.py -v
```
All 60 tests must pass with 0 failures.
