# Forensic Integrity Audit Report

**Work Product**: Frontera Eficiente — Markowitz Quantitative Portfolio Optimization Platform (`src/`, `tests/`, `app.py`)  
**Profile**: General Project  
**Integrity Mode**: Development (`ORIGINAL_REQUEST.md § line 8`)  
**Verdict**: **CLEAN**  

---

## 1. Observation

Direct empirical evidence obtained during forensic static analysis and dynamic execution:

### 1.1 Full Test Suite Execution Trace
Command: `python -m pytest -v tests/`  
Result: **161 passed in 17.39s** (100% pass rate across Tier 1 Unit, Tier 2 Boundary/Corner, Tier 3 Integration, and Tier 4 Real-World suites).

Key test executions verified:
- `tests/tier1_unit/test_markowitz_engine.py`: GMV weight sum and bounds, GMV volatility minimality, Maximum Sharpe optimality, custom bounds, continuous efficient frontier sweep, CAL tangent line, Dirichlet simplex Monte Carlo performance (10,000 samples in <0.03s), and multi-asset GBM trajectory cones.
- `tests/tier1_unit/test_models_covariance.py` & `test_models_stability.py`: Unbiased sample covariance, Ledoit-Wolf Constant Correlation analytical shrinkage ($0 \le \delta^* \le 1$), Ledoit-Wolf Diagonal shrinkage, EWMA covariance, Higham (2002) nearest PSD projection, and condition number computation.
- `tests/tier1_unit/test_models_returns.py`: Arithmetic, CAGR geometric, EWMA decay ($\lambda=0.94$), and CAPM expected return regression ($\mu_i = R_f + \beta_i (\mu_M - R_f)$).
- `tests/tier1_unit/test_risk_analytics.py`: Sortino ratio downside semideviation, Calmar ratio, Underwater Drawdown series, Historical VaR 95%, CVaR 95% (Expected Shortfall), and Parametric Normal VaR/CVaR.
- `tests/tier1_unit/test_export_engine.py`: Styled multi-sheet openpyxl workbook creation with Navy headers (`#1F4E79`), zebra striping, percentage formatting, and formula totals.
- `tests/tier2_boundary_corner/`: Collinear asset stability ($\rho \ge 0.999$), extreme outliers / flash crashes (-80% drops, +300% spikes), bear markets with negative returns, single asset universes ($N=1$), and zero-weight boundary enforcement.
- `tests/tier3_integration/` & `tests/tier4_real_world/`: Complete end-to-end pipelines across all 5 canonical presets: Clásico 60/40, All-Weather Ray Dalio, Big Tech, CEDEARs Argentina (`.BA`), and Cripto + TradFi (`BTC-USD`, `ETH-USD`, `SPY`, `QQQ`).

### 1.2 Static AST & Forensic Pattern Scan Results
Automated AST parser inspected all 24 Python source files and 27 test files:
- **Empty stubs or facade return constants in `src/`**: 0 instances found.
- **Mocking frameworks (`unittest.mock`, `MagicMock`) in `src/`**: 0 instances found (only used in test fixtures for simulating `yfinance` network calls in offline testing).
- **Trivial / Tautological assertions in `tests/` (`assert True`, `assert x == x`)**: 0 instances found.
- **Pre-populated stray artifacts (`.log`, `.out`, `.result`) predating tests**: 0 instances found.

### 1.3 Empirical Subsystem Verification Metrics
Independent script executed all mathematical routines with direct validation:
1. **SLSQP GMV Optimization**: Solved with weight sum $= 1.0000000000$, $\sigma_{gmv} = 0.121571$, bounds strictly inside $[0, 1]$. Volatility verified smaller than 100 random allocations.
2. **SLSQP Max Sharpe Optimization**: Solved with weight sum $= 1.0000000000$, $SR_{ms} = 0.767457 \ge SR_{gmv} = 0.6865$.
3. **Higham (2002) PSD Projection**: Negative eigenvalue matrix ($\lambda_{min} = -0.480523$) repaired to strictly positive semi-definite cone with $\lambda_{min} = 1.0000 \times 10^{-6}$, condition number $\kappa = 2.58 \times 10^6$.
4. **Dirichlet Simplex Sampling**: 10,000 portfolios sampled via standard exponential distribution normalization on $\Delta^{k-1}$; max sampled Sharpe ratio ($0.6957$) strictly bounded below analytical Max Sharpe ($0.7675$).
5. **Stochastic Trajectory Cones**: Multi-asset GBM with Cholesky factor $L L^T = \Sigma_{psd}$ simulated 2,000 paths; quantile hierarchy $P_5 \le P_{25} \le P_{50} \le P_{75} \le P_{95}$ strictly maintained over all time steps.
6. **Risk Metrics**: Coherent risk inequality $CVaR_{95} (0.014050) \ge VaR_{95} (0.011822)$ verified.
7. **Excel Workbook Generation**: Successfully constructed valid binary stream containing all required sheets (`Resumen de Métricas`, `Ponderaciones`, `Matriz de Correlación`, `Matriz de Covarianza`).

---

## 2. Logic Chain

1. **Ground-Truth Requirements & Constraints**: `ORIGINAL_REQUEST.md` specifies `Integrity mode: development`. Under Development Mode, standard scientific libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `openpyxl`, `plotly`, `streamlit`) are fully permitted. Prohibited patterns include hardcoded test results, facade implementations, and fake mocks pretending to be real algorithms.
2. **Implementation Verification**:
   - In `src/optimization/optimizer.py`, `optimize_global_minimum_variance` and `optimize_maximum_sharpe` formulate genuine non-linear constrained optimization problems solved via `scipy.optimize.minimize(method='SLSQP')` equipped with exact analytical Jacobians and a 4-stage fallback cascade (SLSQP analytical $\rightarrow$ SLSQP numerical $\rightarrow$ trust-constr $\rightarrow$ regularized).
   - In `src/models/covariance.py`, `ledoit_wolf_constant_correlation` calculates the full Ledoit & Wolf (2004) analytical formula for asymptotic covariance, asymptotic variance $\hat{\pi}$, target covariance $\hat{\rho}$, Frobenius distance $\hat{\gamma}$, and shrinkage intensity $\delta^* \in [0, 1]$.
   - In `src/models/stability.py`, `nearest_psd_higham` implements Higham's (2002) alternating projection with Dykstra's correction term to guarantee numerical semi-definiteness.
   - In `src/simulation/weight_monte_carlo.py`, uniform simplex sampling is achieved through normalized independent exponential variates $E_i / \sum E_j \sim \text{Dirichlet}(1, \dots, 1)$.
   - In `src/simulation/trajectory_monte_carlo.py`, asset correlation is preserved via Cholesky decomposition $L = \text{chol}(\Sigma)$ with daily stochastic step $S_{t+1} = S_t \exp\left((\mu - \frac{1}{2}\sigma^2)dt + \sqrt{dt} L Z\right)$.
3. **Absence of Malfeasance**:
   - Static AST parsing confirms that no function returns hardcoded constant values to fake optimization or risk metrics.
   - All tests assert computed quantities against mathematical principles rather than static dummy constants.
   - All 5 predefined presets and arbitrary asset universes execute through the full pipeline without exceptions or bypasses.
4. **Conclusion Support**: All evidence converges to show genuine, robust, and authentic mathematical and engineering implementation.

---

## 3. Caveats

- **Network-dependent Yahoo Finance tests**: Tests in the automated suite mock `yfinance.download` with synthetic GBM market paths to ensure offline, deterministic, and hermetic execution. This is standard testing practice and does not compromise implementation authenticity, as the runtime data loader `src/data/loader.py` actively integrates real `yfinance.download` calls when running with internet connectivity.
- **Ruff linter unused import notices**: 111 minor style warnings (primarily `F401` unused imports in test suites and conditional fallback blocks) exist. These do not affect runtime execution, mathematical accuracy, or integrity.

---

## 4. Conclusion

The codebase and test suite of **Frontera Eficiente** are fully authentic, numerically rigorous, and completely free of hardcoded shortcuts, facade implementations, test tampering, or dummy stubs.

**Audit Verdict: CLEAN**

---

## 5. Verification Method

To independently reproduce and verify this audit:

1. **Execute Pytest Suite**:
   ```bash
   python -m pytest -v tests/
   ```
   *Expected result*: 161 tests passing with 0 failures and 0 errors.

2. **Execute Mathematical Engine Self-Check**:
   ```bash
   python -c "
   import numpy as np, pandas as pd
   from src.models.returns import calculate_expected_returns
   from src.models.covariance import estimate_covariance_matrix
   from src.optimization.optimizer import optimize_maximum_sharpe, optimize_global_minimum_variance
   from src.simulation.weight_monte_carlo import run_weight_space_monte_carlo
   from src.simulation.trajectory_monte_carlo import run_trajectory_monte_carlo

   rng = np.random.default_rng(42)
   cov = rng.normal(size=(5,5)); cov = cov.T @ cov + 0.05 * np.eye(5)
   mu = np.array([0.12, 0.18, 0.08, 0.14, 0.10])
   gmv = optimize_global_minimum_variance(cov, mu)
   ms = optimize_maximum_sharpe(mu, cov)
   assert abs(np.sum(gmv.weights) - 1.0) < 1e-9
   assert abs(np.sum(ms.weights) - 1.0) < 1e-9
   assert ms.sharpe_ratio >= gmv.sharpe_ratio
   print('Self-check verified!')
   "
   ```
   *Expected result*: Prints `Self-check verified!`.
