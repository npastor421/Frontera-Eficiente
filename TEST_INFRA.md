# E2E Test Infra: Frontera Eficiente

## Test Philosophy
- Opaque-box, requirement-driven testing. No dependency on implementation internals.
- Systematic 4-tier methodology: Category-Partition, Boundary Value Analysis (BVA), Pairwise Combinatorial Testing, Real-World Workload Testing.
- 100% pass requirement across all analytical calculations, optimizers, and workflow pipelines.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 (Unit) | Tier 2 (Boundary) | Tier 3 (Integration) | Tier 4 (Workload) |
|---|---------|----------------------|:-------------:|:-----------------:|:--------------------:|:-----------------:|
| 1 | `yfinance` Data Ingestion | ORIGINAL_REQUEST § R1 | ≥5 | ≥5 | ✓ | ✓ |
| 2 | Manual CSV/Excel Upload | ORIGINAL_REQUEST § R1 | ≥5 | ≥5 | ✓ | ✓ |
| 3 | Calendar & Missing Data Cleaning | ORIGINAL_REQUEST § R1 | ≥5 | ≥5 | ✓ | ✓ |
| 4 | Data Caching Engine | ORIGINAL_REQUEST § R1 | ≥5 | ≥5 | ✓ | ✓ |
| 5 | Return Estimators (Arith, Geom, EWMA, CAPM) | ORIGINAL_REQUEST § R2 | ≥5 | ≥5 | ✓ | ✓ |
| 6 | Covariance Estimators (Sample, Ledoit-Wolf, EWMA) | ORIGINAL_REQUEST § R2 | ≥5 | ≥5 | ✓ | ✓ |
| 7 | Symmetry & Higham PSD Repair | ORIGINAL_REQUEST § R2 | ≥5 | ≥5 | ✓ | ✓ |
| 8 | Correlation Matrix Presentation | ORIGINAL_REQUEST § R2 | ≥5 | ≥5 | ✓ | ✓ |
| 9 | Maximum Sharpe Optimization | ORIGINAL_REQUEST § R3 | ≥5 | ≥5 | ✓ | ✓ |
| 10 | Global Minimum Variance (GMV) | ORIGINAL_REQUEST § R3 | ≥5 | ≥5 | ✓ | ✓ |
| 11 | Continuous Efficient Frontier Sweep | ORIGINAL_REQUEST § R3 | ≥5 | ≥5 | ✓ | ✓ |
| 12 | Capital Allocation Line (CAL) | ORIGINAL_REQUEST § R3 | ≥5 | ≥5 | ✓ | ✓ |
| 13 | Constraints & Bounds Engine | ORIGINAL_REQUEST § R3 | ≥5 | ≥5 | ✓ | ✓ |
| 14 | Dirichlet Simplex Monte Carlo | ORIGINAL_REQUEST § R4 | ≥5 | ≥5 | ✓ | ✓ |
| 15 | Multi-Year Trajectory Monte Carlo | ORIGINAL_REQUEST § R4 | ≥5 | ≥5 | ✓ | ✓ |
| 16 | Interactive Sliders & Normalization | ORIGINAL_REQUEST § R5 | ≥5 | ≥5 | ✓ | ✓ |
| 17 | 1-Click Preset Portfolios | ORIGINAL_REQUEST § R5 | ≥5 | ≥5 | ✓ | ✓ |
| 18 | Plotly Visualizers Generation | ORIGINAL_REQUEST § R5 | ≥5 | ≥5 | ✓ | ✓ |
| 19 | Comprehensive Risk Metrics Engine | ORIGINAL_REQUEST § R5 | ≥5 | ≥5 | ✓ | ✓ |
| 20 | Multi-Sheet Excel & CSV Exporter | ORIGINAL_REQUEST § R5 | ≥5 | ≥5 | ✓ | ✓ |

## Test Architecture
- Test Runner: `pytest`
- Execution: `pytest -v tests/`
- Directory Structure:
  - `tests/conftest.py`: Synthetic market data generators, deterministic seed fixtures, mock yfinance loaders.
  - `tests/tier1_unit/`: Unit tests for data, models, optimization, simulation, analytics, visualization, presets, export.
  - `tests/tier2_boundary_corner/`: Edge cases (single asset, collinearity, zero weights, flash crashes, missing holidays, negative excess returns).
  - `tests/tier3_integration/`: End-to-end component dataflow (Data -> Covariance -> Optimizer -> Metrics -> Export -> Plotly).
  - `tests/tier4_real_world/`: Realistic workflows for the 5 canonical presets (60/40, All-Weather, Big Tech, CEDEARs.BA, Cripto+TradFi).

## Invariants Under Test
1. $\left|\sum_{i=1}^N w_i - 1.0\right| \le 10^{-5}$ for all optimized/rebalanced allocations.
2. $\sigma(\mathbf{w}_{GMV}) \le \sigma(\mathbf{w})$ for all feasible $\mathbf{w}$.
3. $SR(\mathbf{w}_{MaxSharpe}) \ge SR(\mathbf{w})$ for all random/asset $\mathbf{w}$.
4. $\mathbf{\Sigma} = \mathbf{\Sigma}^T$ and $\lambda_{\min}(\mathbf{\Sigma}) \ge -10^{-8}$.
5. $\text{CVaR}_{95\%} \ge \text{VaR}_{95\%}$ for historical distributions.
6. Execution time of 10,000 Dirichlet Monte Carlo portfolios $< 2.0$ seconds.
