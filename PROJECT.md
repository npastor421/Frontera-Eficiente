# Project: Frontera Eficiente — Markowitz Quantitative Portfolio Optimization Platform

## Architecture
The platform is organized into modular, strictly decoupled layers in Python with clean mathematical separation, vectorized numerical computing (NumPy/SciPy/Pandas/Scikit-Learn), interactive data visualization (Plotly), and dynamic reactive web interface (Streamlit):

```
Frontera Eficiente/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # yfinance downloader & manual CSV/Excel file parser
│   │   ├── cleaner.py            # Calendar harmonization, business day alignment, NaN treatment
│   │   └── cache.py              # Caching wrappers and cache invalidation mechanics
│   ├── models/
│   │   ├── __init__.py
│   │   ├── returns.py            # Arithmetic, CAGR/geometric, EWMA, and CAPM expected return estimators
│   │   ├── covariance.py         # Sample covariance, Ledoit-Wolf Shrinkage (Constant Corr & Diagonal), EWMA
│   │   └── stability.py          # Symmetry enforcement, PSD check, Higham (2002) nearest PSD projection
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── optimizer.py          # SLSQP Max Sharpe, GMV, custom asset bounds, budget constraint enforcement
│   │   └── frontier.py           # Continuous Efficient Frontier curve sweep & Capital Allocation Line (CAL)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── weight_monte_carlo.py # Vectorized Dirichlet uniform simplex weight simulation
│   │   └── trajectory_monte_carlo.py # Multi-asset GBM (Cholesky) & Block Bootstrap with 5%-95% probability cones
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── risk_metrics.py       # Annualized Return, Volatility, Sharpe, Sortino, Calmar, MaxDD, VaR 95%, CVaR 95%
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py              # Interactive Plotly charts (Frontier+CAL+MC, Allocation, Heatmaps, Wealth, Cones)
│   ├── presets/
│   │   ├── __init__.py
│   │   └── portfolio_presets.py  # 60/40, All-Weather, Big Tech, CEDEARs.BA, Cripto+TradFi definitions
│   └── export/
│       ├── __init__.py
│       └── exporter.py           # CSV and multi-sheet styled Excel workbook generation (openpyxl)
├── app.py                        # Streamlit web application entrypoint, sidebar, state sync, and tabs
├── tests/
│   ├── conftest.py               # Shared synthetic data fixtures, seeds, market mock data
│   ├── tier1_unit/               # Module-level unit tests
│   ├── tier2_boundary_corner/    # Boundary value and edge case tests
│   ├── tier3_integration/        # Cross-component integration pipeline tests
│   ├── tier4_real_world/         # Real-world preset and market scenario tests
│   └── test_stress_challenge_harness.py # Adversarial stress testing harness
├── PROJECT.md                    # Project blueprint and milestone tracking
├── TEST_INFRA.md                 # E2E test infrastructure specification
└── TEST_READY.md                 # E2E test completion signal
```

---

## Feature Inventory

Every feature from the Survey phase is fully implemented and tested:

| # | Feature | Description | Milestone | Status | Source |
|---|---------|-------------|-----------|--------|--------|
| 1 | `yfinance` Ingestion | Multi-asset historical prices download (US stocks, ETFs, cryptos `BTC-USD`, CEDEARs `.BA`, benchmarks `SPY`/`^GSPC`) | M1 | DONE | ORIGINAL_REQUEST § R1 |
| 2 | Manual File Parser | CSV/Excel upload supporting Wide prices/returns, Long format, and European/Latin comma decimals | M1 | DONE | ORIGINAL_REQUEST § R1 |
| 3 | Calendar Harmonization | Asynchronous trading calendar alignment (NYSE 252d, BYMA 248d, Crypto 365d) via Business Day reindexing and ffill | M1 | DONE | ORIGINAL_REQUEST § R1 |
| 4 | Streamlit Data Cache | `@st.cache_data` caching with TTL, defensive copies, and explicit UI invalidation | M1 | DONE | ORIGINAL_REQUEST § R1 |
| 5 | Return Estimators | Arithmetic mean, Compound/Geometric CAGR, EWMA returns ($\lambda=0.94$), and CAPM expected returns | M2 | DONE | ORIGINAL_REQUEST § R2 |
| 6 | Covariance Estimators | Classical sample covariance, Ledoit-Wolf Shrinkage (Constant Correlation & Diagonal), and EWMA covariance | M2 | DONE | ORIGINAL_REQUEST § R2 |
| 7 | Matrix PSD Stability | Symmetry enforcement, eigenvalue inspection, Higham (2002) nearest PSD projection, condition number diagnostics | M2 | DONE | ORIGINAL_REQUEST § R2 |
| 8 | Correlation Visual Matrix | Interactive Correlation & Covariance data structures for Plotly heatmaps with custom hover tooltips | M2 | DONE | ORIGINAL_REQUEST § R2 |
| 9 | Maximum Sharpe Portfolio | Tangency portfolio optimization with exact analytical Jacobian and user-editable risk-free rate $R_f$ | M3 | DONE | ORIGINAL_REQUEST § R3 |
| 10 | Global Minimum Variance | GMV portfolio optimization with analytical gradient and constraint enforcement | M3 | DONE | ORIGINAL_REQUEST § R3 |
| 11 | Efficient Frontier Sweep | Continuous 100-point upper Pareto frontier curve calculation with warm-start chained optimization | M3 | DONE | ORIGINAL_REQUEST § R3 |
| 12 | Capital Allocation Line | Tangent CAL line calculation from $(0, R_f)$ through Maximum Sharpe point $(\sigma_{ms}, \mu_{ms})$ | M3 | DONE | ORIGINAL_REQUEST § R3 |
| 13 | Constraints Engine | Long-only ($0 \le w_i \le 1$), Short-selling bounds, custom asset bounds ($w_{min} \le w_i \le w_{max}$), budget verification | M3 | DONE | ORIGINAL_REQUEST § R3 |
| 14 | Dirichlet Weight Monte Carlo | Vectorized Dirichlet uniform simplex weight sampling (5,000–20,000 portfolios in <50ms) colored by Sharpe | M3 | DONE | ORIGINAL_REQUEST § R4 |
| 15 | Stochastic Trajectory MC | 1–5 year wealth forecasting via Cholesky Multi-Asset GBM & Historical Block Bootstrapping with 5%, 25%, 50%, 75%, 95% cones | M3 | DONE | ORIGINAL_REQUEST § R4 |
| 16 | Interactive Weight Sliders | Sliders dynamically bound to session state with instant "Normalizar a 100%" and "Aplicar Cartera Óptima Sharpe" | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 17 | 1-Click Predefined Presets | 5 canonical portfolios: 60/40, All-Weather Dalio, Big Tech, CEDEARs.BA, Cripto+TradFi | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 18 | Interactive Plotly Plots | Frontier+CAL+MC Cloud, Asset Donut/Bar, Heatmaps, $10k Historical Wealth Backtest + Drawdown, Monte Carlo Projection Cones | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 19 | Comprehensive Risk Metrics | Annualized Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, VaR 95% (Hist/Param), CVaR 95% | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 20 | Multi-Format Export | Formatted CSVs and styled multi-sheet Excel workbook (`.xlsx`) via openpyxl | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 21 | Full Streamlit Application | Complete interactive web dashboard integrating sidebar, presets, sliders, tabs, and real-time computation | M4 | DONE | ORIGINAL_REQUEST § R5 |
| 22 | Comprehensive Pytest Suite | 4-tier opaque-box & unit test suite validating 100% of mathematical invariants and end-to-end workflows (177 tests) | E2E Track | DONE | ORIGINAL_REQUEST § Criteria |
| 23 | E2E Pass & Adversarial Hardening | Verification of 100% E2E test pass + Tier 5 adversarial stress testing + Forensic Audit CLEAN | M5 | DONE | Project Pattern |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| Survey | System Survey & Requirements Mining | Map full scope, specifications, formulas, and test plan | none | DONE |
| M1 | Data Ingestion & Cache Engine | `src/data/` (loader, cleaner, cache) | Survey | DONE |
| M2 | Risk Modeling & Robust Covariance | `src/models/` (returns, covariance, stability/Higham) | M1 | DONE |
| M3 | Markowitz Optimization & Dual MC | `src/optimization/`, `src/simulation/` | M2 | DONE |
| M4 | Streamlit UI, Visualizers, Metrics & Export | `src/analytics/`, `src/visualization/`, `src/presets/`, `src/export/`, `app.py` | M3 | DONE |
| E2E | E2E Testing Track Orchestrator | `tests/` (Tier 1-4 suites, `TEST_INFRA.md`, `TEST_READY.md`) | Survey | DONE |
| M5 | Final E2E Hardening & Adversarial Audit | Pass 100% E2E tests + Tier 5 adversarial stress testing (GATE PASS) | M4, E2E | DONE |

---

## Code Layout
- Standard Python package layout under `src/`.
- All automated tests located under `tests/` (177 tests passing).
- App entrypoint: `app.py`.
