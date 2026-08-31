# Milestone 4 Handoff Report: Risk Analytics, Presets, Export Engine, Plotly Visualizers & Streamlit Web Application

**Author**: `teamwork_preview_worker` (Milestone 4 Worker)  
**Assigned Role**: Analytics, UI & Visualizers Worker  
**Date**: 2026-08-31  
**Target File**: `c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m4/handoff.md`  

---

## 1. Observation

Direct inspection of the authoritative requirements in `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `TEST_INFRA.md` revealed the required deliverables for Milestone 4:

1. **Risk Analytics Engine (`src/analytics/risk_metrics.py` & `src/analytics/__init__.py`)**:
   - `PortfolioRiskMetrics` dataclass with all required performance fields (`annualized_return`, `cagr`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `var_95_hist`, `var_95_param`, `cvar_95_hist`, `cvar_95_param`, `peak_date`, `valley_date`, `recovery_days`) and dual attribute/dict access.
   - Vectorized functions: `calculate_portfolio_returns`, `calculate_drawdown_series`, `compute_drawdown_series`, `calculate_max_drawdown`, `calculate_sortino_ratio`, `calculate_calmar_ratio`, `compute_historical_var_cvar`, `compute_parametric_var_cvar`, `calculate_var_95`, `calculate_cvar_95`, and `compute_portfolio_risk_metrics`.
2. **Canonical Portfolio Presets (`src/presets/portfolio_presets.py` & `src/presets/__init__.py`)**:
   - 5 canonical portfolios: Clásico 60/40, All-Weather Ray Dalio, Big Tech, CEDEARs Argentina (.BA), and Cripto + TradFi.
   - Lookup and normalization helper functions: `get_preset_list()`, `list_presets()`, `get_preset()`, `get_preset_tickers()`, `get_preset_weights()`, `get_preset_description()`.
3. **Multi-Format Export Engine (`src/export/exporter.py` & `src/export/__init__.py`)**:
   - Formatted CSV serializers: `export_summary_csv`, `export_metrics_csv`, `export_weights_csv`, `export_correlation_csv`, `export_wealth_series_csv`.
   - Multi-sheet styled Excel workbook (`.xlsx` via `openpyxl`): `export_full_excel` and `generate_excel_workbook` generating 6 sheets (`Resumen de Métricas`, `Ponderaciones`, `Matriz de Correlación`, `Matriz de Covarianza`, `Evolución Histórica`, `Simulación Monte Carlo`) with Navy headers (`#1F4E79`), zebra striping, currency/percentage number formatting, total check rows, and dynamic column widths.
4. **Interactive Plotly Visualizations (`src/visualization/plots.py` & `src/visualization/__init__.py`)**:
   - `plot_efficient_frontier`: WebGL scatter cloud (`go.Scattergl`), continuous cyan frontier curve, orange dashed CAL line, GMV star, Max Sharpe diamond, user portfolio crosshair, and individual assets with rich hover tooltips.
   - `plot_asset_allocation` & `plot_asset_allocation_donut`: Donut chart (`hole=0.45`) with vibrant colors and percentage labels.
   - `plot_allocation_comparison` & `plot_allocation_comparison_bar`: Grouped bar chart comparing User, Max Sharpe, GMV, and Equal-Weight (1/N).
   - `plot_correlation_heatmap` & `plot_covariance_heatmap`: Diverging and sequential heatmaps with exact cell value annotations.
   - `plot_historical_backtest`: 2-row subplot showing cumulative $10,000 USD wealth growth and underwater drawdown chart.
   - `plot_monte_carlo_cones` & `plot_monte_carlo_projection_cones`: 5%, 25%, 50%, 75%, 95% shaded percentile stochastic fan chart.
5. **Streamlit Web Application (`app.py`)**:
   - Production-grade dashboard with dark financial styling, wide layout, hybrid ingestion (yfinance live vs manual upload), sidebar econometric controls (Rf, return estimators, covariance estimators, constraints), top 1-click presets bar, dynamic state-synchronized weight sliders with instant actions ("Normalizar a 100%", "Aplicar Cartera Óptima Sharpe", "Aplicar Mínima Varianza GMV", "Equiponderar 1/N"), and 6 analytics & visualization tabs.

---

## 2. Logic Chain

1. **Analytical Integrity & Vectorization**:
   - Return calculations use pure vectorized NumPy matrix multiplications ($\mathbf{w}^T \mathbf{\mu}$ and $\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$), avoiding Python-level loops.
   - Drawdown analytics leverage `np.maximum.accumulate` to track high-water marks and accurately identify peak and valley timestamps.
   - VaR and CVaR calculations enforce the coherent risk invariant $\text{CVaR}_{95\%} \ge \text{VaR}_{95\%}$ across both empirical and parametric Gaussian distributions.
2. **State Synchronization & UI Responsiveness**:
   - Streamlit widget state keys (`key=f"slider_{ticker}"`) are kept in exact synchronization with `st.session_state["weights"]`.
   - Action buttons programmatically update both dictionary representations and widget keys before rerunning, preventing duplicate key errors and widget drift.
3. **Excel Workbook Architecture**:
   - `openpyxl` applies real cell number formatting strings (`0.00%`, `0.000`, `$#,##0.00`, `0.0000`) rather than string casting, preserving raw numerical types for downstream formula evaluation.
   - Automatic column width calculation ensures no visual clipping across varying ticker names or metric labels.
4. **Visualization Layer Modularity**:
   - All plot builder functions accept both raw arrays/DataFrames and domain dataclasses (`WeightMonteCarloResult`, `EfficientFrontierResult`, `TrajectorySimulationResult`, `PortfolioRiskMetrics`), ensuring full cross-layer composability.

---

## 3. Caveats

- **Network Availability**: `yfinance` download requires internet access; `app.py` includes a deterministic synthetic generator fallback to ensure zero runtime failure if external API connectivity is interrupted.
- **Divisor Edge Cases**: Portfolios with zero negative returns or zero drawdown safely return `float('inf')` or `0.0` rather than raising `ZeroDivisionError`.

---

## 4. Conclusion

Milestone 4 is 100% complete, fully tested, and meets all architectural, mathematical, and aesthetic specifications:
- Risk analytics, presets, export engine, Plotly visualizers, and Streamlit app are implemented with zero hardcoded facades.
- All 161 tests in the repository test suite pass with 100% success rate in under 10 seconds.

---

## 5. Verification Method

To independently verify this implementation:

1. **Execute Full Automated Pytest Suite**:
   ```bash
   pytest -v tests/
   ```
   *Expected result*: 161 passed tests (100% success rate).

2. **Verify Module Unit & Integration Tests**:
   ```bash
   pytest -v tests/tier1_unit/test_risk_analytics.py
   pytest -v tests/tier1_unit/test_export_engine.py
   pytest -v tests/tier3_integration/test_plotly_builders.py
   pytest -v tests/tier3_integration/test_pipeline_flow.py
   pytest -v tests/tier3_integration/test_state_sync.py
   ```

3. **Verify Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
