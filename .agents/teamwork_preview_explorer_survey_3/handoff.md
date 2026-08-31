# Technical Specification & Handoff Report: UI Layer, Interactive Visualizers, Risk Analytics Engine, Export Engine & E2E Testing Strategy

**Author**: `teamwork_preview_explorer_survey_3`  
**Role**: UI, Risk Metrics & E2E Testing Explorer  
**Target Milestone**: R5 & Comprehensive E2E Testing Framework  
**Project**: Markowitz Efficient Frontier & Quantitative Portfolio Optimization Platform  
**Target File**: `c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_explorer_survey_3/handoff.md`  

---

## 1. Observation

Direct examination of the authoritative requirements in `ORIGINAL_REQUEST.md` and project constraints reveals the following specific architectural and technical mandates:

1. **User Interface (R5)**:
   - Modern interactive Streamlit web application with wide layout.
   - Dynamic sidebar controls: data source selector (Yahoo Finance vs CSV/Excel manual upload), date range picker, ticker input (with default presets & CEDEAR `.BA` support), risk-free rate ($R_f$) input, optimization constraints (Long-Only, Short-Selling, Min/Max asset bounds $[w_{min}, w_{max}]$), covariance estimator selector (Sample Covariance, Ledoit-Wolf Shrinkage, EWMA), and return estimator selector.
   - Interactive weight allocation sliders dynamically synced with Streamlit `st.session_state`.
   - Fast Action Buttons:
     * **"Normalizar a 100%"** (rescales all weights $w_i \leftarrow w_i / \sum w_j$).
     * **"Aplicar Cartera Óptima Sharpe"** (sets slider weights to the calculated Tangency portfolio).
     * **"Aplicar Mínima Varianza (GMV)"** (sets slider weights to GMV portfolio).
     * **"Equiponderada (1/N)"** (sets all weights to $1/N$).
   - 1-Click Predefined Portfolios:
     * **Clásico 60/40**: SPY (60%), BND / TLT (40%)
     * **All-Weather Ray Dalio**: SPY (30%), TLT (40%), IEF (15%), GLD (7.5%), DBC (7.5%)
     * **Big Tech**: AAPL (20%), MSFT (20%), GOOGL (20%), AMZN (20%), NVDA (20%)
     * **CEDEARs Argentina**: AAPL.BA (20%), MSFT.BA (20%), GOOGL.BA (15%), MELI.BA (20%), SPY.BA (15%), KO.BA (10%)
     * **Cripto + TradFi**: SPY (50%), QQQ (30%), BTC-USD (15%), ETH-USD (5%)

2. **Interactive Plotly Visualizations (R5 / R4)**:
   - **Markowitz Efficient Frontier Plot**: Monte Carlo cloud scatter (5,000–20,000 points colored by Sharpe Ratio), continuous frontier curve, Capital Allocation Line (CAL), GMV point, Max Sharpe point, User Portfolio point, individual asset markers with rich hover tooltips.
   - **Asset Allocation Visualizers**: Donut chart (`hole=0.45`) and comparative grouped bar chart (User vs GMV vs Max Sharpe vs Presets).
   - **Interactive Heatmaps**: Correlation matrix ($[-1, 1]$ diverging palette) and Covariance matrix with exact numeric annotations.
   - **Historical Backtest & Wealth Growth**: Cumulative evolution of $\$10,000$ USD initial capital over the selected time horizon for User Portfolio, Max Sharpe, GMV, and Benchmark, with secondary Drawdown Under-Water chart.
   - **Monte Carlo Future Wealth Projection Cones**: 1–5 year stochastic wealth fan chart with 5%, 25%, 50%, 75%, and 95% percentile shaded bands.

3. **Advanced Risk Metrics Engine (R5)**:
   - Complete analytical metrics suite: Annualized Return (Arithmetic & CAGR), Annualized Volatility ($\sqrt{w^T \Sigma w} \times \sqrt{252}$), Sharpe Ratio ($R_f$ adjusted), Sortino Ratio (downside semideviation), Calmar Ratio (Annualized Return / Max Drawdown), Maximum Drawdown (peak-to-trough series), Value at Risk (VaR 95% Historical and Parametric), Conditional Value at Risk (CVaR 95% / Expected Shortfall).

4. **Export Engine & Test Strategy (R5 / Acceptance Criteria)**:
   - Export to CSV and multi-tab Excel (`.xlsx` via `openpyxl` / `xlsxwriter`) with formatted tables, metrics, weights, covariance/correlation matrices, and simulation samples.
   - 4-Tier Pytest test suite ensuring 100% test pass rate, validating mathematical invariants ($\sum w_i = 1$, $\sigma_{GMV} \le \sigma_{any}$, $SR_{MaxSharpe} \ge SR_{any}$, positive semi-definite covariance matrices).

---

## 2. Logic Chain & Technical Specifications

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           STREAMLIT UI LAYER                           │
  │  ┌───────────────────────────┐    ┌──────────────────────────────────┐ │
  │  │      Sidebar Controls     │    │       Preset Portfolio Bar       │ │
  │  │ Data Source | Tickers     │    │  [60/40] [All-Weather] [BigTech] │ │
  │  │ Dates | Rf | Estimators   │    │  [CEDEARs.BA] [Cripto+TradFi]    │ │
  │  └─────────────┬─────────────┘    └────────────────┬─────────────────┘ │
  │                ▼                                   ▼                   │
  │  ┌──────────────────────────────────────────────────────────────────┐  │
  │  │       Interactive Weight Sliders & Live State Synchronization    │  │
  │  │   [Slider 1] [Slider 2] ... [Normalizar] [Aplicar Max Sharpe]     │  │
  │  └─────────────────────────────────┬────────────────────────────────┘  │
  └────────────────────────────────────┼───────────────────────────────────┘
                                       ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                      CORE ANALYTICS & COMPUTATION                      │
  │  ┌───────────────────────────┐    ┌──────────────────────────────────┐ │
  │  │   Risk Metrics Engine     │    │     Monte Carlo & Backtest       │ │
  │  │ Return, Vol, Sharpe,      │    │  - Dual MC (Weights & GBM paths) │ │
  │  │ Sortino, Calmar, MaxDD,   │    │  - $10,000 Wealth Backtest       │ │
  │  │ VaR 95%, CVaR 95%         │    │  - Drawdown series               │ │
  │  └─────────────┬─────────────┘    └────────────────┬─────────────────┘ │
  └────────────────┼───────────────────────────────────┼───────────────────┘
                   ▼                                   ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     VISUALIZATION & EXPORT LAYER                       │
  │  ┌──────────────────────────────────────────────────────────────────┐  │
  │  │                   Interactive Plotly Visualizers                 │  │
  │  │  1. Markowitz Frontier + CAL + Cloud   2. Asset Allocation       │  │
  │  │  3. Correlation & Covariance Heatmaps  4. Wealth Backtest        │  │
  │  │  5. 1-5 Year Future Projection Cones (5%, 25%, 50%, 75%, 95%)   │  │
  │  └─────────────────────────────────┬────────────────────────────────┘  │
  │  ┌─────────────────────────────────┴────────────────────────────────┐  │
  │  │            Export Engine (CSV + Multi-Tab Excel Workbook)        │  │
  │  └──────────────────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

### Section 2.1: Streamlit User Interface Architecture & Dynamic State Machine

#### 2.1.1 Application Layout and Component Hierarchy
The UI is organized using a clean, multi-column dashboard design in Streamlit:
- **Header**: Application Title, quantitative subtitle, and status badge showing active data mode (Live yfinance vs Manual CSV).
- **Sidebar**: Configuration pane containing data ingestion parameters, asset universe selectors, econometric estimator toggles, and optimization bounds.
- **Top Row (Presets)**: Horizontal bar of stylized 1-click preset buttons (`st.columns(5)`).
- **Interactive Allocation Panel**: Expandable or dedicated container with dynamic sliders per active ticker, sum indicator badge (with color-coded validation), and instant action buttons.
- **Tabbed Analytics & Visualization Container**:
  * **Tab 1: Frontera Eficiente & Optimización** (Markowitz scatter cloud, continuous curve, CAL, GMV & Max Sharpe indicators, user portfolio marker).
  * **Tab 2: Asignación de Activos** (Interactive Donut chart + Comparative 4-way Bar chart).
  * **Tab 3: Matrices de Riesgo** (Interactive Correlation & Covariance heatmaps).
  * **Tab 4: Backtest Histórico & Drawdown** ($10,000 USD trajectory + Under-water drawdown chart).
  * **Tab 5: Proyección Monte Carlo** (1–5 year GBM / Bootstrap fan chart with 5 shaded percentiles).
  * **Tab 6: Métricas & Exportación** (Comprehensive risk table + CSV/Excel download buttons).

#### 2.1.2 Streamlit Session State Management & Slider Synchronization Pattern
A key technical challenge in Streamlit is updating widget states programmatically (e.g. clicking "Normalizar" or "Aplicar Max Sharpe") without raising `StreamlitDuplicateKeyId` or losing user adjustments on re-render.

**State Machine Specification**:
```python
# Session State Keys
st.session_state['tickers']        # list[str]: e.g. ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
st.session_state['weights']        # dict[str, float]: e.g. {'AAPL': 0.20, 'MSFT': 0.20, ...}
st.session_state['rf_rate']        # float: annualized risk-free rate (e.g. 0.045)
st.session_state['opt_results']    # dict: cached optimization outputs (Max Sharpe, GMV, Frontier)
st.session_state['cov_method']     # str: 'ledoit_wolf' | 'sample' | 'ewma'
st.session_state['price_df']       # pd.DataFrame: cleansed historical price series
st.session_state['returns_df']     # pd.DataFrame: aligned daily returns
```

**Slider Synchronization Pattern**:
To dynamically synchronize sliders when preset or optimization buttons are clicked:
1. Sliders use unique keys tied to active tickers: `key=f"slider_{ticker}"`.
2. When a preset or action button is triggered, update both `st.session_state['weights'][ticker]` and `st.session_state[f"slider_{ticker}"]` before `st.rerun()`.
3. Slider `on_change` callback or direct value reading updates `st.session_state['weights'][ticker]`.
4. Normalization logic:
   $$\text{sum\_w} = \sum_{i=1}^N w_i$$
   If $\text{sum\_w} > 0$: $w_i \leftarrow \frac{w_i}{\text{sum\_w}}$
   If $\text{sum\_w} == 0$: $w_i \leftarrow \frac{1}{N}$

#### 2.1.3 Detailed Specifications of the 5 Predefined Portfolios
Each preset defines a canonical asset allocation tailored to different investor risk profiles:

| Preset Name | Tickers & Target Weights | Rationale & Asset Classes |
|:---|:---|:---|
| **1. Clásico 60/40** | `SPY`: 60.0%<br>`TLT`: 40.0% (or `BND`) | Classic institutional benchmark balancing US Large Cap equities with Long-Term Treasury bonds for equity risk mitigation. |
| **2. All-Weather (Ray Dalio)** | `SPY`: 30.0%<br>`TLT`: 40.0%<br>`IEF`: 15.0%<br>`GLD`: 7.5%<br>`DBC`: 7.5% | Macro-resilient risk-parity allocation across Growth (SPY), Deflation (TLT/IEF), and Inflation/Stagflation (GLD/DBC). |
| **3. Big Tech** | `AAPL`: 20.0%<br>`MSFT`: 20.0%<br>`GOOGL`: 20.0%<br>`AMZN`: 20.0%<br>`NVDA`: 20.0% | Mega-cap technology growth portfolio with high historical Sharpe and high inter-asset correlation. |
| **4. CEDEARs Argentina** | `AAPL.BA`: 20.0%<br>`MSFT.BA`: 20.0%<br>`GOOGL.BA`: 15.0%<br>`MELI.BA`: 20.0%<br>`SPY.BA`: 15.0%<br>`KO.BA`: 10.0% | Argentine market instruments trading in ARS with underlying USD FX hedging (CCL) and local liquidity. |
| **5. Cripto + TradFi** | `SPY`: 50.0%<br>`QQQ`: 30.0%<br>`BTC-USD`: 15.0%<br>`ETH-USD`: 5.0% | Modern barbell hybrid combining traditional equity index ETFs with asymmetric crypto-asset allocation. |

---

### Section 2.2: Interactive Plotly Visualizers Design Specifications

All Plotly visualizers must adhere to high-grade financial aesthetics: dark/modern theme (`plotly_dark` or custom slate palette), clear gridlines (`#2d3748`), high contrast markers, and rich custom hover templates (`hovertemplate`).

#### 2.2.1 Markowitz Efficient Frontier Plot
- **Trace 1 (Monte Carlo Cloud)**:
  * Type: `go.Scattergl` (WebGL for high-performance rendering of 10,000+ points).
  * $x = \sigma_{\text{annual}}$, $y = \mu_{\text{annual}}$, `mode='markers'`.
  * Marker: `size=3.5`, `opacity=0.45`, `color=sharpe_ratios`, `colorscale='Viridis'`, `showscale=True`, `colorbar=dict(title="Sharpe Ratio")`.
  * `hovertemplate`: `<b>Cartera Simulada</b><br>Retorno: %{y:.2%}<br>Volatilidad: %{x:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>`.
- **Trace 2 (Continuous Efficient Frontier Curve)**:
  * Type: `go.Scatter`, `mode='lines'`.
  * Solved for $K=100$ target return points between $\mu_{GMV}$ and $\max(\mu_i)$.
  * Line styling: `color='#00F0FF'`, `width=3.5`.
  * `hovertemplate`: `<b>Frontera Eficiente</b><br>Retorno: %{y:.2%}<br>Volatilidad Mínima: %{x:.2%}<extra></extra>`.
- **Trace 3 (Capital Allocation Line - CAL)**:
  * Type: `go.Scatter`, `mode='lines'`.
  * Linear function: $\mu(\sigma) = R_f + \left(\frac{\mu_{\text{Sharpe}} - R_f}{\sigma_{\text{Sharpe}}}\right) \sigma$.
  * Range: $\sigma \in [0, 1.3 \times \max(\sigma_{\text{assets}})]$.
  * Line styling: `color='#FFA500'`, `dash='dash'`, `width=2`.
  * `hovertemplate`: `<b>Línea de Asignación de Capital (CAL)</b><br>Retorno: %{y:.2%}<br>Volatilidad: %{x:.2%}<extra></extra>`.
- **Trace 4 (Global Minimum Variance - GMV)**:
  * Marker: Star (`symbol='star'`, `size=16`, `color='#FF3366'`, `line=dict(color='white', width=1.5)`).
- **Trace 5 (Maximum Sharpe Tangency Portfolio)**:
  * Marker: Diamond/Star (`symbol='diamond'`, `size=16`, `color='#00FF66'`, `line=dict(color='white', width=1.5)`).
- **Trace 6 (User Custom Portfolio)**:
  * Marker: Large circle with crosshair (`symbol='circle-dot'`, `size=18`, `color='#FFCC00'`, `line=dict(color='white', width=2)`).
- **Trace 7 (Individual Assets)**:
  * Individual $(\sigma_i, \mu_i)$ scatter points with asset text labels (`text=tickers`, `textposition='top center'`).

#### 2.2.2 Asset Allocation Visualizers
- **Donut Chart**:
  * Type: `go.Pie` with `hole=0.45`.
  * Slices: Active asset weights $w_i$.
  * Styling: `textinfo='label+percent'`, `insidetextorientation='radial'`, modern vibrant color sequence (`px.colors.qualitative.Plotly` or `Dark24`).
- **Comparative Bar Chart**:
  * Type: `go.Bar` in grouped mode (`barmode='group'`).
  * X-axis: Asset tickers.
  * Y-axis: Allocation percentage ($0\%$ to $100\%$).
  * Series: User Allocation, Max Sharpe Allocation, GMV Allocation, Equal-Weight (1/N).

#### 2.2.3 Interactive Correlation and Covariance Heatmaps
- **Correlation Heatmap**:
  * Matrix $\mathbf{C} \in [-1, 1]^{N \times N}$.
  * Colorscale: `RdBu_r` or `Tealrose` centered at `zmid=0`, `zmin=-1`, `zmax=1`.
  * Annotations: Display numerical correlation in each cell formatted as `f"{val:.2f}"`.
- **Covariance Heatmap**:
  * Matrix $\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$ (Annualized).
  * Colorscale: `Blues` or `Viridis`.
  * Annotations: Scaled or scientific notation formatted as `f"{val*10000:.2f} bp"`.

#### 2.2.4 Historical Backtest & Drawdown Visualizers
- **Wealth Index Growth ($10,000 USD)**:
  * Calculates daily compounding trajectory:
    $$V_p(t) = 10000 \times \prod_{\tau=1}^t (1 + R_{p,\tau})$$
  * Multi-line Plotly chart showing User Portfolio, Max Sharpe, GMV, and Benchmark (e.g. SPY / 1/N).
  * Hover template shows exact dollar portfolio value and cumulative percentage return.
- **Drawdown Under-Water Chart**:
  * Subplot directly below the wealth chart.
  * Area chart (`fill='tozeroy'`) showing $DD(t) \in [-100\%, 0\%]$ filled with semi-transparent crimson/red (`rgba(255, 65, 54, 0.3)`).

#### 2.2.5 Monte Carlo Future Wealth Projection Cones (Fan Chart)
- **Stochastic Path Simulation**:
  * Simulates $M=1,000$ to $5,000$ geometric Brownian motion trajectories over horizon $T \in [1, 5]$ years (252 to 1260 trading days):
    $$S(t + \Delta t) = S(t) \exp\left( \left(\mu_p - \frac{1}{2}\sigma_p^2\right) \Delta t + \sigma_p \sqrt{\Delta t} Z_t \right), \quad Z_t \sim \mathcal{N}(0, 1)$$
- **Plotly Fan Chart Layering**:
  1. 95th Percentile Upper Bound (Trace: `line=dict(width=0)`).
  2. 5th Percentile Lower Bound (Trace: `fill='tonexty'`, `fillcolor='rgba(0, 240, 255, 0.15)'`, `line=dict(width=0)`).
  3. 75th Percentile Upper Bound (Trace: `line=dict(width=0)`).
  4. 25th Percentile Lower Bound (Trace: `fill='tonexty'`, `fillcolor='rgba(0, 240, 255, 0.35)'`, `line=dict(width=0)`).
  5. 50th Percentile / Median Path (Trace: `line=dict(color='#00F0FF', width=3)`).
  6. Expected Value Path $\mathbb{E}[S(t)] = S_0 e^{\mu_p t}$ (Trace: `line=dict(color='#FFA500', dash='dash', width=2)`).

---

### Section 2.3: Advanced Risk Metrics Engine — Exact Mathematical Formulations

Let $R_{p,t}$ be the daily portfolio return at day $t \in \{1, \dots, T\}$, $\mathbf{w} \in \mathbb{R}^N$ the asset allocation vector, $\mathbf{\mu} \in \mathbb{R}^N$ the annualized expected returns vector, $\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$ the annualized covariance matrix, and $R_f$ the annualized risk-free rate.

| Metric | Mathematical Formula | Implementation Specification & Edge Cases |
|:---|:---|:---|
| **1. Annualized Return ($\mu_p$)** | $\mu_{\text{arith}} = \mathbf{w}^T \mathbf{\mu}$<br>$\text{CAGR} = \left(\prod_{t=1}^T (1 + R_{p,t})\right)^{\frac{252}{T}} - 1$ | Report both Arithmetic Expected Return and Compound Annual Growth Rate (CAGR). Handle leap years and fractional years cleanly. |
| **2. Annualized Volatility ($\sigma_p$)** | $\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}} = \text{std}(R_{p,t}) \times \sqrt{252}$ | Ensure $\mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \ge 0$. If variance $< 10^{-16}$, clip to 0 to prevent numerical NaN. |
| **3. Sharpe Ratio ($SR$)** | $SR = \frac{\mu_p - R_f}{\sigma_p}$ | If $\sigma_p == 0$, return $0.0$ or $\text{NaN}$ with UI label `"N/A"`. |
| **4. Sortino Ratio** | $\text{Sortino} = \frac{\mu_p - R_f}{\sigma_D}$<br>where $\sigma_D = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_{p,t} - \tau)^2} \times \sqrt{252}$ | $\tau = \frac{R_f}{252}$ (daily MAR) or $\tau = 0$. If $\sigma_D == 0$ (no negative returns), display `"Sin riesgo a la baja"`. |
| **5. Calmar Ratio** | $\text{Calmar} = \frac{\text{CAGR}}{\|\text{Max Drawdown}\|}$ | If $\|\text{Max Drawdown}\| == 0$, return $\text{NaN}$ or `"N/A"`. |
| **6. Maximum Drawdown ($MDD$)** | $W_t = \prod_{\tau=1}^t (1 + R_{p,\tau})$<br>$DD_t = \frac{W_t - \max_{\tau \le t} W_\tau}{\max_{\tau \le t} W_\tau}$<br>$MDD = \|\min_{t} DD_t\|$ | Vectorized via `np.maximum.accumulate`. Also compute Peak Date, Valley Date, and Recovery Duration. |
| **7. Value at Risk (VaR 95%)** | **Historical (1-Day)**: $-\text{Percentile}(R_{p}, 5\%)$<br>**Parametric Normal**: $z_{0.95} \sigma_{\text{daily}} - \mu_{\text{daily}}$ ($z_{0.95} = 1.64485$) | Provide both 1-Day VaR and Annualized VaR ($\text{VaR}_{1Y} = z_{0.95}\sigma_p - \mu_p$). |
| **8. Conditional VaR (CVaR 95%)** | **Historical**: $-\frac{1}{\|K\|} \sum_{R_{p,t} \le -\text{VaR}_{95}} R_{p,t}$<br>**Parametric Normal**: $\sigma_{\text{daily}} \frac{\phi(z_{0.95})}{0.05} - \mu_{\text{daily}}$ | Coherent risk measure (Expected Shortfall). Subadditive across assets. $\phi$ is the standard normal PDF ($\phi(1.64485) \approx 0.103135$). |

---

### Section 2.4: Export Engine Specification (CSV & Multi-Sheet Excel)

#### 2.4.1 CSV Export Format
Single or bundled zipped CSVs:
1. `resumen_carteras.csv`: Portfolio Name, Annualized Return, Volatility, Sharpe, Sortino, Calmar, MaxDD, VaR 95%, CVaR 95%.
2. `ponderaciones_optimas.csv`: Ticker, Peso Usuario, Peso Max Sharpe, Peso GMV, Peso Equiponderado.
3. `matriz_correlacion.csv`: $N \times N$ correlation table.
4. `series_historicas.csv`: Date, Asset Prices, Portfolio Wealth Series.

#### 2.4.2 Multi-Sheet Excel Workbook Architecture (`openpyxl` / `xlsxwriter`)
File Name: `optimizacion_frontera_eficiente_YYYYMMDD.xlsx`

- **Sheet 1: `Resumen de Métricas`**:
  * Formatted summary table comparing: Usuario, Máximo Sharpe, Mínima Varianza (GMV), Equiponderada (1/N), and Active Preset.
  * Columns: Métrica, Cartera Usuario, Máx Sharpe, GMV, Equiponderada.
  * Number formatting: Percentages `0.00%`, Ratios `0.000`, Dollar values `$#,##0.00`.
- **Sheet 2: `Ponderaciones`**:
  * Matrix of weights across all compared portfolios.
  * Total check row verifying $\sum w_i = 100.00\%$.
- **Sheet 3: `Matriz de Correlación`**:
  * Formatted correlation matrix with conditional formatting color scale (Red-White-Blue).
- **Sheet 4: `Matriz de Covarianza`**:
  * Annualized covariance values.
- **Sheet 5: `Evolución Histórica`**:
  * Daily dates, cumulative returns, and wealth index of $\$10,000$ USD.
- **Sheet 6: `Simulación Monte Carlo`**:
  * Subsample of 1,000 simulated portfolios (Retorno, Volatilidad, Sharpe, Asignaciones por Activo).

**Workbook Styling Specifications**:
- Header Row: Dark Navy background (`#1F4E79`), White bold text, centered alignment.
- Data Rows: Alternating zebra striping (White / `#F2F4F7`), thin borders (`#D3D3D3`).
- Number Formats applied explicitly via OpenPyXL / XlsxWriter styles (not raw float strings).

---

### Section 2.5: Comprehensive 4-Tier Pytest Suite Architecture

```
tests/
├── conftest.py                   # Shared synthetic fixtures, market data mocks, seed control
├── tier1_unit/                   # Module-level mathematical & algorithmic unit tests
│   ├── test_data_loader.py       # Ticker cleansing, date alignment, CSV upload parser
│   ├── test_covariance_models.py # Sample, Ledoit-Wolf Shrinkage, EWMA, PSD invariant
│   ├── test_markowitz_engine.py  # Quadratic programming, Max Sharpe, GMV, CAL, Dirichlet MC
│   ├── test_risk_analytics.py    # Sharpe, Sortino, Calmar, MaxDD, VaR 95%, CVaR 95%
│   └── test_export_engine.py     # CSV serialization, OpenPyXL multi-sheet binary generation
├── tier2_boundary_corner/        # Edge cases, adversarial inputs, singular matrices
│   ├── test_single_asset.py      # N=1 edge case (w=[1.0], zero cov variance, metrics stability)
│   ├── test_collinear_assets.py  # Perfectly correlated assets (rho=1.0), matrix ill-conditioning
│   ├── test_negative_returns.py  # All assets mu < Rf (negative Sharpe frontier behavior)
│   ├── test_zero_weights.py      # Zero allocation to subset of assets, metric sanity
│   └── test_extreme_outliers.py  # Flash crashes (-90%), extreme returns, holiday misalignments
├── tier3_integration/            # End-to-end component dataflow validation
│   ├── test_pipeline_flow.py     # Ingestion -> Covariance -> Optimization -> Metrics -> Export
│   ├── test_plotly_builders.py   # Plotly figure generation, trace counts, valid schema
│   └── test_state_sync.py        # Normalization logic, slider state transitions
└── tier4_real_world_workflows/   # Full workflow validation against real market tickers
    ├── test_classic_60_40.py     # 60/40 preset end-to-end workflow
    ├── test_all_weather.py       # All-Weather multi-asset (commodities, gold, bonds) workflow
    ├── test_cedears_argentina.py # Argentine CEDEARs (.BA) cross-currency alignment workflow
    └── test_crypto_tradfi.py     # 24/7 Crypto + 5-day TradFi date alignment workflow
```

#### Detailed Test Scenarios & Invariant Assertions:
1. **Mathematical Invariant Assertions**:
   - **Weight Sum**: $\left|\sum_{i=1}^N w_i - 1.0\right| < 10^{-5}$.
   - **Bounds Satisfaction**: $\forall i, w_{min} - 10^{-6} \le w_i \le w_{max} + 10^{-6}$.
   - **GMV Optimality**: $\sigma(\mathbf{w}_{GMV}) \le \sigma(\mathbf{w}) + 10^{-6}$ for all random simplex weights $\mathbf{w}$.
   - **Max Sharpe Optimality**: $SR(\mathbf{w}_{Sharpe}) \ge SR(\mathbf{w}) - 10^{-6}$ for all random simplex weights $\mathbf{w}$.
   - **Covariance Positive Semi-Definiteness**: All eigenvalues $\lambda_k(\mathbf{\Sigma}) \ge -10^{-8}$.
   - **Coherent Risk Invariant**: $\text{CVaR}_{95\%} \ge \text{VaR}_{95\%}$ strictly holds for all historical distributions.

2. **Benchmark & Performance Constraints**:
   - Vectorized Monte Carlo generation of 10,000 portfolios must complete in $< 1.5$ seconds on standard CPU.
   - Efficient frontier optimization (50 points) must solve in $< 0.5$ seconds.
   - Dataframe date alignment with forward-fill and dropna handles missing holidays with zero data leakage.

---

## 3. Caveats & Edge-Case Analysis

1. **Crypto (24/7) vs Traditional Equity (252 Days) Date Misalignment**:
   - *Observation*: Bitcoin (`BTC-USD`) and Ethereum (`ETH-USD`) trade 365 days a year, whereas US/BA equities trade ~252 days.
   - *Design Choice*: Inner join vs Outer join with forward fill (`ffill`). Recommended: Standardize on equity trading calendar (intersection) or reindex with forward-fill, dropping non-trading equity days to prevent artificial zero-volatility weekend artifacts.
2. **Sortino & Calmar Divisor Degeneracy**:
   - *Observation*: Portfolios with zero negative return days yield downside deviation $\sigma_D = 0$. Portfolios with no drawdown yield $MDD = 0$.
   - *Design Choice*: The metrics engine must return `np.nan` or `None` with dedicated UI fallback string (`"N/A"` or `"Sin riesgo a la baja"`) instead of crashing with `ZeroDivisionError`.
3. **Streamlit Component Re-rendering Overhead**:
   - *Observation*: Heavy Monte Carlo simulation recalculation on every slider drag can degrade UI responsiveness.
   - *Design Choice*: Cache computationally intensive steps (`@st.cache_data`) for data fetching and covariance computation, and perform weight metrics calculations vectorially in pure NumPy/Numba.

---

## 4. Conclusion

The specification provides a complete, robust, and production-ready blueprint for:
1. **Interactive UI Layer (R5)**: Seamless Streamlit architecture, state-synchronized weight allocation sliders, 1-click preset portfolios (60/40, All-Weather, Big Tech, CEDEARs.BA, Cripto+TradFi), and instant normalization/Sharpe application.
2. **Interactive Visualizers**: High-performance Plotly visualizers covering the Markowitz Efficient Frontier (with continuous curve, CAL, and 10,000-point Monte Carlo cloud), Donut and comparative Bar allocation charts, Correlation/Covariance heatmaps, historical $\$10,000$ USD backtest with drawdown under-water chart, and 1–5 year stochastic projection cones.
3. **Advanced Risk Metrics Engine**: Rigorous mathematical formulation and vectorized calculation of Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Historical/Parametric VaR (95%), and CVaR (95%).
4. **Multi-Format Export & 4-Tier Pytest Suite**: Full CSV and multi-sheet styled Excel workbook generation, accompanied by a comprehensive 4-Tier Pytest test suite enforcing strict mathematical and functional invariants.

---

## 5. Verification Method

To independently verify this specification and its downstream implementation:
1. **Inspect Handoff Artifact**:
   `view_file` on `c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_explorer_survey_3/handoff.md`.
2. **Execute Automated Pytest Suite (upon code generation)**:
   ```bash
   pytest -v tests/
   ```
3. **Verify Streamlit Application Launch & Visual Rendering**:
   ```bash
   streamlit run app.py
   ```
4. **Verify Excel Multi-Sheet Binary Generation**:
   Run export unit tests and verify generated `.xlsx` file contains all 6 specified sheets with valid formatting.
