## 2026-08-31T13:10:04Z

You are teamwork_preview_spec_miner in a multi-agent quantitative finance software development project.

Your assigned role: Data Ingestion & Risk Modeling Spec Miner
Your working directory: c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_spec_miner_survey_1/

Task:
Read the authoritative user request at:
c:/Nico/Antigravity/Frontera Eficiente/ORIGINAL_REQUEST.md

Investigate and document all technical specifications, mathematical equations, data schemas, edge cases, error handling, and implementation requirements for:
1. **R1: Data Ingestion & Hybrid Handling**:
   - `yfinance` multi-ticker downloading (US stocks, ETFs, crypto such as BTC-USD/ETH-USD, CEDEARs with `.BA` suffix like AAPL.BA).
   - Manual upload of CSV / Excel files with price or return time series.
   - Robust data sanitization, date alignment, calendar handling (holidays, weekends, missing dates across different exchanges), missing value treatment (ffill, bfill, dropna), and Streamlit caching strategy (`@st.cache_data`) with cache invalidation rules.
2. **R2: Statistical Modeling & Robust Risk Estimation**:
   - Expected return estimators: Annualized historical arithmetic/geometric mean, EWMA returns, CAPM expected returns with a market benchmark (e.g. SPY / ^GSPC).
   - Covariance estimators: Classical sample covariance, **Ledoit-Wolf Analytical Shrinkage** (target: constant correlation or diagonal/identity; explain formulas and scikit-learn / custom analytical implementation), and EWMA covariance matrix ($\lambda = 0.94$ or user configurable).
   - Numerical stability guarantees: Symmetry enforcement ($\frac{\Sigma + \Sigma^T}{2}$), Positive Semi-Definite (PSD) validation (eigenvalue decomposition, nearest PSD projection via Higham method if needed), condition number checks, annualization factor (252 trading days for TradFi, 365 for crypto).
   - Interactive Correlation and Covariance matrix data structures and presentation.

Write your comprehensive findings and specification report to:
`c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_spec_miner_survey_1/handoff.md`

Follow the Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method).
When done, notify orchestrator via `send_message` with your report summary and the path to your `handoff.md`.
