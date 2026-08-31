## 2026-08-31T13:14:00Z
You are teamwork_preview_worker in a multi-agent quantitative portfolio optimization project.

Your assigned role: Risk Modeling & Covariance Worker (Milestone 2)
Your working directory: c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m2/

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Task:
Read the authoritative specifications and request:
- c:/Nico/Antigravity/Frontera Eficiente/ORIGINAL_REQUEST.md
- c:/Nico/Antigravity/Frontera Eficiente/PROJECT.md
- c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_spec_miner_survey_1/handoff.md

Implement Milestone 2 (Expected Returns, Robust Covariance Models & PSD Stability).
You exclusively own and write:
- `src/models/__init__.py`
- `src/models/returns.py` (Annualized Arithmetic mean, Compound/Geometric CAGR, EWMA returns with $\lambda=0.94$, CAPM expected returns $\mu_i = R_f + \beta_i(\mu_M - R_f)$ with benchmark regression)
- `src/models/covariance.py` (Unbiased sample covariance, Ledoit-Wolf analytical shrinkage with Constant Correlation target and Diagonal `sklearn` target, EWMA covariance matrix with exponential weighting $\tilde{w}_t$)
- `src/models/stability.py` (Symmetry enforcement, eigenvalue validation, condition number diagnostics, Higham 2002 nearest PSD projection algorithm)

Run unit tests / test scripts to verify all functions and numerical invariants ($\Sigma = \Sigma^T, \lambda_{\min} \ge -10^{-8}$).
Write your handoff report to:
`c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m2/handoff.md`

Follow the Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method). Include passing test output in your report.
When done, notify orchestrator via `send_message` with your summary and report path.
