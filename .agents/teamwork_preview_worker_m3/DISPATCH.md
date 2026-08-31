## 2026-08-31T13:24:18Z

You are teamwork_preview_worker in a multi-agent quantitative portfolio optimization project.

Your assigned role: Optimization & Monte Carlo Worker (Milestone 3)
Your working directory: c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m3/

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Task:
Read the authoritative specifications and request:
- c:/Nico/Antigravity/Frontera Eficiente/ORIGINAL_REQUEST.md
- c:/Nico/Antigravity/Frontera Eficiente/PROJECT.md
- c:/Nico/Antigravity/Frontera Eficiente/TEST_INFRA.md
- c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_explorer_survey_2/handoff.md

Implement Milestone 3 (Markowitz Optimization Engine & Dual Monte Carlo).
You exclusively own and write:
- `src/optimization/__init__.py`
- `src/optimization/optimizer.py`:
  * Dataclass `OptimizationResult(weights, expected_return, volatility, sharpe_ratio, status, success, iterations)`
  * `optimize_global_minimum_variance(cov_matrix, expected_returns=None, rf=0.04, bounds=(0.0, 1.0), custom_bounds=None)` with exact analytical gradient $\nabla f = \mathbf{\Sigma} \mathbf{w}$
  * `optimize_maximum_sharpe(expected_returns, cov_matrix, rf=0.04, bounds=(0.0, 1.0), custom_bounds=None)` with exact analytical Jacobian $\nabla f_{ms} = -\frac{\mathbf{\mu}}{\sigma_p} + \frac{(\mu_p - R_f)}{\sigma_p^3}(\mathbf{\Sigma} \mathbf{w})$
  * Support for Long-Only ($0 \le w_i \le 1$), Short-Selling ($w_{min} \le w_i \le w_{max}$ with $w_{min} < 0$), and custom per-asset bounds
  * Budget constraint validation ($\sum w_{min, i} \le 1.0 \le \sum w_{max, i}$), boundary clamping, and post-optimization normalization ensuring $|\sum w_i - 1.0| < 10^{-12} \ll 10^{-5}$
  * 4-stage solver fallback cascade (SLSQP analytical -> SLSQP numerical -> trust-constr -> Tikhonov jitter regularization)
- `src/optimization/frontier.py`:
  * Dataclass `EfficientFrontierResult(returns, volatilities, weights, sharpe_ratios, gmv_portfolio, max_sharpe_portfolio, cal_line)`
  * `compute_efficient_frontier(expected_returns, cov_matrix, rf=0.04, num_points=100, bounds=(0.0, 1.0), custom_bounds=None)` with warm-start chained quadratic optimization
  * `compute_capital_allocation_line(max_sharpe_result, rf=0.04, max_volatility=None)` tangent line from $(0, R_f)$ through $(\sigma_{ms}, \mu_{ms})$
- `src/simulation/__init__.py`
- `src/simulation/weight_monte_carlo.py`:
  * Dataclass `WeightMonteCarloResult(weights, returns, volatilities, sharpe_ratios, max_sharpe_idx, min_vol_idx)`
  * `run_weight_space_monte_carlo(expected_returns, cov_matrix, rf=0.04, num_portfolios=10000, seed=None)`: Vectorized Dirichlet uniform simplex sampling on $\Delta^{k-1}$ executing 10,000-20,000 portfolios in <50ms
- `src/simulation/trajectory_monte_carlo.py`:
  * Dataclass `TrajectorySimulationResult(days, years, percentile_5, percentile_25, percentile_50, percentile_75, percentile_95, mean_trajectory, sample_paths, initial_wealth, final_wealth_stats)`
  * `run_trajectory_monte_carlo(...)`: Correlated Multi-Asset Geometric Brownian Motion (via Cholesky decomposition $\mathbf{\Sigma} = \mathbf{L} \mathbf{L}^T$) and Historical Block Bootstrapping ($b=10$ days) for 1–5 years ($252 \times T$ days) with 5%, 25%, 50%, 75%, 95% quantile cones from initial capital ($10,000 USD default).

Run pytest on `tests/tier1_unit/test_markowitz_engine.py` and write your handoff report to:
`c:/Nico/Antigravity/Frontera Eficiente/.agents/teamwork_preview_worker_m3/handoff.md`

Follow the Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method).
When done, notify orchestrator via `send_message`.
