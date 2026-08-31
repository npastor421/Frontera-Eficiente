## 2026-08-31T13:10:04Z
From: parent (c339f8ae-776c-436f-bb88-31dba05b700b)
Role: Optimization & Monte Carlo Explorer
Task:
Read the authoritative user request at:
c:/Nico/Antigravity/Frontera Eficiente/ORIGINAL_REQUEST.md

Investigate and design the mathematical optimization engine and dual Monte Carlo simulation architecture:
1. R3: Quantitative Optimization Engine & Efficient Frontier
   - Objective functions: Maximum Sharpe Ratio, GMV, Target return / risk frontier sweep, CAL equations.
   - Constraints engine: Long-only, short-selling, individual asset bounds.
   - Mathematical precision and tolerance criteria (weights sum to 1.0 +- 1e-5, solver fallback).
2. R4: Dual Monte Carlo Simulations
   - Weight Space Simulation (Dirichlet distribution) on simplex, vectorized NumPy (<2s for 10,000).
   - Multi-Year Stochastic Trajectory Forecasting: Multi-asset GBM with Cholesky correlated Wiener processes, historical block bootstrapping, wealth evolution cones (5%, 25%, 50%, 75%, 95%) from $10,000 USD default.
