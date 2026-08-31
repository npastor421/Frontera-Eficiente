"""
Simulation module for portfolio Monte Carlo analysis.
Provides Weight Space Dirichlet simplex simulation and Multi-Year Stochastic
Trajectory forecasting (Correlated Multi-Asset GBM and Historical Block Bootstrapping).
"""

from __future__ import annotations

from src.simulation.trajectory_monte_carlo import (
    TrajectorySimulationResult,
    run_trajectory_monte_carlo,
)
from src.simulation.weight_monte_carlo import (
    WeightMonteCarloResult,
    run_weight_space_monte_carlo,
)

__all__ = [
    "WeightMonteCarloResult",
    "TrajectorySimulationResult",
    "run_weight_space_monte_carlo",
    "run_trajectory_monte_carlo",
]
