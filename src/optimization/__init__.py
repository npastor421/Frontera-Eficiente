"""
Optimization module for Markowitz Portfolio Theory.
Provides GMV, Maximum Sharpe Ratio, Target Return solvers,
Efficient Frontier curve sweep, and Capital Allocation Line (CAL).
"""

from __future__ import annotations

from src.optimization.frontier import (
    EfficientFrontierResult,
    compute_capital_allocation_line,
    compute_efficient_frontier,
)
from src.optimization.optimizer import (
    OptimizationResult,
    normalize_and_clamp_weights,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe,
    optimize_target_return,
    parse_and_validate_bounds,
)

__all__ = [
    "OptimizationResult",
    "EfficientFrontierResult",
    "optimize_global_minimum_variance",
    "optimize_maximum_sharpe",
    "optimize_target_return",
    "compute_efficient_frontier",
    "compute_capital_allocation_line",
    "parse_and_validate_bounds",
    "normalize_and_clamp_weights",
]
