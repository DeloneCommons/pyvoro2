"""Compatibility exports for separator active-set fitting."""

from __future__ import annotations

from ..inverse.separator.active import (
    ActiveSetIteration,
    ActiveSetOptions,
    ActiveSetPathSummary,
    PairConstraintDiagnostics,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)

__all__ = [
    'ActiveSetOptions',
    'ActiveSetIteration',
    'ActiveSetPathSummary',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'solve_self_consistent_power_weights',
]
