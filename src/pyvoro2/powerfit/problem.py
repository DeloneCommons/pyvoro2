"""Compatibility exports for separator fit problems."""

from __future__ import annotations

from ..inverse.separator.problem import (
    PowerFitProblem,
    build_power_fit_problem,
    build_power_fit_result,
)

__all__ = [
    'PowerFitProblem',
    'build_power_fit_problem',
    'build_power_fit_result',
]
