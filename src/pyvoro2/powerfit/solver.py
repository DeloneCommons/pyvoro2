"""Compatibility exports for separator solvers."""

from __future__ import annotations

from ..inverse.separator.solver import (
    ConnectivityDiagnosticsError,
    fit_power_weights,
)

__all__ = ['fit_power_weights', 'ConnectivityDiagnosticsError']
