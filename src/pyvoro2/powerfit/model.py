"""Compatibility exports for separator objective models."""

from __future__ import annotations

from ..inverse.separator.model import (
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HardConstraint,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    ScalarMismatch,
    ScalarPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)

__all__ = [
    'ScalarMismatch',
    'SquaredLoss',
    'HuberLoss',
    'HardConstraint',
    'Interval',
    'FixedValue',
    'ScalarPenalty',
    'SoftIntervalPenalty',
    'ExponentialBoundaryPenalty',
    'ReciprocalBoundaryPenalty',
    'L2Regularization',
    'FitModel',
]
