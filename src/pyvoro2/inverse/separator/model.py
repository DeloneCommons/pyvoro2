"""Objective models for inverse fitting of power weights.

The inverse-fit API is intentionally generic: downstream code specifies which
pairs matter, which periodic image is used for each pair, and which scalar
separator target should be matched. This module defines the objective pieces
used to fit power weights from those constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import numpy as np

from ..._internal.inputs import coerce_finite_1d_array, owned_readonly_array
from ..._internal.validation import (
    require_finite_real,
    require_nonnegative_finite_real,
    require_positive_finite_real,
)


class ScalarMismatch:
    """Base class for mismatch terms applied to predicted separator positions."""


@dataclass(frozen=True, slots=True)
class SquaredLoss(ScalarMismatch):
    """Quadratic mismatch loss ``0.5 * (predicted - target)**2``."""


@dataclass(frozen=True, slots=True)
class HuberLoss(ScalarMismatch):
    """Huber mismatch penalty in the chosen measurement space.

    For residual ``e``, the loss is ``0.5 * e**2`` when
    ``abs(e) <= delta`` and
    ``delta * (abs(e) - 0.5 * delta)`` otherwise.
    """

    delta: float = 1.0

    def __post_init__(self) -> None:
        delta = require_positive_finite_real(
            self.delta,
            name='HuberLoss.delta',
        )
        object.__setattr__(self, 'delta', delta)


class HardConstraint:
    """Base class for hard feasibility restrictions."""


@dataclass(frozen=True, slots=True)
class Interval(HardConstraint):
    """Hard interval restriction in the chosen measurement space."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = require_finite_real(self.lower, name='Interval.lower')
        upper = require_finite_real(self.upper, name='Interval.upper')
        if not upper > lower:
            raise ValueError('Interval requires upper > lower')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)


@dataclass(frozen=True, slots=True)
class FixedValue(HardConstraint):
    """Hard equality restriction in the chosen measurement space."""

    value: float

    def __post_init__(self) -> None:
        value = require_finite_real(self.value, name='FixedValue.value')
        object.__setattr__(self, 'value', value)


class ScalarPenalty:
    """Base class for additional scalar penalties."""


@dataclass(frozen=True, slots=True)
class SoftIntervalPenalty(ScalarPenalty):
    """Quadratic penalty for leaving a preferred interval.

    At scalar value ``y`` it is
    ``strength * (max(lower - y, 0)**2 + max(y - upper, 0)**2)``.
    """

    lower: float
    upper: float
    strength: float

    def __post_init__(self) -> None:
        lower = require_finite_real(
            self.lower,
            name='SoftIntervalPenalty.lower',
        )
        upper = require_finite_real(
            self.upper,
            name='SoftIntervalPenalty.upper',
        )
        strength = require_nonnegative_finite_real(
            self.strength,
            name='SoftIntervalPenalty.strength',
        )
        if not upper > lower:
            raise ValueError('SoftIntervalPenalty requires upper > lower')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)
        object.__setattr__(self, 'strength', strength)


@dataclass(frozen=True, slots=True)
class ExponentialBoundaryPenalty(ScalarPenalty):
    """Repulsive penalty near the boundaries of an interval.

    At scalar value ``y`` it is ``strength`` times
    ``exp((lower + margin - y) / tau)
    + exp((y - (upper - margin)) / tau)``.
    """

    lower: float = 0.0
    upper: float = 1.0
    margin: float = 0.02
    strength: float = 1.0
    tau: float = 0.01

    def __post_init__(self) -> None:
        lower_value = require_finite_real(
            self.lower,
            name='ExponentialBoundaryPenalty.lower',
        )
        upper_value = require_finite_real(
            self.upper,
            name='ExponentialBoundaryPenalty.upper',
        )
        margin_value = require_nonnegative_finite_real(
            self.margin,
            name='ExponentialBoundaryPenalty.margin',
        )
        strength_value = require_nonnegative_finite_real(
            self.strength,
            name='ExponentialBoundaryPenalty.strength',
        )
        tau_value = require_positive_finite_real(
            self.tau,
            name='ExponentialBoundaryPenalty.tau',
        )
        if not upper_value > lower_value:
            raise ValueError('ExponentialBoundaryPenalty requires upper > lower')
        lower = Fraction.from_float(lower_value)
        upper = Fraction.from_float(upper_value)
        margin = Fraction.from_float(margin_value)
        if lower + margin > upper - margin:
            raise ValueError('ExponentialBoundaryPenalty margin is too large')
        object.__setattr__(self, 'lower', lower_value)
        object.__setattr__(self, 'upper', upper_value)
        object.__setattr__(self, 'margin', margin_value)
        object.__setattr__(self, 'strength', strength_value)
        object.__setattr__(self, 'tau', tau_value)


@dataclass(frozen=True, slots=True)
class ReciprocalBoundaryPenalty(ScalarPenalty):
    """Reciprocal repulsion near interval boundaries.

    This penalty is intended to be used together with a hard interval or a
    strong outside penalty. It penalizes separator positions that enter the
    boundary layers ``[lower, lower + margin]`` and ``[upper - margin, upper]``.
    Each inward boundary distance uses reciprocal repulsion above ``epsilon``
    and its finite tangent continuation at and below ``epsilon``.  A boundary
    contribution is zero at and beyond ``margin``.
    """

    lower: float = 0.0
    upper: float = 1.0
    margin: float = 0.05
    strength: float = 1.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        lower = require_finite_real(
            self.lower,
            name='ReciprocalBoundaryPenalty.lower',
        )
        upper = require_finite_real(
            self.upper,
            name='ReciprocalBoundaryPenalty.upper',
        )
        margin = require_positive_finite_real(
            self.margin,
            name='ReciprocalBoundaryPenalty.margin',
        )
        strength = require_nonnegative_finite_real(
            self.strength,
            name='ReciprocalBoundaryPenalty.strength',
        )
        epsilon = require_positive_finite_real(
            self.epsilon,
            name='ReciprocalBoundaryPenalty.epsilon',
        )
        if not upper > lower:
            raise ValueError('ReciprocalBoundaryPenalty requires upper > lower')
        if not 0.0 < epsilon < margin:
            raise ValueError(
                'ReciprocalBoundaryPenalty requires 0 < epsilon < margin'
            )
        if 2 * Fraction.from_float(margin) > (
            Fraction.from_float(upper) - Fraction.from_float(lower)
        ):
            raise ValueError('ReciprocalBoundaryPenalty margin is too large')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)
        object.__setattr__(self, 'margin', margin)
        object.__setattr__(self, 'strength', strength)
        object.__setattr__(self, 'epsilon', epsilon)


@dataclass(frozen=True, slots=True)
class L2Regularization:
    """Optional ``0.5 * strength * ||weights - reference||**2`` term.

    A supplied reference with zero strength contributes nothing to the
    objective.  Existing solver gauge/alignment policy may still use that
    reference to choose otherwise unidentified component offsets.
    """

    strength: float = 0.0
    reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        strength = require_nonnegative_finite_real(
            self.strength,
            name='L2Regularization.strength',
        )
        object.__setattr__(self, 'strength', strength)
        ref = self.reference
        if ref is not None:
            arr = coerce_finite_1d_array(
                ref,
                name='L2Regularization.reference',
            )
            object.__setattr__(
                self,
                'reference',
                owned_readonly_array(arr, dtype=np.float64),
            )


@dataclass(frozen=True, slots=True)
class FitModel:
    """Complete objective definition for inverse power-weight fitting.

    The objective consists of:
      - one required mismatch term multiplied rowwise by confidence,
      - an optional hard feasibility set,
      - zero or more extra penalties not multiplied by confidence,
      - optional L2 regularization on the weights.

    A scalar penalty with zero strength is mathematically absent.
    """

    mismatch: ScalarMismatch = field(default_factory=SquaredLoss)
    feasible: HardConstraint | None = None
    penalties: tuple[ScalarPenalty, ...] = ()
    regularization: L2Regularization = field(default_factory=L2Regularization)

    def __post_init__(self) -> None:
        if not isinstance(self.mismatch, ScalarMismatch):
            raise ValueError('FitModel.mismatch must be a ScalarMismatch instance')
        if self.feasible is not None and not isinstance(self.feasible, HardConstraint):
            raise ValueError('FitModel.feasible must be a HardConstraint or None')
        try:
            penalties = tuple(self.penalties)
        except TypeError:
            raise ValueError(
                'FitModel.penalties must be an iterable of ScalarPenalty instances'
            ) from None
        object.__setattr__(self, 'penalties', penalties)
        if not all(isinstance(p, ScalarPenalty) for p in penalties):
            raise ValueError('FitModel.penalties must contain ScalarPenalty instances')
        if not isinstance(self.regularization, L2Regularization):
            raise ValueError(
                'FitModel.regularization must be an L2Regularization instance'
            )
