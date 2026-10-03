"""Objective models for inverse fitting of power weights.

The inverse-fit API is intentionally generic: downstream code specifies which
pairs matter, which periodic image is used for each pair, and which scalar
separator target should be matched. This module defines the objective pieces
used to fit power weights from those constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
import numpy as np

from ..._internal.inputs import coerce_finite_1d_array, owned_readonly_array
from ..._internal.validation import (
    require_finite_real,
    require_nonnegative_finite_real,
    require_positive_finite_real,
    require_bool,
    require_bool_mask,
    require_string_choice,
)


def _own_row_value(value, *, name, boolean=False, nonnegative=False):
    """Own a scalar or strict 1-D template; length is checked at binding."""
    if np.ndim(value) == 0:
        return (require_bool(value, name=name) if boolean else
                require_nonnegative_finite_real(value, name=name)
                if nonnegative else require_finite_real(value, name=name))
    if boolean:
        original = np.asarray(value, dtype=object)
        if original.ndim != 1:
            raise ValueError(f'{name} must be a scalar or 1D vector')
        return require_bool_mask(value, name=name, length=original.size)
    array = coerce_finite_1d_array(value, name=name)
    if nonnegative and np.any(array < 0.):
        raise ValueError(f'{name} must be non-negative')
    return owned_readonly_array(array, dtype=np.float64)


def _validate_term(term, names, *, hard=False):
    space = (None if term.space is None else require_string_choice(
        term.space, name=f'{type(term).__name__}.space',
        choices=('fraction', 'position'),
    ))
    object.__setattr__(term, 'space', space)
    if hard:
        names = (*names, 'applicable')
    lengths = set()
    for name in names:
        value = _own_row_value(
            getattr(term, name), name=f'{type(term).__name__}.{name}',
            boolean=name == 'applicable', nonnegative=name == 'strength',
        )
        object.__setattr__(term, name, value)
        if isinstance(value, np.ndarray):
            lengths.add(value.size)
    if len(lengths) > 1:
        raise ValueError('row parameters must have compatible vector lengths')
    if 'lower' in names:
        lower, upper = np.broadcast_arrays(term.lower, term.upper)
        if np.any(upper < lower if hard else upper <= lower):
            relation = 'upper >= lower' if hard else 'upper > lower'
            raise ValueError(f'{type(term).__name__} requires {relation}')


def _validate_boundary_width(term):
    lower, upper = np.broadcast_arrays(term.lower, term.upper)
    margin = Fraction.from_float(term.margin)
    if any(2 * margin > Fraction.from_float(float(hi)) -
           Fraction.from_float(float(lo)) for lo, hi in zip(lower.flat, upper.flat)):
        raise ValueError(f'{type(term).__name__} margin is too large')


class ScalarMismatch:
    """Base class for mismatch terms applied to predicted separator positions."""


@dataclass(frozen=True, slots=True)
class SquaredLoss(ScalarMismatch):
    """Quadratic mismatch loss ``0.5 * (predicted - target)**2``."""

    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ())


@dataclass(frozen=True, slots=True)
class HuberLoss(ScalarMismatch):
    """Huber mismatch penalty in the chosen measurement space.

    For residual ``e``, the loss is ``0.5 * e**2`` when
    ``abs(e) <= delta`` and
    ``delta * (abs(e) - 0.5 * delta)`` otherwise.
    """

    delta: float = 1.0
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ())
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

    lower: float | np.ndarray
    upper: float | np.ndarray
    applicable: bool | np.ndarray = field(default=True, kw_only=True)
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ('lower', 'upper'), hard=True)


@dataclass(frozen=True, slots=True)
class FixedValue(HardConstraint):
    """Hard equality restriction in the chosen measurement space."""

    value: float | np.ndarray
    applicable: bool | np.ndarray = field(default=True, kw_only=True)
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ('value',), hard=True)


class ScalarPenalty:
    """Base class for additional scalar penalties."""


@dataclass(frozen=True, slots=True)
class SoftIntervalPenalty(ScalarPenalty):
    """Quadratic penalty for leaving a preferred interval.

    At scalar value ``y`` it is
    ``strength * (max(lower - y, 0)**2 + max(y - upper, 0)**2)``.
    """

    lower: float | np.ndarray
    upper: float | np.ndarray
    strength: float | np.ndarray
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ('lower', 'upper', 'strength'))


@dataclass(frozen=True, slots=True)
class ExponentialBoundaryPenalty(ScalarPenalty):
    """Repulsive penalty near the boundaries of an interval.

    At scalar value ``y`` it is ``strength`` times
    ``exp((lower + margin - y) / tau)
    + exp((y - (upper - margin)) / tau)``.
    """

    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 1.0
    margin: float = 0.02
    strength: float | np.ndarray = 1.0
    tau: float = 0.01
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ('lower', 'upper', 'strength'))
        margin_value = require_nonnegative_finite_real(
            self.margin,
            name='ExponentialBoundaryPenalty.margin',
        )
        tau_value = require_positive_finite_real(
            self.tau,
            name='ExponentialBoundaryPenalty.tau',
        )
        object.__setattr__(self, 'margin', margin_value)
        object.__setattr__(self, 'tau', tau_value)
        _validate_boundary_width(self)


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

    lower: float | np.ndarray = 0.0
    upper: float | np.ndarray = 1.0
    margin: float = 0.05
    strength: float | np.ndarray = 1.0
    epsilon: float = 1e-6
    space: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _validate_term(self, ('lower', 'upper', 'strength'))
        margin = require_positive_finite_real(
            self.margin,
            name='ReciprocalBoundaryPenalty.margin',
        )
        epsilon = require_positive_finite_real(
            self.epsilon,
            name='ReciprocalBoundaryPenalty.epsilon',
        )
        if not 0.0 < epsilon < margin:
            raise ValueError(
                'ReciprocalBoundaryPenalty requires 0 < epsilon < margin'
            )
        object.__setattr__(self, 'margin', margin)
        object.__setattr__(self, 'epsilon', epsilon)
        _validate_boundary_width(self)


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


def _term_getstate(term):
    return [getattr(term, item.name) for item in fields(term)]


def _term_setstate(term, state):
    for item, value in zip(fields(term), state):
        object.__setattr__(term, item.name, value)
    term.__post_init__()


for _term_type in (SquaredLoss, HuberLoss, Interval, FixedValue,
                   SoftIntervalPenalty, ExponentialBoundaryPenalty,
                   ReciprocalBoundaryPenalty, L2Regularization):
    _term_type.__getstate__ = _term_getstate
    _term_type.__setstate__ = _term_setstate
