"""Authoritative scalar formulas for separator inverse objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    localcontext,
)
from fractions import Fraction

import numpy as np

from ._numerics import (
    _array_ball_add,
    _array_ball_divide,
    _array_ball_multiply,
    _array_ball_negate,
    _array_ball_subtract,
    _array_up_add,
    _ArrayBall,
    _ball_add,
    _ball_divide,
    _ball_from_fraction,
    _ball_integer_scale,
    _ball_multiply,
    _ball_negate,
    _ball_subtract,
    _binary_scaled_divide,
    _binary_scaled_from_ball,
    _binary_scaled_multiply,
    _binary_scaled_negate,
    _binary_scaled_sum,
    _BinaryScaledBall,
    _dd_add,
    _dd_difference,
    _dd_divide,
    _dd_divide_float,
    _dd_multiply,
    _dd_multiply_float,
    _dd_negate,
    _dd_sum,
    _DoubleDouble,
    _exact_sum_ratios_scalar,
    _fraction,
    _fraction_float_neighbors,
    _ScaledEnclosure,
    _scaled_signed_enclosure,
    _stable_affine_residual,
    _stable_product,
    _stable_product_scalar,
    _stable_ratio_difference,
    _stable_ratio_product,
    _stable_scaled_difference,
    _stable_scaled_difference_scalar,
    _stable_sum,
    _stable_sum_products,
    _stable_sum_products_sign,
    _stable_sum_scalar,
    _TwofoldBall,
    _up_multiply_nonnegative,
)
from .model import (
    ExponentialBoundaryPenalty,
    HuberLoss,
    ReciprocalBoundaryPenalty,
    ScalarPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)


HARD_ATOL = 1e-12
HARD_RTOL = 64.0 * np.finfo(np.float64).eps
_FLOAT_EPSILON = np.finfo(np.float64).eps
_DD_FAST_MIN = math.ldexp(1.0, -969)
_FLOAT_MAX_FRACTION = _fraction(np.finfo(np.float64).max)


@dataclass(frozen=True, slots=True)
class _QuadraticRowData:
    """Scale-safe row curvature, normal RHS, and diagnostic target."""

    rho: np.ndarray
    rhs: np.ndarray
    z_obs: np.ndarray


@dataclass(frozen=True, slots=True)
class _CompiledLocation:
    """An exact dyadic location and its numeric binary64 neighbors."""

    exact: Fraction
    below: float
    above: float


@dataclass(frozen=True, slots=True)
class _CompiledScalarPenalty:
    """Static exact data for one positive-strength scalar penalty."""

    kind: str
    lower: Fraction
    upper: Fraction
    strength: Fraction
    margin: Fraction | None = None
    epsilon: Fraction | None = None
    tau: Fraction | None = None
    lower_value: float = 0.0
    upper_value: float = 0.0
    strength_value: float = 0.0
    margin_value: float | None = None
    epsilon_value: float | None = None
    tau_value: float | None = None
    lower_epsilon: _CompiledLocation | None = None
    lower_margin: _CompiledLocation | None = None
    upper_margin: _CompiledLocation | None = None
    upper_epsilon: _CompiledLocation | None = None


@dataclass(frozen=True, slots=True)
class _CompiledScalarObjective:
    """Static scalar objective data shared by all proximal coordinates."""

    mismatch_kind: str
    huber_delta: Fraction | None
    penalties: tuple[_CompiledScalarPenalty, ...]
    breakpoints: tuple[Fraction, ...]
    huber_delta_value: float | None = None


@dataclass(frozen=True, slots=True)
class _KernelDerivativeEnclosures:
    """Ordinary-path one-sided derivative and curvature enclosures."""

    minus: _ScaledEnclosure
    plus: _ScaledEnclosure
    curvature: _ScaledEnclosure
    smooth: bool


@dataclass(frozen=True, slots=True)
class _ArrayKernelDerivativeEnclosures:
    """Vectorized moderate-row derivative balls and smoothness masks."""

    derivative: _ArrayBall
    curvature: _ArrayBall
    smooth: np.ndarray


@dataclass(frozen=True, slots=True)
class _ExactExponentialExpression:
    """Exact rational terms plus collected ``coefficient * exp(exponent)``."""

    rational: Fraction
    exponentials: tuple[tuple[Fraction, Fraction], ...]

    @property
    def symbolic_zero(self) -> bool:
        return self.rational == 0 and not self.exponentials

    @property
    def algebraic_sign(self) -> int | None:
        if self.exponentials:
            return None
        return int(self.rational > 0) - int(self.rational < 0)


@dataclass(frozen=True, slots=True)
class _DecimalInterval:
    """Outward Decimal enclosure used only by bounded fallback decisions."""

    lower: Decimal
    upper: Decimal

    @property
    def sign(self) -> int | None:
        if self.lower > 0:
            return 1
        if self.upper < 0:
            return -1
        if self.lower == 0 and self.upper == 0:
            return 0
        return None


def _require_finite_scalar(value: float, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f'{name} must be finite')
    return scalar


def _compile_location(value: Fraction) -> _CompiledLocation:
    below, above = _fraction_float_neighbors(value)
    return _CompiledLocation(exact=value, below=below, above=above)


def _compare_compiled_location(value: float, location: _CompiledLocation) -> int:
    """Compare one binary64 value with a precompiled exact dyadic location."""

    scalar = float(value)
    if location.below == location.above:
        return int(scalar > location.below) - int(scalar < location.below)
    if scalar <= location.below:
        return -1
    if scalar >= location.above:
        return 1
    raise ArithmeticError('no binary64 value lies inside exact location neighbors')


def _compile_scalar_objective(
    mismatch: SquaredLoss | HuberLoss,
    penalties: tuple[ScalarPenalty, ...],
) -> _CompiledScalarObjective:
    """Compile exact branch data without retaining zero-strength terms."""

    breakpoints: set[Fraction] = set()
    if isinstance(mismatch, SquaredLoss):
        mismatch_kind = 'squared'
        huber_delta = None
        huber_delta_value = None
    elif isinstance(mismatch, HuberLoss):
        mismatch_kind = 'huber'
        delta = _require_finite_scalar(
            mismatch.delta,
            name='HuberLoss.delta',
        )
        if delta <= 0.0:
            raise ValueError('HuberLoss.delta must be > 0')
        huber_delta = _fraction(delta)
        huber_delta_value = delta
    else:
        raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')

    compiled: list[_CompiledScalarPenalty] = []
    for penalty in penalties:
        strength_value = float(penalty.strength)
        if strength_value == 0.0:
            continue
        strength_value = _require_finite_scalar(
            strength_value,
            name=f'{type(penalty).__name__}.strength',
        )
        if strength_value < 0.0:
            raise ValueError('scalar penalty strength must be >= 0')
        strength = _fraction(strength_value)
        lower = _fraction(
            _require_finite_scalar(
                penalty.lower,
                name=f'{type(penalty).__name__}.lower',
            )
        )
        upper = _fraction(
            _require_finite_scalar(
                penalty.upper,
                name=f'{type(penalty).__name__}.upper',
            )
        )
        if isinstance(penalty, SoftIntervalPenalty):
            row = _CompiledScalarPenalty(
                kind='soft',
                lower=lower,
                upper=upper,
                strength=strength,
                lower_value=float(penalty.lower),
                upper_value=float(penalty.upper),
                strength_value=strength_value,
            )
            breakpoints.update((lower, upper))
        elif isinstance(penalty, ExponentialBoundaryPenalty):
            margin = _fraction(
                _require_finite_scalar(
                    penalty.margin,
                    name='ExponentialBoundaryPenalty.margin',
                )
            )
            tau = _fraction(
                _require_finite_scalar(
                    penalty.tau,
                    name='ExponentialBoundaryPenalty.tau',
                )
            )
            row = _CompiledScalarPenalty(
                kind='exponential',
                lower=lower,
                upper=upper,
                strength=strength,
                margin=margin,
                tau=tau,
                lower_value=float(penalty.lower),
                upper_value=float(penalty.upper),
                strength_value=strength_value,
                margin_value=float(penalty.margin),
                tau_value=float(penalty.tau),
            )
        elif isinstance(penalty, ReciprocalBoundaryPenalty):
            margin = _fraction(
                _require_finite_scalar(
                    penalty.margin,
                    name='ReciprocalBoundaryPenalty.margin',
                )
            )
            epsilon = _fraction(
                _require_finite_scalar(
                    penalty.epsilon,
                    name='ReciprocalBoundaryPenalty.epsilon',
                )
            )
            row = _CompiledScalarPenalty(
                kind='reciprocal',
                lower=lower,
                upper=upper,
                strength=strength,
                margin=margin,
                epsilon=epsilon,
                lower_value=float(penalty.lower),
                upper_value=float(penalty.upper),
                strength_value=strength_value,
                margin_value=float(penalty.margin),
                epsilon_value=float(penalty.epsilon),
                lower_epsilon=_compile_location(lower + epsilon),
                lower_margin=_compile_location(lower + margin),
                upper_margin=_compile_location(upper - margin),
                upper_epsilon=_compile_location(upper - epsilon),
            )
            breakpoints.update(
                (
                    lower + epsilon,
                    lower + margin,
                    upper - margin,
                    upper - epsilon,
                )
            )
        else:
            raise TypeError(f'unsupported penalty: {type(penalty)!r}')
        compiled.append(row)

    return _CompiledScalarObjective(
        mismatch_kind=mismatch_kind,
        huber_delta=huber_delta,
        penalties=tuple(compiled),
        breakpoints=tuple(sorted(breakpoints)),
        huber_delta_value=huber_delta_value,
    )


def _huber_difference_branches(
    measurement: np.ndarray,
    target: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify ``measurement - target`` against exact ``+/- delta``."""

    lower_sign = _stable_sum_products_sign(
        (
            (measurement,),
            (-1.0, target),
            (delta,),
        )
    )
    upper_sign = _stable_sum_products_sign(
        (
            (measurement,),
            (-1.0, target),
            (-delta,),
        )
    )
    quadratic = (lower_sign >= 0) & (upper_sign <= 0)
    direction = np.where(upper_sign > 0, 1.0, -1.0)
    return quadratic, direction


def _huber_affine_branches(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a complete affine residual against exact ``+/- delta``."""

    common = (
        (beta,),
        (alpha, left),
        (-1.0, alpha, right),
        (-1.0, target),
    )
    lower_sign = _stable_sum_products_sign(common + ((delta,),))
    upper_sign = _stable_sum_products_sign(common + ((-delta,),))
    quadratic = (lower_sign >= 0) & (upper_sign <= 0)
    direction = np.where(upper_sign > 0, 1.0, -1.0)
    return quadratic, direction


# The two dyadic limbs and residual below were generated independently from
# ln(2) = 2 * atanh(1/3).  The generator sums 161 exact rational series terms
# and bounds the remaining geometric tail by 9/8 of the first omitted term.
# The resulting radius is below 2**-110, so range reduction carries more than
# the required 100 effective bits of containment.
_LN2_BALL = _TwofoldBall(
    float.fromhex('0x1.62e42fefa39efp-1'),
    float.fromhex('0x1.abc9e3b39803fp-56'),
    float.fromhex('0x1.7b57a079a1934p-111'),
    True,
)
_EXP_COEFFICIENT_BALLS = tuple(
    _ball_from_fraction(Fraction(1, math.factorial(degree)))
    for degree in range(21)
)
_EXP_REMAINDER_BOUND = float.fromhex('0x1.3d31d3241e942p-75')
_EXP_REMAINDER_COEFFICIENT = float.fromhex('0x1.154ab3925b816p-64')


def _source_sum_ball(*terms: _TwofoldBall | float) -> _TwofoldBall:
    """Evaluate a complete source sum without rounding a subexpression."""

    total = _TwofoldBall.point(0.0)
    for term in terms:
        value = term if isinstance(term, _TwofoldBall) else _TwofoldBall.point(term)
        total = _ball_add(total, value)
    return total


def _source_product_ball(*factors: _TwofoldBall | float) -> _TwofoldBall:
    """Evaluate a complete source product with operation-level radii."""

    value = _TwofoldBall.point(1.0)
    for factor in factors:
        operand = (
            factor
            if isinstance(factor, _TwofoldBall)
            else _TwofoldBall.point(factor)
        )
        value = _ball_multiply(value, operand)
    return value


def _scaled_source_product(
    *factors: _TwofoldBall | float,
) -> _BinaryScaledBall:
    """Multiply source factors without materializing range-risking products."""

    if not factors:
        return _binary_scaled_from_ball(_TwofoldBall.point(1.0))

    def scaled_factor(factor: _TwofoldBall | float) -> _BinaryScaledBall:
        operand = (
            factor
            if isinstance(factor, _TwofoldBall)
            else _TwofoldBall.point(factor)
        )
        return _binary_scaled_from_ball(operand)

    value = scaled_factor(factors[0])
    for factor in factors[1:]:
        value = _binary_scaled_multiply(
            value,
            scaled_factor(factor),
        )
    return value


def _scaled_source_sum(
    *terms: _TwofoldBall | float,
) -> _BinaryScaledBall:
    return _binary_scaled_sum(tuple(
        _binary_scaled_from_ball(
            term
            if isinstance(term, _TwofoldBall)
            else _TwofoldBall.point(term)
        )
        for term in terms
    ))


def _scaled_source_ratio(
    numerator_factors: tuple[_TwofoldBall | float, ...],
    denominator_factors: tuple[_TwofoldBall | float, ...],
) -> _BinaryScaledBall:
    return _binary_scaled_divide(
        _scaled_source_product(*numerator_factors),
        _scaled_source_product(*denominator_factors),
    )


def _with_analytic_remainder(
    value: _TwofoldBall,
    argument: _TwofoldBall,
) -> _TwofoldBall:
    if not value.resolved:
        return _TwofoldBall.unresolved()
    lower, upper = argument.physical_bounds()
    magnitude = max(abs(lower), abs(upper))
    power = 1.0
    for _ in range(21):
        power = _up_multiply_nonnegative(power, magnitude)
    remainder = min(
        _EXP_REMAINDER_BOUND,
        _up_multiply_nonnegative(power, _EXP_REMAINDER_COEFFICIENT),
    )
    return _ball_add(
        value,
        _TwofoldBall(0.0, 0.0, remainder, True),
    )


def _exp_polynomial_ball(argument: _TwofoldBall) -> _TwofoldBall:
    """Enclose degree-20 Taylor ``exp`` on ``[-0.7, 0.7]``."""

    lower, upper = argument.physical_bounds()
    if lower < -0.7 or upper > 0.7:
        return _TwofoldBall.unresolved()
    value = _EXP_COEFFICIENT_BALLS[20]
    for degree in range(19, -1, -1):
        value = _ball_add(
            _ball_multiply(value, argument),
            _EXP_COEFFICIENT_BALLS[degree],
        )
    return _with_analytic_remainder(value, argument)


def _expm1_polynomial_ball(argument: _TwofoldBall) -> _TwofoldBall:
    """Enclose degree-20 Taylor ``expm1`` on ``[-0.7, 0.7]``."""

    lower, upper = argument.physical_bounds()
    if lower < -0.7 or upper > 0.7:
        return _TwofoldBall.unresolved()
    value = _EXP_COEFFICIENT_BALLS[20]
    for degree in range(19, 0, -1):
        value = _ball_add(
            _ball_multiply(value, argument),
            _EXP_COEFFICIENT_BALLS[degree],
        )
    return _with_analytic_remainder(
        _ball_multiply(argument, value),
        argument,
    )


def _certified_exp_ball(argument: _TwofoldBall) -> _BinaryScaledBall:
    """Enclose ``exp(argument)`` without relying on a libm ulp claim."""

    if not argument.resolved:
        return _BinaryScaledBall.unresolved()
    center = argument.center_value
    ln2_center = _LN2_BALL.center_value
    quotient = center / ln2_center
    if not math.isfinite(quotient) or abs(quotient) > 2**53:
        return _BinaryScaledBall.unresolved()
    proposal = int(round(quotient))
    reduced = _ball_subtract(
        argument,
        _ball_integer_scale(_LN2_BALL, proposal),
    )
    reduced_bounds = reduced.physical_bounds()
    if reduced_bounds[0] < -0.7 or reduced_bounds[1] > 0.7:
        adjusted = None
        for candidate in (proposal - 1, proposal + 1):
            trial = _ball_subtract(
                argument,
                _ball_integer_scale(_LN2_BALL, candidate),
            )
            trial_bounds = trial.physical_bounds()
            if trial_bounds[0] >= -0.7 and trial_bounds[1] <= 0.7:
                proposal = candidate
                adjusted = trial
                break
        if adjusted is None:
            return _BinaryScaledBall.unresolved()
        reduced = adjusted
    polynomial = _exp_polynomial_ball(reduced)
    if not polynomial.resolved:
        return _BinaryScaledBall.unresolved()
    return _BinaryScaledBall(polynomial, proposal)


def _certified_exp_difference(
    left_argument: _TwofoldBall,
    right_argument: _TwofoldBall,
    *,
    difference: _TwofoldBall | None = None,
    left_exponential: _BinaryScaledBall | None = None,
    right_exponential: _BinaryScaledBall | None = None,
) -> _BinaryScaledBall:
    """Enclose ``exp(right_argument) - exp(left_argument)``.

    When ordering is proved and separation is at most 0.7, the balanced
    ``exp(a) * expm1(b-a)`` form prevents large common terms from being
    subtracted.  Wider or overlapping arguments use a conservative base-two
    scaled difference.
    """

    if difference is None:
        difference = _ball_subtract(right_argument, left_argument)
    difference_lower, difference_upper = difference.physical_bounds()
    if difference_lower > 0.0 and difference_upper <= 0.7:
        return _binary_scaled_multiply(
            (
                left_exponential
                if left_exponential is not None
                else _certified_exp_ball(left_argument)
            ),
            _binary_scaled_from_ball(_expm1_polynomial_ball(difference)),
        )
    if difference_upper < 0.0 and difference_lower >= -0.7:
        return _binary_scaled_negate(
            _binary_scaled_multiply(
                (
                    right_exponential
                    if right_exponential is not None
                    else _certified_exp_ball(right_argument)
                ),
                _binary_scaled_from_ball(
                    _expm1_polynomial_ball(_ball_negate(difference))
                ),
            )
        )
    return _binary_scaled_sum(
        (
            (
                right_exponential
                if right_exponential is not None
                else _certified_exp_ball(right_argument)
            ),
            _binary_scaled_negate(
                left_exponential
                if left_exponential is not None
                else _certified_exp_ball(left_argument)
            ),
        )
    )


def _compiled_exponential_arguments(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> tuple[_TwofoldBall, _TwofoldBall]:
    """Build both complete ADR 0007 exponential source numerators."""

    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    tau = _TwofoldBall.point(penalty.tau_value)
    lower_numerator = _source_sum_ball(
        penalty.lower_value,
        penalty.margin_value,
        -float(y),
    )
    upper_numerator = _source_sum_ball(
        float(y),
        -penalty.upper_value,
        penalty.margin_value,
    )
    return (
        _ball_divide(lower_numerator, tau),
        _ball_divide(upper_numerator, tau),
    )


def _array_ball_constant(
    value: _TwofoldBall,
    shape: tuple[int, ...],
) -> _ArrayBall:
    return _ArrayBall(
        np.full(shape, value.high, dtype=np.float64),
        np.full(shape, value.low, dtype=np.float64),
        np.full(shape, value.radius, dtype=np.float64),
        np.full(shape, value.resolved, dtype=bool),
    )


def _array_source_sum(*terms: _ArrayBall | np.ndarray | float) -> _ArrayBall:
    shape = next(
        (
            term.high.shape
            if isinstance(term, _ArrayBall)
            else np.asarray(term).shape
            for term in terms
            if isinstance(term, _ArrayBall) or np.asarray(term).shape
        ),
        (),
    )
    total = _ArrayBall.points(np.zeros(shape, dtype=np.float64))
    for term in terms:
        operand = term if isinstance(term, _ArrayBall) else _ArrayBall.points(
            np.broadcast_to(np.asarray(term, dtype=np.float64), shape)
        )
        total = _array_ball_add(total, operand)
    return total


def _array_source_product(
    *factors: _ArrayBall | np.ndarray | float,
) -> _ArrayBall:
    shape = next(
        (
            factor.high.shape
            if isinstance(factor, _ArrayBall)
            else np.asarray(factor).shape
            for factor in factors
            if isinstance(factor, _ArrayBall) or np.asarray(factor).shape
        ),
        (),
    )
    value = _ArrayBall.points(np.ones(shape, dtype=np.float64))
    for factor in factors:
        operand = (
            factor
            if isinstance(factor, _ArrayBall)
            else _ArrayBall.points(
                np.broadcast_to(np.asarray(factor, dtype=np.float64), shape)
            )
        )
        value = _array_ball_multiply(value, operand)
    return value


def _array_certified_exp(argument: _ArrayBall) -> _ArrayBall:
    """Vectorized physical exp balls for moderate heterogeneous rows."""

    shape = argument.high.shape
    ln2 = _array_ball_constant(_LN2_BALL, shape)
    center = argument.high + argument.low
    proposal = np.rint(center / _LN2_BALL.center_value)
    proposal_valid = np.isfinite(proposal) & (np.abs(proposal) <= 2**31 - 1)
    safe_proposal = np.where(proposal_valid, proposal, 0.0)
    integer = _ArrayBall.points(safe_proposal)
    reduced = _array_ball_subtract(
        argument,
        _array_ball_multiply(ln2, integer),
    )
    reduced_lower, reduced_upper = reduced.physical_bounds()
    resolved = (
        argument.resolved
        & proposal_valid
        & reduced.resolved
        & (reduced_lower >= -0.7)
        & (reduced_upper <= 0.7)
    )
    polynomial = _array_ball_constant(_EXP_COEFFICIENT_BALLS[20], shape)
    for degree in range(19, -1, -1):
        polynomial = _array_ball_add(
            _array_ball_multiply(polynomial, reduced),
            _array_ball_constant(_EXP_COEFFICIENT_BALLS[degree], shape),
        )
    magnitude = np.maximum(np.abs(reduced_lower), np.abs(reduced_upper))
    remainder = np.ones_like(magnitude)
    for _ in range(21):
        remainder = np.nextafter(remainder * magnitude, np.inf)
    remainder = np.minimum(
        _EXP_REMAINDER_BOUND,
        np.nextafter(remainder * _EXP_REMAINDER_COEFFICIENT, np.inf),
    )
    polynomial = _ArrayBall(
        polynomial.high,
        polynomial.low,
        np.nextafter(polynomial.radius + remainder, np.inf),
        polynomial.resolved,
    )
    exponent = np.asarray(safe_proposal, dtype=np.intc)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        high = np.ldexp(polynomial.high, exponent)
        low = np.ldexp(polynomial.low, exponent)
        radius = np.ldexp(polynomial.radius, exponent)
    subnormal_low = (
        (polynomial.low != 0.0)
        & (np.abs(low) < np.finfo(np.float64).tiny)
    )
    subnormal_radius = (
        (polynomial.radius != 0.0)
        & (radius < np.finfo(np.float64).tiny)
    )
    scaling_radius = np.where(subnormal_low, math.ulp(0.0), 0.0)
    radius = _array_up_add(radius, scaling_radius)
    radius = np.where(
        subnormal_radius,
        np.nextafter(radius, np.inf),
        radius,
    )
    physical_resolved = (
        resolved
        & polynomial.resolved
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(radius)
        & (high != 0.0)
        & (np.abs(high) >= _DD_FAST_MIN)
    )
    return _ArrayBall(high, low, radius, physical_resolved)


def _array_compiled_exponential_arguments(
    penalty: _CompiledScalarPenalty,
    y: np.ndarray,
) -> tuple[_ArrayBall, _ArrayBall]:
    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    shape = y.shape
    tau = _ArrayBall.points(np.full(shape, penalty.tau_value))
    lower_numerator = _array_source_sum(
        penalty.lower_value,
        penalty.margin_value,
        -y,
    )
    upper_numerator = _array_source_sum(
        y,
        -penalty.upper_value,
        penalty.margin_value,
    )
    return (
        _array_ball_divide(lower_numerator, tau),
        _array_ball_divide(upper_numerator, tau),
    )


def _active_scalar_penalties(
    penalties: tuple[ScalarPenalty, ...],
) -> tuple[ScalarPenalty, ...]:
    """Return scalar penalties that are not exact zero-strength no-ops."""

    return tuple(
        penalty
        for penalty in penalties
        if float(penalty.strength) != 0.0
    )


def _dd_enclosure_terms(
    parts: list[_DoubleDouble],
) -> list[tuple[int, float, float]]:
    """Convert algebraic expansion parts to one scaled signed term."""

    if not parts:
        return []
    if any(
        not math.isfinite(part.high) or not math.isfinite(part.low)
        for part in parts
    ):
        # The short expansion has left its representable range.  Never use a
        # raw ``inf - inf`` sign; force the bounded exact-input fallback.
        return [(0, 0.0, math.inf)]
    try:
        total = _dd_sum(parts).value
        l1 = math.fsum(
            abs(part.high) + abs(part.low)
            for part in parts
        )
    except OverflowError:
        return [(0, 0.0, math.inf)]
    if not math.isfinite(l1):
        return [(0, 0.0, math.inf)]
    absolute_error = 64.0 * (_FLOAT_EPSILON ** 2) * l1
    if math.isfinite(total):
        absolute_error += 4.0 * math.ulp(total)
    if total == 0.0:
        if absolute_error == 0.0:
            return []
        return [(0, math.log(absolute_error), 1.0)]
    if not math.isfinite(total):
        return [(1 if total > 0.0 else -1, math.inf, math.inf)]
    relative_error = absolute_error / abs(total)
    return [(
        1 if total > 0.0 else -1,
        math.log(abs(total)),
        relative_error,
    )]


_LN2_HIGH = 0.6931471805599453
_LN2_LOW = 2.3190468138462996e-17
_EXPANSION_RELATIVE_ERROR = 256.0 * (_FLOAT_EPSILON ** 2)


def _exp_double_double(value: _DoubleDouble) -> _DoubleDouble:
    """Evaluate a moderate exponential as a short binary64 expansion."""

    scalar = value.value
    if scalar < -750.0:
        return _DoubleDouble(0.0)
    if scalar > 700.0:
        return _DoubleDouble(math.inf)
    exponent = int(round(scalar / _LN2_HIGH))
    reduced = _dd_sum(
        (
            value,
            _DoubleDouble(-exponent * _LN2_HIGH),
            _DoubleDouble(-exponent * _LN2_LOW),
        )
    )
    total = _DoubleDouble(1.0)
    term = _DoubleDouble(1.0)
    for order in range(1, 26):
        term = _dd_divide_float(_dd_multiply(term, reduced), float(order))
        total = _dd_add(total, term)
    try:
        return _DoubleDouble(
            math.ldexp(total.high, exponent),
            math.ldexp(total.low, exponent),
        )
    except OverflowError:
        return _DoubleDouble(math.inf)


def _expm1_double_double(value: _DoubleDouble) -> _DoubleDouble:
    """Evaluate ``exp(value) - 1`` without subtractive cancellation."""

    scalar = value.value
    if abs(scalar) > 0.5:
        return _dd_add(_exp_double_double(value), _DoubleDouble(-1.0))
    total = value
    term = value
    for order in range(2, 20):
        term = _dd_divide_float(_dd_multiply(term, value), float(order))
        total = _dd_add(total, term)
    return total


def _exponential_derivative_expansions(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> tuple[_DoubleDouble, _DoubleDouble, float, float] | None:
    """Return moderate derivative/curvature expansions and error logs."""

    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    left_exponent = _dd_divide_float(
        _dd_add(
            _dd_difference(penalty.lower_value, y),
            _DoubleDouble(penalty.margin_value),
        ),
        penalty.tau_value,
    )
    right_exponent = _dd_divide_float(
        _dd_add(
            _dd_difference(y, penalty.upper_value),
            _DoubleDouble(penalty.margin_value),
        ),
        penalty.tau_value,
    )
    if not (-700.0 <= left_exponent.value <= 700.0):
        return None
    if not (-700.0 <= right_exponent.value <= 700.0):
        return None
    left_value = _exp_double_double(left_exponent)
    right_value = _exp_double_double(right_exponent)
    coefficient = _dd_divide_float(
        _DoubleDouble(penalty.strength_value),
        penalty.tau_value,
    )
    derivative = _dd_multiply(
        coefficient,
        _dd_add(right_value, _dd_negate(left_value)),
    )
    curvature_coefficient = _dd_divide_float(
        coefficient,
        penalty.tau_value,
    )
    curvature = _dd_multiply(
        curvature_coefficient,
        _dd_add(right_value, left_value),
    )
    if not (
        math.isfinite(derivative.high)
        and math.isfinite(curvature.high)
    ):
        return None
    maximum = max(left_exponent.value, right_exponent.value)
    balance = math.log1p(
        math.exp(-abs(right_exponent.value - left_exponent.value))
    )
    derivative_error_log = (
        math.log(penalty.strength_value)
        - math.log(penalty.tau_value)
        + maximum
        + balance
        + math.log(_EXPANSION_RELATIVE_ERROR)
    )
    curvature_error_log = derivative_error_log - math.log(
        penalty.tau_value
    )
    return (
        derivative,
        curvature,
        derivative_error_log,
        curvature_error_log,
    )


def _exponential_derivative_logs(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]:
    """Return balanced derivative and curvature log terms for one exponential."""

    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    left_numerator = _dd_add(
        _dd_difference(penalty.lower_value, y),
        _DoubleDouble(penalty.margin_value),
    )
    right_numerator = _dd_add(
        _dd_difference(y, penalty.upper_value),
        _DoubleDouble(penalty.margin_value),
    )
    left_exponent = _dd_divide_float(
        left_numerator,
        penalty.tau_value,
    )
    right_exponent = _dd_divide_float(
        right_numerator,
        penalty.tau_value,
    )
    left = left_exponent.value
    right = right_exponent.value
    if not math.isfinite(left) or not math.isfinite(right):
        return ([(0, 0.0, math.inf)], [(0, 0.0, math.inf)])
    difference = _dd_add(
        right_exponent,
        _dd_negate(left_exponent),
    ).value
    maximum = max(left, right)
    log_strength = math.log(penalty.strength_value)
    log_tau = math.log(penalty.tau_value)
    derivative: list[tuple[int, float, float]] = []
    if difference != 0.0:
        balanced = -math.expm1(-abs(difference))
        if balanced > 0.0:
            derivative.append(
                (
                    1 if difference > 0.0 else -1,
                    log_strength - log_tau + maximum + math.log(balanced),
                    32.0 * _FLOAT_EPSILON,
                )
            )
    curvature_balance = math.log1p(math.exp(-abs(difference)))
    curvature = [(
        1,
        log_strength - 2.0 * log_tau + maximum + curvature_balance,
        32.0 * _FLOAT_EPSILON,
    )]
    return derivative, curvature


def _compiled_exponential_derivative_values(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> tuple[float, float]:
    """Return central derivative values from the shared compiled term kernel."""

    expanded = _exponential_derivative_expansions(penalty, y)
    if expanded is not None:
        return expanded[0].value, expanded[1].value
    derivative_logs, curvature_logs = _exponential_derivative_logs(penalty, y)

    def central(terms: list[tuple[int, float, float]]) -> float:
        enclosure = _scaled_signed_enclosure(tuple(terms))
        lower, upper = enclosure.physical_bounds()
        if math.isfinite(lower) and math.isfinite(upper):
            return 0.5 * lower + 0.5 * upper
        if lower >= 0.0:
            return math.inf
        if upper <= 0.0:
            return -math.inf
        return math.nan

    return central(derivative_logs), central(curvature_logs)


def _legacy_scalar_derivative_terms(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _KernelDerivativeEnclosures:
    """Evaluate ordinary one-sided derivative enclosures in binary64.

    Algebraic terms retain their division/product corrections in short
    expansions before cancellation.  Exponential pairs use a balanced
    ``expm1`` form and join the algebraic result only in a common log scale.
    """

    minus_parts: list[_DoubleDouble] = []
    plus_parts: list[_DoubleDouble] = []
    curvature_parts: list[_DoubleDouble] = []
    minus_logs: list[tuple[int, float, float]] = []
    plus_logs: list[tuple[int, float, float]] = []
    curvature_logs: list[tuple[int, float, float]] = []
    smooth = True

    def append_common(value: _DoubleDouble) -> None:
        minus_parts.append(value)
        plus_parts.append(value)

    def append_common_product(*factors: _DoubleDouble) -> None:
        value = _normalized_product_dd(tuple(factors))
        if value is not None:
            append_common(value)
            return
        term = _normalized_product_log_term(tuple(factors))
        minus_logs.append(term)
        plus_logs.append(term)

    def append_curvature_product(*factors: _DoubleDouble) -> None:
        value = _normalized_product_dd(tuple(factors))
        if value is not None:
            curvature_parts.append(value)
            return
        curvature_logs.append(_normalized_product_log_term(tuple(factors)))

    def append_ratio_term(
        part_destinations: tuple[list[_DoubleDouble], ...],
        log_destinations: tuple[list[tuple[int, float, float]], ...],
        denominator: _DoubleDouble,
        denominator_power: int,
        *numerator_factors: _DoubleDouble,
    ) -> None:
        value = _normalized_ratio_power_dd(
            tuple(numerator_factors),
            denominator,
            denominator_power,
        )
        if value is not None:
            for destination in part_destinations:
                destination.append(value)
            return
        term = _normalized_ratio_power_log_term(
            tuple(numerator_factors),
            denominator,
            denominator_power,
        )
        for destination in log_destinations:
            destination.append(term)

    residual = _dd_difference(y, target)
    if confidence != 0.0:
        if spec.mismatch_kind == 'squared':
            append_common_product(_DoubleDouble(confidence), residual)
            curvature_parts.append(_DoubleDouble(confidence))
        else:
            assert spec.huber_delta_value is not None
            delta = spec.huber_delta_value
            below = int(_stable_sum_products_sign(
                ((y,), (-1.0, target), (delta,))
            ))
            above = int(_stable_sum_products_sign(
                ((y,), (-1.0, target), (-delta,))
            ))
            if below < 0:
                append_common_product(
                    _DoubleDouble(-1.0),
                    _DoubleDouble(confidence),
                    _DoubleDouble(delta),
                )
            elif above > 0:
                append_common_product(
                    _DoubleDouble(confidence),
                    _DoubleDouble(delta),
                )
            else:
                append_common_product(_DoubleDouble(confidence), residual)
                curvature_parts.append(_DoubleDouble(confidence))
            if below == 0 or above == 0:
                smooth = False

    append_common_product(_DoubleDouble(rho), _dd_difference(y, v))
    curvature_parts.append(_DoubleDouble(rho))

    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            if y < penalty.lower_value:
                append_common_product(
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                    _dd_difference(y, penalty.lower_value),
                )
                append_curvature_product(
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                )
            elif y > penalty.upper_value:
                append_common_product(
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                    _dd_difference(y, penalty.upper_value),
                )
                append_curvature_product(
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                )
            if y == penalty.lower_value or y == penalty.upper_value:
                smooth = False
            continue

        if penalty.kind == 'exponential':
            expanded = _exponential_derivative_expansions(penalty, y)
            if expanded is None:
                derivative, curvature = _exponential_derivative_logs(penalty, y)
                minus_logs.extend(derivative)
                plus_logs.extend(derivative)
                curvature_logs.extend(curvature)
            else:
                append_common(expanded[0])
                curvature_parts.append(expanded[1])
                error = (0, expanded[2], 1.0)
                minus_logs.append(error)
                plus_logs.append(error)
                curvature_logs.append((0, expanded[3], 1.0))
            continue

        assert penalty.lower_epsilon is not None
        assert penalty.lower_margin is not None
        assert penalty.upper_margin is not None
        assert penalty.upper_epsilon is not None
        assert penalty.margin_value is not None
        assert penalty.epsilon_value is not None

        lower_margin_cmp = _compare_compiled_location(y, penalty.lower_margin)
        if lower_margin_cmp == 0:
            append_ratio_term(
                (minus_parts,),
                (minus_logs,),
                _DoubleDouble(penalty.margin_value),
                2,
                _DoubleDouble(-1.0),
                _DoubleDouble(penalty.strength_value),
            )
            smooth = False
        elif lower_margin_cmp < 0:
            lower_epsilon_cmp = _compare_compiled_location(
                y,
                penalty.lower_epsilon,
            )
            denominator = (
                _DoubleDouble(penalty.epsilon_value)
                if lower_epsilon_cmp <= 0
                else _dd_difference(y, penalty.lower_value)
            )
            append_ratio_term(
                (minus_parts, plus_parts),
                (minus_logs, plus_logs),
                denominator,
                2,
                _DoubleDouble(-1.0),
                _DoubleDouble(penalty.strength_value),
            )
            if lower_epsilon_cmp > 0:
                append_ratio_term(
                    (curvature_parts,),
                    (curvature_logs,),
                    denominator,
                    3,
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                )
            elif lower_epsilon_cmp == 0:
                smooth = False

        upper_margin_cmp = _compare_compiled_location(y, penalty.upper_margin)
        if upper_margin_cmp == 0:
            append_ratio_term(
                (plus_parts,),
                (plus_logs,),
                _DoubleDouble(penalty.margin_value),
                2,
                _DoubleDouble(penalty.strength_value),
            )
            smooth = False
        elif upper_margin_cmp > 0:
            upper_epsilon_cmp = _compare_compiled_location(
                y,
                penalty.upper_epsilon,
            )
            denominator = (
                _DoubleDouble(penalty.epsilon_value)
                if upper_epsilon_cmp >= 0
                else _dd_difference(penalty.upper_value, y)
            )
            append_ratio_term(
                (minus_parts, plus_parts),
                (minus_logs, plus_logs),
                denominator,
                2,
                _DoubleDouble(penalty.strength_value),
            )
            if upper_epsilon_cmp < 0:
                append_ratio_term(
                    (curvature_parts,),
                    (curvature_logs,),
                    denominator,
                    3,
                    _DoubleDouble(2.0),
                    _DoubleDouble(penalty.strength_value),
                )
            elif upper_epsilon_cmp == 0:
                smooth = False

    minus_logs.extend(_dd_enclosure_terms(minus_parts))
    plus_logs.extend(_dd_enclosure_terms(plus_parts))
    curvature_logs.extend(_dd_enclosure_terms(curvature_parts))
    return _KernelDerivativeEnclosures(
        minus=_scaled_signed_enclosure(tuple(minus_logs)),
        plus=_scaled_signed_enclosure(tuple(plus_logs)),
        curvature=_scaled_signed_enclosure(tuple(curvature_logs)),
        smooth=smooth,
    )


def _scalar_derivative_terms(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _KernelDerivativeEnclosures:
    """Enclose both one-sided derivatives and curvature with balls.

    All algebraic source operands enter as exact point balls.  Exponentials
    use the dependency-free polynomial kernel and every signed contribution is
    accumulated in a common binary scale.  An unsupported operation returns
    an unresolved scaled ball, which the scalar solver routes to its bounded
    exact fallback before any sign is consumed.
    """

    minus: list[_BinaryScaledBall] = []
    plus: list[_BinaryScaledBall] = []
    curvature: list[_BinaryScaledBall] = []
    smooth = True

    def scaled(value: _TwofoldBall) -> _BinaryScaledBall:
        return _binary_scaled_from_ball(value)

    def append_common(value: _TwofoldBall | _BinaryScaledBall) -> None:
        term = value if isinstance(value, _BinaryScaledBall) else scaled(value)
        minus.append(term)
        plus.append(term)

    residual = _source_sum_ball(y, -target)
    if confidence != 0.0:
        if spec.mismatch_kind == 'squared':
            append_common(_scaled_source_product(confidence, residual))
            curvature.append(scaled(_TwofoldBall.point(confidence)))
        else:
            assert spec.huber_delta_value is not None
            delta = spec.huber_delta_value
            below = int(_stable_sum_products_sign(
                ((y,), (-1.0, target), (delta,))
            ))
            above = int(_stable_sum_products_sign(
                ((y,), (-1.0, target), (-delta,))
            ))
            if below < 0:
                append_common(_scaled_source_product(-confidence, delta))
            elif above > 0:
                append_common(_scaled_source_product(confidence, delta))
            else:
                append_common(_scaled_source_product(confidence, residual))
                curvature.append(scaled(_TwofoldBall.point(confidence)))
            if below == 0 or above == 0:
                smooth = False

    append_common(
        _scaled_source_product(rho, _source_sum_ball(y, -v))
    )
    curvature.append(scaled(_TwofoldBall.point(rho)))

    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            if y < penalty.lower_value:
                displacement = _source_sum_ball(y, -penalty.lower_value)
                append_common(
                    _scaled_source_product(
                        2.0,
                        penalty.strength_value,
                        displacement,
                    )
                )
                curvature.append(_scaled_source_product(
                    2.0,
                    penalty.strength_value,
                ))
            elif y > penalty.upper_value:
                displacement = _source_sum_ball(y, -penalty.upper_value)
                append_common(
                    _scaled_source_product(
                        2.0,
                        penalty.strength_value,
                        displacement,
                    )
                )
                curvature.append(_scaled_source_product(
                    2.0,
                    penalty.strength_value,
                ))
            if y == penalty.lower_value or y == penalty.upper_value:
                smooth = False
            continue

        if penalty.kind == 'exponential':
            assert penalty.tau_value is not None
            lower_argument, upper_argument = _compiled_exponential_arguments(
                penalty,
                y,
            )
            lower_exponential = _certified_exp_ball(lower_argument)
            upper_exponential = _certified_exp_ball(upper_argument)
            exponential_difference = _certified_exp_difference(
                lower_argument,
                upper_argument,
                difference=_ball_divide(
                    _source_sum_ball(
                        _source_product_ball(2.0, y),
                        -penalty.upper_value,
                        -penalty.lower_value,
                    ),
                    _TwofoldBall.point(penalty.tau_value),
                ),
                left_exponential=lower_exponential,
                right_exponential=upper_exponential,
            )
            coefficient = _binary_scaled_divide(
                scaled(_TwofoldBall.point(penalty.strength_value)),
                scaled(_TwofoldBall.point(penalty.tau_value)),
            )
            append_common(_binary_scaled_multiply(
                coefficient,
                exponential_difference,
            ))
            exponential_sum = _binary_scaled_sum((
                lower_exponential,
                upper_exponential,
            ))
            curvature_coefficient = _binary_scaled_divide(
                coefficient,
                scaled(_TwofoldBall.point(penalty.tau_value)),
            )
            curvature.append(_binary_scaled_multiply(
                curvature_coefficient,
                exponential_sum,
            ))
            continue

        assert penalty.lower_epsilon is not None
        assert penalty.lower_margin is not None
        assert penalty.upper_margin is not None
        assert penalty.upper_epsilon is not None
        assert penalty.margin_value is not None
        assert penalty.epsilon_value is not None

        def reciprocal_term(
            distance: _TwofoldBall,
            *,
            derivative_sign: float,
            reciprocal_branch: bool,
        ) -> tuple[_BinaryScaledBall, _BinaryScaledBall | None]:
            derivative = _scaled_source_ratio(
                (
                    derivative_sign,
                    penalty.strength_value,
                ),
                (distance, distance),
            )
            second = None
            if reciprocal_branch:
                second = _scaled_source_ratio(
                    (2.0, penalty.strength_value),
                    (distance, distance, distance),
                )
            return derivative, second

        lower_margin_cmp = _compare_compiled_location(
            y,
            penalty.lower_margin,
        )
        if lower_margin_cmp == 0:
            derivative, _ = reciprocal_term(
                _TwofoldBall.point(penalty.margin_value),
                derivative_sign=-1.0,
                reciprocal_branch=False,
            )
            minus.append(derivative)
            smooth = False
        elif lower_margin_cmp < 0:
            lower_epsilon_cmp = _compare_compiled_location(
                y,
                penalty.lower_epsilon,
            )
            distance = (
                _TwofoldBall.point(penalty.epsilon_value)
                if lower_epsilon_cmp <= 0
                else _source_sum_ball(y, -penalty.lower_value)
            )
            derivative, second = reciprocal_term(
                distance,
                derivative_sign=-1.0,
                reciprocal_branch=lower_epsilon_cmp > 0,
            )
            minus.append(derivative)
            plus.append(derivative)
            if second is not None:
                curvature.append(second)
            if lower_epsilon_cmp == 0:
                smooth = False

        upper_margin_cmp = _compare_compiled_location(
            y,
            penalty.upper_margin,
        )
        if upper_margin_cmp == 0:
            derivative, _ = reciprocal_term(
                _TwofoldBall.point(penalty.margin_value),
                derivative_sign=1.0,
                reciprocal_branch=False,
            )
            plus.append(derivative)
            smooth = False
        elif upper_margin_cmp > 0:
            upper_epsilon_cmp = _compare_compiled_location(
                y,
                penalty.upper_epsilon,
            )
            distance = (
                _TwofoldBall.point(penalty.epsilon_value)
                if upper_epsilon_cmp >= 0
                else _source_sum_ball(penalty.upper_value, -y)
            )
            derivative, second = reciprocal_term(
                distance,
                derivative_sign=1.0,
                reciprocal_branch=upper_epsilon_cmp < 0,
            )
            minus.append(derivative)
            plus.append(derivative)
            if second is not None:
                curvature.append(second)
            if upper_epsilon_cmp == 0:
                smooth = False

    return _KernelDerivativeEnclosures(
        minus=_binary_scaled_sum(minus),
        plus=_binary_scaled_sum(plus),
        curvature=_binary_scaled_sum(curvature),
        smooth=smooth,
    )


def _array_scalar_derivative_terms(
    spec: _CompiledScalarObjective,
    *,
    y: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    v: np.ndarray,
    rho: float,
) -> _ArrayKernelDerivativeEnclosures:
    """Vectorized rigorous ordinary evaluation for moderate smooth rows."""

    y = np.asarray(y, dtype=np.float64)
    shape = y.shape
    residual = _array_ball_subtract(
        _ArrayBall.points(y),
        _ArrayBall.points(target),
    )
    smooth = np.ones(shape, dtype=bool)
    if spec.mismatch_kind == 'squared':
        derivative = _array_source_product(confidence, residual)
        curvature = _ArrayBall.points(confidence)
    else:
        assert spec.huber_delta_value is not None
        delta = spec.huber_delta_value
        residual_lower, residual_upper = residual.physical_bounds()
        lower_branch = residual_upper < -delta
        upper_branch = residual_lower > delta
        quadratic = (residual_lower >= -delta) & (residual_upper <= delta)
        branch_resolved = lower_branch | upper_branch | quadratic
        mismatch_ball = _ArrayBall(
            np.where(
                lower_branch,
                -delta,
                np.where(upper_branch, delta, residual.high),
            ),
            np.where(quadratic, residual.low, 0.0),
            np.where(quadratic, residual.radius, 0.0),
            residual.resolved & branch_resolved,
        )
        derivative = _array_source_product(confidence, mismatch_ball)
        curvature = _ArrayBall.points(
            np.where(quadratic, confidence, 0.0)
        )
        smooth &= branch_resolved
    derivative = _array_ball_add(
        derivative,
        _array_source_product(
            rho,
            _array_ball_subtract(_ArrayBall.points(y), _ArrayBall.points(v)),
        ),
    )
    curvature = _array_ball_add(
        curvature,
        _ArrayBall.points(np.full(shape, rho)),
    )
    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            lower_active = y < penalty.lower_value
            upper_active = y > penalty.upper_value
            lower_displacement = _array_ball_subtract(
                _ArrayBall.points(y),
                _ArrayBall.points(np.full(shape, penalty.lower_value)),
            )
            upper_displacement = _array_ball_subtract(
                _ArrayBall.points(y),
                _ArrayBall.points(np.full(shape, penalty.upper_value)),
            )
            displacement = _ArrayBall(
                np.where(
                    lower_active,
                    lower_displacement.high,
                    np.where(upper_active, upper_displacement.high, 0.0),
                ),
                np.where(
                    lower_active,
                    lower_displacement.low,
                    np.where(upper_active, upper_displacement.low, 0.0),
                ),
                np.where(
                    lower_active,
                    lower_displacement.radius,
                    np.where(upper_active, upper_displacement.radius, 0.0),
                ),
                lower_displacement.resolved & upper_displacement.resolved,
            )
            derivative = _array_ball_add(
                derivative,
                _array_source_product(
                    2.0,
                    penalty.strength_value,
                    displacement,
                ),
            )
            curvature = _array_ball_add(
                curvature,
                _array_source_product(
                    np.where(lower_active | upper_active, 2.0, 0.0),
                    penalty.strength_value,
                ),
            )
            smooth &= (y != penalty.lower_value) & (y != penalty.upper_value)
            continue
        if penalty.kind != 'exponential':
            unresolved = np.zeros(shape, dtype=bool)
            derivative = _ArrayBall(
                derivative.high,
                derivative.low,
                derivative.radius,
                unresolved,
            )
            curvature = _ArrayBall(
                curvature.high,
                curvature.low,
                curvature.radius,
                unresolved,
            )
            smooth &= unresolved
            continue
        assert penalty.tau_value is not None
        lower_argument, upper_argument = _array_compiled_exponential_arguments(
            penalty,
            y,
        )
        lower_exponential = _array_certified_exp(lower_argument)
        upper_exponential = _array_certified_exp(upper_argument)
        coefficient = _array_ball_divide(
            _ArrayBall.points(np.full(shape, penalty.strength_value)),
            _ArrayBall.points(np.full(shape, penalty.tau_value)),
        )
        derivative = _array_ball_add(
            derivative,
            _array_ball_multiply(
                coefficient,
                _array_ball_subtract(
                    upper_exponential,
                    lower_exponential,
                ),
            ),
        )
        curvature_coefficient = _array_ball_divide(
            coefficient,
            _ArrayBall.points(np.full(shape, penalty.tau_value)),
        )
        curvature = _array_ball_add(
            curvature,
            _array_ball_multiply(
                curvature_coefficient,
                _array_ball_add(lower_exponential, upper_exponential),
            ),
        )
    return _ArrayKernelDerivativeEnclosures(
        derivative=derivative,
        curvature=curvature,
        smooth=smooth,
    )


def _normalized_expansion(
    value: _DoubleDouble,
) -> tuple[_DoubleDouble, int] | None:
    """Return a finite nonzero expansion mantissa and base-two exponent."""

    if not math.isfinite(value.high) or not math.isfinite(value.low):
        return None
    if value.high == 0.0 and value.low == 0.0:
        return _DoubleDouble(0.0), 0
    leading = value.high if value.high != 0.0 else value.low
    _mantissa, exponent = math.frexp(leading)
    try:
        scaled = _dd_add(
            _DoubleDouble(
                math.ldexp(value.high, -exponent),
                math.ldexp(value.low, -exponent),
            ),
            _DoubleDouble(0.0),
        )
    except OverflowError:
        return None
    magnitude = abs(scaled.high)
    if magnitude >= 1.0:
        scaled = _dd_multiply_float(scaled, 0.5)
        exponent += 1
    elif magnitude < 0.5:
        scaled = _dd_multiply_float(scaled, 2.0)
        exponent -= 1
    return scaled, exponent


def _normalized_product_parts(
    factors: tuple[_DoubleDouble, ...],
) -> tuple[_DoubleDouble, int] | None:
    """Multiply expansions as a double-double mantissa plus integer exponent."""

    result = _DoubleDouble(0.5)
    exponent = 1
    for factor in factors:
        normalized = _normalized_expansion(factor)
        if normalized is None:
            return None
        mantissa, factor_exponent = normalized
        if mantissa.high == 0.0 and mantissa.low == 0.0:
            return _DoubleDouble(0.0), 0
        result = _dd_multiply(result, mantissa)
        normalized_result = _normalized_expansion(result)
        if normalized_result is None:
            return None
        result, adjustment = normalized_result
        exponent += factor_exponent + adjustment
    return result, exponent


def _normalized_sum_parts(
    values: tuple[_DoubleDouble, ...],
) -> tuple[_DoubleDouble, int] | None:
    """Add expansions after one common power-of-two normalization."""

    normalized_values: list[tuple[_DoubleDouble, int]] = []
    for value in values:
        normalized = _normalized_expansion(value)
        if normalized is None:
            return None
        if normalized[0].high != 0.0 or normalized[0].low != 0.0:
            normalized_values.append(normalized)
    return _normalized_parts_sum(tuple(normalized_values))


def _normalized_parts_sum(
    values: tuple[tuple[_DoubleDouble, int], ...],
) -> tuple[_DoubleDouble, int] | None:
    """Add already-normalized expansions at one shared power-of-two scale."""

    if not values:
        return _DoubleDouble(0.0), 0
    common_exponent = max(exponent for _value, exponent in values)
    scaled_values: list[_DoubleDouble] = []
    for value, exponent in values:
        adjustment = exponent - common_exponent
        high = math.ldexp(value.high, adjustment)
        low = math.ldexp(value.low, adjustment)
        if (
            (value.high != 0.0 and high == 0.0)
            or (value.low != 0.0 and low == 0.0)
        ):
            return None
        scaled_values.append(_DoubleDouble(high, low))
    summed = _dd_sum(scaled_values)
    if summed.high == 0.0 and summed.low == 0.0:
        # A true zero and cancellation below the short expansion have different
        # sign consequences.  Let the caller use its exact exceptional source.
        return None
    normalized = _normalized_expansion(summed)
    if normalized is None:
        return None
    mantissa, adjustment = normalized
    return mantissa, common_exponent + adjustment


def _normalized_affine_parts(
    beta: float,
    alpha: float,
    left: float,
    right: float,
    target: float = 0.0,
) -> tuple[_DoubleDouble, int] | None:
    """Return ``beta + alpha*left - alpha*right - target`` normalized."""

    terms: list[tuple[_DoubleDouble, int]] = []
    for scalar in (beta, -target):
        normalized = _normalized_expansion(_DoubleDouble(scalar))
        if normalized is None:
            return None
        if normalized[0].high != 0.0 or normalized[0].low != 0.0:
            terms.append(normalized)
    for sign, operand in ((1.0, left), (-1.0, right)):
        normalized = _normalized_product_parts(
            (
                _DoubleDouble(sign),
                _DoubleDouble(alpha),
                _DoubleDouble(operand),
            )
        )
        if normalized is None:
            return None
        if normalized[0].high != 0.0 or normalized[0].low != 0.0:
            terms.append(normalized)
    return _normalized_parts_sum(tuple(terms))


def _normalized_product_dd(
    factors: tuple[_DoubleDouble, ...],
    *,
    exponent_offset: int = 0,
) -> _DoubleDouble | None:
    """Materialize a normalized product only when safely in normal range."""

    if factors and exponent_offset == 0:
        fast = factors[0]
        if fast.high == 0.0 and fast.low == 0.0:
            return _DoubleDouble(0.0)
        fast_safe = (
            math.isfinite(fast.high)
            and math.isfinite(fast.low)
            and abs(fast.high) >= _DD_FAST_MIN
        )
        for factor in factors[1:]:
            if factor.high == 0.0 and factor.low == 0.0:
                return _DoubleDouble(0.0)
            if not fast_safe:
                break
            fast = (
                _dd_multiply_float(fast, factor.high)
                if factor.low == 0.0
                else _dd_multiply(fast, factor)
            )
            fast_safe = (
                math.isfinite(fast.high)
                and math.isfinite(fast.low)
                and abs(fast.high) >= _DD_FAST_MIN
            )
        if fast_safe:
            return fast

    normalized = _normalized_product_parts(factors)
    if normalized is None:
        return None
    return _materialize_normalized_dd(
        normalized,
        exponent_offset=exponent_offset,
    )


def _materialize_normalized_dd(
    normalized: tuple[_DoubleDouble, int],
    *,
    exponent_offset: int = 0,
) -> _DoubleDouble | None:
    """Materialize normalized parts away from binary64 range boundaries."""

    mantissa, exponent = normalized
    exponent += int(exponent_offset)
    if mantissa.high == 0.0 and mantissa.low == 0.0:
        return _DoubleDouble(0.0)
    # Near the subnormal boundary the low limb can itself underflow and affect
    # final rounding.  Near overflow the leading limb needs an exact range
    # decision.  Those bounded exceptional cases use the exact source dyadic.
    if exponent <= -1019 or exponent >= 1024:
        return None
    try:
        return _dd_add(
            _DoubleDouble(
                math.ldexp(mantissa.high, exponent),
                math.ldexp(mantissa.low, exponent),
            ),
            _DoubleDouble(0.0),
        )
    except OverflowError:
        return None


def _normalized_ratio_power_parts(
    numerator_factors: tuple[_DoubleDouble, ...],
    denominator: _DoubleDouble,
    denominator_power: int,
) -> tuple[_DoubleDouble, int] | None:
    """Normalize a complete product divided by one expansion power."""

    numerator = _normalized_product_parts(numerator_factors)
    normalized_denominator = _normalized_expansion(denominator)
    if numerator is None or normalized_denominator is None:
        return None
    result, exponent = numerator
    divisor, divisor_exponent = normalized_denominator
    if divisor.high == 0.0 and divisor.low == 0.0:
        return None
    for _ in range(int(denominator_power)):
        result = _dd_divide(result, divisor)
        normalized_result = _normalized_expansion(result)
        if normalized_result is None:
            return None
        result, adjustment = normalized_result
        exponent += adjustment - divisor_exponent
    return result, exponent


def _normalized_ratio_power_dd(
    numerator_factors: tuple[_DoubleDouble, ...],
    denominator: _DoubleDouble,
    denominator_power: int,
) -> _DoubleDouble | None:
    """Materialize a normalized product ratio only in the safe normal range."""

    normalized = _normalized_ratio_power_parts(
        numerator_factors,
        denominator,
        denominator_power,
    )
    if normalized is None:
        return None
    return _materialize_normalized_dd(normalized)


def _normalized_ratio_power_log_term(
    numerator_factors: tuple[_DoubleDouble, ...],
    denominator: _DoubleDouble,
    denominator_power: int,
) -> tuple[int, float, float]:
    """Return a range-free signed-log enclosure for a product ratio."""

    normalized = _normalized_ratio_power_parts(
        numerator_factors,
        denominator,
        denominator_power,
    )
    if normalized is None:
        return 0, 0.0, math.inf
    mantissa, exponent = normalized
    value = mantissa.value
    if value == 0.0:
        return 0, 0.0, math.inf
    return (
        1 if value > 0.0 else -1,
        math.log(abs(value)) + exponent * math.log(2.0),
        256.0 * max(1, denominator_power) * (_FLOAT_EPSILON**2),
    )


def _normalized_product_log_term(
    factors: tuple[_DoubleDouble, ...],
    *,
    exponent_offset: int = 0,
) -> tuple[int, float, float]:
    """Return a signed-log enclosure term without materializing its range."""

    normalized = _normalized_product_parts(factors)
    if normalized is None:
        return 0, 0.0, math.inf
    mantissa, exponent = normalized
    exponent += int(exponent_offset)
    value = mantissa.value
    if value == 0.0:
        return 0, 0.0, 0.0
    return (
        1 if value > 0.0 else -1,
        math.log(abs(value)) + exponent * math.log(2.0),
        128.0 * (_FLOAT_EPSILON**2),
    )


def _fraction_to_dd(value: Fraction) -> _DoubleDouble:
    """Round an exact rational to a short expansion without losing range."""

    try:
        high = float(value)
    except OverflowError:
        return _DoubleDouble(-math.inf if value < 0 else math.inf, 0.0)
    if not math.isfinite(high):
        return _DoubleDouble(high, 0.0)
    low = float(value - _fraction(high))
    return _dd_add(_DoubleDouble(high), _DoubleDouble(low))


def _scaled_square_difference_dd(
    left: float,
    right: float,
    *scale_factors: float,
) -> _DoubleDouble:
    """Evaluate a separately factored scaled square before any range loss."""

    difference_parts = _normalized_sum_parts(
        (_DoubleDouble(left), _DoubleDouble(-right))
    )
    if difference_parts is None:
        result = None
    else:
        difference, difference_exponent = difference_parts
        factors = tuple(
            _DoubleDouble(float(factor)) for factor in scale_factors
        ) + (difference, difference)
        result = _normalized_product_dd(
            factors,
            exponent_offset=2 * difference_exponent,
        )
    if result is not None:
        return result
    if not all(
        math.isfinite(value)
        for value in (left, right, *scale_factors)
    ):
        return _DoubleDouble(math.inf, 0.0)
    exact_difference = _fraction(left) - _fraction(right)
    exact = exact_difference**2
    for factor in scale_factors:
        exact *= _fraction(factor)
    return _fraction_to_dd(exact)


def _huber_linear_value_dd(
    left: float,
    right: float,
    delta: float,
    confidence: float,
) -> _DoubleDouble:
    """Evaluate ``c*d*(abs(left-right)-d/2)`` from separate factors."""

    difference = _dd_difference(left, right)
    if math.isfinite(difference.high) and math.isfinite(difference.low):
        sign = -1.0 if (
            difference.high < 0.0
            or (difference.high == 0.0 and difference.low < 0.0)
        ) else 1.0
        magnitude = difference if sign > 0.0 else _dd_negate(difference)
        leading = _normalized_product_dd(
            (
                _DoubleDouble(confidence),
                _DoubleDouble(delta),
                magnitude,
            )
        )
        correction = _normalized_product_dd(
            (
                _DoubleDouble(-0.5),
                _DoubleDouble(confidence),
                _DoubleDouble(delta),
                _DoubleDouble(delta),
            )
        )
        if leading is not None and correction is not None:
            result = _dd_add(leading, correction)
            if math.isfinite(result.high) and math.isfinite(result.low):
                return result
    if not all(
        math.isfinite(value) for value in (left, right, delta, confidence)
    ):
        return _DoubleDouble(math.inf, 0.0)
    exact_delta = _fraction(delta)
    exact = _fraction(confidence) * exact_delta * (
        abs(_fraction(left) - _fraction(right)) - exact_delta / 2
    )
    return _fraction_to_dd(exact)


def _reciprocal_boundary_scalar_value(
    distance: _DoubleDouble,
    *,
    branch: str,
    margin: float,
    epsilon: float,
    strength: float,
) -> _DoubleDouble:
    if branch == 'inactive':
        return _DoubleDouble(0.0)
    if branch == 'reciprocal':
        numerator = _dd_add(_DoubleDouble(margin), _dd_negate(distance))
        return _dd_multiply_float(
            _dd_divide_float(_dd_divide(numerator, distance), margin),
            strength,
        )
    boundary_at_epsilon = _dd_divide_float(
        _dd_divide_float(
            _dd_difference(margin, epsilon),
            epsilon,
        ),
        margin,
    )
    tangent_displacement = _dd_divide_float(
        _dd_divide_float(
            _dd_add(_DoubleDouble(epsilon), _dd_negate(distance)),
            epsilon,
        ),
        epsilon,
    )
    return _dd_multiply_float(
        _dd_add(boundary_at_epsilon, tangent_displacement),
        strength,
    )


def _reciprocal_scalar_branch(
    y: float,
    *,
    epsilon_location: _CompiledLocation,
    margin_location: _CompiledLocation,
    lower_side: bool,
) -> str:
    epsilon_cmp = _compare_compiled_location(y, epsilon_location)
    margin_cmp = _compare_compiled_location(y, margin_location)
    if lower_side:
        if margin_cmp >= 0:
            return 'inactive'
        return 'reciprocal' if epsilon_cmp > 0 else 'tangent'
    if margin_cmp <= 0:
        return 'inactive'
    return 'reciprocal' if epsilon_cmp < 0 else 'tangent'


def _compiled_penalty_scalar_value(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> float:
    """Evaluate one compiled term from complete binary64 source operands."""

    if penalty.kind == 'soft':
        if y < penalty.lower_value:
            boundary = penalty.lower_value
        elif y > penalty.upper_value:
            boundary = penalty.upper_value
        else:
            return 0.0
        return _scaled_square_difference_dd(
            y,
            boundary,
            penalty.strength_value,
        ).value

    if penalty.kind == 'exponential':
        return _compiled_exponential_value(penalty, _DoubleDouble(y))

    assert penalty.lower_epsilon is not None
    assert penalty.lower_margin is not None
    assert penalty.upper_margin is not None
    assert penalty.upper_epsilon is not None
    assert penalty.margin_value is not None
    assert penalty.epsilon_value is not None
    lower_branch = _reciprocal_scalar_branch(
        y,
        epsilon_location=penalty.lower_epsilon,
        margin_location=penalty.lower_margin,
        lower_side=True,
    )
    upper_branch = _reciprocal_scalar_branch(
        y,
        epsilon_location=penalty.upper_epsilon,
        margin_location=penalty.upper_margin,
        lower_side=False,
    )
    lower_value = _reciprocal_boundary_scalar_value(
        _dd_difference(y, penalty.lower_value),
        branch=lower_branch,
        margin=penalty.margin_value,
        epsilon=penalty.epsilon_value,
        strength=penalty.strength_value,
    )
    upper_value = _reciprocal_boundary_scalar_value(
        _dd_difference(penalty.upper_value, y),
        branch=upper_branch,
        margin=penalty.margin_value,
        epsilon=penalty.epsilon_value,
        strength=penalty.strength_value,
    )
    result = _dd_add(lower_value, upper_value).value
    if math.isfinite(result):
        return result
    return _compiled_reciprocal_fraction_value(penalty, _fraction(y))


def _compiled_reciprocal_fraction_value(
    penalty: _CompiledScalarPenalty,
    measurement: Fraction,
) -> float:
    """Evaluate a compiled reciprocal pair exactly before float rounding."""

    assert penalty.kind == 'reciprocal'
    assert penalty.margin is not None
    assert penalty.epsilon is not None
    value = Fraction(0)
    for distance in (
        measurement - penalty.lower,
        penalty.upper - measurement,
    ):
        if distance >= penalty.margin:
            continue
        if distance > penalty.epsilon:
            value += penalty.strength * (
                1 / distance - 1 / penalty.margin
            )
        else:
            value += penalty.strength * (
                1 / penalty.epsilon
                - 1 / penalty.margin
                - (distance - penalty.epsilon) / (penalty.epsilon**2)
            )
    return _fraction_to_extended_float(value)


def _compiled_soft_fraction_value(
    penalty: _CompiledScalarPenalty,
    measurement: Fraction,
) -> float:
    """Evaluate a compiled soft square exactly before float rounding."""

    assert penalty.kind == 'soft'
    if measurement < penalty.lower:
        displacement = measurement - penalty.lower
    elif measurement > penalty.upper:
        displacement = measurement - penalty.upper
    else:
        return 0.0
    return _fraction_to_extended_float(penalty.strength * displacement**2)


def _compiled_exponential_fraction_value(
    penalty: _CompiledScalarPenalty,
    measurement: Fraction,
) -> float:
    """Evaluate an exponential pair from an exact exceptional measurement."""

    assert penalty.kind == 'exponential'
    assert penalty.margin is not None
    assert penalty.tau is not None

    def extended_float(value: Fraction) -> float:
        try:
            return float(value)
        except OverflowError:
            return -math.inf if value < 0 else math.inf

    left = extended_float(
        (penalty.lower + penalty.margin - measurement) / penalty.tau
    )
    right = extended_float(
        (measurement - (penalty.upper - penalty.margin)) / penalty.tau
    )
    maximum = max(left, right)
    if maximum == math.inf:
        return math.inf
    if maximum == -math.inf:
        return 0.0
    return _compiled_exponential_from_exponents(penalty, left, right)


def _compiled_exponential_from_exponents(
    penalty: _CompiledScalarPenalty,
    left: float,
    right: float,
) -> float:
    """Evaluate the shared scaled exponential-pair value primitive."""

    maximum = max(left, right)
    log_value = (
        math.log(penalty.strength_value)
        + maximum
        + math.log1p(math.exp(-abs(right - left)))
    )
    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    if log_value < math.log(math.ulp(0.0)) - 2.0:
        return 0.0
    return math.exp(log_value)


def _compiled_penalty_expansion_value(
    penalty: _CompiledScalarPenalty,
    measurement: _DoubleDouble,
) -> float:
    """Evaluate one compiled term from a complete measurement expansion."""

    if measurement.low == 0.0:
        return _compiled_penalty_scalar_value(penalty, measurement.high)
    if penalty.kind == 'exponential':
        return _compiled_exponential_value(penalty, measurement)

    exact_measurement = (
        _fraction(measurement.high) + _fraction(measurement.low)
    )
    if penalty.kind == 'soft':
        return _compiled_soft_fraction_value(
            penalty,
            exact_measurement,
        )

    assert penalty.kind == 'reciprocal'
    return _compiled_reciprocal_fraction_value(
        penalty,
        exact_measurement,
    )


def _compiled_exponential_value(
    penalty: _CompiledScalarPenalty,
    measurement: _DoubleDouble,
) -> float:
    """Evaluate an exponential pair from one complete measurement expansion."""

    assert penalty.kind == 'exponential'
    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    left = _dd_divide_float(
        _dd_sum(
            (
                _DoubleDouble(penalty.lower_value),
                _DoubleDouble(penalty.margin_value),
                _dd_negate(measurement),
            )
        ),
        penalty.tau_value,
    ).value
    right = _dd_divide_float(
        _dd_sum(
            (
                measurement,
                _DoubleDouble(-penalty.upper_value),
                _DoubleDouble(penalty.margin_value),
            )
        ),
        penalty.tau_value,
    ).value
    return _compiled_exponential_from_exponents(penalty, left, right)


def _scalar_objective_value(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> float:
    """Return the compiled source objective on the binary64 common path."""

    scalar = float(y)
    if not math.isfinite(scalar):
        return math.inf
    parts: list[float] = []
    residual = _dd_difference(scalar, float(target))
    if confidence != 0.0:
        if spec.mismatch_kind == 'squared':
            mismatch = _scaled_square_difference_dd(
                scalar,
                float(target),
                0.5,
                float(confidence),
            ).value
        else:
            assert spec.huber_delta_value is not None
            delta = spec.huber_delta_value
            absolute_residual = abs(residual.value)
            if absolute_residual <= delta:
                mismatch = _scaled_square_difference_dd(
                    scalar,
                    float(target),
                    0.5,
                    float(confidence),
                ).value
            else:
                mismatch = _huber_linear_value_dd(
                    scalar,
                    float(target),
                    delta,
                    float(confidence),
                ).value
        parts.append(mismatch)
    parts.append(
        _scaled_square_difference_dd(
            scalar,
            float(v),
            0.5,
            float(rho),
        ).value
    )
    for penalty in spec.penalties:
        parts.append(_compiled_penalty_scalar_value(penalty, scalar))
    if any(math.isnan(value) for value in parts):
        return math.inf
    if any(value == math.inf for value in parts):
        return math.inf
    try:
        total = math.fsum(parts)
    except OverflowError:
        return math.inf
    return total if math.isfinite(total) else math.inf


def _factored_square_difference(
    lower: float,
    upper: float,
    center: float,
    *scale_factors: float,
) -> _DoubleDouble:
    delta_parts = _normalized_sum_parts(
        (_DoubleDouble(upper), _DoubleDouble(-lower))
    )
    summed_parts = _normalized_sum_parts(
        (
            _DoubleDouble(upper),
            _DoubleDouble(lower),
            _DoubleDouble(-center),
            _DoubleDouble(-center),
        )
    )
    result = None
    if delta_parts is not None and summed_parts is not None:
        delta, delta_exponent = delta_parts
        summed, summed_exponent = summed_parts
        factors = tuple(
            _DoubleDouble(float(factor)) for factor in scale_factors
        ) + (delta, summed)
        result = _normalized_product_dd(
            factors,
            exponent_offset=delta_exponent + summed_exponent,
        )
    if result is not None:
        return result
    if not all(
        math.isfinite(value)
        for value in (lower, upper, center, *scale_factors)
    ):
        return _DoubleDouble(math.nan)
    lower_exact = _fraction(lower) - _fraction(center)
    upper_exact = _fraction(upper) - _fraction(center)
    exact = upper_exact**2 - lower_exact**2
    for factor in scale_factors:
        exact *= _fraction(factor)
    return _fraction_to_dd(exact)


def _fraction_log_abs(value: Fraction) -> float:
    """Return ``log(abs(value))`` without converting the rational to float."""

    absolute = abs(value)
    return math.log(absolute.numerator) - math.log(absolute.denominator)


def _factored_square_difference_log_term(
    lower: float,
    upper: float,
    center: float,
    *scale_factors: float,
) -> tuple[int, float, float] | None:
    """Return a range-free signed-log term for a factored square difference."""

    delta_parts = _normalized_sum_parts(
        (_DoubleDouble(upper), _DoubleDouble(-lower))
    )
    summed_parts = _normalized_sum_parts(
        (
            _DoubleDouble(upper),
            _DoubleDouble(lower),
            _DoubleDouble(-center),
            _DoubleDouble(-center),
        )
    )
    if delta_parts is None or summed_parts is None:
        term = (0, 0.0, math.inf)
    else:
        delta, delta_exponent = delta_parts
        summed, summed_exponent = summed_parts
        factors = tuple(
            _DoubleDouble(float(factor)) for factor in scale_factors
        ) + (delta, summed)
        term = _normalized_product_log_term(
            factors,
            exponent_offset=delta_exponent + summed_exponent,
        )
    if term[0] != 0:
        return term
    if not all(
        math.isfinite(value)
        for value in (lower, upper, center, *scale_factors)
    ):
        return 0, 0.0, math.inf
    lower_exact = _fraction(lower) - _fraction(center)
    upper_exact = _fraction(upper) - _fraction(center)
    exact = upper_exact**2 - lower_exact**2
    for factor in scale_factors:
        exact *= _fraction(factor)
    if exact == 0:
        return None
    return (
        1 if exact > 0 else -1,
        _fraction_log_abs(exact),
        0.0,
    )


def _compiled_penalty_scalar_dd_value(
    penalty: _CompiledScalarPenalty,
    y: float,
) -> _DoubleDouble:
    """Evaluate one non-exponential term without rounding its full value."""

    if penalty.kind == 'soft':
        if y < penalty.lower_value:
            boundary = penalty.lower_value
        elif y > penalty.upper_value:
            boundary = penalty.upper_value
        else:
            return _DoubleDouble(0.0)
        return _scaled_square_difference_dd(
            y,
            boundary,
            penalty.strength_value,
        )

    assert penalty.kind == 'reciprocal'
    assert penalty.lower_epsilon is not None
    assert penalty.lower_margin is not None
    assert penalty.upper_margin is not None
    assert penalty.upper_epsilon is not None
    assert penalty.margin_value is not None
    assert penalty.epsilon_value is not None
    return _dd_add(
        _reciprocal_boundary_scalar_value(
            _dd_difference(y, penalty.lower_value),
            branch=_reciprocal_scalar_branch(
                y,
                epsilon_location=penalty.lower_epsilon,
                margin_location=penalty.lower_margin,
                lower_side=True,
            ),
            margin=penalty.margin_value,
            epsilon=penalty.epsilon_value,
            strength=penalty.strength_value,
        ),
        _reciprocal_boundary_scalar_value(
            _dd_difference(penalty.upper_value, y),
            branch=_reciprocal_scalar_branch(
                y,
                epsilon_location=penalty.upper_epsilon,
                margin_location=penalty.upper_margin,
                lower_side=False,
            ),
            margin=penalty.margin_value,
            epsilon=penalty.epsilon_value,
            strength=penalty.strength_value,
        ),
    )


def _huber_scalar_dd_value(
    value: float,
    *,
    target: float,
    delta: float,
    confidence: float,
) -> _DoubleDouble:
    """Return one Huber term as an unrounded short expansion."""

    residual = _dd_difference(value, target)
    residual_sign = (
        int(residual.high > 0.0) - int(residual.high < 0.0)
        if residual.high != 0.0
        else int(residual.low > 0.0) - int(residual.low < 0.0)
    )
    magnitude = residual if residual_sign >= 0 else _dd_negate(residual)
    outside = _dd_add(magnitude, _DoubleDouble(-delta))
    outside_sign = (
        int(outside.high > 0.0) - int(outside.high < 0.0)
        if outside.high != 0.0
        else int(outside.low > 0.0) - int(outside.low < 0.0)
    )
    if outside_sign <= 0:
        return _scaled_square_difference_dd(
            value,
            target,
            0.5,
            confidence,
        )
    return _huber_linear_value_dd(
        value,
        target,
        delta,
        confidence,
    )


def _reciprocal_boundary_difference(
    penalty: _CompiledScalarPenalty,
    lower: float,
    upper: float,
    *,
    lower_side: bool,
) -> _DoubleDouble:
    """Return one boundary contribution at ``upper`` minus ``lower``.

    Same-branch expressions are factored before evaluation.  Cross-branch
    expressions share the exact continuation value at ``epsilon`` or the
    zero value at ``margin`` so a large additive boundary constant never
    controls the terminal candidate decision.
    """

    assert penalty.margin_value is not None
    assert penalty.epsilon_value is not None
    assert penalty.lower_epsilon is not None
    assert penalty.lower_margin is not None
    assert penalty.upper_margin is not None
    assert penalty.upper_epsilon is not None
    margin = penalty.margin_value
    epsilon = penalty.epsilon_value
    strength = penalty.strength_value
    if lower_side:
        distance_lower = _dd_difference(lower, penalty.lower_value)
        distance_upper = _dd_difference(upper, penalty.lower_value)
        epsilon_location = penalty.lower_epsilon
        margin_location = penalty.lower_margin
    else:
        distance_lower = _dd_difference(penalty.upper_value, lower)
        distance_upper = _dd_difference(penalty.upper_value, upper)
        epsilon_location = penalty.upper_epsilon
        margin_location = penalty.upper_margin
    lower_branch = _reciprocal_scalar_branch(
        lower,
        epsilon_location=epsilon_location,
        margin_location=margin_location,
        lower_side=lower_side,
    )
    upper_branch = _reciprocal_scalar_branch(
        upper,
        epsilon_location=epsilon_location,
        margin_location=margin_location,
        lower_side=lower_side,
    )
    distance_change = _dd_add(
        distance_lower,
        _dd_negate(distance_upper),
    )
    if lower_branch == upper_branch:
        if lower_branch == 'inactive':
            return _DoubleDouble(0.0)
        if lower_branch == 'tangent':
            return _dd_multiply_float(
                _dd_divide_float(
                    _dd_divide_float(distance_change, epsilon),
                    epsilon,
                ),
                strength,
            )
        return _dd_multiply_float(
            _dd_divide(
                _dd_divide(distance_change, distance_lower),
                distance_upper,
            ),
            strength,
        )

    def relative_to_margin(
        distance: _DoubleDouble,
        branch: str,
    ) -> _DoubleDouble:
        if branch == 'inactive':
            return _DoubleDouble(0.0)
        numerator = _dd_add(_DoubleDouble(margin), _dd_negate(distance))
        return _dd_multiply_float(
            _dd_divide_float(_dd_divide(numerator, distance), margin),
            strength,
        )

    if {lower_branch, upper_branch} <= {'reciprocal', 'inactive'}:
        lower_relative = relative_to_margin(distance_lower, lower_branch)
        upper_relative = relative_to_margin(distance_upper, upper_branch)
    else:
        def relative_to_epsilon(
            distance: _DoubleDouble,
            branch: str,
        ) -> _DoubleDouble:
            numerator = _dd_add(
                _DoubleDouble(epsilon),
                _dd_negate(distance),
            )
            if branch == 'tangent':
                relative = _dd_divide_float(
                    _dd_divide_float(numerator, epsilon),
                    epsilon,
                )
            elif branch == 'reciprocal':
                relative = _dd_divide_float(
                    _dd_divide(numerator, distance),
                    epsilon,
                )
            else:
                numerator = _dd_difference(epsilon, margin)
                relative = _dd_divide_float(
                    _dd_divide_float(numerator, epsilon),
                    margin,
                )
            return _dd_multiply_float(
                relative,
                strength,
            )

        lower_relative = relative_to_epsilon(distance_lower, lower_branch)
        upper_relative = relative_to_epsilon(distance_upper, upper_branch)
    return _dd_add(upper_relative, _dd_negate(lower_relative))


def _exponential_difference_logs(
    penalty: _CompiledScalarPenalty,
    lower: float,
    upper: float,
) -> list[tuple[int, float, float]]:
    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    logs: list[tuple[int, float, float]] = []
    log_strength = math.log(penalty.strength_value)
    for lower_numerator, upper_numerator in (
        (
            _dd_add(
                _dd_difference(penalty.lower_value, lower),
                _DoubleDouble(penalty.margin_value),
            ),
            _dd_add(
                _dd_difference(penalty.lower_value, upper),
                _DoubleDouble(penalty.margin_value),
            ),
        ),
        (
            _dd_add(
                _dd_difference(lower, penalty.upper_value),
                _DoubleDouble(penalty.margin_value),
            ),
            _dd_add(
                _dd_difference(upper, penalty.upper_value),
                _DoubleDouble(penalty.margin_value),
            ),
        ),
    ):
        exponent_lower = _dd_divide_float(
            lower_numerator,
            penalty.tau_value,
        )
        exponent_upper = _dd_divide_float(
            upper_numerator,
            penalty.tau_value,
        )
        left = exponent_lower.value
        right = exponent_upper.value
        difference = _dd_add(
            exponent_upper,
            _dd_negate(exponent_lower),
        ).value
        if difference == 0.0:
            continue
        maximum = max(left, right)
        balanced = -math.expm1(-abs(difference))
        if not (
            math.isfinite(maximum)
            and math.isfinite(balanced)
            and balanced > 0.0
        ):
            return [(0, 0.0, math.inf)]
        logs.append(
            (
                1 if difference > 0.0 else -1,
                log_strength + maximum + math.log(balanced),
                32.0 * _FLOAT_EPSILON,
            )
        )
    return logs


def _exponential_difference_expansion(
    penalty: _CompiledScalarPenalty,
    lower: float,
    upper: float,
) -> tuple[_DoubleDouble, float | None] | None:
    """Return a moderate direct difference and its absolute-error log."""

    assert penalty.margin_value is not None
    assert penalty.tau_value is not None
    terms: list[_DoubleDouble] = []
    for lower_numerator, upper_numerator in (
        (
            _dd_add(
                _dd_difference(penalty.lower_value, lower),
                _DoubleDouble(penalty.margin_value),
            ),
            _dd_add(
                _dd_difference(penalty.lower_value, upper),
                _DoubleDouble(penalty.margin_value),
            ),
        ),
        (
            _dd_add(
                _dd_difference(lower, penalty.upper_value),
                _DoubleDouble(penalty.margin_value),
            ),
            _dd_add(
                _dd_difference(upper, penalty.upper_value),
                _DoubleDouble(penalty.margin_value),
            ),
        ),
    ):
        lower_exponent = _dd_divide_float(
            lower_numerator,
            penalty.tau_value,
        )
        upper_exponent = _dd_divide_float(
            upper_numerator,
            penalty.tau_value,
        )
        if not (
            -700.0 <= lower_exponent.value <= 700.0
            and -700.0 <= upper_exponent.value <= 700.0
        ):
            return None
        delta = _dd_add(upper_exponent, _dd_negate(lower_exponent))
        terms.append(
            _dd_multiply(
                _exp_double_double(lower_exponent),
                _expm1_double_double(delta),
            )
        )
    difference = _dd_sum(terms)
    result = _dd_multiply_float(
        difference,
        penalty.strength_value,
    )
    if not math.isfinite(result.high):
        return None
    term_sum = math.fsum(abs(term.value) for term in terms)
    error_log = None if term_sum == 0.0 else (
        math.log(penalty.strength_value)
        + math.log(term_sum)
        + math.log(_EXPANSION_RELATIVE_ERROR)
    )
    return result, error_log


def _legacy_scalar_objective_difference(
    spec: _CompiledScalarObjective,
    *,
    lower: float,
    upper: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _ScaledEnclosure:
    """Enclose ``F(upper) - F(lower)`` directly, term by term."""

    algebraic: list[_DoubleDouble] = []
    logarithmic: list[tuple[int, float, float]] = []

    def append_factored(
        center: float,
        *scale_factors: float,
    ) -> None:
        value = _factored_square_difference(
            lower,
            upper,
            center,
            *scale_factors,
        )
        if (
            math.isfinite(value.high)
            and math.isfinite(value.low)
            and value.value != 0.0
        ):
            algebraic.append(value)
            return
        term = _factored_square_difference_log_term(
            lower,
            upper,
            center,
            *scale_factors,
        )
        if term is not None:
            logarithmic.append(term)

    if confidence != 0.0:
        if spec.mismatch_kind == 'squared':
            append_factored(target, 0.5, confidence)
        else:
            assert spec.huber_delta_value is not None
            lower_value = _huber_scalar_dd_value(
                lower,
                target=target,
                delta=spec.huber_delta_value,
                confidence=confidence,
            )
            upper_value = _huber_scalar_dd_value(
                upper,
                target=target,
                delta=spec.huber_delta_value,
                confidence=confidence,
            )
            algebraic.append(_dd_add(upper_value, _dd_negate(lower_value)))
    append_factored(v, 0.5, rho)
    for penalty in spec.penalties:
        if penalty.kind == 'exponential':
            expanded = _exponential_difference_expansion(
                penalty,
                lower,
                upper,
            )
            if expanded is None:
                logarithmic.extend(
                    _exponential_difference_logs(penalty, lower, upper)
                )
            else:
                algebraic.append(expanded[0])
                if expanded[1] is not None:
                    logarithmic.append((0, expanded[1], 1.0))
            continue
        if penalty.kind == 'soft':
            if lower < penalty.lower_value and upper <= penalty.lower_value:
                append_factored(
                    penalty.lower_value,
                    penalty.strength_value,
                )
                continue
            if lower >= penalty.upper_value and upper > penalty.upper_value:
                append_factored(
                    penalty.upper_value,
                    penalty.strength_value,
                )
                continue
        if penalty.kind == 'reciprocal':
            algebraic.extend(
                (
                    _reciprocal_boundary_difference(
                        penalty,
                        lower,
                        upper,
                        lower_side=True,
                    ),
                    _reciprocal_boundary_difference(
                        penalty,
                        lower,
                        upper,
                        lower_side=False,
                    ),
                )
            )
            continue
        lower_value = _compiled_penalty_scalar_dd_value(penalty, lower)
        upper_value = _compiled_penalty_scalar_dd_value(penalty, upper)
        algebraic.append(_dd_add(upper_value, _dd_negate(lower_value)))
    logarithmic.extend(_dd_enclosure_terms(algebraic))
    return _scaled_signed_enclosure(tuple(logarithmic))


def _direct_factored_square_difference_ball(
    lower: _TwofoldBall,
    upper: _TwofoldBall,
    center: _TwofoldBall,
    *scales: float,
) -> _BinaryScaledBall:
    gap = _ball_subtract(upper, lower)
    centered_sum = _scaled_source_sum(
        upper,
        lower,
        _ball_negate(center),
        _ball_negate(center),
    )
    value = _binary_scaled_multiply(
        _binary_scaled_from_ball(gap),
        centered_sum,
    )
    for scale in scales:
        value = _binary_scaled_multiply(
            value,
            _binary_scaled_from_ball(_TwofoldBall.point(scale)),
        )
    return value


def _direct_reciprocal_difference_terms(
    penalty: _CompiledScalarPenalty,
    lower: float,
    upper: float,
) -> tuple[_BinaryScaledBall, ...]:
    """Partition both reciprocal boundary differences at exact locations."""

    assert penalty.lower_epsilon is not None
    assert penalty.lower_margin is not None
    assert penalty.upper_margin is not None
    assert penalty.upper_epsilon is not None
    assert penalty.epsilon_value is not None
    terms: list[_BinaryScaledBall] = []
    lower_ball = _TwofoldBall.point(lower)
    upper_ball = _TwofoldBall.point(upper)
    epsilon = _TwofoldBall.point(penalty.epsilon_value)

    def reciprocal_segment(
        start: _TwofoldBall,
        end: _TwofoldBall,
        *,
        lower_side: bool,
    ) -> None:
        if lower_side:
            start_distance = _source_sum_ball(start, -penalty.lower_value)
            end_distance = _source_sum_ball(end, -penalty.lower_value)
        else:
            start_distance = _source_sum_ball(
                penalty.upper_value,
                _ball_negate(start),
            )
            end_distance = _source_sum_ball(
                penalty.upper_value,
                _ball_negate(end),
            )
        terms.append(_scaled_source_ratio(
            (
                penalty.strength_value,
                _ball_subtract(start_distance, end_distance),
            ),
            (start_distance, end_distance),
        ))

    lower_epsilon_cmp_lo = _compare_compiled_location(
        lower,
        penalty.lower_epsilon,
    )
    lower_epsilon_cmp_hi = _compare_compiled_location(
        upper,
        penalty.lower_epsilon,
    )
    if lower_epsilon_cmp_lo < 0:
        end = (
            upper_ball
            if lower_epsilon_cmp_hi <= 0
            else _ball_from_fraction(penalty.lower_epsilon.exact)
        )
        terms.append(_scaled_source_ratio(
            (
                -penalty.strength_value,
                _ball_subtract(end, lower_ball),
            ),
            (epsilon, epsilon),
        ))

    lower_margin_cmp_lo = _compare_compiled_location(
        lower,
        penalty.lower_margin,
    )
    lower_margin_cmp_hi = _compare_compiled_location(
        upper,
        penalty.lower_margin,
    )
    if lower_epsilon_cmp_hi > 0 and lower_margin_cmp_lo < 0:
        start = (
            lower_ball
            if lower_epsilon_cmp_lo >= 0
            else _ball_from_fraction(penalty.lower_epsilon.exact)
        )
        end = (
            upper_ball
            if lower_margin_cmp_hi <= 0
            else _ball_from_fraction(penalty.lower_margin.exact)
        )
        reciprocal_segment(start, end, lower_side=True)

    upper_margin_cmp_lo = _compare_compiled_location(
        lower,
        penalty.upper_margin,
    )
    upper_margin_cmp_hi = _compare_compiled_location(
        upper,
        penalty.upper_margin,
    )
    upper_epsilon_cmp_lo = _compare_compiled_location(
        lower,
        penalty.upper_epsilon,
    )
    upper_epsilon_cmp_hi = _compare_compiled_location(
        upper,
        penalty.upper_epsilon,
    )
    if upper_margin_cmp_hi > 0 and upper_epsilon_cmp_lo < 0:
        start = (
            lower_ball
            if upper_margin_cmp_lo >= 0
            else _ball_from_fraction(penalty.upper_margin.exact)
        )
        end = (
            upper_ball
            if upper_epsilon_cmp_hi <= 0
            else _ball_from_fraction(penalty.upper_epsilon.exact)
        )
        reciprocal_segment(start, end, lower_side=False)
    if upper_epsilon_cmp_hi > 0:
        start = (
            lower_ball
            if upper_epsilon_cmp_lo >= 0
            else _ball_from_fraction(penalty.upper_epsilon.exact)
        )
        terms.append(_scaled_source_ratio(
            (
                penalty.strength_value,
                _ball_subtract(upper_ball, start),
            ),
            (epsilon, epsilon),
        ))
    return tuple(terms)


def _scalar_objective_difference(
    spec: _CompiledScalarObjective,
    *,
    lower: float,
    upper: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _BinaryScaledBall:
    """Enclose ``F(upper) - F(lower)`` from direct source differences."""

    lower_ball = _TwofoldBall.point(lower)
    upper_ball = _TwofoldBall.point(upper)
    target_ball = _TwofoldBall.point(target)
    terms: list[_BinaryScaledBall] = []

    def append(value: _TwofoldBall | _BinaryScaledBall) -> None:
        terms.append(
            value
            if isinstance(value, _BinaryScaledBall)
            else _binary_scaled_from_ball(value)
        )

    if confidence != 0.0:
        if spec.mismatch_kind == 'squared':
            append(_direct_factored_square_difference_ball(
                lower_ball,
                upper_ball,
                target_ball,
                0.5,
                confidence,
            ))
        else:
            assert spec.huber_delta_value is not None
            delta = spec.huber_delta_value
            lower_boundary = _source_sum_ball(target, -delta)
            upper_boundary = _source_sum_ball(target, delta)

            def compare_lower(value: float) -> int:
                return int(_stable_sum_products_sign(
                    ((value,), (-1.0, target), (delta,))
                ))

            def compare_upper(value: float) -> int:
                return int(_stable_sum_products_sign(
                    ((value,), (-1.0, target), (-delta,))
                ))

            lower_cmp_lo = compare_lower(lower)
            lower_cmp_hi = compare_lower(upper)
            upper_cmp_lo = compare_upper(lower)
            upper_cmp_hi = compare_upper(upper)
            if lower_cmp_lo < 0:
                end = upper_ball if lower_cmp_hi <= 0 else lower_boundary
                append(_scaled_source_product(
                    -confidence,
                    delta,
                    _ball_subtract(end, lower_ball),
                ))
            if lower_cmp_hi > 0 and upper_cmp_lo < 0:
                start = lower_ball if lower_cmp_lo >= 0 else lower_boundary
                end = upper_ball if upper_cmp_hi <= 0 else upper_boundary
                start_residual = _ball_subtract(start, target_ball)
                end_residual = _ball_subtract(end, target_ball)
                append(_scaled_source_product(
                    0.5,
                    confidence,
                    _ball_subtract(end_residual, start_residual),
                    _ball_add(end_residual, start_residual),
                ))
            if upper_cmp_hi > 0:
                start = lower_ball if upper_cmp_lo >= 0 else upper_boundary
                append(_scaled_source_product(
                    confidence,
                    delta,
                    _ball_subtract(upper_ball, start),
                ))

    append(_direct_factored_square_difference_ball(
        lower_ball,
        upper_ball,
        _TwofoldBall.point(v),
        0.5,
        rho,
    ))

    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            if lower < penalty.lower_value:
                end = min(upper, penalty.lower_value)
                append(_direct_factored_square_difference_ball(
                    lower_ball,
                    _TwofoldBall.point(end),
                    _TwofoldBall.point(penalty.lower_value),
                    penalty.strength_value,
                ))
            if upper > penalty.upper_value:
                start = max(lower, penalty.upper_value)
                append(_direct_factored_square_difference_ball(
                    _TwofoldBall.point(start),
                    upper_ball,
                    _TwofoldBall.point(penalty.upper_value),
                    penalty.strength_value,
                ))
            continue
        if penalty.kind == 'reciprocal':
            for value in _direct_reciprocal_difference_terms(
                penalty,
                lower,
                upper,
            ):
                append(value)
            continue
        lower_arguments = _compiled_exponential_arguments(penalty, lower)
        upper_arguments = _compiled_exponential_arguments(penalty, upper)
        assert penalty.tau_value is not None
        exponent_step = _ball_divide(
            _ball_subtract(upper_ball, lower_ball),
            _TwofoldBall.point(penalty.tau_value),
        )
        exponential_difference = _binary_scaled_sum((
            _certified_exp_difference(
                lower_arguments[0],
                upper_arguments[0],
                difference=_ball_negate(exponent_step),
            ),
            _certified_exp_difference(
                lower_arguments[1],
                upper_arguments[1],
                difference=exponent_step,
            ),
        ))
        append(_binary_scaled_multiply(
            _binary_scaled_from_ball(
                _TwofoldBall.point(penalty.strength_value)
            ),
            exponential_difference,
        ))
    return _binary_scaled_sum(terms)


def _array_expm1_polynomial(argument: _ArrayBall) -> _ArrayBall:
    shape = argument.high.shape
    lower, upper = argument.physical_bounds()
    resolved = argument.resolved & (lower >= -0.7) & (upper <= 0.7)
    polynomial = _array_ball_constant(_EXP_COEFFICIENT_BALLS[20], shape)
    for degree in range(19, 0, -1):
        polynomial = _array_ball_add(
            _array_ball_multiply(polynomial, argument),
            _array_ball_constant(_EXP_COEFFICIENT_BALLS[degree], shape),
        )
    polynomial = _array_ball_multiply(argument, polynomial)
    magnitude = np.maximum(np.abs(lower), np.abs(upper))
    remainder = np.ones_like(magnitude)
    for _ in range(21):
        remainder = np.nextafter(remainder * magnitude, np.inf)
    remainder = np.minimum(
        _EXP_REMAINDER_BOUND,
        np.nextafter(remainder * _EXP_REMAINDER_COEFFICIENT, np.inf),
    )
    return _ArrayBall(
        polynomial.high,
        polynomial.low,
        np.nextafter(polynomial.radius + remainder, np.inf),
        polynomial.resolved & resolved,
    )


def _array_factored_square_difference(
    lower: _ArrayBall,
    upper: _ArrayBall,
    center: _ArrayBall,
    *scales: np.ndarray | float,
) -> _ArrayBall:
    value = _array_ball_multiply(
        _array_ball_subtract(upper, lower),
        _array_source_sum(
            upper,
            lower,
            _array_ball_negate(center),
            _array_ball_negate(center),
        ),
    )
    for scale in scales:
        value = _array_ball_multiply(value, _ArrayBall.points(
            np.broadcast_to(np.asarray(scale, dtype=np.float64), value.high.shape)
        ))
    return value


def _array_scalar_objective_difference(
    spec: _CompiledScalarObjective,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    v: np.ndarray,
    rho: float,
) -> _ArrayBall:
    """Vectorized direct terminal differences for moderate adjacent rows."""

    shape = lower.shape
    lower_ball = _ArrayBall.points(lower)
    upper_ball = _ArrayBall.points(upper)
    target_ball = _ArrayBall.points(target)
    total = _ArrayBall.points(np.zeros(shape, dtype=np.float64))
    if spec.mismatch_kind == 'squared':
        total = _array_ball_add(
            total,
            _array_factored_square_difference(
                lower_ball,
                upper_ball,
                target_ball,
                0.5,
                confidence,
            ),
        )
    else:
        assert spec.huber_delta_value is not None
        delta = spec.huber_delta_value
        lower_residual = _array_ball_subtract(lower_ball, target_ball)
        upper_residual = _array_ball_subtract(upper_ball, target_ball)
        lower_residual_lower, _ = lower_residual.physical_bounds()
        _, upper_residual_upper = upper_residual.physical_bounds()
        lower_linear = upper_residual_upper <= -delta
        upper_linear = lower_residual_lower >= delta
        quadratic = (
            (lower_residual_lower >= -delta)
            & (upper_residual_upper <= delta)
        )
        same_branch = lower_linear | upper_linear | quadratic
        linear_sign = np.where(lower_linear, -1.0, 1.0)
        linear = _array_source_product(
            confidence,
            delta,
            linear_sign,
            _array_ball_subtract(upper_ball, lower_ball),
        )
        quadratic_value = _array_factored_square_difference(
            lower_ball,
            upper_ball,
            target_ball,
            0.5,
            confidence,
        )
        mismatch = _ArrayBall(
            np.where(quadratic, quadratic_value.high, linear.high),
            np.where(quadratic, quadratic_value.low, linear.low),
            np.where(quadratic, quadratic_value.radius, linear.radius),
            same_branch & quadratic_value.resolved & linear.resolved,
        )
        total = _array_ball_add(total, mismatch)
    total = _array_ball_add(
        total,
        _array_factored_square_difference(
            lower_ball,
            upper_ball,
            _ArrayBall.points(v),
            0.5,
            rho,
        ),
    )
    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            lower_active = upper <= penalty.lower_value
            upper_active = lower >= penalty.upper_value
            inactive = (
                (lower >= penalty.lower_value)
                & (upper <= penalty.upper_value)
            )
            branch_resolved = lower_active | upper_active | inactive
            active_center = np.where(
                lower_active,
                penalty.lower_value,
                np.where(upper_active, penalty.upper_value, lower),
            )
            contribution = _array_factored_square_difference(
                lower_ball,
                upper_ball,
                _ArrayBall.points(active_center),
                penalty.strength_value,
            )
            contribution = _ArrayBall(
                np.where(inactive, 0.0, contribution.high),
                np.where(inactive, 0.0, contribution.low),
                np.where(inactive, 0.0, contribution.radius),
                contribution.resolved & branch_resolved,
            )
            total = _array_ball_add(total, contribution)
            continue
        if penalty.kind != 'exponential':
            return _ArrayBall(
                total.high,
                total.low,
                total.radius,
                np.zeros(shape, dtype=bool),
            )
        assert penalty.tau_value is not None
        lower_arguments = _array_compiled_exponential_arguments(penalty, lower)
        upper_arguments = _array_compiled_exponential_arguments(penalty, upper)
        exponent_step = _array_ball_divide(
            _array_ball_subtract(upper_ball, lower_ball),
            _ArrayBall.points(np.full(shape, penalty.tau_value)),
        )
        exponential_step = _array_expm1_polynomial(exponent_step)
        lower_base = _array_certified_exp(upper_arguments[0])
        upper_base = _array_certified_exp(lower_arguments[1])
        exponential_difference = _array_ball_multiply(
            _array_ball_subtract(upper_base, lower_base),
            exponential_step,
        )
        total = _array_ball_add(
            total,
            _array_ball_multiply(
                _ArrayBall.points(
                    np.full(shape, penalty.strength_value)
                ),
                exponential_difference,
            ),
        )
    return total


def _finish_exact_expression(
    rational: Fraction,
    exponentials: dict[Fraction, Fraction],
) -> _ExactExponentialExpression:
    rational += exponentials.pop(Fraction(0), Fraction(0))
    terms = tuple(
        (coefficient, exponent)
        for exponent, coefficient in sorted(exponentials.items())
        if coefficient
    )
    return _ExactExponentialExpression(rational, terms)


def _scalar_derivative_exact_expression(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    side: str,
) -> _ExactExponentialExpression:
    """Build one one-sided derivative exactly from binary64 dyadics."""

    point = _fraction(y)
    residual = point - _fraction(target)
    confidence_exact = _fraction(confidence)
    rational = Fraction(0)
    if confidence_exact:
        if spec.mismatch_kind == 'squared':
            rational += confidence_exact * residual
        else:
            assert spec.huber_delta is not None
            rational += confidence_exact * max(
                -spec.huber_delta,
                min(residual, spec.huber_delta),
            )
    rational += _fraction(rho) * (point - _fraction(v))
    exponentials: dict[Fraction, Fraction] = {}

    def add_exponential(exponent: Fraction, coefficient: Fraction) -> None:
        exponentials[exponent] = (
            exponentials.get(exponent, Fraction(0)) + coefficient
        )

    for penalty in spec.penalties:
        if penalty.kind == 'soft':
            if point < penalty.lower:
                rational += 2 * penalty.strength * (point - penalty.lower)
            elif point > penalty.upper:
                rational += 2 * penalty.strength * (point - penalty.upper)
            continue
        if penalty.kind == 'exponential':
            assert penalty.margin is not None
            assert penalty.tau is not None
            coefficient = penalty.strength / penalty.tau
            add_exponential(
                (penalty.lower + penalty.margin - point) / penalty.tau,
                -coefficient,
            )
            add_exponential(
                (point - (penalty.upper - penalty.margin)) / penalty.tau,
                coefficient,
            )
            continue
        assert penalty.margin is not None
        assert penalty.epsilon is not None
        lower_distance = point - penalty.lower
        upper_distance = penalty.upper - point
        if lower_distance == penalty.margin:
            if side == 'minus':
                rational -= penalty.strength / (penalty.margin**2)
        elif lower_distance < penalty.margin:
            denominator = max(lower_distance, penalty.epsilon)
            rational -= penalty.strength / (denominator**2)
        if upper_distance == penalty.margin:
            if side == 'plus':
                rational += penalty.strength / (penalty.margin**2)
        elif upper_distance < penalty.margin:
            denominator = max(upper_distance, penalty.epsilon)
            rational += penalty.strength / (denominator**2)
    return _finish_exact_expression(rational, exponentials)


def _scalar_objective_difference_exact_expression(
    spec: _CompiledScalarObjective,
    *,
    lower: float,
    upper: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _ExactExponentialExpression:
    """Build ``F(upper) - F(lower)`` exactly except for exponentials."""

    target_exact = _fraction(target)
    confidence_exact = _fraction(confidence)
    v_exact = _fraction(v)
    rho_exact = _fraction(rho)

    def rational_value(point: Fraction) -> Fraction:
        residual = point - target_exact
        if spec.mismatch_kind == 'squared':
            value = confidence_exact * residual**2 / 2
        else:
            assert spec.huber_delta is not None
            delta = spec.huber_delta
            value = confidence_exact * (
                residual**2 / 2
                if abs(residual) <= delta
                else delta * (abs(residual) - delta / 2)
            )
        value += rho_exact * (point - v_exact) ** 2 / 2
        for penalty in spec.penalties:
            if penalty.kind == 'soft':
                value += penalty.strength * (
                    max(penalty.lower - point, Fraction(0)) ** 2
                    + max(point - penalty.upper, Fraction(0)) ** 2
                )
            elif penalty.kind == 'reciprocal':
                assert penalty.margin is not None
                assert penalty.epsilon is not None
                for distance in (
                    point - penalty.lower,
                    penalty.upper - point,
                ):
                    if distance >= penalty.margin:
                        continue
                    if distance > penalty.epsilon:
                        value += penalty.strength * (
                            1 / distance - 1 / penalty.margin
                        )
                    else:
                        value += penalty.strength * (
                            1 / penalty.epsilon
                            - 1 / penalty.margin
                            - (distance - penalty.epsilon)
                            / (penalty.epsilon**2)
                        )
        return value

    lower_exact = _fraction(lower)
    upper_exact = _fraction(upper)
    rational = rational_value(upper_exact) - rational_value(lower_exact)
    exponentials: dict[Fraction, Fraction] = {}

    def add_exponential(exponent: Fraction, coefficient: Fraction) -> None:
        exponentials[exponent] = (
            exponentials.get(exponent, Fraction(0)) + coefficient
        )

    for sign, point in ((-1, lower_exact), (1, upper_exact)):
        for penalty in spec.penalties:
            if penalty.kind != 'exponential':
                continue
            assert penalty.margin is not None
            assert penalty.tau is not None
            add_exponential(
                (penalty.lower + penalty.margin - point) / penalty.tau,
                sign * penalty.strength,
            )
            add_exponential(
                (point - (penalty.upper - penalty.margin)) / penalty.tau,
                sign * penalty.strength,
            )
    return _finish_exact_expression(rational, exponentials)


def _fraction_decimal_bound(
    value: Fraction,
    *,
    precision: int,
    rounding: str,
) -> Decimal:
    with localcontext() as context:
        context.prec = int(precision)
        context.rounding = rounding
        return +(Decimal(value.numerator) / Decimal(value.denominator))


def _positive_exponential_interval(
    exponent: Fraction,
    *,
    precision: int,
) -> _DecimalInterval:
    """Enclose ``exp(exponent)`` using Decimal's correct nearest result."""

    work_precision = int(precision) + 12
    argument_lower = _fraction_decimal_bound(
        exponent,
        precision=work_precision,
        rounding=ROUND_FLOOR,
    )
    argument_upper = _fraction_decimal_bound(
        exponent,
        precision=work_precision,
        rounding=ROUND_CEILING,
    )
    with localcontext() as context:
        context.prec = work_precision
        context.rounding = ROUND_HALF_EVEN
        lower_nearest = argument_lower.exp()
        upper_nearest = argument_upper.exp()
        if not lower_nearest.is_finite() or not upper_nearest.is_finite():
            return _DecimalInterval(Decimal(0), Decimal('Infinity'))
        # Decimal.exp is correctly rounded in ROUND_HALF_EVEN.  The adjacent
        # representable values therefore bracket each exact endpoint image;
        # monotonicity extends those bounds to the exact dyadic argument.
        return _DecimalInterval(
            lower_nearest.next_minus(context),
            upper_nearest.next_plus(context),
        )


def _positive_term_interval(
    coefficient: Fraction,
    exponent: Fraction,
    *,
    precision: int,
) -> _DecimalInterval:
    assert coefficient > 0
    exponential = _positive_exponential_interval(
        exponent,
        precision=precision,
    )
    coefficient_lower = _fraction_decimal_bound(
        coefficient,
        precision=precision,
        rounding=ROUND_FLOOR,
    )
    coefficient_upper = _fraction_decimal_bound(
        coefficient,
        precision=precision,
        rounding=ROUND_CEILING,
    )
    with localcontext() as context:
        context.prec = int(precision)
        context.rounding = ROUND_FLOOR
        lower = +(coefficient_lower * exponential.lower)
    with localcontext() as context:
        context.prec = int(precision)
        context.rounding = ROUND_CEILING
        upper = +(coefficient_upper * exponential.upper)
    return _DecimalInterval(lower, upper)


def _exact_expression_interval(
    expression: _ExactExponentialExpression,
    *,
    precision: int,
) -> _DecimalInterval:
    """Return an outward interval for one exact exponential expression."""

    lower = _fraction_decimal_bound(
        expression.rational,
        precision=precision,
        rounding=ROUND_FLOOR,
    )
    upper = _fraction_decimal_bound(
        expression.rational,
        precision=precision,
        rounding=ROUND_CEILING,
    )
    for coefficient, exponent in expression.exponentials:
        magnitude = _positive_term_interval(
            abs(coefficient),
            exponent,
            precision=precision,
        )
        term_lower, term_upper = (
            (magnitude.lower, magnitude.upper)
            if coefficient > 0
            else (
                magnitude.upper.copy_negate(),
                magnitude.lower.copy_negate(),
            )
        )
        with localcontext() as context:
            context.prec = int(precision)
            context.rounding = ROUND_FLOOR
            lower = +(lower + term_lower)
        with localcontext() as context:
            context.prec = int(precision)
            context.rounding = ROUND_CEILING
            upper = +(upper + term_upper)
    return _DecimalInterval(lower, upper)


def _scalar_derivative_fallback_intervals(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    precision: int,
) -> tuple[_DecimalInterval, _DecimalInterval]:
    """Outward one-sided derivative intervals for one bounded fallback."""

    return tuple(
        _exact_expression_interval(
            _scalar_derivative_exact_expression(
                spec,
                y=y,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
                side=side,
            ),
            precision=precision,
        )
        for side in ('minus', 'plus')
    )


def _scalar_derivative_exact_zero(
    spec: _CompiledScalarObjective,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    side: str,
) -> bool:
    """Prove an exact zero only by symbolic term collection."""

    return _scalar_derivative_exact_expression(
        spec,
        y=y,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
        side=side,
    ).symbolic_zero


def _scalar_objective_difference_fallback_interval(
    spec: _CompiledScalarObjective,
    *,
    lower: float,
    upper: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    precision: int,
) -> _DecimalInterval:
    """Outward interval for a terminal direct objective difference."""

    return _exact_expression_interval(
        _scalar_objective_difference_exact_expression(
            spec,
            lower=lower,
            upper=upper,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        ),
        precision=precision,
    )


def _scalar_objective_difference_exact_zero(
    spec: _CompiledScalarObjective,
    *,
    lower: float,
    upper: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> bool:
    """Prove an exact terminal tie only by symbolic term collection."""

    return _scalar_objective_difference_exact_expression(
        spec,
        lower=lower,
        upper=upper,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
    ).symbolic_zero


def _quadratic_row_data(
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> _QuadraticRowData:
    """Construct authoritative quadratic row curvature and RHS values.

    For row ``r``, the returned values are
    ``rho = confidence * alpha**2`` and
    ``rhs = confidence * alpha * (target - beta)``.  The diagnostic
    ``z_obs`` is computed independently and is never used to reconstruct the
    normal RHS.
    """

    alpha_array, beta_array, target_array, confidence_array = np.broadcast_arrays(
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    rho = np.zeros(alpha_array.shape, dtype=np.float64)
    rhs = np.zeros(alpha_array.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if np.any(active):
        rho[active] = _stable_product(
            confidence_array[active],
            alpha_array[active],
            alpha_array[active],
        )
        rhs[active] = _stable_scaled_difference(
            target_array[active],
            beta_array[active],
            confidence_array[active],
            alpha_array[active],
        )
    z_obs = _stable_ratio_difference(
        target_array,
        beta_array,
        alpha_array,
    )
    return _QuadraticRowData(rho=rho, rhs=rhs, z_obs=z_obs)


def _scaled_square_difference_value(
    left: np.ndarray,
    right: np.ndarray,
    *scale_factors: np.ndarray | float,
) -> np.ndarray:
    """Evaluate a scaled square vectorially, repairing exceptional rows."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (left, right, *scale_factors)
        )
    )
    left_array, right_array = arrays[:2]
    scale_arrays = arrays[2:]
    residual, residual_exceptional = _stable_scaled_difference(
        left_array,
        right_array,
        1.0,
        return_exceptional=True,
    )
    value, product_exceptional = _stable_product(
        *scale_arrays,
        residual,
        residual,
        return_exceptional=True,
    )
    zero_scale = np.logical_or.reduce(
        tuple(array == 0.0 for array in scale_arrays)
    )
    exact_zero = zero_scale | (residual == 0.0)
    value[exact_zero] = 0.0
    exceptional = ~exact_zero & (
        residual_exceptional | product_exceptional
    )
    for flat_index in np.flatnonzero(exceptional):
        value.flat[flat_index] = _scaled_square_difference_dd(
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            *(
                float(array.flat[flat_index])
                for array in scale_arrays
            ),
        ).value
    return value


def _weighted_squared_difference_value(
    left: np.ndarray,
    right: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Return ``0.5 * confidence * (left - right)**2`` scale-safely."""

    return _scaled_square_difference_value(
        left,
        right,
        0.5,
        confidence,
    )


def _weighted_squared_affine_value(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Return one weighted affine square without storing its residual."""

    arrays = np.broadcast_arrays(beta, alpha, left, right, target, confidence)
    beta_array, alpha_array, left_array, right_array, target_array, confidence_array = (
        np.asarray(array, dtype=np.float64) for array in arrays
    )

    residual, residual_exceptional = _stable_affine_residual(
        beta_array,
        alpha_array,
        left_array,
        right_array,
        target_array,
        return_exceptional=True,
    )
    value, product_exceptional = _stable_product(
        0.5,
        confidence_array,
        residual,
        residual,
        return_exceptional=True,
    )
    zero_confidence = confidence_array == 0.0
    value[zero_confidence] = 0.0
    exceptional = ~zero_confidence & (
        residual_exceptional | product_exceptional
    )
    for flat_index in np.flatnonzero(exceptional):
        residual_parts = _normalized_affine_parts(
            float(beta_array.flat[flat_index]),
            float(alpha_array.flat[flat_index]),
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(target_array.flat[flat_index]),
        )
        result = None
        if residual_parts is not None:
            normalized_residual, residual_exponent = residual_parts
            factors = (
                _DoubleDouble(0.5),
                _DoubleDouble(float(confidence_array.flat[flat_index])),
                normalized_residual,
                normalized_residual,
            )
            result = _normalized_product_dd(
                factors,
                exponent_offset=2 * residual_exponent,
            )
        if result is not None:
            value.flat[flat_index] = result.value
            continue
        exact_residual = _exact_affine_fraction(
            float(beta_array.flat[flat_index]),
            float(alpha_array.flat[flat_index]),
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(target_array.flat[flat_index]),
        )
        exact = (
            Fraction(1, 2)
            * _fraction(float(confidence_array.flat[flat_index]))
            * exact_residual**2
        )
        value.flat[flat_index] = _fraction_to_extended_float(exact)
    return value


def _mismatch_terms(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
    *,
    evaluate_value: bool = True,
    evaluate_first: bool = True,
    evaluate_second: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rowwise mismatch value and derivatives in one masked pass."""

    y, target_array, confidence_array = np.broadcast_arrays(
        np.asarray(measurement, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    value = np.zeros(y.shape, dtype=np.float64)
    first = np.zeros(y.shape, dtype=np.float64)
    second = np.zeros(y.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if not np.any(active):
        return value, first, second

    active_y = y[active]
    active_target = target_array[active]
    active_confidence = confidence_array[active]
    if isinstance(mismatch, SquaredLoss):
        if evaluate_value:
            value[active] = _weighted_squared_difference_value(
                active_y,
                active_target,
                active_confidence,
            )
        if evaluate_first:
            first[active] = _stable_scaled_difference(
                active_y,
                active_target,
                active_confidence,
            )
        if evaluate_second:
            second[active] = active_confidence
        return value, first, second

    if not isinstance(mismatch, HuberLoss):
        raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')

    delta = float(mismatch.delta)
    residual = _stable_scaled_difference(
        active_y,
        active_target,
        1.0,
    )
    quadratic, direction = _huber_difference_branches(
        active_y,
        active_target,
        delta,
    )
    linear = ~quadratic
    active_value = np.zeros_like(residual)
    active_first = np.zeros_like(residual)
    active_second = np.zeros_like(residual)
    if np.any(quadratic):
        if evaluate_value:
            active_value[quadratic] = _weighted_squared_difference_value(
                active_y[quadratic],
                active_target[quadratic],
                active_confidence[quadratic],
            )
        if evaluate_first:
            active_first[quadratic] = _stable_scaled_difference(
                active_y[quadratic],
                active_target[quadratic],
                active_confidence[quadratic],
            )
        if evaluate_second:
            active_second[quadratic] = active_confidence[quadratic]
    if np.any(linear):
        linear_y = active_y[linear]
        linear_target = active_target[linear]
        linear_confidence = active_confidence[linear]
        if evaluate_value:
            linear_direction = direction[linear]
            linear_value, exceptional = _stable_sum_products(
                (
                    (
                        linear_direction,
                        linear_confidence,
                        delta,
                        linear_y,
                    ),
                    (
                        -linear_direction,
                        linear_confidence,
                        delta,
                        linear_target,
                    ),
                    (-0.5, linear_confidence, delta, delta),
                ),
                return_exceptional=True,
            )
            for flat_index in np.flatnonzero(exceptional):
                linear_value.flat[flat_index] = _huber_linear_value_dd(
                    float(linear_y.flat[flat_index]),
                    float(linear_target.flat[flat_index]),
                    delta,
                    float(linear_confidence.flat[flat_index]),
                ).value
            active_value[linear] = linear_value
        if evaluate_first:
            active_first[linear] = _stable_product(
                direction[linear],
                linear_confidence,
                delta,
            )
    value[active] = active_value
    first[active] = active_first
    second[active] = active_second
    return value, first, second


def _mismatch_values_from_affine(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    """Evaluate mismatch values without requiring representable residuals."""

    arrays = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )
    (
        beta_array,
        alpha_array,
        left_array,
        right_array,
        target_array,
        confidence_array,
    ) = arrays
    value = np.zeros(beta_array.shape, dtype=np.float64)
    active = confidence_array != 0.0
    if not np.any(active):
        return value

    active_beta = beta_array[active]
    active_alpha = alpha_array[active]
    active_left = left_array[active]
    active_right = right_array[active]
    active_target = target_array[active]
    active_confidence = confidence_array[active]
    if isinstance(mismatch, SquaredLoss):
        value[active] = _weighted_squared_affine_value(
            active_beta,
            active_alpha,
            active_left,
            active_right,
            active_target,
            active_confidence,
        )
        return value
    if not isinstance(mismatch, HuberLoss):
        raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')

    delta = float(mismatch.delta)
    residual = _stable_affine_residual(
        active_beta,
        active_alpha,
        active_left,
        active_right,
        active_target,
    )
    quadratic, direction = _huber_affine_branches(
        active_beta,
        active_alpha,
        active_left,
        active_right,
        active_target,
        delta,
    )
    active_value = np.empty(residual.shape, dtype=np.float64)
    if np.any(quadratic):
        active_value[quadratic] = _weighted_squared_affine_value(
            active_beta[quadratic],
            active_alpha[quadratic],
            active_left[quadratic],
            active_right[quadratic],
            active_target[quadratic],
            active_confidence[quadratic],
        )
    linear = ~quadratic
    if np.any(linear):
        linear_direction = direction[linear]
        linear_confidence = active_confidence[linear]
        active_value[linear] = _stable_sum_products(
            (
                (
                    linear_direction,
                    linear_confidence,
                    delta,
                    active_beta[linear],
                ),
                (
                    linear_direction,
                    linear_confidence,
                    delta,
                    active_alpha[linear],
                    active_left[linear],
                ),
                (
                    -linear_direction,
                    linear_confidence,
                    delta,
                    active_alpha[linear],
                    active_right[linear],
                ),
                (
                    -linear_direction,
                    linear_confidence,
                    delta,
                    active_target[linear],
                ),
                (-0.5, linear_confidence, delta, delta),
            )
        )
    value[active] = active_value
    return value


def _mismatch_values(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_first=False,
        evaluate_second=False,
    )[0]


def _mismatch_first_derivative(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_value=False,
        evaluate_second=False,
    )[1]


def _mismatch_second_derivative(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _mismatch_terms(
        measurement,
        target,
        confidence,
        mismatch,
        evaluate_value=False,
        evaluate_first=False,
    )[2]


def _reciprocal_masks(
    distance: np.ndarray,
    *,
    epsilon: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    continuation = distance <= epsilon
    reciprocal = (distance > epsilon) & (distance < margin)
    return continuation, reciprocal


def _reciprocal_boundary_value(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    continuation, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(continuation):
        continuation_distance = distance[continuation]
        base = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(margin, epsilon, 1.0),
            ),
            (epsilon, margin),
        )
        correction = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(
                    continuation_distance,
                    epsilon,
                    1.0,
                ),
            ),
            (epsilon, epsilon),
        )
        combined = _stable_sum(base, -correction)
        exceptional = ~np.isfinite(combined)
        for flat_index in np.flatnonzero(exceptional):
            d = float(continuation_distance.flat[flat_index])
            if not math.isfinite(d):
                combined.flat[flat_index] = float('inf')
                continue
            combined.flat[flat_index] = _exact_sum_ratios_scalar(
                (
                    ((strength, margin), (epsilon, margin)),
                    ((-strength, epsilon), (epsilon, margin)),
                    ((-strength, d), (epsilon, epsilon)),
                    ((strength, epsilon), (epsilon, epsilon)),
                )
            )
        result[continuation] = combined
    if np.any(reciprocal):
        reciprocal_distance = distance[reciprocal]
        result[reciprocal] = _stable_ratio_product(
            (
                strength,
                _stable_scaled_difference(
                    margin,
                    reciprocal_distance,
                    1.0,
                ),
            ),
            (reciprocal_distance, margin),
        )
    return result


def _reciprocal_boundary_value_from_operands(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    """Evaluate ``q(left - right)`` without requiring a finite difference."""

    return _reciprocal_boundary_terms_from_operands(
        left,
        right,
        margin=margin,
        epsilon=epsilon,
        strength=strength,
        evaluate_value=True,
        evaluate_first=False,
        evaluate_second=False,
    )[0]


def _reciprocal_boundary_first(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    continuation, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(continuation):
        result[continuation] = -_stable_ratio_product(
            (strength,),
            (epsilon, epsilon),
        )
    if np.any(reciprocal):
        d = distance[reciprocal]
        result[reciprocal] = -_stable_ratio_product(
            (strength,),
            (d, d),
        )
    return result


def _reciprocal_boundary_second(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    result = np.zeros_like(distance)
    _, reciprocal = _reciprocal_masks(
        distance,
        epsilon=epsilon,
        margin=margin,
    )
    if np.any(reciprocal):
        d = distance[reciprocal]
        result[reciprocal] = _stable_ratio_product(
            (2.0, strength),
            (d, d, d),
        )
    return result


def _reciprocal_branch_uncertain(
    distance: np.ndarray,
    expression_scale: np.ndarray,
    *,
    epsilon: float,
    margin: float,
    operation_count: int,
) -> np.ndarray:
    """Identify distances whose rounded value may cross a branch boundary."""

    scale = np.maximum(
        np.asarray(expression_scale, dtype=np.float64),
        max(abs(float(epsilon)), abs(float(margin))),
    )
    error_bound = _stable_product(
        scale,
        max(1, int(operation_count)) * _FLOAT_EPSILON,
    )
    uncertain = np.zeros(distance.shape, dtype=bool)
    for boundary in (epsilon, margin):
        gap = _stable_scaled_difference(distance, boundary, 1.0)
        uncertain |= np.abs(gap) <= error_bound
    return uncertain


def _exact_reciprocal_boundary_terms(
    distance: Fraction,
    *,
    margin: float,
    epsilon: float,
    strength: float,
    evaluate_value: bool,
    evaluate_first: bool,
    evaluate_second: bool,
) -> tuple[float, float, float]:
    """Evaluate one reciprocal branch exactly from binary64 operands."""

    margin_fraction = _fraction(margin)
    epsilon_fraction = _fraction(epsilon)
    strength_fraction = _fraction(strength)
    value = Fraction(0)
    first = Fraction(0)
    second = Fraction(0)
    if distance < margin_fraction:
        if distance > epsilon_fraction:
            if evaluate_value:
                value = strength_fraction * (
                    Fraction(1, 1) / distance
                    - Fraction(1, 1) / margin_fraction
                )
            if evaluate_first:
                first = -strength_fraction / (distance * distance)
            if evaluate_second:
                second = (
                    2 * strength_fraction
                    / (distance * distance * distance)
                )
        else:
            if evaluate_value:
                value = strength_fraction * (
                    Fraction(1, 1) / epsilon_fraction
                    - Fraction(1, 1) / margin_fraction
                    - (distance - epsilon_fraction)
                    / (epsilon_fraction * epsilon_fraction)
                )
            if evaluate_first:
                first = -strength_fraction / (
                    epsilon_fraction * epsilon_fraction
                )
    return (
        _fraction_to_extended_float(value) if evaluate_value else 0.0,
        _fraction_to_extended_float(first) if evaluate_first else 0.0,
        _fraction_to_extended_float(second) if evaluate_second else 0.0,
    )


def _reciprocal_boundary_terms_from_operands(
    left: np.ndarray | float,
    right: np.ndarray | float,
    *,
    margin: float,
    epsilon: float,
    strength: float,
    evaluate_value: bool,
    evaluate_first: bool,
    evaluate_second: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate reciprocal terms with branch-safe operand comparisons."""

    left_array, right_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    distance = _stable_scaled_difference(left_array, right_array, 1.0)
    value = (
        _reciprocal_boundary_value(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_value
        else np.zeros_like(distance)
    )
    first = (
        _reciprocal_boundary_first(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_first
        else np.zeros_like(distance)
    )
    second = (
        _reciprocal_boundary_second(
            distance,
            margin=margin,
            epsilon=epsilon,
            strength=strength,
        )
        if evaluate_second
        else np.zeros_like(distance)
    )
    finite_operands = np.isfinite(left_array) & np.isfinite(right_array)
    expression_scale = np.maximum(
        np.maximum(np.abs(left_array), np.abs(right_array)),
        np.abs(distance),
    )
    exceptional = finite_operands & (
        ~np.isfinite(distance)
        | _reciprocal_branch_uncertain(
            distance,
            expression_scale,
            epsilon=epsilon,
            margin=margin,
            operation_count=3,
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact_distance = (
            _fraction(float(left_array.flat[flat_index]))
            - _fraction(float(right_array.flat[flat_index]))
        )
        exact_value, exact_first, exact_second = (
            _exact_reciprocal_boundary_terms(
                exact_distance,
                margin=margin,
                epsilon=epsilon,
                strength=strength,
                evaluate_value=evaluate_value,
                evaluate_first=evaluate_first,
                evaluate_second=evaluate_second,
            )
        )
        if evaluate_value:
            value.flat[flat_index] = exact_value
        if evaluate_first:
            first.flat[flat_index] = exact_first
        if evaluate_second:
            second.flat[flat_index] = exact_second
    return value, first, second


def _penalty_value(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise value for one scalar penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)
    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        for mask, boundary in (
            (y < float(penalty.lower), float(penalty.lower)),
            (y > float(penalty.upper), float(penalty.upper)),
        ):
            if np.any(mask):
                result[mask] = _scaled_square_difference_value(
                    y[mask],
                    boundary,
                    strength,
                )
        return result
    compiled = _compile_scalar_objective(
        SquaredLoss(),
        (penalty,),
    ).penalties[0]
    return np.fromiter(
        (
            _compiled_penalty_scalar_value(compiled, float(value))
            for value in y.flat
        ),
        dtype=np.float64,
        count=y.size,
    ).reshape(y.shape)


def _fraction_to_hard_endpoint_float(value: Fraction) -> float:
    """Round a hard-bound endpoint, saturating only to finite float64."""

    if value > _FLOAT_MAX_FRACTION:
        return np.finfo(np.float64).max
    if value < -_FLOAT_MAX_FRACTION:
        return -np.finfo(np.float64).max
    return float(value)


def _fraction_to_extended_float(value: Fraction) -> float:
    """Round an objective value, preserving true range overflow as infinity."""

    try:
        return float(value)
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


def _exact_affine_fraction(
    beta: float,
    alpha: float,
    left: float,
    right: float,
    target: float,
) -> Fraction:
    """Return an exact affine residual for finite binary64 operands."""

    return (
        _fraction(beta)
        + _fraction(alpha) * _fraction(left)
        - _fraction(alpha) * _fraction(right)
        - _fraction(target)
    )


def _penalty_value_from_affine(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Evaluate a penalty without requiring a representable prediction."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)
    compiled = _compile_scalar_objective(
        SquaredLoss(),
        (penalty,),
    ).penalties[0]
    beta_array, alpha_array, left_array, right_array, y_array = (
        np.broadcast_arrays(
            np.asarray(beta, dtype=np.float64),
            np.asarray(alpha, dtype=np.float64),
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            y,
        )
    )

    def row_value(flat_index: int) -> float:
        measurement_parts = _normalized_affine_parts(
            float(beta_array.flat[flat_index]),
            float(alpha_array.flat[flat_index]),
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
        )
        measurement_expansion = (
            None
            if measurement_parts is None
            else _materialize_normalized_dd(measurement_parts)
        )
        if measurement_expansion is None:
            exact_measurement = _exact_affine_fraction(
                float(beta_array.flat[flat_index]),
                float(alpha_array.flat[flat_index]),
                float(left_array.flat[flat_index]),
                float(right_array.flat[flat_index]),
                0.0,
            )
            if compiled.kind == 'soft':
                return _compiled_soft_fraction_value(
                    compiled,
                    exact_measurement,
                )
            if compiled.kind == 'reciprocal':
                return _compiled_reciprocal_fraction_value(
                    compiled,
                    exact_measurement,
                )
            return _compiled_exponential_fraction_value(
                compiled,
                exact_measurement,
            )
        return _compiled_penalty_expansion_value(
            compiled,
            measurement_expansion,
        )

    return np.fromiter(
        (row_value(index) for index in range(y_array.size)),
        dtype=np.float64,
        count=y_array.size,
    ).reshape(y_array.shape)


def _penalty_first_derivative(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise first derivative for one penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        lower_mask = y < float(penalty.lower)
        upper_mask = y > float(penalty.upper)
        if np.any(lower_mask):
            result[lower_mask] = _stable_scaled_difference(
                y[lower_mask],
                float(penalty.lower),
                2.0,
                strength,
            )
        if np.any(upper_mask):
            result[upper_mask] = _stable_scaled_difference(
                y[upper_mask],
                float(penalty.upper),
                2.0,
                strength,
            )
        return result

    if isinstance(penalty, ExponentialBoundaryPenalty):
        compiled = _compile_scalar_objective(
            SquaredLoss(),
            (penalty,),
        ).penalties[0]
        return np.fromiter(
            (
                _compiled_exponential_derivative_values(
                    compiled,
                    float(value),
                )[0]
                for value in y.flat
            ),
            dtype=np.float64,
            count=y.size,
        ).reshape(y.shape)

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_first = _reciprocal_boundary_terms_from_operands(
            y,
            float(penalty.lower),
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=True,
            evaluate_second=False,
        )[1]
        upper_first = _reciprocal_boundary_terms_from_operands(
            float(penalty.upper),
            y,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=True,
            evaluate_second=False,
        )[1]
        return _stable_sum(lower_first, -upper_first)

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _penalty_second_derivative(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
) -> np.ndarray:
    """Return the authoritative rowwise second derivative for one penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    strength = float(penalty.strength)
    if strength == 0.0:
        return np.zeros_like(y)

    if isinstance(penalty, SoftIntervalPenalty):
        result = np.zeros_like(y)
        result[(y < float(penalty.lower)) | (y > float(penalty.upper))] = (
            2.0 * strength
        )
        return result

    if isinstance(penalty, ExponentialBoundaryPenalty):
        compiled = _compile_scalar_objective(
            SquaredLoss(),
            (penalty,),
        ).penalties[0]
        return np.fromiter(
            (
                _compiled_exponential_derivative_values(
                    compiled,
                    float(value),
                )[1]
                for value in y.flat
            ),
            dtype=np.float64,
            count=y.size,
        ).reshape(y.shape)

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower_second = _reciprocal_boundary_terms_from_operands(
            y,
            float(penalty.lower),
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=False,
            evaluate_second=True,
        )[2]
        upper_second = _reciprocal_boundary_terms_from_operands(
            float(penalty.upper),
            y,
            margin=float(penalty.margin),
            epsilon=float(penalty.epsilon),
            strength=strength,
            evaluate_value=False,
            evaluate_first=False,
            evaluate_second=True,
        )[2]
        return _stable_sum(lower_second, upper_second)

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _penalty_terms(
    measurement: np.ndarray,
    penalty: ScalarPenalty,
    *,
    evaluate_value: bool = True,
    evaluate_first: bool = True,
    evaluate_second: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return authoritative rowwise value and derivatives for one penalty."""

    y = np.asarray(measurement, dtype=np.float64)
    return (
        _penalty_value(y, penalty) if evaluate_value else np.zeros_like(y),
        (
            _penalty_first_derivative(y, penalty)
            if evaluate_first
            else np.zeros_like(y)
        ),
        (
            _penalty_second_derivative(y, penalty)
            if evaluate_second
            else np.zeros_like(y)
        ),
    )


def _l2_value(
    weights: np.ndarray,
    reference: np.ndarray,
    strength: float,
) -> float:
    """Return ``0.5 * strength * ||weights - reference||**2`` stably."""

    strength_value = float(strength)
    if strength_value == 0.0:
        return 0.0
    weights_array, reference_array = np.broadcast_arrays(
        np.asarray(weights, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    terms = _scaled_square_difference_value(
        weights_array,
        reference_array,
        0.5,
        strength_value,
    )
    if np.any(np.isinf(terms)):
        return math.inf
    try:
        return math.fsum(terms.ravel().tolist())
    except OverflowError:
        return math.inf


def _hard_tolerance(*values: np.ndarray | float) -> np.ndarray:
    """Return the shared float64 absolute-plus-relative hard tolerance."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in values)
    )
    scale = np.zeros_like(arrays[0], dtype=np.float64)
    for value in arrays:
        scale = np.maximum(scale, np.abs(value))
    relative = _stable_product(HARD_RTOL, scale)
    return _stable_sum(HARD_ATOL, relative)


def _hard_row_status(
    lower: np.ndarray,
    measurement: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return authoritative row satisfaction, raw violation, and tolerance."""

    lower_array, measurement_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64),
        np.asarray(measurement, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    lower_gap = _stable_scaled_difference(
        lower_array,
        measurement_array,
        1.0,
    )
    upper_gap = _stable_scaled_difference(
        measurement_array,
        upper_array,
        1.0,
    )
    violation = np.maximum(np.maximum(lower_gap, upper_gap), 0.0)
    tolerance = _hard_tolerance(
        lower_array,
        measurement_array,
        upper_array,
    )
    finite = (
        np.isfinite(lower_array)
        & np.isfinite(measurement_array)
        & np.isfinite(upper_array)
    )
    satisfied = finite & (violation <= tolerance)
    return satisfied, violation, tolerance


def _hard_row_satisfied_scalar(
    lower: float,
    measurement: float,
    upper: float,
) -> bool:
    if not (
        math.isfinite(lower)
        and math.isfinite(measurement)
        and math.isfinite(upper)
    ):
        return False
    violation = max(
        _stable_scaled_difference_scalar(lower, measurement, 1.0),
        _stable_scaled_difference_scalar(measurement, upper, 1.0),
        0.0,
    )
    relative = _stable_product_scalar(
        HARD_RTOL,
        max(
            abs(lower),
            abs(measurement),
            abs(upper),
        ),
    )
    tolerance = _stable_sum_scalar(HARD_ATOL, relative)
    return violation <= tolerance


def _tight_lower_accepted_bound(
    lower: float,
    upper: float,
    candidate: float,
) -> float:
    value = (
        -np.finfo(np.float64).max
        if candidate == float('-inf')
        else candidate
    )
    while not _hard_row_satisfied_scalar(lower, value, upper):
        value = math.nextafter(value, float('inf'))
    while True:
        previous = math.nextafter(value, float('-inf'))
        if not _hard_row_satisfied_scalar(lower, previous, upper):
            return value
        value = previous


def _tight_upper_accepted_bound(
    lower: float,
    upper: float,
    candidate: float,
) -> float:
    value = (
        np.finfo(np.float64).max
        if candidate == float('inf')
        else candidate
    )
    while not _hard_row_satisfied_scalar(lower, value, upper):
        value = math.nextafter(value, float('-inf'))
    while True:
        following = math.nextafter(value, float('inf'))
        if not _hard_row_satisfied_scalar(lower, following, upper):
            return value
        value = following


def _hard_accepted_measurement_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact finite interval accepted by the public row predicate."""

    lower_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    accepted_lower = np.empty(lower_array.shape, dtype=np.float64)
    accepted_upper = np.empty(upper_array.shape, dtype=np.float64)
    atol_fraction = _fraction(HARD_ATOL)
    rtol_fraction = _fraction(HARD_RTOL)
    one_minus_rtol = Fraction(1) - rtol_fraction
    for index in np.ndindex(lower_array.shape):
        lower_value = float(lower_array[index])
        upper_value = float(upper_array[index])
        if not (math.isfinite(lower_value) and math.isfinite(upper_value)):
            accepted_lower[index] = float('nan')
            accepted_upper[index] = float('nan')
            continue
        scale = max(abs(lower_value), abs(upper_value))
        scale_fraction = _fraction(scale)
        lower_fraction = _fraction(lower_value)
        upper_fraction = _fraction(upper_value)
        fixed_tolerance = atol_fraction + rtol_fraction * scale_fraction
        lower_candidate = lower_fraction - fixed_tolerance
        if abs(lower_candidate) <= scale_fraction:
            approximate_lower_fraction = lower_candidate
        else:
            approximate_lower_fraction = (
                lower_fraction - atol_fraction
            ) / one_minus_rtol
        upper_candidate = upper_fraction + fixed_tolerance
        if abs(upper_candidate) <= scale_fraction:
            approximate_upper_fraction = upper_candidate
        else:
            approximate_upper_fraction = (
                upper_fraction + atol_fraction
            ) / one_minus_rtol
        approximate_lower = _fraction_to_hard_endpoint_float(
            approximate_lower_fraction
        )
        approximate_upper = _fraction_to_hard_endpoint_float(
            approximate_upper_fraction
        )
        accepted_lower[index] = _tight_lower_accepted_bound(
            lower_value,
            upper_value,
            approximate_lower,
        )
        accepted_upper[index] = _tight_upper_accepted_bound(
            lower_value,
            upper_value,
            approximate_upper,
        )
    return accepted_lower, accepted_upper
