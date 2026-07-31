"""Scale-safe private arithmetic for separator inverse computations."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


_FLOAT_MAX = np.finfo(np.float64).max
_FLOAT_TINY = np.finfo(np.float64).tiny
_FLOAT_EPSILON = np.finfo(np.float64).eps
_CONDITIONING_RELATIVE_LIMIT = math.sqrt(_FLOAT_EPSILON)
_AFFINE_CONDITIONING_RELATIVE_LIMIT = 64.0 * _FLOAT_EPSILON
_NORMAL_MIN_EXPONENT = -1021
_NORMAL_MAX_EXPONENT = 1023
_SPLITTER = 134217729.0
_SIGN_BIT = 1 << 63
_UINT64_MASK = (1 << 64) - 1
_ORDERED_ZERO = _SIGN_BIT
_SPLIT_LOW_BITS = 26


@dataclass(frozen=True, slots=True)
class _DoubleDouble:
    """Two-term nonoverlapping binary64 expansion.

    The expansion is used only on the ordinary scalar-kernel path.  It keeps
    the rounding correction from affine products, ratios, and cancellation in
    binary64 storage; it is not an arbitrary-precision number.
    """

    high: float
    low: float = 0.0

    @property
    def value(self) -> float:
        return math.fsum((self.high, self.low))


@dataclass(frozen=True, slots=True)
class _TwofoldBall:
    """A normalized two-limb center with an explicit outward radius.

    The represented real set is ``high + low +/- radius``.  Finite source
    binary64 values enter as exact point balls.  Every operation below either
    carries all discarded expansion limbs and input radii into ``radius`` or
    returns :meth:`unresolved`; a naked center is never certificate evidence.

    ``resolved`` is deliberately explicit.  Exceptional arithmetic may be
    evaluated by the bounded exact fallback, but it may not manufacture an
    ordinary sign from a non-finite or unproved center operation.
    """

    high: float
    low: float = 0.0
    radius: float = 0.0
    resolved: bool = True

    @classmethod
    def point(cls, value: float) -> '_TwofoldBall':
        scalar = float(value)
        if not math.isfinite(scalar):
            return cls.unresolved()
        return cls(scalar, 0.0, 0.0, True)

    @classmethod
    def unresolved(cls) -> '_TwofoldBall':
        return cls(0.0, 0.0, math.inf, False)

    @property
    def center(self) -> _DoubleDouble:
        return _DoubleDouble(self.high, self.low)

    @property
    def center_value(self) -> float:
        if not self.resolved:
            return math.nan
        return math.fsum((self.high, self.low))

    def physical_bounds(self) -> tuple[float, float]:
        """Return directed binary64 bounds for the complete ball."""

        if (
            not self.resolved
            or not math.isfinite(self.high)
            or not math.isfinite(self.low)
            or not math.isfinite(self.radius)
            or self.radius < 0.0
        ):
            return -math.inf, math.inf
        lower = _down_add(_down_add(self.high, self.low), -self.radius)
        upper = _up_add(_up_add(self.high, self.low), self.radius)
        return lower, upper

    @property
    def strictly_negative(self) -> bool:
        return self.physical_bounds()[1] < 0.0

    @property
    def strictly_positive(self) -> bool:
        return self.physical_bounds()[0] > 0.0


@dataclass(frozen=True, slots=True)
class _BinaryScaledBall:
    """A twofold ball multiplied by an exact power of two.

    Signs are decided from the normalized ball and therefore never require
    materializing an overflowing physical value.  Physical conversion exists
    only for diagnostics and is directed at the ``ldexp`` boundary.
    """

    ball: _TwofoldBall
    exponent: int = 0

    @classmethod
    def unresolved(cls) -> '_BinaryScaledBall':
        return cls(_TwofoldBall.unresolved(), 0)

    @property
    def lower(self) -> float:
        return self.ball.physical_bounds()[0]

    @property
    def upper(self) -> float:
        return self.ball.physical_bounds()[1]

    @property
    def strictly_negative(self) -> bool:
        return self.ball.strictly_negative

    @property
    def strictly_positive(self) -> bool:
        return self.ball.strictly_positive

    @property
    def center_value(self) -> float:
        return self.ball.center_value

    @property
    def log_scale(self) -> float:
        """Legacy proposal-only view; never used to authorize a sign."""

        return self.exponent * math.log(2.0)

    def physical_bounds(self) -> tuple[float, float]:
        lower, upper = self.ball.physical_bounds()
        if not math.isfinite(lower) or not math.isfinite(upper):
            return lower, upper
        return (
            _directed_ldexp(lower, self.exponent, upward=False),
            _directed_ldexp(upper, self.exponent, upward=True),
        )


@dataclass(frozen=True, slots=True)
class _ArrayBall:
    """Vectorized twofold balls for heterogeneous ordinary proximal rows."""

    high: np.ndarray
    low: np.ndarray
    radius: np.ndarray
    resolved: np.ndarray

    @classmethod
    def points(cls, values: object) -> '_ArrayBall':
        high = np.asarray(values, dtype=np.float64)
        return cls(
            high=high,
            low=np.zeros_like(high),
            radius=np.zeros_like(high),
            resolved=np.isfinite(high),
        )

    def physical_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = _array_down_add(
            _array_down_add(self.high, self.low),
            -self.radius,
        )
        upper = _array_up_add(
            _array_up_add(self.high, self.low),
            self.radius,
        )
        lower = np.where(self.resolved, lower, -np.inf)
        upper = np.where(self.resolved, upper, np.inf)
        return lower, upper


@dataclass(frozen=True, slots=True)
class _ScaledEnclosure:
    """Outward interval stored relative to one natural-log scale."""

    lower: float
    upper: float
    log_scale: float

    @property
    def strictly_negative(self) -> bool:
        return self.upper < 0.0

    @property
    def strictly_positive(self) -> bool:
        return self.lower > 0.0

    def physical_bounds(self) -> tuple[float, float]:
        lower = _scaled_physical_float(self.lower, self.log_scale)
        upper = _scaled_physical_float(self.upper, self.log_scale)
        if not math.isfinite(lower) or not math.isfinite(upper):
            return lower, upper
        magnitude = max(abs(lower), abs(upper))
        log_rounding = (
            16.0
            * _FLOAT_EPSILON
            * (1.0 + abs(self.log_scale))
            * magnitude
        )
        radius = log_rounding + 4.0 * max(
            math.ulp(lower),
            math.ulp(upper),
        )
        return (
            math.nextafter(lower - radius, -math.inf),
            math.nextafter(upper + radius, math.inf),
        )


def _float_to_ordered_int(value: float) -> int:
    """Map a numeric binary64 value to a monotone lattice integer.

    The two encodings of zero intentionally map to the same key.  Negative
    keys are shifted by one so ``-minimum_subnormal``, numeric zero, and
    ``+minimum_subnormal`` are consecutive lattice points.
    """

    scalar = float(value)
    if math.isnan(scalar):
        raise ValueError('ordered binary64 values cannot be NaN')
    if scalar == 0.0:
        return _ORDERED_ZERO
    bits = struct.unpack('>Q', struct.pack('>d', scalar))[0]
    if bits & _SIGN_BIT:
        return ((~bits) & _UINT64_MASK) + 1
    return bits | _SIGN_BIT


def _ordered_int_to_float(value: int) -> float:
    """Invert :func:`_float_to_ordered_int`."""

    ordered = int(value)
    if not 0 <= ordered <= _UINT64_MASK:
        raise ValueError('ordered binary64 integer is out of range')
    if ordered == _ORDERED_ZERO:
        return 0.0
    if ordered > _ORDERED_ZERO:
        bits = ordered & ~_SIGN_BIT
    else:
        raw_ordered = ordered - 1
        if raw_ordered < 0:
            raise ValueError('ordered integer is below the numeric lattice')
        bits = (~raw_ordered) & _UINT64_MASK
    scalar = struct.unpack('>d', struct.pack('>Q', bits))[0]
    if math.isnan(scalar):
        raise ValueError('ordered integer maps to NaN')
    return scalar


def _ordered_float_midpoint(lower: float, upper: float) -> float:
    """Return a representable binary64 point strictly between two values."""

    lower_key = _float_to_ordered_int(lower)
    upper_key = _float_to_ordered_int(upper)
    if not float(lower) < float(upper) or lower_key >= upper_key:
        raise ValueError('ordered midpoint requires lower < upper')
    if upper_key - lower_key <= 1:
        raise ValueError('ordered midpoint requires a non-adjacent interval')
    middle_key = lower_key + (upper_key - lower_key) // 2
    if middle_key == lower_key:
        middle_key += 1
    result = _ordered_int_to_float(middle_key)
    if not float(lower) < result < float(upper):
        raise ArithmeticError('ordered midpoint did not contract numerically')
    return result


def _ordered_floats_adjacent(lower: float, upper: float) -> bool:
    """Return whether no binary64 value lies strictly between the inputs."""

    if not float(lower) < float(upper):
        return False
    return (
        _float_to_ordered_int(upper)
        - _float_to_ordered_int(lower)
        == 1
    )


def _two_sum_scalar(left: float, right: float) -> _DoubleDouble:
    """Return the exact finite binary64 sum as a two-term expansion."""

    high = float(left) + float(right)
    virtual_right = high - float(left)
    low = (
        float(left) - (high - virtual_right)
        + (float(right) - virtual_right)
    )
    return _DoubleDouble(high, low)


def _split_scalar(value: float) -> tuple[float, float]:
    """Split one finite value without an overflowing arithmetic multiplier."""

    scalar = float(value)
    if scalar == 0.0:
        return scalar, scalar
    if not math.isfinite(scalar):
        return scalar, 0.0
    bits = struct.unpack('>Q', struct.pack('>d', scalar))[0]
    high_bits = bits & ~((1 << _SPLIT_LOW_BITS) - 1)
    high = struct.unpack('>d', struct.pack('>Q', high_bits))[0]
    return high, scalar - high


def _two_product_scalar(left: float, right: float) -> _DoubleDouble:
    """Return a product and its representable exact rounding correction."""

    left_value = float(left)
    right_value = float(right)
    high = left_value * right_value
    if not math.isfinite(high) or left_value == 0.0 or right_value == 0.0:
        return _DoubleDouble(high, 0.0)
    if abs(high) < _FLOAT_TINY:
        return _DoubleDouble(high, 0.0)

    left_mantissa, left_exponent = math.frexp(left_value)
    right_mantissa, right_exponent = math.frexp(right_value)
    mantissa_product = left_mantissa * right_mantissa
    left_scaled = _SPLITTER * left_mantissa
    left_high = left_scaled - (left_scaled - left_mantissa)
    left_low = left_mantissa - left_high
    right_scaled = _SPLITTER * right_mantissa
    right_high = right_scaled - (right_scaled - right_mantissa)
    right_low = right_mantissa - right_high
    mantissa_error = (
        (
            (left_high * right_high - mantissa_product)
            + left_high * right_low
        )
        + left_low * right_high
        + left_low * right_low
    )
    try:
        low = math.ldexp(
            mantissa_error,
            left_exponent + right_exponent,
        )
    except OverflowError:
        low = 0.0
    return _DoubleDouble(high, low)


def _dd_normalize(high: float, low: float) -> _DoubleDouble:
    summed = _two_sum_scalar(float(high), float(low))
    return _DoubleDouble(summed.high, summed.low)


def _dd_add(left: _DoubleDouble, right: _DoubleDouble) -> _DoubleDouble:
    """Add two short expansions with one final error-free normalization."""

    high = _two_sum_scalar(left.high, right.high)
    low = math.fsum((left.low, right.low, high.low))
    return _dd_normalize(high.high, low)


def _dd_negate(value: _DoubleDouble) -> _DoubleDouble:
    return _DoubleDouble(-value.high, -value.low)


def _dd_difference(left: float, right: float) -> _DoubleDouble:
    return _two_sum_scalar(float(left), -float(right))


def _dd_multiply_float(value: _DoubleDouble, factor: float) -> _DoubleDouble:
    """Multiply an expansion by one binary64 factor."""

    main = _two_product_scalar(value.high, float(factor))
    if not math.isfinite(main.high):
        return _DoubleDouble(main.high, 0.0)
    correction = value.low * float(factor)
    try:
        low = math.fsum((main.low, correction))
    except OverflowError:
        low = correction
    if not math.isfinite(low):
        return _DoubleDouble(low, 0.0)
    return _dd_normalize(main.high, low)


def _dd_multiply(
    left: _DoubleDouble,
    right: _DoubleDouble,
) -> _DoubleDouble:
    """Multiply two short expansions to double-double accuracy."""

    main = _two_product_scalar(left.high, right.high)
    if not math.isfinite(main.high):
        return _DoubleDouble(main.high, 0.0)
    try:
        correction = math.fsum(
            (
                main.low,
                left.high * right.low,
                left.low * right.high,
                left.low * right.low,
            )
        )
    except OverflowError:
        return _DoubleDouble(
            math.copysign(math.inf, left.high * right.high),
            0.0,
        )
    if not math.isfinite(correction):
        return _DoubleDouble(correction, 0.0)
    return _dd_normalize(main.high, correction)


def _dd_divide_float(value: _DoubleDouble, divisor: float) -> _DoubleDouble:
    """Divide an expansion by a finite nonzero binary64 divisor."""

    denominator = float(divisor)
    if denominator == 0.0 or not math.isfinite(denominator):
        return _DoubleDouble(value.value / denominator, 0.0)
    quotient = value.high / denominator
    if not math.isfinite(quotient):
        return _DoubleDouble(quotient, 0.0)
    product = _two_product_scalar(quotient, denominator)
    residual = math.fsum(
        (value.high, -product.high, value.low, -product.low)
    )
    correction = residual / denominator
    return _dd_normalize(quotient, correction)


def _dd_divide(
    numerator: _DoubleDouble,
    denominator: _DoubleDouble,
) -> _DoubleDouble:
    """Divide two finite short expansions to double-double accuracy."""

    divisor = denominator.value
    if divisor == 0.0 or not math.isfinite(divisor):
        return _DoubleDouble(numerator.value / divisor, 0.0)
    first = numerator.high / denominator.high
    if not math.isfinite(first):
        return _DoubleDouble(first, 0.0)
    product = _dd_multiply_float(denominator, first)
    residual = _dd_add(numerator, _dd_negate(product))
    second = residual.value / divisor
    return _dd_normalize(first, second)


def _dd_sum(values: Iterable[_DoubleDouble]) -> _DoubleDouble:
    """Accumulate short expansions without discarding cancelling low parts."""

    parts: list[float] = []
    for value in values:
        parts.extend((value.high, value.low))
    if not parts:
        return _DoubleDouble(0.0, 0.0)
    if any(not math.isfinite(part) for part in parts):
        return _DoubleDouble(_stable_sum_scalar(*parts), 0.0)
    exact_total: Fraction | None = None
    try:
        total = math.fsum(parts)
    except OverflowError:
        exact_total = sum((_fraction(part) for part in parts), Fraction(0))
        try:
            total = float(exact_total)
        except OverflowError:
            return _DoubleDouble(
                -math.inf if exact_total < 0 else math.inf,
                0.0,
            )
    if not math.isfinite(total):
        return _DoubleDouble(total, 0.0)
    # ``fsum`` gives the correctly rounded leading limb.  Summing the source
    # expansion again with that limb negated recovers the rounding residual;
    # attaching zero would round the complete expression prematurely.
    try:
        residual = math.fsum((*parts, -total))
    except OverflowError:
        if exact_total is None:
            exact_total = sum(
                (_fraction(part) for part in parts),
                Fraction(0),
            )
        residual = float(exact_total - _fraction(total))
    return _dd_normalize(total, residual)


def _down_add(left: float, right: float) -> float:
    """Round one finite-source addition toward negative infinity."""

    left_value = float(left)
    right_value = float(right)
    value = left_value + right_value
    if math.isnan(value):
        return -math.inf
    if value == math.inf:
        return _FLOAT_MAX
    if value == -math.inf:
        return -math.inf
    exact = _two_sum_scalar(left_value, right_value)
    return (
        math.nextafter(value, -math.inf)
        if exact.low < 0.0
        else value
    )


def _up_add(left: float, right: float) -> float:
    """Round one finite-source addition toward positive infinity."""

    left_value = float(left)
    right_value = float(right)
    value = left_value + right_value
    if math.isnan(value):
        return math.inf
    if value == math.inf:
        return value
    if value == -math.inf:
        return -_FLOAT_MAX
    exact = _two_sum_scalar(left_value, right_value)
    return (
        math.nextafter(value, math.inf)
        if exact.low > 0.0
        else value
    )


def _up_multiply_nonnegative(left: float, right: float) -> float:
    """Upper bound a product of two nonnegative finite binary64 values."""

    if left < 0.0 or right < 0.0:
        raise ValueError('nonnegative directed product received a negative')
    if left == 0.0 or right == 0.0:
        return 0.0
    value = left * right
    if not math.isfinite(value):
        return math.inf
    return math.nextafter(value, math.inf)


def _up_divide_nonnegative(numerator: float, denominator: float) -> float:
    """Upper bound a nonnegative quotient with a positive denominator."""

    if numerator < 0.0 or not denominator > 0.0:
        raise ValueError('directed quotient requires numerator >= 0, divisor > 0')
    if numerator == 0.0:
        return 0.0
    value = numerator / denominator
    if not math.isfinite(value):
        return math.inf
    return math.nextafter(value, math.inf)


def _directed_ldexp(value: float, exponent: int, *, upward: bool) -> float:
    """Scale by an exact power of two and round in one chosen direction."""

    scalar = float(value)
    if scalar == 0.0 or not math.isfinite(scalar):
        return scalar
    try:
        result = math.ldexp(scalar, int(exponent))
    except OverflowError:
        if scalar > 0.0:
            return math.inf if upward else _FLOAT_MAX
        return -_FLOAT_MAX if upward else -math.inf
    if result == 0.0 or abs(result) < np.finfo(np.float64).tiny:
        direction = math.inf if upward else -math.inf
        return math.nextafter(result, direction)
    return result


def _grow_exact_expansion(
    expansion: list[float],
    term: float,
) -> list[float] | None:
    """Add one limb to an exact floating expansion using only ``two_sum``."""

    if not math.isfinite(term):
        return None
    accumulator = float(term)
    grown: list[float] = []
    for component in expansion:
        pair = _two_sum_scalar(accumulator, component)
        if not math.isfinite(pair.high) or not math.isfinite(pair.low):
            return None
        if pair.low != 0.0:
            grown.append(pair.low)
        accumulator = pair.high
    if accumulator != 0.0 or not grown:
        grown.append(accumulator)
    return grown


def _exact_expansion(terms: Iterable[float]) -> list[float] | None:
    """Return an exact expansion of a finite sequence, or ``None``."""

    expansion: list[float] = []
    for term in terms:
        grown = _grow_exact_expansion(expansion, float(term))
        if grown is None:
            return None
        expansion = grown
    return expansion or [0.0]


def _ball_from_expansion(
    terms: Iterable[float],
    *,
    radius: float = 0.0,
) -> _TwofoldBall:
    """Compress an exact expansion to two limbs and bound every discard."""

    if not math.isfinite(radius) or radius < 0.0:
        return _TwofoldBall.unresolved()
    expansion = _exact_expansion(terms)
    if expansion is None:
        return _TwofoldBall.unresolved()
    ordered = sorted(expansion, key=abs, reverse=True)
    first = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    center = _two_sum_scalar(first, second)
    if not math.isfinite(center.high) or not math.isfinite(center.low):
        return _TwofoldBall.unresolved()
    outward = float(radius)
    for limb in ordered[2:]:
        outward = _up_add(outward, abs(limb))
        if not math.isfinite(outward):
            return _TwofoldBall.unresolved()
    return _TwofoldBall(center.high, center.low, outward, True)


def _ball_from_fraction(value: Fraction) -> _TwofoldBall:
    """Enclose one exact rational with a two-limb dyadic center."""

    try:
        high = float(value)
    except OverflowError:
        return _TwofoldBall.unresolved()
    if not math.isfinite(high):
        return _TwofoldBall.unresolved()
    remainder = value - Fraction.from_float(high)
    try:
        low = float(remainder)
    except OverflowError:
        return _TwofoldBall.unresolved()
    if not math.isfinite(low):
        return _TwofoldBall.unresolved()
    remaining = remainder - Fraction.from_float(low)
    radius = float(abs(remaining))
    if Fraction.from_float(radius) < abs(remaining):
        radius = math.nextafter(radius, math.inf)
    return _ball_from_expansion((high, low), radius=radius)


def _ball_negate(value: _TwofoldBall) -> _TwofoldBall:
    if not value.resolved:
        return _TwofoldBall.unresolved()
    return _TwofoldBall(-value.high, -value.low, value.radius, True)


def _ball_from_leading_and_corrections(
    leading: float,
    corrections: Iterable[float],
    *,
    radius: float,
) -> _TwofoldBall:
    """Keep two leading limbs and attach every exact discarded residual."""

    if not math.isfinite(leading) or not math.isfinite(radius):
        return _TwofoldBall.unresolved()
    parts = tuple(float(term) for term in corrections)
    if any(not math.isfinite(term) for term in parts):
        return _TwofoldBall.unresolved()
    correction = sum(parts, 0.0)
    if not math.isfinite(correction):
        return _TwofoldBall.unresolved()
    count = len(parts)
    # A sequential sum of ``count`` finite values has absolute error at most
    # gamma_n times its 1-norm, with an added minimum-subnormal allowance for
    # every rounded operation.  Using epsilon rather than unit roundoff makes
    # this bound deliberately one factor of two conservative.
    product = count * _FLOAT_EPSILON
    gamma = math.nextafter(product / (1.0 - product), math.inf)
    l1_upper = 0.0
    for term in parts:
        l1_upper = _up_add(l1_upper, abs(term))
    if not math.isfinite(l1_upper):
        return _TwofoldBall.unresolved()
    summation_radius = _up_multiply_nonnegative(gamma, l1_upper)
    summation_radius = _up_add(
        summation_radius,
        sum(term != 0.0 for term in parts) * math.ulp(0.0),
    )
    outward = _up_add(radius, summation_radius)
    center = _two_sum_scalar(leading, correction)
    if not math.isfinite(center.high) or not math.isfinite(center.low):
        return _TwofoldBall.unresolved()
    return _TwofoldBall(center.high, center.low, outward, True)


def _ball_add(left: _TwofoldBall, right: _TwofoldBall) -> _TwofoldBall:
    """Add balls, retaining both input radii and all compression limbs."""

    if not left.resolved or not right.resolved:
        return _TwofoldBall.unresolved()
    leading = _two_sum_scalar(left.high, right.high)
    return _ball_from_leading_and_corrections(
        leading.high,
        (leading.low, left.low, right.low),
        radius=_up_add(left.radius, right.radius),
    )


def _ball_subtract(left: _TwofoldBall, right: _TwofoldBall) -> _TwofoldBall:
    return _ball_add(left, _ball_negate(right))


def _two_product_terms(left: float, right: float) -> tuple[float, ...] | None:
    """Return an exact normal product expansion or mark it unsupported."""

    if left == 0.0 or right == 0.0:
        return (0.0,)
    product = _two_product_scalar(left, right)
    if not math.isfinite(product.high) or not math.isfinite(product.low):
        return None
    if abs(product.high) < _FLOAT_TINY:
        # The EFT correction may itself lie below the binary64 lattice.  An
        # exact fallback is preferable to guessing a subnormal remainder.
        return None
    return (product.high, product.low)


def _two_product_subnormal_allowance(
    left: float,
    right: float,
    low: float,
) -> float:
    """Bound only an actually material correction scaled through underflow."""

    if low != 0.0:
        return math.ulp(0.0) if abs(low) < _FLOAT_TINY else 0.0
    left_mantissa, _left_exponent = math.frexp(left)
    right_mantissa, _right_exponent = math.frexp(right)
    mantissa_product = left_mantissa * right_mantissa
    left_scaled = _SPLITTER * left_mantissa
    left_high = left_scaled - (left_scaled - left_mantissa)
    left_low = left_mantissa - left_high
    right_scaled = _SPLITTER * right_mantissa
    right_high = right_scaled - (right_scaled - right_mantissa)
    right_low = right_mantissa - right_high
    mantissa_error = (
        (left_high * right_high - mantissa_product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return math.ulp(0.0) if mantissa_error != 0.0 else 0.0


def _ball_abs_center_upper(value: _TwofoldBall) -> float:
    if not value.resolved:
        return math.inf
    return _up_add(abs(value.high), abs(value.low))


def _ball_multiply(left: _TwofoldBall, right: _TwofoldBall) -> _TwofoldBall:
    """Multiply balls with the complete bilinear radius formula."""

    if not left.resolved or not right.resolved:
        return _TwofoldBall.unresolved()
    main_terms = _two_product_terms(left.high, right.high)
    if main_terms is None:
        return _TwofoldBall.unresolved()
    leading = main_terms[0]
    corrections = list(main_terms[1:])
    product_rounding_radius = (
        _two_product_subnormal_allowance(
            left.high,
            right.high,
            main_terms[1],
        )
        if len(main_terms) > 1
        else 0.0
    )
    for left_limb, right_limb in (
        (left.high, right.low),
        (left.low, right.high),
        (left.low, right.low),
    ):
        if left_limb == 0.0 or right_limb == 0.0:
            corrections.append(0.0)
            continue
        product = left_limb * right_limb
        if not math.isfinite(product):
            return _TwofoldBall.unresolved()
        corrections.append(product)
        rounding = math.ulp(product) if product != 0.0 else math.ulp(0.0)
        product_rounding_radius = _up_add(
            product_rounding_radius,
            rounding,
        )
    left_magnitude = _ball_abs_center_upper(left)
    right_magnitude = _ball_abs_center_upper(right)
    radius = _up_add(
        _up_multiply_nonnegative(left_magnitude, right.radius),
        _up_multiply_nonnegative(right_magnitude, left.radius),
    )
    radius = _up_add(
        radius,
        _up_multiply_nonnegative(left.radius, right.radius),
    )
    radius = _up_add(radius, product_rounding_radius)
    return _ball_from_leading_and_corrections(
        leading,
        corrections,
        radius=radius,
    )


def _ball_square(value: _TwofoldBall) -> _TwofoldBall:
    return _ball_multiply(value, value)


def _ball_integer_scale(value: _TwofoldBall, factor: int) -> _TwofoldBall:
    integer = int(factor)
    if abs(integer) > 2**53:
        return _TwofoldBall.unresolved()
    return _ball_multiply(value, _TwofoldBall.point(float(integer)))


def _ball_ldexp(value: _TwofoldBall, exponent: int) -> _TwofoldBall:
    """Scale a ball by ``2**exponent`` with explicit range handling."""

    if not value.resolved:
        return _TwofoldBall.unresolved()
    power = int(exponent)
    if power == 0:
        return value
    try:
        high = math.ldexp(value.high, power)
        low = math.ldexp(value.low, power)
        radius = math.ldexp(value.radius, power)
    except OverflowError:
        return _TwofoldBall.unresolved()
    if not all(math.isfinite(part) for part in (high, low, radius)):
        return _TwofoldBall.unresolved()
    tiny = np.finfo(np.float64).tiny
    exact_scaling = all(
        source == 0.0 or abs(scaled) >= tiny
        for source, scaled in (
            (value.high, high),
            (value.low, low),
            (value.radius, radius),
        )
    )
    if exact_scaling:
        return _TwofoldBall(high, low, radius, True)

    # Gradual underflow can discard a limb or radius.  Reconstruct a
    # conservative ball from directed physical endpoints in this exceptional
    # regime; ordinary normalized work never pays this first-order widening.
    lower, upper = value.physical_bounds()
    if not math.isfinite(lower) or not math.isfinite(upper):
        return _TwofoldBall.unresolved()
    scaled_lower = _directed_ldexp(lower, power, upward=False)
    scaled_upper = _directed_ldexp(upper, power, upward=True)
    if not math.isfinite(scaled_lower) or not math.isfinite(scaled_upper):
        return _TwofoldBall.unresolved()
    center = _ball_from_expansion((high, low))
    if not center.resolved:
        return center
    left_gap = _up_add(_up_add(center.high, center.low), -scaled_lower)
    right_gap = _up_add(_up_add(scaled_upper, -center.high), -center.low)
    radius = max(0.0, left_gap, right_gap)
    return _TwofoldBall(center.high, center.low, radius, True)


def _ball_divide(
    numerator: _TwofoldBall,
    denominator: _TwofoldBall,
) -> _TwofoldBall:
    """Divide through a residual proof after excluding zero."""

    if not numerator.resolved or not denominator.resolved:
        return _TwofoldBall.unresolved()
    denominator_lower, denominator_upper = denominator.physical_bounds()
    if denominator_lower <= 0.0 <= denominator_upper:
        return _TwofoldBall.unresolved()
    minimum_denominator = min(
        abs(denominator_lower),
        abs(denominator_upper),
    )
    if not minimum_denominator > 0.0:
        return _TwofoldBall.unresolved()
    if denominator.high == 0.0:
        return _TwofoldBall.unresolved()
    first = numerator.high / denominator.high
    if not math.isfinite(first):
        return _TwofoldBall.unresolved()
    first_ball = _TwofoldBall.point(first)
    first_residual = _ball_subtract(
        numerator,
        _ball_multiply(denominator, first_ball),
    )
    if not first_residual.resolved:
        return _TwofoldBall.unresolved()
    denominator_center = denominator.center_value
    correction = first_residual.center_value / denominator_center
    if not math.isfinite(correction):
        return _TwofoldBall.unresolved()
    proposal = _ball_from_expansion((first, correction))
    if not proposal.resolved:
        return proposal
    point_proposal = _TwofoldBall(
        proposal.high,
        proposal.low,
        0.0,
        True,
    )
    residual = _ball_subtract(
        numerator,
        _ball_multiply(denominator, point_proposal),
    )
    if not residual.resolved:
        return _TwofoldBall.unresolved()
    residual_lower, residual_upper = residual.physical_bounds()
    residual_magnitude = max(abs(residual_lower), abs(residual_upper))
    radius = _up_divide_nonnegative(
        residual_magnitude,
        minimum_denominator,
    )
    return _TwofoldBall(proposal.high, proposal.low, radius, True)


def _ball_reciprocal(value: _TwofoldBall) -> _TwofoldBall:
    return _ball_divide(_TwofoldBall.point(1.0), value)


def _binary_scaled_from_ball(value: _TwofoldBall) -> _BinaryScaledBall:
    """Normalize a finite ball without changing its represented set."""

    if not value.resolved:
        return _BinaryScaledBall.unresolved()
    lower, upper = value.physical_bounds()
    magnitude = max(abs(lower), abs(upper))
    if magnitude == 0.0:
        return _BinaryScaledBall(value, 0)
    if not math.isfinite(magnitude):
        return _BinaryScaledBall.unresolved()
    _mantissa, exponent = math.frexp(magnitude)
    scaled = _ball_ldexp(value, -exponent)
    if not scaled.resolved:
        return _BinaryScaledBall.unresolved()
    return _BinaryScaledBall(scaled, exponent)


def _binary_scaled_multiply(
    left: _BinaryScaledBall,
    right: _BinaryScaledBall,
) -> _BinaryScaledBall:
    product = _ball_multiply(left.ball, right.ball)
    normalized = _binary_scaled_from_ball(product)
    if not normalized.ball.resolved:
        return normalized
    return _BinaryScaledBall(
        normalized.ball,
        left.exponent + right.exponent + normalized.exponent,
    )


def _binary_scaled_negate(value: _BinaryScaledBall) -> _BinaryScaledBall:
    return _BinaryScaledBall(_ball_negate(value.ball), value.exponent)


def _binary_scaled_divide(
    numerator: _BinaryScaledBall,
    denominator: _BinaryScaledBall,
) -> _BinaryScaledBall:
    quotient = _ball_divide(numerator.ball, denominator.ball)
    normalized = _binary_scaled_from_ball(quotient)
    if not normalized.ball.resolved:
        return normalized
    return _BinaryScaledBall(
        normalized.ball,
        numerator.exponent - denominator.exponent + normalized.exponent,
    )


def _binary_scaled_sum(
    values: Iterable[_BinaryScaledBall],
) -> _BinaryScaledBall:
    """Accumulate signed scaled balls without logarithms or ``inf-inf``."""

    material = tuple(values)
    if not material:
        return _BinaryScaledBall(_TwofoldBall.point(0.0), 0)
    if any(not value.ball.resolved for value in material):
        return _BinaryScaledBall.unresolved()
    exponent = max(value.exponent for value in material)
    total = _TwofoldBall.point(0.0)
    for value in material:
        aligned = _ball_ldexp(value.ball, value.exponent - exponent)
        total = _ball_add(total, aligned)
        if not total.resolved:
            return _BinaryScaledBall.unresolved()
    normalized = _binary_scaled_from_ball(total)
    if not normalized.ball.resolved:
        return normalized
    return _BinaryScaledBall(normalized.ball, exponent + normalized.exponent)


def _array_two_sum(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    high = left + right
    virtual_right = high - left
    low = left - (high - virtual_right) + (right - virtual_right)
    return high, low


def _array_down_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    high, low = _array_two_sum(left, right)
    return np.where(low < 0.0, np.nextafter(high, -np.inf), high)


def _array_up_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    high, low = _array_two_sum(left, right)
    return np.where(low > 0.0, np.nextafter(high, np.inf), high)


def _array_up_nonnegative_product(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    product = left * right
    return np.where(
        (left == 0.0) | (right == 0.0),
        0.0,
        np.nextafter(product, np.inf),
    )


def _array_two_product(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Dekker product on finite normal mantissa products."""

    high = left * right
    left_mantissa, left_exponent = np.frexp(left)
    right_mantissa, right_exponent = np.frexp(right)
    mantissa_product = left_mantissa * right_mantissa
    left_scaled = _SPLITTER * left_mantissa
    left_high = left_scaled - (left_scaled - left_mantissa)
    left_low = left_mantissa - left_high
    right_scaled = _SPLITTER * right_mantissa
    right_high = right_scaled - (right_scaled - right_mantissa)
    right_low = right_mantissa - right_high
    mantissa_error = (
        (left_high * right_high - mantissa_product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    exponent = np.asarray(left_exponent + right_exponent, dtype=np.intc)
    low = np.ldexp(mantissa_error, exponent)
    resolved = np.isfinite(high) & np.isfinite(low)
    material = (left != 0.0) & (right != 0.0)
    resolved &= ~material | (np.abs(high) >= _FLOAT_TINY)
    subnormal_correction = (
        material
        & (mantissa_error != 0.0)
        & (np.abs(low) < _FLOAT_TINY)
    )
    return high, low, resolved, subnormal_correction


def _array_ball_negate(value: _ArrayBall) -> _ArrayBall:
    return _ArrayBall(-value.high, -value.low, value.radius, value.resolved)


def _array_ball_add(left: _ArrayBall, right: _ArrayBall) -> _ArrayBall:
    leading, leading_low = _array_two_sum(left.high, right.high)
    parts = np.stack((leading_low, left.low, right.low), axis=0)
    correction = np.sum(parts, axis=0)
    count = parts.shape[0]
    gamma = math.nextafter(
        count * _FLOAT_EPSILON / (1.0 - count * _FLOAT_EPSILON),
        math.inf,
    )
    l1_upper = np.zeros_like(leading)
    for part in parts:
        l1_upper = _array_up_add(l1_upper, np.abs(part))
    summation_radius = np.nextafter(gamma * l1_upper, np.inf)
    summation_radius = _array_up_add(
        summation_radius,
        np.count_nonzero(parts, axis=0) * math.ulp(0.0),
    )
    center_high, center_low = _array_two_sum(leading, correction)
    radius = _array_up_add(
        _array_up_add(left.radius, right.radius),
        summation_radius,
    )
    resolved = (
        left.resolved
        & right.resolved
        & np.isfinite(center_high)
        & np.isfinite(center_low)
        & np.isfinite(radius)
    )
    return _ArrayBall(center_high, center_low, radius, resolved)


def _array_ball_subtract(left: _ArrayBall, right: _ArrayBall) -> _ArrayBall:
    return _array_ball_add(left, _array_ball_negate(right))


def _array_ball_multiply(left: _ArrayBall, right: _ArrayBall) -> _ArrayBall:
    leading, main_low, product_resolved, subnormal_correction = (
        _array_two_product(
            left.high,
            right.high,
        )
    )
    cross = np.stack(
        (
            main_low,
            left.high * right.low,
            left.low * right.high,
            left.low * right.low,
        ),
        axis=0,
    )
    correction = np.sum(cross, axis=0)
    count = cross.shape[0]
    gamma = math.nextafter(
        count * _FLOAT_EPSILON / (1.0 - count * _FLOAT_EPSILON),
        math.inf,
    )
    l1_upper = np.zeros_like(leading)
    for part in cross:
        l1_upper = _array_up_add(l1_upper, np.abs(part))
    summation_radius = np.nextafter(gamma * l1_upper, np.inf)
    summation_radius = _array_up_add(
        summation_radius,
        np.count_nonzero(cross, axis=0) * math.ulp(0.0),
    )
    cross_rounding = np.where(
        product_resolved & subnormal_correction,
        math.ulp(0.0),
        0.0,
    )
    for product, left_limb, right_limb in (
        (cross[1], left.high, right.low),
        (cross[2], left.low, right.high),
        (cross[3], left.low, right.low),
    ):
        material = (left_limb != 0.0) & (right_limb != 0.0)
        ulp = np.abs(np.spacing(product))
        ulp = np.where(material & (product == 0.0), math.ulp(0.0), ulp)
        cross_rounding = _array_up_add(cross_rounding, ulp)
    center_high, center_low = _array_two_sum(leading, correction)
    left_magnitude = _array_up_add(np.abs(left.high), np.abs(left.low))
    right_magnitude = _array_up_add(np.abs(right.high), np.abs(right.low))
    radius = _array_up_add(
        _array_up_nonnegative_product(left_magnitude, right.radius),
        _array_up_nonnegative_product(right_magnitude, left.radius),
    )
    radius = _array_up_add(
        radius,
        _array_up_nonnegative_product(left.radius, right.radius),
    )
    radius = _array_up_add(radius, summation_radius)
    radius = _array_up_add(radius, cross_rounding)
    resolved = (
        left.resolved
        & right.resolved
        & product_resolved
        & np.isfinite(center_high)
        & np.isfinite(center_low)
        & np.isfinite(radius)
    )
    return _ArrayBall(center_high, center_low, radius, resolved)


def _array_ball_divide(
    numerator: _ArrayBall,
    denominator: _ArrayBall,
) -> _ArrayBall:
    denominator_lower, denominator_upper = denominator.physical_bounds()
    excludes_zero = (denominator_lower > 0.0) | (denominator_upper < 0.0)
    minimum_denominator = np.minimum(
        np.abs(denominator_lower),
        np.abs(denominator_upper),
    )
    first = numerator.high / denominator.high
    first_ball = _ArrayBall.points(first)
    first_residual = _array_ball_subtract(
        numerator,
        _array_ball_multiply(denominator, first_ball),
    )
    denominator_center = denominator.high + denominator.low
    correction = (
        first_residual.high + first_residual.low
    ) / denominator_center
    proposal_high, proposal_low = _array_two_sum(first, correction)
    proposal = _ArrayBall(
        proposal_high,
        proposal_low,
        np.zeros_like(proposal_high),
        np.isfinite(proposal_high) & np.isfinite(proposal_low),
    )
    residual = _array_ball_subtract(
        numerator,
        _array_ball_multiply(denominator, proposal),
    )
    residual_lower, residual_upper = residual.physical_bounds()
    residual_magnitude = np.maximum(
        np.abs(residual_lower),
        np.abs(residual_upper),
    )
    radius = np.nextafter(
        residual_magnitude / minimum_denominator,
        np.inf,
    )
    resolved = (
        numerator.resolved
        & denominator.resolved
        & excludes_zero
        & proposal.resolved
        & residual.resolved
        & np.isfinite(radius)
    )
    return _ArrayBall(proposal_high, proposal_low, radius, resolved)


def _scaled_signed_enclosure(
    terms: Sequence[tuple[int, float, float]],
) -> _ScaledEnclosure:
    """Accumulate signed log-magnitude terms with an outward binary64 bound.

    Each term is ``(sign, log(abs(value)), relative_error_bound)``.  The
    common scale prevents raw overflow, while ``fsum`` retains cancellation.
    """

    if any(
        not math.isfinite(log_abs) or not math.isfinite(relative_error)
        for _sign, log_abs, relative_error in terms
    ):
        return _ScaledEnclosure(-math.inf, math.inf, 0.0)
    material = tuple(
        term
        for term in terms
        if term[0] or term[2] > 0.0
    )
    if not material:
        return _ScaledEnclosure(0.0, 0.0, 0.0)
    scale = max(term[1] for term in material)
    scaled_values: list[float] = []
    error_terms: list[float] = []
    for sign, log_abs, relative_error in material:
        offset = log_abs - scale
        magnitude = 0.0 if offset < -746.0 else math.exp(offset)
        if sign > 0:
            scaled_values.append(magnitude)
        elif sign < 0:
            scaled_values.append(-magnitude)
        else:
            scaled_values.append(0.0)
        error_terms.append(abs(magnitude) * max(0.0, relative_error))
    center = math.fsum(scaled_values)
    l1 = math.fsum(abs(value) for value in scaled_values)
    radius = math.fsum(error_terms) + 16.0 * _FLOAT_EPSILON * l1
    radius += 4.0 * math.ulp(center) if math.isfinite(center) else math.inf
    lower = math.nextafter(center - radius, -math.inf)
    upper = math.nextafter(center + radius, math.inf)
    return _ScaledEnclosure(lower, upper, scale)


def _scaled_physical_float(value: float, log_scale: float) -> float:
    """Convert one scaled value to binary64 without unsafe overflow work."""

    scalar = float(value)
    if scalar == 0.0:
        return scalar
    log_abs = math.log(abs(scalar)) + float(log_scale)
    if log_abs > math.log(_FLOAT_MAX):
        return math.copysign(math.inf, scalar)
    if log_abs < math.log(math.ulp(0.0)) - 2.0:
        return math.copysign(0.0, scalar)
    return math.copysign(math.exp(log_abs), scalar)


def _fraction_float_neighbors(value: Fraction) -> tuple[float, float]:
    """Return the bracketing binary64 values for an exact finite rational."""

    rounded = float(value)
    if not math.isfinite(rounded):
        endpoint = math.copysign(_FLOAT_MAX, rounded)
        if value == Fraction.from_float(endpoint):
            return endpoint, endpoint
        if rounded > 0.0:
            return endpoint, float('inf')
        return float('-inf'), endpoint
    rounded_fraction = Fraction.from_float(rounded)
    if rounded_fraction == value:
        return rounded, rounded
    if rounded_fraction < value:
        return rounded, math.nextafter(rounded, float('inf'))
    return math.nextafter(rounded, float('-inf')), rounded


def _ldexp(
    mantissa: object,
    exponent: object,
) -> np.ndarray:
    """Apply ``np.ldexp`` through its portable C-``int`` exponent loop.

    NumPy 1.x defines binary floating-point ``ldexp`` loops for C ``int``
    and C ``long`` exponents.  Both are 32-bit on Windows, so an internal
    ``int64`` exponent accumulator cannot be passed under safe casting there.
    The callers keep wider accumulators and narrow only at this ufunc boundary,
    after the relevant binary-floating-point range checks.
    """

    return np.ldexp(
        mantissa,
        np.asarray(exponent, dtype=np.intc),
    )


def _stable_product_scalar(*factors: float) -> float:
    """Multiply finite scalars without avoidable intermediate range loss."""

    values = tuple(float(value) for value in factors)
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in values):
        return 0.0

    negative = sum(value < 0.0 for value in values) % 2
    if any(math.isinf(value) for value in values):
        return float('-inf') if negative else float('inf')

    mantissa = -1.0 if negative else 1.0
    exponent = 0
    for value in values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _stable_ratio_product_scalar(
    numerators: Iterable[float],
    denominators: Iterable[float],
) -> float:
    """Evaluate a product ratio without first forming reciprocal powers."""

    numerator_values = tuple(float(value) for value in numerators)
    denominator_values = tuple(float(value) for value in denominators)
    values = numerator_values + denominator_values
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in denominator_values):
        if any(value == 0.0 for value in numerator_values):
            return float('nan')
        negative = (
            sum(value < 0.0 for value in numerator_values)
            + sum(value < 0.0 for value in denominator_values)
        ) % 2
        return float('-inf') if negative else float('inf')
    if any(value == 0.0 for value in numerator_values):
        return 0.0

    negative = (
        sum(value < 0.0 for value in numerator_values)
        + sum(value < 0.0 for value in denominator_values)
    ) % 2
    mantissa = -1.0 if negative else 1.0
    exponent = 0
    for value in numerator_values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    for value in denominator_values:
        part, part_exponent = math.frexp(abs(value))
        mantissa /= part
        exponent -= part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _stable_sum_scalar(*values: float) -> float:
    """Sum scalars accurately, returning infinity only on true range overflow."""

    floats = tuple(float(value) for value in values)
    if any(math.isnan(value) for value in floats):
        return float('nan')
    try:
        return float(math.fsum(floats))
    except (OverflowError, ValueError):
        if all(math.isfinite(value) for value in floats):
            total = sum((_fraction(value) for value in floats), Fraction(0))
            try:
                return float(total)
            except OverflowError:
                return float('-inf') if total < 0 else float('inf')
        positive = any(value == float('inf') for value in floats)
        negative = any(value == float('-inf') for value in floats)
        if positive and negative:
            return float('nan')
        if positive:
            return float('inf')
        if negative:
            return float('-inf')
        return float('-inf') if all(value <= 0.0 for value in floats) else float('inf')


def _fraction(value: float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _exact_sum_products_scalar(
    products: Sequence[Sequence[float]],
) -> float:
    """Evaluate a finite sum of finite products exactly before float rounding."""

    total = Fraction(0)
    for factors in products:
        term = Fraction(1)
        for factor in factors:
            value = float(factor)
            if not math.isfinite(value):
                return _stable_sum_scalar(
                    *(
                        _stable_product_scalar(*product)
                        for product in products
                    )
                )
            term *= _fraction(value)
        total += term
    try:
        return float(total)
    except OverflowError:
        return float('-inf') if total < 0 else float('inf')


def _exact_sum_products_sign_scalar(
    products: Sequence[Sequence[float]],
) -> int:
    """Return the exact sign of a finite sum of binary64 products."""

    total = Fraction(0)
    for factors in products:
        term = Fraction(1)
        for factor in factors:
            value = float(factor)
            if not math.isfinite(value):
                approximate = _exact_sum_products_scalar(products)
                return int(approximate > 0.0) - int(approximate < 0.0)
            term *= _fraction(value)
        total += term
    return int(total > 0) - int(total < 0)


def _exact_sum_ratios_scalar(
    terms: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """Evaluate a finite sum of finite product ratios exactly."""

    total = Fraction(0)
    for numerators, denominators in terms:
        term = Fraction(1)
        for numerator in numerators:
            term *= _fraction(float(numerator))
        for denominator in denominators:
            term /= _fraction(float(denominator))
        total += term
    try:
        return float(total)
    except OverflowError:
        return float('-inf') if total < 0 else float('inf')


def _exact_weighted_average_scalar(
    left: float,
    left_weight: float,
    right: float,
    right_weight: float,
) -> float:
    """Evaluate a finite two-term weighted average exactly before rounding."""

    denominator = _fraction(left_weight) + _fraction(right_weight)
    if denominator == 0:
        return float('nan')
    value = (
        _fraction(left_weight) * _fraction(left)
        + _fraction(right_weight) * _fraction(right)
    ) / denominator
    try:
        return float(value)
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


def _same_sign(left: float, right: float) -> bool:
    return (left >= 0.0 and right >= 0.0) or (
        left <= 0.0 and right <= 0.0
    )


def _stable_scaled_difference_scalar(
    left: float,
    right: float,
    *scale_factors: float,
) -> float:
    """Evaluate ``prod(scale_factors) * (left - right)`` stably."""

    scales = tuple(float(value) for value in scale_factors)
    if any(value == 0.0 for value in scales):
        return 0.0
    left_value = float(left)
    right_value = float(right)
    if (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and all(math.isfinite(value) for value in scales)
    ):
        return _exact_sum_products_scalar(
            (
                scales + (left_value,),
                (-1.0,) + scales + (right_value,),
            )
        )
    if _same_sign(left_value, right_value):
        return _stable_product_scalar(
            *scales,
            left_value - right_value,
        )
    return _stable_sum_scalar(
        _stable_product_scalar(*scales, left_value),
        -_stable_product_scalar(*scales, right_value),
    )


def _stable_ratio_difference_scalar(
    left: float,
    right: float,
    denominator: float,
) -> float:
    """Evaluate ``(left - right) / denominator`` stably."""

    left_value = float(left)
    right_value = float(right)
    if _same_sign(left_value, right_value):
        return _stable_ratio_product_scalar(
            (left_value - right_value,),
            (denominator,),
        )
    return _stable_sum_scalar(
        _stable_ratio_product_scalar((left_value,), (denominator,)),
        -_stable_ratio_product_scalar((right_value,), (denominator,)),
    )


def _can_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    opposite = np.signbit(left) != np.signbit(right)
    return finite & (
        opposite
        | (np.abs(left) <= _FLOAT_MAX - np.abs(right))
    )


def _can_subtract(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    same = np.signbit(left) == np.signbit(right)
    return finite & (
        same
        | (np.abs(left) <= _FLOAT_MAX - np.abs(right))
    )


def _can_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    finite = np.isfinite(left) & np.isfinite(right)
    left_absolute = np.abs(left)
    right_absolute = np.abs(right)
    large_left = left_absolute > 1.0
    threshold = np.full(left.shape, _FLOAT_MAX, dtype=np.float64)
    threshold[large_left] = _FLOAT_MAX / left_absolute[large_left]
    range_safe = finite & (
        (left_absolute == 0.0)
        | (right_absolute == 0.0)
        | ~large_left
        | (right_absolute <= threshold)
    )
    nonzero = (
        finite
        & (left_absolute != 0.0)
        & (right_absolute != 0.0)
    )
    underflow_risk = np.zeros(left.shape, dtype=bool)
    if np.any(nonzero):
        left_part, left_exponent = np.frexp(left_absolute[nonzero])
        right_part, right_exponent = np.frexp(right_absolute[nonzero])
        _, normalization = np.frexp(left_part * right_part)
        product_exponent = (
            left_exponent
            + right_exponent
            + normalization
        )
        underflow_risk[nonzero] = (
            product_exponent < _NORMAL_MIN_EXPONENT
        )
    return range_safe & ~underflow_risk


def _direct_product(
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.ones(arrays[0].shape, dtype=np.float64)
    safe = np.ones(result.shape, dtype=bool)
    for array in arrays:
        step_safe = _can_multiply(result, array)
        active = safe & step_safe
        result[active] *= array[active]
        safe &= step_safe
    return result, safe


def _normal_product_parts(
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.zeros(shape, dtype=np.int64)
    finite_nonzero = np.ones(shape, dtype=bool)
    for array in arrays:
        finite_nonzero &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    normal = (
        finite_nonzero
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    return mantissa, exponent, normal


def _stable_product(
    *factors: object,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vectorized product with scalar fallbacks only for exceptional rows."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in factors)
    )
    if not arrays:
        result = np.asarray(1.0, dtype=np.float64)
        if return_exceptional:
            return result, np.asarray(False)
        return result
    result, direct = _direct_product(arrays)
    if np.all(direct):
        if return_exceptional:
            return result, ~direct
        return result
    zero = np.logical_or.reduce(
        tuple(array == 0.0 for array in arrays)
    )
    result[zero] = 0.0
    exceptional = ~direct & ~zero
    exceptional_arrays = tuple(array[exceptional] for array in arrays)
    mantissa, exponent, normal = _normal_product_parts(exceptional_arrays)
    exceptional_result = np.empty(mantissa.shape, dtype=np.float64)
    if np.any(normal):
        exceptional_result[normal] = _ldexp(
            mantissa[normal],
            exponent[normal],
        )
    for flat_index in np.flatnonzero(~normal):
        exceptional_result.flat[flat_index] = _stable_product_scalar(
            *(
                float(array.flat[flat_index])
                for array in exceptional_arrays
            )
        )
    result[exceptional] = exceptional_result
    for flat_index in np.flatnonzero(~direct & zero & np.logical_or.reduce(
        tuple(np.isnan(array) for array in arrays)
    )):
        result.flat[flat_index] = _stable_product_scalar(
            *(float(array.flat[flat_index]) for array in arrays)
        )
    if return_exceptional:
        return result, ~direct
    return result


def _stable_ratio_product(
    numerators: Sequence[object],
    denominators: Sequence[object],
) -> np.ndarray:
    """Vectorized product ratio with exceptional scalar fallbacks."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (*numerators, *denominators)
        )
    )
    numerator_arrays = arrays[:len(numerators)]
    denominator_arrays = arrays[len(numerators):]
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.zeros(shape, dtype=np.int64)
    regular = np.ones(shape, dtype=bool)
    numerator_zero = np.zeros(shape, dtype=bool)
    for array in numerator_arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        numerator_zero |= array == 0.0
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    for array in denominator_arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(array)
        mantissa /= np.where(array == 0.0, 1.0, part)
        exponent -= part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    normal = (
        regular
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    result = np.zeros(shape, dtype=np.float64)
    if np.any(normal):
        result[normal] = _ldexp(mantissa[normal], exponent[normal])
    exceptional = ~normal & ~(
        numerator_zero
        & np.logical_and.reduce(
            tuple(array != 0.0 for array in denominator_arrays)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _stable_ratio_product_scalar(
            tuple(float(array.flat[flat_index]) for array in numerator_arrays),
            tuple(float(array.flat[flat_index]) for array in denominator_arrays),
        )
    return result


def _stable_normalized_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    active: np.ndarray,
) -> np.ndarray:
    """Return nonnegative normalized ratios without unsafe subnormal division."""

    numerator_array, denominator_array, active_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
        np.asarray(active, dtype=bool),
    )
    result = np.zeros(numerator_array.shape, dtype=np.float64)
    material = (
        active_array
        & (numerator_array != 0.0)
        & (denominator_array != 0.0)
    )
    if not np.any(material):
        return result

    numerator_part, numerator_exponent = np.frexp(
        numerator_array[material]
    )
    denominator_part, denominator_exponent = np.frexp(
        denominator_array[material]
    )
    ratio_part, normalization = np.frexp(
        numerator_part / denominator_part
    )
    ratio_exponent = (
        numerator_exponent
        - denominator_exponent
        + normalization
    )
    normal = (
        (ratio_exponent >= _NORMAL_MIN_EXPONENT)
        & (ratio_exponent <= _NORMAL_MAX_EXPONENT)
    )
    material_result = np.empty(ratio_part.shape, dtype=np.float64)
    if np.any(normal):
        material_result[normal] = _ldexp(
            ratio_part[normal],
            ratio_exponent[normal],
        )
    for flat_index in np.flatnonzero(~normal):
        material_result.flat[flat_index] = _stable_ratio_product_scalar(
            (float(numerator_array[material].flat[flat_index]),),
            (float(denominator_array[material].flat[flat_index]),),
        )
    result[material] = material_result
    return result


def _cancellation_error_bound(
    result: np.ndarray,
    terms: Sequence[np.ndarray],
    *,
    operation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixed-sign rows and their first-order forward-error bound."""

    finite_terms = np.logical_and.reduce(
        tuple(np.isfinite(term) for term in terms)
    )
    positive = np.logical_or.reduce(tuple(term > 0.0 for term in terms))
    negative = np.logical_or.reduce(tuple(term < 0.0 for term in terms))
    mixed = finite_terms & positive & negative
    if not np.any(mixed):
        return mixed, np.zeros(result.shape, dtype=np.float64)

    absolute_terms = tuple(np.abs(term) for term in terms)
    scale = np.maximum.reduce(absolute_terms)
    scaled_l1 = np.zeros(result.shape, dtype=np.float64)
    for absolute in absolute_terms:
        material = (
            mixed
            & (absolute != 0.0)
            & (scale != 0.0)
        )
        contribution = _stable_normalized_ratio(
            absolute,
            scale,
            active=material,
        )
        scaled_l1 += contribution
    error_bound = _stable_product(
        scale,
        scaled_l1,
        max(1, int(operation_count)) * _FLOAT_EPSILON,
    )
    return mixed, error_bound


def _cancellation_risk(
    result: np.ndarray,
    terms: Sequence[np.ndarray],
    *,
    operation_count: int,
    relative_limit: float = _CONDITIONING_RELATIVE_LIMIT,
) -> np.ndarray:
    """Identify mixed-sign rows with materially uncertain relative accuracy."""

    mixed, error_bound = _cancellation_error_bound(
        result,
        terms,
        operation_count=operation_count,
    )
    relative_bound = _stable_product(
        np.abs(result),
        float(relative_limit),
    )
    return mixed & (
        ~np.isfinite(result)
        | (error_bound >= relative_bound)
    )


def _two_sum(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rounded sum and its exact finite-input rounding error."""

    summed = left + right
    right_virtual = summed - left
    error = (
        (left - (summed - right_virtual))
        + (right - right_virtual)
    )
    return summed, error


def _split_product_operand(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ordinary finite values into nonoverlapping high/low parts."""

    mantissa, exponent = np.frexp(values)
    scaled = _SPLITTER * mantissa
    high_mantissa = scaled - (scaled - mantissa)
    high = _ldexp(high_mantissa, exponent)
    return high, values - high


def _two_product_error(
    left: np.ndarray,
    right: np.ndarray,
    product: np.ndarray,
) -> np.ndarray:
    """Return the exact ordinary-product rounding error via Dekker splitting."""

    left_mantissa, left_exponent = np.frexp(left)
    right_mantissa, right_exponent = np.frexp(right)
    left_scaled = _SPLITTER * left_mantissa
    left_high = left_scaled - (left_scaled - left_mantissa)
    left_low = left_mantissa - left_high
    right_scaled = _SPLITTER * right_mantissa
    right_high = right_scaled - (right_scaled - right_mantissa)
    right_low = right_mantissa - right_high
    mantissa_product = left_mantissa * right_mantissa
    mantissa_error = (
        (
            (left_high * right_high - mantissa_product)
            + left_high * right_low
        )
        + left_low * right_high
        + left_low * right_low
    )
    return _ldexp(mantissa_error, left_exponent + right_exponent)


def _compensated_affine_residual(
    beta: np.ndarray,
    alpha: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    left_product: np.ndarray,
    right_product: np.ndarray,
) -> np.ndarray:
    """Evaluate ordinary affine rows with product and sum compensation."""

    left_error = _two_product_error(alpha, left, left_product)
    right_error = _two_product_error(-alpha, right, right_product)
    total = np.zeros(beta.shape, dtype=np.float64)
    correction = np.zeros(beta.shape, dtype=np.float64)
    for term in (
        beta,
        left_product,
        right_product,
        -target,
        left_error,
        right_error,
    ):
        total, error = _two_sum(total, term)
        correction += error
    total, error = _two_sum(total, correction)
    return total + error


def _has_compensation_exponent_margin(
    values: np.ndarray,
    *,
    lower_margin: int,
    upper_margin: int,
) -> np.ndarray:
    """Return rows safe for raw compensated arithmetic at every exponent."""

    array = np.asarray(values, dtype=np.float64)
    _, exponent = np.frexp(np.abs(array))
    return (
        (array == 0.0)
        | (
            np.isfinite(array)
            & (exponent >= _NORMAL_MIN_EXPONENT + int(lower_margin))
            & (exponent <= _NORMAL_MAX_EXPONENT - int(upper_margin))
        )
    )


def _stable_sum(*values: object) -> np.ndarray:
    """Vectorized sum with exact range and cancellation fallbacks."""

    arrays = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in values)
    )
    result = np.zeros(arrays[0].shape, dtype=np.float64)
    safe = np.ones(result.shape, dtype=bool)
    for array in arrays:
        add_safe = _can_add(result, array)
        active = safe & add_safe
        result[active] += array[active]
        safe &= add_safe
    exceptional = ~safe
    exceptional |= _cancellation_risk(
        result,
        arrays,
        operation_count=len(arrays),
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple((float(array.flat[flat_index]),) for array in arrays)
        )
    return result


def _stable_sum_products(
    products: Sequence[Sequence[object]],
    *,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate a sum of products with complete-expression fallbacks."""

    flat_values = tuple(
        value
        for product in products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    product_outputs = tuple(
        _stable_product(*product, return_exceptional=True)
        for product in broadcast_products
    )
    product_values = tuple(output[0] for output in product_outputs)
    product_exceptional = np.logical_or.reduce(
        tuple(output[1] for output in product_outputs)
    )
    result = _stable_sum(*product_values)
    vanished_product = np.zeros(result.shape, dtype=bool)
    for product, product_value in zip(
        broadcast_products,
        product_values,
    ):
        finite_nonzero = np.logical_and.reduce(
            tuple(np.isfinite(array) & (array != 0.0) for array in product)
        )
        vanished_product |= finite_nonzero & (product_value == 0.0)
    exceptional = product_exceptional | _cancellation_risk(
        result,
        product_values,
        operation_count=(
            len(product_values)
            + sum(max(0, len(product) - 1) for product in products)
        ),
    )
    exceptional |= vanished_product
    exceptional |= (
        ~np.isfinite(result)
        & np.logical_and.reduce(
            tuple(np.isfinite(array) for array in broadcast)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple(
                tuple(
                    float(array.flat[flat_index])
                    for array in product
                )
                for product in broadcast_products
            )
        )
    if return_exceptional:
        return result, exceptional
    return result


def _stable_sum_products_sign(
    products: Sequence[Sequence[object]],
) -> np.ndarray:
    """Return exact signs for exceptional sums of binary64 products."""

    result, exceptional = _stable_sum_products(
        products,
        return_exceptional=True,
    )
    sign = np.zeros(result.shape, dtype=np.int8)
    sign[result > 0.0] = 1
    sign[result < 0.0] = -1
    if not np.any(exceptional):
        return sign

    flat_values = tuple(
        value
        for product in products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    for flat_index in np.flatnonzero(exceptional):
        sign.flat[flat_index] = _exact_sum_products_sign_scalar(
            tuple(
                tuple(
                    float(array.flat[flat_index])
                    for array in product
                )
                for product in broadcast_products
            )
        )
    return sign


def _power_scaled_product_scalar(
    power: int,
    *factors: float,
) -> float:
    """Return ``2**power * prod(factors)`` without storing the power."""

    values = tuple(float(value) for value in factors)
    if any(math.isnan(value) for value in values):
        return float('nan')
    if any(value == 0.0 for value in values):
        return 0.0
    negative = sum(value < 0.0 for value in values) % 2
    if any(math.isinf(value) for value in values):
        return float('-inf') if negative else float('inf')

    mantissa = -1.0 if negative else 1.0
    exponent = int(power)
    for value in values:
        part, part_exponent = math.frexp(abs(value))
        mantissa *= part
        exponent += part_exponent
        mantissa, normalization = math.frexp(mantissa)
        exponent += normalization
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float('-inf') if negative else float('inf')


def _power_scaled_product(
    power: int,
    *factors: object,
) -> np.ndarray:
    """Vectorized ``2**power * prod(factors)`` with range-safe fallbacks."""

    arrays = np.broadcast_arrays(
        *(np.asarray(factor, dtype=np.float64) for factor in factors)
    )
    shape = arrays[0].shape
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.full(shape, int(power), dtype=np.int64)
    regular = np.ones(shape, dtype=bool)
    zero = np.zeros(shape, dtype=bool)
    for array in arrays:
        regular &= np.isfinite(array) & (array != 0.0)
        zero |= array == 0.0
        part, part_exponent = np.frexp(array)
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)

    normal = (
        regular
        & (exponent >= _NORMAL_MIN_EXPONENT)
        & (exponent <= _NORMAL_MAX_EXPONENT)
    )
    result = np.zeros(shape, dtype=np.float64)
    if np.any(normal):
        result[normal] = _ldexp(mantissa[normal], exponent[normal])
    exceptional = ~normal & ~zero
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _power_scaled_product_scalar(
            int(power),
            *(float(array.flat[flat_index]) for array in arrays),
        )
    return result


def _finite_product_exponents(*factors: object) -> tuple[np.ndarray, np.ndarray]:
    """Return validity and normalized binary exponents for finite products."""

    arrays = np.broadcast_arrays(
        *(np.asarray(factor, dtype=np.float64) for factor in factors)
    )
    valid = np.ones(arrays[0].shape, dtype=bool)
    mantissa = np.ones(arrays[0].shape, dtype=np.float64)
    exponent = np.zeros(arrays[0].shape, dtype=np.int64)
    for array in arrays:
        valid &= np.isfinite(array) & (array != 0.0)
        part, part_exponent = np.frexp(np.abs(array))
        mantissa *= part
        exponent += part_exponent.astype(np.int64)
        mantissa, normalization = np.frexp(mantissa)
        exponent += normalization.astype(np.int64)
    return valid, exponent


def _power_scaled_sum_products(
    power: int,
    products: Sequence[Sequence[object]],
) -> np.ndarray:
    """Evaluate ``2**power * sum(products)`` as one expression."""

    scaled_products = tuple(
        (1.0,) + tuple(product)
        for product in products
    )
    flat_values = tuple(
        value
        for product in scaled_products
        for value in product
    )
    broadcast = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in flat_values)
    )
    broadcast_products: list[tuple[np.ndarray, ...]] = []
    offset = 0
    for product in scaled_products:
        width = len(product)
        broadcast_products.append(tuple(broadcast[offset:offset + width]))
        offset += width
    values = tuple(
        _power_scaled_product(power, *product)
        for product in broadcast_products
    )
    result = _stable_sum(*values)
    finite_inputs = np.logical_and.reduce(
        tuple(np.isfinite(array) for array in broadcast)
    )
    exceptional = finite_inputs & (
        ~np.isfinite(result)
        | _cancellation_risk(
            result,
            values,
            operation_count=(
                len(values)
                + sum(max(0, len(product) - 1) for product in products)
            ),
            relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        exact = Fraction(0)
        for product in broadcast_products:
            term = Fraction(1)
            for array in product:
                term *= _fraction(float(array.flat[flat_index]))
            exact += term
        if power >= 0:
            exact *= 1 << int(power)
        else:
            exact /= 1 << int(-power)
        try:
            result.flat[flat_index] = float(exact)
        except OverflowError:
            result.flat[flat_index] = (
                float('-inf') if exact < 0 else float('inf')
            )
    return result


def _stable_scaled_difference(
    left: object,
    right: object,
    *scale_factors: object,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vectorized ``prod(scale_factors) * (left - right)``."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (left, right, *scale_factors)
        )
    )
    left_array, right_array = arrays[:2]
    scale_arrays = arrays[2:]
    result = np.zeros(left_array.shape, dtype=np.float64)
    zero_scale = np.logical_or.reduce(
        tuple(array == 0.0 for array in scale_arrays)
    )
    difference_safe = _can_subtract(left_array, right_array)
    ordinary = difference_safe & ~zero_scale
    product_exceptional = np.zeros(result.shape, dtype=bool)
    if np.any(ordinary):
        difference = np.zeros_like(result)
        difference[ordinary] = (
            left_array[ordinary] - right_array[ordinary]
        )
        product, product_exceptional = _stable_product(
            *scale_arrays,
            difference,
            return_exceptional=True,
        )
        result[ordinary] = product[ordinary]
    exceptional = ~difference_safe & ~zero_scale
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _stable_scaled_difference_scalar(
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            *(float(array.flat[flat_index]) for array in scale_arrays),
        )
    if return_exceptional:
        return result, exceptional | (ordinary & product_exceptional)
    return result


def _stable_scaled_sum(
    values: Sequence[object],
    *scale_factors: object,
) -> np.ndarray:
    """Evaluate ``prod(scale_factors) * sum(values)`` compositionally."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (*values, *scale_factors)
        )
    )
    value_arrays = arrays[:len(values)]
    scale_arrays = arrays[len(values):]
    summed = _stable_sum(*value_arrays)
    result = _stable_product(*scale_arrays, summed)
    exceptional = (
        ~np.isfinite(summed)
        & np.logical_and.reduce(
            tuple(np.isfinite(array) for array in arrays)
        )
        & ~np.logical_or.reduce(
            tuple(array == 0.0 for array in scale_arrays)
        )
    )
    for flat_index in np.flatnonzero(exceptional):
        scales = tuple(
            float(array.flat[flat_index])
            for array in scale_arrays
        )
        result.flat[flat_index] = _exact_sum_products_scalar(
            tuple(
                scales + (float(array.flat[flat_index]),)
                for array in value_arrays
            )
        )
    return result


def _stable_ratio_difference(
    left: object,
    right: object,
    denominator: object,
) -> np.ndarray:
    """Vectorized scale-safe quotient of a difference."""

    left_array, right_array, denominator_array = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.empty(left_array.shape, dtype=np.float64)
    difference_safe = _can_subtract(left_array, right_array)
    ordinary = difference_safe & np.isfinite(denominator_array)
    if np.any(ordinary):
        difference = left_array[ordinary] - right_array[ordinary]
        result[ordinary] = _stable_ratio_product(
            (difference,),
            (denominator_array[ordinary],),
        )
    for flat_index in np.flatnonzero(~ordinary):
        result.flat[flat_index] = _stable_ratio_difference_scalar(
            float(left_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(denominator_array.flat[flat_index]),
        )
    return result


def _stable_affine_residual(
    beta: object,
    alpha: object,
    left: object,
    right: object,
    target: object,
    *,
    active: object | None = None,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate one complete affine residual with cancellation fallbacks."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (beta, alpha, left, right, target)
        )
    )
    beta_array, alpha_array, left_array, right_array, target_array = arrays
    if active is None:
        active_array = np.ones(beta_array.shape, dtype=bool)
    else:
        active_array = np.broadcast_to(
            np.asarray(active, dtype=bool),
            beta_array.shape,
        )
    result = np.zeros(beta_array.shape, dtype=np.float64)
    exceptional = np.zeros(beta_array.shape, dtype=bool)
    if not np.any(active_array):
        if return_exceptional:
            return result, exceptional
        return result

    active_beta = beta_array[active_array]
    active_alpha = alpha_array[active_array]
    active_left = left_array[active_array]
    active_right = right_array[active_array]
    active_target = target_array[active_array]
    left_product = _stable_product(active_alpha, active_left)
    right_product = _stable_product(
        -1.0,
        active_alpha,
        active_right,
    )
    terms = (
        active_beta,
        left_product,
        right_product,
        -active_target,
    )
    active_result = np.zeros(active_beta.shape, dtype=np.float64)
    range_safe = np.ones(active_beta.shape, dtype=bool)
    for term in terms:
        step_safe = _can_add(active_result, term)
        ordinary = range_safe & step_safe
        active_result[ordinary] += term[ordinary]
        range_safe &= step_safe

    finite_operands = np.logical_and.reduce(
        (
            np.isfinite(active_beta),
            np.isfinite(active_alpha),
            np.isfinite(active_left),
            np.isfinite(active_right),
            np.isfinite(active_target),
        )
    )
    vanished_product = (
        (
            np.isfinite(active_alpha)
            & (active_alpha != 0.0)
            & np.isfinite(active_left)
            & (active_left != 0.0)
            & (left_product == 0.0)
        )
        | (
            np.isfinite(active_alpha)
            & (active_alpha != 0.0)
            & np.isfinite(active_right)
            & (active_right != 0.0)
            & (right_product == 0.0)
        )
    )
    mixed, error_bound = _cancellation_error_bound(
        active_result,
        terms,
        operation_count=7,
    )
    exact_bound = _stable_product(
        np.abs(active_result),
        _CONDITIONING_RELATIVE_LIMIT,
    )
    exact = (
        ~range_safe
        | vanished_product
        | (
            finite_operands
            & ~np.isfinite(active_result)
        )
        | (
            mixed
            & (
                ~np.isfinite(active_result)
                | (error_bound >= exact_bound)
            )
        )
    )
    compensated_bound = _stable_product(
        np.abs(active_result),
        _AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    compensate = (
        ~exact
        & mixed
        & (error_bound >= compensated_bound)
    )
    if np.any(compensate):
        compensation_safe = np.logical_and.reduce(
            (
                finite_operands,
                _has_compensation_exponent_margin(
                    active_alpha,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_left,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_right,
                    lower_margin=32,
                    upper_margin=0,
                ),
                _has_compensation_exponent_margin(
                    active_beta,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    active_target,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    left_product,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    right_product,
                    lower_margin=64,
                    upper_margin=4,
                ),
                _has_compensation_exponent_margin(
                    active_result,
                    lower_margin=64,
                    upper_margin=4,
                ),
            )
        )
        unsafe_compensation = compensate & ~compensation_safe
        exact |= unsafe_compensation
        selected = np.flatnonzero(compensate & compensation_safe)
        if selected.size:
            active_result[selected] = _compensated_affine_residual(
                active_beta[selected],
                active_alpha[selected],
                active_left[selected],
                active_right[selected],
                active_target[selected],
                left_product[selected],
                right_product[selected],
            )

    for flat_index in np.flatnonzero(exact):
        active_result.flat[flat_index] = _exact_sum_products_scalar(
            (
                (float(active_beta.flat[flat_index]),),
                (
                    float(active_alpha.flat[flat_index]),
                    float(active_left.flat[flat_index]),
                ),
                (
                    -1.0,
                    float(active_alpha.flat[flat_index]),
                    float(active_right.flat[flat_index]),
                ),
                (-float(active_target.flat[flat_index]),),
            )
        )
    result[active_array] = active_result
    exceptional[active_array] = exact
    if return_exceptional:
        return result, exceptional
    return result


def _stable_scaled_affine_residual(
    beta: object,
    alpha: object,
    left: object,
    right: object,
    target: object,
    *scale_factors: object,
    active: object | None = None,
    return_exceptional: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate a scaled complete affine residual without storing it first."""

    arrays = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=np.float64)
            for value in (
                beta,
                alpha,
                left,
                right,
                target,
                *scale_factors,
            )
        )
    )
    beta_array, alpha_array, left_array, right_array, target_array = arrays[:5]
    scale_arrays = arrays[5:]
    if active is None:
        active_array = np.ones(beta_array.shape, dtype=bool)
    else:
        active_array = np.broadcast_to(
            np.asarray(active, dtype=bool),
            beta_array.shape,
        )
    result = np.zeros(beta_array.shape, dtype=np.float64)
    exceptional = np.zeros(beta_array.shape, dtype=bool)
    if not np.any(active_array):
        if return_exceptional:
            return result, exceptional
        return result
    active_scales = tuple(array[active_array] for array in scale_arrays)
    active_result = _stable_sum_products(
        (
            active_scales + (beta_array[active_array],),
            active_scales + (
                alpha_array[active_array],
                left_array[active_array],
            ),
            (-1.0,) + active_scales + (
                alpha_array[active_array],
                right_array[active_array],
            ),
            (-1.0,) + active_scales + (target_array[active_array],),
        ),
        return_exceptional=return_exceptional,
    )
    if return_exceptional:
        active_values, active_exceptional = active_result
        result[active_array] = active_values
        exceptional[active_array] = active_exceptional
        return result, exceptional
    result[active_array] = active_result
    return result


def _stable_affine_difference(
    beta: object,
    alpha: object,
    left: object,
    right: object,
) -> np.ndarray:
    """Evaluate ``beta + alpha * left - alpha * right`` directly."""

    return _stable_affine_residual(
        beta,
        alpha,
        left,
        right,
        0.0,
    )


def _stable_weighted_average(
    left: object,
    left_weight: object,
    right: object,
    right_weight: object,
) -> np.ndarray:
    """Return a stable convex weighted average of two arrays."""

    left_array, left_weight_array, right_array, right_weight_array = (
        np.broadcast_arrays(
            np.asarray(left, dtype=np.float64),
            np.asarray(left_weight, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            np.asarray(right_weight, dtype=np.float64),
        )
    )
    scale = np.maximum(left_weight_array, right_weight_array)
    valid_scale = np.isfinite(scale) & (scale != 0.0)
    left_scaled = _stable_normalized_ratio(
        left_weight_array,
        scale,
        active=valid_scale,
    )
    right_scaled = _stable_normalized_ratio(
        right_weight_array,
        scale,
        active=valid_scale,
    )
    denominator = _stable_sum(left_scaled, right_scaled)
    right_fraction = _stable_normalized_ratio(
        right_scaled,
        denominator,
        active=valid_scale & (denominator != 0.0),
    )
    adjustment = _stable_scaled_difference(
        right_array,
        left_array,
        right_fraction,
    )
    result = _stable_sum(left_array, adjustment)
    left_term = _stable_product(left_scaled, left_array)
    right_term = _stable_product(right_scaled, right_array)
    scaled_numerator = _stable_sum(left_term, right_term)
    exceptional = _cancellation_risk(
        scaled_numerator,
        (left_term, right_term),
        operation_count=5,
        relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional |= _cancellation_risk(
        result,
        (left_array, adjustment),
        operation_count=5,
        relative_limit=_AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional |= (
        ((left_weight_array != 0.0) & (left_scaled == 0.0))
        | ((right_weight_array != 0.0) & (right_scaled == 0.0))
    )
    exceptional |= (
        (left_weight_array > 0.0)
        & (right_weight_array > 0.0)
        & (
            np.minimum(left_scaled, right_scaled)
            < _AFFINE_CONDITIONING_RELATIVE_LIMIT
        )
    )
    finite_inputs = (
        np.isfinite(left_array)
        & np.isfinite(left_weight_array)
        & np.isfinite(right_array)
        & np.isfinite(right_weight_array)
    )
    exceptional |= finite_inputs & ~np.isfinite(result)
    exceptional &= finite_inputs & valid_scale
    for flat_index in np.flatnonzero(exceptional):
        result.flat[flat_index] = _exact_weighted_average_scalar(
            float(left_array.flat[flat_index]),
            float(left_weight_array.flat[flat_index]),
            float(right_array.flat[flat_index]),
            float(right_weight_array.flat[flat_index]),
        )
    result[~valid_scale] = float('nan')
    return result


def _stable_incidence_accumulate(
    n_sites: int,
    site_i: np.ndarray,
    site_j: np.ndarray,
    row_values: np.ndarray,
) -> np.ndarray:
    """Return ``B @ row_values`` with masked exact site fallbacks."""

    n = int(n_sites)
    i = np.asarray(site_i, dtype=np.int64)
    j = np.asarray(site_j, dtype=np.int64)
    values = np.asarray(row_values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n, dtype=np.float64)
    degree = (
        np.bincount(i, minlength=n)
        + np.bincount(j, minlength=n)
    )
    if not np.all(np.isfinite(values)):
        terms: list[list[float]] = [[] for _ in range(n)]
        for site_i_value, site_j_value, value in zip(
            i.tolist(),
            j.tolist(),
            values.tolist(),
        ):
            terms[int(site_i_value)].append(float(value))
            terms[int(site_j_value)].append(-float(value))
        return np.asarray(
            [_stable_sum_scalar(*site_terms) for site_terms in terms],
            dtype=np.float64,
        )

    absolute = np.abs(values)
    site_scale = np.zeros(n, dtype=np.float64)
    np.maximum.at(site_scale, i, absolute)
    np.maximum.at(site_scale, j, absolute)
    safe_limit = np.full(n, _FLOAT_MAX, dtype=np.float64)
    occupied = degree != 0
    safe_limit[occupied] /= degree[occupied]
    range_safe = ~occupied | (site_scale <= safe_limit)

    i_values = np.zeros_like(values)
    j_values = np.zeros_like(values)
    i_safe = range_safe[i]
    j_safe = range_safe[j]
    i_values[i_safe] = values[i_safe]
    j_values[j_safe] = values[j_safe]
    result = (
        np.bincount(i, weights=i_values, minlength=n)
        - np.bincount(j, weights=j_values, minlength=n)
    )

    positive = (
        np.bincount(i, weights=values > 0.0, minlength=n)
        + np.bincount(j, weights=values < 0.0, minlength=n)
    ) > 0.0
    negative = (
        np.bincount(i, weights=values < 0.0, minlength=n)
        + np.bincount(j, weights=values > 0.0, minlength=n)
    ) > 0.0
    mixed = positive & negative

    i_material = (
        (absolute != 0.0)
        & (site_scale[i] != 0.0)
    )
    j_material = (
        (absolute != 0.0)
        & (site_scale[j] != 0.0)
    )
    i_normalized = _stable_normalized_ratio(
        absolute,
        site_scale[i],
        active=i_material,
    )
    j_normalized = _stable_normalized_ratio(
        absolute,
        site_scale[j],
        active=j_material,
    )
    scaled_l1 = (
        np.bincount(i, weights=i_normalized, minlength=n)
        + np.bincount(j, weights=j_normalized, minlength=n)
    )
    error_bound = _stable_product(
        site_scale,
        scaled_l1,
        (degree + 2) * _FLOAT_EPSILON,
    )
    relative_limit = _stable_product(
        np.abs(result),
        _AFFINE_CONDITIONING_RELATIVE_LIMIT,
    )
    exceptional = ~range_safe
    exceptional |= mixed & (
        ~np.isfinite(result)
        | (error_bound >= relative_limit)
    )
    exceptional_sites = np.flatnonzero(exceptional)
    if exceptional_sites.size:
        exceptional_index = np.full(n, -1, dtype=np.int64)
        exceptional_index[exceptional_sites] = np.arange(
            exceptional_sites.size,
            dtype=np.int64,
        )
        grouped_terms: list[list[float]] = [
            [] for _ in range(exceptional_sites.size)
        ]
        exceptional_i = exceptional[i]
        for group, value in zip(
            exceptional_index[i[exceptional_i]].tolist(),
            values[exceptional_i].tolist(),
        ):
            grouped_terms[group].append(float(value))
        exceptional_j = exceptional[j]
        for group, value in zip(
            exceptional_index[j[exceptional_j]].tolist(),
            values[exceptional_j].tolist(),
        ):
            grouped_terms[group].append(-float(value))
        result[exceptional_sites] = np.asarray(
            [
                _stable_sum_scalar(*site_terms)
                for site_terms in grouped_terms
            ],
            dtype=np.float64,
        )
    return np.asarray(result, dtype=np.float64)


def _scaled_sum_squares_state(values: object) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    if count == 0:
        return 0.0, 1.0, 0
    absolute = np.abs(array.ravel())
    if np.any(np.isnan(absolute)):
        return float('nan'), float('nan'), count
    scale = float(np.max(absolute))
    if scale == 0.0 or math.isinf(scale):
        return scale, 1.0, count
    scaled = np.zeros_like(absolute)
    _, exponents = np.frexp(absolute)
    _, scale_exponent = math.frexp(scale)
    # Ratios below 2**-500 cannot affect a binary64 norm for any realizable
    # array, and omitting them avoids an underflowing square.
    material = (absolute != 0.0) & (
        exponents >= scale_exponent - 500
    )
    scaled[material] = absolute[material] / scale
    return scale, float(np.dot(scaled, scaled)), count


def _stable_norm(values: object) -> float:
    """Return a scale-safe Euclidean norm."""

    scale, sum_squares, _ = _scaled_sum_squares_state(values)
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, math.sqrt(sum_squares))


def _stable_sum_squares(values: object) -> float:
    """Return a scale-safe sum of squares."""

    scale, sum_squares, _ = _scaled_sum_squares_state(values)
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, scale, sum_squares)


def _stable_rms(values: object) -> float:
    """Return a scale-safe root-mean-square value."""

    scale, sum_squares, count = _scaled_sum_squares_state(values)
    if count == 0 or scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return scale
    return _stable_product_scalar(scale, math.sqrt(sum_squares / count))


def _stable_mean_abs(values: object) -> float:
    """Return a scale-safe mean absolute value."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    absolute = np.abs(array.ravel())
    if np.any(np.isnan(absolute)):
        return float('nan')
    scale = float(np.max(absolute))
    if scale == 0.0 or math.isinf(scale):
        return scale
    scaled = np.zeros_like(absolute)
    _, exponents = np.frexp(absolute)
    _, scale_exponent = math.frexp(scale)
    # A ratio below the normal range cannot affect the rounded mean for any
    # realizable NumPy array.  Omitting it avoids executing an inexact
    # subnormal division under a strict floating-point error policy.
    material = (absolute != 0.0) & (
        exponents >= scale_exponent + _NORMAL_MIN_EXPONENT
    )
    scaled[material] = absolute[material] / scale
    return _stable_product_scalar(scale, float(np.mean(scaled)))
