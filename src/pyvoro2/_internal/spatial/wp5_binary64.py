"""Exact conservative binary64 preimages for the WP5 source replay.

The closed enclosures deliberately retain both midpoint ties. Only the
separate forward replay decides whether a rounded history is compatible.
No function here performs the producer's signed-int32 profile checks.
"""

from __future__ import annotations

from fractions import Fraction
import math
import struct


Interval = tuple[Fraction, Fraction]

_SIGN = 1 << 63
_MASK = (1 << 64) - 1
# Reverse the negative encodings and put positive encodings after them.
# These are the first and last finite keys; signed zeros remain consecutive.
_FIRST_FINITE = 0x0010000000000000
_LAST_FINITE = 0xFFEFFFFFFFFFFFFF
_ZERO_HALF_ULP = Fraction(1, 1 << 1075)
_MAX_HALF_ULP = Fraction(1 << 970)


def bits_equal(left: float, right: float) -> bool:
    """Compare binary64 encodings, including the sign of a zero."""
    return struct.pack('>d', left) == struct.pack('>d', right)


def rounding_bin(value: float) -> Interval:
    """Return the closed exact bin B(value) for a finite binary64 value."""
    if not math.isfinite(value):
        raise ValueError('a binary64 rounding bin requires a finite value')
    if value == 0.0:
        return -_ZERO_HALF_ULP, _ZERO_HALF_ULP
    exact = Fraction(value)
    previous = math.nextafter(value, -math.inf)
    following = math.nextafter(value, math.inf)
    lower = ((Fraction(previous) + exact) / 2
             if math.isfinite(previous) else exact - _MAX_HALF_ULP)
    upper = ((exact + Fraction(following)) / 2
             if math.isfinite(following) else exact + _MAX_HALF_ULP)
    return lower, upper


def _float_at_key(key: int) -> float:
    bits = key & ~_SIGN if key & _SIGN else ~key & _MASK
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


def _finite_boundary(value: Fraction, *, strict: bool) -> int:
    """First finite key >= value (or > value), with one past-end sentinel.

    Every comparison exactifies one binary64 operand. Binary search therefore
    takes at most 64 iterations, even for an interval spanning all floats.
    """
    low, high = _FIRST_FINITE, _LAST_FINITE + 1
    while low < high:
        middle = (low + high) // 2
        exact = Fraction(_float_at_key(middle))
        if exact < value or (strict and exact == value):
            low = middle + 1
        else:
            high = middle
    return low


def rounding_preimage(interval: Interval) -> Interval | None:
    """Return P(interval), or None when it contains no finite binary64.

    The hull is determined by the bins of the exact first and last contained
    values. The binary64 universe is never enumerated or bounded by a float
    conversion of the rational endpoints.
    """
    lower, upper = map(Fraction, interval)
    if lower > upper:
        return None
    first = _finite_boundary(lower, strict=False)
    after_last = _finite_boundary(upper, strict=True)
    if first >= after_last:
        return None
    return (rounding_bin(_float_at_key(first))[0],
            rounding_bin(_float_at_key(after_last - 1))[1])


def interval_add(left: Interval, right: Interval) -> Interval:
    """Add exact closed intervals."""
    return (Fraction(left[0]) + Fraction(right[0]),
            Fraction(left[1]) + Fraction(right[1]))


def interval_sub(left: Interval, right: Interval) -> Interval:
    """Subtract exact closed intervals."""
    return (Fraction(left[0]) - Fraction(right[1]),
            Fraction(left[1]) - Fraction(right[0]))


def interval_div(interval: Interval, positive: Fraction) -> Interval:
    """Divide exact endpoints by a positive source operand."""
    positive = Fraction(positive)
    if positive <= 0:
        raise ValueError('interval division requires a positive divisor')
    return Fraction(interval[0]) / positive, Fraction(interval[1]) / positive


def integer_interval(interval: Interval) -> tuple[int, int]:
    """Inclusive integer bounds; lower > upper represents an empty family."""
    return math.ceil(Fraction(interval[0])), math.floor(Fraction(interval[1]))


def step(value: Fraction | float | int) -> int:
    """Native Step, including trunc(x)-1 at negative exact integers."""
    truncated = int(value)
    return truncated - 1 if value < 0 else truncated


def div(value: int, divisor: int) -> int:
    """Native Div with integer truncation; no floating quotient is formed."""
    if divisor <= 0:
        raise ValueError('source division requires a positive divisor')
    return (value // divisor if value >= 0
            else -1 - (-(value + 1) // divisor))


def mod(value: int, divisor: int) -> int:
    """Native Mod, retaining its explicit negative-source expression."""
    if divisor <= 0:
        raise ValueError('source modulus requires a positive divisor')
    return (value % divisor if value >= 0
            else divisor - 1 - (divisor - 1 - value) % divisor)
