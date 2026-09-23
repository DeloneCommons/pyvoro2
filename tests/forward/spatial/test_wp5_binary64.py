"""Literal boundary expectations for ADR 0021 binary64/source primitives."""

from fractions import Fraction as F
import math

import pytest

from pyvoro2._internal.spatial import wp5_binary64 as binary64


@pytest.mark.parametrize('value, expected', [
    (0.0, (-F(1, 2**1075), F(1, 2**1075))),
    (-0.0, (-F(1, 2**1075), F(1, 2**1075))),
    (float.fromhex('0x0.0000000000001p-1022'),
     (F(1, 2**1075), F(3, 2**1075))),
    (-float.fromhex('0x0.0000000000001p-1022'),
     (-F(3, 2**1075), -F(1, 2**1075))),
    (float.fromhex('0x1p-1022'),
     (F(2**53 - 1, 2**1075), F(2**53 + 1, 2**1075))),
    (1.0, (F(2**54 - 1, 2**54), F(2**53 + 1, 2**53))),
    (-1.0, (-F(2**53 + 1, 2**53), -F(2**54 - 1, 2**54))),
    (float.fromhex('0x1.fffffffffffffp+1023'),
     (F(2**1024 - 3 * 2**970), F(2**1024 - 2**970))),
    (-float.fromhex('0x1.fffffffffffffp+1023'),
     (-F(2**1024 - 2**970), -F(2**1024 - 3 * 2**970))),
])
def test_closed_rounding_bins_include_exact_boundary_endpoints(value, expected):
    assert binary64.rounding_bin(value) == expected


@pytest.mark.parametrize('interval, expected', [
    ((F(0), F(0)), (-F(1, 2**1075), F(1, 2**1075))),
    ((F(1), F(1)),
     (F(2**54 - 1, 2**54), F(2**53 + 1, 2**53))),
    ((F(1) + F(1, 2**54), F(1) + F(3, 2**53)),
     (F(1) + F(1, 2**53), F(1) + F(3, 2**53))),
    ((-F(1) - F(3, 2**53), -F(1) - F(1, 2**54)),
     (-F(1) - F(3, 2**53), -F(1) - F(1, 2**53))),
    ((F(1, 2**1076), F(3, 2**1076)), None),
    ((F(1) + F(1, 2**54), F(1) + F(1, 2**53)), None),
    ((F(2**1024), F(2**1025)), None),
    ((-F(2**1025), -F(2**1024)), None),
    ((F(2), F(1)), None),
    ((-F(2**1025), F(2**1025)),
     (-F(2**1024 - 2**970), F(2**1024 - 2**970))),
])
def test_preimages_select_only_contained_finite_floats(interval, expected):
    assert binary64.rounding_preimage(interval) == expected


def test_conservative_closed_bin_retains_a_tie_rejected_by_forward_rounding():
    # 1 + 2**-52 has odd significand; the lower midpoint rounds down to 1.
    odd = float.fromhex('0x1.0000000000001p+0')
    midpoint = F(1) + F(1, 2**53)
    assert float(midpoint) == 1.0
    assert binary64.rounding_bin(odd)[0] == midpoint


def test_exact_interval_arithmetic_and_inclusive_integer_bounds():
    a, b = (F(-3, 2), F(5, 2)), (F(1, 3), F(2, 3))
    assert binary64.interval_add(a, b) == (F(-7, 6), F(19, 6))
    assert binary64.interval_sub(a, b) == (F(-13, 6), F(13, 6))
    assert binary64.interval_div(a, F(2, 3)) == (F(-9, 4), F(15, 4))
    assert binary64.integer_interval((F(-3), F(4))) == (-3, 4)
    assert binary64.integer_interval((F(-5, 2), F(7, 2))) == (-2, 3)
    assert binary64.integer_interval((F(1, 4), F(3, 4))) == (1, 0)


@pytest.mark.parametrize('value, expected', [
    (-2, -3), (F(-5, 2), -3), (F(-1, 2), -1),
    (-0.0, 0), (0.0, 0), (F(1, 2), 0), (2, 2),
])
def test_source_step_uses_truncation_then_negative_correction(value, expected):
    assert binary64.step(value) == expected


@pytest.mark.parametrize('value, quotient, remainder', [
    (-7, -3, 2), (-6, -2, 0), (-5, -2, 1),
    (-1, -1, 2), (0, 0, 0), (1, 0, 1), (6, 2, 0), (7, 2, 1),
])
def test_source_integer_division_and_modulus(value, quotient, remainder):
    assert binary64.div(value, 3) == quotient
    assert binary64.mod(value, 3) == remainder


def test_integer_source_helpers_do_not_round_large_python_integers():
    value = -(2**100) - 1
    assert binary64.div(value, 2) == -(2**99) - 1
    assert binary64.mod(value, 2) == 1


def test_bit_comparison_distinguishes_signed_zero():
    assert binary64.bits_equal(0.0, 0.0)
    assert binary64.bits_equal(-0.0, -0.0)
    assert not binary64.bits_equal(0.0, -0.0)
    assert not binary64.bits_equal(1.0, math.nextafter(1.0, math.inf))


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan])
def test_rounding_bin_rejects_nonfinite_operands(value):
    with pytest.raises(ValueError, match='finite'):
        binary64.rounding_bin(value)


@pytest.mark.parametrize('divisor', [0, -1])
def test_nonpositive_source_divisors_are_rejected(divisor):
    for function in (binary64.div, binary64.mod):
        with pytest.raises(ValueError, match='positive'):
            function(1, divisor)
    with pytest.raises(ValueError, match='positive'):
        binary64.interval_div((F(0), F(1)), divisor)
