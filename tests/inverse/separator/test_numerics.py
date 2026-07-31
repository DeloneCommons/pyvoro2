"""Cross-platform regressions for private separator numeric kernels."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import pyvoro2.inverse.separator as separator
import pyvoro2.inverse.separator._numerics as separator_numerics
import pyvoro2.inverse.separator._objective as separator_objective
import pyvoro2.inverse.separator._quadratic as separator_quadratic


def test_numeric_binary64_lattice_collapses_signed_zero() -> None:
    tiny = math.ulp(0.0)
    assert separator_numerics._float_to_ordered_int(-0.0) == (
        separator_numerics._float_to_ordered_int(0.0)
    )
    assert separator_numerics._ordered_floats_adjacent(-tiny, -0.0)
    assert separator_numerics._ordered_floats_adjacent(0.0, tiny)
    assert not separator_numerics._ordered_floats_adjacent(-tiny, tiny)
    assert separator_numerics._ordered_float_midpoint(-tiny, tiny) == 0.0


@pytest.mark.parametrize(
    ('lower', 'upper'),
    [
        (-np.finfo(np.float64).max, -1.0),
        (-2.0, -1.0),
        (-math.ulp(0.0), math.ulp(0.0)),
        (1.0, 2.0),
        (1.0, np.finfo(np.float64).max),
    ],
)
def test_ordered_midpoint_strictly_contracts_or_interval_is_adjacent(
    lower: float,
    upper: float,
) -> None:
    if separator_numerics._ordered_floats_adjacent(lower, upper):
        return
    midpoint = separator_numerics._ordered_float_midpoint(lower, upper)
    assert lower < midpoint < upper


def test_adjacent_lattice_at_finite_extrema_and_normal_values() -> None:
    maximum = np.finfo(np.float64).max
    pairs = (
        (-maximum, math.nextafter(-maximum, math.inf)),
        (-1.0, math.nextafter(-1.0, math.inf)),
        (1.0, math.nextafter(1.0, math.inf)),
        (math.nextafter(maximum, -math.inf), maximum),
    )
    for lower, upper in pairs:
        assert separator_numerics._ordered_floats_adjacent(lower, upper)


def test_double_double_ratio_retains_large_cancellation_low_parts() -> None:
    first = separator_numerics._dd_divide_float(
        separator_numerics._dd_divide_float(
            separator_numerics._DoubleDouble(1e12),
            0.1,
        ),
        0.1,
    )
    second = separator_numerics._dd_divide_float(
        separator_numerics._dd_divide_float(
            separator_numerics._DoubleDouble(
                math.nextafter(1e12, math.inf)
            ),
            0.1,
        ),
        0.1,
    )
    difference = separator_numerics._dd_add(
        second,
        separator_numerics._dd_negate(first),
    )
    expected = (
        math.nextafter(1e12, math.inf) - 1e12
    ) / (0.1 * 0.1)
    assert difference.value == pytest.approx(expected, rel=2e-16)


def test_short_expansion_sum_retains_half_ulp_low_limb() -> None:
    result = separator_numerics._dd_sum(
        (
            separator_numerics._DoubleDouble(0.5),
            separator_numerics._DoubleDouble(2.0**-54),
        )
    )
    assert result.high == 0.5
    assert result.low == 2.0**-54


def test_scalar_split_and_product_are_safe_at_finite_extrema() -> None:
    maximum = np.finfo(np.float64).max
    for value in (-maximum, maximum):
        high, low = separator_numerics._split_scalar(value)
        assert math.isfinite(high)
        assert math.isfinite(low)
        assert math.fsum((high, low)) == value
        product = separator_numerics._two_product_scalar(value, 1.0)
        assert product.high == value
        assert product.low == 0.0


def test_scalar_product_retains_exact_normal_mantissa_correction() -> None:
    value = math.nextafter(1.0, 0.0)
    product = separator_numerics._two_product_scalar(-value, value)
    represented = (
        Fraction.from_float(product.high)
        + Fraction.from_float(product.low)
    )
    assert represented == -Fraction.from_float(value) ** 2


def test_vectorized_ldexp_paths_use_c_int_exponents(monkeypatch) -> None:
    """Keep NumPy 1.x Windows ``ldexp`` compatibility on every vector path."""

    original_ldexp = np.ldexp
    exponent_dtypes: list[np.dtype] = []

    def windows_numpy_1x_ldexp(
        mantissa: object,
        exponent: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        exponent_dtype = np.asarray(exponent).dtype
        exponent_dtypes.append(exponent_dtype)
        if exponent_dtype == np.dtype(np.int64):
            raise TypeError(
                "ufunc 'ldexp' not supported for float64/int64 inputs"
            )
        return original_ldexp(mantissa, exponent, *args, **kwargs)

    monkeypatch.setattr(np, 'ldexp', windows_numpy_1x_ldexp)

    fit = separator.fit_weights_from_separators(
        np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64),
        [(0, 1, 0.25)],
        solver='direct',
        linear_backend='dense',
        connectivity_check='diagnose',
    )

    extended_mantissa = np.array([0.75], dtype=np.longdouble)
    extended = separator_numerics._ldexp(
        extended_mantissa,
        np.array([2], dtype=np.int64),
    )
    ratio = separator_numerics._stable_ratio_product(
        (np.array([0.5]),),
        (np.array([0.25]),),
    )
    product = separator_numerics._stable_product(
        np.array([1.0e308]),
        np.array([1.0e308]),
        np.array([1.0e-308]),
    )
    normalized = separator_numerics._stable_normalized_ratio(
        np.array([1.0]),
        np.array([2.0]),
        active=np.array([True]),
    )
    split_high, split_low = separator_numerics._split_product_operand(
        np.array([1.1]),
    )
    scaled = separator_numerics._power_scaled_product(
        2,
        np.array([0.5]),
    )
    product_high, product_low = (
        separator_quadratic._power_scaled_product_parts(
            0,
            np.array([1.1]),
            np.array([1.1]),
        )
    )

    assert fit.status == 'optimal'
    assert fit.solver == 'direct'
    assert fit.linear_backend == 'dense'
    assert extended.dtype == extended_mantissa.dtype
    np.testing.assert_array_equal(extended, np.array([3.0], dtype=np.longdouble))
    np.testing.assert_array_equal(ratio, np.array([2.0]))
    np.testing.assert_allclose(product, np.array([1.0e308]), rtol=2e-15)
    np.testing.assert_array_equal(normalized, np.array([0.5]))
    np.testing.assert_array_equal(split_high + split_low, np.array([1.1]))
    np.testing.assert_array_equal(scaled, np.array([2.0]))
    np.testing.assert_allclose(
        product_high + product_low,
        np.array([1.21]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )
    assert exponent_dtypes
    assert set(exponent_dtypes) == {np.dtype(np.intc)}


def _assert_ball_contains(
    ball: separator_numerics._TwofoldBall,
    exact: Fraction,
) -> None:
    lower, upper = ball.physical_bounds()
    assert ball.resolved
    assert Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)


def test_twofold_ball_source_normalization_and_operation_radii() -> None:
    a = separator_numerics._TwofoldBall.point(1.1)
    b = separator_numerics._TwofoldBall.point(-0.3)
    added = separator_numerics._ball_add(a, b)
    subtracted = separator_numerics._ball_subtract(a, b)
    multiplied = separator_numerics._ball_multiply(a, b)
    divided = separator_numerics._ball_divide(a, b)
    squared = separator_numerics._ball_square(a)
    scaled = separator_numerics._ball_integer_scale(a, -37)
    shifted = separator_numerics._ball_ldexp(a, 17)
    fa = Fraction.from_float(1.1)
    fb = Fraction.from_float(-0.3)
    for ball, exact in (
        (added, fa + fb),
        (subtracted, fa - fb),
        (multiplied, fa * fb),
        (divided, fa / fb),
        (squared, fa * fa),
        (scaled, -37 * fa),
        (shifted, fa * 2**17),
    ):
        _assert_ball_contains(ball, exact)
        if ball.high != 0.0:
            assert abs(ball.low) <= math.ulp(ball.high)


def test_generated_twofold_ball_operations_contain_exact_dyadics() -> None:
    rng = np.random.default_rng(20260803)
    for _ in range(100):
        exponent_a, exponent_b = rng.integers(-300, 301, size=2)
        a = math.ldexp(
            float(rng.uniform(0.5, 1.0)) * (-1.0 if rng.integers(2) else 1.0),
            int(exponent_a),
        )
        b = math.ldexp(
            float(rng.uniform(0.5, 1.0)) * (-1.0 if rng.integers(2) else 1.0),
            int(exponent_b),
        )
        ball_a = separator_numerics._TwofoldBall.point(a)
        ball_b = separator_numerics._TwofoldBall.point(b)
        exact_a = Fraction.from_float(a)
        exact_b = Fraction.from_float(b)
        for ball, exact in (
            (separator_numerics._ball_add(ball_a, ball_b), exact_a + exact_b),
            (
                separator_numerics._ball_subtract(ball_a, ball_b),
                exact_a - exact_b,
            ),
            (
                separator_numerics._ball_multiply(ball_a, ball_b),
                exact_a * exact_b,
            ),
            (
                separator_numerics._ball_divide(ball_a, ball_b),
                exact_a / exact_b,
            ),
        ):
            _assert_ball_contains(ball, exact)


def test_twofold_ball_independent_input_radii_are_not_lost() -> None:
    radius_a = math.ldexp(1.0, -50)
    radius_b = math.ldexp(1.0, -48)
    a = separator_numerics._TwofoldBall(1.25, 0.0, radius_a, True)
    b = separator_numerics._TwofoldBall(-0.75, 0.0, radius_b, True)
    result = separator_numerics._ball_multiply(a, b)
    for exact_a in (
        Fraction.from_float(1.25 - radius_a),
        Fraction.from_float(1.25 + radius_a),
    ):
        for exact_b in (
            Fraction.from_float(-0.75 - radius_b),
            Fraction.from_float(-0.75 + radius_b),
        ):
            _assert_ball_contains(result, exact_a * exact_b)


def test_nonzero_radius_is_retained_by_every_ball_operation() -> None:
    radius_a = math.ldexp(1.0, -40)
    radius_b = math.ldexp(1.0, -42)
    a = separator_numerics._TwofoldBall(1.25, 0.0, radius_a, True)
    b = separator_numerics._TwofoldBall(0.75, 0.0, radius_b, True)
    a_values = tuple(map(Fraction.from_float, (
        1.25 - radius_a,
        1.25 + radius_a,
    )))
    b_values = tuple(map(Fraction.from_float, (
        0.75 - radius_b,
        0.75 + radius_b,
    )))
    operations = (
        (
            separator_numerics._ball_add(a, b),
            tuple(left + right for left in a_values for right in b_values),
        ),
        (
            separator_numerics._ball_subtract(a, b),
            tuple(left - right for left in a_values for right in b_values),
        ),
        (
            separator_numerics._ball_multiply(a, b),
            tuple(left * right for left in a_values for right in b_values),
        ),
        (
            separator_numerics._ball_divide(a, b),
            tuple(left / right for left in a_values for right in b_values),
        ),
        (
            separator_numerics._ball_reciprocal(b),
            tuple(1 / value for value in b_values),
        ),
        (
            separator_numerics._ball_square(a),
            tuple(value * value for value in a_values),
        ),
        (
            separator_numerics._ball_integer_scale(a, -13),
            tuple(-13 * value for value in a_values),
        ),
        (
            separator_numerics._ball_ldexp(a, 37),
            tuple(value * 2**37 for value in a_values),
        ),
    )
    for ball, exact_values in operations:
        for exact in exact_values:
            _assert_ball_contains(ball, exact)


def test_ball_extrema_signed_zero_and_subnormal_boundaries() -> None:
    maximum = np.finfo(np.float64).max
    minimum = math.ulp(0.0)
    signed_zero = separator_numerics._TwofoldBall.point(-0.0)
    assert signed_zero.resolved
    assert signed_zero.physical_bounds() == (-0.0, 0.0)
    cancellation = separator_numerics._ball_add(
        separator_numerics._TwofoldBall.point(maximum),
        separator_numerics._TwofoldBall.point(-maximum),
    )
    _assert_ball_contains(cancellation, Fraction(0))
    finite_product = separator_numerics._ball_multiply(
        separator_numerics._TwofoldBall.point(maximum),
        separator_numerics._TwofoldBall.point(0.5),
    )
    _assert_ball_contains(
        finite_product,
        Fraction.from_float(maximum) / 2,
    )
    finite_ratio = separator_numerics._ball_divide(
        separator_numerics._TwofoldBall.point(maximum),
        separator_numerics._TwofoldBall.point(maximum),
    )
    _assert_ball_contains(finite_ratio, Fraction(1))
    shifted_subnormal = separator_numerics._ball_ldexp(
        separator_numerics._TwofoldBall.point(minimum),
        1,
    )
    _assert_ball_contains(
        shifted_subnormal,
        2 * Fraction.from_float(minimum),
    )


def test_twofold_ball_exceptional_range_is_fail_closed() -> None:
    maximum = np.finfo(np.float64).max
    overflow = separator_numerics._ball_multiply(
        separator_numerics._TwofoldBall.point(maximum),
        separator_numerics._TwofoldBall.point(2.0),
    )
    underflow = separator_numerics._ball_multiply(
        separator_numerics._TwofoldBall.point(math.ulp(0.0)),
        separator_numerics._TwofoldBall.point(0.5),
    )
    zero_denominator = separator_numerics._ball_divide(
        separator_numerics._TwofoldBall.point(1.0),
        separator_numerics._TwofoldBall(0.0, 0.0, math.ulp(0.0), True),
    )
    for ball in (overflow, underflow, zero_denominator):
        assert not ball.resolved
        assert ball.physical_bounds() == (-math.inf, math.inf)


def test_binary_scaled_ball_accumulation_preserves_opposing_terms() -> None:
    leading = separator_numerics._BinaryScaledBall(
        separator_numerics._TwofoldBall.point(0.75),
        900,
    )
    opposite = separator_numerics._BinaryScaledBall(
        separator_numerics._TwofoldBall(
            -0.75,
            math.ldexp(1.0, -53),
            0.0,
            True,
        ),
        900,
    )
    tiny = separator_numerics._BinaryScaledBall(
        separator_numerics._TwofoldBall.point(0.5),
        847,
    )
    result = separator_numerics._binary_scaled_sum((leading, opposite, tiny))
    exact = (
        (Fraction.from_float(0.75)
         + Fraction.from_float(-0.75)
         + Fraction.from_float(math.ldexp(1.0, -53))) * 2**900
        + Fraction.from_float(0.5) * 2**847
    )
    lower, upper = result.physical_bounds()
    assert Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)


def test_binary_scaled_alignment_underflow_is_retained_in_radius() -> None:
    leading = separator_numerics._BinaryScaledBall(
        separator_numerics._TwofoldBall.point(1.0),
        0,
    )
    lost_from_center = separator_numerics._BinaryScaledBall(
        separator_numerics._TwofoldBall.point(-1.0),
        -1100,
    )
    result = separator_numerics._binary_scaled_sum(
        (leading, lost_from_center)
    )
    exact = Fraction(1) - Fraction(1, 2**1100)
    lower, upper = result.physical_bounds()
    assert Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)
    assert result.ball.radius > 0.0


def test_generated_ln2_ball_contains_independent_atanh_series() -> None:
    terms = 161
    lower = sum(
        (
            Fraction(2, (2 * index + 1) * 3 ** (2 * index + 1))
            for index in range(terms)
        ),
        Fraction(0),
    )
    first_omitted = Fraction(
        2,
        (2 * terms + 1) * 3 ** (2 * terms + 1),
    )
    upper = lower + Fraction(9, 8) * first_omitted
    ball = separator_objective._LN2_BALL
    center = Fraction.from_float(ball.high) + Fraction.from_float(ball.low)
    radius = Fraction.from_float(ball.radius)
    assert center - radius <= lower < upper <= center + radius
    assert radius < Fraction(1, 2**100)


def test_factorial_coefficient_balls_contain_exact_rationals() -> None:
    for degree, ball in enumerate(
        separator_objective._EXP_COEFFICIENT_BALLS
    ):
        center = Fraction.from_float(ball.high) + Fraction.from_float(ball.low)
        radius = Fraction.from_float(ball.radius)
        exact = Fraction(1, math.factorial(degree))
        assert center - radius <= exact <= center + radius


@pytest.mark.parametrize('value', np.linspace(-0.7, 0.7, 29))
def test_polynomial_exp_and_expm1_balls_contain_decimal_oracle(
    value: float,
) -> None:
    argument = separator_numerics._TwofoldBall.point(float(value))
    exponential = separator_objective._exp_polynomial_ball(argument)
    exponential_minus_one = separator_objective._expm1_polynomial_ball(argument)
    with localcontext() as context:
        context.prec = 160
        exact = Decimal.from_float(float(value)).exp()
    for ball, oracle in (
        (exponential, exact),
        (exponential_minus_one, exact - Decimal(1)),
    ):
        lower, upper = ball.physical_bounds()
        assert Decimal.from_float(lower) <= oracle <= Decimal.from_float(upper)


def test_random_exact_dyadic_exp_polynomials_contain_decimal_oracles() -> None:
    rng = np.random.default_rng(948103)
    numerators = rng.integers(-700_000, 700_001, size=80)
    for numerator in numerators:
        value = math.ldexp(int(numerator), -20)
        argument = separator_numerics._TwofoldBall.point(value)
        exponential = separator_objective._exp_polynomial_ball(argument)
        exponential_minus_one = (
            separator_objective._expm1_polynomial_ball(argument)
        )
        with localcontext() as context:
            context.prec = 180
            exact = Decimal.from_float(value).exp()
        for ball, oracle in (
            (exponential, exact),
            (exponential_minus_one, exact - Decimal(1)),
        ):
            lower, upper = ball.physical_bounds()
            assert Decimal.from_float(lower) <= oracle <= Decimal.from_float(upper)


def test_k40_range_reduction_contains_independent_decimal_oracle() -> None:
    argument = separator_numerics._TwofoldBall.point(27.691532665133558)
    reduced = separator_numerics._ball_subtract(
        argument,
        separator_numerics._ball_integer_scale(
            separator_objective._LN2_BALL,
            40,
        ),
    )
    lower, upper = reduced.physical_bounds()
    with localcontext() as context:
        context.prec = 160
        exact = Decimal.from_float(argument.high) - Decimal(2).ln() * 40
    assert Decimal.from_float(lower) <= exact <= Decimal.from_float(upper)


@pytest.mark.parametrize('integer', (-1000, -40, -1, 0, 1, 40, 1000))
def test_signed_range_reduction_and_full_exp_contain_oracle(integer: int) -> None:
    argument_value = math.fsum((integer * math.log(2.0), 0.125))
    argument = separator_numerics._TwofoldBall.point(argument_value)
    reduced = separator_numerics._ball_subtract(
        argument,
        separator_numerics._ball_integer_scale(
            separator_objective._LN2_BALL,
            integer,
        ),
    )
    reduced_lower, reduced_upper = reduced.physical_bounds()
    with localcontext() as context:
        context.prec = 180
        exact_reduced = (
            Decimal.from_float(argument_value)
            - Decimal(2).ln() * integer
        )
        exact_exp = Decimal.from_float(argument_value).exp()
    assert Decimal.from_float(reduced_lower) <= exact_reduced
    assert exact_reduced <= Decimal.from_float(reduced_upper)
    exponential = separator_objective._certified_exp_ball(argument)
    exp_lower, exp_upper = exponential.physical_bounds()
    assert Decimal.from_float(exp_lower) <= exact_exp
    assert exact_exp <= Decimal.from_float(exp_upper)


@pytest.mark.parametrize('base', (-20.0, 0.0, 20.0, 700.0))
def test_balanced_exponential_difference_contains_cancellation_oracle(
    base: float,
) -> None:
    adjacent = math.nextafter(base, math.inf)
    left = separator_numerics._TwofoldBall.point(base)
    right = separator_numerics._TwofoldBall.point(adjacent)
    difference = separator_objective._certified_exp_difference(
        left,
        right,
        difference=separator_numerics._ball_subtract(right, left),
    )
    lower, upper = difference.physical_bounds()
    with localcontext() as context:
        context.prec = 180
        exact = (
            Decimal.from_float(adjacent).exp()
            - Decimal.from_float(base).exp()
        )
    assert Decimal.from_float(lower) <= exact <= Decimal.from_float(upper)
    if base == 0.0:
        # The exact positive result is below the normal-product EFT lane.
        # Ordinary arithmetic must decline the sign so the Decimal fallback
        # decides it; an unresolved enclosure is the fail-closed result.
        assert not difference.ball.resolved
    else:
        assert lower > 0.0
