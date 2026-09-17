from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

import pyvoro2
from pyvoro2.diagnostics import _domain_volume


def _f(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _det3(rows: tuple[tuple[float, float, float], ...]) -> Fraction:
    a, b, c = tuple(tuple(_f(value) for value in row) for row in rows)
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _solve_row_exact(
    point: tuple[float, float, float],
    origin: tuple[float, float, float],
    rows: tuple[tuple[float, float, float], ...],
) -> tuple[Fraction, Fraction, Fraction]:
    matrix = [[_f(value) for value in row] for row in rows]
    rhs = [_f(point[index]) - _f(origin[index]) for index in range(3)]
    augmented = [
        [matrix[column][row] for column in range(3)] + [rhs[row]]
        for row in range(3)
    ]
    for column in range(3):
        pivot = next(row for row in range(column, 3)
                     if augmented[row][column])
        augmented[column], augmented[pivot] = (
            augmented[pivot], augmented[column]
        )
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return tuple(row[-1] for row in augmented)  # type: ignore[return-value]


def _reconstruct_exact(
    fractional: tuple[float, float, float],
    origin: tuple[float, float, float],
    rows: tuple[tuple[float, float, float], ...],
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(
        _f(origin[column])
        + sum(
            _f(fractional[row]) * _f(rows[row][column])
            for row in range(3)
        )
        for column in range(3)
    )  # type: ignore[return-value]


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def test_exact_nonsingularity_accepts_both_handedness_and_cancellation() -> None:
    scale = float(2**27)
    right = (
        (scale, scale - 1.0, 0.0),
        (scale + 1.0, scale, 0.0),
        (0.0, 0.0, 1.0),
    )
    left = (right[1], right[0], right[2])
    assert _det3(right) == 1
    assert _det3(left) == -1

    right_cell = pyvoro2.PeriodicCell(right)
    left_cell = pyvoro2.PeriodicCell(left)
    assert right_cell.vectors == right
    assert left_cell.vectors == left
    assert _domain_volume(right_cell) == 1.0
    assert _domain_volume(left_cell) == 1.0


def test_exact_singularity_is_the_only_constructor_rejection() -> None:
    with pytest.raises(ValueError, match='exactly singular'):
        pyvoro2.PeriodicCell(
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        )

    tiny = float.fromhex('0x0.0000000000001p-1022')
    pyvoro2.PeriodicCell(((tiny, 0.0, 0.0),
                          (0.0, tiny, 0.0),
                          (0.0, 0.0, tiny)))
    pyvoro2.PeriodicCell(((1e308, 0.0, 0.0),
                          (0.0, 1e308, 0.0),
                          (0.0, 0.0, 1e308)))


@pytest.mark.parametrize(
    'scale',
    [float.fromhex('0x0.0000000000001p-1022'), 1e308],
)
def test_diagnostic_volume_rejects_an_unrepresentable_float_view(scale) -> None:
    cell = pyvoro2.PeriodicCell(
        ((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, scale))
    )
    with pytest.raises(ValueError, match='positive finite binary64 view'):
        _domain_volume(cell)


def test_user_coordinate_methods_match_independent_exact_oracles() -> None:
    rows = (
        (3.0, 0.5, -0.25),
        (-1.0, 2.0, 0.75),
        (0.5, -0.25, -4.0),
    )
    origin = (0.125, -0.75, 2.0)
    point = (math.nextafter(1.0, 2.0), -3.25, 0.1)
    fractional = (1.0 / 3.0, math.nextafter(-1.0, -2.0), 0.125)
    cell = pyvoro2.PeriodicCell(rows, origin=origin)

    exact_solved = _solve_row_exact(point, origin, rows)
    solved = cell.cart_to_fractional([point])
    assert solved.shape == (1, 3)
    assert solved[0].tolist() == [float(value) for value in exact_solved]

    exact_cart = _reconstruct_exact(fractional, origin, rows)
    cart = cell.fractional_to_cart([fractional])
    assert cart.shape == (1, 3)
    assert cart[0].tolist() == [float(value) for value in exact_cart]


def test_randomized_wrap_cart_matches_fraction_oracle() -> None:
    rows = (
        (1.5, -0.25, 0.125),
        (0.5, 2.0, -0.375),
        (-0.25, 0.75, -1.25),
    )
    origin = (0.125, -0.5, 0.75)
    cell = pyvoro2.PeriodicCell(rows, origin=origin)
    rng = np.random.default_rng(5418)
    integers = rng.integers(-1000, 1001, size=(32, 3))
    exponents = rng.integers(-8, 9, size=(32, 3))
    points = np.ldexp(
        integers.astype(np.float64),
        np.asarray(exponents, dtype=np.intc),
    )

    wrapped, shifts = cell.wrap_cart(points, return_shifts=True)
    for index, point in enumerate(points):
        exact_fractional = _solve_row_exact(tuple(point), origin, rows)
        expected_shift = tuple(
            value.numerator // value.denominator
            for value in exact_fractional
        )
        expected_wrapped = tuple(
            _f(point[column])
            - sum(expected_shift[row] * _f(rows[row][column])
                  for row in range(3))
            for column in range(3)
        )
        assert tuple(shifts[index]) == expected_shift
        assert wrapped[index].tolist() == [
            float(value) for value in expected_wrapped
        ]


def test_exact_floor_is_not_chosen_from_a_rounded_fractional_view() -> None:
    delta = 2.0**-55
    cell = pyvoro2.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        origin=(delta, 0.0, 0.0),
    )
    points = np.array([[1.0, 0.0, 0.0]])

    fractional = cell.cart_to_fractional(points)
    assert fractional[0, 0] == 1.0

    wrapped, shifts = cell.wrap_cart(points, return_shifts=True)
    assert shifts.tolist() == [[0, 0, 0]]
    assert wrapped.tolist() == points.tolist()

    wrapped_again, shifts_again = cell.wrap_cart(
        wrapped, return_shifts=True
    )
    assert shifts_again.tolist() == [[0, 0, 0]]
    assert wrapped_again.tolist() == wrapped.tolist()


@pytest.mark.parametrize(
    'below_zero',
    [
        -2.0**-54,
        -2.0**-55,
        -float.fromhex('0x0.0000000000001p-1022'),
    ],
)
def test_wrap_fractional_preserves_exact_upper_endpoint_remainders(
    below_zero,
) -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 0, 1, 0, 0, 1)
    source = np.array([[below_zero, -0.0, math.nextafter(1.0, 0.0)]])

    wrapped, shifts = cell.wrap_fractional(source, return_shifts=True)
    assert wrapped.tolist() == [[1.0, 0.0, math.nextafter(1.0, 0.0)]]
    assert shifts.tolist() == [[-1, 0, 0]]

    # The rounded view 1.0 is not repaired. Re-wrapping that new binary64
    # source consequently applies a different exact floor.
    wrapped_again, shifts_again = cell.wrap_fractional(
        wrapped, return_shifts=True
    )
    assert wrapped_again[0, 0] == 0.0
    assert shifts_again[0, 0] == 1


def test_wrap_shift_range_is_checked_only_when_materialized() -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 0, 1, 0, 0, 1)
    source = np.array([[float(2**63), 0.25, 0.5]])

    assert cell.wrap_fractional(source).tolist() == [[0.0, 0.25, 0.5]]
    with pytest.raises(ValueError, match='signed int64'):
        cell.wrap_fractional(source, return_shifts=True)

    tiny = float.fromhex('0x0.0000000000001p-1022')
    tiny_cell = pyvoro2.PeriodicCell(
        ((tiny, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    assert tiny_cell.wrap_cart([[1.0, 0.25, 0.5]]).tolist() == [
        [0.0, 0.25, 0.5]
    ]
    with pytest.raises(ValueError, match='signed int64'):
        tiny_cell.wrap_cart([[1.0, 0.25, 0.5]], return_shifts=True)


def test_wrap_shift_int64_boundaries_are_exact() -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 0, 1, 0, 0, 1)
    largest_binary64_below_max = float(2**63 - 1024)
    source = np.array([
        [-float(2**63), 0.0, 0.0],
        [largest_binary64_below_max, 0.0, 0.0],
    ])

    wrapped, shifts = cell.wrap_fractional(source, return_shifts=True)
    assert wrapped.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert shifts.dtype == np.int64
    assert int(shifts[0, 0]) == -(2**63)
    assert int(shifts[1, 0]) == 2**63 - 1024

    with pytest.raises(ValueError, match='signed int64'):
        cell.wrap_fractional([[float(2**63), 0.0, 0.0]], return_shifts=True)


def test_cart_wrap_view_error_preserves_the_exact_reconstruction_envelope() -> None:
    rows = (
        (1.1, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    origin = (0.0, 0.0, 0.0)
    point = (-0.060143602597438485, float(2**52), 0.1)
    cell = pyvoro2.PeriodicCell(rows, origin=origin)

    wrapped, shifts = cell.wrap_cart([point], return_shifts=True)
    exact_fractional = _solve_row_exact(point, origin, rows)
    exact_shifts = tuple(_floor(value) for value in exact_fractional)
    exact_wrapped = tuple(
        _f(point[column])
        - sum(exact_shifts[row] * _f(rows[row][column]) for row in range(3))
        for column in range(3)
    )

    assert tuple(shifts[0]) == exact_shifts
    view_error = tuple(
        _f(wrapped[0, column]) - exact_wrapped[column]
        for column in range(3)
    )
    assert any(error != 0 for error in view_error)
    for column in range(3):
        reconstructed_view = (
            _f(wrapped[0, column])
            + sum(int(shifts[0, row]) * _f(rows[row][column])
                  for row in range(3))
        )
        assert _f(point[column]) - reconstructed_view == -view_error[column]


def test_wrap_cart_is_covariant_under_signed_row_permutations() -> None:
    rows = np.array(
        ((1.5, -0.25, 0.125), (0.5, 2.0, -0.375),
         (-0.25, 0.75, -1.25)),
        dtype=np.float64,
    )
    origin = np.array((0.125, -0.5, 0.75))
    point = np.array((2.75, -1.125, 0.3125))
    reference = pyvoro2.PeriodicCell(
        tuple(map(tuple, rows)), origin=tuple(origin)
    ).wrap_cart([point])[0]

    permutation = np.array((2, 0, 1))
    signs = np.array((-1, 1, -1))
    permuted = rows[permutation]
    signed = signs[:, None] * permuted
    signed_origin = origin + permuted[signs < 0].sum(axis=0)
    actual = pyvoro2.PeriodicCell(
        tuple(map(tuple, signed)), origin=tuple(signed_origin)
    ).wrap_cart([point])[0]

    np.testing.assert_array_equal(actual, reference)


def test_user_coordinate_outputs_are_owned_and_empty_batches_work() -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 0, 2, 0, 0, 3)
    source = np.empty((0, 3), dtype=np.float64)
    for operation in (
        cell.cart_to_fractional,
        cell.fractional_to_cart,
        cell.wrap_fractional,
        cell.wrap_cart,
    ):
        result = operation(source)
        assert result.shape == (0, 3)
        assert result.dtype == np.float64
        assert result.flags.owndata


def test_unavailable_binary64_views_fail_at_the_operation_boundary() -> None:
    tiny = float.fromhex('0x0.0000000000001p-1022')
    cell = pyvoro2.PeriodicCell(
        ((tiny, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    with pytest.raises(ValueError, match='finite binary64'):
        cell.cart_to_fractional([[1.0, 0.0, 0.0]])

    huge = pyvoro2.PeriodicCell.from_params(1e308, 0, 1, 0, 0, 1)
    with pytest.raises(ValueError, match='finite binary64'):
        huge.fractional_to_cart([[2.0, 0.0, 0.0]])
