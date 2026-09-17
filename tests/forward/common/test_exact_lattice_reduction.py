from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import subprocess
import sys

import numpy as np
import pytest

from pyvoro2._internal import exact_lattice as _exact_lattice
from pyvoro2._internal import periodic_images as _production_periodic_images
from pyvoro2._internal.exact_lattice import (
    DEFAULT_REDUCTION_LIMITS,
    ExactLatticeReductionLimits,
    ExactLatticeReductionInvariantError,
    ExactLatticeReductionResourceError,
    _reduction_cache_clear,
    _reduction_cache_info,
    _reduction_from_bits,
    exact_lll_reduce_3d,
    inverse_fraction_matrix,
)

from _lattice_reduction_workload import (
    FROZEN_RANDOM_UNIMODULAR_FIXTURES,
    FROZEN_WORKLOAD_FIXTURES,
    common_alignment_exponent,
    evaluate_proof_workload,
    exact_matrix,
    seeded_unimodular_pair,
)


IntMatrix = tuple[tuple[int, int, int], ...]
ExactMatrix = tuple[tuple[Fraction, Fraction, Fraction], ...]
IDENTITY: IntMatrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _integer_determinant(matrix: IntMatrix) -> int:
    signed_permutations = (
        ((0, 1, 2), 1),
        ((0, 2, 1), -1),
        ((1, 0, 2), -1),
        ((1, 2, 0), 1),
        ((2, 0, 1), 1),
        ((2, 1, 0), -1),
    )
    return sum(
        sign
        * matrix[0][permutation[0]]
        * matrix[1][permutation[1]]
        * matrix[2][permutation[2]]
        for permutation, sign in signed_permutations
    )


def _integer_product(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column]
                for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _integer_exact_product(
    left: IntMatrix,
    right: ExactMatrix,
) -> ExactMatrix:
    return tuple(
        tuple(
            sum(
                (left[row][index] * right[index][column]
                 for index in range(3)),
                Fraction(),
            )
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _coefficient_product(
    coefficients: tuple[int, int, int],
    matrix: IntMatrix,
) -> tuple[int, int, int]:
    return tuple(
        sum(coefficients[index] * matrix[index][column]
            for index in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _cartesian_product(
    coefficients: tuple[int, int, int],
    matrix: ExactMatrix,
) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(
        sum(
            (coefficients[index] * matrix[index][column]
             for index in range(3)),
            Fraction(),
        )
        for column in range(3)
    )  # type: ignore[return-value]


def _gram_schmidt(
    rows,
) -> tuple[
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, Fraction, Fraction],
]:
    stars: list[tuple[Fraction, Fraction, Fraction]] = []
    coefficients: list[list[Fraction]] = [[], [], []]
    squared: list[Fraction] = []
    exact_rows = tuple(
        tuple(Fraction(value) for value in row) for row in rows
    )
    for row_index, row in enumerate(exact_rows):
        star = list(row)
        for previous in range(row_index):
            coefficient = sum(
                row[column] * stars[previous][column]
                for column in range(3)
            ) / squared[previous]
            coefficients[row_index].append(coefficient)
            star = [
                star[column] - coefficient * stars[previous][column]
                for column in range(3)
            ]
        star_tuple = tuple(star)
        norm = sum(value * value for value in star_tuple)
        assert norm > 0
        stars.append(star_tuple)  # type: ignore[arg-type]
        squared.append(norm)
    return (
        tuple(stars),
        tuple(tuple(row) for row in coefficients),
        tuple(squared),  # type: ignore[arg-type]
    )


def _nearest_integer_ties_toward_zero(value: Fraction) -> int:
    numerator = value.numerator
    quotient, remainder = divmod(abs(numerator), value.denominator)
    doubled = 2 * remainder
    if doubled > value.denominator:
        quotient += 1
    return quotient if numerator >= 0 else -quotient


def _rank3_gram_certificate(rows) -> None:
    """Check rank-3 LLL through exact Gram-minor identities."""

    exact_rows = tuple(
        tuple(Fraction(value) for value in row) for row in rows
    )
    gram = tuple(
        tuple(
            sum(
                (left * right for left, right in zip(row, other)),
                Fraction(),
            )
            for other in exact_rows
        )
        for row in exact_rows
    )
    delta_1 = gram[0][0]
    delta_2 = gram[0][0] * gram[1][1] - gram[0][1] ** 2
    delta_3 = sum(
        sign
        * gram[0][permutation[0]]
        * gram[1][permutation[1]]
        * gram[2][permutation[2]]
        for permutation, sign in (
            ((0, 1, 2), 1),
            ((0, 2, 1), -1),
            ((1, 0, 2), -1),
            ((1, 2, 0), 1),
            ((2, 0, 1), 1),
            ((2, 1, 0), -1),
        )
    )
    assert delta_1 > 0
    assert delta_2 > 0
    assert delta_3 > 0

    assert abs(2 * gram[0][1]) <= delta_1
    assert abs(2 * gram[0][2]) <= delta_1
    mu_21_numerator = gram[0][0] * gram[1][2] - (
        gram[0][1] * gram[0][2]
    )
    assert abs(2 * mu_21_numerator) <= delta_2

    assert 4 * gram[1][1] >= 3 * gram[0][0]
    assert 4 * (
        gram[0][0] * gram[2][2] - gram[0][2] ** 2
    ) >= 3 * delta_2


def _prefix_gram_potential(rows) -> Fraction:
    exact_rows = tuple(
        tuple(Fraction(value) for value in row) for row in rows
    )
    first_norm = sum(
        (value * value for value in exact_rows[0]), Fraction()
    )
    second_norm = sum(
        (value * value for value in exact_rows[1]), Fraction()
    )
    cross = sum(
        (exact_rows[0][column] * exact_rows[1][column]
         for column in range(3)),
        Fraction(),
    )
    second_prefix_determinant = first_norm * second_norm - cross * cross
    return first_norm * second_prefix_determinant


def _independent_lll_oracle(
    source: ExactMatrix,
) -> tuple[ExactMatrix, IntMatrix, tuple[tuple[Fraction, Fraction], ...]]:
    """Run the fixed policy and trace descent without production helpers."""

    rows = [list(row) for row in source]
    transform = [list(row) for row in IDENTITY]
    trace = []
    row_index = 1
    steps = 0
    while row_index < 3:
        steps += 1
        assert steps < 10_000
        for previous in range(row_index - 1, -1, -1):
            _stars, coefficients, _squared = _gram_schmidt(
                tuple(tuple(row) for row in rows)  # type: ignore[arg-type]
            )
            nearest = _nearest_integer_ties_toward_zero(
                coefficients[row_index][previous]
            )
            if nearest:
                rows[row_index] = [
                    rows[row_index][column]
                    - nearest * rows[previous][column]
                    for column in range(3)
                ]
                transform[row_index] = [
                    transform[row_index][column]
                    - nearest * transform[previous][column]
                    for column in range(3)
                ]
        current = tuple(tuple(row) for row in rows)
        _stars, coefficients, squared = _gram_schmidt(
            current  # type: ignore[arg-type]
        )
        right = (
            Fraction(3, 4)
            - coefficients[row_index][row_index - 1] ** 2
        ) * squared[row_index - 1]
        if squared[row_index] < right:
            before = _prefix_gram_potential(
                current  # type: ignore[arg-type]
            )
            rows[row_index], rows[row_index - 1] = (
                rows[row_index - 1], rows[row_index]
            )
            transform[row_index], transform[row_index - 1] = (
                transform[row_index - 1], transform[row_index]
            )
            after = _prefix_gram_potential(
                tuple(tuple(row) for row in rows)  # type: ignore[arg-type]
            )
            trace.append((before, after))
            row_index = max(row_index - 1, 1)
        else:
            row_index += 1
    for index, row in enumerate(rows):
        if next(value for value in row if value) < 0:
            rows[index] = [-value for value in row]
            transform[index] = [-value for value in transform[index]]
    return (
        tuple(tuple(row) for row in rows),  # type: ignore[return-value]
        tuple(tuple(row) for row in transform),  # type: ignore[return-value]
        tuple(trace),
    )


def _assert_independently_certified(
    source_values: tuple[tuple[float, float, float], ...] | np.ndarray,
    result,
) -> None:
    source = exact_matrix(source_values)
    assert result.reduced_rows == _integer_exact_product(
        result.transform, source
    )
    assert _integer_determinant(result.transform) in (-1, 1)
    assert _integer_product(result.transform, result.inverse_transform) == IDENTITY
    assert _integer_product(result.inverse_transform, result.transform) == IDENTITY

    _rank3_gram_certificate(result.reduced_rows)
    for row in result.reduced_rows:
        first = next(value for value in row if value)
        assert first > 0
    assert result.certified is True


def test_rational_inverse_coerces_raw_integer_operands_before_division() -> None:
    inverse = inverse_fraction_matrix(((2, 0), (0, 2)))

    assert inverse == (
        (Fraction(1, 2), Fraction()),
        (Fraction(), Fraction(1, 2)),
    )
    assert all(isinstance(value, Fraction) for row in inverse for value in row)


@pytest.mark.parametrize(
    'basis',
    (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 5.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ),
)
def test_identity_handedness_and_signed_permutations_are_exactly_certified(
    basis,
) -> None:
    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    _assert_independently_certified(basis, result)


@pytest.mark.parametrize(
    'basis',
    (
        ((2.0, 0.0, 0.0), (1.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        ((0.0, 2.0, 0.0), (2.0, -1.0, 0.0), (0.0, 0.0, 2.0)),
        ((2.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, -1.0)),
    ),
)
def test_half_ties_and_lovasz_equality_make_no_gratuitous_change(basis) -> None:
    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    assert result.transform == IDENTITY
    assert result.reduced_rows == exact_matrix(basis)
    assert result.diagnostics.swaps == 0
    assert result.diagnostics.size_reductions == 0
    _assert_independently_certified(basis, result)


def test_three_vector_cancellation_does_not_stall_at_pairwise_half_ties() -> None:
    basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.25),
    )

    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    assert result.transform != IDENTITY
    assert result.diagnostics.swaps > 0
    _assert_independently_certified(basis, result)


@pytest.mark.parametrize(
    ('value', 'expected'),
    (
        (Fraction(1, 2), 0),
        (Fraction(-1, 2), 0),
        (Fraction(1, 2) - Fraction(1, 2**20), 0),
        (Fraction(1, 2) + Fraction(1, 2**20), 1),
        (Fraction(-1, 2) + Fraction(1, 2**20), 0),
        (Fraction(-1, 2) - Fraction(1, 2**20), -1),
    ),
)
def test_independent_rounding_uses_divmod_and_half_ties_toward_zero(
    value: Fraction,
    expected: int,
) -> None:
    assert _nearest_integer_ties_toward_zero(value) == expected


def test_independent_oracle_stays_exact_under_determinant_cancellation() -> None:
    magnitude = 2**27
    source = (
        (magnitude, magnitude - 1, 0),
        (magnitude + 1, magnitude, 0),
        (0, 0, 1),
    )

    rows, transform, _trace = _independent_lll_oracle(source)

    assert _integer_determinant(source) == 1
    assert _integer_determinant(transform) in (-1, 1)
    _rank3_gram_certificate(rows)


def test_strict_swaps_follow_an_independent_positive_integer_potential_trace(
) -> None:
    basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.25),
    )
    exponent = common_alignment_exponent(basis)
    aligned = tuple(
        tuple(
            value.numerator
            << (exponent - (value.denominator.bit_length() - 1))
            for value in row
        )
        for row in exact_matrix(basis)
    )

    oracle_rows, oracle_transform, oracle_trace = _independent_lll_oracle(
        aligned  # type: ignore[arg-type]
    )
    result, production_trace = _exact_lattice._trace_exact_lll_reduce_3d(
        np.asarray(basis, dtype=np.float64)
    )

    assert len(oracle_trace) == result.diagnostics.swaps == 4
    assert all(
        before.denominator == after.denominator == 1
        and 0 < after < before
        for before, after in oracle_trace
    )
    swaps = []
    for operation in production_trace:
        before = _prefix_gram_potential(operation.before)
        after = _prefix_gram_potential(operation.after)
        assert before.denominator == after.denominator == 1
        assert before > 0 and after > 0
        if operation.kind == 'size_reduction':
            assert after == before
        elif operation.kind == 'swap':
            assert 4 * after < 3 * before
            swaps.append(operation.row_index)
        elif operation.kind == 'sign_normalization':
            assert after == before
        else:
            raise AssertionError(f'unknown production operation {operation.kind!r}')
    assert len(swaps) == result.diagnostics.swaps
    assert any(right < left for left, right in zip(swaps, swaps[1:]))
    scale = Fraction(1, 1 << exponent)
    expected_rows = tuple(
        tuple(value * scale for value in row) for row in oracle_rows
    )
    assert result.transform == oracle_transform
    assert result.reduced_rows == expected_rows


def test_actual_production_trace_accepts_lovasz_equality_without_a_swap() -> None:
    basis = np.asarray(
        ((2.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, -1.0)),
        dtype=np.float64,
    )
    _stars, coefficients, squared = _gram_schmidt(basis)
    assert squared[1] - (
        Fraction(3, 4) - coefficients[1][0] ** 2
    ) * squared[0] == 0

    result, trace = _exact_lattice._trace_exact_lll_reduce_3d(basis)

    assert result.transform == IDENTITY
    assert not any(operation.kind == 'swap' for operation in trace)


def test_reduced_rows_never_round_through_binary64() -> None:
    epsilon = 2.0**-55
    basis = (
        (1.0, epsilon, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    expected = Fraction(1) - Fraction(1, 2**55)
    assert any(expected in row for row in result.reduced_rows)
    assert float(expected) == 1.0
    _assert_independently_certified(basis, result)


def test_wp2_determinant_cancellation_family_remains_exact() -> None:
    scale = float(2**27)
    right = (
        (scale, scale - 1.0, 0.0),
        (scale + 1.0, scale, 0.0),
        (0.0, 0.0, 1.0),
    )
    left = (right[1], right[0], right[2])

    for basis in (right, left):
        result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))
        _assert_independently_certified(basis, result)


def test_extreme_exact_scale_spread_is_reduced_without_float_authority() -> None:
    basis = (
        (2.0**-500, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 2.0**500),
    )

    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    _assert_independently_certified(basis, result)
    assert result.diagnostics.max_integer_bits >= 1001


def test_known_composed_unimodular_transforms_and_inverses() -> None:
    transforms: tuple[tuple[IntMatrix, IntMatrix], ...] = (
        (
            ((1, 0, 0), (7, 1, 0), (-5, 3, 1)),
            ((1, 0, 0), (-7, 1, 0), (26, -3, 1)),
        ),
        (
            ((0, 1, 0), (1, 11, 0), (4, -9, -1)),
            ((-11, 1, 0), (1, 0, 0), (-53, 4, -1)),
        ),
        (
            ((1, 0, 0), (-13, -1, 0), (8, 5, 1)),
            ((1, 0, 0), (-13, -1, 0), (57, 5, 1)),
        ),
    )
    for transform, known_inverse in transforms:
        assert _integer_product(transform, known_inverse) == IDENTITY
        assert _integer_product(known_inverse, transform) == IDENTITY
        basis = tuple(tuple(float(value) for value in row) for row in transform)

        first = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))
        _reduction_cache_clear()
        second = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

        assert first == second
        _assert_independently_certified(basis, first)


@pytest.mark.parametrize('seed', (0, 1, 17, 56))
def test_seeded_unimodular_compositions_are_exact_and_deterministic(
    seed: int,
) -> None:
    source_transform, known_inverse = seeded_unimodular_pair(seed)
    assert _integer_product(source_transform, known_inverse) == IDENTITY
    assert _integer_product(known_inverse, source_transform) == IDENTITY
    assert all(
        int(float(value)) == value
        for row in source_transform
        for value in row
    )
    basis = tuple(
        tuple(float(value) for value in row) for row in source_transform
    )

    _reduction_cache_clear()
    first = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))
    _reduction_cache_clear()
    second = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    assert first == second
    user_shift = (2**80 + seed, -3, 5)
    reduced_shift = first.map_user_to_reduced(user_shift)
    assert first.map_reduced_to_user(reduced_shift) == user_shift
    assert _cartesian_product(reduced_shift, first.reduced_rows) == (
        _cartesian_product(user_shift, exact_matrix(basis))
    )
    _assert_independently_certified(basis, first)


def test_256_seed_policy_oracle_and_diagnostic_maxima() -> None:
    maxima = {
        'steps': 0,
        'work': 0,
        'integer_bits': 0,
        'rational_bits': 0,
        'transform_bits': 0,
        'inverse_transform_bits': 0,
    }
    for seed in range(256):
        source_transform, known_inverse = seeded_unimodular_pair(seed)
        assert _integer_product(source_transform, known_inverse) == IDENTITY
        assert _integer_product(known_inverse, source_transform) == IDENTITY
        assert all(
            int(float(value)) == value
            for row in source_transform
            for value in row
        )
        basis = tuple(
            tuple(float(value) for value in row)
            for row in source_transform
        )

        oracle_rows, oracle_transform, _trace = _independent_lll_oracle(
            exact_matrix(basis)
        )
        result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

        assert result.transform == oracle_transform
        assert result.reduced_rows == oracle_rows
        _assert_independently_certified(basis, result)
        diagnostics = result.diagnostics
        maxima['steps'] = max(maxima['steps'], diagnostics.steps)
        maxima['work'] = max(maxima['work'], diagnostics.work)
        maxima['integer_bits'] = max(
            maxima['integer_bits'], diagnostics.max_integer_bits
        )
        maxima['rational_bits'] = max(
            maxima['rational_bits'], diagnostics.max_rational_bits
        )
        maxima['transform_bits'] = max(
            maxima['transform_bits'], diagnostics.max_transform_bits
        )
        maxima['inverse_transform_bits'] = max(
            maxima['inverse_transform_bits'],
            diagnostics.max_inverse_transform_bits,
        )

    assert maxima == {
        'steps': 22,
        'work': 3_270,
        'integer_bits': 32,
        'rational_bits': 73,
        'transform_bits': 23,
        'inverse_transform_bits': 18,
    }


def test_large_reduced_coefficients_map_to_small_user_shift_before_int64() -> None:
    shear = 2**80
    basis = (
        (1.0, 0.0, 0.0),
        (float(shear), 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))
    reduced_shift = (shear, 1, 0)
    user_shift = result.map_reduced_to_user(reduced_shift)

    assert result.transform == ((1, 0, 0), (-shear, 1, 0), (0, 0, 1))
    assert user_shift == (0, 1, 0)
    assert result.map_user_to_reduced(user_shift) == reduced_shift
    assert _cartesian_product(reduced_shift, result.reduced_rows) == (
        _cartesian_product(user_shift, exact_matrix(basis))
    )
    _assert_independently_certified(basis, result)


def test_resource_limits_fail_structurally_without_partial_result() -> None:
    basis = np.asarray(
        ((1.0, 0.0, 0.0), (float(2**80), 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    cases = (
        replace(DEFAULT_REDUCTION_LIMITS, max_work=1),
        replace(DEFAULT_REDUCTION_LIMITS, max_steps=1),
        replace(DEFAULT_REDUCTION_LIMITS, max_integer_bits=16),
        replace(DEFAULT_REDUCTION_LIMITS, max_transform_bits=16),
        replace(
            DEFAULT_REDUCTION_LIMITS,
            max_transform_bits=256,
            max_inverse_transform_bits=16,
        ),
    )
    expected_resources = (
        'work',
        'steps',
        'integer_bits',
        'transform_bits',
        'inverse_transform_bits',
    )
    for limits, resource in zip(cases, expected_resources):
        with pytest.raises(ExactLatticeReductionResourceError) as captured:
            exact_lll_reduce_3d(basis, limits=limits)
        error = captured.value
        assert error.resource == resource
        assert error.observed > error.configured_limit
        assert error.method == 'exact-lll-rank3'
        assert error.stage
        assert error.source_summary['lattice_bits']


def test_rational_intermediate_limit_is_distinct_from_integer_limit() -> None:
    limits = replace(
        DEFAULT_REDUCTION_LIMITS,
        max_integer_bits=4096,
        max_rational_bits=8,
    )
    basis = np.asarray(
        ((7.0, 1.0, 0.0), (3.0, 5.0, 1.0), (2.0, -1.0, 6.0)),
        dtype=np.float64,
    )

    with pytest.raises(ExactLatticeReductionResourceError) as captured:
        exact_lll_reduce_3d(basis, limits=limits)

    assert captured.value.resource == 'rational_bits'


def test_integer_determinant_intermediates_are_resource_accounted() -> None:
    scale = float(2**27)
    basis = np.asarray(
        (
            (scale, scale - 1.0, 0.0),
            (scale + 1.0, scale, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    limits = replace(
        DEFAULT_REDUCTION_LIMITS,
        max_integer_bits=40,
        max_rational_bits=4096,
    )

    with pytest.raises(ExactLatticeReductionResourceError) as captured:
        exact_lll_reduce_3d(basis, limits=limits)

    assert captured.value.resource == 'integer_bits'
    assert captured.value.stage == 'source_determinant'


def test_final_certification_is_covered_by_the_work_limit() -> None:
    limits = replace(DEFAULT_REDUCTION_LIMITS, max_work=530)

    with pytest.raises(ExactLatticeReductionResourceError) as captured:
        exact_lll_reduce_3d(np.eye(3), limits=limits)

    assert captured.value.resource == 'work'
    assert captured.value.stage == 'certificate_gram'


def test_final_lovasz_rational_growth_is_limited_and_cache_safe() -> None:
    basis = np.eye(3) * 2.0**-100
    restricted = replace(
        DEFAULT_REDUCTION_LIMITS,
        max_rational_bits=201,
    )
    sufficient = replace(
        DEFAULT_REDUCTION_LIMITS,
        max_rational_bits=203,
    )
    _reduction_cache_clear()

    for _ in range(2):
        with pytest.raises(ExactLatticeReductionResourceError) as captured:
            exact_lll_reduce_3d(basis, limits=restricted)
        error = captured.value
        assert error.stage == 'certificate_lovasz'
        assert error.resource == 'rational_bits'
        assert error.observed == 203
        assert error.configured_limit == 201
    assert _reduction_cache_info().currsize == 0

    result = exact_lll_reduce_3d(basis, limits=sufficient)
    assert result.certified
    assert result.diagnostics.max_rational_bits == 203
    assert exact_lll_reduce_3d(basis, limits=sufficient) is result

    with pytest.raises(ExactLatticeReductionResourceError):
        exact_lll_reduce_3d(basis, limits=restricted)
    assert _reduction_cache_info().currsize == 1


def test_certificate_invariant_checks_run_under_python_optimized_mode() -> None:
    script = """
from fractions import Fraction
from pyvoro2._internal import exact_lattice

zero = Fraction()
one = Fraction(1)
identity = ((one, zero, zero), (zero, one, zero), (zero, zero, one))
bad = ((Fraction(2), zero, zero), identity[1], identity[2])
transform = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
monitor = exact_lattice._ReductionMonitor(
    limits=exact_lattice.DEFAULT_REDUCTION_LIMITS,
    source_summary={},
)
try:
    exact_lattice._certify_reduction(
        source=identity,
        reduced=bad,
        transform=transform,
        inverse_transform=transform,
        monitor=monitor,
    )
except exact_lattice.ExactLatticeReductionInvariantError:
    print('certificate-rejected')
else:
    raise RuntimeError('optimized mode bypassed certification')
"""

    completed = subprocess.run(
        [sys.executable, '-O', '-c', script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == 'certificate-rejected'


def test_limit_validation_rejects_boolean_and_nonpositive_values() -> None:
    with pytest.raises(ValueError, match='max_work'):
        ExactLatticeReductionLimits(max_work=True)
    with pytest.raises(ValueError, match='max_steps'):
        ExactLatticeReductionLimits(max_steps=0)


def test_cache_is_bounded_bit_ordered_and_policy_aware() -> None:
    _reduction_cache_clear()
    first = np.eye(3)
    exact_lll_reduce_3d(first)
    after_first = _reduction_cache_info()
    exact_lll_reduce_3d(first.copy())
    after_same = _reduction_cache_info()
    changed = first.copy()
    changed[0, 0] = np.nextafter(1.0, np.inf)
    exact_lll_reduce_3d(changed)
    after_bits = _reduction_cache_info()
    reordered = first[[1, 0, 2]]
    exact_lll_reduce_3d(reordered)
    after_order = _reduction_cache_info()
    exact_lll_reduce_3d(
        first,
        limits=replace(DEFAULT_REDUCTION_LIMITS, max_work=999_999),
    )
    after_policy = _reduction_cache_info()

    assert after_first.misses == 1
    assert after_same.hits == 1
    assert after_bits.misses == 2
    assert after_order.misses == 3
    assert after_policy.misses == 4
    assert after_policy.maxsize == 128

    for index in range(130):
        matrix = np.diag([1.0 + index * 2.0**-40, 1.0, 1.0])
        exact_lll_reduce_3d(matrix)
    assert _reduction_cache_info().currsize == 128


def test_changed_policy_cannot_reuse_a_current_policy_cache_entry() -> None:
    _reduction_cache_clear()
    matrix = np.eye(3, dtype=np.float64)
    exact_lll_reduce_3d(matrix)
    source_bits = tuple(int(value) for value in matrix.view(np.uint64).flat)

    with pytest.raises(ExactLatticeReductionInvariantError, match='policy'):
        _reduction_from_bits(
            source_bits,
            'different-private-policy',
            DEFAULT_REDUCTION_LIMITS,
        )

    info = _reduction_cache_info()
    assert info.hits == 0
    assert info.misses == 2
    assert info.currsize == 1


def test_cached_result_and_diagnostics_are_immutable() -> None:
    result = exact_lll_reduce_3d(np.eye(3))

    with pytest.raises(FrozenInstanceError):
        result.transform = IDENTITY
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.work = 0


def test_resource_failures_are_not_cached_as_successes() -> None:
    _reduction_cache_clear()
    limits = replace(DEFAULT_REDUCTION_LIMITS, max_work=1)
    for _ in range(2):
        with pytest.raises(ExactLatticeReductionResourceError):
            exact_lll_reduce_3d(np.eye(3), limits=limits)
    info = _reduction_cache_info()
    assert info.hits == 0
    assert info.misses == 2
    assert info.currsize == 0


def test_frozen_source_workload_baselines_do_not_drift() -> None:
    expected = {
        'thin-3e-4': (106_726_048, 10_692_900),
        'thin-1e-3': (9_618_496, 962_360),
        'cubic-shear-2p8': (1_816_657_920, 1_760_452_600),
        'equivalent-composed-a': (86_351_200, 66_311_622),
        'equivalent-composed-b': (257_218_200, 75_021_200),
        'equivalent-composed-c': (3_231_963, 1_328_910),
        'intrinsic-anisotropy': (728_214_795, 331_967_811),
        'r5-sc-001': (363_765, 355_989),
    }
    for fixture in FROZEN_WORKLOAD_FIXTURES:
        if fixture.name not in expected:
            continue
        exponent = common_alignment_exponent(
            fixture.basis, fixture.pi, fixture.pj
        )
        actual = tuple(
            evaluate_proof_workload(
                fixture.basis,
                pi=fixture.pi,
                pj=fixture.pj,
                image_search=seed,
                common_exponent=exponent,
            ).box_count
            for seed in (0, 1)
        )
        assert actual == expected[fixture.name]


def _before_after(fixture, seed: int):
    result = exact_lll_reduce_3d(np.asarray(fixture.basis, dtype=np.float64))
    exponent = common_alignment_exponent(
        fixture.basis, fixture.pi, fixture.pj
    )
    before = evaluate_proof_workload(
        fixture.basis,
        pi=fixture.pi,
        pj=fixture.pj,
        image_search=seed,
        common_exponent=exponent,
    )
    seeded_after = evaluate_proof_workload(
        result.reduced_rows,
        pi=fixture.pi,
        pj=fixture.pj,
        image_search=seed,
        common_exponent=exponent,
    )
    fixed = min(before.incumbent_squared, seeded_after.incumbent_squared)
    before_fixed = evaluate_proof_workload(
        fixture.basis,
        pi=fixture.pi,
        pj=fixture.pj,
        image_search=seed,
        common_exponent=exponent,
        fixed_incumbent_squared=fixed,
    )
    after_fixed = evaluate_proof_workload(
        result.reduced_rows,
        pi=fixture.pi,
        pj=fixture.pj,
        image_search=seed,
        common_exponent=exponent,
        fixed_incumbent_squared=fixed,
    )
    return result, before, seeded_after, before_fixed, after_fixed


@pytest.mark.parametrize('seed', (0, 1))
def test_repository_thin_regressions_meet_proof_box_gates(seed: int) -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'repository-regression']
    assert len(fixtures) == 2
    for fixture in fixtures:
        _result, before, after, before_shared, after_shared = _before_after(
            fixture, seed
        )
        assert after.box_count <= 4096
        assert before.box_count >= 100 * after.box_count
        assert after.box_count <= 1_000_000
        assert before_shared.fixed_box_count >= (
            100 * after_shared.fixed_box_count
        )


def test_six_row_thin_batch_is_below_the_existing_budget() -> None:
    fixture = next(
        fixture for fixture in FROZEN_WORKLOAD_FIXTURES
        if fixture.name == 'thin-1e-3'
    )
    for seed in (0, 1):
        _result, _before, after, _before_shared, _after_shared = (
            _before_after(fixture, seed)
        )
        assert after.box_count == 980
        assert 6 * after.box_count == 5_880
        assert 6 * after.box_count <= 5_000_000


@pytest.mark.parametrize('seed', (0, 1))
def test_exact_cubic_shear_work_is_bounded_independently_of_magnitude(
    seed: int,
) -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'exact-large-shear']
    counts = []
    for fixture in fixtures:
        _result, before, after, before_shared, after_shared = _before_after(
            fixture, seed
        )
        assert before.box_count > after.box_count
        assert after.box_count <= 64
        assert before_shared.fixed_box_count > after_shared.fixed_box_count
        counts.append(after.box_count)
    assert len(set(counts)) == 1


def test_equivalent_well_conditioned_cohort_meets_buffered_gates() -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'equivalent-well-conditioned']
    over_budget = []
    for fixture in fixtures:
        for seed in (0, 1):
            _result, before, after, before_shared, after_shared = _before_after(
                fixture, seed
            )
            assert after.box_count <= 256
            assert after.box_count <= 1_000_000
            if before.box_count > 1_000_000:
                over_budget.append((fixture.name, seed))
                assert before.box_count >= 1000 * after.box_count
            assert after_shared.fixed_box_count <= 256
            assert before_shared.fixed_box_count > after_shared.fixed_box_count
    assert over_budget == [
        ('equivalent-composed-a', 0),
        ('equivalent-composed-a', 1),
        ('equivalent-composed-b', 0),
        ('equivalent-composed-b', 1),
        ('equivalent-composed-c', 0),
        ('equivalent-composed-c', 1),
    ]


def test_frozen_seeded_random_unimodular_workload_cohort() -> None:
    expected = {
        'random-unimodular-seed-0': (
            ((64_537_200, 8), (66_348, 8)),
            (1_580, 8),
        ),
        'random-unimodular-seed-1': (
            ((6_725_203_976_616_300, 8), (20_685_485_506_560, 8)),
            (6_509_580, 8),
        ),
        'random-unimodular-seed-56': (
            ((13_000, 1), (6_656, 1)),
            (13, 1),
        ),
    }
    over_budget = []
    for fixture in FROZEN_RANDOM_UNIMODULAR_FIXTURES:
        seeded_counts = []
        fixed_counts = []
        for image_search in (0, 1):
            _result, seeded_before, seeded_after, before, after = (
                _before_after(fixture, image_search)
            )
            seeded_counts.append(
                (seeded_before.box_count, seeded_after.box_count)
            )
            fixed_counts.append((before.fixed_box_count, after.fixed_box_count))
            assert seeded_after.box_count <= 256
            assert seeded_after.box_count <= 1_000_000
            if seeded_before.box_count > 1_000_000:
                over_budget.append((fixture.name, image_search))
                assert seeded_before.box_count >= 1000 * seeded_after.box_count
            assert after.fixed_box_count <= 256
        expected_seeded, expected_fixed = expected[fixture.name]
        assert tuple(seeded_counts) == expected_seeded
        assert fixed_counts == [expected_fixed, expected_fixed]
    assert over_budget == [
        ('random-unimodular-seed-0', 0),
        ('random-unimodular-seed-1', 0),
        ('random-unimodular-seed-1', 1),
    ]


def test_r5_sc_001_inverse_bound_product_improves_by_at_least_100x() -> None:
    fixture = next(
        fixture for fixture in FROZEN_WORKLOAD_FIXTURES
        if fixture.name == 'r5-sc-001'
    )
    _result, _seeded_before, _seeded_after, before, after = _before_after(
        fixture, seed=1
    )

    assert before.inverse_bound_product >= 100 * after.inverse_bound_product
    assert before.bucket_bins == (3, 132_997, 598_610_259)
    assert after.bucket_bins == (5_137, 210_991, 598_610_259)


def test_intrinsic_anisotropy_is_reported_separately_without_false_gate() -> None:
    fixture = next(
        fixture for fixture in FROZEN_WORKLOAD_FIXTURES
        if fixture.cohort == 'intrinsically-anisotropic'
    )
    result, _seeded_before, _seeded_after, before, after = _before_after(
        fixture, seed=1
    )

    assert result.certified
    assert before.fixed_box_count > 0
    assert after.fixed_box_count > 0
    assert max(after.inverse_column_l1) >= 2**16


def test_workload_default_alignment_includes_uncancelled_endpoints() -> None:
    basis = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ordinary = evaluate_proof_workload(
        basis,
        pi=(0.0, 0.0, 0.0),
        pj=(0.25, 0.0, 0.0),
        image_search=0,
    )
    cancelled = evaluate_proof_workload(
        basis,
        pi=(0.1, 0.0, 0.0),
        pj=(0.1, 0.0, 0.0),
        image_search=0,
    )

    assert ordinary.exponent == common_alignment_exponent(
        basis, (0.0, 0.0, 0.0), (0.25, 0.0, 0.0)
    )
    assert cancelled.exponent == common_alignment_exponent(
        basis, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0)
    )
    assert cancelled.exponent > 0


@pytest.mark.parametrize('image_search', (-1, 2, 8, True, 1.0))
def test_workload_evaluator_rejects_unsupported_seed_domains(
    image_search,
) -> None:
    basis = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match='image_search'):
        evaluate_proof_workload(
            basis,
            pi=(0.0, 0.0, 0.0),
            pj=(0.25, 0.0, 0.0),
            image_search=image_search,
        )


@pytest.mark.parametrize('image_search', (0, 1))
def test_workload_formulas_match_bounded_production_preparation(
    image_search: int,
) -> None:
    basis = np.asarray(
        ((1.0, 0.0, 0.0), (0.25, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    pi = np.asarray((0.1, -0.2, 0.3), dtype=np.float64)
    pj = np.asarray((0.35, 0.15, -0.05), dtype=np.float64)
    workload = evaluate_proof_workload(
        basis,
        pi=pi,
        pj=pj,
        image_search=image_search,
    )

    prepared = _production_periodic_images._prepare_basis(
        *_production_periodic_images._basis_key(
            basis, (True, True, True)
        )
    )
    displacement, lattice, exponent = (
        _production_periodic_images._aligned_integer_geometry(
            pi, pj, prepared
        )
    )
    plan = _production_periodic_images._prepare_triclinic_row(
        displacement,
        lattice,
        exponent,
        pair_index=0,
        basis=prepared,
        orientation=1,
        image_search=image_search,
    )
    layout = _production_periodic_images._exact_triclinic_bucket_layout(
        np.asarray((pi, pj)),
        origin=np.zeros(3),
        lattice_vectors=basis,
        radius=1e-5,
    )

    assert workload.exponent == plan.denominator_exponent
    assert workload.interval_widths == tuple(
        upper - lower + 1
        for lower, upper in zip(plan.lower, plan.upper)
    )
    assert workload.box_count == plan.candidate_count
    assert workload.seed_count == plan.seed_count
    assert workload.bucket_bins == layout.bins
    assert tuple(
        Fraction.from_float(1e-5) * value
        for value in workload.inverse_column_l1
    ) == layout.coefficient_bounds
