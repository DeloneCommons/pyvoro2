from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import numpy as np
import pytest

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
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _integer_inverse(matrix: IntMatrix) -> IntMatrix:
    a, b, c = matrix
    determinant = _integer_determinant(matrix)
    assert abs(determinant) == 1
    adjugate = (
        (
            b[1] * c[2] - b[2] * c[1],
            a[2] * c[1] - a[1] * c[2],
            a[1] * b[2] - a[2] * b[1],
        ),
        (
            b[2] * c[0] - b[0] * c[2],
            a[0] * c[2] - a[2] * c[0],
            a[2] * b[0] - a[0] * b[2],
        ),
        (
            b[0] * c[1] - b[1] * c[0],
            a[1] * c[0] - a[0] * c[1],
            a[0] * b[1] - a[1] * b[0],
        ),
    )
    return tuple(
        tuple(value // determinant for value in row) for row in adjugate
    )  # type: ignore[return-value]


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
    rows: ExactMatrix,
) -> tuple[
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, Fraction, Fraction],
]:
    stars: list[tuple[Fraction, Fraction, Fraction]] = []
    coefficients: list[list[Fraction]] = [[], [], []]
    squared: list[Fraction] = []
    for row_index, row in enumerate(rows):
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
    lower = math.floor(value)
    remainder = value - lower
    if remainder < Fraction(1, 2):
        return lower
    if remainder > Fraction(1, 2):
        return lower + 1
    return lower if value > 0 else lower + 1


def _prefix_gram_potential(rows: ExactMatrix) -> Fraction:
    first_norm = sum(value * value for value in rows[0])
    second_norm = sum(value * value for value in rows[1])
    cross = sum(
        rows[0][column] * rows[1][column] for column in range(3)
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
    assert result.inverse_transform == _integer_inverse(result.transform)
    assert _integer_product(result.transform, result.inverse_transform) == IDENTITY
    assert _integer_product(result.inverse_transform, result.transform) == IDENTITY

    _stars, coefficients, squared = _gram_schmidt(result.reduced_rows)
    for row in range(1, 3):
        for column in range(row):
            assert abs(coefficients[row][column]) <= Fraction(1, 2)
    for row in range(1, 3):
        assert squared[row] >= (
            Fraction(3, 4) - coefficients[row][row - 1] ** 2
        ) * squared[row - 1]
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

    oracle_rows, oracle_transform, trace = _independent_lll_oracle(
        aligned  # type: ignore[arg-type]
    )
    result = exact_lll_reduce_3d(np.asarray(basis, dtype=np.float64))

    assert len(trace) == result.diagnostics.swaps == 4
    assert all(
        before.denominator == after.denominator == 1
        and 0 < after < before
        for before, after in trace
    )
    scale = Fraction(1, 1 << exponent)
    expected_rows = tuple(
        tuple(value * scale for value in row) for row in oracle_rows
    )
    assert result.transform == oracle_transform
    assert result.reduced_rows == expected_rows


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
        'work': 3_264,
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
    return result, before_fixed, after_fixed


@pytest.mark.parametrize('seed', (0, 1))
def test_repository_thin_regressions_meet_proof_box_gates(seed: int) -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'repository-regression']
    assert len(fixtures) == 2
    for fixture in fixtures:
        _result, before, after = _before_after(fixture, seed)
        assert after.fixed_box_count <= 4096
        assert before.fixed_box_count >= 100 * after.fixed_box_count


@pytest.mark.parametrize('seed', (0, 1))
def test_exact_cubic_shear_work_is_bounded_independently_of_magnitude(
    seed: int,
) -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'exact-large-shear']
    counts = []
    for fixture in fixtures:
        _result, before, after = _before_after(fixture, seed)
        assert before.fixed_box_count > after.fixed_box_count
        assert after.fixed_box_count <= 64
        counts.append(after.fixed_box_count)
    assert len(set(counts)) == 1


@pytest.mark.parametrize('seed', (0, 1))
def test_equivalent_well_conditioned_cohort_meets_buffered_gates(seed: int) -> None:
    fixtures = [fixture for fixture in FROZEN_WORKLOAD_FIXTURES
                if fixture.cohort == 'equivalent-well-conditioned']
    for fixture in fixtures:
        _result, before, after = _before_after(fixture, seed)
        assert after.fixed_box_count <= 256
        if before.fixed_box_count > 1_000_000:
            assert before.fixed_box_count >= 1000 * after.fixed_box_count


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
    for fixture in FROZEN_RANDOM_UNIMODULAR_FIXTURES:
        seeded_counts = []
        fixed_counts = []
        for image_search in (0, 1):
            result, before, after = _before_after(fixture, image_search)
            seeded_before = evaluate_proof_workload(
                fixture.basis,
                pi=fixture.pi,
                pj=fixture.pj,
                image_search=image_search,
                common_exponent=common_alignment_exponent(
                    fixture.basis, fixture.pi, fixture.pj
                ),
            )
            seeded_after = evaluate_proof_workload(
                result.reduced_rows,
                pi=fixture.pi,
                pj=fixture.pj,
                image_search=image_search,
                common_exponent=common_alignment_exponent(
                    fixture.basis, fixture.pi, fixture.pj
                ),
            )
            seeded_counts.append(
                (seeded_before.box_count, seeded_after.box_count)
            )
            fixed_counts.append((before.fixed_box_count, after.fixed_box_count))
            assert after.fixed_box_count <= 256
            if before.fixed_box_count > 1_000_000:
                assert before.fixed_box_count >= 1000 * after.fixed_box_count
        expected_seeded, expected_fixed = expected[fixture.name]
        assert tuple(seeded_counts) == expected_seeded
        assert fixed_counts == [expected_fixed, expected_fixed]


def test_r5_sc_001_inverse_bound_product_improves_by_at_least_100x() -> None:
    fixture = next(
        fixture for fixture in FROZEN_WORKLOAD_FIXTURES
        if fixture.name == 'r5-sc-001'
    )
    _result, before, after = _before_after(fixture, seed=1)

    assert before.inverse_bound_product >= 100 * after.inverse_bound_product
    assert before.bucket_bins != after.bucket_bins


def test_intrinsic_anisotropy_is_reported_separately_without_false_gate() -> None:
    fixture = next(
        fixture for fixture in FROZEN_WORKLOAD_FIXTURES
        if fixture.cohort == 'intrinsically-anisotropic'
    )
    result, before, after = _before_after(fixture, seed=1)

    assert result.certified
    assert before.fixed_box_count > 0
    assert after.fixed_box_count > 0
    assert max(after.inverse_column_l1) >= 2**16
