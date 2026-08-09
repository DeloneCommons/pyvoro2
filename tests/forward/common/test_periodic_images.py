from __future__ import annotations

from fractions import Fraction
from itertools import product
import math
import sys

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell
from pyvoro2._internal.periodic_images import (
    _basis_cache_clear,
    _basis_cache_info,
    _MAX_EXACT_CANDIDATES_PER_BATCH,
    _MAX_EXACT_CANDIDATES_PER_PAIR,
    _MAX_SEED_CANDIDATES,
    exact_distance_less_than,
    minimum_image_displacements,
    MinimumImageCertificationError,
)
from pyvoro2._internal.planar.domain_geometry import geometry2d
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from pyvoro2.inverse.separator import resolve_separator_observations
import pyvoro2
import pyvoro2.planar as planar


def _aligned_exact_geometry(
    pi: np.ndarray,
    pj: np.ndarray,
    lattice: np.ndarray,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    """Test-only exact dyadic alignment independent of production helpers."""

    values = [float(value) for value in pi]
    values += [float(value) for value in pj]
    values += [float(value) for value in lattice.ravel()]
    parts = [value.as_integer_ratio() for value in values]
    exponents = [denominator.bit_length() - 1 for _, denominator in parts]
    exponent = max(exponents)
    integers = [
        numerator << (exponent - value_exponent)
        for (numerator, _), value_exponent in zip(parts, exponents)
    ]
    dimension = pi.size
    displacement = tuple(
        integers[dimension + axis] - integers[axis]
        for axis in range(dimension)
    )
    start = 2 * dimension
    rows = tuple(
        tuple(
            integers[start + row * dimension + column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    return displacement, rows, exponent


def _exact_exhaustive_oracle(
    pi: np.ndarray,
    pj: np.ndarray,
    lattice: np.ndarray,
    *,
    radius: int,
    orientation: int,
) -> tuple[tuple[int, ...], tuple[float, ...], float, int]:
    """Enumerate a fixed test cube without using the production proof box."""

    displacement, rows, exponent = _aligned_exact_geometry(pi, pj, lattice)
    dimension = pi.size
    best_shift = None
    best_values = None
    best_distance = None
    tie_count = 0
    for shift in product(range(-radius, radius + 1), repeat=dimension):
        values = tuple(
            displacement[column]
            + sum(
                shift[row] * rows[row][column]
                for row in range(dimension)
            )
            for column in range(dimension)
        )
        distance = sum(value * value for value in values)
        if best_distance is None or distance < best_distance:
            best_shift = shift
            best_values = values
            best_distance = distance
            tie_count = 1
        elif distance == best_distance:
            tie_count += 1
            assert best_shift is not None
            if (
                (orientation == 1 and shift < best_shift)
                or (orientation == -1 and shift > best_shift)
            ):
                best_shift = shift
                best_values = values
    assert best_shift is not None and best_values is not None
    assert best_distance is not None
    assert all(abs(value) < radius for value in best_shift)
    denominator = 1 << exponent
    displacement_float = tuple(
        float(Fraction(value, denominator)) for value in best_values
    )
    distance_float = float(Fraction(best_distance, denominator * denominator))
    return best_shift, displacement_float, distance_float, tie_count


def _minimum(
    pi: np.ndarray,
    pj: np.ndarray,
    lattice: np.ndarray,
    *,
    periodic_axes: tuple[bool, ...] | None = None,
    orientation: int = 1,
    image_search: int = 1,
):
    dimension = pi.size
    axes = (True,) * dimension if periodic_axes is None else periodic_axes
    return minimum_image_displacements(
        pi.reshape(1, dimension),
        pj.reshape(1, dimension),
        lattice_vectors=lattice,
        periodic_axes=axes,
        tie_orientation=np.array([orientation], dtype=np.int8),
        image_search=image_search,
    )


def test_fixed_skewed_triclinic_regression_matches_independent_oracle() -> None:
    cell = PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
    pi = np.array([0.63696169, 0.26978671, 0.04097352])
    pj = np.array([0.01652764, 0.81327024, 0.91275558])
    lattice = np.asarray(cell.vectors, dtype=np.float64)

    expected = _exact_exhaustive_oracle(
        pi,
        pj,
        lattice,
        radius=8,
        orientation=1,
    )
    result = _minimum(pi, pj, lattice)

    assert expected[0] == (2, -1, -1)
    assert tuple(result.shift[0]) == expected[0]
    np.testing.assert_allclose(
        result.displacement[0],
        [-0.12043405000000007, -0.45651646999999995, -0.12821793999999997],
        rtol=0.0,
        atol=1e-16,
    )
    assert float(result.distance_squared[0]) == 0.23935148791850697
    np.testing.assert_array_equal(result.displacement[0], expected[1])
    assert float(result.distance_squared[0]) == expected[2]
    assert result.certified is True
    assert result.method == 'triclinic-finite-box'
    assert int(result.candidate_count[0]) < 100


@pytest.mark.parametrize('dimension', (2, 3))
def test_orthogonal_fast_path_covers_every_periodic_axis_mask(
    dimension: int,
) -> None:
    lengths = np.array([2.0, 4.0, 8.0][:dimension])
    lattice = np.diag(lengths)
    pi = np.zeros(dimension)
    pj = np.array([1.25, 2.25, 4.25][:dimension])

    for periodic_axes in product((False, True), repeat=dimension):
        result = _minimum(
            pi,
            pj,
            lattice,
            periodic_axes=periodic_axes,
        )
        expected = tuple(-1 if periodic_axes[axis] else 0
                         for axis in range(dimension))
        assert tuple(result.shift[0]) == expected
        assert result.method == 'orthogonal-exact'
        assert int(result.candidate_count[0]) <= 2 ** sum(periodic_axes)


def test_orthogonal_exact_ties_use_orientation_and_readonly_owned_arrays() -> None:
    lattice = np.diag([2.0, 4.0, 8.0])
    pi = np.zeros(3)
    pj = np.array([1.0, 2.0, 4.0])

    forward = _minimum(pi, pj, lattice, orientation=1)
    backward = _minimum(pi, pj, lattice, orientation=-1)

    assert tuple(forward.shift[0]) == (-1, -1, -1)
    assert tuple(backward.shift[0]) == (0, 0, 0)
    assert int(forward.tie_count[0]) == 8
    assert int(forward.candidate_count[0]) == 8
    for array in (
        forward.shift,
        forward.displacement,
        forward.distance_squared,
        forward.candidate_count,
        forward.tie_count,
        forward.seed_count,
        forward.seed_truncated,
    ):
        assert array.flags.owndata
        assert not array.flags.writeable


def test_partial_periodic_half_cell_ties_leave_other_shifts_zero() -> None:
    lattice = np.diag([2.0, 4.0, 8.0])
    result = _minimum(
        np.zeros(3),
        np.array([1.0, 2.0, 4.0]),
        lattice,
        periodic_axes=(True, False, True),
        orientation=1,
    )

    assert tuple(result.shift[0]) == (-1, 0, -1)
    assert int(result.tie_count[0]) == 4
    assert int(result.candidate_count[0]) == 4


def test_triclinic_exact_tie_is_repeatable_and_reversal_compatible() -> None:
    lattice = np.array(
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.25, 1.0]],
        dtype=np.float64,
    )
    pi = np.zeros(3)
    pj = 0.5 * lattice[0]

    forward = _minimum(pi, pj, lattice, orientation=1, image_search=0)
    repeated = _minimum(pi, pj, lattice, orientation=1, image_search=0)
    reverse = _minimum(pj, pi, lattice, orientation=-1, image_search=0)

    assert tuple(forward.shift[0]) == (-1, 0, 0)
    assert int(forward.tie_count[0]) == 2
    np.testing.assert_array_equal(repeated.shift, forward.shift)
    np.testing.assert_array_equal(reverse.shift, -forward.shift)
    np.testing.assert_array_equal(reverse.displacement, -forward.displacement)


def test_lattice_translation_pair_reversal_and_common_translation() -> None:
    lattice = np.array(
        [[1.0, 0.0, 0.0], [1.5, 1.0, 0.0], [0.25, -0.5, 1.0]],
        dtype=np.float64,
    )
    pi = np.array([0.125, -0.25, 0.375])
    pj = np.array([0.75, 0.625, -0.125])
    translation = np.array([3, -2, 1], dtype=np.int64)
    cartesian = np.array([2.0, -3.0, 0.25])

    result = _minimum(pi, pj, lattice, orientation=1)
    translated_j = _minimum(
        pi,
        pj + translation @ lattice,
        lattice,
        orientation=1,
    )
    translated_i = _minimum(
        pi + translation @ lattice,
        pj,
        lattice,
        orientation=1,
    )
    reverse = _minimum(pj, pi, lattice, orientation=-1)
    common = _minimum(
        pi + cartesian,
        pj + cartesian,
        lattice,
        orientation=1,
    )

    np.testing.assert_array_equal(
        translated_j.shift[0],
        result.shift[0] - translation,
    )
    np.testing.assert_array_equal(
        translated_i.shift[0],
        result.shift[0] + translation,
    )
    for translated in (translated_j, translated_i):
        np.testing.assert_array_equal(
            translated.displacement,
            result.displacement,
        )
        np.testing.assert_array_equal(
            translated.distance_squared,
            result.distance_squared,
        )
        assert translated.exact_distance_key == result.exact_distance_key
    np.testing.assert_array_equal(reverse.shift, -result.shift)
    np.testing.assert_array_equal(reverse.displacement, -result.displacement)
    np.testing.assert_array_equal(common.shift, result.shift)
    np.testing.assert_array_equal(common.displacement, result.displacement)


def test_random_triclinic_cases_match_fixed_cube_exact_oracle() -> None:
    rng = np.random.default_rng(20260809)
    scales = (1e-3, 1.0, 1e3)
    skews = (
        (0.01, -0.02, 0.03),
        (0.6, -0.4, 0.5),
        (1.5, -0.75, 1.25),
    )
    checked = 0
    candidate_total = 0
    for scale in scales:
        for bxy, bxz, byz in skews:
            lattice = scale * np.array(
                [[1.0, 0.0, 0.0], [bxy, 1.2, 0.0], [bxz, byz, 0.9]],
                dtype=np.float64,
            )
            cell = PeriodicCell(lattice, origin=(2.0, -3.0, 0.5))
            lattice = np.asarray(cell.vectors, dtype=np.float64)
            for _ in range(3):
                pi = scale * rng.uniform(-1.0, 1.0, size=3)
                fractional_delta = rng.uniform(-0.8, 0.8, size=3)
                lattice_translation = rng.integers(-3, 4, size=3)
                pj = pi + (fractional_delta + lattice_translation) @ lattice
                expected = _exact_exhaustive_oracle(
                    pi,
                    pj,
                    lattice,
                    radius=12,
                    orientation=1,
                )
                result = _minimum(pi, pj, lattice)
                assert tuple(result.shift[0]) == expected[0]
                np.testing.assert_array_equal(result.displacement[0], expected[1])
                assert float(result.distance_squared[0]) == expected[2]
                candidate_total += int(result.candidate_count[0])
                checked += 1

    assert checked == 27
    assert candidate_total < 100_000


def test_warning_level_ill_conditioned_cell_still_certifies_exactly() -> None:
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1e-11, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pi = np.zeros(3)
    pj = np.array([0.0, 2.5e-12, 0.0])

    with pytest.warns(RuntimeWarning, match='very ill-conditioned'):
        cell = PeriodicCell(lattice)
    canonical_lattice = np.asarray(cell.vectors, dtype=np.float64)
    expected = _exact_exhaustive_oracle(
        pi,
        pj,
        canonical_lattice,
        radius=3,
        orientation=1,
    )
    result = _minimum(pi, pj, canonical_lattice)

    assert np.linalg.cond(canonical_lattice) == pytest.approx(2e11)
    assert expected[0] == (0, 0, 0)
    assert tuple(result.shift[0]) == expected[0]
    np.testing.assert_array_equal(result.displacement[0], expected[1])
    assert float(result.distance_squared[0]) == expected[2]
    exact_numerator, exact_denominator = float(pj[1]).as_integer_ratio()
    assert result.exact_distance_key[0].numerator == exact_numerator ** 2
    assert result.exact_distance_key[0].denominator_exponent == 2 * (
        exact_denominator.bit_length() - 1
    )
    assert result.certified is True
    assert int(result.candidate_count[0]) <= 8


def test_image_search_changes_only_seed_work() -> None:
    cell = PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
    lattice = np.asarray(cell.vectors, dtype=np.float64)
    pi = np.array([0.63696169, 0.26978671, 0.04097352])
    pj = np.array([0.01652764, 0.81327024, 0.91275558])

    results = [
        _minimum(pi, pj, lattice, image_search=value)
        for value in (0, 1, 5, sys.maxsize)
    ]
    for result in results[1:]:
        np.testing.assert_array_equal(result.shift, results[0].shift)
        np.testing.assert_array_equal(
            result.displacement,
            results[0].displacement,
        )
        np.testing.assert_array_equal(
            result.distance_squared,
            results[0].distance_squared,
        )
        assert result.exact_distance_key == results[0].exact_distance_key
    assert int(results[-1].seed_count[0]) == _MAX_SEED_CANDIDATES
    assert bool(results[-1].seed_truncated[0]) is True


def test_resource_contract_fails_structurally_without_approximation() -> None:
    lattice = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0003, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cell = PeriodicCell(lattice)
    q = np.array([0.49, 0.49, 0.49])
    pi = np.zeros(3)
    pj = q @ np.asarray(cell.vectors)

    with pytest.raises(MinimumImageCertificationError) as captured:
        _minimum(pi, pj, np.asarray(cell.vectors))

    error = captured.value
    assert error.stage == 'pair_candidate_budget'
    assert error.pair_index == 0
    assert error.candidate_bound is not None
    assert error.candidate_bound > _MAX_EXACT_CANDIDATES_PER_PAIR
    assert error.configured_limit == _MAX_EXACT_CANDIDATES_PER_PAIR
    assert error.interval_widths is not None
    assert error.basis_summary['condition_number'] < 1e15


def test_batch_cumulative_budget_is_checked_before_exact_enumeration() -> None:
    lattice = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.001, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cell = PeriodicCell(lattice)
    q = np.array([0.49, 0.49, 0.49])
    pj_row = q @ np.asarray(cell.vectors)
    pi = np.zeros((6, 3))
    pj = np.tile(pj_row, (6, 1))

    with pytest.raises(MinimumImageCertificationError) as captured:
        minimum_image_displacements(
            pi,
            pj,
            lattice_vectors=cell.vectors,
            periodic_axes=(True, True, True),
            tie_orientation=np.ones(6, dtype=np.int8),
            image_search=1,
        )

    error = captured.value
    assert error.stage == 'batch_candidate_budget'
    assert error.pair_index == 5
    assert error.candidate_bound is not None
    assert error.candidate_bound > _MAX_EXACT_CANDIDATES_PER_BATCH
    assert error.configured_limit == _MAX_EXACT_CANDIDATES_PER_BATCH


def test_certified_shift_outside_int64_contract_fails_structurally() -> None:
    beyond = float(2**63 + 2048)

    with pytest.raises(MinimumImageCertificationError) as captured:
        _minimum(
            np.zeros(3),
            np.array([beyond, 0.0, 0.0]),
            np.eye(3),
        )

    error = captured.value
    assert error.stage == 'shift_range'
    assert error.pair_index == 0


def test_basis_cache_uses_exact_geometry_bits_and_ignores_origin() -> None:
    _basis_cache_clear()
    points_i = np.array([[0.0, 0.0, 0.0]])
    points_j = np.array([[0.2, 0.3, 0.4]])
    first = PeriodicCell.from_params(1, 0.25, 1, 0, 0, 1)
    same_basis_new_origin = PeriodicCell.from_params(
        1,
        0.25,
        1,
        0,
        0,
        1,
        origin=(4.0, -3.0, 2.0),
    )
    different_bits = np.asarray(first.vectors, dtype=np.float64).copy()
    different_bits[1, 0] = np.nextafter(different_bits[1, 0], np.inf)
    third = PeriodicCell(different_bits)

    geometry3d(first).minimum_image_displacements(
        points_i,
        points_j,
        tie_orientation=np.array([1], dtype=np.int8),
        image_search=1,
    )
    after_first = _basis_cache_info()
    geometry3d(same_basis_new_origin).minimum_image_displacements(
        points_i,
        points_j,
        tie_orientation=np.array([1], dtype=np.int8),
        image_search=1,
    )
    after_same = _basis_cache_info()
    geometry3d(third).minimum_image_displacements(
        points_i,
        points_j,
        tie_orientation=np.array([1], dtype=np.int8),
        image_search=1,
    )
    after_different = _basis_cache_info()

    assert after_first.misses == 1
    assert after_same.hits == 1
    assert after_same.misses == 1
    assert after_different.misses == 2
    assert after_different.maxsize == 128


def test_basis_cache_is_bounded_to_128_exact_entries() -> None:
    _basis_cache_clear()
    empty = np.empty((0, 3), dtype=np.float64)
    for index in range(130):
        lattice = np.diag([1.0 + index * 2.0 ** -40, 1.0, 1.0])
        minimum_image_displacements(
            empty,
            empty,
            lattice_vectors=lattice,
            periodic_axes=(True, True, True),
            tie_orientation=np.empty(0, dtype=np.int8),
            image_search=1,
        )
    info = _basis_cache_info()
    assert info.currsize == 128
    assert info.misses == 130


def test_basis_cache_key_includes_periodic_axes() -> None:
    _basis_cache_clear()
    point = np.zeros((1, 3))
    for axes in ((True, True, True), (True, False, True)):
        minimum_image_displacements(
            point,
            point,
            lattice_vectors=np.eye(3),
            periodic_axes=axes,
            tie_orientation=np.array([1], dtype=np.int8),
            image_search=1,
        )
    info = _basis_cache_info()
    assert info.misses == 2
    assert info.currsize == 2


def test_geometry_adapters_share_the_certified_orthogonal_path() -> None:
    spatial = OrthorhombicCell(
        ((0.0, 2.0), (0.0, 4.0), (0.0, 8.0)),
        periodic=(True, False, True),
    )
    planar_cell = planar.RectangularCell(
        ((0.0, 2.0), (0.0, 4.0)),
        periodic=(False, True),
    )

    spatial_result = geometry3d(spatial).minimum_image_displacements(
        np.zeros((1, 3)),
        np.array([[1.25, 3.5, 4.25]]),
        tie_orientation=np.array([1], dtype=np.int8),
        image_search=999,
    )
    planar_result = geometry2d(planar_cell).minimum_image_displacements(
        np.zeros((1, 2)),
        np.array([[1.25, 3.5]]),
        tie_orientation=np.array([1], dtype=np.int8),
        image_search=999,
    )

    assert spatial_result.method == 'orthogonal-exact'
    assert planar_result.method == 'orthogonal-exact'
    assert tuple(spatial_result.shift[0]) == (-1, 0, -1)
    assert tuple(planar_result.shift[0]) == (0, -1)


def test_separator_inference_uses_certified_result_and_keeps_explicit_shift() -> None:
    cell = PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
    points = np.array(
        [
            [0.63696169, 0.26978671, 0.04097352],
            [0.01652764, 0.81327024, 0.91275558],
        ]
    )
    inferred = resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        domain=cell,
        image_search=0,
    )
    explicit = resolve_separator_observations(
        points,
        [(0, 1, 0.5, (1, 0, -1))],
        domain=cell,
        image_search=sys.maxsize,
    )

    assert tuple(inferred.shifts[0]) == (2, -1, -1)
    assert float(inferred.distance2[0]) == 0.23935148791850697
    assert not any('boundary' in warning for warning in inferred.warnings)
    assert tuple(explicit.shifts[0]) == (1, 0, -1)
    assert float(explicit.distance2[0]) == pytest.approx(
        0.455884497918507,
        rel=0.0,
        abs=1e-15,
    )
    assert bool(explicit.explicit_shift[0]) is True


def test_separator_exact_tie_is_neutral_to_external_id_labels() -> None:
    cell = OrthorhombicCell(
        ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False, False),
    )
    points = np.array(
        [
            [0.0, 0.25, 0.5],
            [0.5, 0.25, 0.5],
        ]
    )
    first = resolve_separator_observations(
        points,
        [(100, 7, 0.5)],
        domain=cell,
        ids=np.array([100, 7]),
        index_mode='id',
    )
    relabeled = resolve_separator_observations(
        points,
        [(3, 900, 0.5)],
        domain=cell,
        ids=np.array([3, 900]),
        index_mode='id',
    )

    assert tuple(first.shifts[0]) == (-1, 0, 0)
    np.testing.assert_array_equal(first.delta[0], [-0.5, 0.0, 0.0])
    assert float(first.distance2[0]) == 0.25
    np.testing.assert_array_equal(relabeled.i, first.i)
    np.testing.assert_array_equal(relabeled.j, first.j)
    np.testing.assert_array_equal(relabeled.shifts, first.shifts)
    np.testing.assert_array_equal(relabeled.delta, first.delta)
    np.testing.assert_array_equal(relabeled.distance2, first.distance2)


def test_periodic_duplicate_pair_distance_uses_certified_minimum_image() -> None:
    cell = PeriodicCell.from_params(1, 1.5, 1, 0, 0, 1)
    points = np.array(
        [
            [0.63696169, 0.26978671, 0.04097352],
            [0.01652764, 0.81327024, 0.91275558],
        ]
    )

    pairs = pyvoro2.duplicate_check(
        points,
        threshold=0.6,
        domain=cell,
        wrap=True,
        mode='return',
    )

    assert len(pairs) == 1
    assert (pairs[0].i, pairs[0].j) == (0, 1)
    assert pairs[0].distance == math.sqrt(0.23935148791850697)


def test_planar_periodic_duplicate_distance_uses_shared_primitive() -> None:
    _basis_cache_clear()
    cell = planar.RectangularCell(
        ((0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False),
    )
    pairs = planar.duplicate_check(
        np.array([[0.125, 0.25], [0.625, 0.25]]),
        threshold=np.nextafter(0.5, np.inf),
        domain=cell,
        wrap=True,
        mode='return',
    )

    assert len(pairs) == 1
    assert pairs[0].distance == 0.5
    cache = _basis_cache_info()
    assert cache.misses == 1
    assert cache.currsize == 1


def test_exact_distance_key_controls_threshold_comparison() -> None:
    result = _minimum(
        np.zeros(2),
        np.array([0.5, 0.0]),
        np.eye(2),
    )
    key = result.exact_distance_key[0]
    assert exact_distance_less_than(key, np.nextafter(0.5, np.inf))
    assert not exact_distance_less_than(key, 0.5)
