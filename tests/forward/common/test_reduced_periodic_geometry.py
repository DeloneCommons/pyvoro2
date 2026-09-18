"""WP4 independent physical-lattice oracles and consumer regressions."""

from fractions import Fraction as F
from itertools import product
import math

import numpy as np
import pytest

from pyvoro2 import PeriodicCell
import pyvoro2
from pyvoro2._internal import periodic_images as images
from pyvoro2._internal import exact_lattice as exact
from pyvoro2._internal.duplicate_scanning import (
    candidate_pairs, cross_candidate_pairs, scan_close_pairs,
)
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from pyvoro2._internal import generator_preparation as preparation
from _lattice_reduction_workload import (
    exact_matrix, inverse, seeded_unimodular_pair,
)


def minimum(basis, delta, *, start=(0., 0., 0.), orientation=1, search=1):
    return images.minimum_image_displacements(
        [start], [delta], lattice_vectors=basis,
        periodic_axes=(True,) * len(basis),
        tie_orientation=[orientation], image_search=search,
    )


def shear(h):
    return np.array(((1., 0., 0.), (h, 1., 0.), (0., 0., 1.)))


def oracle(basis, pi, pj, orientation=1):
    """Independent cofactor/Fraction CVP, bounded by Cauchy--Schwarz.

    The zero shift is an incumbent with squared norm D. For every minimizer,
    |q_j+s_j|**2 <= D * ||inverse[:,j]||_2**2. The integer ceiling of
    sqrt(ceil(D * column_norm_squared)) gives a conservative finite region.
    No production reduction, box, mapper, or stopping-at-the-edge oracle.
    """
    rows = exact_matrix(basis)
    inv = inverse(rows)
    d = tuple(F(float(y)) - F(float(x)) for x, y in zip(pi, pj))
    distance = sum(v*v for v in d)
    q = tuple(sum(d[k]*inv[k][j] for k in range(3)) for j in range(3))
    bounds = []
    for j in range(3):
        square = math.ceil(distance * sum(inv[k][j]**2 for k in range(3)))
        radius = math.isqrt(square)
        radius += radius*radius < square
        bounds.append(range(math.ceil(-q[j]-radius), math.floor(-q[j]+radius)+1))
    assert math.prod(map(len, bounds)) < 100_000
    candidates = []
    for shift in product(*bounds):
        r = tuple(d[k] + sum(shift[j]*rows[j][k] for j in range(3))
                  for k in range(3))
        candidates.append((sum(v*v for v in r), r, shift))
    best = min(item[0] for item in candidates)
    ties = [item for item in candidates if item[0] == best]
    chosen = (min if orientation == 1 else max)(ties, key=lambda item: item[1])
    return chosen, len(ties)


@pytest.mark.parametrize('h', (8, 2**32, 2**63))
@pytest.mark.parametrize('orientation', (1, -1))
def test_shear_physical_ties_map_before_int64(h, orientation):
    result = minimum(shear(h), (.5, .5, 0.), orientation=orientation)
    expected = (-.5, -.5, 0.) if orientation == 1 else (.5, .5, 0.)
    assert tuple(result.displacement[0]) == expected
    assert tuple(result.shift[0]) == ((h-1, -1, 0) if orientation == 1 else (0, 0, 0))
    assert result.tie_count[0] == 4


@pytest.mark.parametrize('basis', (
    np.diag((-1., 1., 1.)), np.diag((1., -1., -1.)),
    np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., -1.))),
))
def test_signed_rows_choose_physical_tie_in_fixed_cartesian_axes(basis):
    result = minimum(basis, (.5, .5, .5))
    assert tuple(result.displacement[0]) == (-.5, -.5, -.5)
    reverse = minimum(basis, (0., 0., 0.), start=(.5, .5, .5), orientation=-1)
    assert np.array_equal(reverse.displacement, -result.displacement)
    assert np.array_equal(reverse.shift, -result.shift)


@pytest.mark.parametrize('seed', (0, 1, 17, 56))
def test_known_unimodular_inverse_defines_cubic_oracle(seed):
    transform, known_inverse = seeded_unimodular_pair(seed)
    basis = np.array(transform, dtype=float)
    assert exact_matrix(basis) == transform
    assert tuple(tuple(sum(transform[i][k]*known_inverse[k][j] for k in range(3))
                       for j in range(3)) for i in range(3)) == (
        (1, 0, 0), (0, 1, 0), (0, 0, 1))
    result = minimum(basis, (.5, .5, .125))
    expected_shift = tuple(-known_inverse[0][j]-known_inverse[1][j]
                           for j in range(3))
    assert tuple(result.shift[0]) == expected_shift
    assert tuple(result.displacement[0]) == (-.5, -.5, .125)


def test_huge_reduced_coefficients_cancel_to_small_user_shift():
    h = 2**80
    result = minimum(shear(h), (float(h), 1., .25))
    assert tuple(result.shift[0]) == (0, -1, 0)
    assert tuple(result.displacement[0]) == (0., 0., .25)


def test_exact_tie_covariance_with_large_exact_endpoint_translation():
    basis = shear(2**32)
    original = minimum(basis, (.5, .5, .25))
    translated = minimum(basis, (.5+2**32, 1.5, .25), start=(1., 0., 0.))
    assert np.array_equal(original.displacement, translated.displacement)
    assert tuple(translated.shift[0]) == tuple(original.shift[0] + (1, -1, 0))


def test_int64_min_reversal_is_a_materialization_boundary():
    result = minimum(shear(2**63), (0., -1., .25))
    assert tuple(result.shift[0]) == (-2**63, 1, 0)
    with pytest.raises(images.MinimumImageCertificationError) as caught:
        minimum(shear(2**63), (0., 0., 0.), start=(0., -1., .25), orientation=-1)
    assert caught.value.stage == 'shift_range'


def test_distance_only_classifies_huge_user_shift_without_float_views():
    geometry = geometry3d(PeriodicCell(shear(2**80)))
    points = np.array(((0., 0., 0.), (0., 1., 2**-20)))
    result = scan_close_pairs(points, radius=1e-5, geometry=geometry,
                              inclusive_squared=1e-10, max_pairs=10)
    assert result.pairs == ((0, 1, 2**-20),)
    with pytest.raises(images.MinimumImageCertificationError) as caught:
        minimum(shear(2**80), points[1])
    assert caught.value.stage == 'shift_range'


def test_reduced_rows_keep_non_binary64_exact_entry():
    basis = np.array(((1., 2**-55, 0.), (1., 1., 0.), (0., 0., 1.)))
    expected, ties = oracle(basis, (0., 0., 0.), (0., .5, 0.))
    result = minimum(basis, (0., .5, 0.))
    assert tuple(result.shift[0]) == expected[2]
    assert result.tie_count[0] == ties
    key = result.exact_distance_key[0]
    assert F(key.numerator, 1 << key.denominator_exponent) == expected[0]
    # The actual winner contains 1 - 2**-55; rounding B changes its distance.
    assert expected[2] == (1, -1, 0)
    assert expected[0] < F(1, 4)


@pytest.mark.parametrize('reverse_rows', (False, True))
def test_determinant_cancellation_is_still_exact_cubic_geometry(reverse_rows):
    n = 2**27
    basis = np.array(((n, n-1, 0), (n+1, n, 0), (0, 0, 1)), dtype=float)
    if reverse_rows:
        basis = basis[[1, 0, 2]]
    inv = inverse(exact_matrix(basis))
    assert all(x.denominator == 1 for row in inv for x in row)
    result = minimum(basis, (.5, .5, .25))
    assert tuple(result.displacement[0]) == (-.5, -.5, .25)
    assert tuple(result.shift[0]) == tuple(-inv[0][j]-inv[1][j] for j in range(3))


@pytest.mark.parametrize(('thin', 'count', 'rows'), ((.0003, 3270, 1), (.001, 980, 6)))
def test_thin_actual_consumer_counts_under_existing_budgets(thin, count, rows):
    basis = np.array(((1., 0., 0.), (1., thin, 0.), (0., 0., 1.)))
    endpoint = np.array((.49, .49, .49)) @ basis
    result = images.minimum_image_displacements(
        np.zeros((rows, 3)), np.tile(endpoint, (rows, 1)), lattice_vectors=basis,
        periodic_axes=(True,)*3, tie_orientation=[1]*rows, image_search=1,
    )
    assert result.candidate_count.tolist() == [count]*rows
    assert result.seed_count.tolist() == [27]*rows


def test_warm_proof_context_does_not_bypass_stricter_reduction_limit():
    key = images._basis_key(shear(2**32), (True,)*3)
    images._basis_cache_clear()
    exact._reduction_cache_clear()
    first = images._prepare_basis(*key)
    assert images._prepare_basis(*key) is first
    strict = exact.ExactLatticeReductionLimits(max_transform_bits=8)
    with pytest.raises(exact.ExactLatticeReductionResourceError):
        images._prepare_basis(*key, reduction_limits=strict)
    assert images._basis_cache_info().currsize == 1


def test_sparse_cubic_cloud_candidates_do_not_collapse_under_user_shear():
    # Same y/z makes the source x bucket collapse under a huge unimodular
    # shear; analytically the physical lattice is Z^3 with a single seam pair.
    points = np.array([(k/128, .25, .25) for k in range(128)] +
                      [(1.-2**-20, .25, .25)])
    geometry = geometry3d(PeriodicCell(shear(2**32)))
    within = list(candidate_pairs(points, radius=1e-5, geometry=geometry))
    assert within == [(0, 128)]
    cross = list(cross_candidate_pairs(
        points[:128], points[128:], radius=1e-5, geometry=geometry,
    ))
    assert cross == [(0, 0)]
    scan = scan_close_pairs(
        points, radius=1e-5, geometry=geometry,
        inclusive_squared=1e-10, max_pairs=10,
    )
    assert scan.pairs == ((0, 128, 2**-20),)
    assert scan.candidate_count == 1


@pytest.mark.parametrize('search', (0, 1, 4, 10000))
def test_seed_is_correctness_neutral_after_reduction(search):
    result = minimum(shear(256), (.5, .5, .125), search=search)
    assert tuple(result.displacement[0]) == (-.5, -.5, .125)
    assert result.tie_count[0] == 4
    assert result.seed_count[0] <= 4096
    assert bool(result.seed_truncated[0]) == (search == 10000)


def test_exact_core_returns_huge_user_shift_before_public_range_check():
    basis, rows = images._minimum_image_solutions(
        [(0., 0., 0.)], [(.5, .5, 0.)], lattice_vectors=shear(2**80),
        periodic_axes=(True,)*3, tie_orientation=[1], image_search=1,
    )
    assert basis.reduction is not None
    assert rows[0].shift == (2**80-1, -1, 0)
    assert tuple(F(x, 1 << rows[0].denominator_exponent)
                 for x in rows[0].displacement_numerators) == (-F(1, 2), -F(1, 2), 0)


def test_distance_only_does_not_require_an_unrepresentable_squared_float():
    result = images.minimum_image_distances(
        [(0., 0.)], [(0., 1e200)], lattice_vectors=np.eye(2),
        periodic_axes=(True, False), tie_orientation=[1], image_search=1,
    )
    key = result.exact_distance_key[0]
    assert images.exact_distance_less_than(key, 2e200)
    assert images.exact_distance_float(key) == 1e200


def test_extreme_nearly_dependent_basis_preserves_exact_small_residual():
    basis = ((1., 0., 0.), (1., 2**-500, 0.), (0., 0., 2**500))
    result = minimum(basis, (0., 2**-502, 0.))
    assert tuple(result.shift[0]) == (0, 0, 0)
    key = result.exact_distance_key[0]
    assert F(key.numerator, 1 << key.denominator_exponent) == F(1, 2**1004)


def test_partial_signed_diagonal_does_not_invoke_rank_three_reduction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('rectangular fast path invoked rank-3 reduction')
    monkeypatch.setattr(images, 'exact_lll_reduce_3d', forbidden)
    images._basis_cache_clear()
    result = images.minimum_image_displacements(
        [(0., 0., 0.)], [(.5, .5, .5)], lattice_vectors=np.diag((-1., 1., -1.)),
        periodic_axes=(True, False, True), tie_orientation=[1], image_search=1,
    )
    assert tuple(result.displacement[0]) == (-.5, .5, -.5)
    assert tuple(result.shift[0]) == (1, 0, 1)
    assert result.method == 'orthogonal-exact'


def test_generator_and_temporary_preparation_use_reduced_sparse_scans(monkeypatch):
    geometry = geometry3d(PeriodicCell(shear(2**32)))
    snapshot = geometry.native_periodic_snapshot()
    points = np.array([(k/128, .25, .25) for k in range(128)])
    scans = []
    within = preparation.scan_close_pairs
    cross = preparation.scan_cross_close_pairs

    def record_within(*args, **kwargs):
        scan = within(*args, **kwargs)
        scans.append(scan)
        return scan

    def record_cross(*args, **kwargs):
        scan = cross(*args, **kwargs)
        scans.append(scan)
        return scan

    monkeypatch.setattr(preparation, 'scan_close_pairs', record_within)
    monkeypatch.setattr(preparation, 'scan_cross_close_pairs', record_cross)
    policy = dict(duplicate_check='off', duplicate_threshold=1e-5,
                  duplicate_wrap=False, duplicate_max_pairs=10,
                  periodic_snapshot=snapshot, backend_radii=None)
    persistent = preparation.prepare_generators(
        points, geometry=geometry, operation='ghost_cells', external_ids=None,
        **policy,
    )
    # Each query is independent, so repeated safe ghosts are not deduplicated.
    preparation.prepare_temporary_generators(
        [(0., .75, .75)]*2, persistent=persistent, geometry=geometry, **policy,
    )
    assert scans[0].candidate_count == scans[1].candidate_count == 0
    with pytest.raises(pyvoro2.DuplicateError) as caught:
        preparation.prepare_temporary_generators(
            [(1.-2**-20, .25, .25)], persistent=persistent, geometry=geometry,
            **policy,
        )
    assert caught.value.kind == 'backend_safety'
    assert scans[-1].candidate_count == 1
    assert scans[-1].classification_candidate_count == 1
    assert np.array_equal(persistent.primary_points_cart, points)
    assert snapshot.params == (1., 2**32, 1., 0., 0., 1.)


# Predesignated modest backend-resolvable cohort, fixed before native execution.
@pytest.mark.parametrize('h', (2, 4, -4))
@pytest.mark.parametrize('left_handed', (False, True))
@pytest.mark.parametrize('mode', ('standard', 'power'))
def test_poor_representation_forward_smoke_preserves_wp2_backend(h, left_handed, mode):
    basis = shear(h)
    if left_handed:
        basis[2, 2] = -1.
    cell = PeriodicCell(basis)
    points = np.array(((.125, .25, .375), (.625, .75, .875), (.75, .125, .25)))
    kwargs = {'mode': mode, 'domain': cell}
    if mode == 'power':
        kwargs['weights'] = np.array((0., .01, -.01))
    result = pyvoro2.compute(points, **kwargs)
    assert {row['id'] for row in result.cells} == {0, 1, 2}
    assert sum(row['volume'] for row in result.cells) == pytest.approx(1., abs=1e-10)
    assert np.array_equal(cell.vectors, basis)
