"""Signed full-periodic 3D images; independent exact proof-box oracles."""
from fractions import Fraction as F
from itertools import product
import math

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api
from pyvoro2._internal import periodic_images as images
from pyvoro2._internal.duplicate_scanning import (
    scan_close_pairs, scan_cross_close_pairs,
)
from pyvoro2._internal.spatial.domain_geometry import DomainGeometry3D


SIGNS = list(product((-1., 1.), repeat=3))
SKEW = np.array([[-1., 0, 0], [.25, 1., 0], [.125, -.25, 1.]])


def exact_inverse(a):
    n = len(a)
    rows = [[F(float(x)) for x in row] + [F(i == j) for j in range(n)]
            for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[pivot], rows[j] = rows[j], rows[pivot]
        divisor = rows[j][j]
        rows[j] = [x / divisor for x in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[j])]
    return [row[n:] for row in rows]


def oracle(pi, pj, a, orientation):
    d = [F(float(y)) - F(float(x)) for x, y in zip(pi, pj)]
    inverse = exact_inverse(a)
    q = [sum(d[j] * inverse[j][i] for j in range(3)) for i in range(3)]
    # The zero-shift incumbent has norm <= ||d||_1. Every minimizer therefore
    # lies in these rational intervals. Unlike production, no seeded bound,
    # integer square root, cache or exact geometry helper is used.
    radius = sum(map(abs, d))
    bounds = [radius * sum(abs(inverse[j][i]) for j in range(3)) for i in range(3)]
    ranges = [range(math.ceil(-x - b), math.floor(-x + b) + 1)
              for x, b in zip(q, bounds)]
    records = []
    for shift in product(*ranges):
        v = tuple(d[j] + sum(shift[i] * F(float(a[i, j])) for i in range(3))
                  for j in range(3))
        records.append((sum(x*x for x in v), shift, v))
    best = min(row[0] for row in records)
    ties = sorted(row for row in records if row[0] == best)
    return ties[0 if orientation == 1 else -1], len(ties)


def solve(a, pi=(0, 0, 0), pj=(.375, -.25, .125), orientation=1, axes=None,
          image_search=1):
    return images.minimum_image_displacements(
        [pi], [pj], lattice_vectors=a,
        periodic_axes=(True,) * len(a) if axes is None else axes,
        tie_orientation=[orientation], image_search=image_search)


@pytest.mark.parametrize('a', [np.diag(np.array(s)*[1, 2, .5]) for s in SIGNS]
                         + [SKEW, SKEW[[1, 0, 2]]])
@pytest.mark.parametrize('pj', [(.375, -.25, .125), (.5, 0, 0), (1.25, -.5, .25)])
def test_signed_exact_minimum_keys_ties_and_pair_reversal(a, pj):
    (distance, shift, vector), ties = oracle((0, 0, 0), pj, a, 1)
    got = solve(a, pj=pj)
    assert got.certified
    assert tuple(got.shift[0]) == shift
    assert got.displacement[0].tolist() == list(map(float, vector))
    key = got.exact_distance_key[0]
    assert F(key.numerator, 2**key.denominator_exponent) == distance
    assert got.distance_squared[0] == float(distance)
    assert got.tie_count[0] == ties
    reversed_pair = solve(a, pi=pj, pj=(0, 0, 0), orientation=-1)
    np.testing.assert_array_equal(reversed_pair.shift, -got.shift)
    np.testing.assert_array_equal(reversed_pair.displacement, -got.displacement)
    for seed in (0, 2):
        np.testing.assert_array_equal(
            solve(a, pj=pj, image_search=seed).shift, got.shift)
    t = np.array([2, -1, 3])
    translated = solve(a, pj=np.asarray(pj) + t @ a)
    np.testing.assert_array_equal(translated.shift, got.shift - t)
    np.testing.assert_array_equal(translated.displacement, got.displacement)
    translated_first = solve(a, pi=t @ a, pj=pj)
    np.testing.assert_array_equal(translated_first.shift, got.shift + t)


@pytest.mark.parametrize('dimension', [2, 3])
def test_signed_admission_is_not_extended_to_planar_or_partial_axes(dimension):
    for axes in product((False, True), repeat=dimension):
        if dimension == 3 and all(axes):
            continue
        positive = np.diag(np.arange(1, dimension + 1, dtype=float))
        got = solve(positive, pi=[0]*dimension, pj=[.25]*dimension, axes=axes)
        assert got.certified and got.shift.tolist() == [[0]*dimension]
        for signs in product((-1., 1.), repeat=dimension):
            if all(s > 0 for s in signs):
                continue
            with pytest.raises(images.MinimumImageCertificationError) as error:
                solve(positive * np.array(signs), pi=[0]*dimension,
                      pj=[.25]*dimension, axes=axes)
            assert error.value.stage == 'basis_preparation'
        zero = positive.copy()
        zero[0, 0] = 0
        with pytest.raises(images.MinimumImageCertificationError):
            solve(zero, pi=[0]*dimension, pj=[0]*dimension, axes=axes)


def test_general_partial_and_planar_oblique_remain_unsupported():
    for a, axes in ((np.array([[1., .25], [0., 1.]]), (True, True)),
                    (SKEW, (True, False, True))):
        with pytest.raises(images.MinimumImageCertificationError) as error:
            solve(a, pi=[0]*len(a), pj=[0]*len(a), axes=axes)
        assert error.value.stage == 'basis_preparation'


def test_signed_image_budget_and_int64_refusals(monkeypatch):
    monkeypatch.setattr(images, '_MAX_EXACT_CANDIDATES_PER_PAIR', 1)
    with pytest.raises(images.MinimumImageCertificationError) as error:
        solve(SKEW, pj=(.5, .5, .5))
    assert error.value.stage == 'pair_candidate_budget'
    for value in (-2.**63 - 2048, 2.**63):
        with pytest.raises(images.MinimumImageCertificationError) as error:
            solve(np.diag([-1., 1., 1.]), pj=(value, 0, 0))
        assert error.value.stage == 'shift_range'
    got = solve(np.diag([-1., 1., 1.]), pj=(-2.**63, 0, 0))
    assert got.shift[0, 0] == -2**63


@pytest.mark.parametrize('a', [SKEW] + [np.diag(s) for s in SIGNS])
def test_signed_duplicate_buckets_and_cross_role_seams(a):
    origin = np.array([.25, -.5, .125])
    fractional = np.array([[2**-20, .25, .5], [1 - 2**-20, .25, .5],
                           [.5, .5, .5], [2 + 2**-20, -.75, 3.5]])
    points = origin + fractional @ a
    geometry = DomainGeometry3D(pv.PeriodicCell(a, origin))
    scan = scan_close_pairs(points, geometry=geometry, radius=1e-5,
                            inclusive_squared=1e-10, max_pairs=100)
    assert {(i, j) for i, j, _ in scan.pairs} == {(0, 1), (0, 3), (1, 3)}
    cross = scan_cross_close_pairs(points[:2], points[2:], geometry=geometry,
                                   radius=1e-5,
                                   inclusive_squared=1e-10, max_pairs=100)
    assert {(i, j) for i, j, _ in cross.pairs} == {(0, 1), (1, 1)}
    layout = images._exact_triclinic_bucket_layout(
        points, origin=origin, lattice_vectors=a, radius=1e-5)
    assert layout.keys[0] == layout.keys[3]
    for point, keys in zip(fractional, layout.keys):
        assert keys == tuple(math.floor((F(float(v)) % 1) * count)
                             for v, count in zip(point, layout.bins))


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_signed_mandatory_source_safety_cannot_be_disabled(
    monkeypatch, operation, mode,
):
    class ForbiddenCore:
        def __getattr__(self, name):
            raise AssertionError('unsafe native dispatch: ' + name)
    monkeypatch.setattr(api, '_core', ForbiddenCore())
    a = SKEW
    cell = pv.PeriodicCell(a)
    p = np.array([[.25, .25, .25]]) @ a
    q = p + a[0] + np.array([[2**-20, 0, 0]])
    args = (p, q) if operation == 'ghost_cells' else (np.vstack([p, q]),)
    if operation == 'locate':
        args += ([[.5, .5, .5]],)
    options = dict(mode=mode, duplicate_check='off', duplicate_wrap=False,
                   duplicate_threshold=1e-12)
    if mode == 'power':
        options['radii'] = [0.] * len(args[0])
        if operation == 'ghost_cells':
            options['ghost_radius'] = 0.
    with pytest.raises(pv.DuplicateError) as error:
        getattr(pv, operation)(*args, domain=cell, **options)
    assert error.value.kind == 'backend_safety'


def test_signed_batch_budget_and_seed_truncation_keep_failure_stages(monkeypatch):
    got = solve(SKEW, image_search=100)
    assert got.seed_truncated[0]
    assert got.seed_count[0] == images._MAX_SEED_CANDIDATES
    np.testing.assert_array_equal(got.shift, solve(SKEW, image_search=0).shift)
    monkeypatch.setattr(images, '_MAX_EXACT_CANDIDATES_PER_BATCH', 1)
    with pytest.raises(images.MinimumImageCertificationError) as error:
        solve(SKEW, pj=(.5, .5, .5))
    assert error.value.stage == 'batch_candidate_budget'


def test_exact_row_permutation_transports_unique_user_coefficients():
    from itertools import permutations
    pj = np.array([.375, -.25, 1.125])
    original = solve(SKEW, pj=pj)
    assert original.tie_count[0] == 1
    for order in permutations(range(3)):
        permuted = solve(SKEW[list(order)], pj=pj)
        np.testing.assert_array_equal(permuted.shift, original.shift[:, order])
        np.testing.assert_array_equal(permuted.displacement, original.displacement)


@pytest.mark.parametrize('a', [
    np.diag([-1., 0, 1]),
    [[-1., 0, 0], [-1., 0, 0], [0, 0, 1.]],
])
def test_full_periodic_singular_signed_basis_remains_refused(a):
    with pytest.raises(images.MinimumImageCertificationError) as error:
        solve(np.array(a))
    assert error.value.stage == 'basis_preparation'
