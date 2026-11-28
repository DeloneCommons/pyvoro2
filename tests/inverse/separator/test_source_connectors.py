"""Original-source geometry, with independent exact-operand oracles."""
from copy import copy, deepcopy
from dataclasses import replace
from fractions import Fraction as F
from itertools import permutations, product
import pickle

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell, compute
from pyvoro2.inverse.separator import resolve_separator_observations
from pyvoro2.planar import RectangularCell


def _affine(pi, pj, shift, basis):
    return np.array([
        float(F(float(y)) - F(float(x)) + sum(
            int(s) * F(float(row[axis])) for s, row in zip(shift, basis)))
        for axis, (x, y) in enumerate(zip(pi, pj))
    ])


@pytest.mark.parametrize('kind', ['orthorhombic', 'triclinic', 'planar'])
@pytest.mark.parametrize('origin', [0., 2.])
@pytest.mark.parametrize('translation', [0, 3, -2])
def test_source_snap_witness_mixed_rows(kind, origin, translation):
    dim = 2 if kind == 'planar' else 3
    bounds = ((origin, origin + 1),) * dim
    if kind == 'planar':
        domain = RectangularCell(bounds, periodic=(True, False))
    elif kind == 'orthorhombic':
        domain = OrthorhombicCell(bounds)
    else:
        domain = PeriodicCell(np.eye(3), origin=(origin,) * 3)
    points = np.full((2, dim), origin + 1/4)
    points[:, 0] = [origin + 2.**-41, origin + 1/2 + translation]
    shift = (-translation,) + (0,) * (dim - 1)
    reverse = tuple(-s for s in shift)
    rows = [(0, 1, .25), (1, 0, .75), (0, 1, .25, shift)]
    obs = resolve_separator_observations(points, rows, domain=domain)
    d = F(1, 2) - F(1, 2**41)
    assert (d - 1)**2 - d**2 == F(1, 2**40)
    expected = np.array([float(d)] + [0.] * (dim - 1))
    np.testing.assert_array_equal(obs.shifts, [shift, reverse, shift])
    np.testing.assert_array_equal(obs.delta, [expected, -expected, expected])
    np.testing.assert_array_equal(obs.explicit_shift, [False, False, True])
    np.testing.assert_array_equal(obs.input_index, [0, 1, 2])
    assert obs.distance2[0] == np.dot(expected, expected)
    assert obs.target_position[0] == .25 * obs.distance[0]


@pytest.mark.parametrize('order', list(permutations(range(3))))
@pytest.mark.parametrize('sign', [-1, 1])
def test_unique_source_minimizer_preserves_signed_basis_coefficients(order, sign):
    basis = np.eye(3)[list(order)]
    basis[0] *= sign
    points = np.array([[2.**-41, .25, .25], [.5, .25, .25]])
    lift = np.array([2, -1, 3])
    points[1] += lift @ basis
    obs = resolve_separator_observations(
        points, [(0, 1, .5), (1, 0, .5)], domain=PeriodicCell(basis))
    np.testing.assert_array_equal(obs.shifts, [-lift, lift])
    expected = _affine(points[0], points[1], -lift, basis)
    np.testing.assert_array_equal(obs.delta, [expected, -expected])
    assert expected[0] == .5 - 2.**-41


@pytest.mark.parametrize('kind', ['orthorhombic', 'triclinic', 'planar'])
@pytest.mark.parametrize('xi,xj,s,expected', [
    (.75, .25 - 2.**-54, 1, .49999999999999994),
    (0., -2.**53, 2**53 + 1, 1.),
    (.25, 2.**53, -2**53, -.25),
    (.125, 3.375, 2, 5.25),
])
def test_fixed_shift_affine_rounds_complete_expression(kind, xi, xj, s, expected):
    dim = 2 if kind == 'planar' else 3
    domain = (RectangularCell(((0., 1.),) * dim) if dim == 2 else
              OrthorhombicCell(((0., 1.),) * dim) if kind == 'orthorhombic'
              else PeriodicCell(np.eye(dim)))
    points = np.full((2, dim), .25)
    points[:, 0] = [xi, xj]
    shift = (s,) + (0,) * (dim - 1)
    obs = resolve_separator_observations(points, [
        (0, 1, .25, shift), (1, 0, .75, tuple(-v for v in shift)),
    ], domain=domain, image='given_only')
    oracle = _affine(points[0], points[1], shift, np.eye(dim))
    assert oracle[0] == expected
    np.testing.assert_array_equal(obs.delta, [oracle, -oracle])


def test_source_only_resolution_does_not_require_native_admission():
    domain = PeriodicCell(np.diag([2.**-60, 1., 1.]))
    points = np.array([[0., 0., 0.], [2.**-62, 0., 0.]])
    obs = resolve_separator_observations(points, [(0, 1, .5)], domain=domain)
    np.testing.assert_array_equal(obs.shifts, [[0, 0, 0]])
    np.testing.assert_array_equal(obs.delta, [[2.**-62, 0., 0.]])
    with pytest.raises(ValueError):
        compute(points, domain=domain)


def test_source_inference_binding_survives_copy_subset_and_pickle():
    domain = PeriodicCell(np.eye(3))
    points = np.array([[.125, .25, .25], [3.375, .25, .25]])
    obs = resolve_separator_observations(
        points, [(0, 1, .25), (1, 0, .75)], domain=domain, ids=[12, 20])
    for other in (copy(obs), deepcopy(obs), replace(obs),
                  pickle.loads(pickle.dumps(obs)), obs.subset([True, False])):
        assert other.shifts[0].tolist() == [-3, 0, 0]
        assert not other.explicit_shift.any()
        assert other.delta[0].tolist() == [.25, 0., 0.]
        assert not other.delta.flags.writeable
        assert other.to_records()[0]['row_id'] == obs.to_records()[0]['row_id']


def test_fixed_shift_needs_no_representable_translated_endpoint():
    # The translated endpoint exceeds binary64 range, but its full difference
    # cancels exactly. A tiny connector also keeps distance2 finite.
    big = float.fromhex('0x1.fffffffffffffp+1023')
    domain = PeriodicCell(((big, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    points = np.array([[big, 0., 0.], [1., .25, 0.]])
    obs = resolve_separator_observations(
        points, [(0, 1, .5, (1, 0, 0))], domain=domain)
    np.testing.assert_array_equal(obs.delta, [[1., .25, 0.]])


def test_fixed_shift_nonfinite_final_view_refuses():
    big = float.fromhex('0x1.fffffffffffffp+1023')
    domain = PeriodicCell(((big, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match='finite'):
        resolve_separator_observations(
            [[0., 0., 0.], [1., .25, 0.]],
            [(0, 1, .5, (2, 0, 0))], domain=domain)


def test_fixed_shift_oblique_affine_fraction_oracle():
    basis = np.array([[1., .1, 0.], [0., 2., .3], [.2, 0., 3.]])
    points = np.array([[.125, -.3, .75], [3.375, .9, -1.25]])
    shifts = list(product((-2, 0, 3), repeat=3))
    obs = resolve_separator_observations(
        points, [(0, 1, .5, s) for s in shifts], domain=PeriodicCell(basis))
    np.testing.assert_array_equal(obs.delta, [
        _affine(points[0], points[1], s, basis) for s in shifts])


def test_remapped_cached_observation_cannot_acquire_original_source_binding():
    from dataclasses import fields
    from pyvoro2.inverse.separator import (
        SeparatorObservations, fit_weights_from_separators,
    )

    domain = PeriodicCell(np.eye(3))
    source = np.array([[.125, .25, .25], [3.375, .25, .25]])
    primary = source.copy()
    primary[1, 0] -= 3
    old = resolve_separator_observations(primary, [(0, 1, .5)], domain=domain)
    # Direct rows remain valid without source provenance. Binding these old
    # primary-indexed rows to the original source must fail exactly.
    unbound = SeparatorObservations(**{
        f.name: getattr(old, f.name) for f in fields(old)})
    with pytest.raises(ValueError, match='exact geometry source.*shifts'):
        fit_weights_from_separators(source, unbound, domain=domain)
    corrected = resolve_separator_observations(source, [(0, 1, .5)], domain=domain)
    with pytest.raises(ValueError, match='exact geometry source.*shifts'):
        replace(corrected, shifts=np.zeros((1, 3), dtype=np.int64))


def test_canonical_distance_uses_rounded_connector_not_exact_distance_key():
    points = np.array([[.145, .14, .3], [2.18, .96, 1.2]])
    basis = np.array([[2., 0., 0.], [.25, 1., 0.], [.1, .2, 1.5]])
    obs = resolve_separator_observations(
        points, [(0, 1, .5)], domain=PeriodicCell(basis))
    exact_delta = [F(points[1, k]) - F(points[0, k])
                   - sum(F(basis[t, k]) for t in range(3)) for k in range(3)]
    rounded = np.array([float(v) for v in exact_delta])
    canonical = np.einsum('i,i->', rounded, rounded)
    assert canonical != float(sum(v*v for v in exact_delta))
    assert obs.distance2[0] == canonical
    assert obs.distance[0] == np.sqrt(canonical)
