"""R1 source-geometry oracles for bounded coupled upper-boundary snapping."""
from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2 as pv
from pyvoro2._internal.spatial.domain_geometry import _NativePeriodicSnapshot


ORIGIN = np.array([1.25, -.5, .125])
POINTS = ORIGIN + np.array([[.25, .5, .5], [.5, .5, .5], [.75, .5, .5]])
IDS = [11, 13, 17]
REPORTED = np.array([[0., -1., -1.], [1., 0., 0.], [-1., -1., 0.]])
ALTERNATIVE = np.array([[-1., 0., 0.], [0., 1., 1.], [1., 1., 0.]])
BASES = [pytest.param(a, id=name) for name, a in (
    ('reported', REPORTED), ('alternative', ALTERNATIVE),
    ('reported-left', np.diag([-1., 1., 1.]) @ REPORTED),
    ('alternative-left', np.diag([-1., 1., 1.]) @ ALTERNATIVE),
    ('row-swap-left', REPORTED[[1, 0, 2]]),
    ('row-cycle-right', REPORTED[[1, 2, 0]]),
)]


def remapper(cell, route):
    if route == 'public':
        return cell
    return _NativePeriodicSnapshot.from_raw(vectors=cell.vectors, origin=cell.origin)


@pytest.mark.parametrize('route', ['public', 'snapshot'])
@pytest.mark.parametrize('eps', [0., 2.**-38])
@pytest.mark.parametrize(('point', 'wrapped', 'shift', 'seam'), [
    ([1/8, 1/4, 1 - 2.**-40], [1/8, 1/4, 0], [1, 1, 1], 2),
    ([1/8, 1 - 2.**-40, 1/8], [1/8, 0, 1/8], [1, 1, 0], 1),
])
def test_coupled_upper_snap_preserves_dyadic_remainders(
    route, eps, point, wrapped, shift, seam,
):
    cell = pv.PeriodicCell.from_params(1, -1, 1, 0, -1, 1)
    adapter = remapper(cell, route)
    assert adapter.to_internal_params() == (1, -1, 1, 0, -1, 1)
    actual, shifts = adapter.remap_internal([point], eps=eps, return_shifts=True)
    # eps=0 leaves these already-primary points unchanged, without any snap.
    np.testing.assert_array_equal(actual, [wrapped if eps else point])
    np.testing.assert_array_equal(shifts, [shift if eps else [0, 0, 0]])
    # Independent rational reconstruction: only the intended seam loses 2^-40.
    # Non-seam x/y fractions must reconstruct EXACTLY for these dyadic inputs.
    for j in range(3):
        error = (F(float(actual[0, j]))
                 + sum(int(shifts[0, i]) * F(cell.vectors[i][j]) for i in range(3))
                 - F(point[j]))
        assert error == (F(1, 2**40) if eps and j == seam else 0)


@pytest.mark.parametrize('route', ['public', 'snapshot'])
def test_ordinary_endpoint_snaps_and_near_upper_coupled_overshoot(route):
    eps, delta = 2.**-38, 2.**-40
    # A z snap moves y just ABOVE 1, but still inside the epsilon neighborhood.
    cell = pv.PeriodicCell.from_params(1, -1, 1, 0, -delta, 1)
    adapter = remapper(cell, route)
    actual, shifts = adapter.remap_internal(
        [[1/8, 1 - delta/2, 1 - delta]], eps=eps, return_shifts=True)
    np.testing.assert_array_equal(actual, [[1/8, 0, 0]])
    np.testing.assert_array_equal(shifts, [[1, 1, 1]])
    residual = actual + shifts @ np.array(cell.vectors) - [1/8, 1 - delta/2, 1 - delta]
    np.testing.assert_array_equal(residual, [[0, -delta/2, delta]])
    # Normal uncoupled upper, lower and exactly-on-boundary behavior remains.
    adapter = remapper(pv.PeriodicCell(np.eye(3)), route)
    points = [[1 - delta, delta/2, 1/8], [1., 0., 1/8]]
    actual, shifts = adapter.remap_internal(points, eps=eps, return_shifts=True)
    np.testing.assert_array_equal(actual, [[0, 0, 1/8], [0, 0, 1/8]])
    np.testing.assert_array_equal(shifts, [[1, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize('a', BASES)
@pytest.mark.parametrize('route', ['public', 'snapshot'])
def test_integer_lattice_source_reconstruction_and_user_row_shifts(a, route):
    integers = a.astype(np.int64)
    np.testing.assert_array_equal(integers, a)
    assert abs(int(integers[0] @ np.cross(integers[1], integers[2]))) == 1
    adapter = remapper(pv.PeriodicCell(a, origin=ORIGIN), route)
    primary, shifts = adapter.remap_cart(POINTS, return_shifts=True)
    # These small integer unimodular bases generate precisely Z^3. Their
    # diagonals are <= 2 and default eps <= 2e-12. A 32e-12 envelope allows
    # coordinate snaps over three passes and ordinary-scale QR/matmul rounding;
    # it cannot hide the reported 1/4 displacement. This is a local bound.
    np.testing.assert_allclose(POINTS - primary - shifts @ a, 0, rtol=0, atol=32e-12)
    distances = np.linalg.norm(primary[:, None] - primary[None, :], axis=2)
    assert np.min(distances[np.triu_indices(3, k=1)]) > .2
    # Exact integer translations use the ORIGINAL row coefficients even for
    # sign-flipped and permuted bases; native-primary wrapping need not equal
    # wrapping to the user parallelepiped.
    translation = np.array([[2, -1, 3], [-1, 2, -2], [3, 1, -1]])
    translated, translated_shifts = adapter.remap_cart(
        POINTS + translation @ a, return_shifts=True)
    np.testing.assert_allclose(translated, primary, rtol=0, atol=32e-12)
    np.testing.assert_array_equal(translated_shifts - shifts, translation)


def power_options(kind):
    if kind == 'weights':
        return dict(mode='power', weights=[3/32, 0, 0])
    if kind == 'radii':
        return dict(mode='power', radii=[1/4, 0, 0])
    return dict(mode='standard')


def slab_oracle(kind):
    # Z^3 and equal y,z coordinates reduce geometry to periodic 1D power
    # intervals times unit area. Solve the three bisectors in rational math:
    # (x-left)^2-w_left = (x-right)^2-w_right.
    w = {'standard': F(0), 'weights': F(3, 32), 'radii': F(1, 16)}[kind]
    x, weights = [F(-1, 4), F(1, 4), F(1, 2), F(3, 4)], [F(0), w, F(0), F(0)]
    boundaries = [(left + right)/2 + (wl - wr)/(2*(right - left))
                  for left, right, wl, wr in zip(x, x[1:], weights, weights[1:])]
    volumes = [boundaries[1] - boundaries[0], boundaries[2] - boundaries[1],
               boundaries[0] + 1 - boundaries[2]]
    expected = {'standard': [F(3, 8), F(1, 4), F(3, 8)],
                'weights': [F(21, 32), F(1, 16), F(9, 32)],
                'radii': [F(9, 16), F(1, 8), F(5, 16)]}[kind]
    assert volumes == expected
    return dict(zip(IDS, map(float, volumes)))


@pytest.mark.parametrize('a', BASES)
@pytest.mark.parametrize('kind', ['standard', 'weights', 'radii'])
@pytest.mark.parametrize('output', ['cells', 'result'])
def test_integer_lattice_compute_has_source_analytic_id_volumes(a, kind, output):
    # At the defective HEAD tessellation_check='raise' succeeds for the main
    # witness despite WRONG ID-associated volumes. Coverage alone is no oracle.
    result = pv.compute(POINTS, domain=pv.PeriodicCell(a, origin=ORIGIN), ids=IDS,
                        output=output, return_face_shifts=True,
                        tessellation_check='raise', **power_options(kind))
    cells = result if output == 'cells' else result.cells
    actual = {cell['id']: cell['volume'] for cell in cells}
    assert actual == pytest.approx(slab_oracle(kind), rel=0, abs=2e-12)


@pytest.mark.parametrize('a', BASES)
@pytest.mark.parametrize('kind', ['standard', 'radii'])
def test_integer_lattice_locate_has_unique_source_owners(a, kind):
    xs = [F(9, 32), F(19, 32), F(7, 8)]
    w = {'standard': F(0), 'weights': F(3, 32), 'radii': F(1, 16)}[kind]
    expected = []
    for x in xs:
        # These x and sites lie in [0,1]; nearest images are among -1,0,1.
        distances = [min((x - p - s)**2 - weight for s in (-1, 0, 1))
                     for p, weight in zip([F(1, 4), F(1, 2), F(3, 4)], [w, 0, 0])]
        assert distances.count(min(distances)) == 1
        expected.append(IDS[distances.index(min(distances))])
    assert expected == IDS
    queries = ORIGIN + np.array([[float(x), .5, .5] for x in xs])
    result = pv.locate(POINTS, queries, domain=pv.PeriodicCell(a, origin=ORIGIN),
                       ids=IDS, return_owner_position=True, **power_options(kind))
    np.testing.assert_array_equal(result['found'], [True] * 3)
    np.testing.assert_array_equal(result['owner_id'], expected)
    np.testing.assert_allclose(result['owner_pos'], POINTS, rtol=0, atol=32e-12)


@pytest.mark.parametrize('a', BASES)
@pytest.mark.parametrize('count', [1, 2])
def test_integer_lattice_independent_ghost_has_analytic_volume(a, count):
    queries = ORIGIN + np.array([[9/32, .5, .5], [1 + 9/32, .5, .5]])[:count]
    cells = pv.ghost_cells(POINTS, queries,
                           domain=pv.PeriodicCell(a, origin=ORIGIN), ids=IDS)
    assert len(cells) == count
    for index, cell in enumerate(cells):
        assert cell['query_index'] == index
        # Each query's periodic image is inserted independently between
        # x=1/4 and x=1/2. The two ghost images must not compete with each other.
        # The two bisectors are separated by (1/2-1/4)/2 = 1/8, area=1.
        assert cell['volume'] == pytest.approx(1/8, rel=0, abs=2e-12)


@pytest.mark.parametrize('lift', [-2**48, 0, 7, 2**50])
@pytest.mark.parametrize('kind', ['standard', 'weights', 'radii'])
def test_source_lift_preserves_r1_volumes_and_actual_submissions(
    monkeypatch, lift, kind,
):
    import pyvoro2.api as api

    points = POINTS.copy()
    points[1] += lift * REPORTED[0]
    for j in range(3):
        assert F(points[1, j]) == F(POINTS[1, j]) + lift * F(REPORTED[0, j])
    native = api._require_core()
    name = ('compute_periodic_standard' if kind == 'standard'
            else 'compute_periodic_power')
    compute = getattr(native, name)
    prepare = api.prepare_generators
    calls, preparations = [], []

    def capture(*args):
        calls.append(args[0].copy())
        return compute(*args)

    def capture_preparation(*args, **kwargs):
        result = prepare(*args, **kwargs)
        preparations.append(result)
        return result

    monkeypatch.setattr(native, name, capture)
    monkeypatch.setattr(api, 'prepare_generators', capture_preparation)
    domain = pv.PeriodicCell(REPORTED, origin=ORIGIN)
    for source in (POINTS, points):
        cells = pv.compute(source, domain=domain, ids=IDS, output='cells',
                           **power_options(kind))
        assert {c['id']: c['volume'] for c in cells} == pytest.approx(
            slab_oracle(kind), rel=0, abs=2e-12)
    assert len(calls) == len(preparations) == 2
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(calls[1], preparations[1].native_points)
    np.testing.assert_array_equal(preparations[1].input_points_cart, points)
    np.testing.assert_array_equal(preparations[1].external_ids, IDS)
    np.testing.assert_array_equal(
        preparations[1].remap_shifts - preparations[0].remap_shifts,
        [[0, 0, 0], [lift, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize('lift', [0, 7, 2**50])
def test_source_lift_r1_bounded_owners_and_singleton_ghost(monkeypatch, lift):
    import pyvoro2.api as api

    domain = pv.PeriodicCell(REPORTED, origin=ORIGIN)
    points = POINTS.copy()
    points[1] += lift * REPORTED[0]
    # Independent exact periodic distances: at each original site its owner has
    # distance zero and every competing site has strictly positive distance.
    located = pv.locate(points, POINTS, domain=domain, ids=IDS,
                        return_owner_position=True)
    assert located['found'].tolist() == [True] * 3
    assert located['owner_id'].tolist() == IDS
    np.testing.assert_allclose(located['owner_pos'], POINTS, rtol=0, atol=32e-12)

    ghost = api._require_core().ghost_periodic_standard
    calls = []

    def capture(*args):
        calls.append((args[0].copy(), args[-1].copy()))
        return ghost(*args)

    monkeypatch.setattr(api._require_core(), 'ghost_periodic_standard', capture)
    query = ORIGIN + [9/32, .5, .5]
    for translation in (0, 2**50):
        translated = query + translation * REPORTED[0]
        assert all(F(translated[j]) == F(query[j]) + translation * F(REPORTED[0, j])
                   for j in range(3))
        cells = pv.ghost_cells(points, [translated], domain=domain, ids=IDS)
        assert len(cells) == 1
        assert cells[0]['volume'] == pytest.approx(1/8, rel=0, abs=2e-12)
        assert cells[0]['query_index'] == 0
        np.testing.assert_array_equal(cells[0]['query'], translated)
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
