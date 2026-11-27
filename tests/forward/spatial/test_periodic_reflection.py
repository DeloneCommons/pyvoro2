"""Compiled A2 reflection tests with independent cube/slab/owner oracles."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

import pyvoro2 as pv
from pyvoro2._internal.spatial.domain_geometry import _NativePeriodicSnapshot


REFLECT = np.diag([-1., 1., 1.])
ORIGIN = np.array([1.25, -.5, .125])
POINTS = np.array([[.25, .5, .5], [.75, .5, .5]])
FLAGS = list(product((False, True), repeat=3))


def domain(reflected):
    return pv.PeriodicCell(REFLECT if reflected else np.eye(3), ORIGIN)


def cart(points, reflected):
    return ORIGIN + np.asarray(points) @ (REFLECT if reflected else np.eye(3))


def options(kind):
    if kind == 'weights':
        return dict(mode='power', weights=[1/16, 0])
    if kind == 'radii':
        return dict(mode='power', radii=[.25, 0])
    return dict(mode='standard')


def cyclic(values):
    values = tuple(values)
    return min(values[i:] + values[:i] for i in range(len(values))) if values else ()


def assert_inward(cell):
    if cell.get('empty'):
        return
    vertices = np.asarray(cell['vertices'])
    # Mean of all vertices is strictly interior to these full-dimensional
    # convex cells. A power generator itself need not lie inside its cell.
    interior = vertices.mean(axis=0)
    signed_volume = 0.
    for face in cell['faces']:
        polygon = vertices[face['vertices']] - interior
        area2 = sum((np.cross(polygon[i], polygon[(i+1) % len(polygon)])
                     for i in range(len(polygon))), np.zeros(3))
        directed = area2 @ polygon.mean(axis=0)
        assert directed < 0
        signed_volume += directed / 6
    assert signed_volume == pytest.approx(-cell['volume'], abs=2e-13)


def assert_pair(right, left, *, shifts=False):
    rv, lv = np.asarray(right['vertices']), np.asarray(left['vertices'])
    expected = ORIGIN + (rv - ORIGIN) @ REFLECT
    mapping = []
    for vertex in expected:
        matches = np.flatnonzero(np.max(abs(lv - vertex), axis=1) < 3e-14)
        assert len(matches) == 1
        mapping.append(int(matches[0]))
    assert len(set(mapping)) == len(lv)
    for index, row in enumerate(right['adjacency']):
        assert cyclic(left['adjacency'][mapping[index]]) == cyclic(
            [mapping[v] for v in reversed(row)])
    left_faces = {cyclic(f['vertices']): f for f in left['faces']}
    assert len(left_faces) == len(right['faces'])
    for face in right['faces']:
        key = cyclic([mapping[v] for v in reversed(face['vertices'])])
        other = left_faces[key]
        assert other['adjacent_cell'] == face['adjacent_cell']
        if shifts:
            assert other['adjacent_shift'] == face['adjacent_shift']
    assert right['volume'] == pytest.approx(left['volume'], abs=2e-14)
    assert_inward(right)
    assert_inward(left)


@pytest.mark.parametrize('kind', ['standard', 'weights', 'radii'])
@pytest.mark.parametrize('output', ['cells', 'result'])
def test_compute_reflection_has_analytic_volumes_cycles_and_image_planes(kind, output):
    results = [pv.compute(cart(POINTS, reflect), domain=domain(reflect),
                          ids=[41, 73], output=output, return_face_shifts=True,
                          **options(kind)) for reflect in (False, True)]
    cells = [r if output == 'cells' else r.cells for r in results]
    volumes = [1/2, 1/2] if kind == 'standard' else [5/8, 3/8]
    weights = [0, 0] if kind == 'standard' else [1/16, 0]
    for reflect, side in enumerate(cells):
        by_id = {c['id']: c for c in side}
        assert set(by_id) == {41, 73}
        for index, cid in enumerate((41, 73)):
            cell = by_id[cid]
            assert cell['volume'] == pytest.approx(volumes[index], abs=2e-14)
            for face in cell['faces']:
                j = (41, 73).index(face['adjacent_cell'])
                neighbor = by_id[face['adjacent_cell']]
                lattice = np.array(domain(reflect).vectors)
                image = (np.array(neighbor['site'])
                         + np.array(face['adjacent_shift']) @ lattice)
                own = np.array(cell['site'])
                # Independently derived power bisector, in original source rows.
                direction = image - own
                offset = (image @ image - own @ own + weights[index] - weights[j]) / 2
                vertices = np.array(cell['vertices'])[face['vertices']]
                np.testing.assert_allclose(
                    vertices @ direction, offset, atol=3e-14, rtol=0)
    left = {c['id']: c for c in cells[1]}
    for cell in reversed(cells[0]):
        assert_pair(cell, left[cell['id']], shifts=True)
    if output == 'result':
        np.testing.assert_array_equal(results[1].sites, cart(POINTS, True))
        assert not results[1].sites.flags.writeable


@pytest.mark.parametrize('kind', ['standard', 'radii'])
@pytest.mark.parametrize('flags', FLAGS)
@pytest.mark.parametrize(('operation', 'output'), [
    ('compute', 'cells'), ('compute', 'result'), ('ghost_cells', None),
])
def test_reflected_native_output_flags_preserve_directed_cycles(
    kind, flags, operation, output,
):
    common = dict(domain=domain(True), ids=[41, 73], **options(kind))
    args = (cart(POINTS, True),)
    if operation == 'compute':
        common['output'] = output
    else:
        args += (cart([[.5, .5, .5]], True),)
        if kind == 'radii':
            common['ghost_radius'] = .125
    full = getattr(pv, operation)(*args, **common)
    chosen = getattr(pv, operation)(*args, **common,
                                    return_vertices=flags[0],
                                    return_adjacency=flags[1], return_faces=flags[2])
    if output == 'result':
        full, chosen = full.cells, chosen.cells
    assert len(chosen) == len(full)
    for got, expected in zip(chosen, full):
        assert got['volume'] == expected['volume']
        for key, enabled in zip(('vertices', 'adjacency', 'faces'), flags):
            assert (key in got) == enabled
            if enabled:
                # Same native invocation/indices, only extraction flags differ.
                assert got[key] == expected[key]
        assert_inward(expected)


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_compiled_locate_has_analytic_unique_owner_and_cartesian_position(mode):
    # Near the x=.5 boundary the nonzero weight changes the unique owner.
    query = [[.53125, .5, .5], [1.875, .5, .5]]
    expected_ids = [73, 73] if mode == 'standard' else [41, 73]
    for reflected in (False, True):
        opts = dict(mode=mode, radii=[.25, 0]) if mode == 'power' else {}
        got = pv.locate(cart(POINTS, reflected), cart(query, reflected),
                        domain=domain(reflected), ids=[41, 73],
                        return_owner_position=True, **opts)
        assert got['found'].tolist() == [True, True]
        assert got['owner_id'].tolist() == expected_ids
        owner = [.75, .5, .5] if mode == 'standard' else [.25, .5, .5]
        np.testing.assert_allclose(got['owner_pos'], cart(
            [owner, [1.75, .5, .5]], reflected), atol=1e-14, rtol=0)


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_compiled_ghost_batches_have_analytic_volumes_and_reflected_cycles(mode):
    # Equal radii retain ordinary Voronoi geometry: adding the body centre to
    # a cubic lattice gives two cells of volume 1/2. Queries are independent.
    points = [[.25, .25, .25]]
    queries = [[.75, .75, .75], [1.75, .75, .75]]
    opts = dict(mode=mode, radii=[.125], ghost_radius=.125) if mode == 'power' else {}
    sides = [pv.ghost_cells(cart(points, r), cart(queries, r), domain=domain(r),
                            ids=[41], **opts) for r in (False, True)]
    for r, cells in enumerate(sides):
        assert len(cells) == 2
        for i, cell in enumerate(cells):
            assert cell['query_index'] == i
            assert cell['volume'] == pytest.approx(.5, abs=2e-14)
            np.testing.assert_array_equal(cell['query'], cart(queries, r)[i])
            np.testing.assert_allclose(cell['site'], cart(
                [queries[0]], r)[0], atol=1e-14, rtol=0)
    for right, left in zip(*sides):
        assert_pair(right, left)


@pytest.mark.parametrize('include_empty', [False, True])
@pytest.mark.parametrize('output', ['cells', 'result'])
def test_reflected_empty_power_cells_preserve_ids_and_optional_geometry(
    include_empty, output,
):
    for reflected in (False, True):
        result = pv.compute(cart(POINTS, reflected), domain=domain(reflected),
                            ids=[41, 73], mode='power', weights=[4., 0],
                            include_empty=include_empty, output=output,
                            return_face_shifts=True)
        cells = result if output == 'cells' else result.cells
        assert {c['id'] for c in cells} == ({41, 73} if include_empty else {41})
        assert next(c for c in cells if c['id'] == 41)['volume'] == pytest.approx(1.)
        if include_empty:
            empty = next(c for c in cells if c['id'] == 73)
            assert empty['empty'] and empty['volume'] == 0
            for name in ('faces', 'vertices', 'adjacency'):
                assert empty[name] == []
        for cell in cells:
            assert_inward(cell)


def test_reporting_helper_reverses_each_present_cycle_exactly_once():
    # Analytic records isolate reporting from native traversal and output order.
    from pyvoro2.api import _transport_periodic_cells_to_cart_inplace
    snapshot = _NativePeriodicSnapshot.from_raw(vectors=REFLECT, origin=ORIGIN)
    cells = [dict(id=19, volume=.5, site=[.25, .5, .75],
                  faces=[dict(vertices=[2, 0, 1], adjacent_cell=23,
                              adjacent_shift=[-1, 2, 0])], adjacency=[[2, 1, 0]]),
             dict(id=23, volume=0., empty=True, vertices=[], faces=[], adjacency=[])]
    original = deepcopy(cells)
    _transport_periodic_cells_to_cart_inplace(cells, snapshot)
    assert cells[0]['faces'][0] == dict(vertices=[1, 0, 2], adjacent_cell=23,
                                        adjacent_shift=[-1, 2, 0])
    assert cells[0]['adjacency'] == [[0, 1, 2]]
    assert cells[0]['id'] == 19 and cells[0]['volume'] == .5
    assert 'vertices' not in cells[0]
    assert cells[1] == original[1]
    np.testing.assert_array_equal(cells[0]['site'], ORIGIN + [-.25, .5, .75])


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_compiled_missing_owner_sentinel_survives_reflection(mode):
    opts = dict(mode=mode, radii=[]) if mode == 'power' else {}
    result = pv.locate(np.empty((0, 3)), cart([[.5, .5, .5]], True),
                       domain=domain(True), blocks=(1, 1, 1),
                       return_owner_position=True, **opts)
    assert result['found'].tolist() == [False]
    assert result['owner_id'].tolist() == [-1]
    assert np.isnan(result['owner_pos']).all()


@pytest.mark.parametrize('flags', FLAGS)
@pytest.mark.parametrize('include_empty', [False, True])
def test_compiled_reflected_ghost_empty_power_cells(flags, include_empty):
    result = pv.ghost_cells(cart([[.25, .25, .25]], True),
                            cart([[.75, .75, .75]], True), domain=domain(True),
                            mode='power', radii=[2.], ghost_radius=0.,
                            include_empty=include_empty, return_vertices=flags[0],
                            return_adjacency=flags[1], return_faces=flags[2])
    if not include_empty:
        assert result == []
    else:
        assert len(result) == 1
        assert result[0]['empty'] and result[0]['volume'] == 0
        for field, present in zip(('vertices', 'adjacency', 'faces'), flags):
            assert (field in result[0]) == present
            if present:
                assert result[0][field] == []


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_compiled_skew_reflection_keeps_geometry_and_source_shift_rows(mode):
    a = np.array([[1., .25, 0], [0, 1., .125], [.125, 0, 1.]])
    points = np.array([[.25, .25, .25], [.6875, .3125, .375],
                       [.375, .8125, .625], [.875, .75, .9375]]) @ a
    opts = dict(mode=mode, weights=[0., 1/64, 1/32, 3/64]) if mode == 'power' else {}
    sides = [pv.compute(cart(points, r), domain=pv.PeriodicCell(
        a @ (REFLECT if r else np.eye(3)), ORIGIN), ids=[11, 13, 17, 19],
        output='cells', return_face_shifts=True, **opts) for r in (False, True)]
    for cells in sides:
        assert sum(c['volume'] for c in cells) == pytest.approx(257/256, abs=1e-13)
    by_id = {c['id']: c for c in sides[1]}
    for cell in sides[0]:
        assert_pair(cell, by_id[cell['id']], shifts=True)


def test_reflected_diagnostics_face_properties_and_normalization_consumers():
    cell = domain(True)
    cells = pv.compute(cart(POINTS, True), domain=cell, ids=[41, 73],
                       return_face_shifts=True, output='cells')
    diagnostic = pv.analyze_tessellation(cells, cell, expected_ids=[41, 73])
    assert diagnostic.ok and diagnostic.volume_ratio == pytest.approx(1.)
    pv.annotate_face_properties(cells, cell)
    for record in cells:
        for face in record['faces']:
            outward = np.array(face['centroid']) - record['site']
            assert np.dot(face['normal'], outward) > 0
            assert face['area'] > 0
    topology = pv.normalize_topology(cells, domain=cell, tol=1e-8)
    first, second = (next(c for c in topology.cells if c['id'] == cid)
                     for cid in (41, 73))
    first_faces = {tuple(f['adjacent_shift']): int(gid)
                   for f, gid in zip(first['faces'], first['face_global_id'])
                   if f['adjacent_cell'] == 73}
    second_faces = {tuple(f['adjacent_shift']): int(gid)
                    for f, gid in zip(second['faces'], second['face_global_id'])
                    if f['adjacent_cell'] == 41}
    assert first_faces[(-1, 0, 0)] == second_faces[(1, 0, 0)]
