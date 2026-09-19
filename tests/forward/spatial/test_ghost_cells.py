from functools import lru_cache
from itertools import product

import numpy as np
import pytest

import pyvoro2


def test_ghost_cells_box_standard_inside_and_outside_rejected() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    queries = np.array([[0.5, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert isinstance(cells, list)
    assert len(cells) == 1

    c0 = cells[0]

    assert c0['query_index'] == 0
    assert np.allclose(np.asarray(c0['query'], dtype=float), queries[0])
    assert c0['empty'] is False
    # Expected half-space volume: x in [0.25, 1] -> 0.75 * 2 * 2 = 3.0
    assert abs(float(c0['volume']) - 3.0) < 1e-6

    with pytest.raises(ValueError, match='outside the native primary domain'):
        pyvoro2.ghost_cells(
            pts,
            np.array([[2.0, 0.0, 0.0]]),
            domain=box,
        )


def test_ghost_cells_box_power_volume_shift() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    radii = np.array([0.3], dtype=float)
    queries = np.array([[0.5, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        mode='power',
        radii=radii,
        ghost_radii=0.0,
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False

    # Power bisector between (0, r0) and (0.5, rg=0):
    # x = 0.25 + r0^2 - rg^2 = 0.25 + 0.09 = 0.34
    # Ghost region x in [0.34, 1] => (1-0.34)*4 = 2.64
    assert abs(float(c['volume']) - 2.64) < 1e-4


def test_ghost_cells_ids_remap_faces() -> None:
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    ids = [123]
    queries = np.array([[0.5, 0.0, 0.0]], dtype=float)
    box = pyvoro2.Box(bounds=((-1, 1), (-1, 1), (-1, 1)))

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=box,
        ids=ids,
        mode='standard',
        return_vertices=True,
        return_adjacency=False,
        return_faces=True,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False
    neigh = [int(f['adjacent_cell']) for f in c.get('faces', [])]
    assert 123 in neigh


def test_ghost_cells_orthorhombic_periodic_query_wrapping() -> None:
    domain = pyvoro2.OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False, False),
    )
    pts = np.array([[0.1, 0.1, 0.1]], dtype=float)
    queries = np.array([[1.2, 0.1, 0.1]], dtype=float)  # wraps to x=0.2

    cells = pyvoro2.ghost_cells(
        pts,
        queries,
        domain=domain,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        include_empty=True,
    )

    assert len(cells) == 1
    c = cells[0]
    assert c['empty'] is False

    site = np.asarray(c['site'], dtype=float)
    assert abs(site[0] - 0.2) < 1e-12
    assert abs(site[1] - 0.1) < 1e-12
    assert abs(site[2] - 0.1) < 1e-12


_CUBIC_BASES = [
    pytest.param(np.eye(3), id='I-right'),
    pytest.param(np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]),
                 id='T-right'),
    pytest.param(np.array([[-1., -1., 0.], [0., 1., 0.], [0., 0., 1.]]),
                 id='T-left'),
]


def _common_radii(weights, ghost_weights):
    # Independent WP1 representation oracle: gauge the COMPLETE weight set.
    combined = np.concatenate((weights, ghost_weights))
    radii = np.sqrt(combined - np.min(combined))
    return dict(radii=radii[:len(weights)], ghost_radii=radii[len(weights):])


@lru_cache(maxsize=2)
def _independent_cubic_ghost_volume(power):
    """Compute the fixture cell directly from complete power half-spaces."""
    spatial = pytest.importorskip('scipy.spatial')
    points = np.array([
        [1 / 16, 3 / 16, 5 / 16],
        [5 / 8, 7 / 16, 1 / 8],
        [1 / 4, 3 / 4, 11 / 16],
    ])
    query = np.array([-3 / 16, -3 / 16, 3 / 8])
    weights = np.array([0.0, 1 / 32, -1 / 64]) if power else np.zeros(3)
    query_weight = 3 / 64 if power else 0.0

    def bounded_volume(image_radius):
        halfspaces = []
        for shift_tuple in product(
            range(-image_radius, image_radius + 1), repeat=3
        ):
            shift = np.asarray(shift_tuple, dtype=float)
            for point, weight in zip(points, weights):
                image = point + shift
                normal = 2 * (image - query)
                upper = (
                    image @ image - weight
                    - query @ query + query_weight
                )
                halfspaces.append(np.r_[normal, -upper])
            if shift_tuple != (0, 0, 0):
                image = query + shift
                normal = 2 * (image - query)
                upper = image @ image - query @ query
                halfspaces.append(np.r_[normal, -upper])
        intersections = spatial.HalfspaceIntersection(
            np.asarray(halfspaces),
            query,
        ).intersections
        return spatial.ConvexHull(intersections).volume

    # Ghost self-images confine the cell to a unit cube about the query. The
    # radius-two image set contains every generator image that can cut that
    # cube; radius three independently confirms that the oracle is stable.
    radius_two = bounded_volume(2)
    radius_three = bounded_volume(3)
    assert radius_three == pytest.approx(radius_two, abs=1e-14, rel=0)
    return radius_two


def _assert_physical_ghost_equal(actual, expected):
    """Compare physical geometry without asserting future WP7 neighbor IDs."""
    assert actual['empty'] == expected['empty']
    assert actual['volume'] == pytest.approx(
        expected['volume'], abs=1e-12, rel=1e-12,
    )
    np.testing.assert_allclose(actual['site'], expected['site'], atol=1e-12,
                               rtol=0)
    if 'vertices' not in expected:
        assert 'vertices' not in actual
        return
    left = np.asarray(actual['vertices']).reshape(-1, 3)
    right = np.asarray(expected['vertices']).reshape(-1, 3)
    assert left.shape == right.shape
    # These fixtures have distinct vertices; match bijectively without relying
    # on native vertex/face traversal order or rounded lexicographic sorting.
    matches = np.all(np.abs(left[:, None] - right[None, :]) < 1e-12, axis=2)
    np.testing.assert_array_equal(matches.sum(axis=0), np.ones(len(right)))
    np.testing.assert_array_equal(matches.sum(axis=1), np.ones(len(left)))
    vertex_map = np.argmax(matches, axis=1) if len(left) else []
    mapped_faces = sorted(tuple(sorted(vertex_map[v] for v in f['vertices']))
                          for f in actual['faces'])
    expected_faces = sorted(tuple(sorted(f['vertices']))
                            for f in expected['faces'])
    assert mapped_faces == expected_faces
    mapped_edges = sorted((int(vertex_map[i]), int(vertex_map[j]))
                          for i, row in enumerate(actual['adjacency'])
                          for j in row)
    expected_edges = sorted((i, j)
                            for i, row in enumerate(expected['adjacency'])
                            for j in row)
    assert mapped_edges == expected_edges


@pytest.mark.parametrize('basis', _CUBIC_BASES)
@pytest.mark.parametrize('geometry', [False, True], ids=['volume-only', 'geometry'])
@pytest.mark.parametrize('family', ['standard', 'equal-weights', 'weights', 'radii'])
def test_periodic_ghost_batch_analytic_volume(basis, geometry, family):
    points = np.array([[0.125, 0.125, 0.125]])
    queries = np.array([[0.375, 0.125, 0.125], [0.875, 0.125, 0.125]])
    options = dict(domain=pyvoro2.PeriodicCell(basis), blocks=(1, 1, 1),
                   return_vertices=geometry, return_faces=geometry,
                   return_adjacency=geometry)
    if family == 'standard':
        expected_volume = 0.5
    else:
        options['mode'] = 'power'
        weights = np.array([-0.125])
        ghost_weights = np.full(2, -0.125 if family == 'equal-weights' else 0.25)
        if family == 'radii':
            options.update(_common_radii(weights, ghost_weights))
        else:
            options.update(weights=weights, ghost_weights=ghost_weights[0])
        expected_volume = 0.5 if family == 'equal-weights' else 1.0
    # Equal weights bisect the two cyclic intervals: total width 1/2.
    # Otherwise a=1/4, and wg-wp=3/8 > a(1-a)=3/16: the ghost orbit
    # strictly dominates. Only self images bound it, so volume = covolume = 1.
    for order in ([0, 1], [1, 0]):
        ordered = queries[order]
        batch = pyvoro2.ghost_cells(points, ordered, **options)
        assert len(batch) == 2
        for i, cell in enumerate(batch):
            single_options = dict(options)
            if family == 'radii':
                single_options['ghost_radii'] = options['ghost_radii'][i]
            single = pyvoro2.ghost_cells(
                points, ordered[i:i + 1], **single_options,
            )[0]
            assert not single['empty']
            assert single['volume'] == pytest.approx(
                expected_volume, abs=1e-12, rel=0,
            )
            assert cell['volume'] == pytest.approx(
                expected_volume, abs=1e-12, rel=0,
            )
            assert cell['query_index'] == i
            np.testing.assert_array_equal(cell['query'], ordered[i])
            _assert_physical_ghost_equal(cell, single)


@pytest.mark.parametrize('basis', _CUBIC_BASES)
@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize('blocks', [(1, 1, 1), (2, 3, 2)])
def test_periodic_ghost_permutation_matches_independent_geometry(basis, mode, blocks):
    points = np.array([[.125, .125, .125], [.625, .375, .25], [.25, .75, .625]])
    queries = np.array([[.375, .125, .125], [.875, .125, .125], [.5, .625, .875]])
    weights = np.array([0., .015625, .03125])
    ghost_weights = np.array([-.03125, .046875, .015625])
    radii = _common_radii(weights, ghost_weights)
    options = dict(domain=pyvoro2.PeriodicCell(basis), blocks=blocks, mode=mode)
    # Hold the complete batch's explicit radii fixed in every reference call.
    # This also exercises a ghost (not a persistent site) setting the gauge.
    singles = []
    for i, query in enumerate(queries):
        power = (dict(radii=radii['radii'], ghost_radii=radii['ghost_radii'][i])
                 if mode == 'power' else {})
        singles.append(pyvoro2.ghost_cells(
            points, query[None], **options, **power,
        )[0])
    assert all(not cell['empty'] and cell['volume'] > 0 for cell in singles)
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]):
        families = ([
            dict(weights=weights, ghost_weights=ghost_weights[order]),
            dict(radii=radii['radii'], ghost_radii=radii['ghost_radii'][order]),
        ] if mode == 'power' else [{}])
        for power in families:
            batch = pyvoro2.ghost_cells(points, queries[order], **options, **power)
            assert len(batch) == len(queries)
            for i, cell in enumerate(batch):
                assert cell['query_index'] == i
                np.testing.assert_array_equal(cell['query'], queries[order[i]])
                _assert_physical_ghost_equal(cell, singles[order[i]])


@pytest.mark.parametrize('basis', _CUBIC_BASES)
@pytest.mark.parametrize('blocks', [(1, 1, 1), (2, 3, 2)])
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_periodic_singleton_ghost_preserves_cubic_physics(
    basis,
    blocks,
    family,
):
    points = np.array([
        [1 / 16, 3 / 16, 5 / 16],
        [5 / 8, 7 / 16, 1 / 8],
        [1 / 4, 3 / 4, 11 / 16],
    ])
    query = np.array([[-3 / 16, -3 / 16, 3 / 8]])
    options = {}
    if family != 'standard':
        weights = np.array([0.0, 1 / 32, -1 / 64])
        ghost_weights = np.array([3 / 64])
        options['mode'] = 'power'
        if family == 'weights':
            options.update(
                weights=weights,
                ghost_weights=ghost_weights,
            )
        else:
            options.update(_common_radii(weights, ghost_weights))

    cell = pyvoro2.ghost_cells(
        points,
        query,
        domain=pyvoro2.PeriodicCell(basis),
        blocks=blocks,
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
        **options,
    )[0]

    expected = _independent_cubic_ghost_volume(family != 'standard')
    assert not cell['empty']
    assert cell['volume'] == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert 'vertices' not in cell
    assert 'adjacency' not in cell
    assert 'faces' not in cell


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_native_periodic_ghost_batch_isolation(mode):
    # Direct binding oracle: no Python packaging or boundary reconstruction.
    from pyvoro2 import _core

    points = np.array([[.125, .125, .125]])
    queries = np.array([[.375, .125, .125], [.875, .125, .125], [.625, .125, .125]])
    options = dict(points=points, ids=np.array([0], dtype=np.int32),
                   cell_params=(1., 0., 1., 0., 0., 1.), blocks=(1, 1, 1),
                   init_mem=8, opts=(False, False, False))
    if mode == 'power':
        call = _core.ghost_periodic_power
        options.update(radii=np.array([0.]))
        # Advantage 3/8 exceeds a(1-a) <= 1/4 for ALL three queries.
        expected_volume = 1.0
    else:
        call = _core.ghost_periodic_standard
        expected_volume = 0.5
    for order in ([0, 1, 2], [2, 1, 0]):
        power = dict(ghost_radii=np.full(3, np.sqrt(.375))) if mode == 'power' else {}
        batch = call(queries=queries[order], **options, **power)
        assert len(batch) == 3
        for i, cell in enumerate(batch):
            single_power = (dict(ghost_radii=power['ghost_radii'][i:i + 1])
                            if mode == 'power' else {})
            single = call(queries=queries[order[i:i + 1]], **options, **single_power)[0]
            assert single['volume'] == pytest.approx(expected_volume, abs=1e-12, rel=0)
            assert cell['volume'] == pytest.approx(expected_volume, abs=1e-12, rel=0)
            _assert_physical_ghost_equal(cell, single)
