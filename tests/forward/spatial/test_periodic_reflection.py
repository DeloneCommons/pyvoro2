from __future__ import annotations

from itertools import product

import numpy as np
import pytest

import pyvoro2
from pyvoro2.api import _transport_periodic_cells_to_cart_inplace
from pyvoro2._internal.spatial.domain_geometry import DomainGeometry3D


_FRACTIONAL = np.array(
    [
        (0.25, 0.25, 0.25),
        (0.6875, 0.3125, 0.375),
        (0.375, 0.8125, 0.625),
        (0.875, 0.75, 0.9375),
    ],
    dtype=np.float64,
)
_REFLECTION = np.diag((-1.0, 1.0, 1.0))
_OFFSET = np.array((1.0, 0.0, 0.0))


def _domains_and_points():
    right = pyvoro2.PeriodicCell(tuple(map(tuple, np.eye(3))))
    left = pyvoro2.PeriodicCell(
        tuple(map(tuple, _REFLECTION)), origin=tuple(_OFFSET)
    )
    return right, left, _FRACTIONAL, _OFFSET + _FRACTIONAL @ _REFLECTION


def test_transport_helper_has_native_independent_analytic_oracle() -> None:
    _right, left, _right_points, _left_points = _domains_and_points()
    snapshot = DomainGeometry3D(left).native_periodic_snapshot()
    cells = [{
        'id': 23,
        'site': [0.25, 0.5, 0.75],
        'volume': 0.125,
        'vertices': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'adjacency': [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        'faces': [
            {'vertices': [1, 2, 3], 'adjacent_cell': 29, 'token': 'first'},
            {'vertices': [0, 3, 2], 'adjacent_cell': 31, 'token': 'second'},
        ],
    }, {
        'id': 37,
        'site': [0.5, 0.5, 0.5],
        'volume': 0.0,
        'empty': True,
        'vertices': [],
        'adjacency': [],
        'faces': [],
    }]

    _transport_periodic_cells_to_cart_inplace(cells, snapshot)
    assert cells[0]['id'] == 23
    assert cells[0]['volume'] == 0.125
    assert cells[0]['faces'] == [
        {'vertices': [3, 2, 1], 'adjacent_cell': 29, 'token': 'first'},
        {'vertices': [2, 3, 0], 'adjacent_cell': 31, 'token': 'second'},
    ]
    assert cells[0]['adjacency'] == [
        [3, 2, 1], [2, 3, 0], [3, 1, 0], [1, 2, 0]
    ]
    np.testing.assert_array_equal(cells[0]['site'], [0.75, 0.5, 0.75])
    np.testing.assert_array_equal(
        cells[0]['vertices'],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
    )
    assert cells[1]['empty'] is True
    assert cells[1]['volume'] == 0.0
    assert cells[1]['faces'] == []


def test_transport_helper_reverses_cycles_without_vertex_output() -> None:
    _right, left, _right_points, _left_points = _domains_and_points()
    snapshot = DomainGeometry3D(left).native_periodic_snapshot()
    cells = [{
        'id': 41,
        'volume': 0.25,
        'site': [0.25, 0.5, 0.75],
        'adjacency': [[1, 2, 3], [0, 3, 2]],
        'faces': [
            {'vertices': [1, 2, 3], 'adjacent_cell': 43, 'token': 'first'},
            {'vertices': [0, 3, 2], 'adjacent_cell': 47, 'token': 'second'},
        ],
    }, {
        'id': 53,
        'volume': 0.5,
        'site': [0.5, 0.25, 0.125],
        'faces': [
            {'vertices': [4, 5, 6], 'adjacent_cell': 59, 'token': 'only'},
        ],
    }]

    _transport_periodic_cells_to_cart_inplace(cells, snapshot)

    assert cells[0]['id'] == 41
    assert cells[0]['volume'] == 0.25
    assert 'vertices' not in cells[0]
    np.testing.assert_array_equal(cells[0]['site'], [0.75, 0.5, 0.75])
    assert cells[0]['adjacency'] == [[3, 2, 1], [2, 3, 0]]
    assert cells[0]['faces'] == [
        {'vertices': [3, 2, 1], 'adjacent_cell': 43, 'token': 'first'},
        {'vertices': [2, 3, 0], 'adjacent_cell': 47, 'token': 'second'},
    ]
    assert cells[1]['id'] == 53
    assert cells[1]['volume'] == 0.5
    assert 'vertices' not in cells[1]
    assert 'adjacency' not in cells[1]
    np.testing.assert_array_equal(cells[1]['site'], [0.5, 0.25, 0.125])
    assert cells[1]['faces'] == [
        {'vertices': [6, 5, 4], 'adjacent_cell': 59, 'token': 'only'},
    ]


def _same_directed_cycle(actual, expected) -> bool:
    if len(actual) != len(expected):
        return False
    if not expected:
        return True
    doubled = list(expected) + list(expected)
    return any(
        list(actual) == doubled[start:start + len(expected)]
        for start in range(len(expected))
    )


def _reflected_vertex_map(right, left) -> np.ndarray:
    right_vertices = np.asarray(right['vertices'])
    left_vertices = np.asarray(left['vertices'])
    assert right_vertices.shape == left_vertices.shape
    if not right_vertices.size:
        return np.empty((0,), dtype=np.int64)

    expected = _OFFSET + right_vertices @ _REFLECTION
    unmatched = set(range(len(left_vertices)))
    mapping = []
    for point in expected:
        candidates = [
            index for index in unmatched
            if np.all(np.abs(left_vertices[index] - point) <= 3e-14)
        ]
        assert len(candidates) == 1
        matched = candidates[0]
        unmatched.remove(matched)
        mapping.append(matched)
    assert not unmatched
    return np.asarray(mapping, dtype=np.int64)


def _paired_cells(right_cells, left_cells):
    key = (
        'query_index'
        if all('query_index' in cell for cell in right_cells + left_cells)
        else 'id'
    )
    left_by_key = {int(cell[key]): cell for cell in left_cells}
    assert len(left_by_key) == len(left_cells)
    assert set(left_by_key) == {int(cell[key]) for cell in right_cells}
    return [(right, left_by_key[int(right[key])]) for right in right_cells]


def _ghost_neighbor_oracle(
    *, points, lattice_rows, ids, weights, query_weights
):
    """Identify persistent ghost faces from their independent bisector plane."""

    points = np.asarray(points)
    lattice_rows = np.asarray(lattice_rows)
    weights = np.asarray(weights)
    query_weights = np.asarray(query_weights)
    image_shifts = tuple(product(range(-2, 3), repeat=3))

    def identify(cell, face):
        vertices = np.asarray(cell['vertices'])[face['vertices']]
        site = np.asarray(cell['site'])
        query_weight = query_weights[int(cell['query_index'])]
        matches = []
        for generator_id, point, weight in zip(ids, points, weights):
            for shift in image_shifts:
                image = point + np.asarray(shift) @ lattice_rows
                direction = image - site
                plane_offset = (
                    image @ image - weight
                    - site @ site + query_weight
                ) / 2.0
                residual = np.max(
                    np.abs(vertices @ direction - plane_offset)
                )
                if residual <= 2e-12:
                    matches.append(int(generator_id))
        # This dyadic fixture has one unique periodic generator image for each
        # persistent face and no match for its backend-incidental ghost face.
        assert len(matches) <= 1
        return None if not matches else matches[0]

    return identify


def _assert_reflection_covariance(right_cells, left_cells, *, vertices, adjacency,
                                  faces, persistent_neighbor_ids,
                                  neighbor_oracles=None):
    assert len(right_cells) == len(left_cells)
    for right, left in _paired_cells(right_cells, left_cells):
        assert right['id'] == left['id']
        assert right['volume'] == pytest.approx(left['volume'], rel=2e-14, abs=2e-14)
        np.testing.assert_allclose(
            left['site'], _OFFSET + np.asarray(right['site']) @ _REFLECTION,
            rtol=0, atol=2e-14,
        )
        if 'query' in right:
            np.testing.assert_allclose(
                left['query'],
                _OFFSET + np.asarray(right['query']) @ _REFLECTION,
                rtol=0,
                atol=0,
            )
        vertex_map = None
        if vertices:
            vertex_map = _reflected_vertex_map(right, left)
        if adjacency:
            if vertex_map is None:
                assert sorted(map(len, left['adjacency'])) == sorted(
                    map(len, right['adjacency'])
                )
            else:
                assert len(left['adjacency']) == len(right['adjacency'])
                for right_index, left_index in enumerate(vertex_map):
                    expected = [
                        int(vertex_map[index])
                        for index in reversed(right['adjacency'][right_index])
                    ]
                    assert _same_directed_cycle(
                        left['adjacency'][left_index], expected
                    )
        if faces:
            assert len(right['faces']) == len(left['faces'])
            if vertex_map is None:
                assert sorted(map(lambda face: len(face['vertices']),
                                  left['faces'])) == sorted(
                    map(lambda face: len(face['vertices']), right['faces'])
                )
                right_neighbors = sorted(
                    face['adjacent_cell'] for face in right['faces']
                )
                left_neighbors = sorted(
                    face['adjacent_cell'] for face in left['faces']
                )
                assert all(
                    neighbor in persistent_neighbor_ids
                    for neighbor in right_neighbors + left_neighbors
                )
                assert left_neighbors == right_neighbors
                continue

            unmatched_faces = set(range(len(left['faces'])))
            for right_face in right['faces']:
                expected = [
                    int(vertex_map[index])
                    for index in reversed(right_face['vertices'])
                ]
                candidates = [
                    index for index in unmatched_faces
                    if _same_directed_cycle(
                        left['faces'][index]['vertices'], expected
                    )
                ]
                assert len(candidates) == 1
                matched = candidates[0]
                unmatched_faces.remove(matched)
                right_neighbor = right_face['adjacent_cell']
                left_face = left['faces'][matched]
                left_neighbor = left_face['adjacent_cell']
                if neighbor_oracles is None:
                    assert right_neighbor in persistent_neighbor_ids
                    assert left_neighbor == right_neighbor
                else:
                    right_expected = neighbor_oracles[0](right, right_face)
                    left_expected = neighbor_oracles[1](left, left_face)
                    assert left_expected == right_expected
                    if right_expected is not None:
                        assert right_neighbor == right_expected
                        assert left_neighbor == right_expected
            assert not unmatched_faces


@pytest.mark.parametrize('mode', ['standard', 'power'])
@pytest.mark.parametrize(
    ('vertices', 'adjacency', 'faces'),
    [(True, True, True), (False, True, True), (False, False, True)],
)
def test_compute_reflection_transport_covers_output_flags(
    mode, vertices, adjacency, faces
) -> None:
    right, left, right_points, left_points = _domains_and_points()
    kwargs = {'weights': (0.01, 0.02, 0.03, 0.04)} if mode == 'power' else {}
    ids = (11, 13, 17, 19)
    common = dict(
        mode=mode,
        output='cells',
        ids=ids,
        return_vertices=vertices,
        return_adjacency=adjacency,
        return_faces=faces,
        **kwargs,
    )
    right_cells = pyvoro2.compute(right_points, domain=right, **common)
    left_cells = pyvoro2.compute(left_points, domain=left, **common)
    _assert_reflection_covariance(
        right_cells, left_cells, vertices=vertices,
        adjacency=adjacency, faces=faces,
        persistent_neighbor_ids=frozenset(ids),
    )


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_ghost_reflection_transport_preserves_ids_and_cycles(mode) -> None:
    right, left, right_points, left_points = _domains_and_points()
    query_fractional = np.array(
        ((0.3125, 0.4375, 0.5625), (0.9375, 0.125, 0.625))
    )
    right_queries = query_fractional
    left_queries = _OFFSET + query_fractional @ _REFLECTION
    weights = (0.01, 0.02, 0.03, 0.04)
    query_weights = (0.015, 0.025)
    kwargs = (
        {
            'weights': weights,
            'ghost_weights': query_weights,
        }
        if mode == 'power'
        else {}
    )
    ids = (11, 13, 17, 19)
    common = dict(mode=mode, ids=ids, **kwargs)
    right_cells = pyvoro2.ghost_cells(
        right_points, right_queries, domain=right, **common
    )
    left_cells = pyvoro2.ghost_cells(
        left_points, left_queries, domain=left, **common
    )
    _assert_reflection_covariance(
        right_cells, left_cells, vertices=True, adjacency=True, faces=True,
        persistent_neighbor_ids=frozenset(ids),
        neighbor_oracles=(
            _ghost_neighbor_oracle(
                points=right_points,
                lattice_rows=right.vectors,
                ids=ids,
                weights=weights if mode == 'power' else np.zeros(4),
                query_weights=(
                    query_weights if mode == 'power' else np.zeros(2)
                ),
            ),
            _ghost_neighbor_oracle(
                points=left_points,
                lattice_rows=left.vectors,
                ids=ids,
                weights=weights if mode == 'power' else np.zeros(4),
                query_weights=(
                    query_weights if mode == 'power' else np.zeros(2)
                ),
            ),
        ),
    )
    for right_cell, left_cell in _paired_cells(right_cells, left_cells):
        assert right_cell['query_index'] == left_cell['query_index']


def test_structured_power_result_transports_actual_empty_cells() -> None:
    right, left, right_points, left_points = _domains_and_points()
    ids = (11, 13, 17, 19)
    common = dict(
        mode='power',
        weights=(100.0, 0.0, 0.0, 0.0),
        output='result',
        include_empty=True,
        ids=ids,
    )
    right_result = pyvoro2.compute(right_points, domain=right, **common)
    left_result = pyvoro2.compute(left_points, domain=left, **common)

    np.testing.assert_array_equal(right_result.ids, left_result.ids)
    np.testing.assert_array_equal(right_result.empty_mask, [False, True, True, True])
    np.testing.assert_array_equal(left_result.empty_mask, right_result.empty_mask)
    np.testing.assert_allclose(
        left_result.sites,
        _OFFSET + right_result.sites @ _REFLECTION,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        left_result.cell_measures,
        right_result.cell_measures,
        rtol=0,
        atol=2e-14,
    )
    _assert_reflection_covariance(
        right_result.cells,
        left_result.cells,
        vertices=True,
        adjacency=True,
        faces=True,
        persistent_neighbor_ids=frozenset(ids),
    )


def _assert_inward_cycles_and_signed_volume(cells) -> None:
    for cell in cells:
        if cell.get('empty', False):
            continue
        vertices = np.asarray(cell['vertices'])
        site = np.asarray(cell['site'])
        signed_six_volume = 0.0
        for face in cell['faces']:
            polygon = vertices[face['vertices']] - site
            area_twice = sum(
                (np.cross(polygon[index], polygon[(index + 1) % len(polygon)])
                 for index in range(len(polygon))),
                np.zeros(3),
            )
            interior_witness = float(np.dot(area_twice, polygon.mean(axis=0)))
            assert interior_witness < 0.0
            signed_six_volume += interior_witness
        assert signed_six_volume / 6.0 == pytest.approx(
            -cell['volume'], rel=3e-13, abs=3e-13
        )


def test_reflected_face_cycles_give_reflected_cross_products() -> None:
    right, left, right_points, left_points = _domains_and_points()
    right_cells = pyvoro2.compute(right_points, domain=right, output='cells')
    left_cells = pyvoro2.compute(left_points, domain=left, output='cells')
    # Deliberately perturb incidental native result ordering. Correspondence
    # must come from public cell identity and reflected geometry instead.
    left_cells.reverse()
    for cell in left_cells:
        cell['faces'].reverse()

    right_cell, left_cell = next(
        pair for pair in _paired_cells(right_cells, left_cells)
        if pair[0]['id'] == 0
    )
    vertex_map = _reflected_vertex_map(right_cell, left_cell)
    right_face = next(face for face in right_cell['faces']
                      if len(face['vertices']) >= 3)
    expected_cycle = [
        int(vertex_map[index]) for index in reversed(right_face['vertices'])
    ]
    matching_faces = [
        face for face in left_cell['faces']
        if _same_directed_cycle(face['vertices'], expected_cycle)
    ]
    assert len(matching_faces) == 1
    left_face = matching_faces[0]
    rv = np.asarray(right_cell['vertices'])[right_face['vertices']]
    lv = np.asarray(left_cell['vertices'])[left_face['vertices']]
    right_cross = sum(
        (np.cross(rv[index], rv[(index + 1) % len(rv)])
         for index in range(len(rv))),
        np.zeros(3),
    )
    left_cross = sum(
        (np.cross(lv[index], lv[(index + 1) % len(lv)])
         for index in range(len(lv))),
        np.zeros(3),
    )
    np.testing.assert_allclose(
        left_cross, right_cross @ _REFLECTION, rtol=0, atol=2e-13
    )
    _assert_inward_cycles_and_signed_volume(right_cells + left_cells)
