from __future__ import annotations

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
            {'vertices': [1, 2, 3], 'adjacent_cell': 29},
            {'vertices': [0, 3, 2], 'adjacent_cell': 31},
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
    assert cells[0]['faces'][0] == {
        'vertices': [3, 2, 1], 'adjacent_cell': 29
    }
    assert cells[0]['adjacency'][0] == [3, 2, 1]
    np.testing.assert_array_equal(cells[0]['site'], [0.75, 0.5, 0.75])
    np.testing.assert_array_equal(
        cells[0]['vertices'],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
    )
    assert cells[1]['empty'] is True
    assert cells[1]['volume'] == 0.0
    assert cells[1]['faces'] == []


def _assert_reflection_covariance(right_cells, left_cells, *, vertices, adjacency,
                                  faces):
    assert len(right_cells) == len(left_cells)
    for right, left in zip(right_cells, left_cells):
        assert right['id'] == left['id']
        assert right['volume'] == pytest.approx(left['volume'], rel=2e-14, abs=2e-14)
        np.testing.assert_allclose(
            left['site'], _OFFSET + np.asarray(right['site']) @ _REFLECTION,
            rtol=0, atol=2e-14,
        )
        if vertices:
            if right['vertices']:
                np.testing.assert_allclose(
                    left['vertices'],
                    _OFFSET + np.asarray(right['vertices']) @ _REFLECTION,
                    rtol=0, atol=3e-14,
                )
            else:
                assert left['vertices'] == []
        if adjacency:
            assert left['adjacency'] == [
                list(reversed(cycle)) for cycle in right['adjacency']
            ]
        if faces:
            assert len(right['faces']) == len(left['faces'])
            for right_face, left_face in zip(right['faces'], left['faces']):
                assert right_face['adjacent_cell'] == left_face['adjacent_cell']
                assert left_face['vertices'] == list(
                    reversed(right_face['vertices'])
                )


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
    common = dict(
        mode=mode,
        output='cells',
        ids=(11, 13, 17, 19),
        return_vertices=vertices,
        return_adjacency=adjacency,
        return_faces=faces,
        **kwargs,
    )
    right_cells = pyvoro2.compute(right_points, domain=right, **common)
    left_cells = pyvoro2.compute(left_points, domain=left, **common)
    _assert_reflection_covariance(
        right_cells, left_cells, vertices=vertices,
        adjacency=adjacency, faces=faces
    )


@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_ghost_reflection_transport_preserves_ids_and_cycles(mode) -> None:
    right, left, right_points, left_points = _domains_and_points()
    query_fractional = np.array(
        ((0.3125, 0.4375, 0.5625), (0.9375, 0.125, 0.625))
    )
    right_queries = query_fractional
    left_queries = _OFFSET + query_fractional @ _REFLECTION
    kwargs = (
        {
            'weights': (0.01, 0.02, 0.03, 0.04),
            'ghost_weights': (0.015, 0.025),
        }
        if mode == 'power'
        else {}
    )
    common = dict(mode=mode, ids=(11, 13, 17, 19), **kwargs)
    right_cells = pyvoro2.ghost_cells(
        right_points, right_queries, domain=right, **common
    )
    left_cells = pyvoro2.ghost_cells(
        left_points, left_queries, domain=left, **common
    )
    _assert_reflection_covariance(
        right_cells, left_cells, vertices=True, adjacency=True, faces=True
    )
    for right_cell, left_cell in zip(right_cells, left_cells):
        assert right_cell['query_index'] == left_cell['query_index']
        assert right_cell['query'] != left_cell['query']


def test_structured_power_result_transports_actual_empty_cells() -> None:
    right, left, right_points, left_points = _domains_and_points()
    common = dict(
        mode='power',
        weights=(100.0, 0.0, 0.0, 0.0),
        output='result',
        include_empty=True,
        ids=(11, 13, 17, 19),
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
    right_cell = right_cells[0]
    left_cell = left_cells[0]
    right_face = next(face for face in right_cell['faces']
                      if len(face['vertices']) >= 3)
    left_face = left_cell['faces'][right_cell['faces'].index(right_face)]
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
