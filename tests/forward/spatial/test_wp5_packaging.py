"""WP5 consumer contracts using independently specified raw native records."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from pyvoro2 import (
    Box,
    OrthorhombicCell,
    annotate_face_properties,
    normalize_topology,
)
from pyvoro2._internal.power_input import ResolvedPowerInput
from pyvoro2.planar import RectangularCell
from pyvoro2.result import _build_tessellation_result


def _slab_cell() -> dict:
    """A unit cube with one periodic axis, two self images and four walls."""

    return {
        'id': 41,
        'site': [0.25, 0.5, 0.5],
        'volume': 1.0,
        'vertices': [
            [-0.25, 0.0, 0.0], [0.75, 0.0, 0.0],
            [0.75, 1.0, 0.0], [-0.25, 1.0, 0.0],
            [-0.25, 0.0, 1.0], [0.75, 0.0, 1.0],
            [0.75, 1.0, 1.0], [-0.25, 1.0, 1.0],
        ],
        'adjacency': [
            [1, 3, 4], [0, 5, 2], [1, 6, 3], [0, 2, 7],
            [0, 7, 5], [1, 4, 6], [2, 5, 7], [3, 6, 4],
        ],
        'faces': [
            {
                'adjacent_cell': 41,
                'adjacent_shift': (-1, 0, 0),
                'vertices': [0, 3, 7, 4],
            },
            {
                'adjacent_cell': 41,
                'adjacent_shift': (1, 0, 0),
                'vertices': [1, 5, 6, 2],
            },
            {'adjacent_cell': -3, 'vertices': [0, 4, 5, 1]},
            {'adjacent_cell': -4, 'vertices': [3, 2, 6, 7]},
            {'adjacent_cell': -5, 'vertices': [0, 1, 2, 3]},
            {'adjacent_cell': -6, 'vertices': [4, 7, 6, 5]},
        ],
    }


def _slab_domain() -> OrthorhombicCell:
    return OrthorhombicCell(
        ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, False, False),
    )


def _build_certified_result(cells: list[dict]):
    return _build_tessellation_result(
        dimension=3,
        domain=_slab_domain(),
        mode='standard',
        sites=np.array([[0.25, 0.5, 0.5]]),
        ids=np.array([41]),
        cells=cells,
        power_input=ResolvedPowerInput(None, None, None),
        boundaries_available=True,
        periodic_shifts_available=True,
    )


@pytest.mark.parametrize('vertices', [False, True])
@pytest.mark.parametrize('adjacency', [False, True])
def test_certified_result_accepts_walls_without_image_shifts(
    vertices: bool, adjacency: bool,
) -> None:
    cell = _slab_cell()
    if not vertices:
        del cell['vertices']
    if not adjacency:
        del cell['adjacency']
    cells = [cell]
    original = copy.deepcopy(cells)

    result = _build_certified_result(cells)

    assert result.has_periodic_shifts
    assert result.has_boundaries
    assert result.cells is cells
    assert result.require_boundaries() == [cell['faces']]
    assert cells == original
    assert ('vertices' in cell) is vertices
    assert ('adjacency' in cell) is adjacency


def test_certified_result_rechecks_generator_shift_after_wall_mutation() -> None:
    cells = [_slab_cell()]
    result = _build_certified_result(cells)
    cells[0]['faces'][2]['adjacent_cell'] = 41

    with pytest.raises(ValueError, match='without adjacent_shift'):
        result.require_boundaries()


def test_certified_result_does_not_treat_malformed_owner_as_wall() -> None:
    cells = [_slab_cell()]
    cells[0]['faces'][2]['adjacent_cell'] = -3.0

    with pytest.raises(ValueError, match='without adjacent_shift'):
        _build_certified_result(cells)


def test_wp6_planar_wall_exemption_uses_its_own_side_and_mask() -> None:
    result = _build_tessellation_result(
        dimension=2,
        domain=RectangularCell(
            ((0.0, 1.0), (0.0, 1.0)), periodic=(True, False),
        ),
        mode='standard',
        sites=np.array([[0.25, 0.5]]),
        ids=np.array([41]),
        cells=[{
            'id': 41,
            'area': 1.0,
            'edges': [{'adjacent_cell': -3, 'vertices': [0, 1]}],
        }],
        power_input=ResolvedPowerInput(None, None, None),
        boundaries_available=True,
        periodic_shifts_available=True,
    )
    assert result.has_periodic_shifts
    result.cells[0]['edges'][0]['adjacent_cell'] = -1
    with pytest.raises(ValueError, match='not a planar wall'):
        result.require_boundaries()


def test_normalization_accepts_wall_identity_without_image_shift() -> None:
    cells = [_slab_cell()]
    original = copy.deepcopy(cells)

    normalized = normalize_topology(cells, domain=_slab_domain())

    assert len(normalized.cells[0]['faces']) == 6
    face_ids = normalized.cells[0]['face_global_id']
    assert face_ids[0] == face_ids[1]
    assert len(set(face_ids)) == 5
    assert cells == original
    assert all(
        'adjacent_shift' not in face
        for face in normalized.cells[0]['faces'][2:]
    )


@pytest.mark.parametrize(
    ('second_owner', 'second_shift'),
    [(20, (1, 0, 0)), (30, (0, 0, 0))],
)
def test_normalization_keeps_distinct_owner_image_labels(
    second_owner: int, second_shift: tuple[int, int, int],
) -> None:
    # Deliberately coincident numerical cycles isolate the label contract:
    # coordinate quantization cannot merge different semantic boundaries.
    cycle = [[0.25, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]
    cells = [{
        'id': 10,
        'site': [0.125, 0.375, 0.375],
        'volume': 1.0,
        'vertices': cycle + copy.deepcopy(cycle),
        'faces': [
            {
                'adjacent_cell': 20,
                'adjacent_shift': (0, 0, 0),
                'vertices': [0, 1, 2],
            },
            {
                'adjacent_cell': second_owner,
                'adjacent_shift': second_shift,
                'vertices': [3, 4, 5],
            },
        ],
    }]

    normalized = normalize_topology(cells, domain=_slab_domain(), tol=0.125)

    assert len(normalized.global_vertices) == 6
    assert len(normalized.global_faces) == 2
    assert len(set(normalized.cells[0]['face_global_id'])) == 2
    assert {
        (face['cells'], face['cell_shifts'][1])
        for face in normalized.global_faces
    } == {
        ((10, 20), (0, 0, 0)),
        ((10, second_owner), second_shift),
    }


@pytest.mark.parametrize('coincident', [False, True])
@pytest.mark.parametrize('reverse_count', [0, 1, 2])
def test_normalization_keeps_repeated_directed_face_occurrences(
    coincident: bool, reverse_count: int,
) -> None:
    first = [[0.25, 0.125, 0.125], [0.25, 0.25, 0.125],
             [0.25, 0.125, 0.25]]
    second = (copy.deepcopy(first) if coincident else
              [[0.25, 0.625, 0.625], [0.25, 0.75, 0.625],
               [0.25, 0.625, 0.75]])
    cells = [{
        'id': 10,
        'vertices': first + second,
        'faces': [
            {'adjacent_cell': 20, 'adjacent_shift': (0, 0, 0),
             'vertices': [0, 1, 2]},
            {'adjacent_cell': 20, 'adjacent_shift': (0, 0, 0),
             'vertices': [3, 4, 5]},
        ],
    }]
    if reverse_count:
        cells.append({
            'id': 20,
            'vertices': copy.deepcopy(first + second),
            'faces': [
                {'adjacent_cell': 10, 'adjacent_shift': (0, 0, 0),
                 'vertices': [2, 1, 0]},
                {'adjacent_cell': 10, 'adjacent_shift': (0, 0, 0),
                 'vertices': [5, 4, 3]},
            ][:reverse_count],
        })
    original = copy.deepcopy(cells)

    normalized = normalize_topology(cells, domain=_slab_domain())

    # One label can describe several native fragments. Even exact numerical
    # coincidence does not prove which directed occurrences correspond.
    expected_count = 2 + reverse_count
    assert len(normalized.global_faces) == expected_count
    assert len({fid for cell in normalized.cells
                for fid in cell['face_global_id']}) == expected_count
    for cell in normalized.cells:
        for face, fid in zip(cell['faces'], cell['face_global_id']):
            global_face = normalized.global_faces[fid]
            expected = {tuple(cell['vertices'][index])
                        for index in face['vertices']}
            actual = {tuple(normalized.global_vertices[index])
                      for index in global_face['vertices']}
            assert actual == expected
            assert set(global_face) == {
                'cells', 'cell_shifts', 'vertices', 'vertex_shifts',
            }
    assert cells == original


def test_nonplanar_native_face_keeps_triangulated_descriptors() -> None:
    cells = [{
        'id': 4,
        'site': [0.5, 0.5, -1.0],
        'vertices': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        'faces': [{'adjacent_cell': -6, 'vertices': [0, 1, 2, 3]}],
    }]
    original_vertices = copy.deepcopy(cells[0]['vertices'])

    annotate_face_properties(cells, Box(((-2.0, 2.0),) * 3))

    face = cells[0]['faces'][0]
    # Both fan triangles have area sqrt(2)/2, while their vector-area sum
    # has magnitude sqrt(6)/2. These are different for this nonplanar cycle.
    assert face['centroid'] == pytest.approx([0.5, 0.5, 1.0 / 3.0])
    assert face['area'] == pytest.approx(math.sqrt(6.0) / 2.0)
    assert face['normal'] == pytest.approx(
        [-1.0 / math.sqrt(6.0), -1.0 / math.sqrt(6.0),
         2.0 / math.sqrt(6.0)]
    )
    assert cells[0]['vertices'] == original_vertices
