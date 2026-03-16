from __future__ import annotations

import numpy as np

from pyvoro2.planar._edge_shifts2d import _add_periodic_edge_shifts_inplace


def _two_cell_periodic_x() -> list[dict[str, object]]:
    return [
        {
            'id': 0,
            'site': [0.1, 0.5],
            'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            'edges': [
                {'adjacent_cell': -1, 'vertices': [0, 1]},
                {'adjacent_cell': 1, 'vertices': [1, 2]},
                {'adjacent_cell': -2, 'vertices': [2, 3]},
                {'adjacent_cell': 1, 'vertices': [3, 0]},
            ],
        },
        {
            'id': 1,
            'site': [0.9, 0.5],
            'vertices': [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
            'edges': [
                {'adjacent_cell': -1, 'vertices': [0, 1]},
                {'adjacent_cell': 0, 'vertices': [1, 2]},
                {'adjacent_cell': -2, 'vertices': [2, 3]},
                {'adjacent_cell': 0, 'vertices': [3, 0]},
            ],
        },
    ]


def test_planar_edge_shifts_detect_wraparound_standard() -> None:
    cells = _two_cell_periodic_x()

    _add_periodic_edge_shifts_inplace(
        cells,
        lattice_vectors=(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        periodic_mask=(True, False),
        mode='standard',
        search=1,
    )

    c0 = next(cell for cell in cells if cell['id'] == 0)
    c1 = next(cell for cell in cells if cell['id'] == 1)
    s01 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c0['edges']
        if edge['adjacent_cell'] == 1
    }
    s10 = {
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c1['edges']
        if edge['adjacent_cell'] == 0
    }

    assert (-1, 0) in s01
    assert (1, 0) in s10
    assert (0, 0) in s01
    assert (0, 0) in s10


def test_planar_edge_shifts_can_repair_reciprocity() -> None:
    cells = _two_cell_periodic_x()
    for cell in cells:
        for edge in cell['edges']:
            if int(edge['adjacent_cell']) >= 0:
                edge['adjacent_shift'] = (0, 0)

    _add_periodic_edge_shifts_inplace(
        cells,
        lattice_vectors=(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        periodic_mask=(True, False),
        mode='standard',
        search=1,
        validate=True,
        repair=True,
    )

    c0 = next(cell for cell in cells if cell['id'] == 0)
    c1 = next(cell for cell in cells if cell['id'] == 1)

    s01 = sorted(
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c0['edges']
        if edge['adjacent_cell'] == 1
    )
    s10 = sorted(
        tuple(int(v) for v in edge['adjacent_shift'])
        for edge in c1['edges']
        if edge['adjacent_cell'] == 0
    )

    assert s01 == [(-1, 0), (0, 0)]
    assert s10 == [(0, 0), (1, 0)]
