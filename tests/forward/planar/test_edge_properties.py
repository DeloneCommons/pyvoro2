from __future__ import annotations

import numpy as np

from pyvoro2.edge_properties import annotate_edge_properties
from pyvoro2.planar import RectangularCell
from pyvoro2.planar._edge_shifts2d import _add_periodic_edge_shifts_inplace


def test_annotate_edge_properties_basic() -> None:
    cells = [
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
    dom = RectangularCell(bounds=((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))

    _add_periodic_edge_shifts_inplace(
        cells,
        lattice_vectors=dom.lattice_vectors,
        periodic_mask=dom.periodic,
        search=1,
    )
    annotate_edge_properties(cells, dom)

    edge = cells[0]['edges'][1]
    assert np.allclose(edge['midpoint'], [0.5, 0.5])
    assert np.isclose(edge['length'], 1.0)
    assert edge['normal'] is not None
    assert np.allclose(edge['other_site'], [0.9, 0.5])
