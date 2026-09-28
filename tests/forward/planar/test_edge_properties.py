from __future__ import annotations

import numpy as np

from pyvoro2.edge_properties import annotate_edge_properties
from pyvoro2.planar import RectangularCell


def test_annotate_edge_properties_basic() -> None:
    # Unit x period: the x=0/1 edges meet the opposite translated site.
    cells = [
        {
            'id': 0,
            'site': [0.1, 0.5],
            'vertices': [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            'edges': [
                {'adjacent_cell': -1, 'vertices': [0, 1]},
                {'adjacent_cell': 1, 'vertices': [1, 2],
                 'adjacent_shift': (0, 0)},
                {'adjacent_cell': -2, 'vertices': [2, 3]},
                {'adjacent_cell': 1, 'vertices': [3, 0],
                 'adjacent_shift': (-1, 0)},
            ],
        },
        {
            'id': 1,
            'site': [0.9, 0.5],
            'vertices': [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
            'edges': [
                {'adjacent_cell': -1, 'vertices': [0, 1]},
                {'adjacent_cell': 0, 'vertices': [1, 2],
                 'adjacent_shift': (1, 0)},
                {'adjacent_cell': -2, 'vertices': [2, 3]},
                {'adjacent_cell': 0, 'vertices': [3, 0],
                 'adjacent_shift': (0, 0)},
            ],
        },
    ]
    dom = RectangularCell(bounds=((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))

    annotate_edge_properties(cells, dom)

    edge = cells[0]['edges'][1]
    assert np.allclose(edge['midpoint'], [0.5, 0.5])
    assert np.isclose(edge['length'], 1.0)
    assert edge['normal'] is not None
    assert np.allclose(edge['other_site'], [0.9, 0.5])

    wrap_edge = cells[0]['edges'][3]
    assert np.allclose(wrap_edge['other_site'], [-0.1, 0.5])
    assert np.allclose(wrap_edge['midpoint'], [0.0, 0.5])
    assert np.isclose(wrap_edge['length'], 1.0)
