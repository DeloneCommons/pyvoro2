import numpy as np
import pytest

import pyvoro2 as pv
from pyvoro2._face_shifts3d import _add_periodic_face_shifts_inplace


def test_face_shift_tie_break_prefers_best_residual_over_small_l1():
    """Regression test for face-shift tie-breaking.

    Historically, the tie-break tolerance was derived from the *validation*
    tolerance, which could be far too permissive for very small coordinate
    systems. That allowed a clearly-worse candidate to be selected just because
    it had a smaller |shift|.

    This test constructs a synthetic periodic face where:
      - shift (2,0,0) has residual 0 (correct),
      - shift (1,0,0) has a small but nonzero residual,
    and asserts that the algorithm picks (2,0,0).
    """

    # Tiny cubic lattice: chosen specifically to stress any absolute eps floors.
    a = np.array([1e-10, 0.0, 0.0])
    b = np.array([0.0, 1e-10, 0.0])
    c = np.array([0.0, 0.0, 1e-10])

    # Two sites: the neighbor is positioned such that the nearest-image seed
    # includes shifts 1,2,3 along +a.
    p_i = np.array([0.0, 0.0, 0.0])
    p_j = np.array([-2.1e-10, 0.0, 0.0])

    # For shift (2,0,0), the neighbor image is at -1e-11, so the bisector plane
    # is x = -5e-12.
    x_plane = -5e-12
    verts = np.array(
        [
            [x_plane, 0.0, 0.0],
            [x_plane, 1e-10, 0.0],
            [x_plane, 0.0, 1e-10],
            [x_plane, 1e-10, 1e-10],
        ],
        dtype=float,
    )

    cells = [
        {
            'id': 0,
            'site': p_i.tolist(),
            'vertices': verts.tolist(),
            'faces': [
                {
                    'adjacent_cell': 1,
                    'vertices': [0, 1, 3, 2],
                }
            ],
        },
        {
            'id': 1,
            'site': p_j.tolist(),
            'vertices': [],
            'faces': [],
        },
    ]

    _add_periodic_face_shifts_inplace(
        cells,
        lattice_vectors=(a, b, c),
        periodic_mask=(True, True, True),
        search=2,
        validate=False,
        repair=False,
    )

    assert tuple(cells[0]['faces'][0]['adjacent_shift']) == (2, 0, 0)


def test_spatial_power_shift_residual_handles_large_common_radius() -> None:
    points = np.array([[0.25, 0.4, 0.6]], dtype=float)
    domain = pv.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    cells = pv.compute(
        points,
        domain=domain,
        mode='power',
        radii=np.array([1.0]),
        output='cells',
        return_vertices=True,
        return_faces=True,
        return_face_shifts=False,
    )
    assert len(cells) == 1
    assert not bool(cells[0].get('empty', False))
    assert float(cells[0]['volume']) == pytest.approx(1.0)

    _add_periodic_face_shifts_inplace(
        cells,
        lattice_vectors=tuple(
            np.asarray(vector, dtype=float) for vector in domain.vectors
        ),
        periodic_mask=(True, True, True),
        mode='power',
        radii=np.array([1e8]),
    )

    shifts = {
        tuple(int(value) for value in face['adjacent_shift'])
        for face in cells[0]['faces']
    }
    assert {int(face['adjacent_cell']) for face in cells[0]['faces']} == {0}
    assert shifts == {
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    }
