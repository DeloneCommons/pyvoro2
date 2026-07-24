from __future__ import annotations

import numpy as np
import pytest

from pyvoro2.planar import Box, RectangularCell
from pyvoro2.planar._domain_geometry import geometry2d


def test_planar_box_from_points() -> None:
    pts = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=float)
    box = Box.from_points(pts, padding=0.5)
    assert box.bounds == ((-0.5, 2.5), (-1.5, 1.5))


def test_rectangular_cell_remap_cart_returns_shifts() -> None:
    cell = RectangularCell(bounds=((0.0, 1.0), (0.0, 2.0)), periodic=(True, True))
    pts = np.array([[1.2, -0.1], [-0.1, 2.1]], dtype=float)

    remapped, shifts = cell.remap_cart(pts, return_shifts=True)
    assert remapped.shape == (2, 2)
    assert shifts.shape == (2, 2)
    assert np.allclose(remapped[0], [0.2, 1.9])
    assert np.allclose(remapped[1], [0.9, 0.1])
    assert shifts.tolist() == [[1, -1], [-1, 1]]


def test_geometry2d_shift_to_cart_and_block_resolution() -> None:
    dom = RectangularCell(bounds=((0.0, 2.0), (-1.0, 3.0)), periodic=(True, False))
    geom = geometry2d(dom)

    sh = np.array([[1, 0], [-2, 0]], dtype=np.int64)
    cart = geom.shift_to_cart(sh)
    assert np.allclose(cart[0], [2.0, 0.0])
    assert np.allclose(cart[1], [-4.0, 0.0])

    assert geom.resolve_block_counts(
        n_sites=10,
        blocks=(3, 4),
        block_size=None,
    ) == (3, 4)


def test_geometry2d_validate_shifts_rejects_nonperiodic_axis() -> None:
    dom = RectangularCell(bounds=((0.0, 1.0), (0.0, 1.0)), periodic=(True, False))
    geom = geometry2d(dom)
    shifts = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match='non-periodic'):
        geom.validate_shifts(shifts)
