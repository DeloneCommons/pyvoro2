from __future__ import annotations

import numpy as np
import pytest

import pyvoro2.planar as planar


def test_duplicate_check_wrap_controls_minimum_image_distance() -> None:
    domain = planar.RectangularCell(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        periodic=(True, True),
    )
    points = np.array(
        [[0.1, 0.25], [0.9, 0.25]],
        dtype=np.float64,
    )

    wrapped = planar.duplicate_check(
        points,
        threshold=0.5,
        domain=domain,
        wrap=True,
        mode='return',
    )
    unwrapped = planar.duplicate_check(
        points,
        threshold=0.5,
        domain=domain,
        wrap=False,
        mode='return',
    )

    assert len(wrapped) == 1
    assert (wrapped[0].i, wrapped[0].j) == (0, 1)
    assert wrapped[0].distance == pytest.approx(0.2)
    assert unwrapped == tuple()
