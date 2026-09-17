from __future__ import annotations

import pytest

import pyvoro2


def test_periodic_cell_near_coplanar_is_exactly_valid_but_warns() -> None:
    # Numerical conditioning is diagnostic; this binary64 lattice is exactly
    # nonsingular and construction therefore succeeds.
    with pytest.warns(RuntimeWarning, match='ill-conditioned'):
        pyvoro2.PeriodicCell(
            vectors=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1e-12, 1e-12, 1e-25),
            )
        )


def test_periodic_cell_ill_conditioned_warns() -> None:
    # A very large aspect ratio should warn but still be allowed.
    with pytest.warns(RuntimeWarning):
        pyvoro2.PeriodicCell(
            vectors=(
                (1e12, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
