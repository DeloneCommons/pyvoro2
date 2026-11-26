from __future__ import annotations

import warnings

import pytest

import pyvoro2


def test_periodic_cell_near_coplanar_refuses_native_use() -> None:
    cell = pyvoro2.PeriodicCell(
        ((1., 0., 0.), (0., 1., 0.), (1e-12, 1e-12, 1e-25)))
    assert cell.cart_to_fractional([[0, 0, 0]]).tolist() == [[0., 0., 0.]]
    with pytest.raises(ValueError, match='native.*degenerate'):
        cell.to_internal_params()


def test_periodic_cell_ill_conditioned_warns_at_native_use_only() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cell = pyvoro2.PeriodicCell(
            ((1e12, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        cell.cart_to_fractional([[0, 0, 0]])
    with pytest.warns(RuntimeWarning, match='native adapter'):
        cell.to_internal_params()
