from __future__ import annotations

import numpy as np
import pytest

import pyvoro2


def test_duplicate_check_returns_empty_for_small_inputs() -> None:
    assert pyvoro2.duplicate_check(np.zeros((0, 3))) == tuple()
    assert pyvoro2.duplicate_check(np.zeros((1, 3))) == tuple()


def test_duplicate_check_detects_exact_duplicate() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts)


def test_duplicate_check_detects_near_duplicate() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.5e-5, 0.0, 0.0]], dtype=float)
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts, threshold=1e-5)


def test_duplicate_check_periodic_wrap_catches_modulo_duplicates() -> None:
    L = 10.0
    dom = pyvoro2.OrthorhombicCell(
        bounds=((0.0, L), (0.0, L), (0.0, L)), periodic=(True, True, True)
    )
    pts = np.array(
        [
            [0.1, 0.2, 0.3],
            [L + 0.1, 0.2, 0.3],  # same point modulo x-periodicity
        ],
        dtype=float,
    )
    # With wrapping, they coincide -> duplicate.
    with pytest.raises(pyvoro2.DuplicateError):
        pyvoro2.duplicate_check(pts, domain=dom, wrap=True)

    # Without wrapping, distance is ~L -> no duplicate.
    pairs = pyvoro2.duplicate_check(pts, domain=dom, wrap=False, mode='return')
    assert pairs == tuple()


def test_duplicate_check_wrap_controls_minimum_image_distance() -> None:
    domain = pyvoro2.OrthorhombicCell(
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic=(True, True, True),
    )
    points = np.array(
        [[0.1, 0.25, 0.25], [0.9, 0.25, 0.25]],
        dtype=np.float64,
    )

    wrapped = pyvoro2.duplicate_check(
        points,
        threshold=0.5,
        domain=domain,
        wrap=True,
        mode='return',
    )
    unwrapped = pyvoro2.duplicate_check(
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


@pytest.mark.parametrize(
    'threshold',
    [1e-310, float(np.nextafter(0.0, 1.0))],
)
def test_triclinic_duplicate_check_handles_tiny_thresholds(
    threshold: float,
) -> None:
    cell = pyvoro2.PeriodicCell(
        (
            (1.0, 0.0, 0.0),
            (0.2, 1.0, 0.0),
            (0.1, 0.3, 1.0),
        )
    )
    separated = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    exact = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

    with np.errstate(all='raise'):
        assert pyvoro2.duplicate_check(
            separated,
            threshold=threshold,
            domain=cell,
            mode='return',
        ) == ()
        pairs = pyvoro2.duplicate_check(
            exact,
            threshold=threshold,
            domain=cell,
            mode='return',
        )
    assert len(pairs) == 1
    assert (pairs[0].i, pairs[0].j, pairs[0].distance) == (0, 1, 0.0)
