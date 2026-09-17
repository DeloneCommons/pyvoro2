from __future__ import annotations

import numpy as np
import pytest

import pyvoro2
from pyvoro2._internal.spatial.domain_geometry import DomainGeometry3D


@pytest.mark.parametrize('signs', [
    (sx, sy, sz)
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
    for sz in (-1.0, 1.0)
])
def test_backend_frame_has_positive_diagonal_and_exact_parity(signs) -> None:
    matrix = np.diag(np.asarray(signs, dtype=np.float64))
    cell = pyvoro2.PeriodicCell(tuple(map(tuple, matrix)))
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    q = snapshot.rotation_to_internal.T
    lower = matrix @ q

    np.testing.assert_allclose(q.T @ q, np.eye(3), rtol=0, atol=2e-15)
    np.testing.assert_allclose(lower, np.tril(lower), rtol=0, atol=2e-15)
    assert np.all(np.diag(lower) > 0.0)
    assert snapshot.parity == int(np.prod(signs))
    assert np.sign(np.linalg.det(q)) == snapshot.parity


@pytest.mark.parametrize(('rows', 'expected_parity'), [
    (((2.0, 0.25, 0.5), (0.2, 3.0, -0.4), (0.1, 0.7, 4.0)), 1),
    (((0.2, 3.0, -0.4), (2.0, 0.25, 0.5), (0.1, 0.7, 4.0)), -1),
])
def test_public_and_snapshot_backend_transforms_agree(
    rows, expected_parity
) -> None:
    cell = pyvoro2.PeriodicCell(rows, origin=(0.25, -0.5, 1.0))
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 8.0]])

    np.testing.assert_array_equal(
        cell._rotation_to_internal(), snapshot.rotation_to_internal
    )
    assert cell.to_internal_params() == snapshot.params
    assert snapshot.parity == expected_parity

    # The two paths use the identical prepared frame but express the products
    # as row matmul and transposed column matmul. Four ULP cover their three-term
    # dot products plus origin addition without tolerating a different frame.
    internal_public = cell.cart_to_internal(points)
    internal_snapshot = snapshot.cart_to_internal(points)
    np.testing.assert_array_max_ulp(
        internal_public, internal_snapshot, maxulp=4
    )
    np.testing.assert_array_max_ulp(
        cell.internal_to_cart(internal_snapshot),
        snapshot.internal_to_cart(internal_snapshot),
        maxulp=4,
    )


@pytest.mark.parametrize('scale', [1e-100, 1e100])
def test_backend_frame_validation_is_scale_aware(scale) -> None:
    matrix = scale * np.array(
        ((2.0, 0.25, 0.5), (0.2, 3.0, -0.4), (0.1, 0.7, -4.0))
    )
    cell = pyvoro2.PeriodicCell(tuple(map(tuple, matrix)))
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    q = snapshot.rotation_to_internal.T
    np.testing.assert_allclose(
        matrix @ q,
        np.tril(matrix @ q),
        rtol=0.0,
        atol=3e-15 * abs(scale),
    )
    assert snapshot.parity == -1


def test_user_operations_do_not_prepare_the_backend_frame(monkeypatch) -> None:
    cell = pyvoro2.PeriodicCell.from_params(1, 0, 1, 0, 0, 1)

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError('injected backend failure')

    monkeypatch.setattr(np.linalg, 'qr', fail)
    assert cell.wrap_fractional([[1.25, 0.0, 0.0]]).tolist() == [
        [0.25, 0.0, 0.0]
    ]
    assert cell.wrap_cart([[1.25, 0.0, 0.0]]).tolist() == [
        [0.25, 0.0, 0.0]
    ]
    with pytest.raises(ValueError, match='backend frame'):
        cell.to_internal_params()
