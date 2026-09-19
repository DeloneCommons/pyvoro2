from types import SimpleNamespace

import numpy as np
import pytest

from pyvoro2 import PeriodicCell, compute
import pyvoro2.api as api3d
from pyvoro2._internal.spatial.domain_geometry import DomainGeometry3D


def _sheared_cell() -> PeriodicCell:
    # Choose lattice vectors already in Voro++'s lower-triangular form so that
    # the internal basis equals Cartesian (rotation = identity).
    #
    # a = (bx, 0, 0)
    # b = (bxy, by, 0)
    # c = (bxz, byz, bz)
    return PeriodicCell(vectors=((10.0, 0.0, 0.0), (2.0, 10.0, 0.0), (1.0, 3.0, 10.0)))


@pytest.mark.parametrize('handedness', [1.0, -1.0], ids=['right', 'left'])
def test_coupled_upper_snap_preserves_cartesian_tangential_residual(
    handedness,
) -> None:
    lower = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    vectors = lower @ np.diag([handedness, 1.0, 1.0])
    cell = PeriodicCell(vectors)
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    point_internal = np.array([[3 / 16, -(2.0**-55), 3 / 8]])
    point_cart = snapshot.internal_to_cart(point_internal)
    expected_internal = np.array([[3 / 16, 0.0, 3 / 8]])
    expected_cart = snapshot.internal_to_cart(expected_internal)

    for remapper in (cell, snapshot):
        wrapped, shifts = remapper.remap_cart(
            point_cart,
            return_shifts=True,
        )
        np.testing.assert_allclose(wrapped, expected_cart, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(shifts, [[0, 0, 0]])


@pytest.mark.parametrize(
    ('vectors', 'point', 'expected', 'shifts'),
    [
        pytest.param(
            ((1.0, 0.0, 0.0), (0.25, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.5, -(2.0**-55), 3 / 8),
            (0.5, 0.0, 3 / 8),
            (0, 0, 0),
            id='y-to-x-inside',
        ),
        pytest.param(
            ((1.0, 0.0, 0.0), (0.25, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (1.0 - 2.0**-41, -(2.0**-55), 3 / 8),
            (0.0, 0.0, 3 / 8),
            (1, 0, 0),
            id='y-to-x-near-seam',
        ),
        pytest.param(
            ((1.0, 0.0, 0.0), (-0.5, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (2.0**-41, -(2.0**-55), 3 / 8),
            (0.0, 0.0, 3 / 8),
            (0, 0, 0),
            id='y-to-x-just-above-seam',
        ),
        pytest.param(
            ((1.0, 0.0, 0.0), (-2.5, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (3 / 16, -(2.0**-55), 3 / 8),
            (3 / 16, 0.0, 3 / 8),
            (0, 0, 0),
            id='y-to-x-multiple-widths',
        ),
        pytest.param(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.5, 0.0, 1.0)),
            (3 / 16, 3 / 8, -(2.0**-55)),
            (3 / 16, 3 / 8, 0.0),
            (0, 0, 0),
            id='z-to-x',
        ),
        pytest.param(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -0.5, 1.0)),
            (3 / 8, 3 / 16, -(2.0**-55)),
            (3 / 8, 3 / 16, 0.0),
            (0, 0, 0),
            id='z-to-y',
        ),
    ],
)
def test_coupled_upper_snap_only_snaps_coordinates_near_their_boundary(
    vectors,
    point,
    expected,
    shifts,
) -> None:
    cell = PeriodicCell(vectors)
    snapshot = DomainGeometry3D(cell).native_periodic_snapshot()
    points = np.asarray([point])

    for remapper in (cell, snapshot):
        wrapped, actual_shifts = remapper.remap_internal(
            points,
            return_shifts=True,
        )
        np.testing.assert_allclose(wrapped, [expected], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(actual_shifts, [shifts])


def test_persistent_compute_preserves_equivalent_cubic_physics() -> None:
    bases = (
        np.eye(3),
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[-1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    points = np.array([
        [-3 / 16, -3 / 16, 3 / 8],
        [1 / 16, 3 / 16, 5 / 16],
        [5 / 8, 7 / 16, 1 / 8],
        [1 / 4, 3 / 4, 11 / 16],
    ])
    ids = [101, 202, 303, 404]
    weights = np.array([3 / 64, 0.0, 1 / 32, -1 / 64])
    families = (
        ('standard', {}, 0.21879657822851134),
        ('weights', {'mode': 'power', 'weights': weights},
         0.29658409203603464),
        ('radii', {'mode': 'power', 'radii': np.sqrt(weights - weights.min())},
         0.29658409203603464),
    )

    for _family, options, expected_first_volume in families:
        reference = None
        for basis in bases:
            result = compute(
                points,
                ids=ids,
                domain=PeriodicCell(basis),
                return_vertices=False,
                return_adjacency=False,
                return_faces=False,
                **options,
            )
            volumes = {
                int(cell['id']): float(cell['volume'])
                for cell in result.cells
            }
            assert volumes[101] == pytest.approx(
                expected_first_volume,
                abs=1e-12,
                rel=1e-12,
            )
            if reference is None:
                reference = volumes
            else:
                assert volumes == pytest.approx(reference, abs=1e-12, rel=1e-12)


def test_remap_internal_couples_x_when_wrapping_y() -> None:
    cell = _sheared_cell()
    pts_i = np.array([[1.0, 12.0, 0.0]], dtype=float)

    rem, shifts = cell.remap_internal(pts_i, return_shifts=True)
    assert rem.shape == (1, 3)
    assert shifts.shape == (1, 3)

    # Wrapping y by -by also shifts x by -bxy in the sheared basis.
    assert np.allclose(rem[0], [9.0, 2.0, 0.0])

    # Reconstruction: p = rem + na*a + nb*b + nc*c (in internal coords).
    bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
    a = np.array([bx, 0.0, 0.0])
    b = np.array([bxy, by, 0.0])
    c = np.array([bxz, byz, bz])
    na, nb, nc = shifts[0]
    rec = rem[0] + na * a + nb * b + nc * c
    assert np.allclose(rec, pts_i[0])


def test_remap_internal_couples_xy_when_wrapping_z() -> None:
    cell = _sheared_cell()
    pts_i = np.array([[1.0, 1.0, 12.0]], dtype=float)

    rem, shifts = cell.remap_internal(pts_i, return_shifts=True)
    assert np.allclose(rem[0], [2.0, 8.0, 2.0])

    bx, bxy, by, bxz, byz, bz = cell.to_internal_params()
    a = np.array([bx, 0.0, 0.0])
    b = np.array([bxy, by, 0.0])
    c = np.array([bxz, byz, bz])
    na, nb, nc = shifts[0]
    rec = rem[0] + na * a + nb * b + nc * c
    assert np.allclose(rec, pts_i[0])


def test_remap_cart_respects_origin() -> None:
    cell = PeriodicCell(
        vectors=((10.0, 0.0, 0.0), (2.0, 10.0, 0.0), (1.0, 3.0, 10.0)),
        origin=(10.0, 0.0, 0.0),
    )
    pt = np.array([[11.0, 12.0, 0.0]], dtype=float)  # origin + [1,12,0]
    rem = cell.remap_cart(pt)
    assert np.allclose(rem[0], [19.0, 2.0, 0.0])


def test_compute_passes_primary_periodic_points(monkeypatch) -> None:
    """Ensure compute() remaps sheared internal coordinates before dispatch."""
    cell = _sheared_cell()
    pts = np.array([[1.0, 12.0, 0.0], [5.0, 5.0, 5.0]], dtype=float)

    captured = {}

    def fake_compute_periodic_standard(
        pts_i, ids_internal, cell_params, blocks, init_mem, opts
    ):
        captured['pts_i'] = np.asarray(pts_i, dtype=float)
        # Minimal stub cells.
        return [{'id': int(i), 'volume': 0.0} for i in range(len(pts_i))]

    fake_core = SimpleNamespace(
        compute_periodic_standard=fake_compute_periodic_standard,
    )
    monkeypatch.setattr(api3d, '_core', fake_core)
    monkeypatch.setattr(api3d, '_CORE_IMPORT_ERROR', None)

    compute(
        pts,
        domain=cell,
        mode='standard',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )

    assert 'pts_i' in captured
    # Internal basis equals Cartesian and y wrapping couples x through bxy.
    np.testing.assert_allclose(captured['pts_i'][0], [9.0, 2.0, 0.0])


def test_compute_routing_keeps_backend_remap_distinct_from_exact_wrap(
    monkeypatch,
) -> None:
    cell = _sheared_cell()
    points = np.array([[-2.0**-55, 1.0, 1.0], [5.0, 5.0, 5.0]])
    captured = {}

    exact_wrapped, exact_shifts = cell.wrap_cart(
        points[:1], return_shifts=True
    )
    assert exact_wrapped[0, 0] == 10.0
    assert exact_shifts[0, 0] == -1

    def fake_compute_periodic_standard(
        pts_i, ids_internal, cell_params, blocks, init_mem, opts
    ):
        captured['pts_i'] = np.asarray(pts_i, dtype=float)
        return [{'id': int(value), 'volume': 0.0}
                for value in ids_internal]

    monkeypatch.setattr(
        api3d,
        '_core',
        SimpleNamespace(compute_periodic_standard=fake_compute_periodic_standard),
    )
    monkeypatch.setattr(api3d, '_CORE_IMPORT_ERROR', None)

    compute(
        points,
        domain=cell,
        output='cells',
        return_vertices=False,
        return_adjacency=False,
        return_faces=False,
    )

    # Native generator preparation deliberately keeps remap_cart's epsilon
    # and backend-primary convention; exact user wrapping does not replace it.
    assert captured['pts_i'][0, 0] == 0.0
