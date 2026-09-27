"""WP7's private coefficient storage keeps unused translations unbounded."""

import numpy as np
import pytest

from pyvoro2._internal.generator_preparation import (
    prepare_generators,
    prepare_temporary_generators,
)
from pyvoro2._internal.planar.domain_geometry import geometry2d
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from pyvoro2.domains import Box as SpatialBox, OrthorhombicCell, PeriodicCell
from pyvoro2.planar.domains import Box as PlanarBox, RectangularCell


def _prepared(points, geometry, *, snapshot=None, operation='ghost_cells'):
    return prepare_generators(
        points, geometry=geometry, operation=operation, external_ids=None,
        backend_radii=None, duplicate_check='off', duplicate_threshold=1e-5,
        duplicate_wrap=False, duplicate_max_pairs=10,
        periodic_snapshot=snapshot,
    )


def _temporary(points, persistent, geometry, *, snapshot=None):
    return prepare_temporary_generators(
        points, persistent=persistent, geometry=geometry,
        backend_radii=None, duplicate_check='off', duplicate_threshold=1e-5,
        duplicate_wrap=False, duplicate_max_pairs=10,
        periodic_snapshot=snapshot,
    )


@pytest.mark.parametrize('dim', [2, 3])
def test_ghost_preparation_unbounded_translation_retains_original_query(dim):
    bounds = ((0.0, 1.0),) * dim
    periodic = (True,) + (False,) * (dim - 1)
    domain = (RectangularCell(bounds, periodic=periodic) if dim == 2
              else OrthorhombicCell(bounds, periodic=periodic))
    geometry = geometry2d(domain) if dim == 2 else geometry3d(domain)
    original = np.array([[float(2**70), *(0.5,) * (dim - 1)]])
    empty = _prepared(np.empty((0, dim)), geometry)
    temporary = _temporary(original, empty, geometry)

    assert empty.remap_shifts.dtype == object
    assert temporary.remap_shifts.dtype == object
    assert type(temporary.remap_shifts[0, 0]) is int
    assert temporary.remap_shifts[0].tolist() == [2**70] + [0] * (dim - 1)
    np.testing.assert_array_equal(temporary.input_points_cart, original)
    np.testing.assert_array_equal(
        temporary.native_points, [[0.0, *(0.5,) * (dim - 1)]],
    )
    assert not temporary.remap_shifts.flags.writeable

    persistent = _prepared(original, geometry)
    assert persistent.remap_shifts.dtype == object
    assert persistent.remap_shifts[0, 0] == 2**70
    np.testing.assert_array_equal(persistent.input_points_cart, original)
    np.testing.assert_array_equal(persistent.native_points, temporary.native_points)

    with pytest.raises(ValueError, match='signed int64'):
        _prepared(original, geometry, operation='compute')
    with pytest.raises(ValueError, match='signed int64'):
        domain.remap_cart(original, return_shifts=True)


@pytest.mark.parametrize('handedness', [1, -1])
def test_triclinic_ghost_preparation_unbounded_in_native_chart(handedness):
    domain = PeriodicCell(
        vectors=((float(handedness), 0.0, 0.0),
                 (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    geometry = geometry3d(domain)
    snapshot = geometry.native_periodic_snapshot()
    original = np.array([[float(handedness * 2**70), 0.0, 0.0]])
    empty = _prepared(np.empty((0, 3)), geometry, snapshot=snapshot)
    temporary = _temporary(original, empty, geometry, snapshot=snapshot)
    persistent = _prepared(original, geometry, snapshot=snapshot)
    for prepared in (temporary, persistent):
        assert prepared.remap_shifts.dtype == object
        assert prepared.remap_shifts[0].tolist() == [2**70, 0, 0]
        np.testing.assert_array_equal(prepared.input_points_cart, original)
        np.testing.assert_array_equal(prepared.native_points, [[0., 0., 0.]])
    with pytest.raises(ValueError, match='signed int64'):
        domain.remap_internal(snapshot.cart_to_internal(original),
                              return_shifts=True)


@pytest.mark.parametrize('dim', [2, 3])
def test_ghost_private_remap_matches_public_numerical_and_snap(dim):
    bounds = ((0.0, 1.0),) * dim
    domain = (RectangularCell(bounds, periodic=(True, False)) if dim == 2
              else OrthorhombicCell(bounds, periodic=(True, False, True)))
    geometry = geometry2d(domain) if dim == 2 else geometry3d(domain)
    points = np.array([[2.25, .5] if dim == 2 else [2.25, .5, -1.0],
                       [np.nextafter(1., 0.), .75] if dim == 2
                       else [np.nextafter(1., 0.), .75, 1.]])
    expected_points, expected_shifts = domain.remap_cart(points,
                                                         return_shifts=True)
    prepared = _prepared(points, geometry)
    np.testing.assert_array_equal(prepared.native_points, expected_points)
    np.testing.assert_array_equal(prepared.remap_shifts, expected_shifts)
    assert prepared.remap_shifts.dtype == object
    assert prepared.remap_shifts[:, 1].tolist() == [0, 0]


@pytest.mark.parametrize('handedness', [1, -1])
def test_ghost_private_triclinic_preserves_shear_order_and_snapping(handedness):
    domain = PeriodicCell(
        vectors=((float(handedness), 0., 0.),
                 (0.25 * handedness, 1., 0.),
                 (0.125 * handedness, 0.5, 1.)),
    )
    geometry = geometry3d(domain)
    snapshot = geometry.native_periodic_snapshot()
    internal = np.array([[2.5, -1.25, 3.],
                         [np.nextafter(1., 0.), 1., 0.5]])
    cartesian = snapshot.internal_to_cart(internal)
    expected_points, expected_shifts = snapshot.remap_internal(
        snapshot.cart_to_internal(cartesian), return_shifts=True,
    )
    prepared = _prepared(cartesian, geometry, snapshot=snapshot)
    np.testing.assert_array_equal(prepared.native_points, expected_points)
    np.testing.assert_array_equal(prepared.remap_shifts, expected_shifts)


@pytest.mark.parametrize('dim', [2, 3])
def test_ghost_nonperiodic_coefficients_are_python_zero(dim):
    bounds = ((0.0, 1.0),) * dim
    domain = PlanarBox(bounds) if dim == 2 else SpatialBox(bounds)
    geometry = geometry2d(domain) if dim == 2 else geometry3d(domain)
    prepared = _prepared(np.empty((0, dim)), geometry)
    temporary = _temporary(np.full((1, dim), .5), prepared, geometry)
    assert temporary.remap_shifts.dtype == object
    assert all(type(value) is int and value == 0
               for value in temporary.remap_shifts[0])


def test_ghost_private_triclinic_zero_epsilon_coupled_upper_boundary():
    domain = PeriodicCell(
        vectors=((1., 0., 0.), (.25, 1., 0.), (.125, .5, 1.)),
    )
    internal = np.array([[2.0, 1.0, 1.0], [-.5, 0., 0.],
                         [np.nextafter(1., 0.), -1.0, 2.0]])
    expected, public_shifts = domain.remap_internal(
        internal, return_shifts=True, eps=0,
    )
    actual, private_shifts = domain._remap_internal(
        internal, return_shifts=True, eps=0, integer_view=False,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(private_shifts, public_shifts)
    assert private_shifts.dtype == object
