"""A3-R numerical charts, independently evaluated on Fraction operands."""
from fractions import Fraction as F
from itertools import product
import warnings

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell
from pyvoro2.planar import RectangularCell
from pyvoro2._internal.spatial.domain_geometry import _NativePeriodicSnapshot


TINY = float.fromhex('0x0.0000000000001p-1022')
ROUTES = ['planar', 'orthorhombic', 'internal', 'snapshot']


def scalar_oracle(x, lo, period, eps):
    """Exact floor and affine value; only the selected coordinate is rounded."""
    x, lo, period, eps = map(F, (x, lo, period, eps))
    q = (x - lo) // period
    value = float(x - q * period)
    upper = lo + period
    if F(value) >= upper or (eps > 0 and upper - F(value) <= eps):
        return float(lo), q + 1
    if eps > 0 and F(value) - lo < eps:
        return float(lo), q
    return value, q


def triangular_oracle(point, basis, eps):
    residual = list(map(F, point))
    result, shifts = [0.] * 3, [0] * 3
    for axis in (2, 1, 0):
        result[axis], shifts[axis] = scalar_oracle(
            residual[axis], 0, basis[axis][axis], eps)
        residual = [value - shifts[axis] * F(component)
                    for value, component in zip(residual, basis[axis])]
    return result, shifts


def native_remapper(cell, route):
    if route == 'snapshot':
        return _NativePeriodicSnapshot.from_raw(
            vectors=cell.vectors, origin=cell.origin).remap_internal
    return cell.remap_internal


def axis_remapper(route, length):
    dim = 2 if route == 'planar' else 3
    if route in ('internal', 'snapshot'):
        return native_remapper(PeriodicCell(np.eye(3) * length), route), dim
    cls = RectangularCell if dim == 2 else OrthorhombicCell
    return cls(((0., length),) * dim).remap_cart, dim


def assert_fixed(remap, points, eps):
    again, shifts = remap(points, eps=eps, return_shifts=True)
    np.testing.assert_array_equal(again, points)
    np.testing.assert_array_equal(shifts, np.zeros(points.shape, dtype=np.int64))


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('length', [1., 2.])
@pytest.mark.parametrize('eps', [0., None, 1e-12, TINY])
def test_endpoints_subnormals_and_fixed_points(route, length, eps):
    remap, dim = axis_remapper(route, length)
    values = [-TINY, 0., TINY, np.nextafter(length, 0.), length,
              np.nextafter(length, np.inf), -1/8]
    points = np.repeat(np.array(values)[:, None], dim, axis=1)
    original = points.copy()
    effective_eps = 1e-12 * length if eps is None else eps
    expected = [scalar_oracle(x, 0, length, effective_eps) for x in values]
    actual, shifts = remap(points, eps=eps, return_shifts=True)
    np.testing.assert_array_equal(actual, np.repeat(
        np.array([v for v, s in expected])[:, None], dim, axis=1))
    np.testing.assert_array_equal(shifts, np.repeat(
        np.array([s for v, s in expected])[:, None], dim, axis=1))
    assert np.all((actual >= 0) & (actual < length))
    assert actual[0, 0] == shifts[0, 0] == 0
    if eps in (0., TINY):
        assert actual[2, 0] == TINY
    assert actual.dtype == np.float64 and shifts.dtype == np.int64
    assert actual.flags.owndata and shifts.flags.owndata
    assert not np.shares_memory(actual, points)
    np.testing.assert_array_equal(points, original)
    assert_fixed(remap, actual, eps)


@pytest.mark.parametrize('route', ['internal', 'snapshot'])
@pytest.mark.parametrize('eps', [0., None, 1e-12])
@pytest.mark.parametrize('basis, point', [
    ([[1., 0., 0.], [1/4, 1., 0.], [0., 0., 1.]],
     [-2.**-55, -2.**-55, 3/8]),
    ([[1., 0., 0.], [-1/2, 1., 0.], [0., 2.**-54, 1.]],
     [3/16, -2.**-55, -2.**-55]),
    ([[2., 0., 0.], [-1/2, 2., 0.], [1/4, -1/2, 2.]],
     [3/16, 3/8, -TINY]),
    ([[2., 0., 0.], [1/2, 2., 0.], [-1/4, 1/2, 2.]],
     [3/16, -1/8, 2.]),
])
def test_full_row_coupling_and_original_p0_tangential_oracles(
    route, eps, basis, point,
):
    cell = PeriodicCell(basis)
    remap = native_remapper(cell, route)
    translations = np.array([[0, 0, 0], [2, -1, 3], [-2, 1, -3]])
    points = np.array(point) + translations @ np.array(basis)
    effective_eps = 1e-12 * max(np.diag(basis)) if eps is None else eps
    expected = [triangular_oracle(p, basis, effective_eps) for p in points]
    actual, shifts = remap(points, eps=eps, return_shifts=True)
    np.testing.assert_array_equal(actual, [r for r, s in expected])
    np.testing.assert_array_equal(shifts, [s for r, s in expected])
    assert np.all((actual >= 0) & (actual < np.diag(basis)))
    # Original P0 A permits only its tiny x seam error; B has zero tolerance.
    allowed = F(2.**-55) if point[0] < 0 else F(0)
    for p, r, s in zip(points, actual, shifts):
        error = F(r[0]) + sum(int(s[j]) * F(basis[j][0]) for j in range(3)) - F(p[0])
        assert abs(error) <= allowed
    assert_fixed(remap, actual, eps)


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('x, shift', [
    (2.**54, 6004799503160661), (-2.**55, -12009599006321323),
])
def test_exact_quotient_preserves_unit_remainder(route, x, shift):
    remap, dim = axis_remapper(route, 3.)
    actual, shifts = remap([[x] * dim], eps=0., return_shifts=True)
    np.testing.assert_array_equal(actual, [[1.] * dim])
    np.testing.assert_array_equal(shifts, [[shift] * dim])


@pytest.mark.parametrize('route', ['internal', 'snapshot'])
def test_exact_shear_product_preserves_remainder(route):
    cell = PeriodicCell.from_params(1, .1, 1, 0, 0, 1)
    point = [450359962737049.5, 2.**52 - 1, .25]
    expected, shift = triangular_oracle(point, cell.vectors, 0.)
    assert expected == [.975, 0., .25] and shift == [-1, 2**52 - 1, 0]
    actual, shifts = native_remapper(cell, route)([point], eps=0., return_shifts=True)
    np.testing.assert_array_equal(actual, [expected])
    np.testing.assert_array_equal(shifts, [shift])


@pytest.mark.parametrize('route', ['internal', 'snapshot'])
@pytest.mark.parametrize('return_shifts', [False, True])
@pytest.mark.parametrize('x, shear, expected_shift', [
    (2.**63, 1., 2**63 - 1), (-2.**63, 1., None), (2.**63, 2.**-55, None),
])
def test_final_shift_range_after_coupling_and_chart_transition(
    route, return_shifts, x, shear, expected_shift,
):
    remap = native_remapper(PeriodicCell.from_params(1, shear, 1, 0, 0, 1), route)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        if expected_shift is None:
            with pytest.raises(ValueError, match='int64'):
                remap([[x, 1., 0.]], eps=0., return_shifts=return_shifts)
        else:
            result = remap([[x, 1., 0.]], eps=0., return_shifts=return_shifts)
            actual = result[0] if return_shifts else result
            np.testing.assert_array_equal(actual, [[0., 0., 0.]])
            if return_shifts:
                np.testing.assert_array_equal(result[1], [[expected_shift, 1, 0]])


@pytest.mark.parametrize('dim, cls', [(2, RectangularCell), (3, OrthorhombicCell)])
@pytest.mark.parametrize('bounds, x, eps, value, shift', [
    ((-.1, .2), .4, 0., .09999999999999998, 1),
    ((-1., 1.), 2.**-54, 0., 2.**-54, 0),
    ((2.**-54, 1.), 1., 0., None, None),
    ((2.**-54, 1.), 1., 1e-12, 2.**-54, 1),
    ((-2.**60, -1.), -1., 0., None, None),
])
def test_rectangular_represented_period_and_declared_containment(
    dim, cls, bounds, x, eps, value, shift,
):
    remap = cls((bounds,) * dim).remap_cart
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        if value is None:
            for expose in (False, True):
                with pytest.raises(ValueError, match='interval'):
                    remap([[x] * dim], eps=eps, return_shifts=expose)
        else:
            assert scalar_oracle(
                x, bounds[0], bounds[1] - bounds[0], eps) == (value, shift)
            actual, shifts = remap([[x] * dim], eps=eps, return_shifts=True)
            np.testing.assert_array_equal(actual, [[value] * dim])
            np.testing.assert_array_equal(shifts, [[shift] * dim])
            assert_fixed(remap, actual, eps)


@pytest.mark.parametrize('dim, cls', [(2, RectangularCell), (3, OrthorhombicCell)])
def test_rectangular_masks_nonzero_bounds_and_empty_batches(dim, cls):
    bounds = ((-2., 1.), (1., 3.), (0., 4.))[:dim]
    points = np.array([[-3., 3., 5.], [.25, 2., .75]])[:, :dim]
    for mask in product((False, True), repeat=dim):
        remap = cls(bounds, periodic=mask).remap_cart
        for eps in (0., None):
            actual, shifts = remap(points, eps=eps, return_shifts=True)
            for axis, (lo, hi) in enumerate(bounds):
                if mask[axis]:
                    oracle = [scalar_oracle(x, lo, hi - lo, 0.)
                              for x in points[:, axis]]
                    np.testing.assert_array_equal(
                        actual[:, axis], [v for v, s in oracle])
                    np.testing.assert_array_equal(
                        shifts[:, axis], [s for v, s in oracle])
                else:
                    np.testing.assert_array_equal(actual[:, axis], points[:, axis])
                    assert np.all(shifts[:, axis] == 0)
            assert_fixed(remap, actual, eps)
        empty, shifts = remap(np.empty((0, dim)), return_shifts=True)
        assert empty.shape == shifts.shape == (0, dim)
        assert empty.flags.owndata and shifts.flags.owndata


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('count', [0, 1])
def test_snapping_neighborhoods_must_not_touch_or_overlap(route, count):
    remap, dim = axis_remapper(route, 1.)
    points = np.full((count, dim), .25)
    for eps in (.5, .75):
        with pytest.raises(ValueError, match='2.*eps|neighborhood'):
            remap(points, eps=eps)
    remap(points, eps=np.nextafter(.5, 0.))


@pytest.mark.parametrize('dim, cls', [(2, RectangularCell), (3, OrthorhombicCell)])
def test_inactive_rectangular_axes_do_not_constrain_epsilon(dim, cls):
    cell = cls(((0., 1e-13),) + ((0., 1.),) * (dim - 1),
               periodic=(False,) + (True,) * (dim - 1))
    points = np.full((1, dim), .25)
    actual = cell.remap_cart(points)
    np.testing.assert_array_equal(actual, points)
    cell.remap_cart(np.empty((0, dim)))
    inactive = cls(cell.bounds, periodic=(False,) * dim)
    np.testing.assert_array_equal(inactive.remap_cart(points, eps=1e300), points)


@pytest.mark.parametrize('count', [0, 1])
def test_admissible_anisotropic_native_frame_can_refuse_default_remap(count):
    cell = PeriodicCell(np.diag([1., 1., 1e-13]))
    points = np.zeros((count, 3))
    with pytest.warns(RuntimeWarning, match='native adapter'):
        assert cell.to_internal_params()[5] == 1e-13
    with pytest.warns(RuntimeWarning, match='native adapter'):
        with pytest.raises(ValueError, match='2.*eps|neighborhood'):
            cell.remap_internal(points)
    with pytest.warns(RuntimeWarning, match='native adapter'):
        np.testing.assert_array_equal(cell.remap_internal(points, eps=0.), points)


def test_native_chart_does_not_change_a1_exact_floor():
    cell = PeriodicCell(np.eye(3) * 2)
    exact, floor = cell.wrap_cart([[-TINY, 0., 0.]], return_shifts=True)
    native, shift = cell.remap_internal([[-TINY, 0., 0.]], eps=0., return_shifts=True)
    np.testing.assert_array_equal(exact, [[2., 0., 0.]])
    np.testing.assert_array_equal(floor, [[-1, 0, 0]])
    np.testing.assert_array_equal(native, [[0., 0., 0.]])
    np.testing.assert_array_equal(shift, [[0, 0, 0]])


@pytest.mark.parametrize('route', ['internal', 'snapshot'])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_coupled_underflow_on_each_axis(route, axis):
    cell = PeriodicCell.from_params(2, -.5, 2, .25, -.5, 2)
    remap = native_remapper(cell, route)
    points = np.tile([3/16, 3/8, 5/8], (3, 1))
    points[:, axis] = [-TINY, TINY, -1/8]
    actual, shifts = remap(points, eps=0., return_shifts=True)
    oracle = [triangular_oracle(p, cell.vectors, 0.) for p in points]
    np.testing.assert_array_equal(actual, [r for r, s in oracle])
    np.testing.assert_array_equal(shifts, [s for r, s in oracle])
    assert actual[1, axis] == TINY
    assert shifts[2, axis] == -1
    assert_fixed(remap, actual, 0.)


@pytest.mark.parametrize('route', ROUTES)
def test_lower_snap_is_strict_and_upper_snap_is_inclusive(route):
    remap, dim = axis_remapper(route, 1.)
    values = [1/8, np.nextafter(1/8, 0.), 7/8, np.nextafter(7/8, 0.)]
    actual, shifts = remap(np.repeat(np.array(values)[:, None], dim, axis=1),
                           eps=1/8, return_shifts=True)
    expected = [scalar_oracle(x, 0., 1., 1/8) for x in values]
    np.testing.assert_array_equal(actual, [[v] * dim for v, s in expected])
    np.testing.assert_array_equal(shifts, [[s] * dim for v, s in expected])
    assert_fixed(remap, actual, 1/8)


def test_private_coefficients_stay_wide_until_materialization():
    from pyvoro2._internal.native_remap import materialize_remap, remap_triangular

    points = np.array([[2.**63, 1., 0.]])
    primary, wide = remap_triangular(points, (1., 0., 1., 0., 0., 1.), 0.)
    np.testing.assert_array_equal(primary, [[0., 0., 0.]])
    assert wide == [(2**63, 1, 0)]
    assert all(isinstance(s, int) for s in wide[0])
    for expose in (False, True):
        with pytest.raises(ValueError, match='int64'):
            materialize_remap(primary, wide, expose)
