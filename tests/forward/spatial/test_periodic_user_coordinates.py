"""A1 independent rational oracles; no native/frame or production exact helpers."""
from fractions import Fraction as F
from itertools import permutations, product
import math
import warnings

import numpy as np
import pytest

from pyvoro2 import PeriodicCell


def exact(value):
    return F(float(value))


def determinant(rows):
    # Leibniz expansion independent of the production three-term formula.
    total = F(0)
    for p in permutations(range(3)):
        inversions = sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
        total += (-1)**inversions * math.prod(exact(rows[i][p[i]]) for i in range(3))
    return total


def solve(point, origin, basis):
    # Gaussian elimination of A.T f = x-o, independent of cached adjugates.
    rows = [[exact(basis[j][i]) for j in range(3)]
            + [exact(point[i]) - exact(origin[i])] for i in range(3)]
    for j in range(3):
        pivot = next(i for i in range(j, 3) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [v / divisor for v in rows[j]]
        for i in range(3):
            if i != j:
                factor = rows[i][j]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def affine(fractional, origin, basis):
    return [
        exact(origin[j]) + sum(fractional[i] * exact(basis[i][j])
                               for i in range(3))
        for j in range(3)
    ]


BASES = [np.eye(3), [[3, .5, -.25], [-1, 2, .75], [.5, -.25, -4]],
         [[1, 1, 0], [1, 1 + 2**-30, 0], [0, 0, 1]],
         [[2**27, 2**27 - 1, 0], [2**27 + 1, 2**27, 0], [0, 0, 1]]]
BASES += [np.diag(signs) for signs in product((-1., 1.), repeat=3)]
BASES += [np.eye(3) * scale for scale in (1e-110, 1e110)]


@pytest.mark.parametrize('basis', BASES)
def test_exact_user_operations_against_independent_oracles(basis):
    assert determinant(basis) != 0
    origin = (.125, -.75, 2.)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cell = PeriodicCell(basis, origin)
        assert cell.vectors == tuple(tuple(float(v) for v in row) for row in basis)
        points = np.array([[math.nextafter(1., 2.), -3.25, .1],
                           [-2**40, .25, -.125]])
        fractional = np.array([[1/3, math.nextafter(-1., -2.), .125]])
        actual = cell.cart_to_fractional(points)
        wrapped = cell.wrap_cart(points)
        for i, point in enumerate(points):
            f = solve(point, origin, basis)
            shift = [math.floor(v) for v in f]
            remainder = [v - s for v, s in zip(f, shift)]
            assert all(0 <= r < 1 for r in remainder)
            assert affine(f, origin, basis) == list(map(exact, point))
            w = affine(remainder, origin, basis)
            assert actual[i].tolist() == list(map(float, f))
            assert wrapped[i].tolist() == list(map(float, w))
            if all(-2**63 <= s < 2**63 for s in shift):
                result, shifts = cell.wrap_cart([point], return_shifts=True)
                assert shifts.tolist() == [shift]
                assert result[0].tolist() == list(map(float, w))
                # Exact reconstruction of rounded views has ONLY view error.
                for j in range(3):
                    reconstruction = exact(result[0, j]) + sum(
                        shift[k] * exact(basis[k][j]) for k in range(3))
                    error = exact(result[0, j]) - w[j]
                    assert reconstruction - exact(point[j]) == error
        expected = affine(list(map(exact, fractional[0])), origin, basis)
        cart = cell.fractional_to_cart(fractional)
        assert cart[0].tolist() == list(map(float, expected))


@pytest.mark.parametrize('basis', [
    np.zeros((3, 3)),
    [[1, 2, 3], [2, 4, 6], [0, 1, 0]],
    [[1e110, 0, 0], [1e110, 0, 0], [0, 0, 1]],
])
def test_exactly_singular_rejected(basis):
    assert determinant(basis) == 0
    with pytest.raises(ValueError, match='exactly singular'):
        PeriodicCell(basis)


@pytest.mark.parametrize('offset', [-2**-55, -2**-54, -2**-1074])
def test_wrapping_keeps_exact_shift_when_view_rounds_to_one(offset):
    cell = PeriodicCell(np.eye(3))
    for method in (cell.wrap_fractional, cell.wrap_cart):
        view, shifts = method([[offset, 0, 0]], return_shifts=True)
        assert view.tolist() == [[1., 0., 0.]]
        assert shifts.tolist() == [[-1, 0, 0]]
    cell = PeriodicCell(np.eye(3), origin=(-offset, 0, 0))
    assert cell.cart_to_fractional([[1, 0, 0]])[0, 0] == 1.
    view, shifts = cell.wrap_cart([[1, 0, 0]], return_shifts=True)
    assert view.tolist() == [[1., 0., 0.]]
    assert shifts.tolist() == [[0, 0, 0]]


@pytest.mark.parametrize('method', ['wrap_cart', 'wrap_fractional'])
def test_int64_materialization_is_optional_and_checked(method):
    operation = getattr(PeriodicCell(np.eye(3)), method)
    points = [[-2**63, 0, 0], [2**63 - 1024, 0, 0]]
    view, shifts = operation(points, return_shifts=np.bool_(True))
    assert view.tolist() == [[0., 0., 0.]] * 2
    assert shifts.dtype == np.int64
    assert shifts.tolist() == points
    for x in (2**63, np.nextafter(float(-2**63), -np.inf), -1e300):
        assert operation([[x, 0, 0]]).tolist() == [[0., 0., 0.]]
        with pytest.raises(ValueError, match='int64'):
            operation([[x, 0, 0]], return_shifts=True)


def test_unrepresentable_views_and_exact_cancellation_are_operation_local():
    tiny = PeriodicCell(np.diag([2**-1074, 1., 1.]))
    with pytest.raises(ValueError, match='cart_to_fractional.*finite'):
        tiny.cart_to_fractional([[1, 0, 0]])
    assert tiny.wrap_cart([[1, 0, 0]]).tolist() == [[0., 0., 0.]]
    huge = PeriodicCell(np.diag([1e308, 1., 1.]), origin=(-1e308, 0, 0))
    assert huge.cart_to_fractional([[1e308, 0, 0]]).tolist() == [[2., 0., 0.]]
    assert huge.fractional_to_cart([[2., 0, 0]]).tolist() == [[1e308, 0., 0.]]
    with pytest.raises(ValueError, match='fractional_to_cart.*finite'):
        huge.fractional_to_cart([[4., 0, 0]])
    outside = PeriodicCell(np.eye(3) * 1e308, origin=(1e308,) * 3)
    with pytest.raises(ValueError, match='wrap_cart.*finite'):
        outside.wrap_cart([[.9e308] * 3])


METHODS = ['cart_to_fractional', 'fractional_to_cart', 'wrap_fractional', 'wrap_cart']


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('bad', [
    [], [1, 2, 3], [[1, 2]], [[[1, 2, 3]]],
    [[True, 0, 0]], [[1j, 0, 0]], [['1', 0, 0]],
    [[np.inf, 0, 0]], [[np.nan, 0, 0]],
])
def test_strict_coordinate_inputs(method, bad):
    with pytest.raises(ValueError):
        getattr(PeriodicCell(np.eye(3)), method)(bad)


@pytest.mark.parametrize('method', ['wrap_cart', 'wrap_fractional'])
@pytest.mark.parametrize('bad', [0, 1, 'true', None, np.array(True)])
def test_strict_return_shifts(method, bad):
    with pytest.raises(ValueError, match='return_shifts'):
        getattr(PeriodicCell(np.eye(3)), method)([[0, 0, 0]], return_shifts=bad)


@pytest.mark.parametrize('method', METHODS)
def test_owned_outputs_and_empty_batches(method):
    basis, origin = np.eye(3), np.array([.25, 0, 0])
    cell = PeriodicCell(basis, origin)
    operation = getattr(cell, method)
    source = np.array([[.5, .25, .125]])
    expected = operation(source).copy()
    basis[:] = 9
    origin[:] = 8
    result = operation(source)
    assert result.flags.owndata and result.dtype == np.float64
    assert not np.shares_memory(source, result)
    result[:] = 5
    np.testing.assert_array_equal(operation(source), expected)
    source[:] = 6
    np.testing.assert_array_equal(operation([[.5, .25, .125]]), expected)
    empty = operation(np.empty((0, 3)))
    assert empty.shape == (0, 3) and empty.flags.owndata
    if method.startswith('wrap'):
        _, shifts = operation(np.empty((0, 3)), return_shifts=True)
        assert shifts.shape == (0, 3) and shifts.flags.owndata


def test_exact_representable_rotations_translations_and_row_order():
    basis = np.array([[2., .5, 0], [0, 3, .25], [.125, 0, 1]])
    point, origin = np.array([[3.25, -1.5, 2.]]), np.array([.125, .5, 1.])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translated = np.array([4., -2., 1.])
    expected = solve(point[0], origin, basis)
    cell = PeriodicCell(basis @ rotation, origin @ rotation + translated)
    actual = cell.cart_to_fractional(point @ rotation + translated)
    assert actual[0].tolist() == list(map(float, expected))
    for p in permutations(range(3)):
        reordered = PeriodicCell(basis[list(p)], origin)
        actual = reordered.cart_to_fractional(point)
        assert actual[0].tolist() == [float(expected[i]) for i in p]


def test_pure_helpers_do_not_use_native_frame_or_numpy_linalg(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('pure operation requested a numerical frame')
    for name in ('det', 'inv', 'svd', 'qr'):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(PeriodicCell, '_rotation_to_internal', forbidden)
    cell = PeriodicCell(np.diag([-1., 1., 1.]))
    assert cell.cart_to_fractional([[.5, 0, 0]]).tolist() == [[-.5, 0., 0.]]
    assert cell.fractional_to_cart([[.5, 0, 0]]).tolist() == [[-.5, 0., 0.]]
    assert cell.wrap_cart([[.5, 0, 0]]).tolist() == [[-.5, 0., 0.]]
    assert cell.wrap_fractional([[.5, 0, 0]]).tolist() == [[.5, 0., 0.]]


def test_ties_to_even_and_canonical_binary64_inputs():
    cell = PeriodicCell(np.eye(3), origin=(2**-53, 0, 0))
    # Exact affine midpoints round down/up to the even significand.
    f = [[1., 0, 0], [np.nextafter(1., 2.), 0, 0]]
    assert cell.fractional_to_cart(f)[:, 0].tolist() == [1., 1. + 2**-51]
    canonical = PeriodicCell(np.eye(3))
    view, shift = canonical.wrap_cart([[2**53 + 1, 0, 0]], return_shifts=True)
    assert view.tolist() == [[0., 0., 0.]]
    assert shift.tolist() == [[2**53, 0, 0]]  # validated binary64, not source int


def test_repeated_random_batches_match_oracle_and_do_not_mix_origins():
    rng = np.random.default_rng(155)
    for _ in range(16):
        basis = rng.integers(-8, 9, (3, 3)).astype(float) / 4
        if determinant(basis) == 0:
            continue
        points = rng.integers(-100, 101, (8, 3)).astype(float) / 8
        for origin in ([0., 0., 0.], [.125, -.25, .5]):
            cell = PeriodicCell(basis, origin)
            actual, shifts = cell.wrap_cart(points, return_shifts=True)
            for i, point in enumerate(points):
                f = solve(point, origin, basis)
                s = [math.floor(value) for value in f]
                assert shifts[i].tolist() == s
                w = affine([v - t for v, t in zip(f, s)], origin, basis)
                assert actual[i].tolist() == list(map(float, w))


def test_inverse_preparation_is_lazy_bounded_and_reused():
    from pyvoro2 import compute
    from pyvoro2._internal import exact_user_basis as kernel

    kernel._inverse_basis.cache_clear()
    cell = PeriodicCell.from_params(2., .25, 2., 0., 0., 2.)
    compute([[.25, .25, .25]], domain=cell)
    cell.fractional_to_cart([[.25, .25, .25]])
    cell.wrap_fractional([[.25, .25, .25]])
    assert kernel._inverse_basis.cache_info().misses == 0
    cell.cart_to_fractional([[.25, .25, .25]])
    cell.wrap_cart([[.25, .25, .25]])
    info = kernel._inverse_basis.cache_info()
    assert info.misses == 1 and info.hits == 1
    assert info.maxsize == 128
