"""Independent exact-operand oracles for local source projection (F1-A3-P)."""
from fractions import Fraction as F

import numpy as np
import pytest

from pyvoro2 import PeriodicCell
from pyvoro2._internal import native_remap
from pyvoro2._internal.generator_preparation import _prepare_coordinates
from pyvoro2._internal.spatial.domain_geometry import geometry3d


def source_oracle(point, basis, origin, rotation):
    """Solve A.T f.T = (p-o).T by rational elimination, independently of A1."""
    delta = [F(float(p)) - F(float(o)) for p, o in zip(point, origin)]
    system = [[F(float(basis[j][i])) for j in range(3)] + [delta[i]]
              for i in range(3)]
    for col in range(3):
        pivot = next(i for i in range(col, 3) if system[i][col])
        system[col], system[pivot] = system[pivot], system[col]
        scale = system[col][col]
        system[col] = [v / scale for v in system[col]]
        for i in range(3):
            if i != col:
                scale = system[i][col]
                system[i] = [a - scale*b for a, b in zip(system[i], system[col])]
    k = [row[-1].numerator // row[-1].denominator for row in system]
    r = [delta[j] - sum(k[i] * F(float(basis[i][j])) for i in range(3))
         for j in range(3)]
    u = [float(sum(r[i] * F(float(row[i])) for i in range(3))) for row in rotation]
    return u, k, r


def chart_oracle(point, params, eps):
    bx, bxy, by, bxz, byz, bz = map(F, params)
    rows = [(bx, 0, 0), (bxy, by, 0), (bxz, byz, bz)]
    if eps is None:
        # Only diagonal lengths set R's default, irrespective of shear.
        scale = max(params[0], params[2], params[5])
        eps = max(1e-12 * scale, 64 * np.finfo(float).eps * scale)
    remaining, output, shifts = list(map(F, point)), [0.] * 3, [0] * 3
    for axis in (2, 1, 0):
        length = rows[axis][axis]
        ratio = remaining[axis] / length
        q = ratio.numerator // ratio.denominator
        value = float(remaining[axis] - q * length)
        if F(value) == length or (eps > 0 and length - F(value) <= F(float(eps))):
            value, q = 0., q + 1
        elif eps > 0 and F(value) < F(float(eps)):
            value = 0.
        output[axis], shifts[axis] = value, q
        remaining = [v - q*a for v, a in zip(remaining, rows[axis])]
    return output, shifts


BASE = np.array([[0., -1., -1.], [1., 0., 0.], [-1., -1., 0.]])


@pytest.mark.parametrize('basis', [
    BASE, BASE[[1, 0, 2]], -BASE,
    [[2., .5, -.25], [.25, 1., .5], [0., .25, 1.]],
])
@pytest.mark.parametrize('eps', [None, 0., 2.**-38])
@pytest.mark.parametrize('route', ['public', 'snapshot'])
def test_full_local_projection_and_chart_match_fraction_oracles(
    monkeypatch, basis, eps, route,
):
    import pyvoro2.domains as domains

    domain = PeriodicCell(basis, origin=(1.25, -.5, .125))
    snapshot = geometry3d(domain).native_periodic_snapshot()
    points = np.array([[1.5, 0., .625], [-19.125, 7.25, -5.5],
                       [1.25, -.5, .125], [2.**45 + .125, -2.**42, .25]])
    original = points.copy()
    seen = []
    frames = []
    prepare_frame = domains.prepare_native_periodic_frame
    remap = native_remap.remap_triangular

    def frame(vectors):
        result = prepare_frame(vectors)
        frames.append(result)
        return result

    def record(points, params, eps):
        result = remap(points, params, eps)
        seen.append((points.copy(), params, result[0].copy(), result[1]))
        return result

    monkeypatch.setattr(native_remap, 'remap_triangular', record)
    monkeypatch.setattr(domains, 'prepare_native_periodic_frame', frame)
    adapter = domain if route == 'public' else snapshot
    primary, shifts = adapter.remap_cart(points, eps=eps, return_shifts=True)
    assert len(seen) == 1
    assert len(frames) == (1 if route == 'public' else 0)
    actual_frame = frames[0] if frames else snapshot
    u, params, v, n = seen[0]
    assert params == actual_frame.params
    for index, point in enumerate(points):
        expected_u, k, _ = source_oracle(
            point, domain.vectors, domain.origin, actual_frame.rotation_to_internal)
        expected_v, expected_n = chart_oracle(expected_u, params, eps)
        np.testing.assert_array_equal(u[index], expected_u)
        np.testing.assert_array_equal(v[index], expected_v)
        assert tuple(n[index]) == tuple(expected_n)
        assert shifts[index].tolist() == [a+b for a, b in zip(k, expected_n)]
    expected_report = (actual_frame.rotation_to_internal.T @ v.T).T + domain.origin
    np.testing.assert_array_equal(primary, expected_report)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize('route', ['public', 'snapshot', 'preparation'])
@pytest.mark.parametrize('expose', [False, True])
@pytest.mark.parametrize('sign', [-1, 1])
def test_only_final_combined_shift_has_int64_boundary(route, expose, sign):
    domain = PeriodicCell(np.eye(3), origin=(2.**-55, 0., 0.))
    geometry = geometry3d(domain)
    snapshot = geometry.native_periodic_snapshot()
    point = np.array([[sign * 2.**63, 0., 0.]])
    u, k, r = source_oracle(point[0], domain.vectors, domain.origin, np.eye(3))
    assert k[0] == sign * 2**63 - 1
    assert r[0] == 1 - F(1, 2**55)
    assert u == [1., 0., 0.]

    def run():
        if route == 'preparation':
            _, cart, _, shifts = _prepare_coordinates(
                point, geometry=geometry, periodic_snapshot=snapshot, name='points')
            return (cart, shifts) if expose else cart
        adapter = domain if route == 'public' else snapshot
        return adapter.remap_cart(point, eps=0., return_shifts=expose)

    with np.errstate(all='raise'):
        if sign > 0:
            with pytest.raises(ValueError, match='int64'):
                run()
        else:
            result = run()
            cart, shifts = result if expose else (result, None)
            np.testing.assert_array_equal(cart, [[2.**-55, 0., 0.]])
            if expose:
                assert shifts.tolist() == [[-2**63, 0, 0]]


@pytest.mark.parametrize('basis', [BASE, BASE[[1, 0, 2]], -BASE])
def test_exact_translation_native_invariance_across_batch_shape_and_order(basis):
    domain = PeriodicCell(basis, origin=(1.25, -.5, .125))
    geometry = geometry3d(domain)
    snapshot = geometry.native_periodic_snapshot()
    local = np.array([[1.5, 0., .625], [1.75, 0., .625], [2., 0., .625]])
    translations = np.array([[2**48, 0, 0], [-7, 2, 3], [-2**48, 0, 0]])
    lifted = local + translations @ basis
    for index in range(3):
        for j in range(3):
            assert F(lifted[index, j]) == F(local[index, j]) + sum(
                int(translations[index, i]) * F(float(basis[i][j])) for i in range(3))

    def prepare(points):
        return _prepare_coordinates(points, geometry=geometry,
                                    periodic_snapshot=snapshot, name='points')[2:]

    v, k = prepare(local)
    for order in ([0, 1, 2], [2, 0, 1]):
        actual, shifts = prepare(lifted[order])
        np.testing.assert_array_equal(actual, v[order])
        np.testing.assert_array_equal(shifts, k[order] + translations[order])
    for i in range(3):
        actual, shifts = prepare(lifted[i:i+1])
        np.testing.assert_array_equal(actual, v[i:i+1])
        np.testing.assert_array_equal(shifts, k[i:i+1] + translations[i:i+1])
    actual, shifts = prepare(np.concatenate((local, lifted, local[::-1])))
    np.testing.assert_array_equal(actual, np.concatenate((v, v, v[::-1])))
    np.testing.assert_array_equal(shifts, np.concatenate((k, k+translations, k[::-1])))


@pytest.mark.parametrize('route', ['public', 'snapshot'])
@pytest.mark.parametrize('eps', [None, 0., 2.**-38])
def test_empty_shapes_seams_and_owned_views(route, eps):
    domain = PeriodicCell(np.eye(3))
    adapter = (domain if route == 'public'
               else geometry3d(domain).native_periodic_snapshot())
    empty, shifts = adapter.remap_cart(np.empty((0, 3)), eps=eps, return_shifts=True)
    assert empty.shape == shifts.shape == (0, 3)
    assert empty.dtype == np.float64 and shifts.dtype == np.int64
    points = np.array([[0., 1., -1.], [-2.**-55, 2.**-41, .5]])
    actual, shifts = adapter.remap_cart(points, eps=eps, return_shifts=True)
    expected_y = 2.**-41 if eps == 0. else 0.
    np.testing.assert_array_equal(actual, [[0., 0., 0.], [0., expected_y, .5]])
    np.testing.assert_array_equal(shifts, [[0, 1, -1], [0, 0, 0]])
    assert not np.shares_memory(actual, points)
    assert points[1, 0] == -2.**-55


def test_nonfinite_local_projection_refuses_without_runtime_warning():
    from dataclasses import replace

    snapshot = geometry3d(PeriodicCell(np.eye(3))).native_periodic_snapshot()
    # Fault injection at the private projection boundary, not a relaxation of
    # A2 admission: the complete exact dot exceeds the finite binary64 range.
    snapshot = replace(snapshot, rotation_to_internal=np.full(
        (3, 3), np.finfo(float).max))
    with np.errstate(all='raise'), pytest.raises(ValueError, match='finite'):
        snapshot.remap_cart([[.75, .75, .75]])


@pytest.mark.parametrize('alternate_qr', [False, True])
def test_residual_is_not_rounded_before_the_complete_dot(monkeypatch, alternate_qr):
    if alternate_qr:
        # An independently rounded, also A2-qualified QR pair. A projection
        # oracle must consume the operation frame, not pin one LAPACK's bits.
        s = 1 / np.sqrt(2.)
        q = np.array([[0., 1., 0.], [-s, 0., -s], [-s, 0., s]])
        monkeypatch.setattr(np.linalg, 'qr', lambda a: (q, q.T @ a))
    domain = PeriodicCell(BASE, origin=(2.**-55, 2.**-54, 2.**-56))
    snapshot = geometry3d(domain).native_periodic_snapshot()
    point = [.125, .125, .125]
    expected, _, residual = source_oracle(
        point, domain.vectors, domain.origin, snapshot.rotation_to_internal)
    prematurely_rounded = [
        float(sum(F(float(residual[i])) * F(float(row[i])) for i in range(3)))
        for row in snapshot.rotation_to_internal]
    assert expected != prematurely_rounded
    seen = []
    remap = native_remap.remap_triangular

    def capture(points, params, eps):
        seen.append(points.copy())
        return remap(points, params, eps)

    monkeypatch.setattr(native_remap, 'remap_triangular', capture)
    snapshot.remap_cart([point])
    np.testing.assert_array_equal(seen[0], [expected])


@pytest.mark.parametrize('route', ['public', 'snapshot'])
@pytest.mark.parametrize('kwargs', [
    {'return_shifts': 1}, {'eps': True}, {'eps': -1.}, {'eps': float('nan')},
])
def test_stable_wrappers_keep_strict_flags_and_epsilon(route, kwargs):
    domain = PeriodicCell(np.eye(3))
    adapter = (domain if route == 'public'
               else geometry3d(domain).native_periodic_snapshot())
    with pytest.raises(ValueError):
        adapter.remap_cart([[.25, .25, .25]], **kwargs)


@pytest.mark.parametrize('route', ['public', 'snapshot'])
def test_empty_stable_remap_still_validates_snap_overlap(route):
    domain = PeriodicCell(np.diag([1., 1., 2.**-40]))
    with pytest.warns(RuntimeWarning, match='ill-conditioned') as caught:
        adapter = (domain if route == 'public'
                   else geometry3d(domain).native_periodic_snapshot())
        with pytest.raises(ValueError, match='2 \\* eps'):
            adapter.remap_cart(np.empty((0, 3)))
    assert len(caught) == 1
