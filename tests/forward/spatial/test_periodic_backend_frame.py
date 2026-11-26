"""A2 frame qualification against analytic and independent rational oracles."""
from fractions import Fraction as F
from itertools import permutations, product
import math
import warnings

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api
from pyvoro2._internal.spatial.domain_geometry import _NativePeriodicSnapshot


GATE = 60 * 2**-53


def determinant(a):
    return sum((-1)**sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
               * math.prod(F(float(a[i][p[i]])) for i in range(3))
               for p in permutations(range(3)))


def lower(params):
    bx, bxy, by, bxz, byz, bz = params
    return np.array([[bx, 0, 0], [bxy, by, 0], [bxz, byz, bz]])


def snapshot(a, origin=(.25, -.5, 1.)):
    return _NativePeriodicSnapshot.from_raw(vectors=a, origin=origin)


@pytest.mark.parametrize('signs', list(product((-1., 1.), repeat=3)))
@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_signed_permutations_have_analytic_frame_and_row_identity(signs, order):
    # Each row has one nonzero component: its direction is the exact Q column.
    a = np.diag(np.array(signs) * [2, 3, 4])[:, order]
    cell = pv.PeriodicCell(a, origin=(.25, -.5, 1.))
    frame = snapshot(a)
    expected_c = a / np.array([2, 3, 4])[:, None]
    np.testing.assert_array_equal(frame.rotation_to_internal, expected_c)
    np.testing.assert_array_equal(lower(frame.params), np.diag([2, 3, 4]))
    assert frame.parity == (1 if determinant(a) > 0 else -1)
    assert frame.lengths_and_volume == ((2., 3., 4.), 24.)
    assert cell.vectors == tuple(map(tuple, a))
    points = np.array([[1.25, -.25, 2.], [-2., .5, 8.]])
    expected = (points - frame.origin) @ expected_c.T
    np.testing.assert_array_equal(cell.cart_to_internal(points), expected)
    np.testing.assert_array_equal(frame.internal_to_cart(expected), points)
    assert cell.to_internal_params() == frame.params


@pytest.mark.parametrize('parity', [-1, 1])
@pytest.mark.parametrize('scale', [1e-100, 1., 1e100])
def test_analytic_rotated_shear_has_positive_paired_factor(parity, scale):
    # Analytic proper rotation, then optional reflection; no QR used as oracle.
    c = np.array([[.6, .8, 0], [-.8, .6, 0], [0, 0, parity]])
    triangular = np.array([[2., 0, 0], [.25, 3., 0], [-.5, .75, 4.]]) * scale
    a = triangular @ c
    frame = snapshot(a)
    np.testing.assert_allclose(frame.rotation_to_internal, c, rtol=0, atol=1e-15)
    np.testing.assert_allclose(lower(frame.params) / scale, triangular / scale,
                               rtol=2e-15, atol=1e-15)
    assert frame.parity == parity
    assert np.linalg.norm(frame.rotation_to_internal @ frame.rotation_to_internal.T
                          - np.eye(3), ord=np.inf) <= GATE


@pytest.mark.parametrize('power', [30, 38])
def test_near_parallel_isometry_and_backward_error_not_short_metric_certificate(power):
    a = np.array([[1., 1., 0], [1., 1 + 2.**-power, 0], [0, 0, 1.]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        frame = snapshot(a)
        c = pv.PeriodicCell(a)._rotation_to_internal()
    triangular = lower(frame.params)
    assert determinant(a) == F(1, 2**power)
    assert np.linalg.norm(c @ c.T - np.eye(3), ord=np.inf) <= GATE
    residual = np.max(np.sum(abs(triangular - a @ c.T),
                      axis=1) / np.sum(abs(a), axis=1))
    assert residual <= GATE
    # Independent exact arithmetic measures the emitted short translation.
    s = [-1, 1, 0]
    short = [sum(s[i] * F(float(triangular[i, j])) for i in range(3)) for j in range(3)]
    relative_squared_error = abs(sum(v*v for v in short) / F(1, 2**(2*power)) - 1)
    # The 2^-30 gross GS nonisometry/volume blowup is gone. A deliberately
    # condition-scaled bound does NOT impose O(u) relative short-vector error.
    assert relative_squared_error < F(1, 1000)
    if power == 30:
        assert abs(F(frame.lengths_and_volume[1]) / determinant(a) - 1) < F(1, 10**6)


@pytest.mark.parametrize(
    'operation', ['compute', 'locate', 'ghost_cells', 'params', 'snapshot'])
@pytest.mark.parametrize('fault', ['qr_failure', 'svd_failure', 'q_nan', 'r_inf',
                                   'rank_loss', 'parity', 'orthogonality',
                                   'backward', 'volume_underflow', 'volume_overflow'])
def test_frame_failure_is_controlled_and_never_dispatches(
    monkeypatch, operation, fault,
):
    class ForbiddenCore:
        def __getattr__(self, name):
            raise AssertionError(f'unsafe dispatch: {name}')
    monkeypatch.setattr(api, '_core', ForbiddenCore())
    monkeypatch.setattr(api, '_CORE_IMPORT_ERROR', None)
    cell = pv.PeriodicCell(np.eye(3))

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('injected')
    if fault.endswith('failure'):
        monkeypatch.setattr(np.linalg, fault.split('_')[0], fail)
    else:
        q, r = np.eye(3), np.eye(3)
        if fault == 'q_nan':
            q[0, 0] = np.nan
        elif fault == 'r_inf':
            r[0, 1] = np.inf
        elif fault == 'rank_loss':
            r[1, 1] = 0
        elif fault == 'parity':
            q[0, 0] = -1
        elif fault == 'orthogonality':
            q[0, 1] = 1e-10
        elif fault == 'backward':
            r[0, 1] = 1e-10
        elif fault == 'volume_underflow':
            r *= 1e-150
        elif fault == 'volume_overflow':
            r *= 1e150
        monkeypatch.setattr(np.linalg, 'qr', lambda *args, **kwargs: (q, r))
    expected_failure = {
        'qr_failure': 'native frame QR failed',
        'svd_failure': 'native conditioning check failed',
        'q_nan': 'native frame QR factors must be finite',
        'r_inf': 'native frame QR factors must be finite',
        'rank_loss': 'native frame QR has a zero diagonal',
        'parity': 'native frame parity',
        'orthogonality': 'native frame orthogonality',
        'backward': 'native frame backward consistency',
        'volume_underflow': 'native frame volume',
        'volume_overflow': 'native frame volume',
    }[fault]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=expected_failure):
            if operation == 'params':
                cell.to_internal_params()
            elif operation == 'snapshot':
                snapshot(cell.vectors)
            else:
                args = ([[.25, .25, .25]],)
                if operation != 'compute':
                    args += ([[.5, .5, .5]],)
                getattr(pv, operation)(*args, domain=cell)
    assert cell.wrap_fractional([[1.25, 0, 0]]).tolist() == [[.25, 0., 0.]]


def test_frame_and_snapshot_own_readonly_arrays_and_keep_nan_owner_sentinel():
    a, o = np.diag([-2., 3., 4.]), np.array([.25, -.5, 1.])
    frame = _NativePeriodicSnapshot.from_raw(vectors=a, origin=o)
    a[:] = 8
    o[:] = 9
    for array in (frame.vectors, frame.origin, frame.rotation_to_internal):
        assert array.flags.owndata and not array.flags.writeable
    np.testing.assert_array_equal(frame.vectors, np.diag([-2., 3., 4.]))
    np.testing.assert_array_equal(frame.origin, [.25, -.5, 1.])
    assert isinstance(frame.params, tuple)
    assert np.isnan(frame.internal_to_cart([[np.nan] * 3])).all()


def test_sign_normalization_changes_raw_improper_q_as_a_pair():
    a = np.array([[1., 1., 0], [-1., 1., 0], [0, 0, 2.]])
    q0, r0 = np.linalg.qr(a.T)
    assert np.linalg.det(q0) < 0 < float(determinant(a))
    frame = snapshot(a)
    assert np.linalg.det(frame.rotation_to_internal) > 0
    np.testing.assert_allclose(
        a.T, frame.rotation_to_internal.T @ lower(frame.params).T,
        rtol=0, atol=1e-15)
    assert np.any(np.diag(r0) < 0)


def test_parameters_come_from_paired_factor_without_recomputing_source(
    monkeypatch,
):
    r = np.eye(3)
    r[0, 1] = 2**-52  # Within the approved backward gate.
    monkeypatch.setattr(np.linalg, 'qr', lambda a: (np.eye(3), r))
    assert snapshot(np.eye(3)).params[1] == 2**-52
    assert pv.PeriodicCell(np.eye(3)).to_internal_params()[1] == 2**-52


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_forward_prepares_one_frame_and_warns_once(monkeypatch, operation, mode):
    # A spy backend isolates the preparation lifetime from independent native
    # construction limits on this strongly anisotropic lattice.
    calls = []
    qr = np.linalg.qr

    def counted_qr(a):
        calls.append(a.copy())
        return qr(a)

    class Core:
        def __getattr__(self, name):
            def dispatch(*args):
                if name.startswith('locate'):
                    return np.array([False]), np.array([-1]), np.full((1, 3), np.nan)
                if name.startswith('ghost'):
                    return [dict(id=-1, query_index=0, site=[.5, .5, 0.],
                                 volume=0., empty=True)]
                return [dict(id=0, site=[.25, .25, 0.], volume=1.)]
            return dispatch

    monkeypatch.setattr(np.linalg, 'qr', counted_qr)
    monkeypatch.setattr(api, '_core', Core())
    cell = pv.PeriodicCell(np.diag([-1., 1., 2**-34]))
    args = ([[.25, .25, 0]],)
    opts = dict(domain=cell, blocks=(1, 1, 1), mode=mode)
    if operation != 'compute':
        args += ([[.5, .5, 0]],)
    if operation == 'compute':
        opts['output'] = 'cells'
    if operation == 'locate':
        opts['return_owner_position'] = True
    if mode == 'power':
        opts['radii'] = [0.]
        if operation == 'ghost_cells':
            opts['ghost_radius'] = 0.
    with pytest.warns(RuntimeWarning, match='native adapter') as caught:
        result = getattr(pv, operation)(*args, **opts)
    assert len(caught) == len(calls) == 1
    if operation == 'locate':
        assert result['owner_id'].tolist() == [-1]
        assert np.isnan(result['owner_pos']).all()


@pytest.mark.parametrize('fault', ['svd_zero', 'svd_nan', 'det_zero', 'det_wrong_sign'])
def test_invalid_admission_numerics_refuse_without_source_invalidation(
    monkeypatch, fault,
):
    cell = pv.PeriodicCell(np.diag([-1., 1., 1.]))
    if fault.startswith('svd'):
        values = np.array([1., 1., 0 if fault == 'svd_zero' else np.nan])
        monkeypatch.setattr(np.linalg, 'svd', lambda *a, **k: values)
    else:
        monkeypatch.setattr(
            np.linalg, 'det', lambda *a: 0 if fault == 'det_zero' else 1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='native'):
            snapshot(cell.vectors)
    assert cell.wrap_fractional([[1.25, 0, 0]]).tolist() == [[.25, 0., 0.]]
