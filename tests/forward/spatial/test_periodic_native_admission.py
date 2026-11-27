"""Native admission remains narrower than exact user-lattice validity."""
import warnings

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api
from pyvoro2._internal.spatial.domain_geometry import _NativePeriodicSnapshot


REJECTED = [np.diag([1., 1., 1e-16]),
            [[1., 0, 0], [0, 1, 0], [1e-12, 1e-12, 1e-25]],
            np.eye(3) * 1e-110, np.eye(3) * 1e110]


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_overlapping_default_snap_bands_refuse_before_dispatch(
    monkeypatch, operation, mode,
):
    class ForbiddenCore:
        def __getattr__(self, name):
            raise AssertionError(f'unsafe native dispatch: {name}')

    monkeypatch.setattr(api, '_core', ForbiddenCore())
    monkeypatch.setattr(api, '_CORE_IMPORT_ERROR', None)
    cell = pv.PeriodicCell(np.diag([1., 1., 1e-13]))
    options = dict(domain=cell, mode=mode, blocks=(1, 1, 1))
    if mode == 'power':
        options['radii'] = [0.]
        if operation == 'ghost_cells':
            options['ghost_radius'] = 0.
    args = ([[.25, .25, 0.]],)
    if operation != 'compute':
        args += ([[.5, .5, 0.]],)
    with pytest.warns(RuntimeWarning, match='native adapter'):
        with pytest.raises(ValueError, match='2.*eps|neighborhood'):
            getattr(pv, operation)(*args, **options)


@pytest.mark.parametrize('basis', REJECTED)
@pytest.mark.parametrize('method', ['to_internal_params', 'cart_to_internal',
                                    'internal_to_cart', 'remap_internal',
                                    'remap_cart', 'wrap_internal', 'snapshot'])
def test_backend_helpers_refuse_before_frame_work(basis, method):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cell = pv.PeriodicCell(basis)
        assert cell.wrap_fractional([[.5, 0, 0]]).tolist() == [[.5, 0., 0.]]
        with pytest.raises(ValueError, match='native'):
            if method == 'snapshot':
                _NativePeriodicSnapshot.from_raw(
                    vectors=cell.vectors, origin=cell.origin)
            elif method == 'to_internal_params':
                cell.to_internal_params()
            else:
                getattr(cell, method)([[0, 0, 0]])


@pytest.mark.parametrize('basis', REJECTED)
@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_native_routes_refuse_without_dispatch(monkeypatch, basis, operation, mode):
    class ForbiddenCore:
        def __getattr__(self, name):
            raise AssertionError(f'unsafe native dispatch: {name}')
    monkeypatch.setattr(api, '_core', ForbiddenCore())
    monkeypatch.setattr(api, '_CORE_IMPORT_ERROR', None)
    cell = pv.PeriodicCell(basis)
    kwargs = dict(domain=cell, mode=mode)
    if mode == 'power':
        kwargs['radii'] = [0.]
        if operation == 'ghost_cells':
            kwargs['ghost_radius'] = 0.
    args = ([[.25, .25, .25]],)
    if operation != 'compute':
        args += ([[.5, .5, .5]],)
    with pytest.raises(ValueError, match='native'):
        getattr(pv, operation)(*args, **kwargs)


@pytest.mark.parametrize('basis', [
    np.eye(3), [[0., -1., 0], [1., 0, 0], [0, 0, 1.]],
    [[1., 0, 0], [-.5, 1., 0], [0, 0, 1.]],
])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_compiled_native_volume_owner_and_ghost_controls(basis, mode):
    # These lattices generate Z^3 or a unit-volume shear, without changing rows.
    cell = pv.PeriodicCell(basis)
    points = [[.25, .25, .25]]
    kwargs = dict(domain=cell, mode=mode, ids=[19])
    if mode == 'power':
        kwargs['radii'] = [.125]
    cells = pv.compute(points, output='cells', **kwargs)
    assert len(cells) == 1 and cells[0]['id'] == 19
    assert cells[0]['volume'] == pytest.approx(1., abs=1e-12)
    located = pv.locate(points, [[.375, .25, .25]], **kwargs)
    assert located['owner_id'].tolist() == [19]
    ghost_options = {'ghost_radius': .125} if mode == 'power' else {}
    ghosts = pv.ghost_cells(points, [[.75, .75, .75]], **kwargs, **ghost_options)
    assert len(ghosts) == 1
    assert ghosts[0]['volume'] == pytest.approx(.5, abs=1e-12)


def test_svd_failure_is_native_refusal_only(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('did not converge')
    monkeypatch.setattr(np.linalg, 'svd', fail)
    cell = pv.PeriodicCell(np.eye(3))
    assert cell.cart_to_fractional([[.25, 0, 0]]).tolist() == [[.25, 0., 0.]]
    with pytest.raises(ValueError, match='native.*conditioning'):
        cell.to_internal_params()


def test_signed_diagonal_now_passes_independent_image_consumer():
    cell = pv.PeriodicCell(np.diag([-1., -1., 1.]))
    assert cell.to_internal_params() == (1., 0., 1., 0., 0., 1.)
    assert cell.wrap_cart([[.25, .25, .25]]).tolist() == [[-.75, -.75, .25]]
    result = pv.compute([[.25, .25, .25]], domain=cell)
    assert result.cell_measures[0] == pytest.approx(1.)


def test_conditioning_warning_as_error_stops_before_dispatch(monkeypatch):
    class ForbiddenCore:
        def __getattr__(self, name):
            raise AssertionError(f'native dispatch: {name}')
    monkeypatch.setattr(api, '_core', ForbiddenCore())
    cell = pv.PeriodicCell(np.diag([1., 1., 1e-11]))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(RuntimeWarning, match='native adapter'):
            pv.compute([[.25, .25, 0]], domain=cell)


@pytest.mark.parametrize('route', ['to_internal_params', 'snapshot', 'compute'])
def test_underflowed_native_norm_product_refuses_cleanly(route):
    # Rounded norms have product zero although LAPACK returns a subnormal det.
    tiny = 1.1 * 2.0**-537
    cell = pv.PeriodicCell(np.diag([tiny, tiny, .49]))
    assert cell.cart_to_fractional([[0, 0, 0]]).tolist() == [[0., 0., 0.]]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='native'):
            if route == 'to_internal_params':
                cell.to_internal_params()
            elif route == 'snapshot':
                _NativePeriodicSnapshot.from_raw(
                    vectors=cell.vectors, origin=cell.origin)
            else:
                pv.compute([[0, 0, 0]], domain=cell)
