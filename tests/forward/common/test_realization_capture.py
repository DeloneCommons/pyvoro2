"""Private capture shares the public compute worker and its native invocation."""
import inspect
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic', 'triclinic'])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_matching_capture_uses_one_preparation_and_native_call(monkeypatch, kind, mode):
    import pyvoro2.api as spatial
    import pyvoro2.planar.api as planar
    from pyvoro2 import OrthorhombicCell, PeriodicCell
    from pyvoro2.planar import RectangularCell

    dim = 2 if kind == 'planar' else 3
    api = planar if dim == 2 else spatial
    domain = (RectangularCell(((0., 1.),) * 2) if dim == 2 else
              OrthorhombicCell(((0., 1.),) * 3) if kind == 'orthorhombic' else
              PeriodicCell(((1., 1., 0.), (-1., 1., 0.), (0., 0., 1.)),
                           origin=(2., -1., .25)))
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, 3.375]
    native = api._require_core2d() if dim == 2 else api._require_core()
    name = f'compute_{"periodic" if kind == "triclinic" else "box"}_{mode}'
    actual_compute = getattr(native, name)
    actual_prepare = api.prepare_generators
    calls = []
    preparations = []

    def prepare(*args, **kwargs):
        result = actual_prepare(*args, **kwargs)
        preparations.append(result)
        return result

    def compute(*args):
        cells = actual_compute(*args)
        calls.append((args, {c['id']: list(c['site']) for c in cells}))
        return cells[::-1]

    monkeypatch.setattr(api, 'prepare_generators', prepare)
    monkeypatch.setattr(native, name, compute)
    frame_calls = []
    if kind == 'triclinic':
        from pyvoro2._internal.spatial import domain_geometry
        actual_frame = domain_geometry.prepare_native_periodic_frame

        def frame(*args):
            frame_calls.append(1)
            return actual_frame(*args)

        monkeypatch.setattr(domain_geometry, 'prepare_native_periodic_frame', frame)
    kwargs = dict(domain=domain, mode=mode, ids=[30, 10], output='cells')
    if mode == 'power':
        kwargs['weights'] = [0., 0.]
    output, ref = api._compute_for_matching(points, **kwargs)
    assert len(preparations) == len(calls) == 1
    assert len(frame_calls) == (1 if kind == 'triclinic' else 0)
    assert {c['id'] for c in output} == {10, 30}
    np.testing.assert_array_equal(ref.submitted, calls[0][0][0])
    np.testing.assert_array_equal(ref.preparation_shifts, preparations[0].remap_shifts)
    np.testing.assert_array_equal(ref.anchors, [calls[0][1][i] for i in range(2)])
    parameter_index = 3 if mode == 'power' else 2
    params = calls[0][0][parameter_index]
    if kind == 'triclinic':
        bx, bxy, by, bxz, byz, bz = params
        expected_lattice = [[bx, 0, 0], [bxy, by, 0], [bxz, byz, bz]]
    else:
        expected_lattice = np.diag([hi - lo for lo, hi in params])
    np.testing.assert_array_equal(ref.lattice, expected_lattice)
    output[0]['site'][0] = 99
    points[:] = 0
    assert not np.any(ref.anchors == 99)
    assert not ref.anchors.flags.writeable
    assert 'capture_reference' not in inspect.signature(api.compute).parameters

    def forbidden_capture(*args, **kwargs):
        pytest.fail('ordinary compute must not capture realization evidence')

    monkeypatch.setattr(api, 'capture_reference', forbidden_capture)
    # A separate ordinary invocation must still use the same public schema.
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, 3.375]
    ordinary = api.compute(points, **kwargs)
    assert isinstance(ordinary, list)
    assert {c['id'] for c in ordinary} == {10, 30}
    assert all('reference' not in c for c in ordinary)


@pytest.mark.parametrize('dim', [2, 3])
def test_hidden_cell_is_not_native_anchor_evidence(dim):
    import pyvoro2.api as spatial
    import pyvoro2.planar.api as planar
    from pyvoro2 import OrthorhombicCell
    from pyvoro2.planar import RectangularCell
    from pyvoro2.inverse.separator import (
        match_realized_pairs, resolve_separator_observations,
    )

    api = planar if dim == 2 else spatial
    domain = (RectangularCell if dim == 2 else OrthorhombicCell)(((0., 1.),) * dim)
    points = np.full((2, dim), .25)
    points[:, 0] = [.125, .375]
    cells, ref = api._compute_for_matching(
        points, domain=domain, mode='power', weights=[0., 10.],
        include_empty=True, output='cells')
    assert ref.anchor_present.tolist() == [False, True]
    assert next(c for c in cells if c['id'] == 0)['empty']
    obs = resolve_separator_observations(points, [(0, 1, .5)], domain=domain)
    result = match_realized_pairs(
        points, domain=domain, constraints=obs, weights=[0., 10.])
    assert result.endpoint_i_empty.tolist() == [True]
    assert result.realized.tolist() == [False]
    assert result.realized_other_shift.tolist() == [False]


@pytest.mark.parametrize('kind', ['planar', 'orthorhombic', 'triclinic'])
@pytest.mark.parametrize('shear', [0., float(1/3)])
def test_same_operation_native_endpoint_readback(monkeypatch, kind, shear):
    import pyvoro2.api as spatial
    import pyvoro2.planar.api as planar
    from pyvoro2 import OrthorhombicCell, PeriodicCell
    from pyvoro2.planar import RectangularCell
    from pyvoro2._internal.inputs import owned_readonly_array

    dim = 2 if kind == 'planar' else 3
    api = planar if dim == 2 else spatial
    period = 105/1024
    basis = np.eye(dim)
    basis[1, 1] = period
    if kind == 'triclinic':
        basis[1, 0] = shear
        domain = PeriodicCell(basis)
    else:
        bounds = ((0., 1.), (0., period)) + ((0., 1.),) * (dim - 2)
        domain = (RectangularCell if dim == 2 else OrthorhombicCell)(bounds)
    submitted = np.full((2, dim), .25)
    submitted[:, 0] = [.5, .125]
    submitted[:, 1] = [period/4, np.nextafter(period, 0.)]
    actual_prepare = api.prepare_generators

    def prepare(*args, **kwargs):
        prepared = actual_prepare(*args, **kwargs)
        # Force a valid zero-epsilon half-open submission at the native seam.
        # Normal default snapping masks this native insertion witness. Keep
        # all other preflight and the one actual compiled invocation intact.
        return replace(
            prepared, native_points=owned_readonly_array(submitted),
            remap_shifts=owned_readonly_array(np.zeros((2, dim), dtype=np.int64)))

    monkeypatch.setattr(api, 'prepare_generators', prepare)
    _, ref = api._compute_for_matching(
        submitted, domain=domain, mode='power', weights=[0., 0.],
        blocks=(1,) * dim, output='cells')
    # Do not pin a compiler-dependent transition: establish its exact relative
    # membership independently from this invocation's genuine stored anchors.
    residual = [F(float(submitted[1, k])) - F(float(submitted[0, k]))
                - F(float(ref.anchors[1, k])) + F(float(ref.anchors[0, k]))
                for k in range(dim)]
    h = [F(0)] * dim
    for k in reversed(range(dim)):
        higher = sum(h[t] * F(float(basis[t, k])) for t in range(k + 1, dim))
        h[k] = (residual[k] - higher) / F(float(basis[k, k]))
    if all(v.denominator == 1 for v in h):
        assert ref.shift_offset(0, 1) == tuple(-int(v) for v in h)
    else:
        with pytest.raises(ValueError, match='nonintegral'):
            ref.shift_offset(0, 1)
