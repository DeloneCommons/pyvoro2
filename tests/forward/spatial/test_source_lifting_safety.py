"""Preserve both insertion safety stages and the separate bounded query route."""
from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2 as pv
import pyvoro2.api as api
from pyvoro2.duplicates import DuplicateError
from pyvoro2._internal.generator_preparation import _prepare_coordinates
from pyvoro2._internal.spatial.domain_geometry import geometry3d


@pytest.mark.parametrize('operation', ['compute', 'locate', 'ghost_cells'])
@pytest.mark.parametrize('scale', [1, 5])
def test_large_origin_keeps_prepared_and_actual_native_safety(
    monkeypatch, operation, scale,
):
    domain = pv.PeriodicCell(np.diag([scale, 1., 1.]), origin=(2.**54, 0., 0.))
    points = np.full((2, 3), .25)
    points[:, 0] = [.25, .5] if scale == 1 else [1-2.**-22, 1+2.**-22]
    geometry = geometry3d(domain)
    snapshot = geometry.native_periodic_snapshot()
    _, cart, native, shifts = _prepare_coordinates(
        points, geometry=geometry, periodic_snapshot=snapshot, name='points')
    for i in range(2):
        exact = F(points[i, 0]) - 2**54
        quotient = exact // scale
        assert native[i, 0] == float(exact - quotient * scale)
        assert shifts[i, 0] == quotient
    assert native[0, 0] != native[1, 0]
    name = {'compute': 'compute', 'locate': 'locate', 'ghost_cells': 'ghost'}[operation]
    name += '_periodic_standard'
    actual = getattr(api._require_core(), name)
    calls = []

    def capture(*args):
        calls.append(args[0].copy())
        return actual(*args)

    monkeypatch.setattr(api._require_core(), name, capture)
    kwargs = dict(domain=domain, duplicate_check='off', duplicate_wrap=False)
    exception = DuplicateError if scale == 1 else ValueError
    with np.errstate(all='raise'), pytest.raises(exception, match='backend-unsafe'):
        if operation == 'compute':
            pv.compute(points, **kwargs)
        else:
            getattr(pv, operation)(points, [[.25, .75, .75]], **kwargs)
    if scale == 1:
        np.testing.assert_array_equal(cart[0], cart[1])
        assert calls == []
    else:
        assert cart[1, 0] - cart[0, 0] == 4.
        assert F(native[1, 0]) - F(native[0, 0]) == F(1, 2**21)
        assert (F(1, 2**21))**2 < F(1e-10)
        # The actual compiled backstop refuses before insertion.
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], native)


@pytest.mark.parametrize('positive', [False, True])
@pytest.mark.parametrize('positions', [False, True])
def test_uninserted_seam_queries_keep_direct_projection(
    monkeypatch, positive, positions,
):
    points = np.array([[.25, .5, .5], [.75, .5, .5]])
    xs = [-2.**-55]
    if positive:
        points[0, 0] += 2.**-41
        xs = [0., 2.**-41]
    queries = np.array([[x, .5, .5] for x in xs])
    # Compare periodic squared distances in rationals, including the unique gap.
    owners = []
    for x in xs:
        distances = [min((F(x)-F(p)-image)**2 for image in (-1, 0, 1))
                     for p in points[:, 0]]
        assert distances[0] != distances[1]
        owners.append([11, 17][distances.index(min(distances))])
    assert owners == ([17, 11] if positive else [17])
    actual = api._require_core().locate_periodic_standard
    calls = []

    def capture(*args):
        calls.append(args[-1].copy())
        return actual(*args)

    monkeypatch.setattr(api._require_core(), 'locate_periodic_standard', capture)
    result = pv.locate(points, queries, domain=pv.PeriodicCell(np.eye(3)),
                       ids=[11, 17], return_owner_position=positions)
    np.testing.assert_array_equal(calls[0], queries)
    assert result['found'].tolist() == [True] * len(xs)
    assert result['owner_id'].tolist() == owners
    assert ('owner_pos' in result) == positions


def test_nonperiodic_outside_query_and_missing_owner_reporting():
    points = [[.25, .5, .5], [.75, .5, .5]]
    for positions in (False, True):
        result = pv.locate(points, [[2., .5, .5]], domain=pv.Box(((0., 1.),) * 3),
                           ids=[11, 17], return_owner_position=positions)
        assert result['found'].tolist() == [False]
        assert result['owner_id'].tolist() == [-1]
        if positions:
            assert np.isnan(result['owner_pos']).all()
