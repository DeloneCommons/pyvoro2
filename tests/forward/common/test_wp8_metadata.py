"""Independent exact-coordinate oracle and public WP8 contract regressions."""

from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2 as spatial
from pyvoro2 import planar


def exact_wrap(query, rows, origin, periodic):
    """Test-only Gauss-Jordan affine solve; no production geometry helpers."""
    dim = len(origin)
    matrix = [[F(float(rows[j][i])) for j in range(dim)]
              + [F(float(query[i])) - F(float(origin[i]))]
              for i in range(dim)]
    for col in range(dim):
        pivot = next(i for i in range(col, dim) if matrix[i][col])
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [v / divisor for v in matrix[col]]
        for i in range(dim):
            if i != col:
                factor = matrix[i][col]
                matrix[i] = [a - factor * b
                             for a, b in zip(matrix[i], matrix[col])]
    shifts = tuple(int(matrix[i][-1] // 1) if periodic[i] else 0
                   for i in range(dim))
    wrapped = [float(F(float(query[j])) - sum(
        shifts[i] * F(float(rows[i][j])) for i in range(dim)))
        for j in range(dim)]
    return wrapped, shifts


def rectangular(dim, *, periodic=True, bounds=None):
    api = planar if dim == 2 else spatial
    bounds = bounds or ((0., 1.),) * dim
    if not periodic:
        return api, api.Box(bounds)
    cls = planar.RectangularCell if dim == 2 else spatial.OrthorhombicCell
    return api, cls(bounds, periodic=(True,) + (False,) * (dim - 1))


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_native_owner_float_is_preserved_while_exact_image_is_exposed(dim, family):
    api, domain = rectangular(dim)
    points = np.array([[np.nextafter(1., 0.), *([.5] * (dim - 1))]])
    queries = np.array([[0., *([.5] * (dim - 1))],
                        [2., *([.5] * (dim - 1))]])
    opts = {} if family == 'standard' else {'mode': 'power', family: [0.]}
    out = api.locate(points, queries, domain=domain, ids=[713],
                     return_owner_position=True, **opts)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [713, 713]
    assert out['owner_pos'][:, 0].tolist() == [0., 2.]
    assert out['owner_shift'][:, 0].tolist() == [-1, 1]
    assert F(float(points[0, 0])) - 1 == -F(1, 2**53)
    np.testing.assert_array_equal(out['owner_site'], np.repeat(points, 2, axis=0))
    np.testing.assert_array_equal(out['query'], queries)
    assert out['query_shift'][:, 0].tolist() == [0, 2]
    for name in ('owner_shift', 'query_shift'):
        assert out[name].dtype == np.int64
    assert not np.shares_memory(out['query'], queries)
    assert not np.shares_memory(out['owner_site'], points)


@pytest.mark.parametrize('dim', [2, 3])
def test_query_wrap_uses_exact_source_and_not_backend_snap(dim):
    api, domain = rectangular(dim)
    q = np.array([[np.nextafter(1., 0.), *([.25] * (dim - 1))]])
    out = api.locate([[.5] * dim], q, domain=domain)
    assert out['query_shift'].tolist() == [[0] * dim]
    np.testing.assert_array_equal(out['query_wrapped'], q)
    assert domain.remap_cart(q)[0, 0] == 0.
    assert not {'owner_site', 'owner_pos', 'owner_shift'} & out.keys()


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_nonperiodic_native_omission_is_integrity_failure(dim, family):
    api, domain = rectangular(dim, periodic=False, bounds=((-1., 1.),) * dim)
    opts = {} if family == 'standard' else {'mode': 'power', family: [0.]}
    with pytest.raises(ValueError) as caught:
        api.locate([[np.nextafter(1., 0.), *([0.] * (dim - 1))]],
                   [[0.] * dim], domain=domain, blocks=(1,) * dim, **opts)
    assert caught.value.code == 'LOCATE_BACKEND_INSERTION'
    assert caught.value.query_index is None


@pytest.mark.parametrize('dim', [2, 3])
def test_ghost_large_query_shift_materializes_only_retained_records(dim):
    api, domain = rectangular(dim)
    query = [[float(2**70), *([.5] * (dim - 1))]]
    opts = dict(domain=domain, mode='power', weights=[4.], ghost_weights=[0.])
    opts['return_edges' if dim == 2 else 'return_faces'] = False
    assert api.ghost_cells([[.5] * dim], query, include_empty=False, **opts) == []
    with pytest.raises(ValueError) as caught:
        api.ghost_cells([[.5] * dim], query, include_empty=True, **opts)
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'
    assert caught.value.stage == 'materialization'
    assert caught.value.query_index == 0
    assert caught.value.details['field'] == 'query_shift'


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('periodic', [False, True])
def test_ghost_original_query_and_stored_chart_are_distinct(dim, periodic):
    api, domain = rectangular(dim, periodic=periodic)
    query = [[2.25 if periodic else .25, *([.5] * (dim - 1))]]
    opts = {'return_edges': False} if dim == 2 else {'return_faces': False}
    out = api.ghost_cells(np.empty((0, dim)), query, domain=domain, **opts)[0]
    assert out['query'] == query[0]
    assert out['query_index'] == 0
    if periodic:
        assert out['query_shift'] == (2,) + (0,) * (dim - 1)
        assert out['query_wrapped'] == [.25, *([.5] * (dim - 1))]
        assert out['site'] == out['query_wrapped']
    else:
        assert 'query_shift' not in out and 'query_wrapped' not in out
