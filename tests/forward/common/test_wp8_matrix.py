"""WP8 matrix with independently fixed exact images, not output goldens."""
from fractions import Fraction as F
import itertools

import numpy as np
import pytest

import pyvoro2
from pyvoro2 import planar
from test_wp8_metadata import exact_wrap, rectangular


@pytest.mark.parametrize('dim', [2, 3])
def test_binary64_rectangular_span_is_the_affine_operand(dim):
    api, domain = rectangular(dim, bounds=((.1, 1.),) * dim)
    query = np.array([[1., *([.5] * (dim - 1))]])
    span = float(1. - .1)
    # The separately supplied upper bound lies strictly below the exact seam.
    assert F(.1) + F(span) > F(1.)
    rows = np.eye(dim) * span
    wrapped, shift = exact_wrap(query[0], rows, [.1] * dim,
                                (True,) + (False,) * (dim - 1))
    out = api.locate([[.5] * dim], query, domain=domain)
    np.testing.assert_array_equal(out['query_wrapped'], [wrapped])
    assert out['query_shift'].tolist() == [list(shift)] == [[0] * dim]
    assert domain.remap_cart(query)[0, 0] == .1


MASKS = [(d, mask) for d in (2, 3)
         for mask in itertools.product((False, True), repeat=d)]


@pytest.mark.parametrize('dim,mask', MASKS)
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_all_rectangular_masks_families_and_external_id_permutations(dim, mask, family):
    api = planar if dim == 2 else pyvoro2
    cls = planar.RectangularCell if dim == 2 else pyvoro2.OrthorhombicCell
    domain = cls(((0., 1.),) * dim, periodic=mask)
    removal = np.array([2 if p else 0 for p in mask])
    points = np.array([[.25] * dim, [.75] * dim]) + removal
    queries = np.array([[.125] * dim, [.875] * dim]) - removal
    opts = {} if family == 'standard' else {'mode': 'power', family: [0., 0.]}
    out = api.locate(points[::-1], queries, ids=[19, 31], domain=domain,
                     return_owner_position=True, **opts)
    assert out['found'].tolist() == [True, True]
    assert out['owner_id'].tolist() == [31, 19]
    expected_image = np.array([[.25] * dim, [.75] * dim]) - removal
    np.testing.assert_array_equal(out['owner_pos'], expected_image)
    if any(mask):
        np.testing.assert_array_equal(out['owner_site'], points)
        np.testing.assert_array_equal(out['owner_shift'], [-2 * removal] * 2)
        for i, query in enumerate(queries):
            wrapped, shift = exact_wrap(query, np.eye(dim), [0.] * dim, mask)
            assert out['query_wrapped'][i].tolist() == wrapped
            assert out['query_shift'][i].tolist() == list(shift)
    else:
        assert set(out) == {'found', 'owner_id', 'owner_pos'}


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('empty_population', [False, True])
def test_not_found_sentinels_and_owned_arrays(dim, empty_population):
    api, domain = rectangular(dim)
    points = np.empty((0, dim)) if empty_population else [[.5] * dim]
    query = np.array([[.25, 2., *([.5] * (dim - 2))]])
    out = api.locate(points, query, domain=domain, return_owner_position=True)
    assert out['found'].tolist() == [False]
    assert out['owner_id'].tolist() == [-1]
    assert np.isnan(out['owner_site']).all() and np.isnan(out['owner_pos']).all()
    assert out['owner_shift'].tolist() == [[0] * dim]
    for name in ('query', 'query_wrapped'):
        assert not np.shares_memory(out[name], query)
    out['query'][0, 0] = 12.
    assert query[0, 0] == .25 and out['query_wrapped'][0, 0] == .25


@pytest.mark.parametrize('shear', [0, 1, 16])
@pytest.mark.parametrize('hand', [1, -1])
@pytest.mark.parametrize('family', ['standard', 'weights', 'radii'])
def test_exact_equivalent_bases_transport_images(shear, hand, family):
    basis = np.array([[hand, hand * shear, 0], [0, 1, 0], [0, 0, 1.]])
    domain = pyvoro2.PeriodicCell(basis, origin=[.125, -.25, .5])
    original = np.array([[2.25, .375, .625]])
    query = np.array([[-1.625, 2.25, -.25]])
    # This is the cubic lattice. In the original query chart the unique closest
    # image is (-1.75, 2.375, -.375), translation (-4, 2, -1).
    expected_shift = [-4 * hand, 2 + 4 * shear, -1]
    opts = {} if family == 'standard' else {'mode': 'power', family: [0.]}
    out = pyvoro2.locate(original, query, domain=domain,
                         return_owner_position=True, **opts)
    assert out['owner_shift'].tolist() == [expected_shift]
    np.testing.assert_allclose(out['owner_pos'], [[-1.75, 2.375, -.375]], atol=1e-12)
    wrapped, shift = exact_wrap(query[0], basis, domain.origin, (True,) * 3)
    assert out['query_shift'].tolist() == [list(shift)]
    assert out['query_wrapped'].tolist() == [wrapped]


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('original,query,shift', [
    (float(2**63), .125, -(2**63)),
    (-float(2**63), -.75, 2**63 - 1),
])
def test_owner_int64_endpoints_and_cancellation_before_materialization(
        dim, original, query, shift):
    api, domain = rectangular(dim)
    out = api.locate([[original, *([.5] * (dim - 1))]],
                     [[query, *([.5] * (dim - 1))]], domain=domain,
                     return_owner_position=True)
    assert out['owner_shift'][0, 0] == shift


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('vertices,adjacency',
                         tuple(itertools.product((False, True), repeat=2)))
def test_ghost_query_views_without_boundaries_preserve_stored_chart(
        dim, vertices, adjacency):
    api, domain = rectangular(dim)
    opts = {'return_edges' if dim == 2 else 'return_faces': False}
    queries = [[3.5, *([.5] * (dim - 1))], [.5] * dim]
    cells = api.ghost_cells([[2.25, *([.5] * (dim - 1))]], queries,
                            domain=domain, ids=[71], return_vertices=vertices,
                            return_adjacency=adjacency, **opts)
    for i, cell in enumerate(cells):
        assert cell['query'] == queries[i] and cell['query_index'] == i
        assert cell['query_shift'] == (3 if i == 0 else 0,) + (0,) * (dim - 1)
        assert cell['site'] == [.5] * dim
        assert 'site_shift' not in cell
