"""Ghost query metadata with certified boundaries requires the owning WP7 route."""
import itertools

import pytest

from test_wp8_metadata import rectangular


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('vertices,adjacency',
                         tuple(itertools.product((False, True), repeat=2)))
def test_ghost_query_views_do_not_change_stored_boundary_chart(
        dim, vertices, adjacency):
    api, domain = rectangular(dim)
    opts = {'return_edges' if dim == 2 else 'return_faces': True}
    queries = [[3.5, *([.5] * (dim - 1))], [.5] * dim]
    cells = api.ghost_cells([[2.25, *([.5] * (dim - 1))]], queries,
                            domain=domain, ids=[71], return_vertices=vertices,
                            return_adjacency=adjacency, **opts)
    for i, cell in enumerate(cells):
        assert cell['query'] == queries[i] and cell['query_index'] == i
        assert cell['query_shift'] == (3 if i == 0 else 0,) + (0,) * (dim - 1)
        assert cell['site'] == [.5] * dim
        assert 'site_shift' not in cell
        key = 'edges' if dim == 2 else 'faces'
        refs = [r['boundary_reference'] for r in cell[key]]
        assert {r['shift'] for r in refs if r['kind'] == 'generator'} == {
            (-2,) + (0,) * (dim - 1), (-1,) + (0,) * (dim - 1)}


@pytest.mark.parametrize('dim', [2, 3])
def test_certified_ghost_large_query_shift_materializes_only_retained_records(dim):
    api, domain = rectangular(dim)
    query = [[float(2**70), *([.5] * (dim - 1))]]
    opts = dict(domain=domain, mode='power', weights=[4.], ghost_weights=[0.])
    assert api.ghost_cells([[.5] * dim], query, include_empty=False, **opts) == []
    with pytest.raises(ValueError) as caught:
        api.ghost_cells([[.5] * dim], query, include_empty=True, **opts)
    assert caught.value.code == 'GHOST_SHIFT_UNREPRESENTABLE'
    assert caught.value.stage == 'materialization'
    assert caught.value.query_index == 0
    assert caught.value.details['field'] == 'query_shift'
