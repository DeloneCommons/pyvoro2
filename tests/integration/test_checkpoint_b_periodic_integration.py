"""Issue #92: public composition, with dyadic square/cube identity oracles."""
from __future__ import annotations

import copy
import itertools
from collections import deque

import numpy as np
import pytest

import pyvoro2 as spatial
import pyvoro2.planar as planar


def _case(dim, partial=False):
    api = planar if dim == 2 else spatial
    boundary = 'edges' if dim == 2 else 'faces'
    shift_flag = 'return_edge_shifts' if dim == 2 else 'return_face_shifts'
    domain_type = planar.RectangularCell if dim == 2 else spatial.OrthorhombicCell
    mask = (True,) * (dim - 1) + (not partial,)
    return api, domain_type(((0., 1.),) * dim, periodic=mask), boundary, shift_flag


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('translated', [False, True])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_periodic_self_identity_composes(dim, partial, n, translated, mode):
    api, domain, boundary, flag = _case(dim, partial)
    points = np.full((n, dim), .5)
    if n == 2:
        points[:, 0] = [.25, .75]
    translations = np.zeros((n, dim), dtype=int)
    if translated:
        translations[0] = (2, -3, 4)[:dim]
        if n == 2:
            translations[1] = (-3, 2, -1)[:dim]
        if partial:
            translations[:, -1] = 0
    ids = [7919, 104729][:n]
    kwargs = dict(mode=mode, ids=ids, **{flag: True})
    if mode == 'power':
        kwargs['weights'] = [0., 1 / 32][:n]
    result = api.compute(points + translations, domain=domain,
                         return_diagnostics=True, tessellation_check='raise', **kwargs)
    raw = copy.deepcopy(result.cells)
    topology = api.normalize_topology(result.cells, domain=domain)
    diag = api.validate_normalized_topology(topology, domain, level='strict')
    assert result.require_tessellation_diagnostics().ok
    assert diag.ok and diag.ok_incidence
    # Slabs cut the torus into n vertex classes, doubled at the two walls.
    assert topology.global_vertices.shape == (n * (2 if partial else 1), dim)
    assert result.cells == raw
    by_id = {cid: row for row, cid in enumerate(ids)}
    for cell, source in zip(topology.cells, raw):
        assert cell[boundary] == source[boundary]
        assert len(cell[boundary]) == 2 * dim
        vsh = np.asarray(cell['vertex_shift'])
        gids = cell['vertex_global_id']
        np.testing.assert_allclose(topology.global_vertices[gids] + vsh,
                                   cell['vertices'], rtol=0, atol=1e-12)
        # Each periodic axis identifies two local images of every vertex class.
        assert len(set(gids)) == (2 if partial else 1) * (1 if n == 1 else 2)
        assert len(set(zip(gids, map(tuple, vsh)))) == 2 ** dim
        for record in cell[boundary]:
            owner = record['adjacent_cell']
            if owner < 0:
                assert 'adjacent_shift' not in record
                continue
            shift = np.asarray(record['adjacent_shift'])
            base_shift = (shift - translations[by_id[cell['id']]]
                          + translations[by_id[owner]])
            if owner == cell['id']:
                assert sum(abs(base_shift)) == 1
            else:
                assert not np.any(base_shift[1:])
                assert base_shift[0] in ((-1, 0) if cell['id'] == ids[0] else (0, 1))
    if n == 1 and not partial:
        assert len(topology.global_edges) == dim
        if dim == 3:
            assert len(topology.global_faces) == 3


@pytest.mark.parametrize('dim', [2, 3])
def test_validator_compares_self_boundary_image_occurrences(dim):
    api, domain, boundary, flag = _case(dim)
    cells = api.compute([[.5] * dim], domain=domain, **{flag: True}).cells
    topology = api.normalize_topology(cells, domain=domain)
    api.validate_normalized_topology(topology, domain, level='strict')
    # All bare global IDs coincide here. Swap two image assignments, retaining
    # the same set of per-cell representatives; reciprocity must still detect it.
    shifts = topology.cells[0]['vertex_shift']
    shifts[0], shifts[1] = shifts[1], shifts[0]
    with pytest.raises(api.NormalizationError):
        api.validate_normalized_topology(topology, domain, level='strict')


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('n', [1, 2])
def test_annotation_missing_periodic_image_is_unavailable(dim, partial, n):
    api, domain, boundary, _ = _case(dim, partial)
    points = ([[.5] * dim] if n == 1 else
              [[.25] + [.5] * (dim - 1), [.75] + [.5] * (dim - 1)])
    cells = api.compute(points, domain=domain, ids=[7919, 104729][:n]).cells
    annotate = (planar.annotate_edge_properties if dim == 2
                else spatial.annotate_face_properties)
    annotate(cells, domain)
    assert all(record['other_site'] is None for record in cells[0][boundary])


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('n', [1, 2])
def test_planar_annotation_preserves_real_images_and_walls(partial, n):
    _, domain, _, flag = _case(2, partial)
    points = [[.5, .5]] if n == 1 else [[.25, .5], [.75, .5]]
    ids = [7919, 104729][:n]
    cells = planar.compute(points, ids=ids, domain=domain, **{flag: True}).cells
    planar.annotate_edge_properties(cells, domain)
    sites = dict(zip(ids, points))
    seen_zero = False
    for cell in cells:
        for edge in cell['edges']:
            if edge['adjacent_cell'] < 0:
                assert edge['other_site'] is None
            else:
                shift = edge['adjacent_shift']
                seen_zero |= not any(shift)
                expected = np.asarray(sites[edge['adjacent_cell']]) + shift
                np.testing.assert_array_equal(edge['other_site'], expected)
    assert seen_zero == (n == 2)


@pytest.mark.parametrize('bad_shift', [
    None, (), (0,), (0, 0, 0), (.5, 0), (float('nan'), 0), (True, 0),
])
def test_planar_annotation_invalid_shift_never_becomes_primary_image(bad_shift):
    _, domain, _, flag = _case(2)
    cells = planar.compute([[.5, .5]], domain=domain, **{flag: True}).cells
    cells[0]['edges'][0]['adjacent_shift'] = bad_shift
    # The established unavailable-metadata value remains inspectable.
    planar.annotate_edge_properties(cells, domain)
    assert cells[0]['edges'][0]['other_site'] is None


def test_planar_annotation_nonperiodic_neighbors_need_no_shift():
    cells = planar.compute([[.25, .5], [.75, .5]], ids=[7919, 104729],
                           domain=planar.Box(((0., 1.),) * 2)).cells
    planar.annotate_edge_properties(cells, planar.Box(((0., 1.),) * 2))
    sites = {cell['id']: cell['site'] for cell in cells}
    for cell in cells:
        for edge in cell['edges']:
            assert edge['other_site'] == sites.get(edge['adjacent_cell'])


_SELECTORS = [flags for flags in itertools.product((False, True), repeat=4)
              if flags[0] or not flags[1]]


@pytest.mark.parametrize('normalize', ['vertices', 'topology'])
@pytest.mark.parametrize('edges,shifts,vertices,adjacency', _SELECTORS)
def test_planar_normalization_owns_metadata_across_raw_selectors(
        normalize, edges, shifts, vertices, adjacency):
    domain = planar.RectangularCell(((0., 1.),) * 2)
    result = planar.compute([[.125, .125], [.625, .25], [.375, .75]],
                            domain=domain, normalize=normalize,
                            return_edges=edges, return_edge_shifts=shifts,
                            return_vertices=vertices, return_adjacency=adjacency)
    norm = (result.require_normalized_topology() if normalize == 'topology'
            else result.require_normalized_vertices())
    assert planar.validate_normalized_topology(norm, domain, level='strict').ok
    assert result.has_periodic_shifts == shifts
    assert result.has_normalized_vertices
    assert result.has_normalized_topology == (normalize == 'topology')
    for raw, normalized in zip(result.cells, norm.cells):
        assert ('edges' in raw) == edges
        assert ('vertices' in raw) == vertices
        assert ('adjacency' in raw) == adjacency
        if edges:
            assert all(('adjacent_shift' in e) == shifts for e in raw['edges'])
        assert all('adjacent_shift' in e for e in normalized['edges'])
        assert 'vertices' in normalized
        assert len(normalized['vertex_global_id']) == len(normalized['vertices'])
        if edges:
            assert raw['edges'][0] is not normalized['edges'][0]


@pytest.mark.parametrize('dim', [2, 3])
def test_standalone_exact_unit_cell_incidence(dim):
    # Independent cube/square oracle: binary corner tuples and axial sides.
    api, domain, boundary, _ = _case(dim)
    vertices = list(itertools.product((0., 1.), repeat=dim))
    records = []
    for axis in range(dim):
        for side in (0, 1):
            indices = [i for i, v in enumerate(vertices) if v[axis] == side]
            if dim == 3:
                indices = [indices[i] for i in (0, 1, 3, 2)]
            shift = tuple((2 * side - 1) if j == axis else 0 for j in range(dim))
            records.append(dict(vertices=indices, adjacent_cell=7919,
                                adjacent_shift=shift))
    cells = [dict(id=7919, vertices=vertices, **{boundary: records})]
    snapshot = copy.deepcopy(cells)
    topology = api.normalize_topology(cells, domain=domain)
    diag = api.validate_normalized_topology(topology, domain, level='strict')
    assert diag.ok and diag.ok_incidence
    assert len(topology.global_vertices) == 1
    assert len(topology.global_edges) == dim
    assert cells == snapshot
    # A permutation keeps all bare gids and the per-cell shift set, but moves
    # one boundary endpoint to the wrong image. Reciprocal checks must fail.
    shifts = topology.cells[0]['vertex_shift']
    shifts[0], shifts[1] = shifts[1], shifts[0]
    with pytest.raises(api.NormalizationError):
        api.validate_normalized_topology(topology, domain, level='strict')


def test_standalone_planar_boundary_metadata_ownership():
    cells = [dict(id=7919, vertices=[[.25, .25], [.25, .75]], edges=[
        dict(vertices=[0, 1], adjacent_cell=104729, adjacent_shift=(0, 0))])]
    _, domain, _, _ = _case(2)
    vertices = planar.normalize_vertices(cells, domain=domain)
    topology = planar.normalize_edges(vertices, domain=domain)
    cells[0]['edges'][0].pop('adjacent_shift')
    assert vertices.cells[0]['edges'][0]['adjacent_shift'] == (0, 0)
    vertices.cells[0]['edges'][0].pop('adjacent_shift')
    assert topology.cells[0]['edges'][0]['adjacent_shift'] == (0, 0)


@pytest.mark.parametrize('shift', [None, (0,), (.5, 0), (0, 1)])
def test_standalone_planar_unusable_image(shift):
    domain = planar.RectangularCell(((0., 1.),) * 2, periodic=(True, False))
    cells = [dict(id=7919, site=[.5, .5], vertices=[[0., 0.], [0., 1.]],
                  edges=[dict(vertices=[0, 1], adjacent_cell=7919,
                              adjacent_shift=shift)])]
    planar.annotate_edge_properties(cells, domain)
    assert cells[0]['edges'][0]['other_site'] is None


@pytest.mark.parametrize('left_handed', [False, True])
@pytest.mark.parametrize('translated', [False, True])
def test_spatial_self_images_in_user_basis(left_handed, translated):
    vectors = np.array([[(-1. if left_handed else 1.), 0., 0.],
                        [.25, 1., 0.], [.125, .25, 1.]])
    domain = spatial.PeriodicCell(vectors)
    points = np.array([[.5, .5, .5]])
    if translated:
        points += np.array([2, -3, 1]) @ vectors
    result = spatial.compute(points, domain=domain, ids=[7919],
                             return_face_shifts=True,
                             return_diagnostics=True, tessellation_check='raise')
    original = copy.deepcopy(result.cells)
    topology = spatial.normalize_topology(result.cells, domain=domain)
    diag = spatial.validate_normalized_topology(topology, domain, level='strict')
    assert diag.ok and diag.ok_incidence
    cell = topology.cells[0]
    np.testing.assert_allclose(
        topology.global_vertices[cell['vertex_global_id']]
        + np.asarray(cell['vertex_shift']) @ vectors, cell['vertices'],
        rtol=0, atol=1e-12)
    assert result.cells == original
    assert cell['faces'] == original[0]['faces']
    assert len(topology.global_vertices) - len(topology.global_edges) + len(
        topology.global_faces) - 1 == 0


def test_standalone_spatial_reciprocity_keeps_all_fragments():
    # A malformed first occurrence must not disappear behind a later record
    # carrying the same directed owner/image label.
    cells = [dict(id=0, vertices=[[0., 0., 0.], [0., 1., 0.]],
                  vertex_global_id=[0, 0], vertex_shift=[(0, 0, 0), (0, 1, 0)],
                  faces=[dict(adjacent_cell=1, adjacent_shift=(0, 0, 0), vertices=[1]),
                         dict(adjacent_cell=1, adjacent_shift=(0, 0, 0),
                              vertices=[0])]),
             dict(id=1, vertices=[[0., 0., 0.]], vertex_global_id=[0],
                  vertex_shift=[(0, 0, 0)],
                  faces=[dict(adjacent_cell=0, adjacent_shift=(0, 0, 0),
                              vertices=[0])])]
    normalized = spatial.NormalizedVertices(np.array([[0., 0., 0.]]), cells)
    diag = spatial.validate_normalized_topology(
        normalized, spatial.OrthorhombicCell(((0., 1.),) * 3),
        check_vertex_face_shift=False, check_incidence=False, check_euler=False)
    assert not diag.ok
    assert diag.n_face_vertex_set_mismatch == 1


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('topology', [False, True])
@pytest.mark.parametrize('transitive', [False, True])
def test_standalone_completion_preserves_coincident_classes(dim, topology, transitive):
    # Independent review oracle: coincident local cycles have distinct exact
    # incident owner sets. Neither direct nor transitive completion may erase
    # that distinction. These numerical records make no N/E/S claim.
    api, domain, boundary, _flag = _case(dim)
    cycle = ([[.25, .25], [.25, .5]] if dim == 2 else
             [[.25, .25, .25], [.25, .5, .25], [.25, .25, .5]])
    count = len(cycle)

    def record(adjacent, offset=0):
        return dict(adjacent_cell=adjacent, adjacent_shift=(0,) * dim,
                    vertices=list(range(offset, offset + count)))

    if transitive:
        # All individual reciprocal buckets are unique along this path:
        # 10:first -> 20 -> 40 -> 10:second.
        cells = [dict(id=10, vertices=cycle + copy.deepcopy(cycle),
                      **{boundary: [record(20), record(30), record(40, count),
                                    record(50, count)]}),
                 dict(id=20, vertices=copy.deepcopy(cycle),
                      **{boundary: [record(10), record(40)]}),
                 dict(id=40, vertices=copy.deepcopy(cycle),
                      **{boundary: [record(20), record(10)]})]
    else:
        cells = [dict(id=owner, vertices=cycle + copy.deepcopy(cycle),
                      **{boundary: [record(peer), record(30), record(peer, count),
                                    record(40, count)]})
                 for owner, peer in ((10, 20), (20, 10))]
    original = copy.deepcopy(cells)
    normalize = api.normalize_topology if topology else api.normalize_vertices
    result = normalize(cells, domain=domain)
    assert cells == original
    gids = result.cells[0]['vertex_global_id']
    assert set(gids[:count]).isdisjoint(gids[count:])
    if not transitive:
        assert len(result.global_vertices) == 2 * count
        assert gids == result.cells[1]['vertex_global_id']


@pytest.mark.parametrize('edges', [None, 0, ['unconsumed'], [None], [(0, 1)], (None,)])
def test_standalone_nonperiodic_vertices_preserve_unused_edge_metadata(edges):
    cell = dict(id=7, vertices=[[.25, .5]], edges=edges)
    normalized = planar.normalize_vertices(
        [cell], domain=planar.Box(((0., 1.),) * 2))
    assert normalized.global_vertices.tolist() == [[.25, .5]]
    assert normalized.cells[0]['edges'] == edges


@pytest.mark.parametrize('kind', ['deque', 'ndarray'])
@pytest.mark.parametrize('stage', ['vertices', 'edges'])
def test_standalone_consumed_edge_containers_own_metadata(kind, stage):
    edge = dict(vertices=[0, 1], adjacent_cell=104729, adjacent_shift=(0, 0))
    edges = deque([edge]) if kind == 'deque' else np.asarray([edge], dtype=object)
    cells = [dict(id=7919, vertices=[[.25, .25], [.25, .75]], edges=edges)]
    domain = planar.RectangularCell(((0., 1.),) * 2)
    if stage == 'vertices':
        normalized = planar.normalize_vertices(cells, domain=domain)
    else:
        cells[0]['vertex_global_id'] = [0, 1]
        cells[0]['vertex_shift'] = [(0, 0), (0, 0)]
        vertices = planar.NormalizedVertices(np.asarray(cells[0]['vertices']), cells)
        normalized = planar.normalize_edges(vertices, domain=domain)
    edge.pop('adjacent_shift')
    assert normalized.cells[0]['edges'][0].get('adjacent_shift') == (0, 0)
    assert normalized.cells[0]['edges'][0] is not edge
