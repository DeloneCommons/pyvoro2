"""WP7 exact semantic anchors and RED public ghost-certificate tests.

These expectations precede the production certifier and never call its ideal,
producer, or image helpers. Native occurrences may fragment one semantic facet;
coincident classes need only geometric support coverage, not invented labels.
"""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

import pyvoro2
import pyvoro2.planar as planar

from wp7_rational_oracle import ghost_ideal


def _key(reference, ids, dim, mask):
    assert set(reference) == {'kind', 'generator_id', 'shift', 'wall_id'}
    kind = reference['kind']
    if kind == 'generator':
        assert reference['generator_id'] in ids
        assert reference['wall_id'] is None
        shift = reference['shift']
        if any(mask):
            assert isinstance(shift, tuple) and len(shift) == dim
            assert all(isinstance(s, int) and s == 0 for s, p in zip(shift, mask)
                       if not p)
        else:
            assert shift is None
            shift = (0,) * dim
        return kind, ids.index(reference['generator_id']), tuple(shift)
    if kind == 'ghost_self':
        assert reference['generator_id'] is reference['wall_id'] is None
        shift = reference['shift']
        assert isinstance(shift, tuple) and len(shift) == dim and any(shift)
        assert all(s == 0 for s, p in zip(shift, mask) if not p)
        return kind, tuple(shift)
    assert kind == 'wall'
    assert reference['generator_id'] is reference['shift'] is None
    assert reference['wall_id'] in range(-2 * dim, 0)
    return kind, reference['wall_id']


def _assert_certified(cell, ideal, dim, mask, ids, boundary_name):
    assert bool(cell['empty']) == (ideal.dimension != dim)
    if ideal.dimension != dim:
        assert float(cell['volume' if dim == 3 else 'area']) == 0
        assert cell[boundary_name] == []
        return
    assert abs(float(cell['volume' if dim == 3 else 'area'])
               - float(ideal.measure)) < 1e-8
    refs = set()
    for boundary in cell[boundary_name]:
        assert 'boundary_reference' in boundary
        ref = boundary['boundary_reference']
        if ref is None:
            # A retained raw artifact has no positive semantic authority.
            continue
        key = _key(ref, ids, dim, mask)
        assert key in ideal.positive, (key, ideal.contacts.get(key))
        if ref['kind'] == 'ghost_self':
            assert 'adjacent_cell' not in boundary
        elif ref['kind'] == 'generator':
            assert boundary['adjacent_cell'] == ref['generator_id']
        else:
            assert boundary['adjacent_cell'] == ref['wall_id']
        refs.add(key)
    assert all(classes & refs for classes in ideal.facets.values())
    return refs


def _call(dim, points, queries, mask, *, mode='standard', power=None,
          ids=None, geometry=None, **selectors):
    bounds = ((0., 1.),) * dim
    if dim == 2:
        domain = (planar.RectangularCell(bounds=bounds, periodic=mask)
                  if any(mask) else planar.Box(bounds=bounds))
        call = planar.ghost_cells
        boundary = 'edges'
        selectors.setdefault('return_edges', True)
    else:
        domain = geometry or (pyvoro2.OrthorhombicCell(bounds=bounds,
                                                       periodic=mask)
                              if any(mask) else pyvoro2.Box(bounds=bounds))
        call = pyvoro2.ghost_cells
        boundary = 'faces'
        selectors.setdefault('return_faces', True)
    cells = call(np.asarray(points, dtype=float).reshape(-1, dim),
                 np.asarray(queries, dtype=float).reshape(-1, dim),
                 domain=domain, ids=ids, mode=mode,
                 return_vertices=False, return_adjacency=False,
                 **(power or {}), **selectors)
    return cells, boundary


@pytest.mark.parametrize('dim', (2, 3))
def test_oracle_axial_ghost_self_walls_diagonal_contacts(dim):
    g = (F(1, 2),) * dim
    for mask in product((False, True), repeat=dim):
        ideal = ghost_ideal([], g, ((0, 1),) * dim, (1,) * dim, mask)
        assert ideal.dimension == dim and ideal.measure == 1
        expected = set()
        for axis, periodic in enumerate(mask):
            for sign in (-1, 1):
                if periodic:
                    shift = tuple(sign if j == axis else 0
                                  for j in range(dim))
                    expected.add(('ghost_self', shift))
                else:
                    wall = -(2 * axis + (1 if sign < 0 else 2))
                    expected.add(('wall', wall))
        assert set(ideal.positive) == expected
        if sum(mask) >= 2:
            diagonal = tuple(1 if periodic else 0 for periodic in mask)
            status = ideal.contact('ghost_self', shift=diagonal).status
            assert status == ('point' if dim == 2 or all(mask) else 'line')


@pytest.mark.parametrize('dim', (2, 3))
def test_oracle_standard_power_multiple_and_zero_contacts(dim):
    suffix = (F(1, 2),) * (dim - 2)
    g = (F(1, 2), F(3, 4)) + suffix
    points = [(F(1, 4), F(1, 2)) + suffix,
              (F(3, 4), F(1, 2)) + suffix]
    # The unweighted triangle has width 1 and height 1/2; weighting both
    # competitors by 1/16 leaves width 3/4 and height 3/8.
    for weight, expected in ((F(0), F(1, 4)),
                             (F(1, 16), F(9, 64))):
        ideal = ghost_ideal(points, g, ((0, 1),) * dim, (1,) * dim,
                            (False,) * dim, [weight, weight], 0)
        assert ideal.dimension == dim and ideal.measure == expected
        assert ideal.contact('generator', 0, (0,) * dim).status == 'positive'
        assert ideal.contact('generator', 1, (0,) * dim).status == 'positive'
    # A weighted competitor can reduce the ghost to the high-x wall; a still
    # larger weight eliminates it. The exact cell rank distinguishes both.
    p = [(F(1, 4),) + (F(1, 2),) * (dim - 1)]
    center = (F(1, 2),) * dim
    lower = ghost_ideal(p, center, ((0, 1),) * dim, (1,) * dim,
                        (False,) * dim, [F(5, 16)], 0)
    hidden = ghost_ideal(p, center, ((0, 1),) * dim, (1,) * dim,
                         (False,) * dim, [1], 0)
    assert lower.dimension == dim - 1 and lower.measure == 0
    assert lower.contact('generator', 0, (0,) * dim).status == 'lower-dimensional'
    assert hidden.dimension == -1 and hidden.measure == 0
    assert not hidden.positive


def test_oracle_stored_chart_coincident_labels_and_radius_order():
    chart = ghost_ideal([(2.25, .5)], (.5, .5), ((0, 1),) * 2,
                        (1, 1), (True, True))
    assert chart.measure == F(1, 2)
    assert chart.contact('generator', 0, (-2, 0)).status == 'positive'
    assert chart.contact('generator', 0, (1, 0)).status == 'absent'

    for dim in (2, 3):
        g = (F(1, 4), F(1, 2)) + (F(1, 2),) * (dim - 2)
        p = (F(3, 4), F(1, 2)) + (F(1, 2),) * (dim - 2)
        ideal = ghost_ideal([p], g, ((0, 1),) * dim, (1,) * dim,
                            (True,) * dim, [-F(1, 4)], 0)
        coactive = {('ghost_self', (1,) + (0,) * (dim - 1)),
                    ('generator', 0, (0,) * dim)}
        assert coactive <= ideal.positive.keys()
        assert any(coactive <= classes for classes in ideal.facets.values())

    radius = F(.1)
    assert radius * radius != F(.1 ** 2)
    weighted = ghost_ideal([(.25, .5)], (.5, .5), ((0, 1),) * 2,
                           (1, 1), (False, False), [radius * radius], 0)
    rounded = ghost_ideal([(.25, .5)], (.5, .5), ((0, 1),) * 2,
                          (1, 1), (False, False), [F(.1 ** 2)], 0)
    assert weighted.vertices != rounded.vertices


def test_oracle_triclinic_complete_family_exceeds_unit_coefficient_cube():
    # The physical cubic y facet is image (-3, 1, 0) in this exact user basis.
    # A native producer's +/-1 coefficient stencil would be an invalid S
    # completeness argument; the independent outer polytope reduces its rows.
    basis = ((1, 0, 0), (3, 1, 0), (0, 0, 1))
    ideal = ghost_ideal([], (.5, .5, .5), None, None,
                        (True,) * 3, lattice=basis)
    assert ideal.dimension == 3 and ideal.measure == 1
    assert len(ideal.positive) == 6
    assert ideal.contact('ghost_self', shift=(-3, 1, 0)).status == 'positive'
    assert ideal.contact('ghost_self', shift=(3, -1, 0)).status == 'positive'


def test_oracle_exact_unimodular_handedness_preserves_physical_supports():
    bases = (((1, 1, 0), (0, 1, 0), (0, 0, 1)),
             ((-1, -1, 0), (0, 1, 0), (0, 0, 1)),
             ((1, 0, 0), (3, 1, 0), (0, 0, 1)))
    support_sets = []
    for basis in bases:
        ideal = ghost_ideal([], (F(1, 4), F(1, 4), F(1, 2)),
                            None, None, (True,) * 3, lattice=basis)
        assert ideal.measure == 1
        support_sets.append({tuple(sum(s[i] * basis[i][j] for i in range(3))
                                   for j in range(3))
                             for kind, s in ideal.positive if kind == 'ghost_self'})
    assert support_sets[0] == support_sets[1] == support_sets[2]


@pytest.mark.parametrize('dim, expected', ((2, 'point'), (3, 'line')))
def test_oracle_lower_rank_contact_is_not_positive(dim, expected):
    center = (F(1, 2),) * dim
    p = (F(3, 4), F(3, 4)) + center[2:]
    ideal = ghost_ideal([p], center, ((0, 1),) * dim, (1,) * dim,
                        (False,) * dim, [-F(3, 8)], 0)
    assert ideal.dimension == dim and ideal.measure == 1
    assert ideal.contact('generator', 0, (0,) * dim).status == expected
    assert ('generator', 0, (0,) * dim) not in ideal.positive


def test_oracle_zero_normal_equal_and_strict_power_constraints():
    args = ([(F(1, 2), F(1, 2))], (F(1, 2), F(1, 2)),
            ((0, 1), (0, 1)), (1, 1), (False, False))
    equal = ghost_ideal(*args, [0], 0)
    assert equal.dimension == 2 and equal.measure == 1
    assert equal.contact('generator', 0, (0, 0)).status == 'identical'
    stronger = ghost_ideal(*args, [1], 0)
    assert stronger.dimension == -1 and stronger.measure == 0


@pytest.mark.parametrize('dim', (2, 3))
def test_public_ghost_only_every_rectangular_mask(dim):
    g = (.5,) * dim
    for mask in product((False, True), repeat=dim):
        cells, boundary = _call(dim, [], [g], mask, ids=[])
        assert len(cells) == 1 and cells[0]['query_index'] == 0
        assert tuple(cells[0]['site']) == g
        ideal = ghost_ideal([], cells[0]['site'], ((0, 1),) * dim,
                            (1,) * dim, mask)
        refs = _assert_certified(cells[0], ideal, dim, mask, [], boundary)
        assert refs == set(ideal.positive)
        assert 'vertices' not in cells[0] and 'adjacency' not in cells[0]


@pytest.mark.parametrize('family', ('standard', 'weights', 'radii'))
def test_public_spatial_every_mask_with_generator_and_walls(family):
    p = [(.25, .25, .5)]
    g = (.5, .75, .5)
    power = ({'weights': [-1 / 16], 'ghost_weights': 0}
             if family == 'weights' else
             {'radii': [.25], 'ghost_radii': .5} if family == 'radii'
             else {})
    mathematical = ([F(-1, 16)] if family == 'weights' else
                    [F(1, 16)] if family == 'radii' else None)
    ghost_weight = F(1, 4) if family == 'radii' else F(0)
    for mask in product((False, True), repeat=3):
        cells, boundary = _call(3, p, [g], mask, ids=[201],
                                mode='standard' if not power else 'power',
                                power=power)
        ideal = ghost_ideal(p, cells[0]['site'], ((0, 1),) * 3,
                            (1,) * 3, mask, mathematical, ghost_weight)
        refs = _assert_certified(cells[0], ideal, 3, mask, [201], boundary)
        assert any(key[0] == 'generator' for key in refs)
        if not all(mask):
            assert any(key[0] == 'wall' for key in refs)
        if any(mask):
            assert any(key[0] == 'ghost_self' for key in refs)


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('family', ('standard', 'weights', 'radii'))
def test_public_multiple_generators_power_families(dim, family):
    suffix = (.5,) * (dim - 2)
    points = [(0.25, .5) + suffix, (.75, .5) + suffix]
    g = (.5, .75) + suffix
    power = ({'weights': [1 / 16, 1 / 16], 'ghost_weights': 0}
             if family == 'weights' else
             {'radii': [.25, .25], 'ghost_radii': 0} if family == 'radii'
             else {})
    mode = 'standard' if family == 'standard' else 'power'
    cells, boundary = _call(dim, points, [g], (False,) * dim,
                            ids=[70, 42], mode=mode, power=power)
    ideal = ghost_ideal(points, cells[0]['site'], ((0, 1),) * dim,
                        (1,) * dim, (False,) * dim,
                        [F(1, 16)] * 2 if power else None, 0)
    refs = _assert_certified(cells[0], ideal, dim, (False,) * dim,
                             [70, 42], boundary)
    assert ('generator', 0, (0,) * dim) in refs
    assert ('generator', 1, (0,) * dim) in refs


@pytest.mark.parametrize('dim', (2, 3))
def test_public_batch_global_weight_gauge_preserves_original_math(dim):
    suffix = (.5,) * (dim - 2)
    points = [(.25, .5) + suffix, (.75, .5) + suffix]
    queries = [(.5, .75) + suffix, (.5, .875) + suffix]
    mask = (False,) * dim
    weights = [-1 / 8, 0]
    ghost_weights = [0, 1 / 8]
    cells, boundary = _call(dim, points, queries, mask, ids=[15, 28],
                            mode='power', power={'weights': weights,
                                                 'ghost_weights': ghost_weights})
    assert [c['query_index'] for c in cells] == [0, 1]
    for cell in cells:
        qi = cell['query_index']
        ideal = ghost_ideal(points, cell['site'], ((0, 1),) * dim,
                            (1,) * dim, mask, weights, ghost_weights[qi])
        _assert_certified(cell, ideal, dim, mask, [15, 28], boundary)


@pytest.mark.parametrize('dim', (2, 3))
def test_public_permutation_and_external_ids_change_only_labels(dim):
    suffix = (.5,) * (dim - 2)
    original = [(.25, .5) + suffix, (.75, .5) + suffix]
    g = (.5, .75) + suffix
    mask = (False,) * dim
    canonical = []
    for order, ids in (((0, 1), (101, 707)), ((1, 0), (707, 101))):
        points = [original[i] for i in order]
        cells, boundary = _call(dim, points, [g], mask, ids=ids)
        ideal = ghost_ideal(points, cells[0]['site'], ((0, 1),) * dim,
                            (1,) * dim, mask)
        refs = _assert_certified(cells[0], ideal, dim, mask,
                                 list(ids), boundary)
        canonical.append({(kind, ids[owner], shift)
                          if kind == 'generator' else (kind, owner)
                          for kind, owner, *rest in refs
                          for shift in [rest[0] if rest else None]})
    assert canonical[0] == canonical[1]


@pytest.mark.parametrize('dim', (2, 3))
def test_public_translated_original_site_and_query_keep_stored_chart(dim):
    tail = (.5,) * (dim - 2)
    mask = (True,) * dim
    classes = []
    cases = ((2.25, 3.5, -2), (7.25, -1.5, -7))
    for persistent_x, query_x, public_shift in cases:
        p = [(persistent_x, .5) + tail]
        cells, boundary = _call(dim, p, [(query_x, .5) + tail], mask,
                                ids=[909])
        cell, = cells
        assert tuple(cell['site']) == (.5,) * dim
        if dim == 3:
            assert tuple(cell['query']) == (query_x, .5) + tail
        ideal = ghost_ideal(p, cell['site'], ((0, 1),) * dim,
                            (1,) * dim, mask)
        refs = _assert_certified(cell, ideal, dim, mask, [909], boundary)
        assert ('generator', 0, (public_shift,) + (0,) * (dim - 1)) in refs
        classes.append({(key[0], key[2][0] + persistent_x)
                        if key[0] == 'generator' else key
                        for key in refs})
    assert classes[0] == classes[1]


def test_public_stored_ghost_chart_and_shift_selector():
    points = [(2.25, .5)]
    cells, boundary = _call(2, points, [(3.5, .5)], (True, True),
                            ids=[91], return_edge_shifts=True)
    cell, = cells
    assert tuple(cell['site']) == (.5, .5)
    ideal = ghost_ideal(points, cell['site'], ((0, 1),) * 2,
                        (1, 1), (True, True))
    refs = _assert_certified(cell, ideal, 2, (True, True), [91], boundary)
    assert ('generator', 0, (-2, 0)) in refs
    assert ('generator', 0, (1, 0)) not in refs
    for edge in cell['edges']:
        ref = edge['boundary_reference']
        if ref is not None and ref['kind'] in ('generator', 'ghost_self'):
            assert tuple(edge['adjacent_shift']) == ref['shift']


@pytest.mark.parametrize('dim', (2, 3))
def test_public_coincident_sources_cover_geometry_without_invented_labels(dim):
    suffix = (.5,) * (dim - 2)
    points = [(.75, .5) + suffix]
    cells, boundary = _call(dim, points, [(.25, .5) + suffix],
                            (True,) * dim, ids=[17], mode='power',
                            power={'weights': [-.25], 'ghost_weights': 0})
    ideal = ghost_ideal(points, cells[0]['site'], ((0, 1),) * dim,
                        (1,) * dim, (True,) * dim, [-F(1, 4)], 0)
    _assert_certified(cells[0], ideal, dim, (True,) * dim, [17], boundary)


@pytest.mark.parametrize('dim', (2, 3))
def test_public_hidden_and_lower_dimensional_power_ghosts(dim):
    center = (.5,) * dim
    p = [(.25,) + (.5,) * (dim - 1)]
    # The lower-dimensional S cell may be deleted by N (legitimate empty),
    # or N may retain a nonempty record (then the required outcome is an
    # atomic semantic inconsistency). Never silently report that record as a
    # certified positive ghost cell.
    try:
        cells, boundary = _call(dim, p, [center], (False,) * dim,
                                mode='power', ids=[3],
                                power={'weights': [5 / 16],
                                       'ghost_weights': 0})
    except ValueError as exc:
        assert exc.code == 'GHOST_SEMANTIC_INCONSISTENT'
    else:
        ideal = ghost_ideal(p, cells[0]['site'], ((0, 1),) * dim,
                            (1,) * dim, (False,) * dim, [F(5, 16)], 0)
        assert ideal.dimension == dim - 1
        _assert_certified(cells[0], ideal, dim, (False,) * dim, [3], boundary)

    cells, boundary = _call(dim, p, [center], (False,) * dim,
                            mode='power', ids=[3],
                            power={'weights': [5 / 16],
                                   'ghost_weights': -1})
    ideal = ghost_ideal(p, cells[0]['site'], ((0, 1),) * dim,
                        (1,) * dim, (False,) * dim, [F(5, 16)], -1)
    assert ideal.dimension == -1
    _assert_certified(cells[0], ideal, dim, (False,) * dim, [3], boundary)
    filtered, _ = _call(dim, p, [center], (False,) * dim,
                        mode='power', ids=[3], power={'weights': [5 / 16],
                        'ghost_weights': -1}, include_empty=False)
    assert filtered == []


@pytest.mark.parametrize('dim', (2, 3))
def test_public_batch_hidden_filter_preserves_survivor_query_index(dim):
    suffix = (.5,) * (dim - 1)
    p = [(.25,) + suffix]
    queries = [(.5,) + suffix, (.75,) + suffix]
    mask = (False,) * dim
    power = {'weights': [1], 'ghost_weights': [0, 1]}
    cells, boundary = _call(dim, p, queries, mask, ids=[8],
                            mode='power', power=power, include_empty=True)
    assert [c['query_index'] for c in cells] == [0, 1]
    hidden = ghost_ideal(p, cells[0]['site'], ((0, 1),) * dim,
                         (1,) * dim, mask, [1], 0)
    visible = ghost_ideal(p, cells[1]['site'], ((0, 1),) * dim,
                          (1,) * dim, mask, [1], 1)
    assert hidden.dimension == -1 and visible.dimension == dim
    _assert_certified(cells[0], hidden, dim, mask, [8], boundary)
    _assert_certified(cells[1], visible, dim, mask, [8], boundary)
    filtered, _ = _call(dim, p, queries, mask, ids=[8], mode='power',
                        power=power, include_empty=False)
    assert len(filtered) == 1 and filtered[0]['query_index'] == 1
    assert tuple(filtered[0]['site']) == tuple(cells[1]['site'])


@pytest.mark.parametrize('basis', (((1., 1., 0.), (0., 1., 0.), (0., 0., 1.)),
                                   ((-1., -1., 0.), (0., 1., 0.), (0., 0., 1.)),
                                   ((1., 0., 0.), (3., 1., 0.), (0., 0., 1.))))
def test_public_triclinic_ghost_only_basis_orientation(basis):
    domain = pyvoro2.PeriodicCell(vectors=basis)
    query = tuple(.5 * sum(row[j] for row in basis) for j in range(3))
    cells, boundary = _call(3, [], [query], (True,) * 3, ids=[],
                            geometry=domain)
    ideal = ghost_ideal([], cells[0]['site'], None, None, (True,) * 3,
                        lattice=basis)
    assert ideal.measure == 1
    _assert_certified(cells[0], ideal, 3, (True,) * 3, [], boundary)


def test_public_zero_queries_and_boundary_without_public_vertices():
    cells, _ = _call(2, [], [], (True, True), ids=[],
                     return_edge_shifts=True)
    assert cells == []
    cells, _ = _call(3, [], [], (False,) * 3, ids=[])
    assert cells == []


def test_planar_ghost_retains_unrelated_resource_and_duplicate_controls():
    cells = planar.ghost_cells(
        np.empty((0, 2)), np.empty((0, 2)),
        domain=planar.Box(bounds=((0, 1), (0, 1))),
        blocks=(2, 3), init_mem=2, duplicate_check='off',
        duplicate_threshold=.001, duplicate_wrap=False,
        duplicate_max_pairs=1,
    )
    assert cells == []


@pytest.mark.parametrize('obsolete', ('edge_shift_search',
                                      'validate_edge_shifts',
                                      'repair_edge_shifts',
                                      'edge_shift_tol'))
def test_obsolete_planar_ghost_controls_rejected(obsolete):
    with pytest.raises(TypeError, match=obsolete):
        planar.ghost_cells(np.empty((0, 2)), np.empty((0, 2)),
                           domain=planar.Box(bounds=((0, 1), (0, 1))),
                           **{obsolete: 0})
