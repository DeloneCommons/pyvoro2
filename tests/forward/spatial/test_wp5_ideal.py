"""Independent exact fixtures for the WP5 full-cell semantic engine."""

from fractions import Fraction as F
from itertools import combinations, permutations, product
import random

import numpy as np
import pytest

from pyvoro2._internal.spatial.wp5_common import WP5Budget, WP5Failure, WP5Limits


IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
ALL_PERIODIC = (True, True, True)


def _ideal(*args, **kwargs):
    # Keep missing implementation a test failure, rather than collection failure.
    from pyvoro2._internal.spatial.wp5_ideal import ExactIdeal

    return ExactIdeal(*args, **kwargs)


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F())


def _det(rows):
    total = F()
    for perm in permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3))
        term = rows[0][perm[0]] * rows[1][perm[1]] * rows[2][perm[2]]
        total += (-1) ** inversions * term
    return total


def _oracle_vertices(cuts):
    """Test-only Cramer intersection oracle, independent of clipping/reduction."""

    result = set()
    for triple in combinations(cuts, 3):
        normals, offsets = zip(*triple)
        det = _det(normals)
        if not det:
            continue
        point = tuple(
            _det(tuple(tuple(offsets[r] if c == axis else normals[r][c]
                             for c in range(3)) for r in range(3))) / det
            for axis in range(3)
        )
        if all(_dot(n, point) <= q for n, q in cuts):
            result.add(point)
    return result


def test_one_cubic_site_classifies_facets_edges_vertices_and_absence():
    cell = _ideal([(0, 0, 0)], IDENTITY, [0], ALL_PERIODIC).cell(0)

    assert cell.dimension == 3
    assert set(cell.vertices) == set(product((-1, 1), repeat=3))
    assert len(cell.facets) == 6
    for axis in range(3):
        for sign in (-1, 1):
            shift = tuple(sign if k == axis else 0 for k in range(3))
            contact = cell.contact(0, shift)
            assert contact.status == 'positive'
            assert contact.dimension == 2
            assert contact.area_squared == 1
            assert cell.facets[(0, shift)] == contact

    assert cell.contact(0, (1, 1, 0)).status == 'zero'
    assert cell.contact(0, (1, 1, 0)).dimension == 1
    assert cell.contact(0, (1, 1, 1)).dimension == 0
    assert cell.contact(0, (10**40, 0, 0)).status == 'absent'
    assert cell.contact(0, (10**40, 0, 0)).area_squared == 0


def test_numpy_integer_operands_keep_python_exact_arithmetic():
    cell = _ideal(np.zeros((1, 3), dtype=np.int64),
                  np.eye(3, dtype=np.int64), np.zeros(1, dtype=np.int64),
                  ALL_PERIODIC).cell(0)

    assert set(cell.vertices) == set(product((-1, 1), repeat=3))
    assert all(contact.area_squared == 1 for contact in cell.facets.values())


def test_rigorous_outer_norm_bound_keeps_cubic_complete_region_small():
    budget = WP5Budget(WP5Limits(candidate_limit=27))
    cell = _ideal([(0, 0, 0)], IDENTITY, [0], ALL_PERIODIC,
                  budget=budget).cell(0)

    assert len(cell.facets) == 6
    # The farthest retained image has norm sqrt(3), equal to the true outer
    # radius. Outward rounding must keep this exact corner contact.
    corner = cell.contact(0, (1, 1, 1))
    assert corner.dimension == 0
    assert corner.vertices == ((1, 1, 1),)


@pytest.mark.parametrize('periodic', list(product((False, True), repeat=3)))
def test_partial_periodicity_uses_real_walls_and_source_local_slabs(periodic):
    lengths = (2, 3, 4)
    lattice = ((2, 0, 0), (0, 3, 0), (0, 0, 4))
    site = (F(1, 4), F(1, 3), F(2, 3))
    bounds = ((0, 2), (0, 3), (0, 4))
    cell = _ideal([site], lattice, [0], periodic, bounds=bounds).cell(0)
    intervals = [(-lengths[k], lengths[k]) if periodic[k]
                 else (-2 * site[k], 2 * (lengths[k] - site[k])) for k in range(3)]

    assert set(cell.vertices) == set(product(*intervals))
    assert cell.dimension == 3
    assert len(cell.facets) == 6
    for axis in range(3):
        expected_area_squared = (24 // lengths[axis]) ** 2
        for side, sign in enumerate((-1, 1)):
            if periodic[axis]:
                shift = tuple(sign if k == axis else 0 for k in range(3))
                contact = cell.contact(0, shift)
            else:
                wall_id = -(2 * axis + side + 1)
                contact = cell.wall(wall_id)
                assert ('wall', wall_id) in cell.facets
            assert contact.status == 'positive'
            assert contact.area_squared == expected_area_squared


def test_contacts_are_classified_against_full_cell_not_outer_bound():
    cell = _ideal([(0, 0, 0), (F(1, 2), 0, 0)], IDENTITY,
                  [0, 0], ALL_PERIODIC).cell(0)

    assert set(cell.vertices) == set(product((F(-1, 2), F(1, 2)), (-1, 1), (-1, 1)))
    assert cell.contact(0, (1, 0, 0)).status == 'absent'
    assert cell.contact(1, (0, 0, 0)).status == 'positive'
    assert cell.contact(1, (-1, 0, 0)).status == 'positive'
    assert cell.contact(0, (0, 1, 0)).area_squared == F(1, 4)


@pytest.mark.parametrize('constrained_axes', [1, 2, 3])
def test_lower_dimensional_cells_remain_distinct_from_empty(constrained_axes):
    sites = [(0, 0, 0)]
    for axis in range(constrained_axes):
        sites.extend(tuple(sign if k == axis else 0 for k in range(3))
                     for sign in (-1, 1))
    weights = [0] + [1] * (len(sites) - 1)
    cell = _ideal(sites, IDENTITY, weights, (False, False, False),
                  bounds=((-2, 2),) * 3).cell(0)

    assert cell.dimension == 3 - constrained_axes
    assert set(cell.vertices) == set(product(*[(0,) if k < constrained_axes
                                               else (-4, 4) for k in range(3)]))
    contact = cell.contact(1, (0, 0, 0))
    assert contact.dimension == cell.dimension
    assert contact.status == ('positive' if constrained_axes == 1 else 'zero')
    if constrained_axes == 1:
        assert contact.area_squared == 256


def test_hidden_weighted_cell_is_empty():
    cell = _ideal([(0, 0, 0), (-1, 0, 0), (1, 0, 0)], IDENTITY,
                  [0, 2, 2], (False, False, False),
                  bounds=((-2, 2),) * 3).cell(0)

    assert cell.dimension == -1
    assert cell.vertices == ()
    assert not cell.facets
    assert cell.contact(1, (0, 0, 0)).status == 'absent'


def test_effective_and_semantic_inputs_have_independent_contact_statuses():
    sites = [(0, 0, 0), (-1, 0, 0), (1, 0, 0)]
    kwargs = dict(periodic=(False, True, True),
                  bounds=((-2, 2), (0, 1), (0, 1)))
    effective = _ideal(sites, IDENTITY, [0, 0, 0], **kwargs).cell(0)
    semantic = _ideal(sites, IDENTITY, [0, 1, 1], **kwargs).cell(0)

    assert effective.dimension == 3
    assert semantic.dimension == 2
    assert effective.contact(0, (0, 1, 0)).status == 'positive'
    assert semantic.contact(0, (0, 1, 0)).status == 'zero'


def test_coincident_support_preserves_each_owner_provenance():
    cell = _ideal([(0, 0, 0), (1, 0, 0), (2, 0, 0)], IDENTITY,
                  [0, 0, 2], (False, True, True),
                  bounds=((-3, 3), (0, 1), (0, 1))).cell(0)

    left = cell.contact(1, (0, 0, 0))
    right = cell.contact(2, (0, 0, 0))
    assert left.status == right.status == 'positive'
    assert left.vertices == right.vertices
    assert (1, (0, 0, 0)) in cell.facets
    assert (2, (0, 0, 0)) in cell.facets


def test_signed_partial_lattice_retains_allowed_images_and_real_walls():
    lattice = ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
    sites = ((F(1, 8), F(1, 8), F(1, 8)),
             (F(7, 8), F(1, 8), F(1, 8)))
    cell = _ideal(sites, lattice, [0, 0], (True, False, False),
                  bounds=((0, 1),) * 3).cell(0)

    expected = product((F(-1, 4), F(3, 4)),
                       (F(-1, 4), F(7, 4)), (F(-1, 4), F(7, 4)))
    assert set(cell.vertices) == set(expected)
    assert cell.contact(1, (1, 0, 0)).status == 'positive'
    assert cell.contact(1, (-1, 0, 0)).status == 'absent'
    assert cell.contact(0, (0, 1, 0)).status == 'absent'
    assert cell.wall(-1).status == 'absent'
    assert cell.wall(-3).area_squared == F(1, 4)


def test_exact_weight_gauge_and_binary64_radius_square_remain_exact():
    radius = F.from_float(float(2**27 + 1))
    weights = (F(2**54), radius * radius)
    distance = 2**14 + 1
    length = 2 * distance
    lattice = ((length, 0, 0), (0, length, 0), (0, 0, length))
    geometry = ([(0, 0, 0), (distance, 0, 0)], lattice)
    cell = _ideal(*geometry, weights, ALL_PERIODIC).cell(0)
    shifted = _ideal(*geometry, [w - 2**54 for w in weights], ALL_PERIODIC).cell(0)

    assert cell.vertices == shifted.vertices
    assert {v[0] for v in cell.contact(1, (0, 0, 0)).vertices} == {F(2**15, distance)}
    assert cell.contact(1, (0, 0, 0)).area_squared == length**4


def test_tiny_positive_semantic_faces_do_not_require_a_float_measure():
    width = F(1, 2**1100)
    cell = _ideal([(0, 0, 0), (-1, 0, 0), (1, 0, 0)], IDENTITY,
                  [0, 1 - width, 1 - width], (False, True, True),
                  bounds=((-2, 2), (0, 1), (0, 1))).cell(0)

    contact = cell.contact(0, (0, 1, 0))
    assert cell.dimension == 3
    assert contact.status == 'positive'
    assert contact.area_squared == width * width


def test_large_semantic_faces_do_not_materialize_a_float_measure():
    length = 2**600
    lattice = ((length, 0, 0), (0, length, 0), (0, 0, length))
    cell = _ideal([(0, 0, 0)], lattice, [0], ALL_PERIODIC).cell(0)

    assert cell.contact(0, (1, 0, 0)).area_squared == F(2**2400)


def test_reduced_triclinic_coefficients_map_back_without_fixed_width():
    shear = 2**80
    lattice = ((1, 0, 0), (shear, 1, 0), (0, 0, -1))
    cell = _ideal([(0, 0, 0)], lattice, [0], ALL_PERIODIC).cell(0)
    positive = ((1, 0, 0), (-shear, 1, 0), (0, 0, -1))

    assert set(cell.vertices) == set(product((-1, 1), repeat=3))
    assert set(cell.facets) == {
        (0, tuple(sign * k for k in row))
        for row in positive for sign in (-1, 1)
    }
    assert all(f.area_squared == 1 for f in cell.facets.values())


def test_original_source_translation_moves_labels_without_rounding_geometry():
    translation = 2**80
    cell = _ideal([(translation, 0, 0), (F(1, 4), 0, 0)], IDENTITY,
                  [0, 0], ALL_PERIODIC).cell(0)

    assert set(cell.vertices) == set(product((F(-3, 4), F(1, 4)), (-1, 1), (-1, 1)))
    assert cell.contact(1, (translation, 0, 0)).status == 'positive'
    assert cell.contact(1, (translation - 1, 0, 0)).status == 'positive'
    assert cell.contact(1, (0, 0, 0)).status == 'absent'


def test_triclinic_prism_preserves_exact_irrational_facet_measure():
    lattice = ((1, 0, 0), (F(1, 2), 1, 0), (0, 0, 1))
    cell = _ideal([(0, 0, 0)], lattice, [0], ALL_PERIODIC).cell(0)
    hexagon = ((1, F(3, 4)), (0, F(5, 4)), (-1, F(3, 4)),
               (-1, F(-3, 4)), (0, F(-5, 4)), (1, F(-3, 4)))

    assert set(cell.vertices) == {(x, y, z) for (x, y), z in product(hexagon, (-1, 1))}
    assert len(cell.facets) == 8
    assert cell.contact(0, (1, 0, 0)).area_squared == F(9, 16)
    assert cell.contact(0, (0, 1, 0)).area_squared == F(5, 16)
    assert cell.contact(0, (0, 0, 1)).area_squared == 1


def test_weighted_cell_matches_independent_exact_supercell_oracle():
    sites = ((F(), F(), F()), (F(1, 4), F(1, 8), F()))
    weights = (F(), F(1, 32))
    cuts = []
    # Beyond this 3x3x3 supercell, some displacement component is at least
    # 7/4. Its t*t-|t| contribution exceeds 21/16; the other two contribute
    # at least -1/2 in total, and |Delta| <= 1/32. Such a cut cannot touch
    # the self cube, proving this test's finite oracle complete independently.
    for owner, site in enumerate(sites):
        for shift in product((-1, 0, 1), repeat=3):
            if owner == 0 and shift == (0, 0, 0):
                continue
            d = tuple(site[k] + shift[k] for k in range(3))
            q = _dot(d, d) + weights[0] - weights[owner]
            if q <= sum(map(abs, d)):
                cuts.append((d, q))
    expected = _oracle_vertices(cuts)
    cell = _ideal(sites, IDENTITY, weights, ALL_PERIODIC).cell(0)

    assert expected
    assert set(cell.vertices) == expected
    for (owner, shift), contact in cell.facets.items():
        d = tuple(sites[owner][k] + shift[k] for k in range(3))
        q = _dot(d, d) + weights[0] - weights[owner]
        assert set(contact.vertices) == {v for v in expected if _dot(d, v) == q}


def test_weighted_triclinic_cells_match_independent_exact_supercell_oracle():
    lattice = ((F(1), F(), F()), (F(1, 4), F(1), F()),
               (F(), F(1, 4), F(1)))
    sites = ((F(), F(), F()), (F(1, 4), F(1, 8), F()))
    weights = (F(), F(1, 32))
    outer = []
    for x, t, u in product((-1, 1), repeat=3):
        y = F(17, 16) * t - F(1, 4) * x
        z = F(17, 16) * u - F(1, 4) * y
        outer.append((F(x), y, z))
    # These are literal self-slab vertices, independent of production reduction.
    # M < 9/4 and |Delta| <= 1/32 imply R < 23/10. The exact inverse columns
    # have 1-norm at most 21/16, and source displacement inverse coordinates
    # have absolute value at most 7/32. Thus every touching coefficient obeys
    # |s_k| < (23/10)*(21/16)+7/32 < 4: [-3,3]^3 is independently complete.
    assert max(_dot(v, v) for v in outer) == F(19073, 4096) < F(9, 4)**2
    assert F(23, 10)**2 - F(9, 4) * F(23, 10) - F(1, 32) > 0
    assert F(23, 10) * F(21, 16) + F(7, 32) < 4
    ideal = _ideal(sites, lattice, weights, ALL_PERIODIC)
    for source, expected_count in enumerate((20, 32)):
        cuts = {}
        for owner, site in enumerate(sites):
            for shift in product(range(-3, 4), repeat=3):
                if owner == source and shift == (0, 0, 0):
                    continue
                normal = tuple(site[k] - sites[source][k]
                               + sum(shift[j] * lattice[j][k] for j in range(3))
                               for k in range(3))
                offset = _dot(normal, normal) + weights[source] - weights[owner]
                if offset <= max(_dot(normal, v) for v in outer):
                    cuts[(owner, shift)] = normal, offset
        expected = _oracle_vertices(tuple(cuts.values()))
        assert len(expected) == expected_count
        cell = ideal.cell(source)
        assert set(cell.vertices) == expected
        expected_facets = {}
        for label, (normal, offset) in cuts.items():
            points = sorted(v for v in expected if _dot(normal, v) == offset)
            if not points:
                continue
            differences = [tuple(a - b for a, b in zip(v, points[0]))
                           for v in points[1:]]
            if any(_det((a, b, normal)) for a, b in combinations(differences, 2)):
                expected_facets[label] = set(points)
        assert set(cell.facets) == set(expected_facets)
        for label, points in expected_facets.items():
            assert set(cell.facets[label].vertices) == points


def test_weighted_bounded_cells_match_independent_intersection_oracle():
    generator = random.Random(20260923)
    grid = list(product((F(-3, 4), F(-1, 4), F(1, 4), F(3, 4)), repeat=3))
    for _ in range(4):
        sites = generator.sample(grid, 5)
        weights = [F(generator.randrange(-2, 5), 4) for _ in sites]
        ideal = _ideal(sites, IDENTITY, weights, (False, False, False),
                       bounds=((-1, 1),) * 3)
        for source, point in enumerate(sites):
            cuts = []
            for axis in range(3):
                for sign in (-1, 1):
                    normal = tuple(F(sign if k == axis else 0) for k in range(3))
                    cuts.append((normal, 2 * (1 - sign * point[axis])))
            for owner, other in enumerate(sites):
                if owner != source:
                    d = tuple(x - y for x, y in zip(other, point))
                    cuts.append((d, _dot(d, d) + weights[source] - weights[owner]))
            assert set(ideal.cell(source).vertices) == _oracle_vertices(cuts)


@pytest.mark.parametrize('limits', [
    WP5Limits(candidate_limit=1), WP5Limits(work_limit=1), WP5Limits(bit_limit=1),
])
def test_resource_guards_refuse_instead_of_publishing_a_prefix(limits):
    with pytest.raises(WP5Failure) as caught:
        _ideal([(0, 0, 0)], IDENTITY, [0], ALL_PERIODIC,
               budget=WP5Budget(limits)).cell(0)
    assert caught.value.code == 'WP5_RESOURCE_LIMIT'


@pytest.mark.parametrize('resource', [False, True])
def test_reduction_failure_keeps_resource_and_invariant_reasons_distinct(
    monkeypatch, resource,
):
    from pyvoro2._internal.exact_lattice import (
        ExactLatticeReductionInvariantError, ExactLatticeReductionResourceError,
    )
    from pyvoro2._internal.spatial import wp5_ideal

    if resource:
        failure = ExactLatticeReductionResourceError(
            'bounded reduction refused', stage='reduction', resource='steps',
            observed=5, configured_limit=4, source_summary={},
        )
        expected = 'WP5_RESOURCE_LIMIT'
    else:
        failure = ExactLatticeReductionInvariantError('invalid reduction certificate')
        expected = 'WP5_SOURCE_PROFILE_MISMATCH'

    def refuse(*args, **kwargs):
        raise failure

    monkeypatch.setattr(wp5_ideal, '_prepare_basis', refuse)
    with pytest.raises(WP5Failure) as caught:
        _ideal([(0, 0, 0)], IDENTITY, [0], ALL_PERIODIC)
    assert caught.value.code == expected
    assert caught.value.__cause__ is failure
