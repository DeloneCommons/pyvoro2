"""#99: numerical spatial normalization is not an exact-S reconstruction.

Inputs are selected unchanged from the hash-verified #96 atlas. The independent
Fraction polygon oracle recomputes ideal facts; no archived native topology is
used as a golden answer. See docs/development/spatial-degeneracy-regressions.md.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pytest

import pyvoro2 as spatial
from pyvoro2 import api

from _degeneracy_oracle import (
    add, cross, diagram, dimension, inverse, mul, quotient_key, rowmat, sub,
)


INPUTS = json.loads((Path(__file__).parent / 'data/degeneracy_scope.json').read_text())
CARTESIAN = [
    ('spatial_self_cube', (1, 3, 3, 1)),
    ('spatial_mixed_slab8', (2, 6, 6, 2)),
    ('spatial_mixed_square_extruded8', (4, 12, 12, 4)),
    ('spatial_distinct_cube8', (8, 24, 24, 8)),
]
# Literal #96 facts, independently reconstructed in each test. Native counts
# are deliberately not golden: conforming producers may refine N differently.
OTHER_BASES = [
    ('spatial_generic_tetra4', (12, 32, 24, 4)),
    ('spatial_bipyramid5', (11, 30, 24, 5)),
    ('spatial_octa6', (2, 11, 15, 6)),
    ('spatial_asymmetric_power6', (17, 40, 29, 6)),
    ('spatial_self_triclinic4', (6, 12, 7, 1)),
    ('spatial_partial_cube8', (12, 32, 28, 8)),
    ('spatial_rotated_self_cube8', (1, 3, 3, 1)),
    ('spatial_asymmetric_power8', (44, 92, 56, 8)),
    ('spatial_asymmetric_extruded_power4', (6, 16, 14, 4)),
]
STRICT_DIFFERENCES = [
    ('spatial_bipyramid5__radial_out_2m24', (15, 36, 26, 5), False),
    ('spatial_bipyramid5__radial_in_2m24', (16, 37, 26, 5), True),
    ('spatial_octa6__radial_in_2m24', (9, 23, 20, 6), False),
    ('spatial_octa6__tangential_2m24', (9, 23, 20, 6), True),
    ('spatial_asymmetric_power5__radial_out_2m36', (30, 61, 36, 5), False),
    ('spatial_asymmetric_power5__tangential_2m36', (30, 61, 36, 5), False),
    ('spatial_distinct_cube8__radial_in_2m52', (26, 64, 46, 8), False),
]


def _weights(fixture):
    if 'radii' in fixture:
        return [F(r) ** 2 for r in fixture['radii']]
    return fixture.get('weights')


@lru_cache(None)
def _ideal(name):
    f = INPUTS[name]
    return diagram(f['points'], _weights(f), f['lattice'], f['periodic'], f['bounds'])


def _compute(name, *, action='diagnose'):
    f = INPUTS[name]
    if f['domain'] == 'periodic':
        domain = spatial.PeriodicCell([[float(F(x)) for x in r] for r in f['lattice']])
    else:
        domain = spatial.OrthorhombicCell(f['bounds'], periodic=f['periodic'])
    options = dict(domain=domain, mode=f['mode'], return_face_shifts=True,
                   output='cells', include_empty=True, tessellation_check=action)
    if 'ids' in f:
        options['ids'] = f['ids']
    if 'radii' in f:
        options['radii'] = [float(F(x)) for x in f['radii']]
    elif 'weights' in f:
        options['weights'] = [float(F(x)) for x in f['weights']]
    cells, cert = api._compute_with_certificate(
        [[float(F(x)) for x in p] for p in f['points']], **options)
    return domain, cells, cert


def _counts(view):
    return (len(view.global_vertices), len(view.global_edges),
            len(view.global_faces), len(view.cells))


def _raw_view(domain, cells):
    original = deepcopy(cells)
    view = spatial.normalize_topology(cells, domain=domain)
    # Ordering, multiplicity, labels and image-qualified native cycles survive.
    assert cells == original
    for raw, normalized in zip(original, view.cells):
        assert normalized['faces'] == raw['faces']
        assert normalized['vertices'] == raw['vertices']
        assert len(normalized['face_global_id']) == len(raw['faces'])
    return view


def _check_semantic_vertices(cert, ideal):
    # WP5 uses doubled source-local coordinates; the oracle uses ordinary local
    # coordinates. This is exact algebra, not tolerance-generated membership.
    for i, c in enumerate(ideal['cells']):
        assert set(cert.semantic.cell(i).vertices) == {mul(v, 2) for v in c['vertices']}


def _target_lifts(fixture, ideal):
    q = tuple(F(x) for x in fixture['target'])
    inv = inverse([tuple(F(x) for x in r) for r in fixture['lattice']])
    lifts = set()
    for i, (site, c) in enumerate(zip(fixture['points'], ideal['cells'])):
        site = tuple(F(x) for x in site)
        for vertex in c['vertices']:
            s = rowmat(sub(q, add(site, vertex)), inv)
            if all(x.denominator == 1 and (flag or x == 0)
                   for x, flag in zip(s, fixture['periodic'])):
                lifts.add((i, s))
    return lifts


@pytest.mark.parametrize('name,expected', CARTESIAN)
def test_cartesian_local_geometry_does_not_determine_owner_quotient(name, expected):
    ideal = _ideal(name)
    assert ideal['counts'] == expected
    for c in ideal['cells']:
        assert (len(c['vertices']), len(c['edges']), len(c['facets'])) == (8, 12, 6)
        assert [sum(dimension(p) == d for p in c['contacts'].values())
                for d in (0, 1, 2)] == [8, 12, 6]
    # Analytic Cartesian torus: n boxes each contribute one corner class,
    # three axis edge classes and three face classes after image identification.
    n = len(INPUTS[name]['points'])
    assert expected == (n, 3 * n, 3 * n, n)
    assert len(_target_lifts(INPUTS[name], ideal)) == 8
    domain, cells, cert = _compute(name)
    _check_semantic_vertices(cert, ideal)
    assert cert.audit_complete and cert.semantic_consistent
    assert spatial.validate_normalized_topology(
        _raw_view(domain, cells), domain, level='strict').ok


def test_hexagonal_prism_has_six_way_incidence_without_point_collapse():
    name = 'spatial_self_hexprism6'
    ideal = _ideal(name)
    # Bisectors x=+-1/2 and +-x/2+-y=5/8 bound the exact hexagon.
    xy = {(F(sign, 2), F(t, 8)) for sign in (-1, 1) for t in (-3, 3)}
    xy.update({(F(), F(-5, 8)), (F(), F(5, 8))})
    expected = {(x, y, z) for x, y in xy for z in (F(-1, 2), F(1, 2))}
    c = ideal['cells'][0]
    assert set(c['vertices']) == expected
    assert (len(c['vertices']), len(c['edges']), len(c['facets'])) == (12, 18, 8)
    assert ideal['counts'] == (2, 5, 4, 1)
    assert len(_target_lifts(INPUTS[name], ideal)) == 6
    assert {dimension(p) for p in c['contacts'].values()} == {1, 2}
    domain, cells, cert = _compute(name)
    _check_semantic_vertices(cert, ideal)
    for label, points in c['contacts'].items():
        contact = cert.semantic.cell(0).contact(*label)
        assert contact.dimension == dimension(points)
        assert set(contact.vertices) == {mul(v, 2) for v in points}
    assert spatial.validate_normalized_topology(
        _raw_view(domain, cells), domain, level='strict').ok


def test_partial_periodic_ridge_retains_two_distinct_quotient_endpoints():
    name = 'spatial_partial_square_ridge4'
    f, ideal = INPUTS[name], _ideal(name)
    local = ideal['cells'][0]['contacts'][3, (0, 0, 0)]
    assert ideal['counts'] == (8, 20, 16, 4)
    endpoints = {add(tuple(F(x) for x in f['points'][0]), p) for p in local}
    assert dimension(local) == 1
    assert endpoints == {(F(1, 2), F(1, 2), F()), (F(1, 2), F(1, 2), F(1))}
    assert len({quotient_key([p], f['lattice'], f['periodic']) for p in endpoints}) == 2
    domain, cells, cert = _compute(name)
    for owning in (cert.effective, cert.semantic):
        contact = owning.cell(0).contact(3, (0, 0, 0))
        assert contact.status == 'zero' and contact.dimension == 1
        assert contact.area_squared == 0
        assert set(contact.vertices) == {mul(v, 2) for v in local}
    view = _raw_view(domain, cells)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    # Both physical wall roles survive numerical normalization. Equal x/y and
    # zero contact area cannot authorize merging through nonperiodic z.
    ids = {0: set(), 1: set()}
    for c in view.cells:
        assert all(s[2] == 0 for s in c['vertex_shift'])
        for p, gid in zip(c['vertices'], c['vertex_global_id']):
            if p[:2] == [.5, .5]:
                ids[int(p[2])].add(gid)
    assert ids[0] and ids[1] and ids[0].isdisjoint(ids[1])


@pytest.mark.parametrize('direct_radii', [False, True])
def test_successful_power_five_audit_is_not_e_s_vertex_bijection(direct_radii):
    suffix = '__exact_dyadic_radii' if direct_radii else ''
    name = 'spatial_asymmetric_power5' + suffix
    domain, cells, cert = _compute(name, action='raise')
    s = _ideal(name)
    e = diagram(cert.effective.sites, cert.effective.weights, cert.effective.lattice)
    assert cert.audit_complete and cert.semantic_consistent and not cert.issues
    assert s['counts'] == (28, 58, 35, 5)
    assert e['counts'] == ((28, 58, 35, 5) if direct_radii else (29, 59, 35, 5))
    _check_semantic_vertices(cert, s)
    for i, c in enumerate(e['cells']):
        expected = {mul(v, 2) for v in c['vertices']}
        assert set(cert.effective.cell(i).vertices) == expected
    assert spatial.validate_normalized_topology(
        _raw_view(domain, cells), domain, level='strict').ok


def test_power_six_direct_dyadic_radius_control_has_identical_e_s_geometry():
    name = 'spatial_asymmetric_power6__exact_dyadic_radii'
    domain, cells, cert = _compute(name, action='raise')
    ideal = _ideal(name)
    e = diagram(cert.effective.sites, cert.effective.weights, cert.effective.lattice)
    assert ideal['counts'] == e['counts'] == (17, 40, 29, 6)
    assert cert.audit_complete and cert.semantic_consistent
    for left, right in zip(ideal['cells'], e['cells']):
        assert left['vertices'] == right['vertices']
        assert left['contacts'] == right['contacts']
    assert spatial.validate_normalized_topology(
        _raw_view(domain, cells), domain, level='strict').ok


@pytest.mark.parametrize('name,expected,semantic_ok', STRICT_DIFFERENCES)
def test_all_seven_saved_strict_views_may_differ_from_s(name, expected, semantic_ok):
    ideal = _ideal(name)
    assert ideal['counts'] == expected
    domain, cells, cert = _compute(name)
    _check_semantic_vertices(cert, ideal)
    view = _raw_view(domain, cells)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    assert _counts(view) != ideal['counts']
    assert cert.audit_complete and cert.semantic_consistent == semantic_ok
    if not semantic_ok:
        assert any(i.code == 'WP5_POSITIVE_FACET_MISSING' for i in cert.issues)


def test_tiny_exact_positive_facet_loss_remains_wp5_coverage_failure():
    name = 'spatial_bipyramid5__radial_out_2m36'
    ideal = _ideal(name)
    domain, cells, cert = _compute(name)
    missing = [i for i in cert.issues
               if i.code == 'WP5_POSITIVE_FACET_MISSING' and i.context['ideal'] == 'S']
    assert missing and cert.audit_complete and not cert.semantic_consistent
    for issue in missing:
        i, label = issue.context['source_id'], issue.context['label']
        points = ideal['cells'][i]['contacts'][label]
        assert dimension(points) == 2
        # Exact noncollinear vectors prove positive area independently of any
        # native floating measure or tolerance.
        assert any(any(cross(sub(a, points[0]), sub(b, points[0])))
                   for a in points[1:] for b in points[1:])
        contact = cert.semantic.cell(i).contact(*label)
        assert contact.status == 'positive' and contact.area_squared > 0
    original = deepcopy(cells)
    view = _raw_view(domain, cells)
    with pytest.raises(spatial.NormalizationError):
        spatial.validate_normalized_topology(view, domain, level='strict')
    assert cells == original
    with pytest.raises(spatial.TessellationError) as raised:
        _compute(name, action='raise')
    assert any(i.code == 'WP5_POSITIVE_FACET_MISSING'
               for i in raised.value.diagnostics.issues)


def test_equal_public_triples_do_not_identify_exact_semantic_vertices():
    name = 'spatial_distinct_cube8__radial_in_2m52'
    ideal = _ideal(name)
    buckets = defaultdict(list)
    for p in ideal['quotient_vertices']:
        buckets[tuple(float(x) for x in p)].append(p)
    collisions = {p: pairs for p, pairs in buckets.items() if len(pairs) > 1}
    assert set(collisions) == {(0., .5, .5), (.5, 0., .5), (.5, .5, 0.)}
    assert all(len(pairs) == 2 and pairs[0] != pairs[1]
               for pairs in collisions.values())
    domain, cells, cert = _compute(name)
    _check_semantic_vertices(cert, ideal)
    assert not cert.semantic_consistent
    assert any(i.code == 'WP5_POSITIVE_FACET_MISSING' for i in cert.issues)
    # The weaker raw view may pass; this does not grant an exact coordinate-to-S
    # vertex decoder or manufacture the missing exact facets.
    view = _raw_view(domain, cells)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    assert _counts(view) != ideal['counts']
    with pytest.raises(spatial.TessellationError):
        _compute(name, action='raise')


@pytest.mark.parametrize('name,expected', OTHER_BASES)
def test_other_spatial_archetypes_keep_exact_incidence_and_raw_scope(name, expected):
    ideal = _ideal(name)
    assert ideal['counts'] == expected
    domain, cells, cert = _compute(name)
    _check_semantic_vertices(cert, ideal)
    assert cert.audit_complete
    view = _raw_view(domain, cells)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok


@pytest.mark.parametrize('name', [
    'spatial_self_cube__equivalent_sheared_basis',
    'spatial_distinct_cube8__equivalent_sheared_basis',
    'spatial_self_cube__left_handed_basis',
    'spatial_distinct_cube8__left_handed_basis',
    'spatial_self_hexprism6__left_handed_equivalent',
    'spatial_distinct_cube8__external_ids',
    'spatial_self_cube__independent_lattice_translations',
])
@pytest.mark.parametrize('power', [False, True])
def test_basis_self_image_and_sparse_id_controls_preserve_public_shifts(name, power):
    f = deepcopy(INPUTS[name])
    lattice = np.asarray([[float(F(x)) for x in r] for r in f['lattice']])
    domain = (spatial.PeriodicCell(lattice) if f['domain'] == 'periodic' else
              spatial.OrthorhombicCell(f['bounds'], periodic=f['periodic']))
    points = np.asarray([[float(F(x)) for x in p] for p in f['points']])
    kwargs = dict(ids=f.get('ids'), return_face_shifts=True,
                  tessellation_check='raise', return_diagnostics=True)
    if power:
        kwargs.update(mode='power', radii=[.25] * len(points))
    result = spatial.compute(points, domain=domain, **kwargs)
    view = _raw_view(domain, result.cells)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    ids = f.get('ids', list(range(len(points))))
    assert [c['id'] for c in view.cells] == ids
    for c in view.cells:
        np.testing.assert_allclose(
            view.global_vertices[c['vertex_global_id']]
            + np.asarray(c['vertex_shift']) @ lattice,
            c['vertices'], rtol=0, atol=1e-12)
        for face in c['faces']:
            assert face['adjacent_cell'] in ids
            assert len(face['adjacent_shift']) == 3
    # Query/owner transport consumes WP8's own user-lattice authority.
    query = points[:1] + np.array([2, -3, 1]) @ lattice
    located = spatial.locate(points, query, domain=domain, return_owner_position=True,
                             **{k: v for k, v in kwargs.items()
                                if k in ('ids', 'mode', 'radii')})
    np.testing.assert_allclose(located['query_wrapped']
                               + located['query_shift'] @ lattice, query,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(located['owner_site']
                               + located['owner_shift'] @ lattice,
                               query, rtol=0, atol=1e-12)


def test_spatial_routes_do_not_consume_planar_proof_adapter(monkeypatch):
    from pyvoro2._internal.planar import normalization_context

    def forbidden(*args, **kwargs):
        raise AssertionError('Spatial route entered the planar proof adapter')

    monkeypatch.setattr(normalization_context, 'compile_context', forbidden)
    domain = spatial.OrthorhombicCell(((0., 1.),) * 3)
    points = [[float(F(x)) for x in p]
              for p in INPUTS['spatial_distinct_cube8']['points']]
    result = spatial.compute(points, domain=domain,
                             return_face_shifts=True,
                             tessellation_check='raise')
    view = spatial.normalize_topology(result.cells, domain=domain)
    assert spatial.validate_normalized_topology(view, domain, level='strict').ok
    assert not hasattr(view, '_normalization_context')
