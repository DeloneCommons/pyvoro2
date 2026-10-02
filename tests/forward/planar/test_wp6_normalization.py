"""ADR 0025: raw-preserving, compute-owned planar identity consumption.

Inputs are small dyadic constructions, rather than archived golden packets.
The four-owner square has four quotient corners, eight positive boundary
classes and four retained point artifacts. Higher-way circle centers exercise
the same identity graph without requiring all other slots to project to S.
"""
from __future__ import annotations

import copy
import itertools
import pickle
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest

import pyvoro2.planar as p
from pyvoro2.planar.api import _compute_with_certificate
from pyvoro2._internal.planar.normalization_context import compile_context
from pyvoro2._internal.normalization_proof import ProofFailure


SQUARE = np.array([[.25, .25], [.25, .75], [.75, .25], [.75, .75]])
IDS = [7919, 104729, 17, 999983]
TRANSLATIONS = np.array([[2, -3], [-3, 2], [1, 4], [-1, -2]])
STALE = 'NORMALIZATION_PROOF_CONTEXT_STALE'


def _domain(periodic=(True, True), lengths=(1., 1.)):
    return p.RectangularCell(tuple((0., x) for x in lengths), periodic=periodic)


def _controls():
    """The 18 named controls and 18 one-site perturbations from issue #94."""
    controls = [
        ('square', SQUARE, {}),
        ('equal-radii', SQUARE, dict(mode='power', radii=[.125] * 4)),
        ('equal-weights', SQUARE, dict(mode='power', weights=[.125] * 4)),
        ('sparse-ids', SQUARE, dict(ids=IDS)),
        ('common-translation', SQUARE + [2, -3], {}),
        ('independent-translations', SQUARE + TRANSLATIONS, dict(ids=IDS)),
        ('translated-power', SQUARE + TRANSLATIONS,
         dict(ids=IDS, mode='power', radii=[.125] * 4)),
        ('rigid-offset', SQUARE + [1 / 16, 1 / 8], {}),
        ('reversed', SQUARE[::-1], {}),
        ('product', [[.25, .25], [.25, .625], [.75, .25], [.75, .625]], {}),
        ('staggered', [[.25, .25], [.25, .75], [.75, .25], [.6875, .6875]], {}),
        ('rectangular', [[.25, .25], [.375, .75], [1.25, .3125], [1.6875, .6875]],
         dict(domain=_domain(lengths=(2., 1.)))),
        ('one', [[.5, .5]], {}),
        ('one-power', [[.5, .5]], dict(mode='power', radii=[.125])),
        ('slab', [[.25, .5], [.75, .5]], {}),
        ('slab-power', [[.25, .5], [.75, .5]],
         dict(mode='power', radii=[.125, .125])),
        ('partial-x', SQUARE, dict(domain=_domain((True, False)))),
        ('partial-x-power', SQUARE,
         dict(domain=_domain((True, False)), mode='power', radii=[.125] * 4)),
    ]
    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, 2)):
        for denominator in (16, 256, 4096):
            points = SQUARE.copy()
            points[0] += np.array(direction) / denominator
            controls.append((f'perturb-{direction}-{denominator}', points, {}))
    assert len(controls) == 36
    return [pytest.param(points, options, id=name)
            for name, points, options in controls]


@pytest.mark.parametrize('points,options', _controls())
def test_characterized_controls_strict_compute_owned(points, options):
    options = dict(options)
    domain = options.pop('domain', _domain())
    result = p.compute(points, domain=domain, normalize='topology',
                       return_edge_shifts=True, return_diagnostics=True,
                       tessellation_check='raise', **options)
    assert result.require_tessellation_diagnostics().ok
    topology = result.require_normalized_topology()
    assert p.validate_normalized_topology(topology, domain, level='strict').ok
    for raw, normalized in zip(result.cells, topology.cells):
        assert raw['edges'] == normalized['edges']
        assert raw['vertices'] == normalized['vertices']
        assert len(normalized['edge_global_id']) == len(raw['edges'])


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('mode', ['standard', 'power'])
def test_square_preserves_all_native_occurrences(diagnostics, mode):
    domain = _domain()
    options = dict(mode=mode)
    if mode == 'power':
        options['radii'] = [.125] * 4
    result = p.compute(SQUARE, domain=domain, normalize='topology',
                       return_edge_shifts=True, return_diagnostics=diagnostics,
                       **options)
    topology = result.require_normalized_topology()
    assert topology.global_vertices.shape == (4, 2)
    assert len(topology.global_edges) == 12
    assert sum(len(c['edges']) for c in topology.cells) == 20
    assert p.validate_normalized_topology(topology, domain, level='strict').ok
    assert result.has_tessellation_diagnostics is diagnostics
    context = topology._normalization_context
    positive = {topology.cells[pos]['edge_global_id'][slot]
                for pos, slot in context.positive}
    artifacts = {topology.cells[pos]['edge_global_id'][slot]
                 for pos, slot in context.artifacts}
    assert len(context.positive) == 16 and len(context.artifacts) == 4
    assert len(positive) == 8 and len(artifacts) == 4 and not positive & artifacts
    for o in context.certificate.occurrences:
        edge = topology.cells[dict(context.source_positions)[o.source]]['edges'][o.slot]
        assert edge['vertices'] == [o.slot, o.next]
        assert edge['adjacent_cell'] == o.owner
        assert edge.get('adjacent_shift') == o.shift


@pytest.mark.parametrize('n,nslots,npairs', [(5, 10, 45), (6, 15, 105)])
def test_higher_way_center_identity_closure(n, nslots, npairs):
    vectors = [(5, 0), (3, 4), (-3, 4)]
    if n == 6:
        vectors.append((-5, 0))
    vectors.extend([(-3, -4), (3, -4)])
    points = .5 + np.array(vectors) / 32
    result = p.compute(points, domain=_domain(), normalize='vertices')
    normalized = result.require_normalized_vertices()
    central = [(c['vertex_global_id'][k], tuple(c['vertex_shift'][k]))
               for c in normalized.cells for k, vertex in enumerate(c['vertices'])
               if vertex == [.5, .5]]
    assert len(central) == nslots
    assert sum(a == b for a, b in itertools.combinations(central, 2)) == npairs
    closure = dict(normalized._normalization_context.closure)
    proved = [closure[position, slot] for position, cell in enumerate(normalized.cells)
              for slot, vertex in enumerate(cell['vertices']) if vertex == [.5, .5]]
    assert sum(a == b for a, b in itertools.combinations(proved, 2)) == npairs


def test_stripped_square_keeps_standalone_conservatism():
    domain = _domain()
    result = p.compute(SQUARE, domain=domain, normalize='topology',
                       return_edge_shifts=True)
    topology = result.require_normalized_topology()
    assert p.validate_normalized_topology(topology, domain, level='strict').ok
    standalone = p.normalize_topology(copy.deepcopy(result.cells), domain=domain)
    assert len(standalone.global_vertices) == 11
    assert len(standalone.global_edges) == 17
    with pytest.raises(p.NormalizationError):
        p.validate_normalized_topology(standalone, domain, level='strict')


def _square_result():
    return p.compute(SQUARE, domain=_domain(), normalize='topology',
                     return_edge_shifts=True)


@pytest.mark.parametrize('mutation', [
    lambda t: t.global_vertices.__setitem__((0, 0), .125),
    lambda t: t.cells[0]['vertices'][0].__setitem__(0, .125),
    lambda t: t.cells[0]['vertex_global_id'].__setitem__(0, 999),
    lambda t: t.cells[0]['vertex_shift'].__setitem__(0, (1, 0)),
    lambda t: t.cells[0].__setitem__('id', 999),
    lambda t: t.cells[0]['edges'][0].__setitem__('adjacent_cell', 999),
    lambda t: t.cells[0]['edges'][0].__setitem__('vertices', [0, 0]),
    lambda t: t.cells[0]['edges'][0].__setitem__('adjacent_shift', (1, 1)),
    lambda t: t.cells[0]['edge_global_id'].__setitem__(0, 999),
    lambda t: t.global_edges[0].__setitem__('vertices', (0, 0)),
])
def test_normalized_mutation_explicitly_invalidates_proof(mutation):
    topology = _square_result().require_normalized_topology()
    mutation(topology)
    with pytest.raises(p.NormalizationError) as caught:
        p.validate_normalized_topology(topology, _domain(), level='strict')
    assert caught.value.diagnostics.issues[0].code == STALE
    assert not p.validate_normalized_topology(topology, _domain()).ok


@pytest.mark.parametrize('domain', [_domain((True, False)), _domain(lengths=(2., 1.))])
def test_mismatching_domain_is_stale(domain):
    with pytest.raises(p.NormalizationError) as caught:
        p.validate_normalized_topology(_square_result().normalized_topology,
                                       domain, level='strict')
    assert caught.value.diagnostics.issues[0].code == STALE


def test_separate_raw_parent_and_normalized_vertices_remain_independent():
    result = _square_result()
    topology = result.require_normalized_topology()
    vertices = result.require_normalized_vertices()
    result.cells[0]['vertices'][0][0] = .123
    result.cells[0]['edges'][0]['vertices'][0] = 123
    result.cells[0]['edges'][0]['adjacent_shift'] = (999, 999)
    result.cells[0]['id'] = 999
    assert p.validate_normalized_topology(topology, _domain(), level='strict').ok
    assert p.validate_normalized_topology(vertices, _domain(), level='strict').ok
    assert topology.cells[0] is not vertices.cells[0]


@pytest.mark.parametrize('copy_view', [
    copy.deepcopy,
    lambda t: pickle.loads(pickle.dumps(t)),
    lambda t: p.NormalizedTopology(**asdict(t)),
    lambda t: p.NormalizedTopology(
        t.global_vertices.copy(), copy.deepcopy(t.global_edges),
        copy.deepcopy(t.cells)),
])
def test_copy_or_public_reconstruction_preserves_data_without_authority(copy_view):
    original = _square_result().normalized_topology
    copied = copy_view(original)
    assert copied.cells == original.cells
    np.testing.assert_array_equal(copied.global_vertices, original.global_vertices)
    assert not hasattr(copied, '_normalization_context')
    with pytest.raises(p.NormalizationError) as caught:
        p.validate_normalized_topology(copied, _domain(), level='strict')
    assert any(i.code == 'EDGE_VERTEX_SET_MISMATCH'
               for i in caught.value.diagnostics.issues)
    assert p.validate_normalized_topology(original, _domain(), level='strict').ok


@pytest.mark.parametrize('copy_result', [
    copy.deepcopy, lambda r: pickle.loads(pickle.dumps(r)),
])
def test_result_copy_retains_capabilities_without_proof(copy_result):
    copied = copy_result(_square_result())
    assert copied.has_normalized_topology
    assert not hasattr(copied.normalized_topology, '_normalization_context')
    with pytest.raises(p.NormalizationError):
        p.validate_normalized_topology(copied.normalized_topology,
                                       _domain(), level='strict')


def test_context_transplant_does_not_renew_authority():
    original, other = _square_result(), _square_result()
    topology = original.normalized_topology
    object.__setattr__(topology, '_normalization_context',
                       other.normalized_topology._normalization_context)
    with pytest.raises(p.NormalizationError) as caught:
        p.validate_normalized_topology(topology, _domain(), level='strict')
    assert caught.value.diagnostics.issues[0].code == STALE


def test_edge_normalization_never_silently_downgrades_a_stale_vertex_view():
    vertices = _square_result().normalized_vertices
    vertices.cells[0]['vertex_shift'][0] = (999, 0)
    with pytest.raises(p.NormalizationError) as caught:
        p.normalize_edges(vertices, domain=_domain())
    assert caught.value.diagnostics.issues[0].code == STALE


@pytest.mark.parametrize('mutation', [
    lambda c: setattr(c, 'audit_complete', False),
    lambda c: c.rows[0]['next'].__setitem__(0, 0),
    lambda c: c.packet.__setitem__('periodic', (False, False)),
    lambda c: setattr(c, 'transport', ((999, 0),) + c.transport[1:]),
    lambda c: setattr(c.semantic, 'weights', (F(1),) * 4),
])
def test_stale_audit_or_provenance_is_detected(mutation):
    topology = _square_result().normalized_topology
    mutation(topology._normalization_context.certificate)
    with pytest.raises(p.NormalizationError) as caught:
        p.validate_normalized_topology(topology, _domain(), level='strict')
    assert caught.value.diagnostics.issues[0].code == STALE


def _circle(n=6):
    vectors = [(5, 0), (3, 4), (-3, 4), (-5, 0), (-3, -4), (3, -4)]
    return .5 + np.array(vectors[:n]) / 32


def test_equal_internal_but_displaced_collapse_is_exempt_without_alias():
    result, c = _compute_with_certificate(
        _circle()[::-1], domain=_domain(), return_edge_shifts=True)
    assert c.semantic_consistent
    context = compile_context(c, result.cells, _domain())
    o = next(o for o in c.occurrences if o.source == 4 and o.slot == 3)
    assert o.collapsed
    point = c.semantic.cell(o.source).contact(o.owner, o.shift).endpoints[0]
    local = tuple(F(v) / 2 for v in c.rows[o.source]['local2'][o.slot])
    assert point != local
    position = dict(context.source_positions)[o.source]
    assert (position, o.slot) in context.artifacts
    assert not any(e.kind == 'point-alias' and e.first == (position, o.slot)
                   for e in context.identities)


@pytest.mark.parametrize('family,expected', [
    ('unequal', 'WP6_NONPOSITIVE_NATIVE_EDGE'),
    ('weights', 'WP6_EFFECTIVE_SEMANTIC_CONFLICT'),
    ('tiny-positive', 'WP6_MISSING_POSITIVE_COVERAGE'),
])
def test_failed_semantic_family_cannot_obtain_normalization_authority(family, expected):
    if family == 'unequal':
        points = .5 + np.array([(-8, 0), (-3, 4), (3, 4), (0, -6)]) / 16
        options = dict(mode='power', radii=np.array([8, 5, 5, 6]) / 16)
    else:
        points = .5 + np.array([(5, 0), (0, 6), (-7, 0), (0, -8)]) / 32
        radii = np.array([5, 6, 7, 8]) / 32
        options = (dict(mode='power', weights=radii * radii) if family == 'weights' else
                   dict(mode='power', radii=radii + 2. ** -45
                        * np.array([0, 1, -2, 3])))
    raw = p.compute(points, domain=_domain(), return_edge_shifts=True,
                    return_diagnostics=True, **options)
    assert expected in {i.code for i in raw.tessellation_diagnostics.issues}
    try:
        normalized = p.compute(points, domain=_domain(), normalize='topology',
                               return_diagnostics=True, **options)
    except p.TessellationError as exc:
        assert expected in {i.code for i in exc.diagnostics.issues}
    else:
        assert expected in {i.code for i in normalized.tessellation_diagnostics.issues}
        assert not hasattr(normalized.normalized_topology, '_normalization_context')


def test_diagnostics_disabled_audit_resource_refusal_has_no_partial_proof(monkeypatch):
    from pyvoro2._internal.planar import wp6_certificate as certificate
    from pyvoro2._internal.planar.wp6_ideal import ExactAuditBudget
    original = certificate._audit

    def refuse(c, power_input, semantic_weights, reciprocity_required, budget):
        return original(c, power_input, semantic_weights, reciprocity_required,
                        ExactAuditBudget(max_candidates=0))

    monkeypatch.setattr(certificate, '_audit', refuse)
    numerical = p.compute(SQUARE, domain=_domain(), normalize='topology',
                          return_diagnostics=False)
    assert not hasattr(numerical.normalized_topology, '_normalization_context')
    assert not numerical.has_tessellation_diagnostics
    with pytest.raises(p.NormalizationError):
        p.validate_normalized_topology(numerical.normalized_topology,
                                       _domain(), level='strict')
    diagnosed = p.compute(SQUARE, domain=_domain(), normalize='topology',
                          return_diagnostics=True)
    assert any(i.code == 'WP6_AUDIT_RESOURCE'
               for i in diagnosed.tessellation_diagnostics.issues)


@pytest.mark.parametrize('mutation', [
    lambda cells: cells[0]['edges'][0].__setitem__('adjacent_cell', 999),
    lambda cells: cells[0]['edges'][0].__setitem__('adjacent_shift', (999, 0)),
    lambda cells: cells[0]['edges'].pop(0),
    lambda cells: cells[0]['vertices'][0].__setitem__(0, .123),
])
def test_adapter_requires_actual_raw_occurrence_alignment(mutation):
    result, c = _compute_with_certificate(
        SQUARE, domain=_domain(), return_edge_shifts=True)
    mutation(result.cells)
    with pytest.raises(ProofFailure) as caught:
        compile_context(c, result.cells, _domain())
    assert caught.value.code == STALE


@pytest.mark.parametrize('mutation', [
    lambda c: c.rows[0]['next'].__setitem__(0, 0),
    lambda c: setattr(c.semantic, 'weights', (F(1),) * 4),
])
def test_adapter_cannot_approve_stale_audit_operands_by_flags_alone(mutation):
    result, c = _compute_with_certificate(
        SQUARE, domain=_domain(), return_edge_shifts=True)
    mutation(c)
    assert c.semantic_consistent
    with pytest.raises(ProofFailure) as caught:
        compile_context(c, result.cells, _domain())
    assert caught.value.code == STALE


def test_distinct_semantic_endpoints_with_colliding_public_coordinates():
    points = .5 + np.array([(8, 0), (0, 8), (-8, 0), (0, -8)]) / 32
    points[:, 0] += 2. ** 40
    radii = .125 + 2. ** -20 * np.array([0, 1, -2, 3])
    result, c = _compute_with_certificate(
        points, domain=_domain(), mode='power', radii=radii, return_edge_shifts=True)
    collision = next(o for o in c.occurrences if not o.collapsed and
                     result.cells[o.source]['vertices'][o.slot]
                     == result.cells[o.source]['vertices'][o.next])
    s = c.semantic.cell(collision.source).contact(collision.owner, collision.shift)
    assert s.status == 'positive' and s.length_squared > 0
    assert s.endpoints[0] != s.endpoints[1]
    if not c.semantic_consistent:
        with pytest.raises(p.TessellationError):
            compile_context(c, result.cells, _domain())
    else:
        topology = p.compute(points, domain=_domain(), mode='power', radii=radii,
                             normalize='topology').normalized_topology
        anchors = dict(topology._normalization_context.anchors)
        for a, b in itertools.combinations(anchors, 2):
            if anchors[a][0] != anchors[b][0]:
                assert topology.cells[a[0]]['vertex_global_id'][a[1]] != (
                    topology.cells[b[0]]['vertex_global_id'][b[1]])


def test_proved_distinctions_partition_a_coincident_numerical_bucket():
    # Schema control: two separately proved S anchors have identical float
    # rows and incident labels. Numerical equality cannot erase their IDs.
    from types import SimpleNamespace
    from pyvoro2.planar.normalize import _proof_pool_keys
    prepared = [dict(position=0, quantized=((1, 1), (1, 1)))]
    context = SimpleNamespace(identities=(), anchors=(
        ((0, 0), ((F(1, 2), F(1, 2)), (0, 0))),
        ((0, 1), ((F(1, 2) + F(1, 2 ** 60), F(1, 2)), (0, 0))),
    ))
    pools, _anchors = _proof_pool_keys(prepared, None, context)
    assert pools[0, 0] != pools[0, 1]


@pytest.mark.parametrize('domain', [
    p.Box(((0., 1.),) * 2), _domain((True, False)),
])
def test_standalone_vertex_lifts_vanish_on_nonperiodic_axes(domain):
    raw = p.compute(SQUARE, domain=domain,
                    return_edge_shifts=isinstance(domain, p.RectangularCell))
    vertices = p.normalize_vertices(raw.cells, domain=domain)
    topology = p.normalize_edges(vertices, domain=domain)
    for view in (vertices, topology):
        view.cells[0]['vertex_shift'] = [(0, 7)] * len(view.cells[0]['vertices'])
        diagnostics = p.validate_normalized_topology(view, domain, level='basic')
        assert not diagnostics.ok
        assert diagnostics.issues[0].code == 'INVALID_NORMALIZED_MAPPING'
        with pytest.raises(p.NormalizationError):
            p.validate_normalized_topology(view, domain, level='strict')
    with pytest.raises(ValueError):
        p.normalize_edges(vertices, domain=domain)


def test_vertices_only_view_does_not_require_unused_boundaries():
    domain = p.Box(((0., 1.),) * 2)
    vertices = p.normalize_vertices([dict(id=0, vertices=[[.25, .5]])],
                                    domain=domain)
    assert p.validate_normalized_topology(
        vertices, domain, level='strict', check_polygon=False).ok
    vertices.cells[0]['vertex_global_id'][0] = len(vertices.global_vertices)
    diagnostics = p.validate_normalized_topology(
        vertices, domain, level='basic', check_polygon=False)
    assert not diagnostics.ok
    assert diagnostics.issues[0].code == 'INVALID_NORMALIZED_MAPPING'


@pytest.mark.parametrize('operation', ['strict', 'basic', 'edges'])
def test_later_admission_loss_has_public_normalization_diagnostics(
        monkeypatch, operation):
    from pyvoro2._internal import native_qualification as authority
    result = _square_result()

    def refuse(module, component):
        raise authority.NativeQualificationError(
            'source_schema_mismatch', 'installed consumer changed')

    monkeypatch.setattr(authority, 'require_native', refuse)
    if operation == 'basic':
        diagnostics = p.validate_normalized_topology(
            result.normalized_topology, _domain(), level='basic')
        assert not diagnostics.ok
    else:
        with pytest.raises(p.NormalizationError) as caught:
            if operation == 'edges':
                p.normalize_edges(result.normalized_vertices, domain=_domain())
            else:
                p.validate_normalized_topology(result.normalized_topology,
                                               _domain(), level='strict')
        diagnostics = caught.value.diagnostics
    assert diagnostics.issues[0].code == 'WP6_PROFILE_UNSUPPORTED'
