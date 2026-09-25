"""Atomic refusal tests; tampered packets below are test constructions."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pyvoro2.planar import RectangularCell, TessellationError, compute
from pyvoro2.planar import api
from pyvoro2._internal.planar import wp6_certificate as certifier
from pyvoro2._internal.planar import wp6_ideal

DOMAIN = RectangularCell(((0.0, 1.0), (0.0, 1.0)))
POINTS = [[0.25, 0.25], [0.75, 0.75]]


def test_private_preparation_shift_is_unbounded_until_public_view():
    # A common private preparation translation cancels for self images.
    result = compute(
        [[float(2**100), 0.25]],
        domain=DOMAIN,
        return_vertices=False,
        return_edge_shifts=True,
    )
    assert {tuple(e['adjacent_shift']) for e in result.cells[0]['edges']} == {
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    }
    points = [[float(2**100), 0.25], [0.0, 0.75]]
    assert len(compute(points, domain=DOMAIN, return_vertices=False).cells) == 2
    with pytest.raises(TessellationError) as caught:
        compute(points, domain=DOMAIN, return_vertices=False, return_edge_shifts=True)
    assert caught.value.diagnostics.issues[0].code == 'WP6_SHIFT_REPRESENTATION'
    # The existing public remapping representation is outside WP6's removals.
    with pytest.raises(ValueError, match='signed int64'):
        DOMAIN.remap_cart(np.array([[float(2**100), 0.25]]), return_shifts=True)


@pytest.mark.parametrize('mask', [(False, True), (True, False), (True, True)])
def test_private_preparation_preserves_existing_coordinate_arithmetic(mask):
    domain = RectangularCell(((-0.5, 0.1), (-1.0, 1.0)), periodic=mask)
    points = np.random.default_rng(74).uniform(-20.0, 20.0, size=(32, 2))
    points = np.vstack((points, [[-0.5, -0.0], [np.nextafter(0.1, 0.0), 1.0]]))
    actual, big_shifts = domain._remap_cart(
        points, return_shifts=True, integer_view=False
    )
    expected, small_shifts = domain.remap_cart(points, return_shifts=True)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(big_shifts, small_shifts)


def corrupt_native(monkeypatch, transform):
    core = api._require_core2d()
    original = core._compute_box_standard_witness

    def changed(*args):
        cells, packet = original(*args)
        cells, packet = deepcopy(cells), deepcopy(packet)
        transform(cells, packet)
        return cells, packet

    monkeypatch.setattr(core, '_compute_box_standard_witness', changed)


@pytest.mark.parametrize('normalization', ['vertices', 'topology'])
def test_required_normalization_integer_view_refuses_structurally(normalization):
    # Native self shifts fit; only the requested normalized source chart needs
    # an unavailable signed-int64 coordinate-shift representation.
    with pytest.raises(TessellationError) as caught:
        compute([[float(2**100), .25]], domain=DOMAIN,
                return_vertices=False, return_edges=False, normalize=normalization)
    assert caught.value.diagnostics.issues[0].code == 'WP6_NORMALIZATION_REPRESENTATION'


@pytest.mark.parametrize('policy', ['none', 'diagnose', 'warn', 'raise'])
@pytest.mark.parametrize(
    'fault,code',
    [
        ('source', 'WP6_PROVENANCE_INVALID'),
        ('missing', 'WP6_INSERTION_OMITTED'),
        ('side', 'WP6_PROVENANCE_INVALID'),
        ('profile', 'WP6_PROFILE_UNSUPPORTED'),
    ],
)
def test_hard_attribution_errors_even_without_public_shifts(
    monkeypatch, policy, fault, code
):
    def alter(cells, packet):
        if fault == 'source':
            packet['sources'][0]['origins'][0]['source'] = 123
        elif fault == 'missing':
            packet['inserted'].pop()
        elif fault == 'profile':
            packet['profile']['source_sha256'] = '0' * 64
        else:
            for row in packet['sources']:
                for origin in row['origins']:
                    if origin['kind'] == 'initialization':
                        origin['side'] = -99
                        return

    corrupt_native(monkeypatch, alter)
    with pytest.raises(TessellationError) as caught:
        compute(
            POINTS,
            domain=DOMAIN,
            return_vertices=False,
            return_edge_shifts=False,
            tessellation_check=policy,
        )
    assert [issue.code for issue in caught.value.diagnostics.issues] == [code]


def test_attribution_resource_is_hard_and_distinct_from_audit(monkeypatch):
    limits = certifier.AttributionLimits(max_occurrences=1)
    monkeypatch.setattr(certifier, 'AttributionLimits', lambda: limits)
    with pytest.raises(TessellationError) as caught:
        compute(POINTS, domain=DOMAIN)
    assert caught.value.diagnostics.issues[0].code == 'WP6_ATTRIBUTION_RESOURCE'


def test_numerical_refusal_is_not_a_successful_reciprocal_subcheck(monkeypatch):
    from pyvoro2._internal.planar import wp6_numerical
    monkeypatch.setattr(wp6_numerical, '_MAX_SEGMENT_COMPARISONS', 0)
    result, certificate = api._compute_with_certificate(
        POINTS, domain=DOMAIN, return_diagnostics=True, return_vertices=False)
    assert certificate.audit_complete
    assert certificate.semantic_consistent
    diag = result.tessellation_diagnostics
    assert not diag.ok and not diag.ok_reciprocity
    assert not diag.reciprocity_checked
    assert any(i.code == 'WP6_NUMERICAL_AUDIT_RESOURCE' for i in diag.issues)


@pytest.mark.parametrize('policy', ['none', 'diagnose', 'warn', 'raise'])
def test_semantic_resource_refusal_never_claims_complete_audit(monkeypatch, policy):
    original_budget = wp6_ideal.ExactAuditBudget
    monkeypatch.setattr(
        wp6_ideal, 'ExactAuditBudget', lambda: original_budget(max_candidates=1)
    )
    options = dict(
        domain=DOMAIN,
        return_vertices=False,
        return_edge_shifts=True,
        tessellation_check=policy,
        return_diagnostics=True,
    )
    if policy == 'raise':
        with pytest.raises(TessellationError) as caught:
            compute(POINTS, **options)
        diag = caught.value.diagnostics
    else:
        if policy == 'warn':
            with pytest.warns(UserWarning, match='tessellation_check failed'):
                result = compute(POINTS, **options)
        else:
            result = compute(POINTS, **options)
        diag = result.tessellation_diagnostics
        assert result.has_periodic_shifts
        assert all('adjacent_shift' in e for c in result.cells for e in c['edges'])
    assert not diag.ok
    assert not diag.reciprocity_checked
    assert any(i.code == 'WP6_AUDIT_RESOURCE' for i in diag.issues)


def test_public_shift_int64_overflow_is_only_a_required_view_failure():
    # Accepted archive transport_int64_refusal: each preparation shift fits,
    # but their difference does not. No expectation calls the certifier.
    points = [[float(2**63 - 1024), 0.25], [-float(2**63), 0.75]]
    plain = compute(points, domain=DOMAIN, return_vertices=False)
    assert {c['id'] for c in plain.cells} == {0, 1}
    with pytest.raises(TessellationError) as caught:
        compute(points, domain=DOMAIN, return_vertices=False, return_edge_shifts=True)
    assert caught.value.diagnostics.issues[0].code == 'WP6_SHIFT_REPRESENTATION'
    assert certifier.public_shift((1, 0), (2**63 - 1, 0), (-1, 0)) == (2**63 + 1, 0)


def test_public_rounding_collapse_is_not_internal_native_collapse():
    # Accepted public_rounding_4 fixture:2 internally collapsed occurrences,
    # plus2 that collapse only in the public coordinate serialization.
    origin = float(2**53)
    points = np.array([[12, 4], [14, 4], [12, 0], [12, 14], [6, 4], [10, 6]]) + origin
    domain = RectangularCell(((origin, origin + 16),) * 2)
    result, certificate = api._compute_with_certificate(points, domain=domain)
    cells = {c['id']: c for c in result.cells}
    public_only = [
        o
        for o in certificate.occurrences
        if not o.collapsed
        and cells[o.source]['vertices'][o.slot] == cells[o.source]['vertices'][o.next]
    ]
    assert sum(o.collapsed for o in certificate.occurrences) == 2
    assert len(public_only) == 2


def test_large_equal_radii_do_not_rewrite_native_source_expression_results():
    # Accepted large_radius_134217728_grid24. Native source arithmetic
    # collapses all9 occurrences although equal weights cancel in both ideals.
    result = compute(
        [[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]],
        domain=DOMAIN,
        blocks=(24, 24),
        mode='power',
        radii=[2**27] * 3,
        return_edge_shifts=True,
        return_diagnostics=True,
    )
    assert len([e for c in result.cells for e in c['edges']]) == 9
    assert not result.tessellation_diagnostics.ok
    assert any(
        i.code == 'WP6_COLLAPSED_POSITIVE_COVERAGE'
        for i in result.tessellation_diagnostics.issues
    )


def test_nonzero_insertion_transport_private_eps0_stress(monkeypatch):
    original = api.prepare_generators

    def prepare_without_seam_nudge(points, **kwargs):
        prepared = original(points, **kwargs)
        # Deliberate private preparation construction, not default public history.
        return replace(
            prepared,
            native_points=np.array(points, dtype=float),
            remap_shifts=np.zeros((2, 2), dtype=object),
        )

    monkeypatch.setattr(api, 'prepare_generators', prepare_without_seam_nudge)
    domain = RectangularCell(((0.0, 0.1), (0.0, 1.0)))
    _, certificate = api._compute_with_certificate(
        [[np.nextafter(0.1, 0.0), 0.5], [0.025, 0.5]],
        domain=domain,
        blocks=(5, 1),
        return_vertices=False,
        return_edge_shifts=True,
    )
    assert certificate.storage[0]['h'] == (1, 0)
    assert certificate.transport == ((1, 0), (0, 0))
    for occurrence in certificate.occurrences:
        if occurrence.source == 0 and occurrence.owner == 1:
            assert occurrence.shift == (occurrence.sigma[0] + 1, occurrence.sigma[1])
