"""Independent composition and malformed-packet regressions for WP5."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from pyvoro2 import OrthorhombicCell, PeriodicCell, api
from pyvoro2.diagnostics import TessellationError


def _domain():
    return OrthorhombicCell(((0, 4),) * 3)


def _certificate(points=None, domain=None):
    return api._compute_with_certificate(
        [[1., 1., 1.]] if points is None else points,
        domain=_domain() if domain is None else domain,
        return_face_shifts=True, return_vertices=False, return_adjacency=False,
    )[1]


@pytest.mark.parametrize('damage', ['missing_id', 'bad_id', 'missing_faces',
                                    'missing_cells', 'bad_edge'])
def test_malformed_witness_is_a_structured_hard_failure(monkeypatch, damage):
    packet = deepcopy(_certificate().packet)
    if damage == 'missing_id':
        del packet['cells'][0]['id']
    elif damage == 'bad_id':
        packet['cells'][0]['id'] = 19
    elif damage == 'missing_faces':
        del packet['cells'][0]['faces']
    elif damage == 'missing_cells':
        del packet['cells']
    else:
        packet['cells'][0]['adjacency'][0][0] = 19
    monkeypatch.setattr(api._core, '_observe_box', lambda *a, **k: packet)
    with pytest.raises(TessellationError) as raised:
        api.compute([[1., 1., 1.]], domain=_domain(), return_face_shifts=True,
                    tessellation_check='none')
    diagnostics = raised.value.diagnostics
    assert not diagnostics.face_shift_available
    assert any(issue.code == 'WP5_SOURCE_PROFILE_MISMATCH'
               for issue in diagnostics.issues)


def test_frame_bridge_retains_nonzero_binary64_frame_defects():
    # These decimal source bits make Q and its inverse only numerically
    # orthogonal. Check the exact algebra using the original inputs, not a
    # nearest-image or face residual oracle.
    vectors = np.array([[4., 0.2, 0.1], [0.3, 4., 0.4], [0.1, 0.6, 4.]])
    points = np.array([[9., 1., 1.], [2., 2., 2.]])
    cert = _certificate(points, PeriodicCell(vectors))
    A = [[Fraction(float(x)) for x in row] for row in vectors]
    p = [[Fraction(float(x)) for x in row] for row in points]
    b = [[Fraction(float(x)) for x in row['site']]
         for row in cert.packet['sites']]
    L = cert.effective.lattice
    sigma = (2, -3, 1)
    public = tuple(sigma[k] + cert.shifts[0][k] - cert.shifts[1][k]
                   for k in range(3))
    delta_s = [p[1][k] - p[0][k] + sum(public[t] * A[t][k]
                                       for t in range(3)) for k in range(3)]
    delta_e = [b[1][k] - b[0][k] + sum(sigma[t] * L[t][k]
                                       for t in range(3)) for k in range(3)]
    mapped = [sum(delta_s[t] * cert.frame[t][k] for t in range(3))
              for k in range(3)]
    assert tuple(delta_e[k] - mapped[k] for k in range(3)) == (
        cert.bridge_defect(0, 1, sigma)
    )
    assert any(x for row in cert.lattice_defect for x in row)
    for i in range(2):
        assert all(p[i][k] == cert.charts[i][k] + sum(
            cert.shifts[i][t] * A[t][k] for t in range(3)) for k in range(3))


def test_public_integer_overflow_never_selects_another_image():
    from pyvoro2._internal.spatial.wp5_certificate import _public_shift
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    with pytest.raises(WP5Failure) as raised:
        _public_shift((1, 0, 0), (2**63 - 1, 0, 0), (0, 0, 0))
    assert raised.value.code == 'WP5_SHIFT_REPRESENTATION'
    assert raised.value.context['shift'] == (2**63, 0, 0)
    assert _public_shift((0, 0, 0), (-2**63, 0, 0), (0, 0, 0)) == (
        -2**63, 0, 0,
    )


def test_completed_multiple_producer_images_are_unresolved(monkeypatch):
    from pyvoro2._internal.spatial.wp5_producer import Producer, Route
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    cert = _certificate([[1., 2., 2.], [3., 2., 2.]])
    cell = cert.packet['cells'][0]
    origin = next(row for row in cell['origins'] if row['kind'] == 'particle')
    # The deliberately injected full source-history set tests the composition
    # rule: both images survive even if only one could have a positive ideal.
    routes = [Route('independent-a', (0, 0, 0), 'standard'),
              Route('independent-b', (-1, 0, 0), 'standard')]
    monkeypatch.setattr(Producer, '_box_particle', lambda *args: routes)
    producer = Producer(cert.packet, cert.prepared.native_points,
                        cert.prepared.internal_ids)
    with pytest.raises(WP5Failure) as raised:
        producer.attribute(cell, origin)
    assert raised.value.code == 'WP5_IMAGE_UNRESOLVED'
    assert set(raised.value.context['candidates']) == {(0, 0, 0), (-1, 0, 0)}


def _audit(cert, *, omit_first=False, duplicate_first=False, packet=None):
    from pyvoro2._internal.spatial.wp5_certificate import _audit_semantics
    from pyvoro2._internal.spatial.wp5_common import WP5Budget
    from pyvoro2._internal.spatial.wp5_cycle import audit_cycle

    occurrences = []
    for cell in cert.packet['cells']:
        origins = {row['token']: row for row in cell['origins']}
        for index, face in enumerate(cell['faces']):
            owner, shift = cert.labels[cell['id'], index]
            # The independently known cube is already in its source chart.
            occurrences.append((cell, face, origins[face['token']],
                                owner, shift, shift))
    if omit_first:
        occurrences = occurrences[1:]
    if duplicate_first:
        occurrences.append(occurrences[0])
    findings = []
    _audit_semantics(cert.packet if packet is None else packet, occurrences,
                     cert.effective, cert.semantic, cert.shifts, WP5Budget(),
                     findings, audit_cycle)
    return findings


def test_missing_positive_cube_occurrence_reports_both_coverage_directions():
    cert = _certificate()
    findings = _audit(cert, omit_first=True)
    codes = {finding.code for finding in findings}
    assert codes == {'WP5_POSITIVE_FACET_MISSING', 'WP5_RECIPROCAL_MISSING'}
    missing = [f for f in findings if f.code == 'WP5_POSITIVE_FACET_MISSING']
    assert {f.context['ideal'] for f in missing} == {'E', 'S'}
    assert len(cert.labels) == 6  # No opposite shift was repaired or substituted.


def test_repeated_occurrence_is_reported_without_merging_native_faces():
    cert = _certificate()
    findings = _audit(cert, duplicate_first=True)
    assert [finding.code for finding in findings] == ['WP5_OCCURRENCE_MULTIPLICITY']


def test_missing_volumetric_owner_is_distinct_from_missing_facet_coverage():
    cert = _certificate()
    packet = deepcopy(cert.packet)
    packet['cells'][0]['computed'] = False
    findings = _audit(cert, packet=packet)
    assert [finding.code for finding in findings] == ['WP5_OWNER_COVERAGE_MISSING']


@pytest.mark.parametrize('policy', ['none', 'diagnose', 'warn', 'raise'])
@pytest.mark.parametrize('code', [
    'WP5_EXACT_ZERO', 'WP5_IDEAL_ABSENT', 'WP5_REPRESENTATION_CONFLICT',
    'WP5_POSITIVE_FACET_MISSING', 'WP5_OWNER_COVERAGE_MISSING',
    'WP5_RECIPROCAL_MISSING', 'WP5_OCCURRENCE_MULTIPLICITY',
    'WP5_PROVENANCE_COINCIDENT', 'WP5_NATIVE_CYCLE_COLLAPSED',
    'WP5_NATIVE_CYCLE_INVALID',
])
def test_semantic_findings_use_the_existing_action_policy(monkeypatch, policy, code):
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    certify = api._certify_wp5

    def with_finding(*args, **kwargs):
        return replace(certify(*args, **kwargs), issues=(
            WP5Failure(code, 'deliberate semantic audit finding'),
        ))

    monkeypatch.setattr(api, '_certify_wp5', with_finding)
    options = dict(domain=_domain(), return_face_shifts=True,
                   return_vertices=False, return_adjacency=False,
                   return_diagnostics=True,
                   tessellation_check=policy)
    if policy == 'raise':
        with pytest.raises(TessellationError) as raised:
            api.compute([[1., 1., 1.]], **options)
        assert raised.value.diagnostics.face_shift_available
        assert code in {i.code for i in raised.value.diagnostics.issues}
        return
    if policy == 'warn':
        with pytest.warns(UserWarning, match=code):
            result = api.compute([[1., 1., 1.]], **options)
    else:
        result = api.compute([[1., 1., 1.]], **options)
    assert result.has_periodic_shifts
    assert len(result.cells[0]['faces']) == 6
    assert not result.tessellation_diagnostics.ok


def test_semantic_diagnostics_use_external_owner_and_label_ids(monkeypatch):
    from pyvoro2._internal.spatial.wp5_common import WP5Failure

    certify = api._certify_wp5

    def with_finding(*args, **kwargs):
        return replace(certify(*args, **kwargs), issues=(WP5Failure(
            'WP5_PROVENANCE_COINCIDENT', 'test owner labels', source_id=0,
            owner=1, label=(1, (0, 0, 0)),
            labels=((1, (0, 0, 0)), ('wall', -1)),
        ),))

    monkeypatch.setattr(api, '_certify_wp5', with_finding)
    result = api.compute([[1., 2., 2.], [3., 2., 2.]], domain=_domain(),
                         ids=[17, 99], return_face_shifts=True,
                         return_diagnostics=True)
    issue = result.tessellation_diagnostics.issues[-1]
    assert issue.examples[0] == {
        'source_id': 17, 'owner': 99, 'label': (99, (0, 0, 0)),
        'labels': ((99, (0, 0, 0)), ('wall', -1)),
    }


def test_decimal_cross_tiny_exact_facet_is_never_erased_by_area_tolerance():
    points = [[.1, .5, .5], [.9, .5, .5], [.5, .1, .5], [.5, .9, .5]]
    domain = OrthorhombicCell(((0, 1),) * 3, periodic=(True, True, False))
    result, cert = api._compute_with_certificate(
        points, domain=domain, return_face_shifts=True,
        return_vertices=False, return_adjacency=False,
    )
    contact = cert.semantic.cell(1).contact(3, (0, -1, 0))
    assert contact.status == 'positive'
    assert contact.area_squared == Fraction(
        21093705987797736880617171147817,
        33699933333938303484777960425886112789416772978412816530493407232,
    )
    assert result.has_periodic_shifts
    # Native tolerance can omit this face; portable expectations follow the
    # observed packet instead of freezing one compiler's marginal topology.
    label = (3, (0, -1, 0))
    if label not in [value for (i, _), value in cert.labels.items() if i == 1]:
        assert any(f.code == 'WP5_POSITIVE_FACET_MISSING'
                   and f.context['source_id'] == 1
                   and f.context['label'] == label for f in cert.issues)
