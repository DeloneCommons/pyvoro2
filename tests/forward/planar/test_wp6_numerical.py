"""Constructed descriptor tests, not native producer qualification evidence.

The descriptors deliberately alter raw endpoint geometry without changing
known class labels. They exercise only the numerical diagnostic after exact
class attribution/auditing; they do not claim to be native-produced packets.
"""

from fractions import Fraction as F
import importlib
import math

import pytest

from pyvoro2._internal.planar.wp6_certificate import EdgeCertificate, EdgeOccurrence
from pyvoro2._internal.planar.wp6_ideal import ExactIdeal


LEFT = (((0.25, -0.5), (0.25, 0.5)),)
RIGHT = (((-0.25, -0.5), (-0.25, 0.5)),)


def _numerical():
    name = 'pyvoro2._internal.planar.wp6_numerical'
    assert importlib.util.find_spec(name) is not None, 'numerical audit not implemented'
    return importlib.import_module(name)


def _constructed_certificate(left=LEFT, right=RIGHT, *, transport=None,
                             semantic_weights=(0, 0)):
    points = ((0.0, 0.5), (0.5, 0.5))
    transport = ((0, 0), (0, 0)) if transport is None else transport
    public = tuple(tuple(F(p[k]) + transport[i][k] for k in range(2))
                   for i, p in enumerate(points))
    effective = ExactIdeal(points, (0, 0), ((0, 1), (0, 1)),
                           (1, 1), (True, False))
    semantic = ExactIdeal(public, semantic_weights, ((0, 1), (0, 1)),
                          (1, 1), (True, False))
    for ideal in (effective, semantic):
        ideal.cell(0)
        ideal.cell(1)
    rows, occurrences = {}, []
    for source, segments in enumerate((left, right)):
        local2 = []
        for segment in segments:
            slot = len(local2)
            local2.extend(tuple(2 * float(v) for v in point) for point in segment)
            shift = tuple(transport[source][k] - transport[1 - source][k]
                          for k in range(2))
            occurrences.append(EdgeOccurrence(
                source, slot, slot + 1, 1 - source, (0, 0), shift, None,
                segment[0] == segment[1],
            ))
        rows[source] = {'local2': tuple(local2)}
    return EdgeCertificate(
        native_cells={}, packet={'periods': (1.0, 1.0)}, prepared=None,
        domain=None, mode='standard', rows=rows,
        storage={i: {'point': point} for i, point in enumerate(points)},
        transport=transport, occurrences=tuple(occurrences), periods=(F(1), F(1)),
        epsilon=(), lattice_defect=(), audit_complete=True,
        effective=effective, semantic=semantic,
    )


def test_constructed_offset_tolerance_changes_only_numerical_finding():
    right = (((-0.2499, -0.5), (-0.2499, 0.5)),)
    certificate = _constructed_certificate(right=right)
    before = certificate.occurrences
    findings = _numerical().numeric_findings(certificate, offset_tol=1e-6)
    assert len(findings) == 1
    assert findings[0].code == 'RECIPROCAL_MISMATCH'
    assert findings[0].severity == 'error'
    assert _numerical().numeric_findings(certificate, offset_tol=1e-3) == ()
    assert certificate.occurrences is before
    assert certificate.issues == ()


def test_constructed_angle_tolerance_and_optional_severity():
    right = (((-0.26, -0.5), (-0.24, 0.5)),)
    certificate = _constructed_certificate(right=right)
    findings = _numerical().numeric_findings(
        certificate, offset_tol=0.1, angle_tol=0.001, required=False,
    )
    assert [(f.code, f.severity) for f in findings] == [
        ('RECIPROCAL_MISMATCH', 'warning'),
    ]
    assert _numerical().numeric_findings(
        certificate, offset_tol=0.1, angle_tol=0.1,
    ) == ()


def test_constructed_fragments_compare_complete_union_without_pairing():
    left = (((0.25, -0.5), (0.25, 0.0)), ((0.25, 0.0), (0.25, 0.5)))
    certificate = _constructed_certificate(left=left + left)
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()
    gapped = (((-0.25, -0.5), (-0.25, -0.1)),
              ((-0.25, 0.1), (-0.25, 0.5)))
    certificate = _constructed_certificate(left=left, right=gapped)
    assert [f.code for f in _numerical().numeric_findings(certificate)] == [
        'RECIPROCAL_MISMATCH',
    ]


def test_constructed_ideal_coordinates_do_not_select_native_segment_geometry():
    certificate = _constructed_certificate(semantic_weights=(F(1, 10), 0))
    effective = certificate.effective.cell(0).contact(1, (0, 0))
    semantic = certificate.semantic.cell(0).contact(1, (0, 0))
    assert effective.endpoints != semantic.endpoints
    assert effective.status == semantic.status == 'positive'
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()


def test_constructed_public_shift_outside_int64_needs_no_public_coordinate_view():
    certificate = _constructed_certificate(transport=((2 ** 63, 0), (0, 0)))
    assert certificate.occurrences[0].shift == (2 ** 63, 0)
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()


def test_constructed_native_period_translates_union_independently_of_public_period():
    certificate = _constructed_certificate(left=RIGHT, right=LEFT)
    certificate.occurrences = tuple(
        EdgeOccurrence(o.source, o.slot, o.next, o.owner,
                       (-1, 0) if o.source == 0 else (1, 0),
                       (-1, 0) if o.source == 0 else (1, 0), None, False)
        for o in certificate.occurrences
    )
    certificate.periods = (F(9, 8), F(1))
    certificate.semantic = ExactIdeal(((0, F(1, 2)), (F(1, 2), F(1, 2))),
                                      (0, 0), ((0, F(9, 8)), (0, 1)),
                                      certificate.periods, (True, False))
    certificate.semantic.cell(0)
    certificate.semantic.cell(1)
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()


def test_constructed_common_exact_reference_preserves_translated_endpoint_offset():
    # Adding a one-unit y translation to these endpoints in binary64 loses
    # the translation. Subtracting the common exact reference first retains
    # source interval [0,8] and reciprocal interval [1,9], a real mismatch.
    big = 2 ** 54
    left = (((0.25, big), (0.25, big + 8)),)
    right = (((-0.25, big), (-0.25, big + 8)),)
    certificate = _constructed_certificate(left=left, right=right)
    points = ((0.0, 0.5), (0.5, 1.5))
    certificate.storage[1]['point'] = points[1]
    certificate.periods = (F(1), F(2))
    certificate.packet['periods'] = (1.0, 2.0)
    certificate.effective = ExactIdeal(points, (0, 0), ((0, 1), (0, 2)),
                                       (1, 2), (True, False))
    certificate.semantic = ExactIdeal(points, (0, 0), ((0, 1), (0, 2)),
                                      (1, 2), (True, False))
    for ideal in (certificate.effective, certificate.semantic):
        ideal.cell(0)
        ideal.cell(1)
    findings = _numerical().numeric_findings(certificate, offset_tol=0)
    assert [f.code for f in findings] == ['RECIPROCAL_MISMATCH']


def test_constructed_collapsed_occurrence_does_not_change_positive_union():
    extra = (((0.25, 1234.0), (0.25, 1234.0)),)
    certificate = _constructed_certificate(left=LEFT + extra)
    assert certificate.occurrences[1].collapsed
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()


def test_constructed_nonpositive_class_is_not_numerical_positive_authority():
    certificate = _constructed_certificate(right=(((20.0, 0.0), (20.0, 1.0)),))
    records = tuple(
        EdgeOccurrence(o.source, o.slot, o.next, o.source, (2, 0), (2, 0), None, False)
        for o in certificate.occurrences
    )
    certificate.occurrences = records
    assert _numerical().numeric_findings(certificate, offset_tol=0) == ()
    assert certificate.occurrences is records


def test_constructed_incomplete_exact_audit_cannot_report_numerical_success():
    certificate = _constructed_certificate()
    certificate.audit_complete = False
    findings = _numerical().numeric_findings(certificate)
    assert len(findings) == 1 and findings[0].code == 'WP6_AUDIT_INCOMPLETE'


def test_constructed_complete_comparison_workload_refuses_before_any_prefix():
    certificate = _constructed_certificate(left=LEFT * 400, right=RIGHT * 400)
    findings = _numerical().numeric_findings(certificate)
    assert len(findings) == 1
    assert findings[0].code == 'WP6_NUMERICAL_AUDIT_RESOURCE'
    assert findings[0].context['comparisons'] == 320000
    assert findings[0].context['audit_complete'] is False


def test_constructed_local_half_coordinate_underflow_is_representation_refusal():
    certificate = _constructed_certificate()
    smallest = math.ldexp(1.0, -1074)
    certificate.rows[0]['local2'] = ((0.5, 0.0), (0.5, smallest))
    certificate.rows[1]['local2'] = ((-0.5, 0.0), (-0.5, smallest))
    findings = _numerical().numeric_findings(certificate, offset_tol=0)
    assert len(findings) == 1
    assert findings[0].code == 'WP6_NUMERICAL_AUDIT_REPRESENTATION'
    assert findings[0].context['audit_complete'] is False


def test_constructed_internal_finite_coordinates_can_have_unrepresentable_union():
    certificate = _constructed_certificate()
    largest = float.fromhex('0x1.fffffffffffffp1023')
    certificate.rows[0]['local2'] = ((-largest, -largest), (largest, largest))
    certificate.rows[1]['local2'] = ((-largest, -largest), (largest, largest))
    findings = _numerical().numeric_findings(certificate)
    assert len(findings) == 1
    assert findings[0].code == 'WP6_NUMERICAL_AUDIT_REPRESENTATION'


@pytest.mark.parametrize('kwargs', [dict(offset_tol=-1), dict(angle_tol=math.inf),
                                    dict(offset_tol=True), dict(required=1)])
def test_numerical_options_preserve_strict_categories(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _numerical().numeric_findings(_constructed_certificate(), **kwargs)


def test_simple_native_descriptor_with_no_public_vertices():
    from pyvoro2.planar import RectangularCell, api
    _, certificate = api._compute_with_certificate(
        [[0.25, 0.25], [0.75, 0.75]],
        domain=RectangularCell(((0.0, 1.0), (0.0, 1.0))),
        return_vertices=False, return_diagnostics=True, tessellation_check='diagnose',
    )
    assert certificate.audit_complete
    assert _numerical().numeric_findings(certificate) == ()
