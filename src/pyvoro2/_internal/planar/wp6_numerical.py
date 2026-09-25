"""Numerical reciprocal segment-union diagnostics on certified descriptors.

Attribution and the complete E/S audit precede this optional numerical view.
Only classes positive in both ideals participate; only noncollapsed native
occurrences supply segments. No native image or exact positivity is selected,
discarded, repaired or inferred by a floating tolerance here.

Opposite occurrence groups are compared as complete unions, without fragment
pairing. Their common coordinate chart comes from the actual stored points
and native image periods. Each operand is exactified before translation and
subtraction of a shared local reference; public coordinates and public int64
shift materialization are unnecessary. Ideal endpoint coordinates are never
compared to native endpoint coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

import numpy as np

from ..validation import require_bool, require_optional_nonnegative_finite_real
from .wp6_certificate import WP6Failure


# Both directed union checks inspect at most 2*m*n source/target segment pairs.
# Bounding the COMPLETE sum also bounds all coordinate materialization for
# compared groups, and interval sorting by that sum times its logarithm. This
# permits ordinary singleton classes while refusing quadratic multiplicity.
_MAX_SEGMENT_COMPARISONS = 262_144


def _positive_groups(certificate):
    groups = defaultdict(list)
    eligible = {}
    for occurrence in certificate.occurrences:
        if occurrence.collapsed or occurrence.shift is None:
            continue
        key = occurrence.source, occurrence.owner, occurrence.shift
        if key not in eligible:
            source = occurrence.source
            effective = certificate.effective.cell(source)
            semantic = certificate.semantic.cell(source)
            eligible[key] = (occurrence.native_label in effective.positive
                             and occurrence.label in semantic.positive)
        if eligible[key]:
            groups[key].append(occurrence)
    return groups


def _comparison_pairs(groups):
    checked = set()
    pairs = []
    comparisons = 0
    for key in sorted(groups):
        if key in checked:
            continue
        source, owner, shift = key
        opposite = owner, source, (-shift[0], -shift[1])
        checked.update((key, opposite))
        # Missing or collapsed-only coverage belongs to the exact audit.
        if opposite not in groups:
            continue
        left, right = groups[key], groups[opposite]
        comparisons += 2 * len(left) * len(right)
        pairs.append((key, left, right))
    if comparisons > _MAX_SEGMENT_COMPARISONS:
        raise WP6Failure(
            'WP6_NUMERICAL_AUDIT_RESOURCE',
            'Complete reciprocal segment-union comparison exceeds its work limit',
            stage='numerical_audit', audit_complete=False, resource='comparisons',
            comparisons=comparisons, limit=_MAX_SEGMENT_COMPARISONS,
        )
    return pairs


def _local_point(certificate, occurrence, slot):
    return tuple(Fraction(value) / 2
                 for value in certificate.rows[occurrence.source]['local2'][slot])


def _segments(certificate, occurrences, reference, translation):
    result = []
    for occurrence in occurrences:
        endpoints = tuple(_local_point(certificate, occurrence, slot)
                          for slot in (occurrence.slot, occurrence.next))
        array = np.array([
            [float(point[k] + translation[k] - reference[k]) for k in range(2)]
            for point in endpoints
        ], dtype=np.float64)
        if not np.isfinite(array).all() or np.array_equal(array[0], array[1]):
            raise ArithmeticError(
                'noncollapsed native segment has no finite distinct view',
            )
        result.append(array)
    return result


def numeric_findings(certificate, offset_tol=None, angle_tol=None, required=True):
    """Return findings without modifying the descriptor or its exact audit.

    ``RECIPROCAL_MISMATCH`` is an error when reciprocity is required and a
    warning otherwise. An incomplete exact audit, unavailable numerical view,
    or comparison resource refusal is explicitly incomplete. A refusal never
    returns a successful prefix of the requested numerical inspection.
    """

    offset_tol = require_optional_nonnegative_finite_real(offset_tol, name='offset_tol')
    angle_tol = require_optional_nonnegative_finite_real(angle_tol, name='angle_tol')
    required = require_bool(required, name='required')
    if (not certificate.audit_complete or certificate.effective is None
            or certificate.semantic is None):
        return (WP6Failure(
            'WP6_AUDIT_INCOMPLETE',
            'Reciprocal numerical inspection requires a complete exact E/S audit',
            stage='numerical_audit', audit_complete=False,
        ),)

    try:
        pairs = _comparison_pairs(_positive_groups(certificate))
    except WP6Failure as failure:
        return (failure,)
    if not pairs:
        return ()

    from ...planar.diagnostics import _segment_union_covers

    findings = []
    try:
        length_scale = float(max(certificate.periods))
        offset = 1e-6 * length_scale if offset_tol is None else offset_tol
        angle = 1e-6 if angle_tol is None else angle_tol
        coordinate = max(1000.0 * offset, 128.0 * np.finfo(float).eps * length_scale)
        if not all(math.isfinite(v) for v in (length_scale, offset, coordinate)):
            raise ArithmeticError('nonfinite numerical comparison scale')
        for key, left, right in pairs:
            source, owner, shift = key
            reference = _local_point(certificate, left[0], left[0].slot)
            sigma = left[0].sigma
            displacement = tuple(
                Fraction(certificate.storage[owner]['point'][k])
                + sigma[k] * Fraction(certificate.packet['periods'][k])
                - Fraction(certificate.storage[source]['point'][k]) for k in range(2)
            )
            union_left = _segments(certificate, left, reference,
                                   (Fraction(), Fraction()))
            union_right = _segments(certificate, right, reference, displacement)
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                matching = all(
                    _segment_union_covers(
                        first, second, offset_tol=offset, angle_tol=angle,
                        coordinate_tol=coordinate,
                    ) for first, second in ((union_left, union_right),
                                            (union_right, union_left))
                )
            if not matching:
                findings.append(WP6Failure(
                    'RECIPROCAL_MISMATCH',
                    'Reciprocal native occurrence groups have different segment unions',
                    severity='error' if required else 'warning',
                    stage='numerical_audit',
                    source_id=source, label=(owner, shift),
                    source_slots=tuple(o.slot for o in left),
                    reciprocal_slots=tuple(o.slot for o in right),
                    offset_tol=offset, angle_tol=angle,
                ))
    except (ArithmeticError, ValueError) as exc:
        return (WP6Failure(
            'WP6_NUMERICAL_AUDIT_REPRESENTATION',
            'Reciprocal native segment union has no usable finite numerical view',
            stage='numerical_audit', audit_complete=False, refusal='representation',
            detail=str(exc),
        ),)
    return tuple(findings)
