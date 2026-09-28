"""Ghost-specific failure, exact semantic eligibility and reference policy.

Native attribution is supplied by dimension-specific producers. This module
never chooses a native owner or image from mathematical contact geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral


from .ghost_failure import GhostFailure, _CODES
from .ghost_failure import _bounded as _bounded_details

_bounded = _bounded_details


@dataclass(frozen=True)
class GhostOccurrence:
    """Private already-attributed source class; n is the selected ghost."""

    owner: int | None
    shift: tuple[int, ...] | None
    wall_id: int | None = None
    collapsed: bool = False


@dataclass(frozen=True)
class SemanticCertificate:
    """Private exact disposition retained through the packaging boundary."""

    references: tuple
    semantic_dimension: int
    native_present: bool
    query_index: int


def reject_ghost_records(cells):
    """Ordinary normalization and partition diagnostics cannot consume ghosts."""
    for cell in cells:
        if isinstance(cell, dict) and isinstance(cell.get('id'), Integral):
            if int(cell['id']) == -1:
                raise ValueError(
                    'Independent ghost cells do not form a tessellation partition; '
                    'ordinary normalization and partition diagnostics reject them'
                )


def validate_materialized_cells(cells, dimension):
    """Check requested finite public views before any batch can escape."""
    measure = 'area' if dimension == 2 else 'volume'
    for cell in cells:
        fields = [('site', cell['site']), (measure, (cell[measure],))]
        fields.extend(('vertices', vertex) for vertex in cell.get('vertices', ()))
        for field, values in fields:
            try:
                finite = all(math.isfinite(float(value)) for value in values)
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise GhostFailure(
                    'GHOST_SHIFT_UNREPRESENTABLE',
                    f'Cannot materialize finite ghost {field}',
                    stage='materialization', query_index=cell['query_index'],
                    dimension=dimension, field=field,
                )


def from_native_failure(exc, dimension, query_index, stage='provenance'):
    """Translate private reasons without turning native IDs into public owners."""
    original = getattr(exc, 'code', '')
    text = str(exc)
    if text.startswith('ghost_native:'):
        parts = text.split(':', 4)
        if len(parts) == 5 and parts[3] in _CODES:
            try:
                selected = None if parts[2] == 'None' else int(parts[2])
            except ValueError:
                pass
            else:
                return GhostFailure(parts[3], parts[4], stage=parts[1],
                                    query_index=selected, dimension=dimension)
    if ('RESOURCE' in original or getattr(exc, 'reason', '') == 'resource'
            or ':resource:' in text):
        code = 'GHOST_CERTIFICATION_RESOURCE'
    elif original in ('WP5_IMAGE_UNRESOLVED', 'WP6_PROVENANCE_AMBIGUOUS'):
        code = 'GHOST_PROVENANCE_AMBIGUOUS'
    elif 'REPRESENTATION' in original or 'NONFINITE_OUTPUT' in original:
        code = 'GHOST_SHIFT_UNREPRESENTABLE'
        stage = 'materialization'
    elif ('UNSUPPORTED' in original or 'PROFILE_UNSUPPORTED' in original
          or text.startswith('planar_certification:profile:')):
        code = 'GHOST_NATIVE_UNSUPPORTED'
        stage = 'native'
    elif 'INSERTION' in original or ':insertion:' in text:
        code = 'GHOST_BACKEND_INSERTION'
        stage = 'insertion'
    else:
        code = 'GHOST_PROVENANCE_INCONSISTENT'
    context = getattr(exc, 'context', {})
    safe = {key: context[key] for key in
            ('resource', 'required', 'limit', 'stage') if key in context}
    for key in ('resource', 'observed', 'limit'):
        value = getattr(exc, key, None)
        if value is not None:
            safe[key] = value
    return GhostFailure(code, text, stage=stage, query_index=query_index,
                        dimension=dimension, native_code=original, native=safe)


def semantic_weights(power_input, query_index, n):
    """Keep original mathematical weights distinct from rounded radius squares."""
    if power_input.backend_radii is None:
        return (Fraction(),) * n, Fraction()
    if power_input.input_weights is not None:
        return (tuple(Fraction(float(v)) for v in power_input.input_weights),
                Fraction(float(power_input.input_ghost_weights[query_index])))
    return (tuple(Fraction(float(v)) ** 2 for v in power_input.backend_radii),
            Fraction(float(power_input.backend_ghost_radii[query_index])) ** 2)


def boundary_reference(occurrence, *, external_ids, periodic, query_index):
    """Materialize exactly the four-field public schema after attribution."""
    o = occurrence
    dim = len(periodic)

    def invalid(message):
        raise GhostFailure('GHOST_PROVENANCE_INCONSISTENT', message,
                           stage='provenance', query_index=query_index,
                           dimension=dim)

    if o.wall_id is not None:
        if (o.owner is not None or o.shift is not None
                or type(o.wall_id) is not int or not -2 * dim <= o.wall_id < 0
                or periodic[(-o.wall_id - 1) // 2]):
            invalid('Unqualified physical wall source')
        reference = dict(kind='wall', generator_id=None, shift=None,
                         wall_id=o.wall_id)
    else:
        if (type(o.owner) is not int or not 0 <= o.owner <= len(external_ids)
                or not isinstance(o.shift, tuple) or len(o.shift) != dim
                or any(type(v) is not int for v in o.shift)
                or any(v and not p for v, p in zip(o.shift, periodic))):
            invalid('Owner/image is outside the verified augmented population')
        self_image = o.owner == len(external_ids)
        if self_image and not any(o.shift):
            invalid('A primary self cut is not a ghost boundary')
        reference = dict(
            kind='ghost_self' if self_image else 'generator',
            generator_id=None if self_image else int(external_ids[o.owner]),
            shift=o.shift if any(periodic) else None, wall_id=None,
        )
    if o.collapsed:
        return None
    if reference['shift'] is not None and any(
            v < -(2**63) or v >= 2**63 for v in reference['shift']):
        raise GhostFailure('GHOST_SHIFT_UNREPRESENTABLE',
                           'Required public ghost shift exceeds signed int64',
                           stage='materialization', query_index=query_index,
                           dimension=dim, kind=reference['kind'],
                           generator_id=reference['generator_id'])
    return reference


def certify_semantics(*, points, ghost_site, lattice, bounds, periodic,
                      weights, ghost_weight, occurrences, present, query_index,
                      external_ids, budget=None):
    """Require complete S positivity, facet coverage and native disposition.

    Equality of full exact contact vertex sets groups coincident geometric
    facets for coverage while each source label stays independently attributed.
    No native measure or public-coordinate equality enters eligibility.
    """
    dim = len(periodic)
    n = len(points)
    sites = tuple(tuple(p) for p in points) + (tuple(ghost_site),)
    all_weights = tuple(weights) + (ghost_weight,)

    def fail(message, **details):
        raise GhostFailure('GHOST_SEMANTIC_INCONSISTENT', message,
                           stage='semantic', query_index=query_index,
                           dimension=dim, **details)

    try:
        if dim == 2:
            from .planar.wp6_ideal import ExactIdeal

            ideal = ExactIdeal(sites, all_weights, bounds,
                               tuple(lattice[k][k] for k in range(dim)),
                               periodic, budget=budget)
        else:
            from .spatial.wp5_ideal import ExactIdeal

            ideal = ExactIdeal(sites, lattice, all_weights, periodic,
                               bounds=bounds, budget=budget)
        cell = ideal.cell(n)
        if bool(present) != (cell.dimension == dim):
            fail('Native cell disposition disagrees with exact S dimension',
                 invariant='empty_disposition', native_present=bool(present),
                 semantic_dimension=cell.dimension)
        if not present:
            if occurrences:
                fail('Deleted native cell carries boundary occurrences',
                     invariant='empty_occurrences')
            return SemanticCertificate((), cell.dimension, False, query_index)
        facets = cell.positive if dim == 2 else cell.facets

        def geometry(contact):
            return frozenset(contact.endpoints if dim == 2 else contact.vertices)

        required = {geometry(contact) for contact in facets.values()}
        covered = set()
        references = []
        for slot, o in enumerate(occurrences):
            # Validate source class even for collapsed raw geometry. Range is
            # checked only once an eligible public reference is materialized.
            reference = boundary_reference(
                GhostOccurrence(o.owner, o.shift, o.wall_id, True),
                external_ids=external_ids, periodic=periodic,
                query_index=query_index,
            )
            if o.collapsed:
                references.append(reference)
                continue
            contact = (cell.wall(o.wall_id) if o.wall_id is not None
                       else cell.contact(o.owner, o.shift))
            if contact.status != 'positive':
                source = (dict(kind='wall', wall_id=o.wall_id)
                          if o.wall_id is not None else
                          dict(kind='ghost_self' if o.owner == n else 'generator',
                               generator_id=(None if o.owner == n else
                                             int(external_ids[o.owner])),
                               shift=o.shift))
                fail('Noncollapsed native occurrence has no positive S contact',
                     invariant='positive_reference', boundary_slot=slot,
                     contact_status=contact.status, source=source)
            covered.add(geometry(contact))
            references.append(boundary_reference(
                o, external_ids=external_ids, periodic=periodic,
                query_index=query_index,
            ))
        if required - covered:
            fail('Positive S geometric facets lack eligible native coverage',
                 invariant='positive_facet_coverage', required_count=len(required),
                 covered_count=len(required & covered),
                 missing_count=len(required - covered))
        return SemanticCertificate(tuple(references), cell.dimension, True,
                                   query_index)
    except GhostFailure:
        raise
    except Exception as exc:
        from .planar.wp6_ideal import ExactAuditRefusal
        from .spatial.wp5_common import WP5Failure

        if isinstance(exc, (ExactAuditRefusal, WP5Failure)):
            raise from_native_failure(exc, dim, query_index, 'semantic') from exc
        raise
