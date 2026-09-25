"""Source-associated ordinary planar provenance and independent exact audits.

Native labels are established before either ideal is constructed. Neither
semantic contact nor reciprocal geometry can select or mutate those labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
import struct

from .domain_geometry import geometry2d


class WP6Failure(Exception):
    """Private structured reason, translated to planar TessellationError."""

    def __init__(self, code, message, *, severity='error', **context):
        super().__init__(message)
        self.code = code
        self.severity = severity
        self.context = context


@dataclass(frozen=True)
class AttributionLimits:
    # A complete million-scalar occurrence packet is already a substantial
    # Python allocation. Bound the final object, never a searched prefix.
    max_occurrences: int = 262_144


def _integer(value):
    if type(value) is not int:
        raise ValueError('native integer field is not an exact integer')
    return value


def _pair(values, function):
    if len(values) != 2:
        raise ValueError('planar vector must contain two components')
    return tuple(function(v) for v in values)


def _finite(value):
    if isinstance(value, bool):
        raise ValueError('Boolean native coordinate')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite native coordinate or measure')
    return result


def _bits(value):
    return struct.pack('>d', float(value))


def public_shift(sigma, first, second, *, materialize=False):
    """Transport in Python integers; int64 is only a requested-view boundary."""
    result = tuple(int(sigma[k]) + int(first[k]) - int(second[k]) for k in range(2))
    if materialize and any(s < -(2**63) or s >= 2**63 for s in result):
        raise WP6Failure(
            'WP6_SHIFT_REPRESENTATION',
            'Public planar shift is outside signed int64',
            stage='representation',
            shift=result,
        )
    return result


@dataclass(frozen=True)
class EdgeOccurrence:
    source: int
    slot: int
    next: int
    owner: int
    sigma: tuple[int, int] | None
    shift: tuple[int, int] | None
    side: int | None
    collapsed: bool

    @property
    def label(self):
        return self.side if self.shift is None else (self.owner, self.shift)

    @property
    def native_label(self):
        return self.side if self.sigma is None else (self.owner, self.sigma)


@dataclass
class EdgeCertificate:
    native_cells: dict
    packet: dict
    prepared: object
    domain: object
    mode: str
    rows: dict
    storage: dict
    transport: tuple
    occurrences: tuple[EdgeOccurrence, ...]
    periods: tuple
    epsilon: tuple
    lattice_defect: tuple
    issues: tuple[WP6Failure, ...] = ()
    audit_complete: bool = False
    effective: object = None
    semantic: object = None
    audit_work: int = 0

    @property
    def semantic_consistent(self):
        return self.audit_complete and not any(
            issue.severity == 'error' for issue in self.issues
        )

    def require_semantic_consistency(self):
        if not self.semantic_consistent:
            from ...planar.api import _raise_wp6_failure

            errors = tuple(e for e in self.issues if e.severity == 'error') or (
                WP6Failure(
                    'WP6_AUDIT_INCOMPLETE',
                    'Required exact audit is incomplete',
                    stage='semantic_audit',
                ),
            )
            _raise_wp6_failure(errors, self.domain, self.prepared, self.mode)

    def positive_boundaries(self):
        """Complete positive S generator classes, once per exact segment."""
        self.require_semantic_consistency()
        return {
            (i, label[0], label[1]): contact
            for i in range(len(self.rows))
            for label, contact in self.semantic.cell(i).positive.items()
            if not isinstance(label, int)
        }

    def boundary_measures(self):
        from .wp6_ideal import ExactAuditRefusal, length_from_squared

        try:
            return {
                key: length_from_squared(contact.length_squared)
                for key, contact in self.positive_boundaries().items()
            }
        except ExactAuditRefusal as exc:
            from ...planar.api import _raise_wp6_failure

            error = WP6Failure(
                'WP6_MEASURE_REPRESENTATION', str(exc), stage='representation'
            )
            _raise_wp6_failure(error, self.domain, self.prepared, self.mode)

    def bridge_defect(self, source, owner, sigma):
        return tuple(
            self.epsilon[owner][k]
            - self.epsilon[source][k]
            + sigma[k] * self.lattice_defect[k]
            for k in range(2)
        )

    def public_cells(self, *, vertices, adjacency, edges, shifts, include_empty):
        """Materialize source-centered native views without using public points
        as exact ideal vertices. Unrequested views are never constructed.
        """
        by_source = defaultdict(list)
        for occurrence in self.occurrences:
            by_source[occurrence.source].append(occurrence)
        ids = list(self.native_cells)
        if include_empty:
            missing = set(self.rows) - set(ids)
            if missing:
                ids = sorted(set(ids) | missing)
        result = []
        for i in ids:
            row = self.rows[i]
            present = row['present']
            site = self.prepared.input_points_cart[i].tolist()
            out = {
                'id': i,
                'site': site,
                'area': self.native_cells[i]['area'] if present else 0.0,
            }
            if not present:
                out['empty'] = True
            if vertices:
                vv = []
                for p in row.get('local2', ()):
                    try:
                        vv.append([_finite(site[k] + 0.5 * p[k]) for k in range(2)])
                    except (ValueError, OverflowError) as exc:
                        raise WP6Failure(
                            'WP6_VERTEX_REPRESENTATION',
                            'Requested native vertex view is nonfinite',
                            source_id=i,
                            stage='representation',
                        ) from exc
                out['vertices'] = vv
            if adjacency:
                outgoing = row.get('next', ())
                previous = [0] * len(outgoing)
                for k, next_slot in enumerate(outgoing):
                    previous[next_slot] = k
                out['adjacency'] = [[q, previous[k]] for k, q in enumerate(outgoing)]
            if edges:
                out['edges'] = []
                for o in by_source[i]:
                    edge = {'vertices': [o.slot, o.next], 'adjacent_cell': o.owner}
                    if shifts and o.shift is not None:
                        edge['adjacent_shift'] = public_shift(
                            o.sigma,
                            self.transport[i],
                            self.transport[o.owner],
                            materialize=True,
                        )
                    out['edges'].append(edge)
            result.append(out)
        return result


def _checked_packet(cells, packet, prepared, domain, mode, limits):
    from .wp6_profile import validate_profile

    try:
        validate_profile(packet['profile'])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise WP6Failure('WP6_PROFILE_UNSUPPORTED', str(exc), stage='profile') from exc
    geom = geometry2d(domain)
    n = len(prepared.internal_ids)
    periodic = tuple(packet['periodic'])
    if (
        len(periodic) != 2
        or any(type(v) is not bool for v in periodic)
        or periodic != geom.periodic_axes
    ):
        raise ValueError('native/public periodic mask mismatch')
    bounds = tuple(_pair(row, _finite) for row in packet['bounds'])
    if len(bounds) != 2 or any(
        _bits(bounds[k][q]) != _bits(geom.native_bounds[k][q])
        for k in range(2)
        for q in range(2)
    ):
        raise ValueError('native/public insertion bounds mismatch')
    native_periods = _pair(packet['periods'], _finite)
    if any(
        _bits(native_periods[k]) != _bits(bounds[k][1] - bounds[k][0]) for k in range(2)
    ):
        raise ValueError('native period does not match observed bounds')
    storage = {}
    transport = [None] * n
    for row in packet['inserted']:
        i = _integer(row['id'])
        if not 0 <= i < n or i in storage:
            raise ValueError('native inserted identity is duplicated or out of range')
        point = _pair(row['point'], _finite)
        radius = _finite(row['radius'])
        expected_radius = 0.0 if mode == 'standard' else prepared.backend_radii[i]
        if radius < 0 or _bits(radius) != _bits(expected_radius):
            raise ValueError('native stored radius differs from dispatched radius')
        h = _pair(row['h'], _integer)
        if any(
            (not periodic[k] and h[k] != 0) or not -(2**31) <= h[k] < 2**31
            for k in range(2)
        ):
            raise ValueError('invalid actual insertion transport')
        if _integer(row['block']) < 0 or _integer(row['slot']) < 0:
            raise ValueError('invalid native storage location')
        transport[i] = tuple(int(prepared.remap_shifts[i, k]) + h[k] for k in range(2))
        storage[i] = dict(row, point=point, radius=radius, h=h)
    if len(storage) != n:
        raise WP6Failure(
            'WP6_INSERTION_OMITTED',
            'Native inserted population does not account for every input',
            stage='insertion',
            missing=tuple(sorted(set(range(n)) - set(storage))),
        )
    rows = {}
    for row in packet['sources']:
        i = _integer(row['id'])
        if i not in storage or i in rows or type(row['present']) is not bool:
            raise ValueError('invalid source computation identity/disposition')
        rows[i] = row
    if set(rows) != set(storage):
        raise ValueError('source computation population is incomplete')
    count = sum(len(row.get('origins', ())) for row in rows.values())
    if count > limits.max_occurrences:
        raise WP6Failure(
            'WP6_ATTRIBUTION_RESOURCE',
            'Complete planar occurrence object exceeds attribution budget',
            stage='attribution',
            resource='occurrences',
            required=count,
            limit=limits.max_occurrences,
        )
    native_cells = {}
    for cell in cells:
        i = _integer(cell['id'])
        if i not in rows or i in native_cells or not rows[i]['present']:
            raise ValueError('returned geometry does not match source disposition')
        if _finite(cell['area']) < 0:
            raise ValueError('negative native area')
        if any(
            _bits(cell['site'][k]) != _bits(storage[i]['point'][k]) for k in range(2)
        ):
            raise ValueError('returned native source does not match stored source')
        native_cells[i] = cell
    if set(native_cells) != {i for i, row in rows.items() if row['present']}:
        raise ValueError('returned geometry has incomplete source associations')
    occurrences = []
    for i, row in rows.items():
        if not row['present']:
            if row.get('origins') or row.get('local2') or row.get('next'):
                raise ValueError('deleted source carries surviving occurrences')
            continue
        local2 = tuple(_pair(p, _finite) for p in row['local2'])
        outgoing = tuple(_integer(q) for q in row['next'])
        origins = tuple(row['origins'])
        m = len(local2)
        if m < 1 or len(outgoing) != m or len(origins) != m:
            raise ValueError('incomplete final outgoing occurrence association')
        if set(outgoing) != set(range(m)):
            raise ValueError('native outgoing topology is not a permutation')
        visited, q = set(), 0
        while q not in visited:
            visited.add(q)
            q = outgoing[q]
        if len(visited) != m or q != 0:
            raise ValueError('native outgoing topology is not one closed cycle')
        raw_edges = native_cells[i].get('edges')
        if raw_edges is not None and len(raw_edges) != m:
            raise ValueError('serialized/native outgoing occurrence count mismatch')
        for slot, origin in enumerate(origins):
            next_slot = outgoing[slot]
            if (
                _integer(origin['source']) != i
                or _integer(origin['slot']) != slot
                or _integer(origin['next']) != next_slot
            ):
                raise ValueError('source/slot/outgoing origin association mismatch')
            side = None
            if origin['kind'] == 'initialization':
                side = _integer(origin['side'])
                if side not in (-1, -2, -3, -4):
                    raise ValueError('unknown initialization side')
                axis = (-side - 1) // 2
                sigma = [0, 0]
                sigma[axis] = -1 if side in (-1, -3) else 1
                owner = i if periodic[axis] else side
                sigma = tuple(sigma) if periodic[axis] else None
                raw_owner = side
            elif origin['kind'] == 'particle':
                owner = _integer(origin['owner'])
                sigma = _pair(origin['sigma'], _integer)
                if owner not in storage or any(
                    s not in (-1, 0, 1) or (not periodic[k] and s != 0)
                    for k, s in enumerate(sigma)
                ):
                    raise ValueError('particle owner/image outside qualified producer')
                if owner == i and sigma == (0, 0):
                    raise ValueError('qualified producer cannot cut primary self')
                raw_owner = owner
            else:
                raise ValueError('unknown source origin kind')
            if raw_edges is not None:
                edge = raw_edges[slot]
                if (
                    tuple(edge['vertices']) != (slot, next_slot)
                    or _integer(edge['adjacent_cell']) != raw_owner
                ):
                    raise ValueError('serialized edge and source origin disagree')
            shift = (
                None
                if sigma is None
                else public_shift(sigma, transport[i], transport[owner])
            )
            occurrences.append(
                EdgeOccurrence(
                    i,
                    slot,
                    next_slot,
                    owner,
                    sigma,
                    shift,
                    side,
                    local2[slot] == local2[next_slot],
                )
            )
    periods = tuple(Fraction(float(geom.lattice_vectors_cart[k][k])) for k in range(2))
    epsilon = tuple(
        tuple(
            Fraction(storage[i]['point'][k])
            - (
                Fraction(float(prepared.input_points_cart[i, k]))
                - transport[i][k] * periods[k]
            )
            for k in range(2)
        )
        for i in range(n)
    )
    defect = tuple(Fraction(native_periods[k]) - periods[k] for k in range(2))
    return EdgeCertificate(
        native_cells,
        packet,
        prepared,
        domain,
        mode,
        rows,
        storage,
        tuple(transport),
        tuple(occurrences),
        periods,
        epsilon,
        defect,
    )


def certify_packet(
    cells,
    packet,
    prepared,
    domain,
    power_input,
    mode,
    *,
    semantic_weights=None,
    audit=False,
    reciprocity_required=True,
    limits=None,
    audit_budget=None,
):
    """All-or-error attribution; semantic refusal cannot erase known provenance."""
    try:
        certificate = _checked_packet(
            cells, packet, prepared, domain, mode, limits or AttributionLimits()
        )
    except WP6Failure:
        raise
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise WP6Failure(
            'WP6_PROVENANCE_INVALID', str(exc), stage='attribution'
        ) from exc
    if audit:
        _audit(
            certificate,
            power_input,
            semantic_weights,
            reciprocity_required,
            audit_budget,
        )
    return certificate


def _audit(certificate, power_input, semantic_weights, reciprocity_required, budget):
    from .wp6_ideal import ExactAuditBudget, ExactAuditRefusal, ExactIdeal

    c = certificate
    issues = []
    budget = budget or ExactAuditBudget()
    n = len(c.rows)
    weights = (
        semantic_weights if semantic_weights is not None else power_input.input_weights
    )
    if weights is not None and (
        len(weights) != n or any(not math.isfinite(float(w)) for w in weights)
    ):
        raise WP6Failure(
            'WP6_TRANSPORT_INVALID',
            'Invalid private semantic weights',
            stage='attribution',
        )
    if c.mode == 'standard':
        sw = ew = (Fraction(0),) * n
    else:
        ew = tuple(Fraction(c.storage[i]['radius']) ** 2 for i in range(n))
        sw = (
            tuple(Fraction(float(w)) for w in weights)
            if weights is not None
            else tuple(Fraction(float(r)) ** 2 for r in c.prepared.backend_radii)
        )
    try:
        c.effective = ExactIdeal(
            tuple(c.storage[i]['point'] for i in range(n)),
            ew,
            c.packet['bounds'],
            c.packet['periods'],
            c.packet['periodic'],
            budget=budget,
        )
        c.semantic = ExactIdeal(
            c.prepared.input_points_cart,
            sw,
            geometry2d(c.domain).native_bounds,
            c.periods,
            c.packet['periodic'],
            budget=budget,
        )
        _audit_cells(c, issues, reciprocity_required)
        c.audit_complete = True
    except ExactAuditRefusal as exc:
        issues.append(
            WP6Failure(
                'WP6_AUDIT_RESOURCE',
                str(exc),
                stage='semantic_audit',
                audit_complete=False,
                refusal=getattr(exc, 'reason', 'resource'),
            )
        )
    c.issues = tuple(issues)
    c.audit_work = getattr(budget, 'work', 0)


def _contact(cell, label):
    return cell.wall(label) if isinstance(label, int) else cell.contact(*label)


def _mapped(label, source, transport, *, reverse=False):
    if isinstance(label, int):
        return label
    owner, s = label
    if reverse:
        return owner, public_shift(s, transport[owner], transport[source])
    return owner, public_shift(s, transport[source], transport[owner])


def _audit_cells(c, issues, reciprocity_required):
    by_source = defaultdict(lambda: defaultdict(list))
    for o in c.occurrences:
        by_source[o.source][o.label].append(o)
    for i in range(len(c.rows)):
        ec, sc = c.effective.cell(i), c.semantic.cell(i)
        for authority, cell in (('E', ec), ('S', sc)):
            if cell.dimension in (0, 1):
                issues.append(
                    WP6Failure(
                        'WP6_LOWER_DIMENSIONAL_CELL',
                        'Exact cell has no full-dimensional interior',
                        source_id=i,
                        authority=authority,
                        dimension=cell.dimension,
                    )
                )
        labels = set(sc.contacts)
        labels.update(_mapped(label, i, c.transport) for label in ec.contacts)
        labels.update(by_source[i])
        for label in labels:
            e_label = _mapped(label, i, c.transport, reverse=True)
            e = _contact(ec, e_label)
            s = _contact(sc, label)
            records = by_source[i].get(label, ())
            context = dict(
                source_id=i, label=label, effective=e.status, semantic=s.status
            )
            if e.status != s.status:
                issues.append(
                    WP6Failure(
                        'WP6_EFFECTIVE_SEMANTIC_CONFLICT',
                        'E and S contact classifications differ',
                        **context,
                    )
                )
            if 'identical' in (e.status, s.status):
                issues.append(
                    WP6Failure(
                        'WP6_IDENTICAL_FUNCTION',
                        'Different provenance has an identical power function',
                        **context,
                    )
                )
            positive = e.status == 'positive' or s.status == 'positive'
            if positive and not records:
                issues.append(
                    WP6Failure(
                        'WP6_MISSING_POSITIVE_COVERAGE',
                        'Positive ideal boundary class has no native occurrence',
                        **context,
                    )
                )
            elif positive and all(o.collapsed for o in records):
                issues.append(
                    WP6Failure(
                        'WP6_COLLAPSED_POSITIVE_COVERAGE',
                        'Positive ideal boundary has only collapsed native records',
                        **context,
                    )
                )
            if records and not positive:
                if any(not o.collapsed for o in records):
                    issues.append(
                        WP6Failure(
                            'WP6_NONPOSITIVE_NATIVE_EDGE',
                            'Noncollapsed native edge has no positive ideal contact',
                            **context,
                        )
                    )
                elif e.status == s.status:
                    issues.append(
                        WP6Failure(
                            'WP6_COLLAPSED_ARTIFACT',
                            'Retained collapsed native occurrence is nonpositive',
                            severity='info',
                            occurrences=len(records),
                            **context,
                        )
                    )
            if len(records) > 1:
                issues.append(
                    WP6Failure(
                        'WP6_NATIVE_MULTIPLICITY',
                        'Several native occurrences share a provenance class',
                        severity='info',
                        occurrences=len(records),
                        **context,
                    )
                )
        _audit_reciprocal(c, i, ec, c.effective, False, issues, reciprocity_required)
        _audit_reciprocal(c, i, sc, c.semantic, True, issues, reciprocity_required)


def _audit_reciprocal(c, source, cell, ideal, public, issues, required):
    for label, contact in cell.positive.items():
        if isinstance(label, int):
            continue
        owner, shift = label
        opposite = tuple(-v for v in shift)
        reciprocal = ideal.cell(owner).contact(source, opposite)
        points = (
            c.prepared.input_points_cart
            if public
            else tuple(c.storage[i]['point'] for i in range(len(c.rows)))
        )
        periods = (
            c.periods if public else tuple(Fraction(v) for v in c.packet['periods'])
        )
        displacement = tuple(
            Fraction(float(points[owner][k]))
            + shift[k] * periods[k]
            - Fraction(float(points[source][k]))
            for k in range(2)
        )
        translated = tuple(
            sorted(
                tuple(p[k] + displacement[k] for k in range(2))
                for p in reciprocal.endpoints
            )
        )
        if (
            reciprocal.status != 'positive'
            or tuple(sorted(contact.endpoints)) != translated
        ):
            issues.append(
                WP6Failure(
                    'WP6_RECIPROCAL_CONTACT',
                    'Exact reciprocal boundary contact is inconsistent',
                    severity='error' if required else 'warning',
                    source_id=source,
                    label=label,
                    authority='S' if public else 'E',
                )
            )
