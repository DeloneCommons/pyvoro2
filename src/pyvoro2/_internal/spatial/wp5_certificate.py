"""WP5 native attribution with independent E/S semantic consistency findings.

No ideal predicate participates in producer image selection. Native labels are
atomic; semantic findings follow the public tessellation_check action policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from fractions import Fraction
import math

import numpy as np

from .wp5_common import WP5Budget, WP5Failure


def _vector(values):
    return tuple(Fraction(float(value)) for value in values)


def _matrix(values):
    return tuple(_vector(row) for row in values)


def _row_product(row, matrix):
    return tuple(sum((row[k] * matrix[k][v] for k in range(3)), Fraction())
                 for v in range(3))


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def _finite(value, *, view):
    try:
        result = float(value)
    except OverflowError as exc:
        raise WP5Failure('WP5_NONFINITE_OUTPUT_VIEW',
                         f'WP5 {view} is outside binary64') from exc
    if not math.isfinite(result):
        raise WP5Failure('WP5_NONFINITE_OUTPUT_VIEW',
                         f'WP5 {view} is nonfinite')
    return result


def _sqrt_view(value):
    """Finite scaled binary64 sqrt view; exact positivity was decided earlier."""
    if value == 0:
        return 0.0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    exponent -= exponent % 2
    scale = Fraction(2)**exponent
    try:
        result = math.ldexp(math.sqrt(float(value / scale)), exponent // 2)
    except OverflowError as exc:
        raise WP5Failure('WP5_NONFINITE_OUTPUT_VIEW',
                         'S boundary measure exceeds binary64') from exc
    return _finite(result, view='S boundary measure')


def _public_shift(sigma, first, second, *, materialize=True):
    shift = tuple(int(sigma[k]) + first[k] - second[k] for k in range(3))
    if materialize and any(value < -(2**63) or value >= 2**63 for value in shift):
        raise WP5Failure('WP5_SHIFT_REPRESENTATION',
                         'Certified public shift is outside signed int64',
                         shift=shift)
    return shift


@dataclass(frozen=True)
class FaceCertificate:
    """Complete native attribution plus an independently reported semantic audit."""

    packet: dict
    prepared: object
    domain: object
    snapshot: object
    mode: str
    shifts: tuple
    lattice: tuple
    labels: dict
    semantic: object
    effective: object
    epsilon: tuple
    lattice_defect: tuple
    frame: tuple
    charts: tuple
    work: int
    issues: tuple[WP5Failure, ...] = ()
    audit_complete: bool = True

    @property
    def semantic_consistent(self):
        return self.audit_complete and not self.issues

    def require_semantic_consistency(self):
        """Scientific realization requires more than native image identity."""
        if not self.semantic_consistent:
            from ...api import _raise_wp5_failure

            failures = self.issues or (
                WP5Failure('WP5_RESOURCE_LIMIT', 'Exact semantic audit incomplete',
                           audit_scope='semantic'),
            )
            _raise_wp5_failure(failures, self.packet, self.domain,
                               self.prepared, self.mode)

    def bridge_defect(self, source, owner, sigma):
        """Exact ADR identity, retaining both site and lattice frame defects."""
        actual = _add(
            _subtract(self.epsilon[owner], self.epsilon[source]),
            _row_product(sigma, self.lattice_defect),
        )
        return actual

    def boundary_measures(self):
        """Binary64 S-area view, scaled to avoid intermediate square overflow.

        Very small positive exact areas can round to zero in this numerical
        view. Their scientific positivity and adjacency were already certified.
        No native triangulated area enters this map.
        """
        self.require_semantic_consistency()
        try:
            return {
                (i, owner, shift): _sqrt_view(
                    self.semantic.cell(i).contact(owner, shift).area_squared
                )
                for (i, _face), (owner, shift) in self.labels.items()
                if shift is not None
            }
        except WP5Failure as exc:
            from ...api import _raise_wp5_failure

            _raise_wp5_failure(exc, self.packet, self.domain,
                               self.prepared, self.mode)

    def public_cells(self, *, return_vertices, return_adjacency, include_empty):
        """Serialize the matched ordinary native geometry in the source chart."""
        result = []
        reflected = self.snapshot is not None and self.snapshot.parity < 0
        for native in self.packet['cells']:
            i = native['id']
            if not native['computed'] and not include_empty:
                continue
            cell = {'id': i, 'site': self.prepared.input_points_cart[i].tolist(),
                    'volume': _finite(native['volume'], view='native volume')}
            if not native['computed']:
                cell['empty'] = True
            cell['faces'] = []
            for index, face in enumerate(native['faces']):
                owner, shift = self.labels[i, index]
                cycle = list(face['vertices'])
                if reflected:
                    cycle.reverse()
                item = {'vertices': cycle, 'adjacent_cell': owner}
                if shift is not None:
                    item['adjacent_shift'] = shift
                cell['faces'].append(item)
            if return_adjacency:
                cell['adjacency'] = [
                    list(reversed(row)) if reflected else list(row)
                    for row in native['adjacency']
                ]
            if return_vertices:
                # Ordinary Voro++ vertices: RN(b + RN(0.5*Y)), then the
                # existing numerical backend/user frame transform. Keep its
                # defect from p distinct; only source-chart translation is exact.
                values = []
                for row in native['vertices_doubled']:
                    values.append([
                        _finite(float(native['site'][k]) + 0.5 * float(row[k]),
                                view='native vertices') for k in range(3)
                    ])
                if values and self.snapshot is not None:
                    with np.errstate(over='ignore', invalid='ignore'):
                        values = self.snapshot.internal_to_cart(
                            np.asarray(values, dtype=np.float64)
                        ).tolist()
                translation = _row_product(self.shifts[i], self.lattice)
                cell['vertices'] = [
                    [_finite(Fraction(_finite(value, view='native frame'))
                             + translation[k], view='source-chart vertices')
                     for k, value in enumerate(row)] for row in values
                ]
            result.append(cell)
        result.sort(key=lambda cell: cell['id'])
        return result


def _check_packet(packet, n):
    """Validate packet association, including every final directed edge token."""
    try:
        _check_packet_fields(packet, n)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                         f'Malformed native witness packet: {exc}') from exc


def _check_packet_fields(packet, n):
    if len(packet.get('sites', ())) != n or len(packet.get('cells', ())) != n:
        raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                         'Witness does not cover every persistent insertion')
    if {c['id'] for c in packet['cells']} != set(range(n)):
        raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                         'Witness persistent cell identities disagree')
    for cell in packet['cells']:
        i = cell['id']
        parity = cell.get('noninterference', {})
        if parity.get('computed') is not True or (
            cell['computed'] and any(parity.get(key) is not True
                                     for key in ('geometry', 'topology',
                                                 'owners', 'volume'))
        ):
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                             'Ordinary/witness noninterference failed', source_id=i)
        origins = {origin['token']: origin for origin in cell['origins']}
        if len(origins) != len(cell['origins']):
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                             'Witness occurrence token reused', source_id=i)
        adjacency = cell['adjacency']
        if (len(adjacency) != len(cell['vertices_doubled'])
                or cell['vertex_orders'] != [len(row) for row in adjacency]):
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                             'Malformed indexed witness topology', source_id=i)
        edges = Counter()
        for face in cell['faces']:
            token = face['token']
            cycle = face['vertices']
            if (token not in origins
                    or origins[token]['kind'] == 'construction_bound'
                    or len(cycle) < 3
                    or face['edge_tokens'] != [token] * len(cycle)
                    or any(v < 0 or v >= len(cell['vertices_doubled'])
                           for v in cycle)
                    or face['legacy_owner'] != origins[token]['legacy_owner']):
                raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                                 'Malformed final witness occurrence',
                                 source_id=i, token=token)
            for first, second in zip(cycle, cycle[1:] + cycle[:1]):
                if second not in adjacency[first]:
                    raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                                     'Face cycle disagrees with indexed topology',
                                     source_id=i, token=token)
                edges[first, second] += 1
        expected_edges = Counter((i, j) for i, row in enumerate(adjacency)
                                 for j in row)
        if edges != expected_edges or any(count != 1 for count in edges.values()):
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                             'Face occurrences do not cover native indexed edges',
                             source_id=i)


def certify_packet(packet, *, prepared, power_input, domain, snapshot=None,
                   semantic_weights=None, budget=None, audit=True,
                   materialize_shifts=True):
    """Compose the frozen ADR 0021 certificate, or raise a structured refusal."""
    from .wp5_cycle import audit_cycle
    from .wp5_ideal import ExactIdeal
    from .wp5_producer import Producer

    budget = budget if budget is not None else WP5Budget()
    n = len(prepared.internal_ids)
    _check_packet(packet, n)
    producer = Producer(packet, prepared.native_points, prepared.internal_ids,
                        prepared.backend_radii, budget=budget)
    shifts = tuple(
        tuple(int(prepared.remap_shifts[i, k]) + producer.removals[i][k]
              for k in range(3)) for i in range(n)
    )
    periodic = tuple(prepared.periodic_axes)
    if snapshot is None:
        lattice = _matrix(domain.lattice_vectors)
        native_lattice = lattice
        bounds = domain.bounds
        frame = _matrix(np.eye(3))
        origin = (Fraction(),) * 3
    else:
        lattice = _matrix(snapshot.vectors)
        bx, bxy, by, bxz, byz, bz = packet['context']['cell_params']
        native_lattice = _matrix(((bx, 0, 0), (bxy, by, 0), (bxz, byz, bz)))
        bounds = None
        frame = _matrix(snapshot.rotation_to_internal.T)
        origin = _vector(snapshot.origin)
    source = tuple(_vector(row) for row in prepared.input_points_cart)
    stored = tuple(_vector(site['site']) for site in packet['sites'])
    charts = tuple(_subtract(source[i], _row_product(shifts[i], lattice))
                   for i in range(n))
    epsilon = tuple(_subtract(stored[i], _row_product(
        _subtract(charts[i], origin), frame)) for i in range(n))
    defect = tuple(_subtract(native_lattice[k], _row_product(lattice[k], frame))
                   for k in range(3))
    weighted = prepared.backend_radii is not None
    effective_weights = tuple(
        Fraction(float(site['radius']))**2 if weighted else Fraction()
        for site in packet['sites']
    )
    public_weights = (semantic_weights if semantic_weights is not None
                      else power_input.input_weights)
    if public_weights is not None:
        if not weighted or np.asarray(public_weights).shape != (n,):
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                             'Resolved semantic weights do not match insertion')
        semantic = tuple(Fraction(float(value)) for value in public_weights)
    else:
        semantic = tuple(Fraction(float(value))**2
                         for value in prepared.backend_radii) if weighted else (
                             (Fraction(),) * n)
    # Finish source attribution before starting any independent ideal work.
    # A semantic audit refusal can never erase a known native image.
    occurrences = []
    labels = {}
    for cell in packet['cells']:
        i = cell['id']
        origins = {item['token']: item for item in cell['origins']}
        for index, face in enumerate(cell['faces']):
            attribution = producer.attribute(cell, origins[face['token']])
            owner, sigma = attribution.owner, attribution.shift
            shift = (None if sigma is None else
                     _public_shift(sigma, shifts[i], shifts[owner],
                                   materialize=materialize_shifts))
            labels[i, index] = (owner, shift)
            occurrences.append((cell, face, origins[face['token']],
                                owner, sigma, shift))

    # Independent refusal accounting: exhausted ideal work is a diagnostic,
    # while producer exhaustion above prevents honest requested shift output.
    audit_budget = WP5Budget(budget.limits)
    findings = []
    effective = semantic_ideal = None
    audit_complete = bool(audit)
    try:
        if audit:
            effective = ExactIdeal(
                stored, native_lattice, effective_weights, periodic,
                bounds=bounds, budget=audit_budget,
            )
            semantic_ideal = ExactIdeal(
                source, lattice, semantic, periodic,
                bounds=bounds, budget=audit_budget,
            )
            _audit_semantics(packet, occurrences, effective, semantic_ideal,
                             shifts, audit_budget, findings, audit_cycle)
    except WP5Failure as exc:
        if exc.code != 'WP5_RESOURCE_LIMIT':
            raise
        exc.context.update(audit_scope='semantic', shifts_attributed=True,
                           audit_complete=False)
        findings.append(exc)
        audit_complete = False

    return FaceCertificate(
        packet, prepared, domain, snapshot, 'power' if weighted else 'standard',
        shifts, lattice, labels, semantic_ideal, effective, epsilon, defect,
        frame, charts, budget.work + audit_budget.work, tuple(findings),
        audit_complete,
    )


def _audit_semantics(packet, occurrences, effective, semantic_ideal,
                     shifts, budget, findings, audit_cycle):
    """Collect semantic findings without changing native labels or geometry."""
    covered = set()
    native_labels = set()
    positive = set()
    for cell, face, support, owner, sigma, shift in occurrences:
        i = cell['id']
        cell_e, cell_s = effective.cell(i), semantic_ideal.cell(i)
        if shift is None:
            contact_e, contact_s = cell_e.wall(owner), cell_s.wall(owner)
            label = ('wall', owner)
        else:
            contact_e = cell_e.contact(owner, sigma)
            contact_s = cell_s.contact(owner, shift)
            label = (owner, shift)
        context = dict(source_id=i, token=support['token'], owner=owner,
                       native_shift=sigma, public_shift=shift,
                       effective=contact_e.status, semantic=contact_s.status)
        if (i, label) in native_labels:
            findings.append(WP5Failure(
                'WP5_OCCURRENCE_MULTIPLICITY',
                'Several native occurrences cover one semantic label', **context,
            ))
        native_labels.add((i, label))
        if contact_e.status != contact_s.status:
            findings.append(WP5Failure(
                'WP5_REPRESENTATION_CONFLICT',
                'E and S exact contact statuses disagree', **context,
            ))
        elif contact_s.status != 'positive':
            findings.append(WP5Failure(
                'WP5_EXACT_ZERO' if contact_s.status == 'zero'
                else 'WP5_IDEAL_ABSENT',
                'Native occurrence has no positive ideal facet', **context,
            ))
        cycle_valid = True
        try:
            audit_cycle([cell['vertices_doubled'][v] for v in face['vertices']],
                        support['normal'], support['offset'], budget=budget)
        except WP5Failure as exc:
            if exc.code not in ('WP5_NATIVE_CYCLE_INVALID',
                                'WP5_NATIVE_CYCLE_COLLAPSED'):
                raise
            exc.context.update(context)
            findings.append(exc)
            cycle_valid = False
        if (contact_e.status == contact_s.status == 'positive'
                and cycle_valid):
            covered.add((i, label))
            if shift is not None:
                positive.add((i, owner, shift))

    # Hidden ownership remains genuine producer evidence. A missing reverse
    # positive occurrence is a finding, never an ownership or shift rewrite.
    for i, owner, shift in sorted(positive):
        reverse = (owner, i, tuple(-value for value in shift))
        if reverse not in positive:
            findings.append(WP5Failure(
                'WP5_RECIPROCAL_MISSING',
                'Positive native/ideal occurrence lacks its positive reverse',
                source_id=i, owner=owner, shift=shift,
            ))

    cells_by_id = {cell['id']: cell for cell in packet['cells']}
    for i in range(len(packet['sites'])):
        cell_e, cell_s = effective.cell(i), semantic_ideal.cell(i)
        if max(cell_e.dimension, cell_s.dimension) == 3 and not (
            cells_by_id[i]['computed']
        ):
            findings.append(WP5Failure(
                'WP5_OWNER_COVERAGE_MISSING',
                'Exact volumetric owner has no returned native cell', source_id=i,
            ))
        for ideal_name, exact_cell in (('E', cell_e), ('S', cell_s)):
            supports = {}
            for label, contact in exact_cell.facets.items():
                geometry = frozenset(contact.vertices)
                if geometry in supports and supports[geometry] != label:
                    findings.append(WP5Failure(
                        'WP5_PROVENANCE_COINCIDENT',
                        'Exact support has distinct positive labels',
                        source_id=i, ideal=ideal_name,
                        labels=(supports[geometry], label),
                    ))
                supports[geometry] = label
                public_label = label
                if ideal_name == 'E' and label[0] != 'wall':
                    owner, sigma = label
                    # This is an ideal label, not a requested returned shift.
                    public_label = (owner, tuple(
                        int(sigma[k]) + shifts[i][k] - shifts[owner][k]
                        for k in range(3)
                    ))
                if (i, public_label) not in covered:
                    findings.append(WP5Failure(
                        'WP5_POSITIVE_FACET_MISSING',
                        'Exact positive facet lacks positive native coverage',
                        source_id=i, ideal=ideal_name, label=public_label,
                    ))
