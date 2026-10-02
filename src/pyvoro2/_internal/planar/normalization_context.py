"""Compile WP6 authority into raw-normalization identities and obligations.

Only source-associated exact endpoint predicates emit identities. Nonpositive
artifact exemptions are separate, and do not enter the identity graph.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from fractions import Fraction
import secrets

from ..normalization_proof import (
    ProofFailure, VertexIdentity, identity_closure, normalized_operands, value_digest,
)
from .domain_geometry import geometry2d
from .wp6_certificate import WP6Failure, require_wp6
from .wp6_ideal import ExactAuditRefusal


def domain_operands(domain):
    g = geometry2d(domain)
    return g.kind, g.native_bounds, g.periodic_axes, g.lattice_vectors_cart


def _contact_operands(contact):
    return contact.status, contact.dimension, contact.endpoints, contact.length_squared


def _ideal_operands(ideal):
    return (ideal.points, ideal.weights, ideal.bounds, ideal.periods, ideal.periodic,
            tuple((i, cell.dimension, cell.vertices, cell.area,
                   {label: _contact_operands(c) for label, c in cell.contacts.items()},
                   {label: _contact_operands(c) for label, c in cell.positive.items()},
                   {label: _contact_operands(c) for label, c in cell._contacts.items()})
                  for i, cell in sorted(ideal._cells.items())))


def certificate_operands(c):
    """Bind the actual provenance, interpretation, transports and checked scope."""
    return (c.native_cells, c.packet, c.rows, c.storage, c.mode,
            tuple((f.name, getattr(c.prepared, f.name)) for f in fields(c.prepared)),
            domain_operands(c.domain), c.transport, c.periods, c.epsilon,
            c.lattice_defect,
            tuple(tuple(getattr(o, f.name) for f in fields(o)) for o in c.occurrences),
            c.audit_complete, c.audit_work,
            tuple((e.code, e.severity, str(e), e.context) for e in c.issues),
            _ideal_operands(c.effective), _ideal_operands(c.semantic))


def seal_certificate(c):
    """Bind the completed owning audit at its producer boundary, before export."""
    module = require_wp6()
    c._normalization_audit_binding = value_digest(
        (certificate_operands(c), module._qualification_identity()))
    require_wp6(artifact=False)


def _contact(c, occurrence, ideal):
    cell = ideal.cell(occurrence.source)
    return (cell.wall(occurrence.side) if occurrence.shift is None else
            cell.contact(occurrence.owner,
                         occurrence.sigma if ideal is c.effective
                         else occurrence.shift))


@dataclass(frozen=True)
class PlanarContext:
    certificate: object
    domain: bytes
    certificate_digest: bytes
    qualification_digest: bytes
    domain_digest: bytes
    raw_digest: bytes
    # Nodes use output cell position, local slot. Occurrences retain N sources.
    nodes: tuple
    identities: tuple
    anchors: tuple
    closure: tuple
    artifacts: frozenset
    positive: frozenset
    source_positions: tuple

    def operands(self):
        return (self.domain, self.certificate_digest, self.qualification_digest,
                self.domain_digest, self.raw_digest, self.nodes,
                tuple(e.operands() for e in self.identities), self.anchors,
                self.closure, tuple(sorted(self.artifacts)),
                tuple(sorted(self.positive)),
                self.source_positions)

    def check_source(self, domain):
        module = require_wp6()
        try:
            live = (value_digest(domain_operands(domain)) == self.domain_digest
                    and value_digest(certificate_operands(self.certificate))
                    == self.certificate_digest
                    and value_digest(module._qualification_identity())
                    == self.qualification_digest)
        except (TypeError, ValueError, AttributeError, KeyError):
            live = False
        require_wp6(artifact=False)
        if not live:
            raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                               'Planar proof operands, audit, provenance '
                               'or domain changed')
        closure = identity_closure(self.nodes, self.identities, domain=self.domain,
                                   dimension=2, anchors=self.anchors)
        if tuple(closure.items()) != self.closure:
            raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                               'Certified identity closure changed')

    def check_raw(self, cells, domain):
        self.check_source(domain)
        if value_digest(cells) != self.raw_digest:
            raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                               'Consumed raw occurrence snapshot changed')
        require_wp6(artifact=False)

    def bind(self, normalized, domain):
        self.check_source(domain)
        binding = value_digest((self.operands(), normalized_operands(normalized)))
        object.__setattr__(normalized, '_normalization_context', self)
        object.__setattr__(normalized, '_normalization_binding', binding)

    def check_view(self, normalized, domain):
        self.check_source(domain)
        try:
            live = value_digest((self.operands(), normalized_operands(normalized)))
        except (TypeError, ValueError, AttributeError):
            live = None
        require_wp6(artifact=False)
        if live != getattr(normalized, '_normalization_binding', None):
            raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                               'Proof-bound normalized snapshot changed '
                               'or was replaced')

        node_values = {}
        for node in self.nodes:
            position, slot = node
            cell = normalized.cells[position]
            node_values[node] = (cell['vertex_global_id'][slot],
                                 tuple(cell['vertex_shift'][slot]))
        for edge in self.identities:
            first, second = node_values[edge.first], node_values[edge.second]
            if (first[0] != second[0] or first[1] != tuple(
                    second[1][a] + edge.shift[a] for a in range(2))):
                raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                                   'Normalized mapping violates a certified '
                                   'identity lift')
        protected = {}
        for node, (point, lift) in self.anchors:
            gid = node_values[node][0]
            if node_values[node][1] != lift:
                raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                                   'Normalized vertex lift differs from its '
                                   'proved S anchor')
            if gid in protected and protected[gid] != point:
                raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                                   'Normalized mapping merges distinct '
                                   'proved S anchors')
            protected[gid] = point

    def distinct(self, position, first, second):
        anchors = dict(self.anchors)
        a, b = anchors.get((position, first)), anchors.get((position, second))
        return a is not None and b is not None and a[0] != b[0]


def context_for(normalized, domain):
    context = getattr(normalized, '_normalization_context', None)
    if context is not None:
        if not isinstance(context, PlanarContext):
            raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                               'Unknown retained normalization proof context')
        try:
            context.check_view(normalized, domain)
        except WP6Failure as exc:
            raise ProofFailure(exc.code, str(exc)) from exc
    elif hasattr(normalized, '_normalization_binding'):
        raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                           'Normalization proof context is missing')
    return context


def compile_context(c, cells, domain):
    """Use the accepted complete audit, never proximity or support guessing."""
    module = require_wp6()
    c.require_semantic_consistency()
    if getattr(c, '_normalization_audit_binding', None) != value_digest(
            (certificate_operands(c), module._qualification_identity())):
        raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                           'Completed audit operands changed before normalization')
    _check_occurrence_snapshot(c, cells)
    semantic, effective = c.semantic, c.effective
    arithmetic = semantic._arithmetic
    budget = semantic.budget
    external = tuple(int(i) for i in c.prepared.external_ids)
    positions = {int(cell['id']): p for p, cell in enumerate(cells)}
    source_positions = tuple((i, positions[external[i]]) for i in sorted(c.rows)
                             if external[i] in positions)
    by_source = dict(source_positions)
    nodes, local, physical, native_physical, anchors = [], {}, {}, {}, {}
    contacts, by_label, by_wall = {}, defaultdict(list), defaultdict(list)
    artifacts, positive = set(), set()

    def add(a, b):
        return tuple(arithmetic.add(a[k], b[k]) for k in range(2))

    def transported(point, shift, periods):
        return tuple(arithmetic.add(point[k], arithmetic.mul(shift[k], periods[k]))
                     for k in range(2))

    for source, position in source_positions:
        for slot, local2 in enumerate(c.rows[source]['local2']):
            budget.charge(stage='normalization_membership')
            node = position, slot
            nodes.append(node)
            local[node] = tuple(arithmetic.div(Fraction(float(v)), 2) for v in local2)
            physical[node] = add(semantic.points[source], local[node])
            native_physical[node] = add(effective.points[source], local[node])
            if (local[node] in semantic.cell(source).vertices
                    and local[node] in effective.cell(source).vertices):
                point, lift = [], []
                for axis in range(2):
                    quotient = physical[node][axis]
                    shift = 0
                    if semantic.periodic[axis]:
                        ratio = arithmetic.div(arithmetic.sub(
                            quotient, semantic.bounds[axis][0]), semantic.periods[axis])
                        shift = ratio.numerator // ratio.denominator
                        budget.integer(shift, stage='normalization_lift')
                        quotient = arithmetic.sub(quotient, arithmetic.mul(
                            shift, semantic.periods[axis]))
                    point.append(quotient)
                    lift.append(shift)
                anchors[node] = tuple(point), tuple(lift)

    for o in c.occurrences:
        budget.charge(stage='normalization_occurrence')
        e, s = _contact(c, o, effective), _contact(c, o, semantic)
        contacts[o.source, o.slot] = e, s
        by_label[o.source, o.owner, o.shift].append(o)
        if o.shift is None:
            by_wall[o.source, o.side].append(o)
        if o.source in by_source:
            occurrence = by_source[o.source], o.slot
            if o.collapsed and e.status == s.status and e.status in ('point', 'absent'):
                artifacts.add(occurrence)
            if e.status == s.status == 'positive':
                positive.add(occurrence)

    # Raw positive coverage and reverse N obligations are not ideal existence.
    for source, position in source_positions:
        for label in semantic.cell(source).positive:
            records = (by_wall[source, label] if isinstance(label, int)
                       else by_label[source, label[0], label[1]])
            if not any((position, o.slot) in positive for o in records):
                raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                                   'Positive S boundary lacks positive raw '
                                   'occurrence coverage')
            if isinstance(label, int):
                continue
            owner, shift = label
            if owner not in by_source or not any(
                    (by_source[owner], peer.slot) in positive
                    for peer in by_label[owner, source, tuple(-v for v in shift)]):
                raise ProofFailure('NORMALIZATION_IDENTITY_CONFLICT',
                                   'Positive S boundary lacks a positive '
                                   'reverse N class')

    execution = secrets.token_bytes(32)
    identities = []
    seen = set()

    def identity(first, second, shift, kind, sources):
        key = first, second, shift, kind
        if key in seen:
            return
        seen.add(key)
        if first not in anchors or second not in anchors:
            return
        identities.append(VertexIdentity(execution, 2, first, second,
                                         tuple(int(s) for s in shift), kind, sources))

    for o in c.occurrences:
        if o.source not in by_source:
            continue
        position = by_source[o.source]
        first, last = (position, o.slot), (position, o.next)
        e, s = contacts[o.source, o.slot]
        if o.collapsed:
            if (e.status == s.status == 'point' and e.dimension == s.dimension == 0
                    and len(e.endpoints) == len(s.endpoints) == 1
                    and local[first] == local[last] == e.endpoints[0] == s.endpoints[0]
                    and physical[first] == physical[last]
                    and native_physical[first] == native_physical[last]):
                identity(first, last, (0, 0), 'point-alias', ((o.source, o.slot),))
            continue
        if o.shift is None or (position, o.slot) not in positive:
            continue
        peers = by_label[o.owner, o.source, tuple(-v for v in o.shift)]
        for slot in (o.slot, o.next):
            budget.charge(stage='normalization_correspondence')
            first = position, slot
            if local[first] not in e.endpoints or local[first] not in s.endpoints:
                continue
            matches = []
            for peer in peers:
                if peer.source not in by_source or peer.collapsed:
                    continue
                peer_position = by_source[peer.source]
                pe, ps = contacts[peer.source, peer.slot]
                if pe.status != 'positive' or ps.status != 'positive':
                    continue
                for peer_slot in (peer.slot, peer.next):
                    budget.charge(stage='normalization_correspondence')
                    second = peer_position, peer_slot
                    if (local[second] in pe.endpoints and local[second] in ps.endpoints
                            and peer.sigma == tuple(-v for v in o.sigma)
                            and physical[first] == transported(
                                physical[second], o.shift, semantic.periods)
                            and native_physical[first] == transported(
                                native_physical[second], o.sigma, effective.periods)):
                        matches.append((peer, second))
            if len(matches) == 1:
                peer, second = matches[0]
                identity(first, second, o.shift, 'reciprocal-endpoint',
                         ((o.source, o.slot), (peer.source, peer.slot)))

    closure = identity_closure(nodes, identities, domain=execution, dimension=2,
                               anchors=anchors.items())
    require_wp6(artifact=False)
    # Snapshot after all admitted scope/contact queries. No partial graph escapes.
    return PlanarContext(c, execution, value_digest(certificate_operands(c)),
                         value_digest(module._qualification_identity()),
                         value_digest(domain_operands(domain)), value_digest(cells),
                         tuple(nodes), tuple(identities), tuple(anchors.items()),
                         tuple(closure.items()), frozenset(artifacts),
                         frozenset(positive),
                         source_positions)


def _check_occurrence_snapshot(c, cells):
    """Reconcile the consumed public views with admitted same-execution N."""
    external = tuple(int(i) for i in c.prepared.external_ids)
    expected = c.public_cells(vertices=True, adjacency=False, edges=True,
                              shifts=any(c.semantic.periodic), include_empty=True)
    by_id = {external[cell['id']]: cell for cell in expected}
    seen = set()
    for cell in cells:
        cid = cell.get('id')
        original = by_id.get(cid)
        valid = original is not None and cid not in seen
        seen.add(cid)
        if valid:
            for field in ('site', 'area', 'vertices'):
                valid = valid and value_digest(cell.get(field)) == value_digest(
                    original.get(field))
            edges = cell.get('edges', ())
            valid = valid and len(edges) == len(original.get('edges', ()))
            for edge, admitted in zip(edges, original.get('edges', ())):
                owner = admitted['adjacent_cell']
                mapped = external[owner] if owner >= 0 else owner
                valid = valid and edge.get('adjacent_cell') == mapped
                valid = valid and value_digest(edge.get('vertices')) == value_digest(
                    admitted['vertices'])
                valid = valid and (
                    value_digest(edge.get('adjacent_shift'))
                    == value_digest(admitted.get('adjacent_shift')))
        if not valid:
            raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                               'Raw slots, owner/images or ID mapping differ '
                               'from admitted N')
    present = {external[i] for i, row in c.rows.items() if row['present']}
    if not present.issubset(seen):
        raise ProofFailure('NORMALIZATION_PROOF_CONTEXT_STALE',
                           'Consumed inventory omits an admitted nonempty source cell')


def audit_resource_failure(exc):
    assert isinstance(exc, ExactAuditRefusal)
    from .wp6_certificate import WP6Failure
    return WP6Failure('WP6_AUDIT_RESOURCE', str(exc), stage=exc.stage,
                      audit_complete=False, resource=exc.resource,
                      observed=exc.observed, limit=exc.limit)
