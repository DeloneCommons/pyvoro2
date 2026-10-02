"""Canonical owning-audit operands shared by its producer and consumer.

This leaf knows the planar certificate values, but imports neither producer
nor adapter. Sealing receives an identity from the already admitted producer.
"""
from dataclasses import fields

from ..normalization_proof import value_digest
from .domain_geometry import geometry2d


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


def seal_certificate(c, qualification_identity):
    """Bind the completed owning audit at its admitted producer boundary."""
    c._normalization_audit_binding = value_digest(
        (certificate_operands(c), qualification_identity))
