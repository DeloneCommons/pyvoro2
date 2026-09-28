"""Arithmetic-free ghost failure protocol for guarded first entry."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral


_CODES = frozenset({
    'GHOST_BACKEND_INSERTION', 'GHOST_NATIVE_UNSUPPORTED',
    'GHOST_PROVENANCE_AMBIGUOUS', 'GHOST_PROVENANCE_INCONSISTENT',
    'GHOST_SEMANTIC_INCONSISTENT', 'GHOST_CERTIFICATION_RESOURCE',
    'GHOST_SHIFT_UNREPRESENTABLE',
})


def _bounded(value, depth=0):
    """Bound diagnostic presentation, never the certification computation."""
    if depth > 4:
        return '<detail depth limit>'
    if isinstance(value, dict):
        return {str(k)[:80]: _bounded(v, depth + 1)
                for k, v in list(value.items())[:16]}
    if isinstance(value, (tuple, list)):
        return tuple(_bounded(v, depth + 1) for v in value[:8])
    if isinstance(value, Integral):
        value = int(value)
        if value.bit_length() > 256:
            return f'<integer of {value.bit_length()} bits>'
        return value
    if isinstance(value, Fraction):
        return {'numerator': _bounded(value.numerator),
                'denominator': _bounded(value.denominator)}
    if value is None or isinstance(value, (float, bool)):
        return value
    return str(value)[:512]


class GhostFailure(ValueError):
    """Private inspectable exception; its attribute protocol is public."""

    def __init__(self, code, message, *, stage, query_index=None, **details):
        if code not in _CODES:
            raise ValueError('unknown ghost certification failure code')
        self.code = code
        self.stage = stage
        self.query_index = query_index
        self.details = _bounded(details)
        super().__init__(f'{code} [{stage}]: {str(message)[:512]}')
