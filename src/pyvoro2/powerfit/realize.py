"""Compatibility exports for separator realization matching."""

from __future__ import annotations

from ..inverse.separator.realize import (
    RealizedPairDiagnostics,
    UnaccountedRealizedPair,
    UnaccountedRealizedPairError,
    match_realized_pairs,
)

__all__ = [
    'UnaccountedRealizedPair',
    'UnaccountedRealizedPairError',
    'RealizedPairDiagnostics',
    'match_realized_pairs',
]
