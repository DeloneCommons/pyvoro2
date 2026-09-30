"""Internal spatial domain helpers.

This module exists to avoid duplicating small pieces of domain logic across
`api`, `normalize`, and `viz3d`.

The helpers here are intentionally lightweight and have **no** dependency on
the compiled extension.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from ...domains import Box, OrthorhombicCell, PeriodicCell
from ..inputs import coerce_finite_matrix, coerce_finite_vector
from ..native_runtime import checked_call, checked_tuple
from ..validation import require_ordered_bounds


Domain: TypeAlias = Box | OrthorhombicCell | PeriodicCell


def is_periodic_domain(domain: Domain) -> bool:
    """Return True if *any* periodic boundary condition is active."""

    if isinstance(domain, PeriodicCell):
        return True
    if isinstance(domain, OrthorhombicCell):
        return any(checked_call(bool, value) for value in checked_tuple(
            checked_call(getattr, domain, 'periodic')))
    return False


def domain_length_scale(domain: Domain) -> float:
    """Return a characteristic length scale of the domain.

    The value is used for heuristic tolerances and visualization defaults.
    It is **not** guaranteed to be a rigorous bound on any geometric quantity.
    """

    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = require_ordered_bounds(
            checked_call(getattr, domain, 'bounds'), name='domain bounds', dim=3)
        return float(max(xmax - xmin, ymax - ymin, zmax - zmin))

    vec = coerce_finite_matrix(checked_call(getattr, domain, 'vectors'),
                               name='vectors', shape=(3, 3))
    # vectors: (3,3) where each row is a lattice vector
    return float(np.max(np.linalg.norm(vec, axis=1)))


def domain_origin(domain: Domain) -> np.ndarray:
    """Return the domain origin in Cartesian coordinates."""

    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, _), (ymin, _), (zmin, _) = require_ordered_bounds(
            checked_call(getattr, domain, 'bounds'), name='domain bounds', dim=3)
        return np.array([xmin, ymin, zmin], dtype=float)
    return coerce_finite_vector(checked_call(getattr, domain, 'origin'),
                                name='origin', n=3)


def domain_lattice_vectors(
    domain: OrthorhombicCell | PeriodicCell,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, b, c) lattice translation vectors for the domain."""

    if isinstance(domain, OrthorhombicCell):
        vectors = checked_call(getattr, domain, 'lattice_vectors')
    else:
        vectors = checked_call(getattr, domain, 'vectors')
    a, b, c = coerce_finite_matrix(vectors, name='vectors', shape=(3, 3))
    return a, b, c
