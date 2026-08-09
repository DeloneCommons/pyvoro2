"""Duplicate / near-duplicate point detection.

Voro++ contains an internal "duplicate" safeguard that can terminate the
process (via `exit(1)`) if it detects two points closer than an absolute
threshold (~1e-5 in container distance units).

This module provides a fast *Python-side* pre-check to detect such cases before
calling into the C++ library.

The check is intentionally simple:
  - spatial hashing on an integer grid with cell size == threshold
  - compare each point only to points in its own grid cell and neighboring 26
    cells

When periodic wrapping is enabled (``wrap=True``), distances for candidate
pairs that are evaluated use the shared certified minimum-image primitive.
With wrapping disabled, the established unwrapped Cartesian check is
preserved.  Candidate generation itself is unchanged and is not yet a complete
periodic seam scanner; mandatory safety independent of wrapping is owned by
v0.8 R5.

Expected complexity is O(n) for typical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import sys

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.spatial.domain_utils import is_periodic_domain
from ._internal.spatial.domain_geometry import geometry3d
from ._internal.inputs import coerce_point_array, floor_to_int64
from ._internal.periodic_images import exact_distance_less_than
from ._internal.validation import (
    require_bool,
    require_positive_finite_real,
    require_positive_index,
    require_string_choice,
)


Domain = Box | OrthorhombicCell | PeriodicCell


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    i: int
    j: int
    distance: float


class DuplicateError(ValueError):
    """Raised when near-duplicate points are detected."""

    def __init__(
        self, message: str, pairs: tuple[DuplicatePair, ...], threshold: float
    ):
        super().__init__(message, pairs, threshold)
        self.pairs = pairs
        self.threshold = float(threshold)

    def __str__(self) -> str:
        return str(self.args[0])


def duplicate_check(
    points: Any,
    *,
    threshold: float = 1e-5,
    domain: Domain | None = None,
    wrap: bool = True,
    mode: Literal['raise', 'warn', 'return'] = 'raise',
    max_pairs: int = 10,
) -> tuple[DuplicatePair, ...]:
    """Detect point pairs closer than an absolute threshold.

    Args:
        points: Array-like of shape (n, 3).
        threshold: Absolute distance threshold. The default (1e-5) matches the
            effective Voro++ duplicate check (distance < 1e-5).
        domain: Optional domain. If provided and `wrap=True`, points are first
            remapped into the primary periodic domain for periodic domains,
            matching what Voro++ will do internally. Distances for evaluated
            periodic candidate pairs use certified minimum-image geometry;
            candidate generation is not yet complete across every seam.
        wrap: Whether to remap points into the primary domain when `domain` has
            periodicity. With `wrap=False`, preserve the unwrapped Cartesian
            distance check.
        mode: Behavior when duplicates are found:
            - 'raise' (default): raise :class:`DuplicateError`
            - 'warn': emit a RuntimeWarning and return the pairs
            - 'return': return the pairs without warnings
        max_pairs: Maximum number of pairs to include in the report.

    Returns:
        Tuple of DuplicatePair records (possibly empty).
    """

    mode = require_string_choice(
        mode,
        name='mode',
        choices=('raise', 'warn', 'return'),
    )

    thr = require_positive_finite_real(threshold, name='threshold')
    wrap_value = require_bool(wrap, name='wrap')
    max_pairs_i = require_positive_index(
        max_pairs,
        name='max_pairs',
        maximum=sys.maxsize,
    )

    pts = coerce_point_array(points, name='points', dim=3)
    n = int(pts.shape[0])
    if n <= 1:
        return tuple()

    periodic_geometry = None
    if domain is not None and wrap_value and is_periodic_domain(domain):
        # Domain remap is authoritative for how Voro++ will interpret periodic
        # coordinates. (For PeriodicCell, this matches the internal remap used
        # when inserting points.)
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)
        periodic_geometry = geometry3d(domain)

    h = thr
    h2 = h * h
    # grid index for each point
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        quotient = pts / h
    g = floor_to_int64(quotient, name='duplicate grid coordinates')

    # Precompute neighbor offsets
    neigh = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]

    buckets: dict[tuple[int, int, int], list[int]] = {}
    found: list[DuplicatePair] = []

    for i in range(n):
        key = (int(g[i, 0]), int(g[i, 1]), int(g[i, 2]))
        x = pts[i]

        # Check points in this bucket and adjacent buckets
        for dx, dy, dz in neigh:
            nk = (key[0] + dx, key[1] + dy, key[2] + dz)
            cand = buckets.get(nk)
            if not cand:
                continue
            for j in cand:
                if periodic_geometry is None:
                    d = x - pts[j]
                    dist2 = float(
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                    )
                    close = dist2 < h2
                else:
                    minimum = periodic_geometry.minimum_image_displacements(
                        pts[j:j + 1],
                        pts[i:i + 1],
                        tie_orientation=np.array([1], dtype=np.int8),
                        image_search=1,
                    )
                    dist2 = float(minimum.distance_squared[0])
                    close = exact_distance_less_than(
                        minimum.exact_distance_key[0],
                        thr,
                    )
                if close:
                    found.append(
                        DuplicatePair(
                            i=int(j), j=int(i), distance=float(np.sqrt(dist2))
                        )
                    )
                    if len(found) >= max_pairs_i:
                        break
            if len(found) >= max_pairs_i:
                break
        if len(found) >= max_pairs_i:
            break

        buckets.setdefault(key, []).append(i)

    pairs = tuple(found)
    if not pairs:
        return pairs

    msg = (
        f'Found {len(pairs)} point pair(s) closer than threshold={thr:g}. '
        'Such near-duplicates may cause Voro++ to terminate the process.'
    )

    if mode == 'raise':
        raise DuplicateError(msg, pairs, thr)
    if mode == 'warn':
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return pairs
