"""Duplicate / near-duplicate point detection.

Voro++ contains an internal "duplicate" safeguard that can terminate the
process (via `exit(1)`) if it detects two points closer than an absolute
threshold (~1e-5 in container distance units).

This module provides the public diagnostic helper. Native-facing operations add
an independent mandatory safety floor in the private generator-preparation
layer, so turning an optional diagnostic off never weakens backend safety.

Expected complexity is O(n) for typical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import sys

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.spatial.domain_geometry import geometry3d
from ._internal.duplicate_scanning import scan_close_pairs
from ._internal.inputs import coerce_point_array
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
        self,
        message: str,
        pairs: tuple[DuplicatePair, ...],
        threshold: float,
        *,
        kind: str = 'user_threshold',
        safety_distance_squared: float = 1e-10,
        safety_distance: float = 1e-5,
        user_threshold: float | None = None,
        minimum_image_used: bool = False,
        optional_wrap_used: bool = False,
        truncated: bool = False,
        operation: str = 'duplicate_check',
        external_ids: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__(message, pairs, threshold)
        self.pairs = pairs
        self.threshold = float(threshold)
        self.kind = str(kind)
        self.safety_distance_squared = float(safety_distance_squared)
        self.safety_distance = float(safety_distance)
        self.user_threshold = (
            float(threshold) if user_threshold is None else float(user_threshold)
        )
        self.minimum_image_used = bool(minimum_image_used)
        self.optional_wrap_used = bool(optional_wrap_used)
        self.truncated = bool(truncated)
        self.operation = str(operation)
        self.external_ids = (
            tuple((pair.i, pair.j) for pair in pairs)
            if external_ids is None
            else tuple(external_ids)
        )

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
            public strict-distance diagnostic threshold.
        domain: Optional domain. If provided and `wrap=True`, points are first
            remapped into the primary periodic domain for periodic domains,
            matching native-facing preparation. Periodic candidate generation
            is seam-complete and final distances use certified minimum-image
            geometry.
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
    if domain is not None and wrap_value:
        geometry = geometry3d(domain)
        if geometry.has_any_periodic_axis:
            # Domain remap is authoritative for primary-coordinate candidate
            # generation; R4 remains authoritative for final distances.
            periodic_geometry = geometry
    if periodic_geometry is not None:
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)
    scan = scan_close_pairs(
        pts,
        radius=thr,
        geometry=periodic_geometry,
        max_pairs=max_pairs_i,
    )
    pairs = tuple(
        DuplicatePair(i=i, j=j, distance=distance)
        for i, j, distance in scan.pairs
    )
    if not pairs:
        return pairs

    count = (
        f'at least {len(pairs)}; showing {len(pairs)}'
        if scan.truncated
        else str(len(pairs))
    )
    msg = f'Found {count} point pair(s) closer than threshold={thr:g}.'

    if mode == 'raise':
        raise DuplicateError(
            msg,
            pairs,
            thr,
            kind='user_threshold',
            user_threshold=thr,
            minimum_image_used=scan.minimum_image_used,
            optional_wrap_used=bool(periodic_geometry is not None),
            truncated=scan.truncated,
            operation='duplicate_check',
        )
    if mode == 'warn':
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return pairs
