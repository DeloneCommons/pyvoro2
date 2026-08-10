"""Planar near-duplicate detection with wrapped periodic certification.

When periodic wrapping is enabled, seam-complete candidate generation is
followed by the shared certified minimum-image primitive. Native-facing calls
also apply an independent mandatory safety floor.
"""

from __future__ import annotations

from typing import Any, Literal
import sys

import warnings

import numpy as np

from ..duplicates import DuplicateError, DuplicatePair
from .._internal.duplicate_scanning import scan_close_pairs
from .._internal.inputs import coerce_point_array
from .._internal.planar.domain_geometry import geometry2d
from .._internal.validation import (
    require_bool,
    require_positive_finite_real,
    require_positive_index,
    require_string_choice,
)
from .domains import Box, RectangularCell

Domain2D = Box | RectangularCell


def duplicate_check(
    points: Any,
    *,
    threshold: float = 1e-5,
    domain: Domain2D | None = None,
    wrap: bool = True,
    mode: Literal['raise', 'warn', 'return'] = 'raise',
    max_pairs: int = 10,
) -> tuple[DuplicatePair, ...]:
    """Detect candidate planar pairs closer than an absolute threshold."""

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

    pts = coerce_point_array(points, name='points', dim=2)
    n = int(pts.shape[0])
    if n <= 1:
        return tuple()

    periodic_geometry = None
    if domain is not None and wrap_value and isinstance(domain, RectangularCell):
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)
        if any(domain.periodic):
            periodic_geometry = geometry2d(domain)

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
    msg = f'Found {count} planar point pair(s) closer than threshold={thr:g}.'
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
