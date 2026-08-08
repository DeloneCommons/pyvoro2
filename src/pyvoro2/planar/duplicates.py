"""Planar near-duplicate point detection."""

from __future__ import annotations

from typing import Any, Literal
import sys

import warnings

import numpy as np

from ..duplicates import DuplicateError, DuplicatePair
from .._internal.inputs import coerce_point_array, floor_to_int64
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
    """Detect planar point pairs closer than an absolute threshold."""

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

    if domain is not None and wrap_value and isinstance(domain, RectangularCell):
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)

    h2 = thr * thr
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        quotient = pts / thr
    grid = floor_to_int64(quotient, name='duplicate grid coordinates')
    neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    buckets: dict[tuple[int, int], list[int]] = {}
    found: list[DuplicatePair] = []
    for i in range(n):
        key = (int(grid[i, 0]), int(grid[i, 1]))
        x = pts[i]
        for dx, dy in neigh:
            cand = buckets.get((key[0] + dx, key[1] + dy))
            if not cand:
                continue
            for j in cand:
                d = x - pts[j]
                dist2 = float(d[0] * d[0] + d[1] * d[1])
                if dist2 < h2:
                    found.append(
                        DuplicatePair(
                            i=int(j),
                            j=int(i),
                            distance=float(np.sqrt(dist2)),
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
        f'Found {len(pairs)} planar point pair(s) closer than '
        f'threshold={thr:g}. Such near-duplicates may cause Voro++ '
        'to terminate the process.'
    )
    if mode == 'raise':
        raise DuplicateError(msg, pairs, thr)
    if mode == 'warn':
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return pairs
