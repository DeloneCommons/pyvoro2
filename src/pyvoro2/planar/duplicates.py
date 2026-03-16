"""Planar near-duplicate point detection."""

from __future__ import annotations

from typing import Any, Literal

import warnings

import numpy as np

from ..duplicates import DuplicateError, DuplicatePair
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

    if mode not in ('raise', 'warn', 'return'):
        raise ValueError("mode must be one of: 'raise', 'warn', 'return'")

    thr = float(threshold)
    if not np.isfinite(thr) or thr <= 0.0:
        raise ValueError('threshold must be a positive finite number')
    max_pairs_i = int(max_pairs)
    if max_pairs_i <= 0:
        raise ValueError('max_pairs must be > 0')

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('points must have shape (n, 2)')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')
    n = int(pts.shape[0])
    if n <= 1:
        return tuple()

    if domain is not None and wrap and isinstance(domain, RectangularCell):
        pts = np.asarray(domain.remap_cart(pts), dtype=np.float64)

    h2 = thr * thr
    grid = np.floor(pts / thr).astype(np.int64)
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
