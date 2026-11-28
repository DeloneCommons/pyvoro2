"""Stable source lifting into one accepted numerical native frame.

This is a local numerical chart, not exact source/native equivalence. It uses
A1 source arithmetic and R's unchanged reducer; it never prepares another frame.
Uninserted locate queries deliberately do not use this path.
"""
from __future__ import annotations

import numpy as np

from .exact_user_basis import local_residuals
from . import native_remap


def prepare_periodic_points(
    points, vectors, origin, rotation_to_internal, params, eps=None,
):
    """Return primary native coordinates and final nominal int64 source shifts.

    Canonical finite source operands and the actual paired Q/L are supplied by
    the caller. Evaluate each complete r @ Q dot exactly, then round once to
    binary64. Both source k and reducer n remain wide until materializing k+n.
    """
    coefficients, rotation_denominator = native_remap._align(
        tuple(float(v).as_integer_ratio() for v in rotation_to_internal.flat))
    rotation = [coefficients[j:j+3] for j in (0, 3, 6)]
    projected = np.empty(points.shape, dtype=np.float64)
    source_shifts = []
    try:
        for index, (residual, denominator, shift) in enumerate(
                local_residuals(points, vectors, origin)):
            projected[index] = [sum(r * q for r, q in zip(residual, row))
                                / (denominator * rotation_denominator)
                                for row in rotation]
            source_shifts.append(shift)
    except OverflowError:
        raise ValueError('local projection requires a finite binary64 view') from None
    if not np.all(np.isfinite(projected)):
        raise ValueError('local projection requires a finite binary64 view')
    primary, native_shifts = native_remap.remap_triangular(projected, params, eps)
    combined = [tuple(k + n for k, n in zip(source, native))
                for source, native in zip(source_shifts, native_shifts)]
    return native_remap.materialize_remap(primary, combined, True)
