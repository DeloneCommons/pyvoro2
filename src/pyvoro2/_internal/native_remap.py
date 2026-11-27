"""Exact dyadic arithmetic for numerical native-primary representatives.

Inputs are canonical finite binary64 values and qualified native parameters.
Integer-scaled affine arithmetic stays wide until ``materialize_remap``; only
the final coordinate is rounded. This chart is distinct from A1 user wrapping.
"""
from __future__ import annotations

import numpy as np

from .inputs import coerce_point_array
from .validation import (
    INT64_MAX, INT64_MIN, require_bool, require_nonnegative_finite_real,
)


def validate_remap_inputs(points, *, dim, return_shifts, eps, name='points'):
    """Keep strict public categories at both public and snapshot remap entries."""
    expose = require_bool(return_shifts, name='return_shifts')
    pts = coerce_point_array(points, name=name, dim=dim)
    epsilon = None if eps is None else require_nonnegative_finite_real(eps, name='eps')
    return pts, expose, epsilon


def _snap_epsilon(periods, eps):
    if eps is None:
        # Preserve the scale-relative default, with no absolute floor.
        scale = max(periods, default=0.)
        eps = float(max(1e-12 * scale, 64. * float(np.finfo(float).eps) * scale))
    en, ed = eps.as_integer_ratio()
    for period in periods:
        dn, dd = period.as_integer_ratio()
        if 2 * en * dd >= dn * ed:
            raise ValueError('remap snapping neighborhoods require 2 * eps < period')
    return eps


def _align(ratios):
    """Put binary64 ratios over their largest (power-of-two) denominator."""
    denominator = max(d for _, d in ratios)
    return [n * (denominator // d) for n, d in ratios], denominator


def _reduce(residual, period, denominator, eps, lower=0):
    """Finalize one scalar chart; all arguments are exact scaled integers."""
    quotient, remainder = divmod(residual, period)
    try:
        # Integer true division rounds the entire affine expression to nearest
        # binary64, without first rounding remainder or a wide coefficient.
        value = (lower + remainder) / denominator
    except OverflowError:
        raise ValueError('remapped coordinates need a finite binary64 view') from None
    vn, vd = value.as_integer_ratio()
    # Compare represented operands exactly, including lower + period and 2*eps.
    upper_gap = (lower + period) * vd - vn * denominator
    if upper_gap <= 0 or (eps > 0 and upper_gap <= eps * vd):
        return lower / denominator, quotient + 1
    if eps > 0 and vn * denominator - lower * vd < eps * vd:
        return lower / denominator, quotient
    return value, quotient


def remap_rectangular(points, bounds, periodic, eps):
    """Return owned coordinates and wide shift rows for supported 2D/3D masks."""
    periods = [float(hi - lo) for lo, hi in bounds]
    epsilon = _snap_epsilon([d for d, active in zip(periods, periodic) if active], eps)
    result = np.array(points, dtype=np.float64, copy=True, order='C')
    shifts = [[0] * len(bounds) for _ in points]
    for axis, ((lo, hi), period, active) in enumerate(zip(bounds, periods, periodic)):
        if not active:
            continue
        constants = tuple(v.as_integer_ratio() for v in (lo, period, epsilon))
        for index, point in enumerate(points):
            (x, lower, length, tolerance), denominator = _align(
                (float(point[axis]).as_integer_ratio(),) + constants)
            value, shift = _reduce(x - lower, length, denominator, tolerance, lower)
            if not lo <= value < hi:
                raise ValueError(
                    f'remapped coordinate on axis {axis} cannot satisfy the '
                    f'declared interval [{lo!r}, {hi!r}) with its represented period')
            result[index, axis] = value
            shifts[index][axis] = shift
    return result, shifts


def remap_triangular(points, params, eps):
    """Finalize z, y, x once, propagating each committed full-row translation.

    Return wide coefficients so later source/native composition can precede
    fixed-width materialization. No integer is converted to float for subtraction.
    """
    bx, bxy, by, bxz, byz, bz = params
    epsilon = _snap_epsilon((bx, by, bz), eps)
    constants = tuple(float(v).as_integer_ratio() for v in (*params, epsilon))
    result = np.empty(points.shape, dtype=np.float64)
    shifts = []
    for index, point in enumerate(points):
        values, denominator = _align(
            tuple(float(v).as_integer_ratio() for v in point) + constants)
        x, y, z, ax, b_x, b_y, c_x, c_y, c_z, tolerance = values
        vz, nc = _reduce(z, c_z, denominator, tolerance)
        x -= nc * c_x
        y -= nc * c_y
        vy, nb = _reduce(y, b_y, denominator, tolerance)
        x -= nb * b_x
        vx, na = _reduce(x, ax, denominator, tolerance)
        result[index] = (vx, vy, vz)
        shifts.append((na, nb, nc))
    return result, shifts


def materialize_remap(points, shifts, return_shifts):
    """Check final coefficients even when a legacy caller does not request them."""
    if any(s < INT64_MIN or s > INT64_MAX for row in shifts for s in row):
        raise ValueError('remap shifts must be representable as signed int64')
    if not return_shifts:
        return points
    result = np.empty(points.shape, dtype=np.int64)
    if len(points):
        result[:] = shifts
    return points, result
