"""Planar domain specifications for 2D tessellations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domains import _default_snap_eps
from .._internal.inputs import coerce_point_array, floor_to_int64
from .._internal.validation import (
    INT64_MAX,
    require_bool,
    require_bool_tuple,
    require_nonnegative_finite_real,
    require_ordered_bounds,
)


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned non-periodic planar box."""

    bounds: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        bounds = require_ordered_bounds(self.bounds, name='bounds', dim=2)
        object.__setattr__(self, 'bounds', bounds)

    @classmethod
    def from_points(cls, points: np.ndarray, padding: float = 2.0) -> 'Box':
        """Create a bounding box that encloses planar points."""

        pts = coerce_point_array(points, name='points', dim=2)
        if pts.shape[0] == 0:
            raise ValueError('points must contain at least one point')
        padding_value = require_nonnegative_finite_real(
            padding,
            name='padding',
        )
        with np.errstate(over='ignore', invalid='ignore'):
            mins = pts.min(axis=0) - padding_value
            maxs = pts.max(axis=0) + padding_value
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            raise ValueError('points and padding must produce finite bounds')
        return cls(
            bounds=((float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1])))
        )


@dataclass(frozen=True, slots=True)
class RectangularCell:
    """Axis-aligned planar cell with optional x/y periodicity.

    This is the honest first public 2D domain scope for pyvoro2.planar.
    It intentionally does **not** cover non-orthogonal periodic cells.
    """

    bounds: tuple[tuple[float, float], tuple[float, float]]
    periodic: tuple[bool, bool] = (True, True)

    def __post_init__(self) -> None:
        bounds = require_ordered_bounds(self.bounds, name='bounds', dim=2)
        periodic = require_bool_tuple(self.periodic, name='periodic', length=2)
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'periodic', periodic)

    @property
    def lattice_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lattice vectors ``(a, b)`` in Cartesian coordinates."""

        (xmin, xmax), (ymin, ymax) = self.bounds
        a = np.array([xmax - xmin, 0.0], dtype=np.float64)
        b = np.array([0.0, ymax - ymin], dtype=np.float64)
        return a, b

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points into the primary rectangular domain."""

        return_shifts_value = require_bool(
            return_shifts,
            name='return_shifts',
        )
        pts = coerce_point_array(points, name='points', dim=2)

        (xmin, xmax), (ymin, ymax) = self.bounds
        lx = float(xmax - xmin)
        ly = float(ymax - ymin)

        if eps is None:
            lp = 0.0
            if self.periodic[0]:
                lp = max(lp, lx)
            if self.periodic[1]:
                lp = max(lp, ly)
            eps_val = _default_snap_eps(lp)
        else:
            eps_val = require_nonnegative_finite_real(eps, name='eps')

        x = pts[:, 0].astype(float, copy=True)
        y = pts[:, 1].astype(float, copy=True)
        shifts = np.zeros((pts.shape[0], 2), dtype=np.int64)

        for axis, (lo, hi, length, is_periodic) in enumerate(
            (
                (xmin, xmax, lx, self.periodic[0]),
                (ymin, ymax, ly, self.periodic[1]),
            )
        ):
            if not is_periodic:
                continue
            coord = x if axis == 0 else y
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                quotient = (coord - lo) / length
            s = floor_to_int64(quotient, name=f'points axis {axis} shift')
            with np.errstate(over='ignore', invalid='ignore'):
                coord -= s * length
            if not np.all(np.isfinite(coord)):
                raise ValueError('remapped points must contain only finite values')
            shifts[:, axis] = s

            if eps_val > 0.0:
                m0 = np.abs(coord - lo) < eps_val
                if np.any(m0):
                    coord[m0] = lo
                m1 = coord >= (hi - eps_val)
                if np.any(m1):
                    if np.any(shifts[m1, axis] == INT64_MAX):
                        raise ValueError(
                            'remap shifts must be representable as signed int64'
                        )
                    coord[m1] = lo
                    shifts[m1, axis] += 1

            if axis == 0:
                x = coord
            else:
                y = coord

        out = np.stack([x, y], axis=1).astype(np.float64)
        if return_shifts_value:
            return out, shifts
        return out
