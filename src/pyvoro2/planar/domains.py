"""Planar domain specifications for 2D tessellations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._internal import native_remap
from .._internal.inputs import coerce_point_array
from .._internal.validation import (
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
        """Remap finite (n, 2) Cartesian points into the numerical primary cell.

        Active axes use the represented binary64 period ``D = hi - lo``;
        inactive coordinates are copied with zero shifts. Quotients, affine
        subtraction and snap comparisons use exact represented operands. Only
        the final coordinate is rounded, then required to lie in ``[lo, hi)``.
        Rounded upper endpoints advance to lo even with ``eps=0``.

        Epsilon defaults to 1e-12 times the maximum active period, with no
        absolute floor. Explicit eps must be finite and nonnegative. Exact
        ``2*eps < D`` is required on each active axis, including empty batches;
        overlapping/touching neighborhoods raise ValueError. An incompatible
        represented-period chart also raises ValueError.

        Outputs are owned float64 coordinates and, if return_shifts is True,
        owned int64 shifts with ``input ~= primary + shifts @ lattice``.
        Final shifts must fit int64 even when not returned; intermediate
        arithmetic stays wide. Reconstruction includes coordinate rounding and
        intentional snapping. Repeating an accepted remap with unchanged eps
        preserves coordinates and adds zero shifts. Caller inputs are unchanged.
        """

        pts, expose, epsilon = native_remap.validate_remap_inputs(
            points, dim=2, return_shifts=return_shifts, eps=eps)
        primary, shifts = native_remap.remap_rectangular(
            pts, self.bounds, self.periodic, epsilon)
        return native_remap.materialize_remap(primary, shifts, expose)
