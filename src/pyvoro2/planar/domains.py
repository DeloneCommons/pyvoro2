"""Planar domain specifications for 2D tessellations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domains import _default_snap_eps


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned non-periodic planar box."""

    bounds: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.bounds) != 2:
            raise ValueError('bounds must have length 2')
        for lo, hi in self.bounds:
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError('bounds must be finite')
            if not hi > lo:
                raise ValueError('each bound must satisfy hi > lo')

    @classmethod
    def from_points(cls, points: np.ndarray, padding: float = 2.0) -> 'Box':
        """Create a bounding box that encloses planar points."""

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError('points must have shape (n, 2)')
        mins = pts.min(axis=0) - float(padding)
        maxs = pts.max(axis=0) + float(padding)
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
        if len(self.bounds) != 2:
            raise ValueError('bounds must have length 2')
        for lo, hi in self.bounds:
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError('bounds must be finite')
            if not hi > lo:
                raise ValueError('each bound must satisfy hi > lo')
        if len(self.periodic) != 2:
            raise ValueError('periodic must have length 2')
        object.__setattr__(
            self,
            'periodic',
            (bool(self.periodic[0]), bool(self.periodic[1])),
        )

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

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError('points must have shape (n, 2)')

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
            eps_val = float(eps)
            if eps_val < 0.0:
                raise ValueError('eps must be >= 0')

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
            s = np.floor((coord - lo) / length).astype(np.int64)
            coord -= s * length
            shifts[:, axis] = s

            if eps_val > 0.0:
                m0 = np.abs(coord - lo) < eps_val
                if np.any(m0):
                    coord[m0] = lo
                m1 = coord >= (hi - eps_val)
                if np.any(m1):
                    coord[m1] = lo
                    shifts[m1, axis] += 1

            if axis == 0:
                x = coord
            else:
                y = coord

        out = np.stack([x, y], axis=1).astype(np.float64)
        if return_shifts:
            return out, shifts
        return out
