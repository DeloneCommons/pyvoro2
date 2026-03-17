"""Internal geometry adapter for planar domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domains import Box, RectangularCell

Domain2D = Box | RectangularCell


@dataclass(frozen=True, slots=True)
class DomainGeometry2D:
    """Minimal internal adapter for 2D domains."""

    domain: Domain2D | None

    @property
    def dim(self) -> int:
        return 2

    @property
    def kind(self) -> str:
        if self.domain is None:
            return 'none'
        if isinstance(self.domain, Box):
            return 'box'
        return 'rectangular'

    @property
    def periodic_axes(self) -> tuple[bool, bool]:
        if self.domain is None or isinstance(self.domain, Box):
            return (False, False)
        return tuple(bool(v) for v in self.domain.periodic)

    @property
    def has_any_periodic_axis(self) -> bool:
        return any(self.periodic_axes)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.domain is None:
            raise ValueError('a domain is required to determine planar bounds')
        return self.domain.bounds

    @property
    def lattice_vectors_cart(self) -> tuple[np.ndarray, np.ndarray]:
        """Return planar lattice/edge vectors in Cartesian coordinates."""

        if self.domain is None:
            raise ValueError('a domain is required to determine lattice vectors')
        if isinstance(self.domain, RectangularCell):
            return self.domain.lattice_vectors

        (xmin, xmax), (ymin, ymax) = self.domain.bounds
        a = np.array([xmax - xmin, 0.0], dtype=np.float64)
        b = np.array([0.0, ymax - ymin], dtype=np.float64)
        return a, b

    def remap_cart(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if self.domain is None or isinstance(self.domain, Box):
            return pts
        return self.domain.remap_cart(pts, return_shifts=False)

    def shift_to_cart(self, shifts: np.ndarray) -> np.ndarray:
        sh = np.asarray(shifts, dtype=np.int64)
        if sh.ndim != 2 or sh.shape[1] != 2:
            raise ValueError('shifts must have shape (m, 2)')
        if self.domain is None or isinstance(self.domain, Box):
            return np.zeros((sh.shape[0], 2), dtype=np.float64)
        a, b = self.lattice_vectors_cart
        return sh[:, 0:1] * a[None, :] + sh[:, 1:2] * b[None, :]

    def shift_vector(self, shift: Sequence[int] | np.ndarray) -> np.ndarray:
        sh = np.asarray(shift, dtype=np.int64)
        if sh.shape != (2,):
            raise ValueError('shift must have shape (2,)')
        return self.shift_to_cart(sh.reshape(1, 2)).reshape(2)

    def validate_shifts(self, shifts: np.ndarray) -> None:
        sh = np.asarray(shifts, dtype=np.int64)
        if sh.ndim != 2 or sh.shape[1] != 2:
            raise ValueError('shifts must have shape (m, 2)')

        if self.domain is None or isinstance(self.domain, Box):
            if np.any(sh != 0):
                raise ValueError('constraint shifts require a periodic domain')
            return

        periodic = self.periodic_axes
        for ax in range(2):
            if not periodic[ax] and np.any(sh[:, ax] != 0):
                raise ValueError(
                    'shifts on non-periodic axes must be 0 for RectangularCell'
                )

    def nearest_image_shifts(
        self,
        pi: np.ndarray,
        pj: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(self.domain, RectangularCell):
            raise ValueError('nearest-image shifts require a periodic planar domain')
        (xmin, xmax), (ymin, ymax) = self.domain.bounds
        lengths = np.array([xmax - xmin, ymax - ymin], dtype=float)
        periodic = np.array(self.domain.periodic, dtype=bool)
        delta = np.asarray(pj, dtype=float) - np.asarray(pi, dtype=float)
        shifts = np.zeros_like(delta, dtype=np.int64)
        for ax in range(2):
            if not periodic[ax]:
                continue
            shifts[:, ax] = (-np.round(delta[:, ax] / lengths[ax])).astype(np.int64)
        return shifts

    def resolve_block_counts(
        self,
        *,
        n_sites: int,
        blocks: tuple[int, int] | None,
        block_size: float | None,
    ) -> tuple[int, int]:
        if blocks is not None:
            if len(blocks) != 2:
                raise ValueError('blocks must have length 2')
            nx, ny = (int(v) for v in blocks)
            if nx <= 0 or ny <= 0:
                raise ValueError('blocks must contain positive integers')
            return nx, ny

        lengths, area = self._lengths_and_area()
        if block_size is None:
            spacing = (area / max(int(n_sites), 1)) ** 0.5
            block_size_eff = max(1e-6, 2.5 * spacing)
        else:
            block_size_eff = float(block_size)
            if not np.isfinite(block_size_eff) or block_size_eff <= 0.0:
                raise ValueError('block_size must be a positive finite scalar')

        return tuple(max(1, int(length / block_size_eff)) for length in lengths)

    def _lengths_and_area(self) -> tuple[tuple[float, float], float]:
        if self.domain is None:
            raise ValueError('a domain is required to derive block counts')
        (xmin, xmax), (ymin, ymax) = self.domain.bounds
        lx = float(xmax - xmin)
        ly = float(ymax - ymin)
        return (lx, ly), float(lx * ly)


def geometry2d(domain: Domain2D | None) -> DomainGeometry2D:
    """Return the internal geometry adapter for a planar domain."""

    return DomainGeometry2D(domain)
