"""Internal geometry adapter for planar domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...planar.domains import Box, RectangularCell
from ..inputs import (
    coerce_native_block_parameters,
    coerce_point_array,
    round_to_int64,
)
from ..validation import (
    CPP_INT_MAX,
    INT64_MAX,
    INT64_MIN,
    require_index_array,
    require_ordered_bounds,
)

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
    def native_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return a validated snapshot of rectangular native bounds."""

        validated = require_ordered_bounds(
            self.bounds,
            name='domain bounds',
            dim=2,
        )
        return validated  # type: ignore[return-value]

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
        pts = coerce_point_array(points, name='points', dim=2)
        if self.domain is None or isinstance(self.domain, Box):
            return pts
        return self.domain.remap_cart(pts, return_shifts=False)

    def shift_to_cart(self, shifts: np.ndarray) -> np.ndarray:
        raw = np.asarray(shifts, dtype=object)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError('shifts must have shape (m, 2)')
        sh = require_index_array(
            raw,
            name='shifts',
            shape=raw.shape,
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        if self.domain is None or isinstance(self.domain, Box):
            return np.zeros((sh.shape[0], 2), dtype=np.float64)
        a, b = self.lattice_vectors_cart
        with np.errstate(over='ignore', invalid='ignore'):
            translated = sh[:, 0:1] * a[None, :] + sh[:, 1:2] * b[None, :]
        if not np.all(np.isfinite(translated)):
            raise ValueError('shifts produce non-finite Cartesian translations')
        return translated

    def shift_vector(self, shift: Sequence[int] | np.ndarray) -> np.ndarray:
        raw = np.asarray(shift, dtype=object)
        if raw.shape != (2,):
            raise ValueError('shift must have shape (2,)')
        sh = require_index_array(
            raw,
            name='shift',
            shape=(2,),
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        return self.shift_to_cart(sh.reshape(1, 2)).reshape(2)

    def validate_shifts(self, shifts: np.ndarray) -> None:
        raw = np.asarray(shifts, dtype=object)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError('shifts must have shape (m, 2)')
        sh = require_index_array(
            raw,
            name='shifts',
            shape=raw.shape,
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )

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
        pi_array = coerce_point_array(pi, name='pi', dim=2)
        pj_array = coerce_point_array(pj, name='pj', dim=2)
        if pi_array.shape != pj_array.shape:
            raise ValueError('pi and pj must have the same shape')
        delta = pj_array - pi_array
        shifts = np.zeros_like(delta, dtype=np.int64)
        for ax in range(2):
            if not periodic[ax]:
                continue
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                quotient = -delta[:, ax] / lengths[ax]
            shifts[:, ax] = round_to_int64(
                quotient,
                name=f'nearest-image axis {ax} shift',
            )
        return shifts

    def resolve_block_counts(
        self,
        *,
        n_sites: int,
        blocks: tuple[int, int] | None,
        block_size: float | None,
    ) -> tuple[int, int]:
        validated_blocks, validated_block_size = coerce_native_block_parameters(
            blocks=blocks,
            block_size=block_size,
            dim=2,
        )
        if validated_blocks is not None:
            nx, ny = validated_blocks
            return nx, ny

        lengths, area = self._lengths_and_area()
        if validated_block_size is None:
            spacing = (area / max(int(n_sites), 1)) ** 0.5
            block_size_eff = max(1e-6, 2.5 * spacing)
        else:
            block_size_eff = validated_block_size

        return tuple(
            self._block_count(length, block_size_eff) for length in lengths
        )

    @staticmethod
    def _block_count(length: float, block_size: float) -> int:
        ratio = length / block_size
        if not np.isfinite(ratio) or ratio > CPP_INT_MAX:
            raise ValueError(
                'derived blocks must fit the C++ int destination range'
            )
        return max(1, int(ratio))

    def _lengths_and_area(self) -> tuple[tuple[float, float], float]:
        if self.domain is None:
            raise ValueError('a domain is required to derive block counts')
        (xmin, xmax), (ymin, ymax) = self.native_bounds
        lx = float(xmax - xmin)
        ly = float(ymax - ymin)
        return (lx, ly), float(lx * ly)


def geometry2d(domain: Domain2D | None) -> DomainGeometry2D:
    """Return the internal geometry adapter for a planar domain."""

    return DomainGeometry2D(domain)
