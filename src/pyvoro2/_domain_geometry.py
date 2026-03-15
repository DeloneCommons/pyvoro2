"""Internal domain-geometry adapter for 3D code paths.

The current public package is still 3D-first, but centralizing the geometry
logic behind a small adapter makes the eventual 2D addition much less invasive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell

Domain3D = Box | OrthorhombicCell | PeriodicCell


@dataclass(frozen=True, slots=True)
class DomainGeometry3D:
    """Minimal internal adapter for 3D domains.

    It exposes the geometry operations that are currently duplicated across the
    API wrapper and the inverse-fitting code: primary-cell remapping, lattice
    shift conversion, nearest-image search, and block-grid heuristics.
    """

    domain: Domain3D | None

    @property
    def kind(self) -> str:
        if self.domain is None:
            return 'none'
        if isinstance(self.domain, Box):
            return 'box'
        if isinstance(self.domain, OrthorhombicCell):
            return 'orthorhombic'
        return 'triclinic'

    @property
    def is_rectangular(self) -> bool:
        return isinstance(self.domain, (Box, OrthorhombicCell))

    @property
    def is_triclinic(self) -> bool:
        return isinstance(self.domain, PeriodicCell)

    @property
    def periodic_axes(self) -> tuple[bool, bool, bool]:
        if self.domain is None or isinstance(self.domain, Box):
            return (False, False, False)
        if isinstance(self.domain, OrthorhombicCell):
            return tuple(bool(v) for v in self.domain.periodic)
        return (True, True, True)

    @property
    def has_any_periodic_axis(self) -> bool:
        return any(self.periodic_axes)

    @property
    def bounds(self) -> tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None:
        if self.is_rectangular:
            return self.domain.bounds  # type: ignore[return-value]
        return None

    @property
    def internal_params(self) -> tuple[float, float, float, float, float, float] | None:
        if isinstance(self.domain, PeriodicCell):
            return self.domain.to_internal_params()
        return None

    def remap_cart(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if self.domain is None or isinstance(self.domain, Box):
            return pts
        if isinstance(self.domain, OrthorhombicCell):
            return self.domain.remap_cart(pts, return_shifts=False)
        return self.domain.remap_cart(pts, return_shifts=False)

    def shift_to_cart(self, shifts: np.ndarray) -> np.ndarray:
        sh = np.asarray(shifts, dtype=np.int64)
        if sh.ndim != 2 or sh.shape[1] != 3:
            raise ValueError('shifts must have shape (m,3)')
        if self.domain is None or isinstance(self.domain, Box):
            return np.zeros((sh.shape[0], 3), dtype=np.float64)
        if isinstance(self.domain, OrthorhombicCell):
            a, b, c = self.domain.lattice_vectors
        else:
            a, b, c = (np.asarray(v, dtype=float) for v in self.domain.vectors)
        return (
            sh[:, 0:1] * a[None, :]
            + sh[:, 1:2] * b[None, :]
            + sh[:, 2:3] * c[None, :]
        )

    def validate_shifts(self, shifts: np.ndarray) -> None:
        sh = np.asarray(shifts, dtype=np.int64)
        if sh.ndim != 2 or sh.shape[1] != 3:
            raise ValueError('shifts must have shape (m,3)')

        if self.domain is None:
            if np.any(sh != 0):
                raise ValueError('constraint shifts require a periodic domain')
            return

        if isinstance(self.domain, Box):
            if np.any(sh != 0):
                raise ValueError('Box domain does not support periodic shifts')
            return

        if isinstance(self.domain, OrthorhombicCell):
            per = self.periodic_axes
            for ax in range(3):
                if not per[ax] and np.any(sh[:, ax] != 0):
                    raise ValueError(
                        'shifts on non-periodic axes must be 0 for OrthorhombicCell'
                    )

    def nearest_image_shifts(
        self,
        pi: np.ndarray,
        pj: np.ndarray,
        *,
        search: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-image shifts and a boundary-hit mask.

        The boundary-hit mask is only informative for triclinic search, where a
        best candidate lying on the search boundary suggests that a larger
        search window may be advisable.
        """

        if isinstance(self.domain, OrthorhombicCell):
            shifts = _nearest_image_shifts_orthorhombic(pi, pj, self.domain)
            return shifts, np.zeros(shifts.shape[0], dtype=bool)
        if isinstance(self.domain, PeriodicCell):
            return _nearest_image_shifts_triclinic(pi, pj, self.domain, search=search)
        raise ValueError('nearest-image shifts require a periodic domain')

    def resolve_block_counts(
        self,
        *,
        n_sites: int,
        blocks: tuple[int, int, int] | None,
        block_size: float | None,
    ) -> tuple[int, int, int]:
        """Resolve the internal Voro++ block grid."""

        if blocks is not None:
            if len(blocks) != 3:
                raise ValueError('blocks must have length 3')
            nx, ny, nz = (int(v) for v in blocks)
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise ValueError('blocks must contain positive integers')
            return nx, ny, nz

        lengths, volume = self._lengths_and_volume()
        if block_size is None:
            spacing = (volume / max(int(n_sites), 1)) ** (1.0 / 3.0)
            block_size_eff = max(1e-6, 2.5 * spacing)
        else:
            block_size_eff = float(block_size)
            if not np.isfinite(block_size_eff) or block_size_eff <= 0.0:
                raise ValueError('block_size must be a positive finite scalar')

        return tuple(max(1, int(length / block_size_eff)) for length in lengths)

    def _lengths_and_volume(self) -> tuple[tuple[float, float, float], float]:
        if self.domain is None:
            raise ValueError('a domain is required to derive block counts')
        if isinstance(self.domain, (Box, OrthorhombicCell)):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.domain.bounds
            lx = float(xmax - xmin)
            ly = float(ymax - ymin)
            lz = float(zmax - zmin)
            return (lx, ly, lz), float(lx * ly * lz)
        bx, _bxy, by, _bxz, _byz, bz = self.domain.to_internal_params()
        return (float(bx), float(by), float(bz)), float(bx * by * bz)


def geometry3d(domain: Domain3D | None) -> DomainGeometry3D:
    """Return the internal geometry adapter for a 3D domain."""

    return DomainGeometry3D(domain)



def _nearest_image_shifts_orthorhombic(
    pi: np.ndarray, pj: np.ndarray, cell: OrthorhombicCell
) -> np.ndarray:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = cell.bounds
    lengths = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)
    periodic = np.array(cell.periodic, dtype=bool)
    delta = np.asarray(pj, dtype=float) - np.asarray(pi, dtype=float)
    shifts = np.zeros_like(delta, dtype=np.int64)
    for ax in range(3):
        if not periodic[ax]:
            continue
        shifts[:, ax] = (-np.round(delta[:, ax] / lengths[ax])).astype(np.int64)
    return shifts



def _nearest_image_shifts_triclinic(
    pi: np.ndarray,
    pj: np.ndarray,
    cell: PeriodicCell,
    *,
    search: int,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = (np.asarray(v, dtype=float) for v in cell.vectors)
    rng = np.arange(-search, search + 1, dtype=np.int64)
    cand = np.array(np.meshgrid(rng, rng, rng, indexing='ij')).reshape(3, -1).T
    base = np.asarray(pj, dtype=float) - np.asarray(pi, dtype=float)
    trans = (
        cand[:, 0:1] * a[None, :]
        + cand[:, 1:2] * b[None, :]
        + cand[:, 2:3] * c[None, :]
    )
    diff = base[:, None, :] + trans[None, :, :]
    d2 = np.einsum('mki,mki->mk', diff, diff)
    best = np.argmin(d2, axis=1)
    shifts = cand[best].astype(np.int64)
    boundary_hits = np.any(np.abs(shifts) == int(search), axis=1)
    return shifts, boundary_hits.astype(bool)
