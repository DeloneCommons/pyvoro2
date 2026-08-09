"""Internal domain-geometry adapter for spatial code paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...domains import Box, OrthorhombicCell, PeriodicCell
from ..inputs import (
    coerce_finite_matrix,
    coerce_finite_vector,
    coerce_native_block_parameters,
    coerce_point_array,
)
from ..periodic_images import (
    MinimumImageBatch,
    minimum_image_displacements as _minimum_image_displacements,
)
from ..validation import (
    CPP_INT_MAX,
    INT64_MAX,
    INT64_MIN,
    require_index_array,
    require_ordered_bounds,
    require_positive_finite_real,
)

Domain3D = Box | OrthorhombicCell | PeriodicCell


@dataclass(frozen=True, slots=True)
class _NativePeriodicSnapshot:
    """Detached periodic data prepared once for one native-facing operation."""

    vectors: np.ndarray
    origin: np.ndarray
    rotation_to_internal: np.ndarray
    params: tuple[float, float, float, float, float, float]
    length_scale: float

    @classmethod
    def from_raw(
        cls,
        *,
        vectors: object,
        origin: object,
    ) -> '_NativePeriodicSnapshot':
        """Validate canonical values before preparing derived geometry."""

        vectors_array = coerce_finite_matrix(
            vectors,
            name='PeriodicCell.vectors',
            shape=(3, 3),
        )
        origin_array = coerce_finite_vector(
            origin,
            name='PeriodicCell.origin',
            n=3,
        )

        # Match PeriodicCell's Cartesian-to-internal basis construction, but do
        # it on validated detached values without replaying constructor
        # conditioning policy or warnings.
        a, b, _c = vectors_array
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            norm_a = float(np.linalg.norm(a))
        if not np.isfinite(norm_a) or norm_a <= 0.0:
            raise ValueError(
                'PeriodicCell.vectors must define a finite non-zero first vector'
            )
        e1 = a / norm_a
        b_perp = b - np.dot(b, e1) * e1
        with np.errstate(over='ignore', invalid='ignore'):
            norm_b_perp = float(np.linalg.norm(b_perp))
        if not np.isfinite(norm_b_perp) or norm_b_perp <= 0.0:
            raise ValueError(
                'PeriodicCell.vectors must define non-colinear first and '
                'second vectors'
            )
        e2 = b_perp / norm_b_perp
        e3 = np.cross(e1, e2)
        rotation = np.vstack([e1, e2, e3])

        transformed = (rotation @ vectors_array.T).T
        a_internal, b_internal, c_internal = transformed
        params_array = coerce_finite_vector(
            np.array(
                [
                    a_internal[0],
                    b_internal[0],
                    b_internal[1],
                    c_internal[0],
                    c_internal[1],
                    c_internal[2],
                ],
                dtype=np.float64,
            ),
            name='periodic cell parameters',
            n=6,
        )
        for index, label in ((0, 'bx'), (2, 'by'), (5, 'bz')):
            require_positive_finite_real(
                params_array[index],
                name=f'periodic cell parameter {label}',
            )

        with np.errstate(over='ignore', invalid='ignore'):
            vector_lengths = np.linalg.norm(vectors_array, axis=1)
        length_scale = float(np.max(vector_lengths))

        vectors_snapshot = np.array(vectors_array, copy=True, order='C')
        origin_snapshot = np.array(origin_array, copy=True, order='C')
        rotation_snapshot = np.array(rotation, copy=True, order='C')
        for array in (vectors_snapshot, origin_snapshot, rotation_snapshot):
            array.setflags(write=False)

        params = tuple(float(value) for value in params_array)
        return cls(
            vectors=vectors_snapshot,
            origin=origin_snapshot,
            rotation_to_internal=rotation_snapshot,
            params=params,  # type: ignore[arg-type]
            length_scale=length_scale,
        )

    def to_internal_params(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the already prepared Voro++ periodic parameters."""

        return self.params

    @property
    def lengths_and_volume(self) -> tuple[tuple[float, float, float], float]:
        """Return native diagonal lengths and their primary-cell volume."""

        bx, _bxy, by, _bxz, _byz, bz = self.params
        return (bx, by, bz), float(bx * by * bz)

    def cart_to_internal(self, points: np.ndarray) -> np.ndarray:
        """Transform Cartesian points using the prepared rotation and origin."""

        pts = np.asarray(points, dtype=np.float64) - self.origin[None, :]
        return (self.rotation_to_internal @ pts.T).T

    def internal_to_cart(self, points_internal: np.ndarray) -> np.ndarray:
        """Transform prepared internal coordinates back to Cartesian space."""

        pts = np.asarray(points_internal, dtype=np.float64)
        return (
            self.rotation_to_internal.T @ pts.T
        ).T + self.origin[None, :]

    def remap_internal(
        self,
        points_internal: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Reuse the established lower-triangular periodic remap algorithm."""

        return PeriodicCell.remap_internal(
            self,  # type: ignore[arg-type]
            points_internal,
            return_shifts=return_shifts,
            eps=eps,
        )

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points through the prepared periodic geometry."""

        points_internal = self.cart_to_internal(points)
        if return_shifts:
            remapped, shifts = self.remap_internal(
                points_internal,
                return_shifts=True,
                eps=eps,
            )
            return self.internal_to_cart(remapped), shifts
        remapped = self.remap_internal(
            points_internal,
            return_shifts=False,
            eps=eps,
        )
        return self.internal_to_cart(remapped)


@dataclass(frozen=True, slots=True)
class DomainGeometry3D:
    """Minimal internal adapter for 3D domains.

    It exposes the geometry operations that are currently duplicated across the
    API wrapper and the inverse-fitting code: primary-cell remapping, lattice
    shift conversion, nearest-image search, and block-grid heuristics.
    """

    domain: Domain3D | None

    @property
    def dim(self) -> int:
        return 3

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
    def native_bounds(self) -> tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]:
        """Return a validated snapshot of rectangular native bounds."""

        bounds = self.bounds
        if bounds is None:
            raise ValueError('rectangular native bounds require a rectangular domain')
        validated = require_ordered_bounds(
            bounds,
            name='domain bounds',
            dim=3,
        )
        return validated  # type: ignore[return-value]

    @property
    def internal_params(self) -> tuple[float, float, float, float, float, float] | None:
        if isinstance(self.domain, PeriodicCell):
            return self.native_periodic_snapshot().params
        return None

    def native_periodic_snapshot(
        self,
    ) -> _NativePeriodicSnapshot:
        """Return detached validated geometry for one native operation.

        ``PeriodicCell`` retains canonical owned tuples. Revalidate those
        values at the native boundary, then prepare one side-effect-free
        float64 snapshot for all derived native geometry.
        """

        if not isinstance(self.domain, PeriodicCell):
            raise ValueError('periodic native parameters require a PeriodicCell')
        return _NativePeriodicSnapshot.from_raw(
            vectors=self.domain.vectors,
            origin=self.domain.origin,
        )

    @property
    def native_internal_params(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return finite Voro++ periodic parameters with positive diagonals."""

        return self.native_periodic_snapshot().params

    @property
    def lattice_vectors_cart(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the 3D lattice/edge vectors in Cartesian coordinates."""

        if self.domain is None:
            raise ValueError('a domain is required to determine lattice vectors')
        if isinstance(self.domain, PeriodicCell):
            a, b, c = (
                np.asarray(vec, dtype=np.float64).reshape(3)
                for vec in self.domain.vectors
            )
            return a, b, c

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.domain.bounds
        a = np.array([xmax - xmin, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, ymax - ymin, 0.0], dtype=np.float64)
        c = np.array([0.0, 0.0, zmax - zmin], dtype=np.float64)
        return a, b, c

    def remap_cart(self, points: np.ndarray) -> np.ndarray:
        pts = coerce_point_array(points, name='points', dim=3)
        if self.domain is None or isinstance(self.domain, Box):
            return pts
        return self.domain.remap_cart(pts, return_shifts=False)

    def shift_to_cart(self, shifts: np.ndarray) -> np.ndarray:
        raw = np.asarray(shifts, dtype=object)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError('shifts must have shape (m,3)')
        sh = require_index_array(
            raw,
            name='shifts',
            shape=raw.shape,
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        if self.domain is None or isinstance(self.domain, Box):
            return np.zeros((sh.shape[0], 3), dtype=np.float64)
        a, b, c = self.lattice_vectors_cart
        with np.errstate(over='ignore', invalid='ignore'):
            translated = (
                sh[:, 0:1] * a[None, :]
                + sh[:, 1:2] * b[None, :]
                + sh[:, 2:3] * c[None, :]
            )
        if not np.all(np.isfinite(translated)):
            raise ValueError('shifts produce non-finite Cartesian translations')
        return translated

    def shift_vector(self, shift: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return the Cartesian translation vector for one integer lattice shift."""

        raw = np.asarray(shift, dtype=object)
        if raw.shape != (3,):
            raise ValueError('shift must have shape (3,)')
        sh = require_index_array(
            raw,
            name='shift',
            shape=(3,),
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        return self.shift_to_cart(sh.reshape(1, 3)).reshape(3)

    def validate_shifts(self, shifts: np.ndarray) -> None:
        raw = np.asarray(shifts, dtype=object)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError('shifts must have shape (m,3)')
        sh = require_index_array(
            raw,
            name='shifts',
            shape=raw.shape,
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )

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

    def minimum_image_displacements(
        self,
        pi: np.ndarray,
        pj: np.ndarray,
        *,
        tie_orientation: np.ndarray,
        image_search: int,
    ) -> MinimumImageBatch:
        """Return shared exact-certified minimum-image geometry."""

        if isinstance(self.domain, (OrthorhombicCell, PeriodicCell)):
            return _minimum_image_displacements(
                pi,
                pj,
                lattice_vectors=self.lattice_vectors_cart,
                periodic_axes=self.periodic_axes,
                tie_orientation=tie_orientation,
                image_search=image_search,
            )
        raise ValueError('nearest-image shifts require a periodic domain')

    def nearest_image_shifts(
        self,
        pi: np.ndarray,
        pj: np.ndarray,
        *,
        tie_orientation: np.ndarray,
        image_search: int,
    ) -> np.ndarray:
        """Return shifts from the shared certified minimum-image primitive."""

        return self.minimum_image_displacements(
            pi,
            pj,
            tie_orientation=tie_orientation,
            image_search=image_search,
        ).shift

    def resolve_block_counts(
        self,
        *,
        n_sites: int,
        blocks: tuple[int, int, int] | None,
        block_size: float | None,
        periodic_snapshot: _NativePeriodicSnapshot | None = None,
    ) -> tuple[int, int, int]:
        """Resolve the internal Voro++ block grid."""

        validated_blocks, validated_block_size = coerce_native_block_parameters(
            blocks=blocks,
            block_size=block_size,
            dim=3,
        )
        if validated_blocks is not None:
            nx, ny, nz = validated_blocks
            return nx, ny, nz

        lengths, volume = self._lengths_and_volume(
            periodic_snapshot=periodic_snapshot,
        )
        if validated_block_size is None:
            spacing = (volume / max(int(n_sites), 1)) ** (1.0 / 3.0)
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

    def _lengths_and_volume(
        self,
        *,
        periodic_snapshot: _NativePeriodicSnapshot | None = None,
    ) -> tuple[tuple[float, float, float], float]:
        if self.domain is None:
            raise ValueError('a domain is required to derive block counts')
        if isinstance(self.domain, (Box, OrthorhombicCell)):
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.native_bounds
            lx = float(xmax - xmin)
            ly = float(ymax - ymin)
            lz = float(zmax - zmin)
            return (lx, ly, lz), float(lx * ly * lz)
        snapshot = periodic_snapshot
        if snapshot is None:
            snapshot = self.native_periodic_snapshot()
        return snapshot.lengths_and_volume


def geometry3d(domain: Domain3D | None) -> DomainGeometry3D:
    """Return the internal geometry adapter for a 3D domain."""

    return DomainGeometry3D(domain)
