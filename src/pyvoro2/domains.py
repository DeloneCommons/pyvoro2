"""Domain specifications for Voronoi tessellation.

pyvoro2 currently supports:
- Box: orthogonal bounding box (non-periodic, for finite systems)
- OrthorhombicCell: orthogonal cell with optional per-axis periodicity
  (1D/2D/3D periodic)
- PeriodicCell: fully periodic triclinic cell (3D crystals), implemented via
  a coordinate transform into Voro++'s lower-triangular representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._internal import exact_user_basis
from ._internal.native_periodic_frame import prepare_native_periodic_frame
from ._internal.inputs import (
    checked_int64_add,
    coerce_finite_matrix,
    coerce_finite_vector,
    coerce_point_array,
    floor_to_int64,
)
from ._internal.validation import (
    INT64_MAX,
    require_bool,
    require_bool_tuple,
    require_finite_real,
    require_nonnegative_finite_real,
    require_ordered_bounds,
    require_positive_finite_real,
)


def _default_snap_eps(L: float, *, rel: float = 1e-12) -> float:
    """Return a scale-relative snapping epsilon for remapping.

    The returned value scales with ``L`` and has **no** hard absolute floor.
    A small machine-epsilon-based lower bound keeps it from becoming
    numerically ineffective for typical floating-point ranges.
    """

    Lf = float(L)
    if not np.isfinite(Lf) or Lf <= 0.0:
        return 0.0
    epsf = float(np.finfo(float).eps)
    # Keep everything scale-relative: both terms are proportional to L.
    return float(max(rel * Lf, 64.0 * epsf * Lf))


@dataclass(frozen=True, slots=True)
class Box:
    """Orthogonal bounding box domain.

    Args:
        bounds: Three (min, max) pairs for x, y, z.

    Raises:
        ValueError: If bounds are malformed or degenerate.
    """

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        bounds = require_ordered_bounds(self.bounds, name='bounds', dim=3)
        object.__setattr__(self, 'bounds', bounds)

    @classmethod
    def from_points(cls, points: np.ndarray, padding: float = 2.0) -> 'Box':
        """Create a box that encloses points with optional padding.

        Args:
            points: Array of shape (n, 3).
            padding: Padding added on each side, in the same units as points.

        Returns:
            Box: Bounding box.

        Raises:
            ValueError: If points shape is invalid.
        """
        pts = coerce_point_array(points, name='points', dim=3)
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
            bounds=(
                (float(mins[0]), float(maxs[0])),
                (float(mins[1]), float(maxs[1])),
                (float(mins[2]), float(maxs[2])),
            )
        )


@dataclass(frozen=True, slots=True)
class OrthorhombicCell:
    """Orthorhombic cell with optional periodicity along each axis.

    This domain corresponds to Voro++'s rectangular containers (`container` and
    `container_poly`) with per-axis periodic flags.

    Args:
        bounds: Three (min, max) pairs for x, y, z.
        periodic: Tuple of three booleans (px, py, pz). If an axis is periodic,
            points may lie outside the corresponding bounds and will be remapped
            by Voro++ into the primary domain.

    Notes:
        - For periodic axes, the primary domain uses a half-open convention:
          x in [xmin, xmax), etc.
        - For non-periodic axes, the container has walls at the bounds.
    """

    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        bounds = require_ordered_bounds(self.bounds, name='bounds', dim=3)
        periodic = require_bool_tuple(self.periodic, name='periodic', length=3)
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'periodic', periodic)

    @property
    def lattice_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return lattice vectors (a, b, c) for this orthorhombic cell.

        Vectors are returned in Cartesian coordinates and correspond to
        translations that map the cell onto itself.
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        a = np.array([xmax - xmin, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, ymax - ymin, 0.0], dtype=np.float64)
        c = np.array([0.0, 0.0, zmax - zmin], dtype=np.float64)
        return a, b, c

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points into the primary orthorhombic domain.

        For each periodic axis, points are wrapped into the half-open interval
        [min, max). For non-periodic axes, coordinates are left unchanged.

        Args:
            points: Array of shape (n, 3).
            return_shifts: If True, also return integer shifts (nx, ny, nz)
                such that:

                    p_original ~= p_remapped + nx*a + ny*b + nz*c

                where a/b/c are the cell lattice vectors.
            eps: Optional snapping tolerance. If None, defaults to
                1e-12 * L where L is the maximum periodic axis length.

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (nx, ny, nz).
        """
        return_shifts_value = require_bool(
            return_shifts,
            name='return_shifts',
        )
        pts = coerce_point_array(points, name='points', dim=3)

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounds
        Lx = float(xmax - xmin)
        Ly = float(ymax - ymin)
        Lz = float(zmax - zmin)

        if eps is None:
            # Use the maximum length among periodic axes (no hard floor).
            Lp = 0.0
            if bool(self.periodic[0]):
                Lp = max(Lp, Lx)
            if bool(self.periodic[1]):
                Lp = max(Lp, Ly)
            if bool(self.periodic[2]):
                Lp = max(Lp, Lz)
            eps_val = _default_snap_eps(Lp)
        else:
            eps_val = require_nonnegative_finite_real(eps, name='eps')

        x = pts[:, 0].astype(float, copy=True)
        y = pts[:, 1].astype(float, copy=True)
        z = pts[:, 2].astype(float, copy=True)

        shifts = np.zeros((pts.shape[0], 3), dtype=np.int64)

        for axis, (lo, hi, L, is_per) in enumerate(
            (
                (xmin, xmax, Lx, self.periodic[0]),
                (ymin, ymax, Ly, self.periodic[1]),
                (zmin, zmax, Lz, self.periodic[2]),
            )
        ):
            if not is_per:
                continue
            coord = x if axis == 0 else y if axis == 1 else z
            # Wrap into [lo, hi) using floor.
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                quotient = (coord - lo) / L
            s = floor_to_int64(quotient, name=f'points axis {axis} shift')
            with np.errstate(over='ignore', invalid='ignore'):
                coord -= s * L
            if not np.all(np.isfinite(coord)):
                raise ValueError('remapped points must contain only finite values')
            shifts[:, axis] = s

            if eps_val > 0.0:
                # Snap near the lower boundary to lo.
                m0 = np.abs(coord - lo) < eps_val
                if np.any(m0):
                    coord[m0] = lo
                # Snap near the upper boundary to lo with shift increment.
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
            elif axis == 1:
                y = coord
            else:
                z = coord

        out = np.stack([x, y, z], axis=1).astype(np.float64)
        if return_shifts_value:
            return out, shifts
        return out


@dataclass(frozen=True, slots=True)
class PeriodicCell:
    """Fully periodic triclinic cell for 3D crystals.

    The supplied finite binary64 rows A define the user lattice exactly by
    x = origin + f @ A. Either handedness is accepted when det(A) is exactly
    nonzero; rows and origin are preserved. The four provisional user-coordinate
    methods make exact decisions and return rounded float64 views without Voro++.

    Native operations separately require the existing right-handed numerical
    admission policy before converting to the Voro++ representation:
        a = (bx, 0, 0)
        b = (bxy, by, 0)
        c = (bxz, byz, bz)

    and transforms points into that coordinate system before tessellation.

    Args:
        vectors: Three lattice vectors (a, b, c), each length-3.
        origin: Origin of the unit cell in Cartesian coordinates.

    Raises:
        ValueError: If vectors/origin are malformed or the rows are exactly
            singular. Native-admission and coordinate-view errors occur on use.
    """

    vectors: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        vec = coerce_finite_matrix(
            self.vectors,
            name='vectors',
            shape=(3, 3),
        )
        org = coerce_finite_vector(self.origin, name='origin', n=3)

        vectors = tuple(tuple(float(value) for value in row) for row in vec)
        origin = tuple(float(value) for value in org)
        if exact_user_basis.determinant_sign(vectors) == 0:
            raise ValueError('cell vectors are exactly singular')
        object.__setattr__(self, 'vectors', vectors)
        object.__setattr__(self, 'origin', origin)

    def cart_to_fractional(self, points: np.ndarray) -> np.ndarray:
        """Round the exact user-basis solve ``(points - origin) @ inverse(A)``.

        Provisional. Accept finite real (n, 3) binary64 inputs and return a new
        owned float64 array. Subtraction and solving precede all rounding.
        A non-finite result view raises ValueError at this operation.
        """
        pts = coerce_point_array(points, name='points', dim=3)
        return exact_user_basis.cart_to_fractional(pts, self.vectors, self.origin)

    def fractional_to_cart(self, fractional: np.ndarray) -> np.ndarray:
        """Round the exact affine expression ``origin + fractional @ A``.

        Provisional. Inputs have shape (n, 3); the returned float64 array is
        owned and rounds each exact binary64-input expression to nearest-even.
        A non-finite result view raises ValueError at this operation.
        """
        f = coerce_point_array(fractional, name='fractional', dim=3)
        return exact_user_basis.fractional_to_cart(f, self.vectors, self.origin)

    def wrap_fractional(
        self, fractional: np.ndarray, *, return_shifts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return a rounded view of ``fractional - floor(fractional)``.

        Provisional. Exact floors use the canonical binary64 input. An interior
        remainder may round to 1.0; no snapping or shift correction is applied.
        Optional shifts are owned int64 (n, 3), checked before casting. Without
        returned shifts there is no int64 restriction. Inputs have shape (n, 3).
        """
        expose = require_bool(return_shifts, name='return_shifts')
        f = coerce_point_array(fractional, name='fractional', dim=3)
        return exact_user_basis.wrap_fractional(f, expose)

    def wrap_cart(
        self, points: np.ndarray, *, return_shifts: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Wrap into the original user parallelepiped with exact decisions.

        Provisional. Solve f = (x-o) @ inverse(A) exactly, then take s = floor(f)
        and round x - s @ A to nearest-even. This does not compose the rounded
        converters and does not use native remapping or an epsilon. Coordinates
        are owned float64 (n, 3); optional shifts are owned int64 (n, 3).
        Non-finite views or requested out-of-range shifts raise ValueError.
        Unreturned shifts have no int64 restriction. Rounded views need not be
        idempotent under wrapping.
        """
        expose = require_bool(return_shifts, name='return_shifts')
        pts = coerce_point_array(points, name='points', dim=3)
        return exact_user_basis.wrap_cart(pts, self.vectors, self.origin, expose)

    @classmethod
    def from_params(
        cls,
        bx: float,
        bxy: float,
        by: float,
        bxz: float,
        byz: float,
        bz: float,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> 'PeriodicCell':
        """Create a PeriodicCell from Voro++ internal cell parameters.

        Voro++ represents a triclinic periodic cell via the lower-triangular
        lattice vectors:

            a = (bx,  0,  0)
            b = (bxy, by, 0)
            c = (bxz, byz, bz)

        This constructor allows users to initialize a :class:`PeriodicCell`
        directly using those parameters.

        Args:
            bx: x component of vector a.
            bxy: x component of vector b.
            by: y component of vector b.
            bxz: x component of vector c.
            byz: y component of vector c.
            bz: z component of vector c.
            origin: Origin of the unit cell in Cartesian coordinates.

        Returns:
            PeriodicCell: A fully periodic triclinic cell.

        Notes:
            All parameters are validated before vector construction. The three
            diagonal lengths ``bx``, ``by``, and ``bz`` must be positive.
        """
        bx_value = require_positive_finite_real(bx, name='bx')
        bxy_value = require_finite_real(bxy, name='bxy')
        by_value = require_positive_finite_real(by, name='by')
        bxz_value = require_finite_real(bxz, name='bxz')
        byz_value = require_finite_real(byz, name='byz')
        bz_value = require_positive_finite_real(bz, name='bz')
        a = (bx_value, 0.0, 0.0)
        b = (bxy_value, by_value, 0.0)
        c = (bxz_value, byz_value, bz_value)
        return cls(vectors=(a, b, c), origin=origin)

    def _rotation_to_internal(self) -> np.ndarray:
        """Return the orthogonal Cartesian-to-native frame, possibly improper."""
        return prepare_native_periodic_frame(self.vectors).rotation_to_internal

    def to_internal_params(self) -> tuple[float, float, float, float, float, float]:
        """Return (bx, bxy, by, bxz, byz, bz) from the qualified QR factor.

        Native diagonal lengths are positive for either source handedness.
        The numerical frame does not certify exact source/native equivalence.
        """
        return prepare_native_periodic_frame(self.vectors).params

    def cart_to_internal(self, points: np.ndarray) -> np.ndarray:
        """Transform Cartesian points into the internal coordinate system."""
        r = self._rotation_to_internal()
        origin = np.asarray(self.origin, dtype=float)
        pts = coerce_point_array(points, name='points', dim=3)
        with np.errstate(over='ignore', invalid='ignore'):
            result = (r @ (pts - origin[None, :]).T).T
        if not np.all(np.isfinite(result)):
            raise ValueError('points produce non-finite internal coordinates')
        return result

    def internal_to_cart(self, points_internal: np.ndarray) -> np.ndarray:
        """Transform internal points back into Cartesian coordinates."""
        r = self._rotation_to_internal()
        origin = np.asarray(self.origin, dtype=float)
        pts = coerce_point_array(
            points_internal,
            name='points_internal',
            dim=3,
        )
        with np.errstate(over='ignore', invalid='ignore'):
            result = (r.T @ pts.T).T + origin[None, :]
        if not np.all(np.isfinite(result)):
            raise ValueError('points_internal produce non-finite coordinates')
        return result

    def remap_internal(
        self,
        points_internal: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap internal coordinates into the primary periodic domain.

        Voro++ stores a triclinic periodic domain using the lower-triangular
        vectors:

            a = (bx,  0,  0)
            b = (bxy, by, 0)
            c = (bxz, byz, bz)

        Remapping into the primary domain is *not* an independent modulo on
        x/y/z when the cell is sheared (bxy/bxz/byz != 0). In particular,
        wrapping in z shifts x and y, and wrapping in y shifts x.

        This routine performs a lattice-consistent remap equivalent to applying
        integer translations along c, then b, then a. It enforces a half-open
        convention for the primary cell:

            x in [0, bx), y in [0, by), z in [0, bz)

        To make the mapping deterministic at boundaries (and stable under
        repeated remapping), an epsilon snapping rule is applied:

            - values within `eps` of 0 are set to 0
            - values within `eps` of the upper boundary are wrapped to 0 and the
              corresponding lattice shift is incremented

        Notes:
            - This method is provided as an explicit helper.
            - Forward operations pre-wrap generators through a detached native
              snapshot. This numerical backend remap is not exact user wrapping.

        Args:
            points_internal: Points in the internal coordinate system,
                shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance in internal distance units. If None,
                defaults to 1e-12 * L where L = max(bx, by, bz).

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (na, nb, nc).
        """
        return_shifts_value = require_bool(
            return_shifts,
            name='return_shifts',
        )
        pts = coerce_point_array(
            points_internal,
            name='points_internal',
            dim=3,
        )
        explicit_eps = (
            None
            if eps is None
            else require_nonnegative_finite_real(eps, name='eps')
        )
        bx, bxy, by, bxz, byz, bz = self.to_internal_params()

        x = pts[:, 0].astype(float, copy=True)
        y = pts[:, 1].astype(float, copy=True)
        z = pts[:, 2].astype(float, copy=True)

        if explicit_eps is None:
            eps_val = _default_snap_eps(max(bx, by, bz))
        else:
            eps_val = explicit_eps

        na = np.zeros_like(x, dtype=np.int64)
        nb = np.zeros_like(x, dtype=np.int64)
        nc = np.zeros_like(x, dtype=np.int64)

        # Iterate several times: remap -> (optional) snap upper boundary.
        # In normal cases, this converges in 1 iteration. Extra iterations
        # handle points that land extremely close to periodic boundaries
        # and would otherwise flip images due to floating-point round-off.
        for _ in range(3):
            # Remap into [0,b) using lower-triangular lattice steps.
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                dc_quotient = z / bz
            dc = floor_to_int64(dc_quotient, name='points_internal c shift')
            with np.errstate(over='ignore', invalid='ignore'):
                z -= dc * bz
                y -= dc * byz
                x -= dc * bxz
            _require_finite_remap_coordinates(x, y, z)

            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                db_quotient = y / by
            db = floor_to_int64(db_quotient, name='points_internal b shift')
            with np.errstate(over='ignore', invalid='ignore'):
                y -= db * by
                x -= db * bxy
            _require_finite_remap_coordinates(x, y, z)

            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                da_quotient = x / bx
            da = floor_to_int64(da_quotient, name='points_internal a shift')
            with np.errstate(over='ignore', invalid='ignore'):
                x -= da * bx
            _require_finite_remap_coordinates(x, y, z)

            na = checked_int64_add(na, da, name='remap a shifts')
            nb = checked_int64_add(nb, db, name='remap b shifts')
            nc = checked_int64_add(nc, dc, name='remap c shifts')

            if eps_val == 0.0:
                break

            # Snap tiny values to 0 (does not change shifts).
            x[np.abs(x) < eps_val] = 0.0
            y[np.abs(y) < eps_val] = 0.0
            z[np.abs(z) < eps_val] = 0.0

            # Snap near upper boundaries to 0 with the corresponding shift increment.
            changed = False

            mz = z >= (bz - eps_val)
            if np.any(mz):
                z[mz] = 0.0
                y[mz] -= byz
                x[mz] -= bxz
                increment = np.zeros_like(nc)
                increment[mz] = 1
                nc = checked_int64_add(nc, increment, name='remap c shifts')
                changed = True

            my = y >= (by - eps_val)
            if np.any(my):
                y[my] = 0.0
                x[my] -= bxy
                increment = np.zeros_like(nb)
                increment[my] = 1
                nb = checked_int64_add(nb, increment, name='remap b shifts')
                changed = True

            mx = x >= (bx - eps_val)
            if np.any(mx):
                x[mx] = 0.0
                increment = np.zeros_like(na)
                increment[mx] = 1
                na = checked_int64_add(na, increment, name='remap a shifts')
                changed = True

            _require_finite_remap_coordinates(x, y, z)

            if not changed:
                break

        # Final remap to guarantee we are inside the primary cell after any snapping.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            dc_quotient = z / bz
        dc = floor_to_int64(dc_quotient, name='points_internal c shift')
        with np.errstate(over='ignore', invalid='ignore'):
            z -= dc * bz
            y -= dc * byz
            x -= dc * bxz
        _require_finite_remap_coordinates(x, y, z)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            db_quotient = y / by
        db = floor_to_int64(db_quotient, name='points_internal b shift')
        with np.errstate(over='ignore', invalid='ignore'):
            y -= db * by
            x -= db * bxy
        _require_finite_remap_coordinates(x, y, z)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            da_quotient = x / bx
        da = floor_to_int64(da_quotient, name='points_internal a shift')
        with np.errstate(over='ignore', invalid='ignore'):
            x -= da * bx
        _require_finite_remap_coordinates(x, y, z)

        na = checked_int64_add(na, da, name='remap a shifts')
        nb = checked_int64_add(nb, db, name='remap b shifts')
        nc = checked_int64_add(nc, dc, name='remap c shifts')

        # Snap tiny values to 0 again for cleanliness.
        if eps_val > 0.0:
            x[np.abs(x) < eps_val] = 0.0
            y[np.abs(y) < eps_val] = 0.0
            z[np.abs(z) < eps_val] = 0.0

        remapped = np.stack([x, y, z], axis=1)

        if not return_shifts_value:
            return remapped
        shifts = np.stack([na, nb, nc], axis=1).astype(np.int64)
        return remapped, shifts

    def wrap_internal(self, points_internal: np.ndarray) -> np.ndarray:
        """Alias for :meth:`remap_internal`.

        This name existed in early versions of pyvoro2 but previously used an
        incorrect independent modulo for sheared cells.
        """
        return self.remap_internal(points_internal, return_shifts=False)

    def remap_cart(
        self,
        points: np.ndarray,
        *,
        return_shifts: bool = False,
        eps: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Remap Cartesian points into the primary cell.

        This is a convenience wrapper around :meth:`cart_to_internal`,
        :meth:`remap_internal`, and :meth:`internal_to_cart`.

        Args:
            points: Cartesian coordinates, shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance passed through to :meth:`remap_internal`.
        """
        return_shifts_value = require_bool(
            return_shifts,
            name='return_shifts',
        )
        eps_value = (
            None
            if eps is None
            else require_nonnegative_finite_real(eps, name='eps')
        )
        pts_i = self.cart_to_internal(points)
        if return_shifts_value:
            pts_i2, shifts = self.remap_internal(
                pts_i,
                return_shifts=True,
                eps=eps_value,
            )
            return self.internal_to_cart(pts_i2), shifts
        pts_i2 = self.remap_internal(
            pts_i,
            return_shifts=False,
            eps=eps_value,
        )
        return self.internal_to_cart(pts_i2)


def _require_finite_remap_coordinates(*coordinates: np.ndarray) -> None:
    """Reject non-finite shear arithmetic before another floor operation."""

    if not all(np.all(np.isfinite(values)) for values in coordinates):
        raise ValueError('remapped points must contain only finite values')
