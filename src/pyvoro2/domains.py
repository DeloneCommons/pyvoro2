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

from ._internal import exact_user_basis, native_remap
from ._internal.native_periodic_frame import prepare_native_periodic_frame
from ._internal.native_periodic_preparation import prepare_periodic_points
from ._internal.inputs import (
    coerce_finite_matrix,
    coerce_finite_vector,
    coerce_point_array,
)
from ._internal.validation import (
    require_bool,
    require_bool_tuple,
    require_finite_real,
    require_nonnegative_finite_real,
    require_ordered_bounds,
    require_positive_finite_real,
)


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
        The period is the represented binary64 ``max - min``. Quotients and
        affine subtraction use exact binary64 operands; only the final
        coordinate is rounded. A value reaching exact ``min + D`` advances the
        chart to min even with ``eps=0``. This chart endpoint can differ from
        max; an emitted value equal to max can instead require refusal.

        Args:
            points: Array of shape (n, 3).
            return_shifts: If True, also return integer shifts (nx, ny, nz)
                such that:

                    p_original ~= p_remapped + nx*a + ny*b + nz*c

                where a/b/c are the cell lattice vectors.
            eps: Optional snapping tolerance. If None, defaults to
                1e-12 * L where L is the maximum periodic axis length.
                Must be finite, nonnegative and satisfy exact ``2*eps < D``
                for every active represented period D, even for empty input.

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (nx, ny, nz).

        Raises:
            ValueError: If snapping neighborhoods touch/overlap, a numerical
                representative cannot satisfy the declared interval, or final
                shifts do not fit int64 (also when return_shifts is False).

        Notes:
            Outputs are owned float64 coordinates and optional int64 shifts.
            Repeating this remap with unchanged epsilon preserves coordinates
            and adds zero shifts. Reconstruction includes final rounding and
            intentional snapping error, with no universal absolute error bound.
        """
        pts, expose, epsilon = native_remap.validate_remap_inputs(
            points, dim=3, return_shifts=return_shifts, eps=eps)
        primary, shifts = native_remap.remap_rectangular(
            pts, self.bounds, self.periodic, epsilon)
        return native_remap.materialize_remap(primary, shifts, expose)


@dataclass(frozen=True, slots=True)
class PeriodicCell:
    """Fully periodic triclinic cell for 3D crystals.

    The supplied finite binary64 rows A define the user lattice exactly by
    x = origin + f @ A. Either handedness is accepted when det(A) is exactly
    nonzero; rows and origin are preserved. The four provisional user-coordinate
    methods make exact decisions and return rounded float64 views without Voro++.

    Native operations support either handedness through a qualified orthogonal
    QR frame, subject to separate numerical and native admission safeguards.
    The positive-diagonal Voro++ representation is:
        a = (bx, 0, 0)
        b = (bxy, by, 0)
        c = (bxz, byz, bz)

    Points are transformed into that coordinate system before tessellation;
    reflection-aware reporting preserves Cartesian face and adjacency winding.

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

        Exact dyadic affine arithmetic applies committed integer translations
        along c, then b, then a, finalizing each axis once. Each translation
        includes the complete row before the next axis is evaluated. Only the
        emitted coordinates are rounded to binary64, in the half-open domain:

            x in [0, bx), y in [0, by), z in [0, bz)

        A rounded remainder equal to its period emits zero and increments the
        coefficient, even with ``eps=0``. Otherwise, for positive epsilon,
        a rounded remainder strictly below eps snaps to zero; a remainder at
        most eps below the period snaps to zero and increments the coefficient.
        Comparisons use exact represented operands. No physical tangential
        remainder is discarded during coupled translations.

        Notes:
            - This method is provided as an explicit helper.
            - Forward operations pre-wrap generators through a detached native
              snapshot. This numerical backend remap is not exact user wrapping.
            - Repeating an accepted internal remap with unchanged epsilon
              preserves coordinates and adds zero shifts. This does not promise
              a fixed point after a Cartesian QR round trip.
            - ``p_input ~= p_remapped + shifts @ L`` includes final-coordinate
              rounding and intentional snapping, without a scale-independent
              reconstruction tolerance or exact source/native equivalence.

        Args:
            points_internal: Points in the internal coordinate system,
                shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance in internal distance units. If None,
                defaults to 1e-12 * L where L = max(bx, by, bz).
                Must be finite, nonnegative and satisfy exact ``2*eps < D``
                for each native diagonal D, including empty batches. Some
                admissible anisotropic frames therefore refuse the default;
                this helper permits a smaller explicit epsilon.

        Returns:
            If return_shifts is False:
                Remapped coordinates, shape (n, 3).
            If return_shifts is True:
                (remapped_points, shifts) where shifts has shape (n, 3)
                and contains integer (na, nb, nc).

        Raises:
            ValueError: If native admission fails, snap neighborhoods touch or
                overlap, or final shifts cannot fit int64. Intermediate
                coefficients stay exact and wide; final range is required even
                when return_shifts is False. Caller inputs are never mutated.
        """
        pts, expose, epsilon = native_remap.validate_remap_inputs(
            points_internal, dim=3, return_shifts=return_shifts, eps=eps,
            name='points_internal')
        primary, shifts = native_remap.remap_triangular(
            pts, self.to_internal_params(), epsilon)
        return native_remap.materialize_remap(primary, shifts, expose)

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

        Uses one qualified native frame. First compute the exact source lift
        ``k = floor((p-o) @ inverse(A))`` and local residual ``r = p-o-k@A``.
        Project the exact residual through the represented frame, rounding each
        complete dot once to nearest-even. Apply :meth:`remap_internal`'s chart
        to this local projection and combine its shift n with k before checking
        final int64 range, even when shifts are not requested.

        Exactly represented source-lattice translations preserve the native
        submission with the same frame and epsilon; their coefficients add to
        the nominal shift k+n. This cannot recover bits already lost by caller
        arithmetic. Reporting uses the existing native-to-Cartesian transform.
        The result need not equal a composition of separately rounded public
        converters or be a fixed point after a Cartesian round trip. It differs
        from exact user-parallelepiped :meth:`wrap_cart` and does not certify
        exact source/native equivalence. Native admission and safety remain.

        Args:
            points: Cartesian coordinates, shape (n, 3).
            return_shifts: If True, also return integer lattice shifts
                (na, nb, nc) applied to each point.
            eps: Snapping tolerance passed through to :meth:`remap_internal`.
        """
        pts, expose, epsilon = native_remap.validate_remap_inputs(
            points, dim=3, return_shifts=return_shifts, eps=eps)
        frame = prepare_native_periodic_frame(self.vectors)
        r = frame.rotation_to_internal
        origin = np.asarray(self.origin, dtype=float)
        primary, shifts = prepare_periodic_points(
            pts, self.vectors, self.origin, r, frame.params, epsilon)
        with np.errstate(over='ignore', invalid='ignore'):
            result = (r.T @ primary.T).T + origin[None, :]
        if not np.all(np.isfinite(result)):
            raise ValueError('points_internal produce non-finite coordinates')
        return (result, shifts) if expose else result
