"""Central private preparation boundary for every native generator insertion."""

from __future__ import annotations

from dataclasses import dataclass, replace
import warnings

import numpy as np

from ..duplicates import DuplicateError, DuplicatePair
from .duplicate_scanning import (
    PairScan,
    scan_close_pairs,
    scan_cross_close_pairs,
)
from .inputs import (
    coerce_external_id_array,
    coerce_nonnegative_vector,
    coerce_point_array,
    owned_readonly_array,
    require_internal_id_range,
    require_planar_ghost_site_id_range,
    require_query_index_range,
    validate_duplicate_check_mode,
    validate_duplicate_options,
)


BACKEND_SAFETY_DISTANCE_SQUARED = 1e-10
BACKEND_SAFETY_DISTANCE = 1e-5


@dataclass(frozen=True, slots=True)
class PreparedGenerators:
    """Owned immutable data for one native-facing generator role."""

    input_points_cart: np.ndarray
    primary_points_cart: np.ndarray
    native_points: np.ndarray
    remap_shifts: np.ndarray
    internal_ids: np.ndarray
    external_ids: np.ndarray
    backend_radii: np.ndarray | None
    periodic_axes: tuple[bool, ...]
    domain_kind: str
    operation: str

    def with_backend_radii(
        self,
        radii: np.ndarray | None,
    ) -> 'PreparedGenerators':
        """Attach already resolved backend radii without losing ownership."""

        if radii is None:
            return replace(self, backend_radii=None)
        values = coerce_nonnegative_vector(
            radii,
            name='backend_radii',
            n=self.input_points_cart.shape[0],
        )
        return replace(
            self,
            backend_radii=owned_readonly_array(values, dtype=np.float64),
        )


def _prepare_coordinates(
    points: object,
    *,
    geometry,
    periodic_snapshot,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dim = int(geometry.dim)
    input_points = coerce_point_array(points, name=name, dim=dim)
    n = int(input_points.shape[0])
    shifts = np.zeros((n, dim), dtype=np.int64)

    if getattr(geometry, 'is_triclinic', False):
        if periodic_snapshot is None:
            raise ValueError(
                'triclinic generator preparation requires one periodic snapshot'
            )
        with np.errstate(over='ignore', invalid='ignore'):
            internal = periodic_snapshot.cart_to_internal(input_points)
        internal = coerce_point_array(internal, name=name, dim=dim)
        primary_internal, shifts = periodic_snapshot.remap_internal(
            internal,
            return_shifts=True,
        )
        native_points = coerce_point_array(
            primary_internal,
            name=f'primary {name}',
            dim=dim,
        )
        primary_cart = periodic_snapshot.internal_to_cart(native_points)
        primary_cart = coerce_point_array(
            primary_cart,
            name=f'primary Cartesian {name}',
            dim=dim,
        )
        return input_points, primary_cart, native_points, shifts

    if geometry.has_any_periodic_axis:
        primary_cart, shifts = geometry.domain.remap_cart(
            input_points,
            return_shifts=True,
        )
        primary_cart = coerce_point_array(
            primary_cart,
            name=f'primary {name}',
            dim=dim,
        )
    else:
        primary_cart = np.array(input_points, copy=True, order='C')
    return input_points, primary_cart, primary_cart, shifts


def _require_contained(
    points: np.ndarray,
    *,
    geometry,
    external_ids: np.ndarray,
    operation: str,
    role: str,
    periodic_snapshot=None,
) -> None:
    if getattr(geometry, 'is_triclinic', False):
        if periodic_snapshot is None:
            raise ValueError(
                'triclinic containment requires the operation periodic snapshot'
            )
        bx, _bxy, by, _bxz, _byz, bz = periodic_snapshot.params
        bounds = ((0.0, bx), (0.0, by), (0.0, bz))
        checked_axes = range(3)
    else:
        bounds = geometry.native_bounds
        checked_axes = (
            axis
            for axis, periodic in enumerate(geometry.periodic_axes)
            if not periodic
        )

    for axis in checked_axes:
        lo, hi = (float(value) for value in bounds[axis])
        invalid = np.flatnonzero(
            (points[:, axis] < lo) | (points[:, axis] >= hi)
        )
        if invalid.size:
            index = int(invalid[0])
            value = float(points[index, axis])
            external_id = int(external_ids[index])
            raise ValueError(
                f'{operation} {role} index {index} (external ID '
                f'{external_id}) is outside the native primary domain on '
                f'axis {axis}: value={value!r}, required interval '
                f'[{lo!r}, {hi!r})'
            )


def _public_pairs(scan: PairScan) -> tuple[DuplicatePair, ...]:
    return tuple(
        DuplicatePair(i=int(i), j=int(j), distance=float(distance))
        for i, j, distance in scan.pairs
    )


def _duplicate_message(
    *,
    scan: PairScan,
    kind: str,
    threshold: float,
    operation: str,
) -> str:
    count_text = (
        f'at least {len(scan.pairs)}; showing {len(scan.pairs)}'
        if scan.truncated
        else str(len(scan.pairs))
    )
    if kind == 'backend_safety':
        return (
            f'Found {count_text} backend-unsafe generator pair(s) during '
            f'{operation} at or below safety distance '
            f'{BACKEND_SAFETY_DISTANCE:g} (squared distance '
            f'{BACKEND_SAFETY_DISTANCE_SQUARED:g}). Native insertion was '
            'not attempted.'
        )
    return (
        f'Found {count_text} safe generator pair(s) closer than user '
        f'threshold={threshold:g} during {operation}.'
    )


def _raise_duplicate_error(
    *,
    scan: PairScan,
    kind: str,
    threshold: float,
    user_threshold: float,
    optional_wrap_used: bool,
    operation: str,
    external_id_pairs: tuple[tuple[int, int], ...],
) -> None:
    message = _duplicate_message(
        scan=scan,
        kind=kind,
        threshold=threshold,
        operation=operation,
    )
    raise DuplicateError(
        message,
        _public_pairs(scan),
        threshold,
        kind=kind,
        safety_distance_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        safety_distance=BACKEND_SAFETY_DISTANCE,
        user_threshold=user_threshold,
        minimum_image_used=scan.minimum_image_used,
        optional_wrap_used=optional_wrap_used,
        truncated=scan.truncated,
        operation=operation,
        external_ids=external_id_pairs,
    )


def _scan_persistent_policy(
    prepared: PreparedGenerators,
    *,
    geometry,
    duplicate_check: str,
    duplicate_threshold: float,
    duplicate_wrap: bool,
    duplicate_max_pairs: int,
) -> None:
    periodic_geometry = geometry if geometry.has_any_periodic_axis else None
    mandatory = scan_close_pairs(
        prepared.primary_points_cart,
        radius=BACKEND_SAFETY_DISTANCE,
        geometry=periodic_geometry,
        inclusive_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        max_pairs=duplicate_max_pairs,
    )
    if mandatory.pairs:
        external = tuple(
            (
                int(prepared.external_ids[i]),
                int(prepared.external_ids[j]),
            )
            for i, j, _distance in mandatory.pairs
        )
        _raise_duplicate_error(
            scan=mandatory,
            kind='backend_safety',
            threshold=BACKEND_SAFETY_DISTANCE,
            user_threshold=duplicate_threshold,
            optional_wrap_used=False,
            operation=prepared.operation,
            external_id_pairs=external,
        )

    if duplicate_check == 'off' or duplicate_threshold <= BACKEND_SAFETY_DISTANCE:
        return
    optional_geometry = periodic_geometry if duplicate_wrap else None
    optional_points = (
        prepared.primary_points_cart
        if optional_geometry is not None
        else prepared.input_points_cart
    )
    optional = scan_close_pairs(
        optional_points,
        radius=duplicate_threshold,
        geometry=optional_geometry,
        max_pairs=duplicate_max_pairs,
    )
    if not optional.pairs:
        return
    external = tuple(
        (
            int(prepared.external_ids[i]),
            int(prepared.external_ids[j]),
        )
        for i, j, _distance in optional.pairs
    )
    message = _duplicate_message(
        scan=optional,
        kind='user_threshold',
        threshold=duplicate_threshold,
        operation=prepared.operation,
    )
    if duplicate_check == 'warn':
        warnings.warn(message, RuntimeWarning, stacklevel=4)
        return
    _raise_duplicate_error(
        scan=optional,
        kind='user_threshold',
        threshold=duplicate_threshold,
        user_threshold=duplicate_threshold,
        optional_wrap_used=bool(optional_geometry is not None),
        operation=prepared.operation,
        external_id_pairs=external,
    )


def prepare_generators(
    points: object,
    *,
    geometry,
    operation: str,
    external_ids: object | None,
    backend_radii: object | None,
    duplicate_check: object,
    duplicate_threshold: object,
    duplicate_wrap: object,
    duplicate_max_pairs: object,
    periodic_snapshot=None,
    reserve_ghost_id: bool = False,
) -> PreparedGenerators:
    """Validate, remap, contain, de-duplicate, and assign native IDs."""

    mode = validate_duplicate_check_mode(duplicate_check)
    threshold, wrap, max_pairs = validate_duplicate_options(
        threshold=duplicate_threshold,
        wrap=duplicate_wrap,
        max_pairs=duplicate_max_pairs,
    )
    input_points, primary_cart, native_points, shifts = _prepare_coordinates(
        points,
        geometry=geometry,
        periodic_snapshot=periodic_snapshot,
        name='points',
    )
    n = int(input_points.shape[0])
    if reserve_ghost_id and int(geometry.dim) == 2:
        require_planar_ghost_site_id_range(n)
    else:
        require_internal_id_range(n)
    supplied_external_ids = None
    if external_ids is not None:
        supplied_external_ids = coerce_external_id_array(
            external_ids,
            name='ids',
            n=n,
        )
    external = (
        np.arange(n, dtype=np.int64)
        if supplied_external_ids is None
        else supplied_external_ids
    )
    radii = None
    if backend_radii is not None:
        radii = coerce_nonnegative_vector(
            backend_radii,
            name='backend_radii',
            n=n,
        )

    _require_contained(
        native_points,
        geometry=geometry,
        external_ids=external,
        operation=operation,
        role='generator',
        periodic_snapshot=periodic_snapshot,
    )
    prepared = PreparedGenerators(
        input_points_cart=owned_readonly_array(input_points, dtype=np.float64),
        primary_points_cart=owned_readonly_array(primary_cart, dtype=np.float64),
        native_points=owned_readonly_array(native_points, dtype=np.float64),
        remap_shifts=owned_readonly_array(shifts, dtype=np.int64),
        internal_ids=owned_readonly_array(
            np.arange(n, dtype=np.int32),
            dtype=np.int32,
        ),
        external_ids=owned_readonly_array(external, dtype=np.int64),
        backend_radii=(
            None
            if radii is None
            else owned_readonly_array(radii, dtype=np.float64)
        ),
        periodic_axes=tuple(bool(value) for value in geometry.periodic_axes),
        domain_kind=str(geometry.kind),
        operation=str(operation),
    )
    _scan_persistent_policy(
        prepared,
        geometry=geometry,
        duplicate_check=mode,
        duplicate_threshold=threshold,
        duplicate_wrap=wrap,
        duplicate_max_pairs=max_pairs,
    )
    return prepared


def prepare_temporary_generators(
    points: object,
    *,
    persistent: PreparedGenerators,
    geometry,
    backend_radii: object | None,
    duplicate_check: object,
    duplicate_threshold: object,
    duplicate_wrap: object,
    duplicate_max_pairs: object,
    periodic_snapshot=None,
) -> PreparedGenerators:
    """Prepare ghost generators and compare each only with persistent sites."""

    mode = validate_duplicate_check_mode(duplicate_check)
    threshold, wrap, max_pairs = validate_duplicate_options(
        threshold=duplicate_threshold,
        wrap=duplicate_wrap,
        max_pairs=duplicate_max_pairs,
    )
    input_points, primary_cart, native_points, shifts = _prepare_coordinates(
        points,
        geometry=geometry,
        periodic_snapshot=periodic_snapshot,
        name='queries',
    )
    m = int(input_points.shape[0])
    require_query_index_range(m)
    external = np.arange(m, dtype=np.int64)
    radii = None
    if backend_radii is not None:
        radii = coerce_nonnegative_vector(
            backend_radii,
            name='ghost_radius',
            n=m,
        )
    _require_contained(
        native_points,
        geometry=geometry,
        external_ids=external,
        operation='ghost_cells',
        role='temporary ghost generator',
        periodic_snapshot=periodic_snapshot,
    )
    temporary = PreparedGenerators(
        input_points_cart=owned_readonly_array(input_points, dtype=np.float64),
        primary_points_cart=owned_readonly_array(primary_cart, dtype=np.float64),
        native_points=owned_readonly_array(native_points, dtype=np.float64),
        remap_shifts=owned_readonly_array(shifts, dtype=np.int64),
        internal_ids=owned_readonly_array(external, dtype=np.int64),
        external_ids=owned_readonly_array(external, dtype=np.int64),
        backend_radii=(
            None
            if radii is None
            else owned_readonly_array(radii, dtype=np.float64)
        ),
        periodic_axes=persistent.periodic_axes,
        domain_kind=persistent.domain_kind,
        operation='ghost_cells',
    )

    periodic_geometry = geometry if geometry.has_any_periodic_axis else None
    mandatory = scan_cross_close_pairs(
        persistent.primary_points_cart,
        temporary.primary_points_cart,
        radius=BACKEND_SAFETY_DISTANCE,
        geometry=periodic_geometry,
        inclusive_squared=BACKEND_SAFETY_DISTANCE_SQUARED,
        max_pairs=max_pairs,
    )
    if mandatory.pairs:
        external_pairs = tuple(
            (
                int(persistent.external_ids[i]),
                int(temporary.external_ids[j]),
            )
            for i, j, _distance in mandatory.pairs
        )
        _raise_duplicate_error(
            scan=mandatory,
            kind='backend_safety',
            threshold=BACKEND_SAFETY_DISTANCE,
            user_threshold=threshold,
            optional_wrap_used=False,
            operation='ghost_cells',
            external_id_pairs=external_pairs,
        )

    if mode == 'off' or threshold <= BACKEND_SAFETY_DISTANCE:
        return temporary
    optional_geometry = periodic_geometry if wrap else None
    reference_points = (
        persistent.primary_points_cart
        if optional_geometry is not None
        else persistent.input_points_cart
    )
    query_points = (
        temporary.primary_points_cart
        if optional_geometry is not None
        else temporary.input_points_cart
    )
    optional = scan_cross_close_pairs(
        reference_points,
        query_points,
        radius=threshold,
        geometry=optional_geometry,
        max_pairs=max_pairs,
    )
    if not optional.pairs:
        return temporary
    external_pairs = tuple(
        (
            int(persistent.external_ids[i]),
            int(temporary.external_ids[j]),
        )
        for i, j, _distance in optional.pairs
    )
    message = _duplicate_message(
        scan=optional,
        kind='user_threshold',
        threshold=threshold,
        operation='ghost_cells',
    )
    if mode == 'warn':
        warnings.warn(message, RuntimeWarning, stacklevel=4)
        return temporary
    _raise_duplicate_error(
        scan=optional,
        kind='user_threshold',
        threshold=threshold,
        user_threshold=threshold,
        optional_wrap_used=bool(optional_geometry is not None),
        operation='ghost_cells',
        external_id_pairs=external_pairs,
    )
    return temporary  # pragma: no cover


def validate_compute_internal_ids(
    cells: object,
    *,
    n: int,
    mode: str,
) -> None:
    """Enforce the standard/power raw native ID postconditions."""

    try:
        rows = list(cells)  # type: ignore[arg-type]
    except TypeError:
        raise RuntimeError(
            'native compute output must be an iterable of cells'
        ) from None
    seen: set[int] = set()
    for position, cell in enumerate(rows):
        if not isinstance(cell, dict) or 'id' not in cell:
            raise RuntimeError(
                f'native compute output cell {position} has no internal ID'
            )
        value = cell['id']
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise RuntimeError(
                f'native compute output cell {position} has a malformed '
                'internal ID'
            )
        internal_id = int(value)
        if not 0 <= internal_id < n:
            raise RuntimeError(
                f'native compute output cell {position} has out-of-range '
                f'internal ID {internal_id}'
            )
        if internal_id in seen:
            raise RuntimeError(
                f'native compute output contains duplicate internal ID '
                f'{internal_id}'
            )
        seen.add(internal_id)
    if mode == 'standard' and seen != set(range(n)):
        missing = sorted(set(range(n)) - seen)
        raise RuntimeError(
            'native standard compute output must contain every internal ID '
            f'exactly once; missing={missing[:10]}'
        )
