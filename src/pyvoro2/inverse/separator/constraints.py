"""Constraint parsing and geometric normalization for inverse power fitting."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Literal, Sequence

import numpy as np

from ..._internal.inputs import (
    coerce_finite_matrix,
    coerce_finite_vector,
    coerce_point_array,
    owned_readonly_array,
)
from ..._internal.validation import (
    INT64_MAX,
    INT64_MIN,
    require_bool,
    require_bool_mask,
    require_finite_real,
    require_index,
    require_index_array,
    require_nonnegative_index,
    require_string_choice,
    require_string_tuple,
)
from ..._internal.periodic_images import MinimumImageBatch
from ..._internal.spatial.domain_geometry import geometry3d
from ...domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ..._internal.planar.domain_geometry import geometry2d
from ...planar.domains import Box as Box2D, RectangularCell

ConstraintRow = (
    tuple[int | np.integer, int | np.integer, float]
    | tuple[int | np.integer, int | np.integer, float, Sequence[int]]
)
ConstraintInput = Sequence[ConstraintRow]
Domain3D = Box3D | OrthorhombicCell | PeriodicCell
Domain2D = Box2D | RectangularCell
DomainAny = Domain2D | Domain3D


def _plain_value(value: object) -> object:
    return value.item() if hasattr(value, 'item') else value


def _validated_ids_array(
    ids: Sequence[int | np.integer] | np.ndarray,
    n_points: int | None = None,
) -> np.ndarray:
    """Return validated non-negative unique integer external IDs."""

    try:
        ids_arr = np.asarray(ids, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError('ids must be a 1D sequence of integers') from exc
    if n_points is not None and ids_arr.shape != (n_points,):
        if ids_arr.ndim == 1:
            raise ValueError('ids must have length n_points')
        raise ValueError('ids must be a 1D sequence of length n_points')
    if n_points is None and ids_arr.ndim != 1:
        raise ValueError('ids must be a 1D sequence')
    if n_points is not None and ids_arr.size != n_points:
        raise ValueError('ids must have length n_points')
    values: list[int] = []
    for position, value in enumerate(ids_arr):
        try:
            values.append(require_index(value, name=f'ids[{position}]'))
        except ValueError:
            raise ValueError(
                f'ids[{position}] must be an integer '
                '(exact and non-Boolean)'
            ) from None
    if any(value < 0 for value in values):
        raise ValueError('ids must be non-negative')
    largest = max(values, default=0)
    if largest <= np.iinfo(np.int64).max:
        dtype: np.dtype | type = np.int64
    elif largest <= np.iinfo(np.uint64).max:
        dtype = np.uint64
    else:
        dtype = object
    owned = np.array(values, dtype=dtype, copy=True)
    if np.unique(owned).size != owned.size:
        raise ValueError('ids must be unique')
    owned.setflags(write=False)
    return owned


def _external_id_label(ids: np.ndarray, site_index: int) -> int:
    """Return an external ID for one internal site index."""

    if not 0 <= int(site_index) < int(ids.size):
        raise ValueError(f'ids do not cover site index {site_index}')
    return int(ids[int(site_index)])


def _readonly_index_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a read-only int64 array without lossy element conversion."""

    array = np.asarray(value, dtype=object)
    if array.ndim != 1:
        raise ValueError(f'{name} must have shape (m,)')
    owned = require_index_array(
        array,
        name=name,
        shape=array.shape,
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class SeparatorObservations:
    """Resolved pairwise separator observations.

    This object is the public boundary between downstream pair-selection logic
    and pyvoro2's inverse solver. Each row refers to a specific ordered pair
    ``(i, j, shift)`` where ``shift`` is the lattice image applied to site ``j``.
    """

    n_points: int
    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    measurement: Literal['fraction', 'position']
    distance: np.ndarray
    distance2: np.ndarray
    delta: np.ndarray
    target_fraction: np.ndarray
    target_position: np.ndarray
    input_index: np.ndarray
    explicit_shift: np.ndarray
    ids: np.ndarray | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        measurement = require_string_choice(
            self.measurement,
            name='measurement',
            choices=('fraction', 'position'),
        )
        warnings = require_string_tuple(self.warnings, name='warnings')
        object.__setattr__(self, 'measurement', measurement)
        object.__setattr__(self, 'warnings', warnings)
        n_points = require_nonnegative_index(
            self.n_points,
            name='SeparatorObservations.n_points',
            maximum=sys.maxsize,
        )
        object.__setattr__(self, 'n_points', n_points)

        i = _readonly_index_array(self.i, name='SeparatorObservations.i')
        j = _readonly_index_array(self.j, name='SeparatorObservations.j')
        m = int(i.shape[0])
        if j.shape != (m,):
            raise ValueError('SeparatorObservations.j must have shape (m,)')
        raw_shifts = np.asarray(self.shifts, dtype=object)
        if raw_shifts.ndim != 2 or raw_shifts.shape[0] != m:
            raise ValueError('SeparatorObservations.shifts must have shape (m, d)')
        shifts = require_index_array(
            raw_shifts,
            name='SeparatorObservations.shifts',
            shape=raw_shifts.shape,
            minimum=INT64_MIN,
            maximum=INT64_MAX,
        )
        shifts.setflags(write=False)

        target = owned_readonly_array(
            coerce_finite_vector(
                self.target,
                name='SeparatorObservations.target',
                n=m,
            ),
            dtype=np.float64,
        )
        confidence = owned_readonly_array(
            coerce_finite_vector(
                self.confidence,
                name='SeparatorObservations.confidence',
                n=m,
            ),
            dtype=np.float64,
        )
        distance = owned_readonly_array(
            coerce_finite_vector(
                self.distance,
                name='SeparatorObservations.distance',
                n=m,
            ),
            dtype=np.float64,
        )
        distance2 = owned_readonly_array(
            coerce_finite_vector(
                self.distance2,
                name='SeparatorObservations.distance2',
                n=m,
            ),
            dtype=np.float64,
        )
        delta = owned_readonly_array(
            coerce_finite_matrix(
                self.delta,
                name='SeparatorObservations.delta',
                shape=(m, raw_shifts.shape[1]),
            ),
            dtype=np.float64,
        )
        target_fraction = owned_readonly_array(
            coerce_finite_vector(
                self.target_fraction,
                name='SeparatorObservations.target_fraction',
                n=m,
            ),
            dtype=np.float64,
        )
        target_position = owned_readonly_array(
            coerce_finite_vector(
                self.target_position,
                name='SeparatorObservations.target_position',
                n=m,
            ),
            dtype=np.float64,
        )
        input_index_raw = np.asarray(self.input_index, dtype=object)
        if input_index_raw.shape != (m,):
            raise ValueError('SeparatorObservations.input_index must have shape (m,)')
        input_index = require_index_array(
            input_index_raw,
            name='SeparatorObservations.input_index',
            shape=(m,),
            minimum=0,
            maximum=INT64_MAX,
        )
        input_index.setflags(write=False)
        explicit_shift = require_bool_mask(
            self.explicit_shift,
            name='SeparatorObservations.explicit_shift',
            length=m,
        )

        object.__setattr__(self, 'i', i)
        object.__setattr__(self, 'j', j)
        object.__setattr__(self, 'shifts', shifts)
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(self, 'distance', distance)
        object.__setattr__(self, 'distance2', distance2)
        object.__setattr__(self, 'delta', delta)
        object.__setattr__(self, 'target_fraction', target_fraction)
        object.__setattr__(self, 'target_position', target_position)
        object.__setattr__(self, 'input_index', input_index)
        object.__setattr__(self, 'explicit_shift', explicit_shift)
        object.__setattr__(
            self,
            'ids',
            (
                None
                if self.ids is None
                else _validated_ids_array(self.ids, n_points)
            ),
        )
        if np.any(self.i < 0) or np.any(self.i >= n_points):
            raise ValueError(
                'SeparatorObservations.i contains a site index out of range'
            )
        if np.any(self.j < 0) or np.any(self.j >= n_points):
            raise ValueError(
                'SeparatorObservations.j contains a site index out of range'
            )
        if np.any(self.confidence < 0.0):
            raise ValueError('SeparatorObservations.confidence must be non-negative')
        if np.any(self.distance <= 0.0) or np.any(self.distance2 <= 0.0):
            raise ValueError(
                'SeparatorObservations distances must be strictly positive'
            )

    @property
    def n_constraints(self) -> int:
        return int(self.i.shape[0])

    @property
    def dim(self) -> int:
        return int(self.shifts.shape[1])

    def pair_labels(self, *, use_ids: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return the left/right pair labels as indices or external ids."""

        use_ids_value = require_bool(use_ids, name='use_ids')
        if use_ids_value:
            if self.ids is None:
                raise ValueError(
                    'use_ids=True requires ids on the resolved constraint set'
                )
            return self.ids[self.i].copy(), self.ids[self.j].copy()
        return self.i.copy(), self.j.copy()

    def to_records(self, *, use_ids: bool = False) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per constraint row."""

        use_ids_value = require_bool(use_ids, name='use_ids')
        left, right = self.pair_labels(use_ids=use_ids_value)
        rows: list[dict[str, object]] = []
        left_is_int = np.issubdtype(np.asarray(left).dtype, np.integer)
        right_is_int = np.issubdtype(np.asarray(right).dtype, np.integer)
        for k in range(self.n_constraints):
            site_i = int(left[k]) if left_is_int else _plain_value(left[k])
            site_j = int(right[k]) if right_is_int else _plain_value(right[k])
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': site_i,
                    'site_j': site_j,
                    'shift': tuple(int(v) for v in self.shifts[k]),
                    'target': float(self.target[k]),
                    'confidence': float(self.confidence[k]),
                    'measurement': self.measurement,
                    'distance': float(self.distance[k]),
                    'target_fraction': float(self.target_fraction[k]),
                    'target_position': float(self.target_position[k]),
                    'input_index': int(self.input_index[k]),
                    'explicit_shift': bool(self.explicit_shift[k]),
                }
            )
        return tuple(rows)

    def subset(self, mask: np.ndarray) -> SeparatorObservations:
        """Return a subset with row order preserved."""

        mask = require_bool_mask(
            mask,
            name='mask',
            length=self.n_constraints,
        )
        return SeparatorObservations(
            n_points=self.n_points,
            i=self.i[mask].copy(),
            j=self.j[mask].copy(),
            shifts=self.shifts[mask].copy(),
            target=self.target[mask].copy(),
            confidence=self.confidence[mask].copy(),
            measurement=self.measurement,
            distance=self.distance[mask].copy(),
            distance2=self.distance2[mask].copy(),
            delta=self.delta[mask].copy(),
            target_fraction=self.target_fraction[mask].copy(),
            target_position=self.target_position[mask].copy(),
            input_index=self.input_index[mask].copy(),
            explicit_shift=self.explicit_shift[mask].copy(),
            ids=None if self.ids is None else self.ids.copy(),
            warnings=self.warnings,
        )


def resolve_separator_observations(
    points: np.ndarray,
    constraints: ConstraintInput,
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: DomainAny | None = None,
    ids: Sequence[int | np.integer] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: Sequence[float] | None = None,
    allow_empty: bool = False,
) -> SeparatorObservations:
    """Parse and resolve pairwise separator observations.

    Args:
        points: Site coordinates with shape ``(n, d)`` where ``d`` is currently
            supported for planar (2D) and spatial (3D) workflows.
        constraints: Raw constraint tuples ``(i, j, value[, shift])``.
        measurement: Whether ``value`` is interpreted as a normalized fraction
            in ``[0, 1]`` or as an absolute position along the connector.
        domain: Optional non-periodic or periodic domain.
        ids: Unique non-negative integer external IDs aligned with ``points``.
            Python integers and NumPy integer scalars are accepted.
        index_mode: Interpret the first two tuple entries as internal indices or
            external IDs. Endpoint values must be integers in either mode;
            floats, numeric strings, and booleans are rejected.
        image: Shift resolution policy for tuples that do not specify a shift.
        image_search: Bounded incumbent-seeding hint for certified periodic
            nearest-image inference. It cannot change a successful result.
        confidence: Optional non-negative per-constraint weights.
        allow_empty: Allow zero constraints and return an empty resolved object.
    """

    measurement = require_string_choice(
        measurement,
        name='measurement',
        choices=('fraction', 'position'),
    )
    index_mode = require_string_choice(
        index_mode,
        name='index_mode',
        choices=('index', 'id'),
    )
    image = require_string_choice(
        image,
        name='image',
        choices=('nearest', 'given_only'),
    )
    image_search_value = require_nonnegative_index(
        image_search,
        name='image_search',
        maximum=sys.maxsize,
    )
    allow_empty_value = require_bool(allow_empty, name='allow_empty')

    raw_points = np.asarray(points, dtype=object)
    if raw_points.ndim != 2 or raw_points.shape[1] not in (2, 3):
        raise ValueError('points must have shape (n, d) with d in {2, 3}')
    pts = coerce_point_array(
        raw_points,
        name='points',
        dim=int(raw_points.shape[1]),
    )

    ids_arr = None if ids is None else _validated_ids_array(ids, int(pts.shape[0]))

    i_idx, j_idx, target, shifts, shift_given, warnings = _parse_constraints(
        constraints,
        n_points=pts.shape[0],
        ids=ids_arr,
        index_mode=index_mode,
        allow_empty=allow_empty_value,
        shift_dim=pts.shape[1],
    )

    target_arr = np.asarray(target, dtype=np.float64)
    m = int(i_idx.shape[0])
    if confidence is None:
        omega = np.ones(m, dtype=np.float64)
    else:
        omega = coerce_finite_vector(confidence, name='confidence', n=m)
        if np.any(omega < 0):
            raise ValueError('confidence must be non-negative')

    pts2 = _maybe_remap_points(pts, domain)
    shifts_used, warnings2, inferred_geometry = _resolve_constraint_shifts(
        pts2,
        i_idx,
        j_idx,
        shifts,
        shift_given,
        domain=domain,
        image=image,
        image_search=image_search_value,
    )
    warnings = warnings + warnings2

    if m == 0:
        zeros_i = np.zeros(0, dtype=np.int64)
        zeros_f = np.zeros(0, dtype=np.float64)
        zeros_s = np.zeros((0, pts.shape[1]), dtype=np.int64)
        zeros_b = np.zeros(0, dtype=bool)
        return SeparatorObservations(
            n_points=int(pts.shape[0]),
            i=zeros_i,
            j=zeros_i.copy(),
            shifts=zeros_s,
            target=zeros_f,
            confidence=zeros_f,
            measurement=measurement,
            distance=zeros_f,
            distance2=zeros_f,
            delta=np.zeros((0, pts.shape[1]), dtype=np.float64),
            target_fraction=zeros_f,
            target_position=zeros_f,
            input_index=zeros_i,
            explicit_shift=zeros_b,
            ids=ids_arr,
            warnings=warnings,
        )

    if inferred_geometry is None:
        shift_cart = shift_to_cart(shifts_used, domain)
        with np.errstate(all='ignore'):
            pj_star = pts2[j_idx] + shift_cart
        _require_finite_connector_geometry(
            pj_star,
            stage='endpoint translation',
        )
        with np.errstate(all='ignore'):
            delta = pj_star - pts2[i_idx]
    else:
        missing = ~shift_given
        delta = np.empty((m, pts.shape[1]), dtype=np.float64)
        if np.any(shift_given):
            explicit_shift_cart = shift_to_cart(
                shifts_used[shift_given],
                domain,
            )
            with np.errstate(all='ignore'):
                explicit_endpoint = (
                    pts2[j_idx[shift_given]] + explicit_shift_cart
                )
            _require_finite_connector_geometry(
                explicit_endpoint,
                stage='explicit endpoint translation',
            )
            with np.errstate(all='ignore'):
                delta[shift_given] = (
                    explicit_endpoint - pts2[i_idx[shift_given]]
                )
        delta[missing] = inferred_geometry.displacement
    _require_finite_connector_geometry(delta, stage='coordinate difference')

    with np.errstate(all='ignore'):
        d2 = np.einsum('mi,mi->m', delta, delta)
    if inferred_geometry is not None:
        d2[~shift_given] = inferred_geometry.distance_squared
    _require_finite_connector_geometry(d2, stage='squared distance')
    if np.any(d2 <= 0.0):
        raise ValueError(
            'some constraints have zero distance (coincident points/image)'
        )
    with np.errstate(all='ignore'):
        d = np.sqrt(d2)
    _require_finite_connector_geometry(d, stage='distance')

    if measurement == 'fraction':
        target_fraction = target_arr.copy()
        with np.errstate(all='ignore'):
            target_position = target_fraction * d
        _require_finite_connector_geometry(
            target_position,
            stage='fraction-to-position conversion',
        )
    else:
        target_position = target_arr.copy()
        with np.errstate(all='ignore'):
            target_fraction = target_position / d
        _require_finite_connector_geometry(
            target_fraction,
            stage='position-to-fraction conversion',
        )

    return SeparatorObservations(
        n_points=int(pts.shape[0]),
        i=np.asarray(i_idx, dtype=np.int64),
        j=np.asarray(j_idx, dtype=np.int64),
        shifts=np.asarray(shifts_used, dtype=np.int64),
        target=target_arr,
        confidence=omega,
        measurement=measurement,
        distance=np.asarray(d, dtype=np.float64),
        distance2=np.asarray(d2, dtype=np.float64),
        delta=np.asarray(delta, dtype=np.float64),
        target_fraction=np.asarray(target_fraction, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
        input_index=np.arange(m, dtype=np.int64),
        explicit_shift=np.asarray(shift_given, dtype=bool),
        ids=ids_arr,
        warnings=warnings,
    )


# ---------------------------- internal helpers ----------------------------


def _require_finite_connector_geometry(
    values: np.ndarray,
    *,
    stage: str,
) -> None:
    """Reject non-representable derived connector geometry without warnings."""

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f'derived separator connector {stage} must contain only finite '
            'values'
        )


def _parse_constraints(
    constraints: ConstraintInput,
    *,
    n_points: int,
    ids: np.ndarray | None,
    index_mode: Literal['index', 'id'],
    allow_empty: bool,
    shift_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse raw tuple/list constraints.

    Accepted forms:
        ``(i, j, value)``
        ``(i, j, value, shift)``
    """

    index_mode = require_string_choice(
        index_mode,
        name='index_mode',
        choices=('index', 'id'),
    )
    if index_mode == 'id':
        if ids is None:
            raise ValueError('ids must be provided when index_mode="id"')
        id_to_index = {
            _external_id_label(ids, k): k for k in range(int(ids.size))
        }
    else:
        id_to_index = None

    m = len(constraints)
    if m == 0 and not allow_empty:
        raise ValueError('constraints must be non-empty')

    i_idx = np.empty(m, dtype=np.int64)
    j_idx = np.empty(m, dtype=np.int64)
    val = np.empty(m, dtype=np.float64)
    shifts = np.zeros((m, shift_dim), dtype=np.int64)
    shift_given = np.zeros(m, dtype=bool)
    warnings: list[str] = []

    for k, c in enumerate(constraints):
        if not isinstance(c, (tuple, list)):
            raise ValueError(f'constraint {k} must be a tuple/list')
        if len(c) not in (3, 4):
            raise ValueError(
                f'constraint {k} must have length 3 or 4: (i, j, value[, shift])'
            )
        try:
            ii = require_index(c[0], name=f'constraint {k} endpoint i')
        except ValueError:
            raise ValueError(
                f'constraint {k} endpoint i must be an integer '
                '(exact and non-Boolean)'
            ) from None
        try:
            jj = require_index(c[1], name=f'constraint {k} endpoint j')
        except ValueError:
            raise ValueError(
                f'constraint {k} endpoint j must be an integer '
                '(exact and non-Boolean)'
            ) from None
        if id_to_index is not None:
            if ii not in id_to_index or jj not in id_to_index:
                raise ValueError(f'constraint {k} uses id not present in ids')
            ii = id_to_index[ii]
            jj = id_to_index[jj]
        if not (0 <= ii < n_points and 0 <= jj < n_points):
            raise ValueError(f'constraint {k} index out of range')
        if ii == jj:
            raise ValueError(f'constraint {k} has i == j (degenerate)')
        i_idx[k] = ii
        j_idx[k] = jj
        val[k] = require_finite_real(
            c[2],
            name=f'constraint {k} value',
        )

        if len(c) == 4:
            sh = c[3]
            if (
                not isinstance(sh, (tuple, list))
                or len(sh) != shift_dim
            ):
                raise ValueError(
                    f'constraint {k} shift must be a length-{shift_dim} tuple'
                )
            shifts[k] = tuple(
                require_index(
                    value,
                    name=f'constraint {k} shift[{axis}]',
                    minimum=INT64_MIN,
                    maximum=INT64_MAX,
                )
                for axis, value in enumerate(sh)
            )
            shift_given[k] = True

    return i_idx, j_idx, val, shifts, shift_given, tuple(warnings)


def maybe_remap_points(points: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    return _maybe_remap_points(points, domain)


def _geometry_for_dim(dim: int, domain: DomainAny | None):
    if dim == 2:
        if domain is not None and not isinstance(domain, (Box2D, RectangularCell)):
            raise ValueError(
                '2D points require domain=None or a planar domain '
                '(pyvoro2.planar.Box or RectangularCell)'
            )
        return geometry2d(domain)
    if dim == 3:
        if domain is not None and not isinstance(
            domain, (Box3D, OrthorhombicCell, PeriodicCell)
        ):
            raise ValueError(
                '3D points require domain=None or a 3D domain '
                '(Box, OrthorhombicCell, or PeriodicCell)'
            )
        return geometry3d(domain)
    raise ValueError('only 2D and 3D points are supported')


def _maybe_remap_points(points: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    raw = np.asarray(points, dtype=object)
    if raw.ndim != 2:
        raise ValueError('points must have shape (n, d)')
    pts = coerce_point_array(
        raw,
        name='points',
        dim=int(raw.shape[1]),
    )
    return _geometry_for_dim(int(pts.shape[1]), domain).remap_cart(pts)


def _resolve_constraint_shifts(
    points: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    shifts: np.ndarray,
    shift_given: np.ndarray,
    *,
    domain: DomainAny | None,
    image: Literal['nearest', 'given_only'],
    image_search: int,
) -> tuple[np.ndarray, tuple[str, ...], MinimumImageBatch | None]:
    """Return per-constraint integer shifts to apply to site j."""

    image = require_string_choice(
        image,
        name='image',
        choices=('nearest', 'given_only'),
    )
    m = i_idx.shape[0]
    warnings: list[str] = []
    dim = int(points.shape[1])
    geom = _geometry_for_dim(dim, domain)

    shifts = require_index_array(
        shifts,
        name='shifts',
        shape=(m, dim),
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )
    shift_given = require_bool_mask(
        shift_given,
        name='shift_given',
        length=m,
    )

    if not geom.has_any_periodic_axis:
        geom.validate_shifts(shifts[shift_given])
        return np.zeros((m, dim), dtype=np.int64), tuple(warnings), None

    shifts2 = shifts.copy()
    provided_mask = shift_given.copy()

    if image == 'given_only':
        if np.any(~provided_mask):
            raise ValueError('some constraints are missing shifts (image="given_only")')
        geom.validate_shifts(shifts2)
        return shifts2, tuple(warnings), None

    image_search = require_nonnegative_index(
        image_search,
        name='image_search',
        maximum=sys.maxsize,
    )

    missing = ~provided_mask
    inferred_geometry = None
    if np.any(missing):
        tie_orientation = np.where(
            i_idx[missing] < j_idx[missing],
            1,
            -1,
        ).astype(np.int8)
        inferred_geometry = geom.minimum_image_displacements(
            points[i_idx[missing]],
            points[j_idx[missing]],
            tie_orientation=tie_orientation,
            image_search=image_search,
        )
        shifts2[missing] = inferred_geometry.shift
        warnings.append(
            'some constraints did not specify shifts; using nearest-image shifts'
        )

    geom.validate_shifts(shifts2)
    return shifts2, tuple(warnings), inferred_geometry


def shift_to_cart(shifts: np.ndarray, domain: DomainAny | None) -> np.ndarray:
    raw = np.asarray(shifts, dtype=object)
    if raw.ndim != 2:
        raise ValueError('shifts must have shape (m, d)')
    sh = require_index_array(
        raw,
        name='shifts',
        shape=raw.shape,
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )
    return _geometry_for_dim(int(sh.shape[1]), domain).shift_to_cart(sh)
