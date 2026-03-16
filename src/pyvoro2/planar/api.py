"""High-level 2D API for planar Voronoi and power tessellations."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import warnings

import numpy as np

from .._cell_output import add_empty_cells_inplace, remap_ids_inplace
from .._inputs import (
    coerce_id_array,
    coerce_nonnegative_scalar_or_vector,
    coerce_nonnegative_vector,
    coerce_point_array,
    validate_duplicate_check_mode,
)
from ._domain_geometry import geometry2d
from ._edge_shifts2d import _add_periodic_edge_shifts_inplace
from .domains import Box, RectangularCell
from .duplicates import duplicate_check as _duplicate_check

try:
    from .. import _core2d  # type: ignore[attr-defined]

    _CORE2D_IMPORT_ERROR: BaseException | None = None
except Exception as _e:  # pragma: no cover
    _core2d = None  # type: ignore[assignment]
    _CORE2D_IMPORT_ERROR = _e


Domain2D = Box | RectangularCell


def _require_core2d():
    """Return the compiled 2D extension module or raise a helpful ImportError."""

    if _core2d is None:  # pragma: no cover
        raise ImportError(
            "pyvoro2 C++ extension module '_core2d' is not available. "
            'Install a prebuilt wheel with planar support or build from '
            'source to use pyvoro2.planar.compute/locate/ghost_cells.'
        ) from _CORE2D_IMPORT_ERROR
    return _core2d


def _warn_if_scale_suspicious(*, pts: np.ndarray, domain: Domain2D) -> None:
    """Warn if the planar coordinate scale is likely to be problematic."""

    if pts.size == 0:
        return

    geom = geometry2d(domain)
    (lx, ly), _area = geom._lengths_and_area()
    length_scale = max(float(lx), float(ly), 0.0)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        return

    if length_scale < 1e-3:
        warnings.warn(
            'The planar domain length scale appears very small '
            f'(L≈{length_scale:.3g}). Voro++ uses fixed absolute tolerances '
            '(~1e-5) and may terminate the process if points are too close in '
            'these units. Consider rescaling your coordinates before calling '
            'pyvoro2.planar.',
            RuntimeWarning,
            stacklevel=3,
        )
    elif length_scale > 1e9:
        warnings.warn(
            'The planar domain length scale appears very large '
            f'(L≈{length_scale:.3g}). Floating-point precision may be poor at '
            'this scale; consider rescaling your coordinates.',
            RuntimeWarning,
            stacklevel=3,
        )


def compute(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Domain2D,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_edges: bool = True,
    return_edge_shifts: bool = False,
    edge_shift_search: int = 2,
    include_empty: bool = False,
    validate_edge_shifts: bool = True,
    repair_edge_shifts: bool = False,
    edge_shift_tol: float | None = None,
) -> list[dict[str, Any]]:
    """Compute planar Voronoi or power tessellation cells.

    Supported domains:
      - :class:`~pyvoro2.planar.domains.Box`
      - :class:`~pyvoro2.planar.domains.RectangularCell`
    """

    pts = coerce_point_array(points, name='points', dim=2)
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    n = int(pts.shape[0])

    if int(edge_shift_search) < 0:
        raise ValueError('edge_shift_search must be >= 0')

    geom = geometry2d(domain)
    if return_edge_shifts:
        if not geom.has_any_periodic_axis:
            raise ValueError(
                'return_edge_shifts is only supported for periodic domains '
                '(RectangularCell with any periodic axis)'
            )
        if not return_edges:
            raise ValueError('return_edge_shifts requires return_edges=True')
        if not return_vertices:
            raise ValueError('return_edge_shifts requires return_vertices=True')

    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)
    core = _require_core2d()

    validate_duplicate_check_mode(duplicate_check)
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks,
        block_size=block_size,
    )
    bounds = geom.bounds
    periodic_flags = geom.periodic_axes
    opts = (bool(return_vertices), bool(return_adjacency), bool(return_edges))

    rr: np.ndarray | None = None
    if mode == 'standard':
        cells = core.compute_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            opts,
        )
    elif mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        rr = coerce_nonnegative_vector(radii, name='radii', n=n)
        cells = core.compute_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            opts,
        )
        if include_empty:
            add_empty_cells_inplace(
                cells,
                n=n,
                sites=pts,
                opts=opts,
                measure_key='area',
                boundary_key='edges',
            )
    else:
        raise ValueError(f'unknown mode: {mode}')

    if return_edge_shifts:
        _add_periodic_edge_shifts_inplace(
            cells,
            lattice_vectors=geom.lattice_vectors_cart,
            periodic_mask=geom.periodic_axes,
            mode=mode,
            radii=rr,
            search=int(edge_shift_search),
            tol=edge_shift_tol,
            validate=bool(validate_edge_shifts),
            repair=bool(repair_edge_shifts),
        )

    if ids_user is not None:
        remap_ids_inplace(cells, ids_user, boundary_key='edges')
    return cells


def locate(
    points: Sequence[Sequence[float]] | np.ndarray,
    queries: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Domain2D,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    return_owner_position: bool = False,
) -> dict[str, np.ndarray]:
    """Locate the owning generator for each planar query point."""

    pts = coerce_point_array(points, name='points', dim=2)
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    q = coerce_point_array(queries, name='queries', dim=2)

    n = int(pts.shape[0])
    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)
    core = _require_core2d()

    validate_duplicate_check_mode(duplicate_check)
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    geom = geometry2d(domain)
    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks,
        block_size=block_size,
    )
    bounds = geom.bounds
    periodic_flags = geom.periodic_axes

    if mode == 'standard':
        found, owner_id, owner_pos = core.locate_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            q,
        )
    elif mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        rr = coerce_nonnegative_vector(radii, name='radii', n=n)
        found, owner_id, owner_pos = core.locate_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            q,
        )
    else:
        raise ValueError(f'unknown mode: {mode}')

    owner_id = np.asarray(owner_id)
    found = np.asarray(found, dtype=bool)
    if ids_user is not None:
        out_ids = owner_id.astype(np.int64, copy=True)
        mask = out_ids >= 0
        if np.any(mask):
            out_ids[mask] = ids_user[out_ids[mask]]
        owner_id = out_ids

    out: dict[str, np.ndarray] = {
        'found': found,
        'owner_id': owner_id,
    }
    if return_owner_position:
        out['owner_pos'] = np.asarray(owner_pos, dtype=np.float64)
    return out


def ghost_cells(
    points: Sequence[Sequence[float]] | np.ndarray,
    queries: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Domain2D,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    ghost_radius: float | Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_edges: bool = True,
    return_edge_shifts: bool = False,
    edge_shift_search: int = 2,
    include_empty: bool = True,
    validate_edge_shifts: bool = True,
    repair_edge_shifts: bool = False,
    edge_shift_tol: float | None = None,
) -> list[dict[str, Any]]:
    """Compute ghost Voronoi/Laguerre cells at planar query points."""

    pts = coerce_point_array(points, name='points', dim=2)
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    q = coerce_point_array(queries, name='queries', dim=2)

    if int(edge_shift_search) < 0:
        raise ValueError('edge_shift_search must be >= 0')

    geom = geometry2d(domain)
    if return_edge_shifts:
        if not geom.has_any_periodic_axis:
            raise ValueError(
                'return_edge_shifts is only supported for periodic domains '
                '(RectangularCell with any periodic axis)'
            )
        if not return_edges:
            raise ValueError('return_edge_shifts requires return_edges=True')
        if not return_vertices:
            raise ValueError('return_edge_shifts requires return_vertices=True')

    n = int(pts.shape[0])
    m = int(q.shape[0])
    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)
    core = _require_core2d()

    validate_duplicate_check_mode(duplicate_check)
    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=float(duplicate_threshold),
            domain=domain,
            wrap=bool(duplicate_wrap),
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=int(duplicate_max_pairs),
        )

    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks,
        block_size=block_size,
    )
    bounds = geom.bounds
    periodic_flags = geom.periodic_axes
    opts = (bool(return_vertices), bool(return_adjacency), bool(return_edges))

    rr: np.ndarray | None = None
    if mode == 'standard':
        cells = core.ghost_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            opts,
            q,
        )
    elif mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        if ghost_radius is None:
            raise ValueError('ghost_radius is required for mode="power"')
        rr = coerce_nonnegative_vector(radii, name='radii', n=n)
        gr = coerce_nonnegative_scalar_or_vector(
            ghost_radius,
            name='ghost_radius',
            n=m,
            length_name='m',
        )
        cells = core.ghost_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            int(init_mem),
            opts,
            q,
            gr,
        )
    else:
        raise ValueError(f'unknown mode: {mode}')

    if return_edge_shifts:
        _add_periodic_edge_shifts_inplace(
            cells,
            lattice_vectors=geom.lattice_vectors_cart,
            periodic_mask=geom.periodic_axes,
            mode=mode,
            radii=rr,
            search=int(edge_shift_search),
            tol=edge_shift_tol,
            validate=bool(validate_edge_shifts),
            repair=bool(repair_edge_shifts),
        )

    if not include_empty:
        cells = [cell for cell in cells if not bool(cell.get('empty', False))]

    if ids_user is not None:
        remap_ids_inplace(cells, ids_user, boundary_key='edges')
    return cells
