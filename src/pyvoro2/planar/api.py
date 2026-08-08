"""High-level 2D API for planar Voronoi and power tessellations."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import warnings

import numpy as np

from .._internal.cell_output import add_empty_cells_inplace, remap_ids_inplace
from .._internal.inputs import (
    coerce_id_array,
    coerce_native_block_parameters,
    coerce_nonnegative_scalar_or_vector,
    coerce_nonnegative_vector,
    coerce_point_array,
    require_internal_id_range,
    require_planar_ghost_site_id_range,
    require_query_index_range,
    validate_forward_mode,
    validate_duplicate_check_mode,
    validate_duplicate_options,
)
from .._internal.power_input import ResolvedPowerInput, resolve_power_input
from .._internal.validation import (
    CPP_INT_MAX,
    PY_SSIZE_T_MAX,
    require_bool,
    require_nonnegative_finite_real,
    require_nonnegative_index,
    require_optional_bool,
    require_optional_nonnegative_finite_real,
    require_positive_finite_real,
    require_positive_index,
    require_string_choice,
)
from ..result import TessellationResult, _build_tessellation_result
from .._internal.planar.domain_geometry import geometry2d
from .._internal.planar.edge_shifts import _add_periodic_edge_shifts_inplace
from .diagnostics import (
    TessellationDiagnostics,
    TessellationError,
    analyze_tessellation,
)
from .domains import Box, RectangularCell
from .duplicates import duplicate_check as _duplicate_check
from .normalize import normalize_edges, normalize_vertices

_core2d = None
_CORE2D_IMPORT_ERROR: BaseException | None = None


Domain2D = Box | RectangularCell


class _DefaultOutput(str):
    """String sentinel whose public signature representation is ``'result'``."""


_DEFAULT_OUTPUT = _DefaultOutput('result')


def _strip_internal_geometry_inplace(
    cells: list[dict[str, Any]],
    *,
    keep_vertices: bool,
    keep_adjacency: bool,
    keep_edges: bool,
    keep_edge_shifts: bool,
) -> None:
    """Drop internal geometry fields that were requested only for analysis.

    Periodic diagnostics may require temporary vertices/edges/edge shifts even
    when the caller only wants a lightweight high-level answer. This helper
    removes those internal extras before the final result is returned.
    """

    for cell in cells:
        if not keep_vertices:
            cell.pop('vertices', None)
        if not keep_adjacency:
            cell.pop('adjacency', None)
        if not keep_edges:
            cell.pop('edges', None)
            continue
        if not keep_edge_shifts:
            for edge in cell.get('edges') or []:
                edge.pop('adjacent_shift', None)


def _require_core2d():
    """Return the compiled 2D extension module or raise a helpful ImportError."""

    global _core2d, _CORE2D_IMPORT_ERROR
    if _core2d is None and _CORE2D_IMPORT_ERROR is None:
        try:
            from importlib import import_module

            _core2d = import_module('.._core2d', __package__)
        except Exception as exc:  # pragma: no cover
            _CORE2D_IMPORT_ERROR = exc
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


def _resolve_compute_output(
    *,
    output: object,
    normalize: object,
) -> tuple[
    Literal['result', 'cells'],
    Literal['none', 'vertices', 'topology'],
]:
    """Validate and resolve the planar output selector."""

    output_value = require_string_choice(
        output,
        name='output',
        choices=('result', 'cells'),
    )
    normalize_value = require_string_choice(
        normalize,
        name='normalize',
        choices=('none', 'vertices', 'topology'),
    )

    resolved: Literal['result', 'cells'] = (
        'result' if output_value == 'result' else 'cells'
    )

    if normalize_value != 'none':
        if resolved == 'cells':
            raise ValueError(
                'output="cells" cannot be combined with normalization; '
                'use output="result"'
            )
        resolved = 'result'

    return resolved, normalize_value  # type: ignore[return-value]


def _finish_compute_output(
    *,
    output: Literal['result', 'cells'],
    return_diagnostics: bool,
    domain: Domain2D,
    mode: Literal['standard', 'power'],
    sites: np.ndarray,
    ids: np.ndarray | None,
    cells: list[dict[str, Any]],
    power_input: ResolvedPowerInput,
    diagnostics: TessellationDiagnostics | None,
    normalized_vertices: object | None,
    normalized_topology: object | None,
    boundaries_available: bool,
    periodic_shifts_available: bool,
) -> (
    TessellationResult
    | list[dict[str, Any]]
    | tuple[list[dict[str, Any]], TessellationDiagnostics]
):
    """Return structured output or the explicit raw output shape."""

    if output == 'cells':
        if return_diagnostics:
            assert diagnostics is not None
            return cells, diagnostics
        return cells

    return _build_tessellation_result(
        dimension=2,
        domain=domain,
        mode=mode,
        sites=sites,
        ids=ids,
        cells=cells,
        power_input=power_input,
        tessellation_diagnostics=diagnostics,
        normalized_vertices=normalized_vertices,
        normalized_topology=normalized_topology,
        boundaries_available=boundaries_available,
        periodic_shifts_available=periodic_shifts_available,
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
    weights: Sequence[float] | np.ndarray | None = None,
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
    return_diagnostics: bool = False,
    output: Literal['result', 'cells'] = _DEFAULT_OUTPUT,
    normalize: Literal['none', 'vertices', 'topology'] = 'none',
    normalization_tol: float | None = None,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'none',
    tessellation_require_reciprocity: bool | None = None,
    tessellation_area_tol_rel: float = 1e-8,
    tessellation_area_tol_abs: float = 1e-12,
    tessellation_line_offset_tol: float | None = None,
    tessellation_line_angle_tol: float | None = None,
) -> (
    list[dict[str, Any]]
    | tuple[list[dict[str, Any]], TessellationDiagnostics]
    | TessellationResult
):
    """Compute planar Voronoi or power tessellation cells.

    Supported domains:
      - :class:`~pyvoro2.planar.domains.Box`
      - :class:`~pyvoro2.planar.domains.RectangularCell`

    By default, planar compute returns one
    :class:`~pyvoro2.TessellationResult`. Set ``output="cells"`` for the
    explicit raw cell list, or ``(cells, diagnostics)``
    when ``return_diagnostics=True``. Structured results always carry computed
    diagnostics inside ``result.tessellation_diagnostics`` and never return a
    tuple.

    Wrapper-level normalization convenience is also available via
    ``normalize='vertices'`` or ``'topology'``. Any request for normalized
    output returns a :class:`~pyvoro2.TessellationResult`.
    The normalized structures intentionally carry their own augmented cell
    copies, so the raw ``cells`` field can stay lightweight even when internal
    geometry was needed for diagnostics or normalization.

    For periodic domains, diagnostics and normalization automatically compute
    temporary edge shifts and the required edge/vertex geometry internally,
    even when those fields were not requested by the caller. Any such
    temporary fields are stripped from the raw returned cells unless they were
    explicitly requested.

    In ``mode='power'``, supply exactly one of ``weights`` or ``radii``.
    Mathematical weights follow the power convention
    ``||x - p_i||^2 - w_i`` and have squared-length units; positive, zero, and
    negative finite weights are valid when the common-shift conversion remains
    finite and representable. Non-finite input or overflow during conversion
    raises ``ValueError`` before native computation. Finite representability
    does not guarantee a numerically resolvable native tessellation. Voro++
    evaluates radical geometry with binary64 squared-radius arithmetic, so very
    large absolute ``radii**2`` values or genuine weight ranges relative to
    squared coordinate/domain scales can lose geometric resolution. There is no
    universal safe cutoff: the onset depends on scale, geometry, platform, and
    compiler, and periodic power tessellations are a particularly sensitive
    regime.
    pyvoro2 converts valid weights to non-negative backend radii with one common
    global shift, so adding the same constant to every weight does not change
    the diagram. Radii have length units and are a non-unique backend
    representation, not necessarily physical radii. Standard mode rejects both
    ``weights`` and ``radii`` because neither representation has meaning there.

    ``init_mem`` and explicit length-2 ``blocks`` must contain positive exact
    non-Boolean integers in the C++ ``int`` range. ``block_size``, when
    supplied, must be positive and finite. Points and radii are validated for
    shape and finiteness before construction, and the aggregate estimate of
    known eager native construction allocations may be at most exactly 1 GiB.
    """

    resolved_output, normalize = _resolve_compute_output(
        output=output,
        normalize=normalize,
    )
    mode = validate_forward_mode(mode)  # type: ignore[assignment]
    duplicate_check = validate_duplicate_check_mode(  # type: ignore[assignment]
        duplicate_check
    )
    tessellation_check = require_string_choice(  # type: ignore[assignment]
        tessellation_check,
        name='tessellation_check',
        choices=('none', 'diagnose', 'warn', 'raise'),
    )
    duplicate_threshold_value, duplicate_wrap_value, duplicate_max_pairs_value = (
        validate_duplicate_options(
            threshold=duplicate_threshold,
            wrap=duplicate_wrap,
            max_pairs=duplicate_max_pairs,
        )
    )
    edge_shift_search_value = require_nonnegative_index(
        edge_shift_search,
        name='edge_shift_search',
        maximum=PY_SSIZE_T_MAX,
    )
    user_return_vertices = require_bool(
        return_vertices,
        name='return_vertices',
    )
    user_return_adjacency = require_bool(
        return_adjacency,
        name='return_adjacency',
    )
    user_return_edges = require_bool(return_edges, name='return_edges')
    user_return_edge_shifts = require_bool(
        return_edge_shifts,
        name='return_edge_shifts',
    )
    include_empty_value = require_bool(include_empty, name='include_empty')
    validate_edge_shifts_value = require_bool(
        validate_edge_shifts,
        name='validate_edge_shifts',
    )
    repair_edge_shifts_value = require_bool(
        repair_edge_shifts,
        name='repair_edge_shifts',
    )
    return_diagnostics_value = require_bool(
        return_diagnostics,
        name='return_diagnostics',
    )
    tessellation_require_reciprocity_value = require_optional_bool(
        tessellation_require_reciprocity,
        name='tessellation_require_reciprocity',
    )
    edge_shift_tol_value = require_optional_nonnegative_finite_real(
        edge_shift_tol,
        name='edge_shift_tol',
    )
    normalization_tol_value = (
        None
        if normalization_tol is None
        else require_positive_finite_real(
            normalization_tol,
            name='normalization_tol',
        )
    )
    area_tol_rel_value = require_nonnegative_finite_real(
        tessellation_area_tol_rel,
        name='tessellation_area_tol_rel',
    )
    area_tol_abs_value = require_nonnegative_finite_real(
        tessellation_area_tol_abs,
        name='tessellation_area_tol_abs',
    )
    line_offset_tol_value = require_optional_nonnegative_finite_real(
        tessellation_line_offset_tol,
        name='tessellation_line_offset_tol',
    )
    line_angle_tol_value = require_optional_nonnegative_finite_real(
        tessellation_line_angle_tol,
        name='tessellation_line_angle_tol',
    )
    if repair_edge_shifts_value:
        validate_edge_shifts_value = True
    init_mem_value = require_positive_index(
        init_mem,
        name='init_mem',
        maximum=CPP_INT_MAX,
    )
    blocks_value, block_size_value = coerce_native_block_parameters(
        blocks=blocks,
        block_size=block_size,
        dim=2,
    )
    pts = coerce_point_array(points, name='points', dim=2)
    n = int(pts.shape[0])
    require_internal_id_range(n)
    power_input = resolve_power_input(
        mode=mode,
        weights=weights,
        radii=radii,
        n=n,
    )
    rr = power_input.backend_radii

    geom = geometry2d(domain)
    bounds = geom.native_bounds
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
    )

    periodic = bool(geom.has_any_periodic_axis)
    need_diag = return_diagnostics_value or tessellation_check != 'none'
    need_norm_vertices = normalize in ('vertices', 'topology')
    need_norm_topology = normalize == 'topology'

    need_periodic_diag_geometry = bool(need_diag and periodic)
    need_periodic_norm_geometry = bool(need_norm_vertices and periodic)

    internal_return_vertices = (
        user_return_vertices or need_periodic_diag_geometry or need_norm_vertices
    )
    internal_return_adjacency = user_return_adjacency
    internal_return_edges = (
        user_return_edges
        or need_periodic_diag_geometry
        or need_norm_topology
        or need_periodic_norm_geometry
    )
    internal_return_edge_shifts = (
        user_return_edge_shifts
        or need_periodic_diag_geometry
        or need_periodic_norm_geometry
    )

    if user_return_edge_shifts:
        if not periodic:
            raise ValueError(
                'return_edge_shifts is only supported for periodic domains '
                '(RectangularCell with any periodic axis)'
            )
        if not user_return_edges:
            raise ValueError('return_edge_shifts requires return_edges=True')
        if not user_return_vertices:
            raise ValueError('return_edge_shifts requires return_vertices=True')

    if internal_return_edge_shifts:
        if repair_edge_shifts_value:
            validate_edge_shifts_value = True

    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)

    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=duplicate_threshold_value,
            domain=domain,
            wrap=duplicate_wrap_value,
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=duplicate_max_pairs_value,
        )

    periodic_flags = geom.periodic_axes
    opts = (
        internal_return_vertices,
        internal_return_adjacency,
        internal_return_edges,
    )
    core = _require_core2d()

    if mode == 'standard':
        cells = core.compute_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
        )
    elif mode == 'power':
        assert rr is not None
        cells = core.compute_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
        )
        if include_empty_value:
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

    if internal_return_edge_shifts:
        _add_periodic_edge_shifts_inplace(
            cells,
            lattice_vectors=geom.lattice_vectors_cart,
            periodic_mask=geom.periodic_axes,
            mode=mode,
            radii=rr,
            search=edge_shift_search_value,
            tol=edge_shift_tol_value,
            validate=validate_edge_shifts_value,
            repair=repair_edge_shifts_value,
        )

    if ids_user is not None:
        remap_ids_inplace(cells, ids_user, boundary_key='edges')

    diag: TessellationDiagnostics | None = None
    if need_diag:
        expected = ids_user.tolist() if ids_user is not None else list(range(n))
        diag = analyze_tessellation(
            cells,
            domain,
            expected_ids=expected,
            mode=mode,
            area_tol_rel=area_tol_rel_value,
            area_tol_abs=area_tol_abs_value,
            check_reciprocity=bool(periodic),
            check_line_mismatch=bool(periodic),
            line_offset_tol=line_offset_tol_value,
            line_angle_tol=line_angle_tol_value,
            mark_edges=bool(periodic),
        )

        if tessellation_require_reciprocity_value is None:
            tessellation_require_reciprocity_value = bool(periodic) and mode in (
                'standard',
                'power',
            )

        if tessellation_check in ('warn', 'raise'):
            ok = bool(diag.ok_area) and (
                bool(diag.ok_reciprocity)
                if tessellation_require_reciprocity_value
                else True
            )
            if not ok:
                msg = (
                    f'tessellation_check failed (mode={mode!r}): '
                    f'area_ratio={diag.area_ratio:g}, '
                    f'orphan_edges={diag.n_edges_orphan}, '
                    f'mismatched_edges={diag.n_edges_mismatched}'
                )
                if tessellation_check == 'raise':
                    raise TessellationError(msg, diag)
                warnings.warn(msg, stacklevel=2)

    normalized_vertices = None
    normalized_topology = None
    if need_norm_vertices:
        normalized_vertices = normalize_vertices(
            cells,
            domain=domain,
            tol=normalization_tol_value,
            require_edge_shifts=True,
            copy_cells=True,
        )
        if need_norm_topology:
            normalized_topology = normalize_edges(
                normalized_vertices,
                domain=domain,
                tol=normalization_tol_value,
                copy_cells=False,
            )

    _strip_internal_geometry_inplace(
        cells,
        keep_vertices=user_return_vertices,
        keep_adjacency=user_return_adjacency,
        keep_edges=user_return_edges,
        keep_edge_shifts=user_return_edge_shifts,
    )

    return _finish_compute_output(
        output=resolved_output,
        return_diagnostics=return_diagnostics_value,
        domain=domain,
        mode=mode,
        sites=pts,
        ids=ids_user,
        cells=cells,
        power_input=power_input,
        diagnostics=diag,
        normalized_vertices=normalized_vertices,
        normalized_topology=normalized_topology,
        boundaries_available=user_return_edges,
        periodic_shifts_available=user_return_edge_shifts,
    )


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
    """Locate the owning generator for each planar query point.

    ``init_mem`` and explicit length-2 ``blocks`` use positive exact-integer
    semantics; ``block_size`` is positive and finite. Malformed or non-finite
    points, queries, radii, domains, and over-cap known eager native allocation
    estimates raise ``ValueError`` before construction.
    """

    mode = validate_forward_mode(mode)  # type: ignore[assignment]
    duplicate_check = validate_duplicate_check_mode(  # type: ignore[assignment]
        duplicate_check
    )
    duplicate_threshold_value, duplicate_wrap_value, duplicate_max_pairs_value = (
        validate_duplicate_options(
            threshold=duplicate_threshold,
            wrap=duplicate_wrap,
            max_pairs=duplicate_max_pairs,
        )
    )
    return_owner_position_value = require_bool(
        return_owner_position,
        name='return_owner_position',
    )
    init_mem_value = require_positive_index(
        init_mem,
        name='init_mem',
        maximum=CPP_INT_MAX,
    )
    blocks_value, block_size_value = coerce_native_block_parameters(
        blocks=blocks,
        block_size=block_size,
        dim=2,
    )
    pts = coerce_point_array(points, name='points', dim=2)
    q = coerce_point_array(queries, name='queries', dim=2)

    n = int(pts.shape[0])
    require_internal_id_range(n)
    rr: np.ndarray | None = None
    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        rr = coerce_nonnegative_vector(radii, name='radii', n=n)
    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)

    geom = geometry2d(domain)
    bounds = geom.native_bounds
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
    )

    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=duplicate_threshold_value,
            domain=domain,
            wrap=duplicate_wrap_value,
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=duplicate_max_pairs_value,
        )

    periodic_flags = geom.periodic_axes
    core = _require_core2d()

    if mode == 'standard':
        found, owner_id, owner_pos = core.locate_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            q,
        )
    elif mode == 'power':
        assert rr is not None
        found, owner_id, owner_pos = core.locate_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
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
    if return_owner_position_value:
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
    """Compute ghost Voronoi/Laguerre cells at planar query points.

    ``init_mem`` and explicit length-2 ``blocks`` use positive exact-integer
    semantics; ``block_size`` is positive and finite. Malformed or non-finite
    points, queries, radii, domains, and over-cap known eager native allocation
    estimates raise ``ValueError`` before construction.
    """

    mode = validate_forward_mode(mode)  # type: ignore[assignment]
    duplicate_check = validate_duplicate_check_mode(  # type: ignore[assignment]
        duplicate_check
    )
    duplicate_threshold_value, duplicate_wrap_value, duplicate_max_pairs_value = (
        validate_duplicate_options(
            threshold=duplicate_threshold,
            wrap=duplicate_wrap,
            max_pairs=duplicate_max_pairs,
        )
    )
    edge_shift_search_value = require_nonnegative_index(
        edge_shift_search,
        name='edge_shift_search',
        maximum=PY_SSIZE_T_MAX,
    )
    return_vertices_value = require_bool(
        return_vertices,
        name='return_vertices',
    )
    return_adjacency_value = require_bool(
        return_adjacency,
        name='return_adjacency',
    )
    return_edges_value = require_bool(return_edges, name='return_edges')
    return_edge_shifts_value = require_bool(
        return_edge_shifts,
        name='return_edge_shifts',
    )
    include_empty_value = require_bool(include_empty, name='include_empty')
    validate_edge_shifts_value = require_bool(
        validate_edge_shifts,
        name='validate_edge_shifts',
    )
    repair_edge_shifts_value = require_bool(
        repair_edge_shifts,
        name='repair_edge_shifts',
    )
    edge_shift_tol_value = require_optional_nonnegative_finite_real(
        edge_shift_tol,
        name='edge_shift_tol',
    )
    if repair_edge_shifts_value:
        validate_edge_shifts_value = True
    init_mem_value = require_positive_index(
        init_mem,
        name='init_mem',
        maximum=CPP_INT_MAX,
    )
    blocks_value, block_size_value = coerce_native_block_parameters(
        blocks=blocks,
        block_size=block_size,
        dim=2,
    )
    pts = coerce_point_array(points, name='points', dim=2)
    q = coerce_point_array(queries, name='queries', dim=2)

    n = int(pts.shape[0])
    m = int(q.shape[0])
    require_planar_ghost_site_id_range(n)
    require_query_index_range(m)

    rr: np.ndarray | None = None
    gr: np.ndarray | None = None
    if mode == 'power':
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
    ids_internal = np.arange(n, dtype=np.int32)
    ids_user = coerce_id_array(ids, n=n)

    geom = geometry2d(domain)
    bounds = geom.native_bounds
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
    )

    if return_edge_shifts_value:
        if not geom.has_any_periodic_axis:
            raise ValueError(
                'return_edge_shifts is only supported for periodic domains '
                '(RectangularCell with any periodic axis)'
            )
        if not return_edges_value:
            raise ValueError('return_edge_shifts requires return_edges=True')
        if not return_vertices_value:
            raise ValueError('return_edge_shifts requires return_vertices=True')

    if duplicate_check != 'off' and n > 1:
        _duplicate_check(
            pts,
            threshold=duplicate_threshold_value,
            domain=domain,
            wrap=duplicate_wrap_value,
            mode='warn' if duplicate_check == 'warn' else 'raise',
            max_pairs=duplicate_max_pairs_value,
        )

    periodic_flags = geom.periodic_axes
    opts = (
        return_vertices_value,
        return_adjacency_value,
        return_edges_value,
    )

    core = _require_core2d()
    if mode == 'standard':
        cells = core.ghost_box_standard(
            pts,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
            q,
        )
    elif mode == 'power':
        assert rr is not None
        assert gr is not None
        cells = core.ghost_box_power(
            pts,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
            q,
            gr,
        )
    else:
        raise ValueError(f'unknown mode: {mode}')

    if return_edge_shifts_value:
        _add_periodic_edge_shifts_inplace(
            cells,
            lattice_vectors=geom.lattice_vectors_cart,
            periodic_mask=geom.periodic_axes,
            mode=mode,
            radii=rr,
            site_positions=pts,
            ghost_radii=gr if mode == 'power' else None,
            search=edge_shift_search_value,
            tol=edge_shift_tol_value,
            validate=validate_edge_shifts_value,
            repair=repair_edge_shifts_value,
        )

    if not include_empty_value:
        cells = [cell for cell in cells if not bool(cell.get('empty', False))]

    if ids_user is not None:
        remap_ids_inplace(cells, ids_user, boundary_key='edges')
    return cells
