"""High-level API for computing Voronoi tessellations."""

from __future__ import annotations

from typing import Any, Sequence, Literal

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.spatial.domain_utils import domain_length_scale
from ._internal.inputs import (
    coerce_native_block_parameters,
    coerce_nonnegative_scalar_or_vector,
    coerce_nonnegative_vector,
    coerce_point_array,
    validate_forward_mode,
    validate_duplicate_check_mode,
    validate_duplicate_options,
)
from ._internal.generator_preparation import (
    prepare_generators,
    prepare_temporary_generators,
    validate_compute_internal_ids,
)
from ._internal.spatial.domain_geometry import geometry3d
from ._internal.spatial.face_shifts import _add_periodic_face_shifts_inplace
from ._internal.power_input import ResolvedPowerInput, resolve_power_input
from ._internal.validation import (
    CPP_INT_MAX,
    PY_SSIZE_T_MAX,
    require_bool,
    require_nonnegative_finite_real,
    require_nonnegative_index,
    require_optional_bool,
    require_optional_nonnegative_finite_real,
    require_positive_index,
    require_string_choice,
)
from .diagnostics import (
    TessellationDiagnostics,
    TessellationError,
    analyze_tessellation,
)
from .result import TessellationResult, _build_tessellation_result

# The compiled C++ extension is loaded only when a geometry operation needs it.
# Documentation builds and inverse-only imports therefore work without a
# compiled wheel and do not pay the native-import cost.
_core = None
_CORE_IMPORT_ERROR: BaseException | None = None


def _require_core():
    """Return the compiled extension module or raise a helpful ImportError."""
    global _core, _CORE_IMPORT_ERROR
    if _core is None and _CORE_IMPORT_ERROR is None:
        try:
            from importlib import import_module

            _core = import_module('._core', __package__)
        except Exception as exc:  # pragma: no cover
            _CORE_IMPORT_ERROR = exc
    if _core is None:  # pragma: no cover
        raise ImportError(
            'pyvoro2 C++ extension module \'_core\' is not available. '
            'Install a prebuilt wheel or build from source to use '
            'compute/locate/ghost_cells.'
        ) from _CORE_IMPORT_ERROR
    return _core


def _warn_if_scale_suspicious(*, pts: np.ndarray, length_scale: float) -> None:
    """Warn if the coordinate scale is likely to be numerically problematic.

    Voro++ uses fixed absolute tolerances internally. pyvoro2 rejects every
    generator pair in the fixed backend-safety regime before insertion, but an
    extremely small or large coordinate system can still lose geometric
    accuracy.

    pyvoro2 intentionally does **not** rescale user inputs automatically.
    Instead we emit a warning to encourage explicit rescaling by the caller.
    """

    try:
        L = float(length_scale)
    except Exception:
        return
    if not np.isfinite(L) or L <= 0:
        return

    # Heuristic thresholds: conservative enough to avoid noisy warnings for
    # typical coordinate systems (~1..1e3), but still highlight the most common
    # failure mode (very small unit systems, e.g. SI meters for atomistic data).
    if L < 1e-3:
        warnings.warn(
            'The domain length scale appears very small (L≈{:.3g}). '
            'pyvoro2 reserves distances through 1e-5 for backend safety, and '
            'Voro++ uses other fixed absolute tolerances. Consider '
            'rescaling your coordinates (e.g. multiply by a constant) before '
            'calling pyvoro2.'.format(L),
            RuntimeWarning,
            stacklevel=3,
        )
    elif L > 1e9:
        warnings.warn(
            'The domain length scale appears very large (L≈{:.3g}). '
            'Floating-point precision may be poor at this scale; consider '
            'rescaling your coordinates.'.format(L),
            RuntimeWarning,
            stacklevel=3,
        )


def _remap_ids_inplace(cells: list[dict[str, Any]], ids_user: np.ndarray) -> None:
    """Remap internal IDs (0..n-1) to user IDs in-place."""
    for c in cells:
        pid = int(c.get('id', -1))
        if 0 <= pid < ids_user.size:
            c['id'] = int(ids_user[pid])

        faces = c.get('faces')
        if faces is None:
            continue

        for f in faces:
            adj = int(f.get('adjacent_cell', -999999))
            # In Voro++, negative neighbor IDs can encode walls; keep them unchanged.
            if 0 <= adj < ids_user.size:
                f['adjacent_cell'] = int(ids_user[adj])


def _add_empty_cells_inplace(
    cells: list[dict[str, Any]],
    *,
    n: int,
    sites: np.ndarray,
    opts: tuple[bool, bool, bool],
) -> None:
    """Insert explicit empty-cell records for missing particle IDs.

    In power (Laguerre) diagrams, some sites may have empty cells and Voro++
    will omit them from iteration. This helper restores a full length-n
    output (IDs 0..n-1), marking missing entries as empty.

    The inserted records are intentionally minimal but include the same top-level
    keys as non-empty cells for the requested outputs.

    Args:
        cells: List of per-cell dictionaries returned by the C++ layer.
        n: Total number of input sites.
        sites: Site positions aligned with internal IDs (shape (n,3)).
        opts: (return_vertices, return_adjacency, return_faces)
    """
    if n <= 0:
        return

    present = {int(c.get('id', -1)) for c in cells}
    missing = [i for i in range(n) if i not in present]
    if not missing:
        return

    ret_vertices, ret_adjacency, ret_faces = opts
    for i in missing:
        rec: dict[str, Any] = {
            'id': int(i),
            'empty': True,
            'volume': 0.0,
            'site': np.asarray(sites[i], dtype=np.float64).reshape(3).tolist(),
        }
        if ret_vertices:
            rec['vertices'] = []
        if ret_adjacency:
            rec['adjacency'] = []
        if ret_faces:
            rec['faces'] = []
        cells.append(rec)

    # Deterministic order is convenient for debugging and testing.
    cells.sort(key=lambda cc: int(cc.get('id', 0)))


def _validate_output(output: object) -> Literal['result', 'cells']:
    """Validate and resolve the public compute output selector."""

    return require_string_choice(
        output,
        name='output',
        choices=('result', 'cells'),
    )  # type: ignore[return-value]


def _finish_compute_output(
    *,
    output: Literal['result', 'cells'],
    return_diagnostics: bool,
    dimension: Literal[3],
    domain: Box | OrthorhombicCell | PeriodicCell,
    mode: Literal['standard', 'power'],
    sites: np.ndarray,
    ids: np.ndarray | None,
    cells: list[dict[str, Any]],
    power_input: ResolvedPowerInput,
    diagnostics: TessellationDiagnostics | None,
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
        dimension=dimension,
        domain=domain,
        mode=mode,
        sites=sites,
        ids=ids,
        cells=cells,
        power_input=power_input,
        tessellation_diagnostics=diagnostics,
        boundaries_available=boundaries_available,
        periodic_shifts_available=periodic_shifts_available,
    )


def compute(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    weights: Sequence[float] | np.ndarray | None = None,
    radii: Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_faces: bool = True,
    return_face_shifts: bool = False,
    face_shift_search: int = 2,
    include_empty: bool = False,
    validate_face_shifts: bool = True,
    repair_face_shifts: bool = False,
    face_shift_tol: float | None = None,
    return_diagnostics: bool = False,
    output: Literal['result', 'cells'] = 'result',
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'none',
    tessellation_require_reciprocity: bool | None = None,
    tessellation_volume_tol_rel: float = 1e-8,
    tessellation_volume_tol_abs: float = 1e-12,
    tessellation_plane_offset_tol: float | None = None,
    tessellation_plane_angle_tol: float | None = None,
) -> (
    TessellationResult
    | list[dict[str, Any]]
    | tuple[list[dict[str, Any]], TessellationDiagnostics]
):
    """Compute Voronoi tessellation cells.

    Supported domains:
      - :class:`~pyvoro2.domains.Box` (non-periodic)
      - :class:`~pyvoro2.domains.OrthorhombicCell` (orthogonal with optional
        per-axis periodicity)
      - :class:`~pyvoro2.domains.PeriodicCell` (fully periodic triclinic)

    Supported modes:
      - ``mode='standard'``: classic Voronoi midplanes; both ``weights`` and
        ``radii`` must be ``None``
      - ``mode='power'``: power/Laguerre (radical) diagram using exactly one of
        mathematical power ``weights`` or backend-compatible ``radii``

    Notes:
        Internally, the C++ layer always uses point indices 0..n-1 as particle IDs.
        If `ids` is provided, results are remapped back to those user IDs on return.

    Args:
        points: Point coordinates, shape (n, 3).
        domain: Domain object.
        ids: Optional integer IDs returned in output. Defaults to `range(n)`.
        duplicate_check: Optional policy above the mandatory backend-safety
            distance. ``"off"`` skips that additional policy, ``"warn"`` emits
            a warning, and ``"raise"`` raises :class:`pyvoro2.DuplicateError`.
            Backend-unsafe pairs always raise before native insertion.
        duplicate_threshold: Absolute distance for the optional policy. Values
            at or below ``1e-5`` add no range above mandatory safety.
        duplicate_wrap: Whether the optional policy uses periodic minimum-image
            distance. Mandatory periodic safety always uses certified wrapping.
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Positive finite approximate grid block size. If provided,
            block counts are derived unless explicit ``blocks`` are supplied.
        blocks: Explicit positive exact-integer ``(nx, ny, nz)`` grid counts.
            These select the counts instead of ``block_size`` derivation.
        init_mem: Positive exact-integer initial per-block particle capacity in
            Voro++. Known eager native construction allocations are subject to
            an aggregate cap of exactly 1 GiB.
        mode: 'standard' or 'power'.
        weights: Per-point mathematical power weights for ``mode='power'``,
            with shape ``(n,)`` and squared-length units. Positive, zero, and
            negative finite values are accepted when the common-shift
            conversion remains finite and representable. Non-finite input or
            overflow during conversion raises ``ValueError`` before native
            computation. One common global shift is applied before conversion
            to non-negative backend radii; adding a common constant
            to every weight therefore leaves the diagram unchanged. The
            convention is ``||x - p_i||^2 - w_i``. Supplying both ``weights``
            and ``radii`` is an error in power mode. Standard mode rejects both
            arguments.
        radii: Per-point non-negative backend radii for ``mode='power'``, with
            length units. This backend representation is not unique and
            should not be interpreted as physical radii. Radii are rejected in
            standard mode. Finite values do not guarantee a numerically
            resolvable native tessellation. Voro++ evaluates radical geometry
            with binary64 squared-radius arithmetic, so very large absolute
            ``radii**2`` values or genuine weight ranges relative to squared
            coordinate/domain scales can lose geometric resolution. There is no
            universal safe cutoff: the onset depends on scale, geometry,
            platform, and compiler, and periodic power tessellations are a
            particularly sensitive regime.
        return_vertices: Include vertex coordinates.
        return_adjacency: Include vertex adjacency.
        return_faces: Include faces with adjacent cell IDs.
        return_face_shifts: For periodic domains, include an integer lattice shift
            (na, nb, nc) for each face neighbor indicating which periodic image
            of the adjacent cell generated that face.
            Requires `return_faces=True` and `return_vertices=True`.
        face_shift_search: Search radius S for determining neighbor shifts.
            Candidate shifts (na,nb,nc) in [-S..S]^3 are considered (restricted to
            periodic axes for :class:`~pyvoro2.domains.OrthorhombicCell`).
        include_empty: If True, include explicit empty-cell records for sites that
            do not produce a Voronoi/Laguerre cell (possible in extreme power
            settings). Empty records have 'empty': True, volume 0.0, and empty
            geometry lists.
        validate_face_shifts: If True and return_face_shifts=True, validate that
            each face's chosen adjacent_shift yields a near-zero plane residual,
            and that reciprocal faces carry opposite shifts.
        repair_face_shifts: If True and return_face_shifts=True, attempt to repair
            rare reciprocity mismatches by enforcing opposite shifts on reciprocal
            faces.
        face_shift_tol: Optional absolute tolerance (in container distance units) for
            the face-shift plane residual check. If None, a conservative default is
            used.

        output: ``"result"`` (the default) returns one
            :class:`~pyvoro2.TessellationResult`. ``"cells"`` selects the
            explicit raw cell list, or ``(cells, diagnostics)`` when
            ``return_diagnostics=True``.

    Returns:
        A :class:`~pyvoro2.TessellationResult` by default. The explicit
        ``output="cells"`` route returns raw cell dictionaries and is a
        supported low-level output mode.

    Raises:
        ValueError: If inputs are inconsistent or an unknown mode is provided.

    Every generator must lie in each non-periodic half-open interval
    ``[lo, hi)``; periodic axes are remapped before native dispatch. Generator
    pairs at squared distance at most ``1e-10`` always raise before insertion.
    The public duplicate options control only additional diagnostics above this
    backend-safety floor.
    """
    resolved_output = _validate_output(output)
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
    face_shift_search_value = require_nonnegative_index(
        face_shift_search,
        name='face_shift_search',
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
    return_faces_value = require_bool(return_faces, name='return_faces')
    return_face_shifts_value = require_bool(
        return_face_shifts,
        name='return_face_shifts',
    )
    include_empty_value = require_bool(include_empty, name='include_empty')
    validate_face_shifts_value = require_bool(
        validate_face_shifts,
        name='validate_face_shifts',
    )
    repair_face_shifts_value = require_bool(
        repair_face_shifts,
        name='repair_face_shifts',
    )
    return_diagnostics_value = require_bool(
        return_diagnostics,
        name='return_diagnostics',
    )
    tessellation_require_reciprocity_value = require_optional_bool(
        tessellation_require_reciprocity,
        name='tessellation_require_reciprocity',
    )
    face_shift_tol_value = require_optional_nonnegative_finite_real(
        face_shift_tol,
        name='face_shift_tol',
    )
    volume_tol_rel_value = require_nonnegative_finite_real(
        tessellation_volume_tol_rel,
        name='tessellation_volume_tol_rel',
    )
    volume_tol_abs_value = require_nonnegative_finite_real(
        tessellation_volume_tol_abs,
        name='tessellation_volume_tol_abs',
    )
    plane_offset_tol_value = require_optional_nonnegative_finite_real(
        tessellation_plane_offset_tol,
        name='tessellation_plane_offset_tol',
    )
    plane_angle_tol_value = require_optional_nonnegative_finite_real(
        tessellation_plane_angle_tol,
        name='tessellation_plane_angle_tol',
    )
    if repair_face_shifts_value:
        validate_face_shifts_value = True
    init_mem_value = require_positive_index(
        init_mem,
        name='init_mem',
        maximum=CPP_INT_MAX,
    )
    blocks_value, block_size_value = coerce_native_block_parameters(
        blocks=blocks,
        block_size=block_size,
        dim=3,
    )
    pts = coerce_point_array(points, name='points', dim=3)
    n = int(pts.shape[0])
    power_input = resolve_power_input(
        mode=mode,
        weights=weights,
        radii=radii,
        n=n,
    )
    rr = power_input.backend_radii

    geom = geometry3d(domain)
    if isinstance(domain, (Box, OrthorhombicCell)):
        native_bounds = geom.native_bounds
        native_cell = None
        native_params = None
    else:
        native_bounds = None
        native_cell = geom.native_periodic_snapshot()
        native_params = native_cell.params
    prepared = prepare_generators(
        pts,
        geometry=geom,
        operation='compute',
        external_ids=ids,
        backend_radii=rr,
        duplicate_check=duplicate_check,
        duplicate_threshold=duplicate_threshold_value,
        duplicate_wrap=duplicate_wrap_value,
        duplicate_max_pairs=duplicate_max_pairs_value,
        periodic_snapshot=native_cell,
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
    native_scale = (
        native_cell.length_scale
        if native_cell is not None
        else domain_length_scale(domain)
    )
    _warn_if_scale_suspicious(pts=pts, length_scale=native_scale)
    nx, ny, nz = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
        periodic_snapshot=native_cell,
    )

    opts = (
        return_vertices_value,
        return_adjacency_value,
        return_faces_value,
    )

    core = _require_core()

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        assert native_bounds is not None
        bounds = native_bounds
        periodic_flags = geom.periodic_axes
        is_periodic = geom.has_any_periodic_axis
        if return_face_shifts_value:
            if not is_periodic:
                raise ValueError(
                    'return_face_shifts is only supported for periodic domains '
                    '(PeriodicCell, or OrthorhombicCell with any periodic axis)'
                )
            if not return_faces_value:
                raise ValueError('return_face_shifts requires return_faces=True')
            if not return_vertices_value:
                raise ValueError('return_face_shifts requires return_vertices=True')

        if mode == 'standard':
            cells = core.compute_box_standard(
                pts_native,
                ids_internal,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                opts,
            )

        elif mode == 'power':
            assert rr is not None
            cells = core.compute_box_power(
                pts_native,
                ids_internal,
                rr,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                opts,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

        validate_compute_internal_ids(cells, n=n, mode=mode)

        if include_empty_value:
            sites_for_empty = prepared.primary_points_cart
            _add_empty_cells_inplace(cells, n=n, sites=sites_for_empty, opts=opts)

        if return_face_shifts_value:
            assert isinstance(domain, OrthorhombicCell)
            a, b, cvec = domain.lattice_vectors
            _add_periodic_face_shifts_inplace(
                cells,
                lattice_vectors=(a, b, cvec),
                periodic_mask=periodic_flags,
                mode=mode,
                radii=rr,
                search=face_shift_search_value,
                tol=face_shift_tol_value,
                validate=validate_face_shifts_value,
                repair=repair_face_shifts_value,
            )
        if ids_user is not None:
            _remap_ids_inplace(cells, ids_user)

        diag: TessellationDiagnostics | None = None
        do_diag = return_diagnostics_value or tessellation_check != 'none'
        if do_diag:
            expected = ids_user.tolist() if ids_user is not None else list(range(n))
            diag = analyze_tessellation(
                cells,
                domain,
                expected_ids=expected,
                mode=mode,
                volume_tol_rel=volume_tol_rel_value,
                volume_tol_abs=volume_tol_abs_value,
                check_reciprocity=bool(is_periodic),
                check_plane_mismatch=bool(is_periodic),
                plane_offset_tol=plane_offset_tol_value,
                plane_angle_tol=plane_angle_tol_value,
                mark_faces=bool(is_periodic),
            )

            if tessellation_require_reciprocity_value is None:
                tessellation_require_reciprocity_value = bool(is_periodic) and mode in (
                    'standard',
                    'power',
                )

            if tessellation_check in ('warn', 'raise'):
                ok = bool(diag.ok_volume) and (
                    bool(diag.ok_reciprocity)
                    if tessellation_require_reciprocity_value
                    else True
                )
                if not ok:
                    msg = (
                        f'tessellation_check failed (mode={mode!r}): '
                        f'volume_ratio={diag.volume_ratio:g}, '
                        f'orphan_faces={diag.n_faces_orphan}, '
                        f'mismatched_faces={diag.n_faces_mismatched}'
                    )
                    if tessellation_check == 'raise':
                        raise TessellationError(msg, diag)
                    warnings.warn(msg, stacklevel=2)

        return _finish_compute_output(
            output=resolved_output,
            return_diagnostics=return_diagnostics_value,
            dimension=3,
            domain=domain,
            mode=mode,
            sites=pts,
            ids=ids_user,
            cells=cells,
            power_input=power_input,
            diagnostics=diag,
            boundaries_available=return_faces_value,
            periodic_shifts_available=return_face_shifts_value,
        )

    # --- PeriodicCell (triclinic) ---
    #
    # Generator preparation has already transformed and remapped points into
    # Voro++'s primary internal half-open cell.
    cell = native_cell
    assert cell is not None
    assert native_params is not None
    bx, bxy, by, bxz, byz, bz = native_params
    pts_i = pts_native

    if return_face_shifts_value:
        if not return_faces_value:
            raise ValueError('return_face_shifts requires return_faces=True')
        if not return_vertices_value:
            raise ValueError('return_face_shifts requires return_vertices=True')

    if mode == 'standard':
        cells = core.compute_periodic_standard(
            pts_i,
            ids_internal,
            (bx, bxy, by, bxz, byz, bz),
            (nx, ny, nz),
            init_mem_value,
            opts,
        )

    elif mode == 'power':
        assert rr is not None
        cells = core.compute_periodic_power(
            pts_i,
            ids_internal,
            rr,
            (bx, bxy, by, bxz, byz, bz),
            (nx, ny, nz),
            init_mem_value,
            opts,
        )

    else:
        raise ValueError(f'unknown mode: {mode}')

    validate_compute_internal_ids(cells, n=n, mode=mode)

    # Determine periodic-image shifts for face neighbors (optional)
    if include_empty_value:
        # Voro++ remaps inserted points into the primary cell; mirror that here for
        # any empty-cell records we inject.
        sites_for_empty = pts_i
        _add_empty_cells_inplace(cells, n=n, sites=sites_for_empty, opts=opts)

    if return_face_shifts_value:
        a = np.array([bx, 0.0, 0.0], dtype=np.float64)
        b = np.array([bxy, by, 0.0], dtype=np.float64)
        cvec = np.array([bxz, byz, bz], dtype=np.float64)
        _add_periodic_face_shifts_inplace(
            cells,
            lattice_vectors=(a, b, cvec),
            periodic_mask=(True, True, True),
            mode=mode,
            radii=rr,
            search=face_shift_search_value,
            tol=face_shift_tol_value,
            validate=validate_face_shifts_value,
            repair=repair_face_shifts_value,
        )

    # Remap ids (and face neighbor ids) to user ids if requested
    if ids_user is not None:
        _remap_ids_inplace(cells, ids_user)

    # Transform vertices back to Cartesian if requested
    if return_vertices_value:
        for c in cells:
            verts = np.asarray(c.get('vertices', []), dtype=np.float64)
            if verts.size:
                c['vertices'] = cell.internal_to_cart(verts).tolist()

    # Transform site positions back to Cartesian for periodic cells
    for c in cells:
        site_i = np.asarray(c.get('site', []), dtype=np.float64)
        if site_i.size == 3:
            c['site'] = cell.internal_to_cart(site_i.reshape(1, 3)).reshape(3).tolist()

    diag = None
    do_diag = return_diagnostics_value or tessellation_check != 'none'
    if do_diag:
        expected = ids_user.tolist() if ids_user is not None else list(range(n))
        diag = analyze_tessellation(
            cells,
            domain,
            expected_ids=expected,
            mode=mode,
            volume_tol_rel=volume_tol_rel_value,
            volume_tol_abs=volume_tol_abs_value,
            check_reciprocity=True,
            check_plane_mismatch=True,
            plane_offset_tol=plane_offset_tol_value,
            plane_angle_tol=plane_angle_tol_value,
            mark_faces=True,
        )

        if tessellation_require_reciprocity_value is None:
            # Standard Voronoi and power diagrams are true tessellations; missing
            # reciprocity/mismatch indicates a bug or numerical issue.
            tessellation_require_reciprocity_value = mode in ('standard', 'power')

        if tessellation_check in ('warn', 'raise'):
            ok = bool(diag.ok_volume) and (
                bool(diag.ok_reciprocity)
                if tessellation_require_reciprocity_value
                else True
            )
            if not ok:
                msg = (
                    f'tessellation_check failed (mode={mode!r}): '
                    f'volume_ratio={diag.volume_ratio:g}, '
                    f'orphan_faces={diag.n_faces_orphan}, '
                    f'mismatched_faces={diag.n_faces_mismatched}'
                )
                if tessellation_check == 'raise':
                    raise TessellationError(msg, diag)
                warnings.warn(msg, stacklevel=2)

    return _finish_compute_output(
        output=resolved_output,
        return_diagnostics=return_diagnostics_value,
        dimension=3,
        domain=domain,
        mode=mode,
        sites=pts,
        ids=ids_user,
        cells=cells,
        power_input=power_input,
        diagnostics=diag,
        boundaries_available=return_faces_value,
        periodic_shifts_available=return_face_shifts_value,
    )


def locate(
    points: Sequence[Sequence[float]] | np.ndarray,
    queries: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    return_owner_position: bool = False,
) -> dict[str, Any]:
    """Locate which generator owns each query point.

    This is a stateless wrapper around Voro++'s ``find_voronoi_cell``.
    Persistent generators use half-open containment and mandatory duplicate
    safety. Query points are not inserted and retain the existing query
    semantics, including queries outside a non-periodic domain.

    Args:
        points: Generator coordinates, shape (n, 3).
        queries: Query coordinates, shape (m, 3).
        domain: Domain object (Box, OrthorhombicCell, or PeriodicCell).
        ids: Optional user IDs aligned with points. If provided, returned
            owner IDs are remapped to these values.
        duplicate_check: Optional off/warn/raise policy above the mandatory
            backend-safety distance. Backend-unsafe pairs always raise.
        duplicate_threshold: Absolute distance for the optional policy.
        duplicate_wrap: Whether the optional policy uses periodic minimum-image
            distance. Mandatory periodic safety always wraps.
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Positive finite approximate grid block size. If provided,
            block counts are derived unless explicit ``blocks`` are supplied.
        blocks: Explicit positive exact-integer ``(nx, ny, nz)`` grid counts.
            These select the counts instead of ``block_size`` derivation.
        init_mem: Positive exact-integer initial per-block particle capacity in
            Voro++. Known eager native construction allocations are subject to
            an aggregate cap of exactly 1 GiB.
        mode: 'standard' or 'power'.
        radii: Per-point radii for `mode='power'`.
        return_owner_position: If True, also return the (possibly periodic-image)
            position of the owning generator as reported by Voro++.

    Returns:
        A dict with:
            - ``found``: (m,) boolean array
            - ``owner_id``: (m,) integer array (internal 0..n-1, or remapped to `ids`)
            - ``owner_pos``: (m, 3) float array (only if ``return_owner_position=True``)

    Notes:
        For periodic domains, Voro++ may return the owner position in a periodic
        image of the primary domain. This is useful when you need a consistent
        nearest-image geometry for a given query.
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
        dim=3,
    )
    pts = coerce_point_array(points, name='points', dim=3)
    q = coerce_point_array(queries, name='queries', dim=3)

    n = int(pts.shape[0])
    rr: np.ndarray | None = None
    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        rr = coerce_nonnegative_vector(radii, name='radii', n=n)
    geom = geometry3d(domain)
    if isinstance(domain, (Box, OrthorhombicCell)):
        native_bounds = geom.native_bounds
        native_cell = None
        native_params = None
    else:
        native_bounds = None
        native_cell = geom.native_periodic_snapshot()
        native_params = native_cell.params
    prepared = prepare_generators(
        pts,
        geometry=geom,
        operation='locate',
        external_ids=ids,
        backend_radii=rr,
        duplicate_check=duplicate_check,
        duplicate_threshold=duplicate_threshold_value,
        duplicate_wrap=duplicate_wrap_value,
        duplicate_max_pairs=duplicate_max_pairs_value,
        periodic_snapshot=native_cell,
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
    native_scale = (
        native_cell.length_scale
        if native_cell is not None
        else domain_length_scale(domain)
    )
    _warn_if_scale_suspicious(pts=pts, length_scale=native_scale)
    nx, ny, nz = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
        periodic_snapshot=native_cell,
    )

    core = _require_core()

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        assert native_bounds is not None
        bounds = native_bounds
        periodic_flags = geom.periodic_axes

        if mode == 'standard':
            found, owner_id, owner_pos = core.locate_box_standard(
                pts_native,
                ids_internal,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                q,
            )
        elif mode == 'power':
            assert rr is not None
            found, owner_id, owner_pos = core.locate_box_power(
                pts_native,
                ids_internal,
                rr,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                q,
            )
        else:
            raise ValueError(f'unknown mode: {mode}')

    # --- PeriodicCell (triclinic) ---
    else:
        cell = native_cell
        assert cell is not None
        assert native_params is not None
        bx, bxy, by, bxz, byz, bz = native_params
        with np.errstate(over='ignore', invalid='ignore'):
            q_i = cell.cart_to_internal(q)
        pts_i = pts_native
        q_i = coerce_point_array(q_i, name='queries', dim=3)

        if mode == 'standard':
            found, owner_id, owner_pos = core.locate_periodic_standard(
                pts_i,
                ids_internal,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem_value,
                q_i,
            )
        elif mode == 'power':
            assert rr is not None
            found, owner_id, owner_pos = core.locate_periodic_power(
                pts_i,
                ids_internal,
                rr,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem_value,
                q_i,
            )
        else:
            raise ValueError(f'unknown mode: {mode}')

        # Convert owner positions back to Cartesian if requested.
        # Note: owner_pos may already be outside the primary cell due to
        # periodic images.
        if return_owner_position_value:
            owner_pos = cell.internal_to_cart(np.asarray(owner_pos, dtype=np.float64))

    # Remap owner IDs to user IDs if requested.
    owner_id = np.asarray(owner_id)
    found = np.asarray(found, dtype=bool)

    if ids_user is not None:
        out_ids = owner_id.astype(np.int64, copy=True)
        mask = out_ids >= 0
        if np.any(mask):
            out_ids[mask] = ids_user[out_ids[mask]]
        owner_id = out_ids

    out: dict[str, Any] = {
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
    domain: Box | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int] | None = None,
    duplicate_check: Literal['off', 'warn', 'raise'] = 'off',
    duplicate_threshold: float = 1e-5,
    duplicate_wrap: bool = True,
    duplicate_max_pairs: int = 10,
    block_size: float | None = None,
    blocks: tuple[int, int, int] | None = None,
    init_mem: int = 8,
    mode: Literal['standard', 'power'] = 'standard',
    radii: Sequence[float] | np.ndarray | None = None,
    ghost_radius: float | Sequence[float] | np.ndarray | None = None,
    return_vertices: bool = True,
    return_adjacency: bool = True,
    return_faces: bool = True,
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    """Compute ghost Voronoi/Laguerre cells at arbitrary query positions.

    This is a stateless wrapper around Voro++'s ``compute_ghost_cell`` routine.
    Each query is temporarily inserted, so it uses the same containment and
    mandatory duplicate-safety rules as persistent generators. An outside
    non-periodic query therefore raises before native dispatch; a contained,
    distinct query may still have an empty cell geometrically.
    It is useful for probing the tessellation at positions that are not part of
    the generator set (e.g. along a line/trajectory, or at grid points).

    Compared to :func:`pyvoro2.compute`, ghost cells are *not* part of a global
    tessellation and therefore:

      - Ghost cells are returned with ``id = -1``.
      - The returned faces' ``adjacent_cell`` values refer to *generator* IDs
        (0..n-1, or remapped to `ids` if provided).
      - No periodic face-shift annotation is performed.

    Args:
        points: Generator coordinates, shape (n, 3).
        queries: Query coordinates, shape (m, 3).
        domain: Domain (Box, OrthorhombicCell, or PeriodicCell).
        ids: Optional user IDs aligned with points. If provided, face neighbor
            IDs are remapped to these values.
        duplicate_check: Optional off/warn/raise policy above the mandatory
            backend-safety distance. Backend-unsafe persistent or temporary
            ghost generators always raise.
        duplicate_threshold: Absolute distance for the optional policy.
        duplicate_wrap: Whether the optional policy uses periodic minimum-image
            distance. Mandatory periodic safety always wraps.
        duplicate_max_pairs: Maximum number of near-duplicate pairs reported.
        block_size: Positive finite approximate grid block size. If provided,
            block counts are derived unless explicit ``blocks`` are supplied.
        blocks: Explicit positive exact-integer ``(nx, ny, nz)`` grid counts.
            These select the counts instead of ``block_size`` derivation.
        init_mem: Positive exact-integer initial per-block particle capacity in
            Voro++. Known eager native construction allocations are subject to
            an aggregate cap of exactly 1 GiB.
        mode: 'standard' or 'power'.
        radii: Per-point radii for `mode='power'`.
        ghost_radius: Radius (or array of radii) for each ghost query point in
            `mode='power'`. Must be provided for power mode.
        return_vertices: Include vertex coordinates.
        return_adjacency: Include vertex adjacency.
        return_faces: Include faces with adjacent generator IDs.
        include_empty: If True, return an explicit empty record for queries for
            which Voro++ cannot compute a cell (e.g. outside a non-periodic box).
            Empty records have ``empty=True`` and volume 0.0. If False, those
            queries are omitted from the output list.

    Returns:
        A list of cell dicts (length ``m`` unless ``include_empty=False``).

        Each element contains:
            - ``query_index``: index of the query in the input array
            - ``query``: original query coordinate (Cartesian)
            - ``site``: coordinate used by Voro++ for the ghost. For periodic
              domains, this is wrapped into the primary domain. Returned in
              Cartesian coordinates.
            - ``empty``: boolean
            - ``volume``: float
            - optional ``vertices``, ``adjacency``, ``faces``

    Raises:
        ValueError: if inputs are inconsistent.
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
    return_vertices_value = require_bool(
        return_vertices,
        name='return_vertices',
    )
    return_adjacency_value = require_bool(
        return_adjacency,
        name='return_adjacency',
    )
    return_faces_value = require_bool(return_faces, name='return_faces')
    include_empty_value = require_bool(include_empty, name='include_empty')
    init_mem_value = require_positive_index(
        init_mem,
        name='init_mem',
        maximum=CPP_INT_MAX,
    )
    blocks_value, block_size_value = coerce_native_block_parameters(
        blocks=blocks,
        block_size=block_size,
        dim=3,
    )
    pts = coerce_point_array(points, name='points', dim=3)
    q = coerce_point_array(queries, name='queries', dim=3)

    n = int(pts.shape[0])
    m = int(q.shape[0])

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

    geom = geometry3d(domain)
    if isinstance(domain, (Box, OrthorhombicCell)):
        native_bounds = geom.native_bounds
        native_cell = None
        native_params = None
    else:
        native_bounds = None
        native_cell = geom.native_periodic_snapshot()
        native_params = native_cell.params
    prepared = prepare_generators(
        pts,
        geometry=geom,
        operation='ghost_cells',
        external_ids=ids,
        backend_radii=rr,
        duplicate_check=duplicate_check,
        duplicate_threshold=duplicate_threshold_value,
        duplicate_wrap=duplicate_wrap_value,
        duplicate_max_pairs=duplicate_max_pairs_value,
        periodic_snapshot=native_cell,
    )
    temporary = prepare_temporary_generators(
        q,
        persistent=prepared,
        geometry=geom,
        backend_radii=gr,
        duplicate_check=duplicate_check,
        duplicate_threshold=duplicate_threshold_value,
        duplicate_wrap=duplicate_wrap_value,
        duplicate_max_pairs=duplicate_max_pairs_value,
        periodic_snapshot=native_cell,
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    q_native = temporary.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
    gr = temporary.backend_radii
    native_scale = (
        native_cell.length_scale
        if native_cell is not None
        else domain_length_scale(domain)
    )
    _warn_if_scale_suspicious(pts=pts, length_scale=native_scale)
    nx, ny, nz = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
        periodic_snapshot=native_cell,
    )

    opts = (
        return_vertices_value,
        return_adjacency_value,
        return_faces_value,
    )
    core = _require_core()

    # --- Rectangular containers (Box / OrthorhombicCell) ---
    if isinstance(domain, (Box, OrthorhombicCell)):
        assert native_bounds is not None
        bounds = native_bounds
        periodic_flags = geom.periodic_axes

        q_call = q_native

        if mode == 'standard':
            cells = core.ghost_box_standard(
                pts_native,
                ids_internal,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                opts,
                q_call,
            )

        elif mode == 'power':
            assert rr is not None
            assert gr is not None

            cells = core.ghost_box_power(
                pts_native,
                ids_internal,
                rr,
                bounds,
                (nx, ny, nz),
                periodic_flags,
                init_mem_value,
                opts,
                q_call,
                gr,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

    # --- PeriodicCell (triclinic) ---
    else:
        cell = native_cell
        assert cell is not None
        assert native_params is not None
        bx, bxy, by, bxz, byz, bz = native_params

        pts_i = pts_native
        q_i = q_native

        if mode == 'standard':
            cells = core.ghost_periodic_standard(
                pts_i,
                ids_internal,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem_value,
                opts,
                q_i,
            )

        elif mode == 'power':
            assert rr is not None
            assert gr is not None

            cells = core.ghost_periodic_power(
                pts_i,
                ids_internal,
                rr,
                (bx, bxy, by, bxz, byz, bz),
                (nx, ny, nz),
                init_mem_value,
                opts,
                q_i,
                gr,
            )

        else:
            raise ValueError(f'unknown mode: {mode}')

        # Convert vertices/site back to Cartesian for PeriodicCell.
        if return_vertices_value:
            for c in cells:
                verts = np.asarray(c.get('vertices', []), dtype=np.float64)
                if verts.size:
                    c['vertices'] = cell.internal_to_cart(verts).tolist()

        for c in cells:
            site_i = np.asarray(c.get('site', []), dtype=np.float64)
            if site_i.size == 3:
                c['site'] = (
                    cell.internal_to_cart(site_i.reshape(1, 3)).reshape(3).tolist()
                )

    # Remap generator IDs on faces to user IDs if requested.
    if ids_user is not None:
        _remap_ids_inplace(cells, ids_user)

    # Add original query coordinates (Cartesian) to each record.
    q_list = q.tolist()
    for c in cells:
        qi = int(c.get('query_index', -1))
        if 0 <= qi < m:
            c['query'] = q_list[qi]
        else:
            c['query'] = None

    if not include_empty_value:
        cells = [c for c in cells if not bool(c.get('empty', False))]

    return cells
