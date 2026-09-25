"""High-level 2D API for planar Voronoi and power tessellations."""

from __future__ import annotations

from typing import Any, Literal, Sequence
from dataclasses import replace

import warnings

import numpy as np

from .._internal.cell_output import remap_ids_inplace
from .._internal.inputs import (
    coerce_native_block_parameters,
    coerce_point_array,
    validate_forward_mode,
    validate_duplicate_check_mode,
    validate_duplicate_options,
)
from .._internal.generator_preparation import (
    prepare_generators,
    prepare_temporary_generators,
    validate_compute_internal_ids,
)
from .._internal.power_input import (
    ResolvedPowerInput,
    resolve_ghost_power_input,
    resolve_power_input,
)
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
    TessellationIssue,
    _analyze_tessellation,
)
from .domains import Box, RectangularCell
from .._internal.planar.wp6_certificate import WP6Failure
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
            f'(L≈{length_scale:.3g}). pyvoro2 reserves distances through '
            '1e-5 for backend safety, and Voro++ uses other fixed absolute '
            'tolerances. Consider rescaling your coordinates before calling '
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


def _certify_wp6(*args, **kwargs):
    from .._internal.planar.wp6_certificate import certify_packet
    return certify_packet(*args, **kwargs)


def _native_wp6_failure(exc):
    parts = str(exc).split(':', 3)
    stage = parts[1] if len(
        parts) > 2 and parts[0] == 'planar_certification' else 'native'
    codes = {
        'profile': 'WP6_PROFILE_UNSUPPORTED',
        'insertion': 'WP6_INSERTION_FAILED',
        'provenance': 'WP6_PROVENANCE_INVALID',
        'resource': 'WP6_ATTRIBUTION_RESOURCE',
        'native': 'WP6_BACKEND_FAILURE',
    }
    return WP6Failure(codes.get(stage, 'WP6_BACKEND_FAILURE'), str(exc), stage=stage)


def _attach_wp6_findings(diag, findings, prepared):
    ids = prepared.external_ids.tolist()

    def public_id(value):
        return ids[value] if type(value) is int and 0 <= value < len(ids) else value

    issues = []
    for finding in findings:
        context = dict(finding.context)
        if 'source_id' in context:
            context['source_id'] = public_id(context['source_id'])
        if 'label' in context and isinstance(context['label'], tuple):
            owner, shift = context['label']
            context['label'] = (public_id(owner), shift)
        issues.append(TessellationIssue(finding.code, finding.severity,
                                        str(finding), (context,)))
    return replace(diag, issues=(*diag.issues, *issues),
                   ok=diag.ok and not any(i.severity == 'error' for i in issues))


def _raise_wp6_failure(failure, domain, prepared, mode):
    """Abort atomically without claiming that unobserved native geometry passed."""
    findings = (failure,) if isinstance(failure, WP6Failure) else tuple(failure)
    diag = TessellationDiagnostics(
        domain_area=float('nan'), sum_cell_area=float('nan'), area_ratio=float('nan'),
        area_gap=float('nan'), area_overlap=float('nan'),
        n_sites_expected=len(prepared.internal_ids), n_cells_returned=0,
        missing_ids=(), empty_ids=(), edge_shift_available=False,
        reciprocity_checked=False, n_edges_total=0, n_edges_orphan=0,
        n_edges_mismatched=0, issues=(), ok_area=False, ok_reciprocity=False, ok=False,
    )
    diag = _attach_wp6_findings(diag, findings, prepared)
    first = findings[0]
    raise TessellationError(f'{first.code}: {first}', diag) from first


def _wp6_reciprocal_occurrences(cells, certificate, numerical_findings):
    """Count raw missing opposite classes without assigning error severity."""
    occurrences = [o for o in certificate.occurrences if o.shift is not None]
    keys = {(o.source, o.owner, o.shift) for o in occurrences}
    orphan_keys = {key for key in keys
                   if (key[1], key[0], tuple(-s for s in key[2])) not in keys}
    by_id = {cell['id']: cell for cell in cells}

    def mark(source, slots, field):
        public_id = int(certificate.prepared.external_ids[source])
        edges = by_id.get(public_id, {}).get('edges', ())
        if edges:
            for slot in slots:
                edges[slot][field] = True

    count = 0
    for o in occurrences:
        if (o.source, o.owner, o.shift) in orphan_keys:
            count += 1
            mark(o.source, (o.slot,), 'orphan')
            mark(o.source, (o.slot,), 'reciprocal_missing')
    for issue in numerical_findings:
        if issue.code == 'RECIPROCAL_MISMATCH':
            context = issue.context
            mark(context['source_id'], context['source_slots'], 'reciprocal_mismatch')
            mark(context['label'][0], context['reciprocal_slots'],
                 'reciprocal_mismatch')
    return count


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
    include_empty: bool = False,
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

    ``tessellation_check='diagnose'`` attaches the completed diagnostics without
    acting on failure. ``'warn'`` emits one summary warning when the final
    diagnostic is not okay, and ``'raise'`` raises
    :class:`~pyvoro2.planar.TessellationError` in the same case.
    ``tessellation_require_reciprocity=None`` preserves the default requirement
    for periodic standard and power tessellations; ``False`` retains optional
    reciprocity findings without making them fatal.

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
    Generators must lie in each non-periodic half-open interval ``[lo, hi)``;
    periodic axes are remapped before native dispatch. Backend-unsafe pairs at
    squared distance at most ``1e-10`` always raise. ``duplicate_check`` and
    its threshold/wrap options control only diagnostics above that floor.
    """

    return _compute_impl(**locals())


def _compute_with_certificate(points, *, semantic_weights=None, **options):
    """Private resolved-state channel keeping mathematical weights beside radii."""
    sink = []
    result = _compute_impl(points, _semantic_weights=semantic_weights,
                           _certificate_sink=sink, **options)
    if len(sink) != 1:
        raise RuntimeError('private planar certificate channel was not completed')
    return result, sink[0]


def _compute_impl(
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
    include_empty: bool = False,
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
    _semantic_weights=None,
    _certificate_sink=None,
) -> (
    list[dict[str, Any]]
    | tuple[list[dict[str, Any]], TessellationDiagnostics]
    | TessellationResult
):
    """Validated implementation shared with private semantic consumers."""

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
    return_diagnostics_value = require_bool(
        return_diagnostics,
        name='return_diagnostics',
    )
    tessellation_require_reciprocity_value = require_optional_bool(
        tessellation_require_reciprocity,
        name='tessellation_require_reciprocity',
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
    power_input = resolve_power_input(
        mode=mode,
        weights=weights,
        radii=radii,
        n=n,
    )
    rr = power_input.backend_radii

    geom = geometry2d(domain)
    bounds = geom.native_bounds
    prepared = prepare_generators(
        pts,
        geometry=geom,
        operation='compute',
        unbounded_planar_shifts=True,
        external_ids=ids,
        backend_radii=rr,
        duplicate_check=duplicate_check,
        duplicate_threshold=duplicate_threshold_value,
        duplicate_wrap=duplicate_wrap_value,
        duplicate_max_pairs=duplicate_max_pairs_value,
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
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

    internal_return_vertices = user_return_vertices or need_norm_vertices
    internal_return_adjacency = user_return_adjacency
    internal_return_edges = user_return_edges or need_norm_vertices
    internal_return_edge_shifts = user_return_edge_shifts or (
        need_norm_vertices and periodic
    )

    if user_return_edge_shifts:
        if not periodic:
            raise ValueError(
                'return_edge_shifts is only supported for periodic domains '
                '(RectangularCell with any periodic axis)'
            )
        if not user_return_edges:
            raise ValueError('return_edge_shifts requires return_edges=True')

    opts = (False, internal_return_adjacency, internal_return_edges)
    core = _require_core2d()
    packet = None
    try:
        name = ('_compute_box_standard_witness' if mode == 'standard'
                else '_compute_box_power_witness')
        native_compute = getattr(core, name, None)
        if native_compute is None:
            raise WP6Failure('WP6_PROFILE_UNSUPPORTED',
                             'The planar extension has no supported source witness',
                             stage='profile')
        args = [pts_native, ids_internal]
        if mode == 'power':
            args.append(rr)
        native_result = native_compute(
            *args, bounds, (nx, ny), geom.periodic_axes, init_mem_value, opts)
        if not isinstance(native_result, tuple) or len(native_result) != 2:
            raise WP6Failure('WP6_PROVENANCE_INVALID',
                             'Native computation returned no associated witness',
                             stage='attribution')
        native_cells, packet = native_result
        certificate = _certify_wp6(
            native_cells, packet, prepared, domain, power_input, mode,
            semantic_weights=_semantic_weights,
            audit=need_diag or _certificate_sink is not None,
            reciprocity_required=(
                periodic if tessellation_require_reciprocity_value is None
                else tessellation_require_reciprocity_value
            ),
        )
        cells = certificate.public_cells(
            vertices=internal_return_vertices, adjacency=internal_return_adjacency,
            edges=internal_return_edges, shifts=internal_return_edge_shifts,
            include_empty=include_empty_value,
        )
        validate_compute_internal_ids(cells, n=n, mode=mode)
    except WP6Failure as exc:
        _raise_wp6_failure(exc, domain, prepared, mode)
    except RuntimeError as exc:
        error = _native_wp6_failure(exc)
        _raise_wp6_failure(error, domain, prepared, mode)
    if _certificate_sink is not None:
        _certificate_sink.append(certificate)

    if ids_user is not None:
        remap_ids_inplace(cells, ids_user, boundary_key='edges')

    diag: TessellationDiagnostics | None = None
    if need_diag:
        expected = ids_user.tolist() if ids_user is not None else list(range(n))
        if tessellation_require_reciprocity_value is None:
            tessellation_require_reciprocity_value = bool(periodic) and mode in (
                'standard',
                'power',
            )
        diag = _analyze_tessellation(
            cells,
            domain,
            expected_ids=expected,
            mode=mode,
            area_tol_rel=area_tol_rel_value,
            area_tol_abs=area_tol_abs_value,
            check_reciprocity=False,
            reciprocity_required=bool(
                tessellation_require_reciprocity_value
            ),
            check_line_mismatch=False,
            line_offset_tol=line_offset_tol_value,
            line_angle_tol=line_angle_tol_value,
            mark_edges=bool(periodic),
        )

        numerical_findings = ()
        if periodic and certificate.audit_complete:
            from .._internal.planar.wp6_numerical import numeric_findings
            numerical_findings = numeric_findings(
                certificate, offset_tol=line_offset_tol_value,
                angle_tol=line_angle_tol_value,
                required=tessellation_require_reciprocity_value,
            )
        findings = certificate.issues + numerical_findings
        diag = _attach_wp6_findings(diag, findings, prepared)
        reciprocal_complete = certificate.audit_complete and all(
            issue.context.get('audit_complete') is not False
            for issue in numerical_findings)
        orphan_count = _wp6_reciprocal_occurrences(
            cells, certificate, numerical_findings)
        diag = replace(
            diag, edge_shift_available=bool(periodic),
            reciprocity_checked=bool(periodic and reciprocal_complete),
            n_edges_total=sum(o.shift is not None for o in certificate.occurrences),
            n_edges_orphan=orphan_count,
            n_edges_mismatched=sum(
                issue.code == 'RECIPROCAL_MISMATCH' for issue in numerical_findings),
            ok_reciprocity=reciprocal_complete and not any(
                issue.code in ('WP6_RECIPROCAL_CONTACT', 'RECIPROCAL_MISMATCH')
                for issue in findings),
        )

        if tessellation_check in ('warn', 'raise'):
            if not diag.ok:
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
    try:
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
    except (ValueError, OverflowError) as exc:
        _raise_wp6_failure(
            WP6Failure('WP6_NORMALIZATION_REPRESENTATION',
                       'Required planar normalization has no valid representation',
                       stage='representation', normalization=normalize,
                       detail=str(exc)),
            domain, prepared, mode,
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
    weights: Sequence[float] | np.ndarray | None = None,
    radii: Sequence[float] | np.ndarray | None = None,
    return_owner_position: bool = False,
) -> dict[str, np.ndarray]:
    """Locate the owning generator for each planar query point.

    ``init_mem`` and explicit length-2 ``blocks`` use positive exact-integer
    semantics; ``block_size`` is positive and finite. Malformed or non-finite
    points, queries, weights, radii, domains, and over-cap known eager native
    allocation estimates raise ``ValueError`` before construction. Power mode
    requires exactly one of mathematical ``weights`` or explicit non-negative
    backend ``radii``.
    Generator points use the same half-open containment and mandatory duplicate
    safety as :func:`compute`; locate queries themselves are not inserted and
    retain their existing query semantics.
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
    power_input = resolve_power_input(
        mode=mode,
        weights=weights,
        radii=radii,
        n=n,
    )
    rr = power_input.backend_radii
    geom = geometry2d(domain)
    bounds = geom.native_bounds
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
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
    _warn_if_scale_suspicious(pts=pts, domain=domain)
    nx, ny = geom.resolve_block_counts(
        n_sites=n,
        blocks=blocks_value,
        block_size=block_size_value,
    )

    periodic_flags = geom.periodic_axes
    core = _require_core2d()

    if mode == 'standard':
        found, owner_id, owner_pos = core.locate_box_standard(
            pts_native,
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
            pts_native,
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
    weights: Sequence[float] | np.ndarray | None = None,
    radii: Sequence[float] | np.ndarray | None = None,
    ghost_weights: float | Sequence[float] | np.ndarray | None = None,
    ghost_radii: float | Sequence[float] | np.ndarray | None = None,
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
    points, queries, weights, radii, domains, and over-cap known eager native
    allocation estimates raise ``ValueError`` before construction. Power mode
    accepts exactly one complete ``weights``/``ghost_weights`` or
    ``radii``/``ghost_radii`` family. Persistent and ghost weights share one
    common backend-radius conversion gauge.
    Both persistent sites and each temporary ghost generator use half-open
    containment and mandatory duplicate safety. Periodic axes are remapped;
    an outside non-periodic ghost query raises instead of producing an empty
    ghost cell.
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

    power_input = resolve_ghost_power_input(
        mode=mode,
        weights=weights,
        radii=radii,
        ghost_weights=ghost_weights,
        ghost_radii=ghost_radii,
        n=n,
        m=m,
    )
    rr = power_input.backend_radii
    gr = power_input.backend_ghost_radii
    geom = geometry2d(domain)
    bounds = geom.native_bounds
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
        reserve_ghost_id=True,
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
    )
    pts = prepared.input_points_cart
    pts_native = prepared.native_points
    q_native = temporary.native_points
    ids_internal = prepared.internal_ids
    ids_user = prepared.external_ids if ids is not None else None
    rr = prepared.backend_radii
    gr = temporary.backend_radii
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

    periodic_flags = geom.periodic_axes
    opts = (
        return_vertices_value,
        return_adjacency_value,
        return_edges_value,
    )

    core = _require_core2d()
    if mode == 'standard':
        cells = core.ghost_box_standard(
            pts_native,
            ids_internal,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
            q_native,
        )
    elif mode == 'power':
        assert rr is not None
        assert gr is not None
        cells = core.ghost_box_power(
            pts_native,
            ids_internal,
            rr,
            bounds,
            (nx, ny),
            periodic_flags,
            init_mem_value,
            opts,
            q_native,
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
            site_positions=prepared.primary_points_cart,
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
