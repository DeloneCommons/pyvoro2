"""Severity-complete planar tessellation diagnostics and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import warnings

import numpy as np

from .._internal.inputs import coerce_external_id_array
from .._internal.planar.domain_geometry import geometry2d
from .._internal.tessellation_diagnostics import (
    classify_expected_ids,
    diagnostics_ok,
    reciprocity_issue_severity,
    reset_owned_annotations,
    stable_measure_sum,
    validate_cell_measure,
)
from .._internal.validation import (
    require_bool,
    require_nonnegative_finite_real,
    require_optional_bool,
    require_optional_nonnegative_finite_real,
    require_string,
    require_string_choice,
)
from .domains import Box, RectangularCell


Domain2D = Box | RectangularCell


@dataclass(frozen=True, slots=True)
class TessellationIssue:
    code: str
    severity: Literal['info', 'warning', 'error']
    message: str
    examples: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'code', require_string(self.code, name='code'))
        object.__setattr__(
            self,
            'severity',
            require_string_choice(
                self.severity,
                name='severity',
                choices=('info', 'warning', 'error'),
            ),
        )
        object.__setattr__(
            self,
            'message',
            require_string(self.message, name='message'),
        )


@dataclass(frozen=True, slots=True)
class TessellationDiagnostics:
    domain_area: float
    sum_cell_area: float
    area_ratio: float
    area_gap: float
    area_overlap: float
    n_sites_expected: int
    n_cells_returned: int
    missing_ids: tuple[int, ...]
    empty_ids: tuple[int, ...]
    edge_shift_available: bool
    reciprocity_checked: bool
    n_edges_total: int
    n_edges_orphan: int
    n_edges_mismatched: int
    issues: tuple[TessellationIssue, ...]
    ok_area: bool
    ok_reciprocity: bool
    ok: bool


class TessellationError(ValueError):
    """Raised when planar tessellation sanity checks fail."""

    def __init__(self, message: str, diagnostics: TessellationDiagnostics):
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


def _domain_area(domain: Domain2D) -> float:
    geom = geometry2d(domain)
    (_lengths, area) = geom._lengths_and_area()
    return float(area)


def _characteristic_length(domain: Domain2D) -> float:
    geom = geometry2d(domain)
    (lx, ly), _area = geom._lengths_and_area()
    L = float(max(lx, ly))
    return L if np.isfinite(L) else 0.0


def _is_periodic_domain(domain: Domain2D) -> bool:
    return bool(geometry2d(domain).has_any_periodic_axis)


def _normal_from_vertices(v: np.ndarray) -> np.ndarray | None:
    """Return a numerical normal, or None for coincident public endpoints."""

    if v.shape[0] < 2:
        return None
    dv = v[1] - v[0]
    nn = float(np.hypot(dv[0], dv[1]))
    if nn == 0.0:
        return None
    tangent = dv / nn
    return np.array([-tangent[1], tangent[0]], dtype=np.float64)


def _segment_union_covers(
    source: list[np.ndarray],
    target: list[np.ndarray],
    *,
    offset_tol: float,
    angle_tol: float,
    coordinate_tol: float,
) -> bool:
    """Compare numerical segment coverage without pairing raw occurrences.

    Projection intervals from every compatible target segment form a union.
    Tolerances describe only public-coordinate agreement, never exact contact
    status, native collapse, or positivity.
    """

    for segment in source:
        direction = segment[1] - segment[0]
        length = float(np.hypot(direction[0], direction[1]))
        if length == 0.0:
            covered = False
            for other in target:
                delta = other[1] - other[0]
                other_length = float(np.hypot(delta[0], delta[1]))
                if other_length == 0.0:
                    distance = segment[0] - other[0]
                else:
                    tangent = delta / other_length
                    position = float(np.dot(segment[0] - other[0], tangent))
                    position = min(other_length, max(0.0, position))
                    distance = segment[0] - (other[0] + position * tangent)
                if float(np.hypot(distance[0], distance[1])) <= coordinate_tol:
                    covered = True
                    break
            if not covered:
                return False
            continue

        tangent = direction / length
        normal = np.array([-tangent[1], tangent[0]])
        intervals: list[tuple[float, float]] = []
        for other in target:
            relative = other - segment[0]
            other_normal = _normal_from_vertices(other)
            if other_normal is not None:
                dot = abs(float(np.dot(normal, other_normal)))
                cross = abs(float(
                    normal[0] * other_normal[1] - normal[1] * other_normal[0]
                ))
                if float(np.arctan2(cross, dot)) > angle_tol:
                    continue
            if float(np.max(np.abs(relative @ normal))) > offset_tol:
                continue
            projected = relative @ tangent
            intervals.append((float(np.min(projected)), float(np.max(projected))))

        covered_until = 0.0
        started = False
        for lo, hi in sorted(intervals):
            if hi < -coordinate_tol:
                continue
            if lo > covered_until + coordinate_tol:
                break
            started = True
            covered_until = max(covered_until, hi)
            if covered_until >= length - coordinate_tol:
                break
        if not started or covered_until < length - coordinate_tol:
            return False
    return True


def analyze_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Domain2D,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    area_tol_rel: float = 1e-8,
    area_tol_abs: float = 1e-12,
    check_reciprocity: bool = True,
    check_line_mismatch: bool = True,
    line_offset_tol: float | None = None,
    line_angle_tol: float | None = None,
    mark_edges: bool = True,
) -> TessellationDiagnostics:
    """Analyze planar tessellation sanity and optionally annotate edges.

    Missing expected IDs are errors in standard mode, informational hidden
    cells in power mode, and warnings when ``mode=None``. Requested periodic
    reciprocity is required. Invalid cell areas and closure failures are
    explicit errors. When marking is enabled, analyzer-owned edge flags are
    reset before current findings are marked.

    This standalone utility inspects mutable raw records, area closure, and
    numerical reciprocal class coverage. It does not have the stored native
    population, mathematical weights, or private occurrence witness needed for
    an exact N/E/S audit. All generator occurrences, including tiny or publicly
    coincident segments, count as raw records. Repeated owner/image labels are
    compared as segment unions when public geometry is available, without
    requiring one-to-one fragment pairing. No tolerance establishes semantic
    positivity or certifies native collapse.
    """

    return _analyze_tessellation(
        cells,
        domain,
        expected_ids=expected_ids,
        mode=mode,
        area_tol_rel=area_tol_rel,
        area_tol_abs=area_tol_abs,
        check_reciprocity=check_reciprocity,
        reciprocity_required=True,
        check_line_mismatch=check_line_mismatch,
        line_offset_tol=line_offset_tol,
        line_angle_tol=line_angle_tol,
        mark_edges=mark_edges,
    )


def _analyze_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Domain2D,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    area_tol_rel: float = 1e-8,
    area_tol_abs: float = 1e-12,
    check_reciprocity: bool = True,
    reciprocity_required: bool,
    check_line_mismatch: bool = True,
    line_offset_tol: float | None = None,
    line_angle_tol: float | None = None,
    mark_edges: bool = True,
) -> TessellationDiagnostics:
    """Private analyzer with an explicit required/optional reciprocity policy."""

    if mode is not None:
        mode = require_string_choice(
            mode,
            name='mode',
            choices=('standard', 'power'),
        )
    area_tol_rel = require_nonnegative_finite_real(
        area_tol_rel,
        name='area_tol_rel',
    )
    area_tol_abs = require_nonnegative_finite_real(
        area_tol_abs,
        name='area_tol_abs',
    )
    check_reciprocity = require_bool(
        check_reciprocity,
        name='check_reciprocity',
    )
    check_line_mismatch = require_bool(
        check_line_mismatch,
        name='check_line_mismatch',
    )
    line_offset_tol = require_optional_nonnegative_finite_real(
        line_offset_tol,
        name='line_offset_tol',
    )
    line_angle_tol = require_optional_nonnegative_finite_real(
        line_angle_tol,
        name='line_angle_tol',
    )
    mark_edges = require_bool(mark_edges, name='mark_edges')
    expected_ids_array = (
        None
        if expected_ids is None
        else coerce_external_id_array(expected_ids, name='expected_ids')
    )

    issues: list[TessellationIssue] = []

    dom_area = _domain_area(domain)
    valid_areas: list[float] = []
    measures_valid = True
    empty_ids: list[int] = []
    present_ids: list[int] = []
    for cell in cells:
        cid = int(cell.get('id', -1))
        if cid >= 0:
            present_ids.append(cid)
        empty = bool(cell.get('empty', False))
        if empty:
            if cid >= 0:
                empty_ids.append(cid)
        measure = validate_cell_measure(cell, field='area', empty=empty)
        if not measure.valid:
            measures_valid = False
            for code in measure.issue_codes:
                if code == 'MISSING_CELL_MEASURE':
                    message = f'Non-empty cell {cid} is missing its area'
                elif code == 'INVALID_CELL_MEASURE':
                    message = f'Cell {cid} has an invalid non-real area'
                elif code == 'NONFINITE_CELL_MEASURE':
                    message = f'Cell {cid} has a non-finite area'
                elif code == 'NEGATIVE_CELL_MEASURE':
                    message = f'Cell {cid} has a negative area'
                else:
                    message = f'Empty cell {cid} has a nonzero area'
                issues.append(
                    TessellationIssue(
                        code,
                        'error',
                        message,
                        examples=((cid,) if cid >= 0 else ()),
                    )
                )
        elif not empty:
            assert measure.value is not None
            valid_areas.append(measure.value)

    sum_area = stable_measure_sum(valid_areas)
    domain_area_valid = bool(np.isfinite(dom_area) and dom_area > 0.0)
    if not domain_area_valid:
        issues.append(
            TessellationIssue('DOMAIN_AREA', 'error', 'Domain area is non-positive')
        )
        dom_area = max(dom_area, 0.0)

    area_tol = max(float(area_tol_abs), float(area_tol_rel) * dom_area)
    diff = sum_area - dom_area
    aggregate_overflow = bool(np.isposinf(sum_area))
    ok_area = bool(
        measures_valid
        and domain_area_valid
        and not aggregate_overflow
        and abs(diff) <= area_tol
    )
    gap = max(0.0, dom_area - sum_area)
    overlap = max(0.0, sum_area - dom_area)
    if measures_valid and domain_area_valid:
        if aggregate_overflow:
            issues.append(
                TessellationIssue(
                    'OVERLAP',
                    'error',
                    'Sum of cell areas exceeds the finite representable range',
                )
            )
        elif not ok_area:
            if gap > area_tol:
                issues.append(
                    TessellationIssue(
                        'GAP',
                        'error',
                        f'Sum of cell areas is smaller than domain area by {gap:g}',
                    )
                )
            if overlap > area_tol:
                issues.append(
                    TessellationIssue(
                        'OVERLAP',
                        'error',
                        f'Sum of cell areas exceeds domain area by {overlap:g}',
                    )
                )

    missing_ids: list[int] = []
    if expected_ids_array is not None:
        classification = classify_expected_ids(
            expected_ids_array.tolist(),
            present_ids,
            mode=mode,
        )
        missing_ids = list(classification.missing_ids)
        empty_ids.extend(classification.hidden_ids)
        if classification.issue_code is not None:
            subject = (
                'hidden power'
                if classification.issue_code == 'HIDDEN_IDS'
                else 'expected'
            )
            issues.append(
                TessellationIssue(
                    classification.issue_code,
                    classification.severity,
                    f'{len(missing_ids)} {subject} ids are absent from output',
                    examples=tuple(missing_ids[:10]),
                )
            )

    if mark_edges:
        reset_owned_annotations(
            (
                edge
                for cell in cells
                for edge in (cell.get('edges') or [])
            ),
            fields=('orphan', 'reciprocal_missing', 'reciprocal_mismatch'),
        )

    edge_shift_available = False
    reciprocity_checked = False
    relevant_edges: list[dict[str, Any]] = []
    invalid_adjacency: list[tuple[int, int, int]] = []
    periodic_axes = geometry2d(domain).periodic_axes
    for cell in cells:
        for edge_index, edge in enumerate(cell.get('edges') or []):
            adjacent = int(edge.get('adjacent_cell', -999999))
            if adjacent >= 0:
                relevant_edges.append(edge)
            elif (
                adjacent not in (-1, -2, -3, -4)
                or periodic_axes[(-adjacent - 1) // 2]
            ):
                invalid_adjacency.append(
                    (int(cell.get('id', -1)), edge_index, adjacent)
                )
    if invalid_adjacency:
        issues.append(
            TessellationIssue(
                'INVALID_EDGE_ADJACENCY', 'error',
                f'{len(invalid_adjacency)} negative edge references are not known '
                'planar wall sides on nonperiodic axes',
                examples=tuple(invalid_adjacency[:10]),
            )
        )
    n_edges_total = len(relevant_edges)
    n_orphan = 0
    n_mismatch = 0

    if _is_periodic_domain(domain) and check_reciprocity:
        edge_shift_available = all(
            'edges' in cell or bool(cell.get('empty', False)) for cell in cells
        ) and all('adjacent_shift' in edge for edge in relevant_edges)

        if not edge_shift_available:
            issues.append(
                TessellationIssue(
                    'NO_EDGE_SHIFTS',
                    reciprocity_issue_severity(
                        required=reciprocity_required,
                        missing_shifts=True,
                    ),
                    'Edge shifts are not available; set return_edge_shifts=True '
                    'to enable reciprocity diagnostics',
                )
            )
        else:
            reciprocity_checked = True

            geom = geometry2d(domain)
            avec, bvec = geom.lattice_vectors_cart

            cell_by_id: dict[int, dict[str, Any]] = {}
            for cell in cells:
                cid = int(cell.get('id', -1))
                if cid >= 0:
                    cell_by_id[cid] = cell

            L = _characteristic_length(domain)
            if (line_offset_tol is None or line_angle_tol is None) and (
                float(L) < 1e-3 or float(L) > 1e9
            ):
                warnings.warn(
                    'analyze_tessellation is using default periodic line-mismatch '
                    'tolerances derived from the planar domain length scale '
                    f'(L≈{float(L):.3g}). For very small/large units this may '
                    'be too strict/too loose. Consider rescaling inputs or '
                    'passing line_offset_tol=... and/or line_angle_tol=... '
                    'explicitly.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            off_tol = (1e-6 * L) if line_offset_tol is None else float(line_offset_tol)
            ang_tol = 1e-6 if line_angle_tol is None else float(line_angle_tol)
            eps_f = float(np.finfo(float).eps)
            coord_tol = float(max(1000.0 * off_tol, 128.0 * eps_f * L))

            def _skey(s: Any) -> tuple[int, int]:
                return int(s[0]), int(s[1])

            edge_map: dict[
                tuple[int, int, tuple[int, int]], list[tuple[int, int]]
            ] = {}
            for cell in cells:
                i = int(cell.get('id', -1))
                if i < 0:
                    continue
                edges = cell.get('edges') or []
                for ei, edge in enumerate(edges):
                    j = int(edge.get('adjacent_cell', -999999))
                    if j < 0:
                        continue
                    s = _skey(edge['adjacent_shift'])
                    key = (i, j, s)
                    edge_map.setdefault(key, []).append((i, ei))

            def _edge_segment(
                cell_id: int,
                edge_index: int,
                *,
                translate: np.ndarray | None = None,
            ) -> np.ndarray | None:
                cell = cell_by_id.get(cell_id)
                if cell is None:
                    return None
                verts = np.asarray(cell.get('vertices', []), dtype=np.float64)
                if verts.size == 0:
                    verts = verts.reshape((0, 2))
                edges = cell.get('edges') or []
                if edge_index < 0 or edge_index >= len(edges):
                    return None
                idx = np.asarray(edges[edge_index].get('vertices', []), dtype=np.int64)
                if idx.shape != (2,) or verts.size == 0:
                    return None
                vv = verts[idx]
                if translate is not None:
                    vv = vv + translate.reshape(1, 2)
                return vv

            checked: set[tuple[int, int, tuple[int, int]]] = set()
            examples_missing: list[tuple[int, int, tuple[int, int]]] = []
            examples_mismatch: list[tuple[int, int, tuple[int, int]]] = []

            for (i, j, s), locations in edge_map.items():
                if (i, j, s) in checked:
                    continue
                recip = (j, i, (-s[0], -s[1]))
                checked.add((i, j, s))
                checked.add(recip)
                if recip not in edge_map:
                    n_orphan += len(locations)
                    if len(examples_missing) < 10:
                        examples_missing.append((i, j, s))
                    if mark_edges:
                        for ci, ei in locations:
                            cell_by_id[ci]['edges'][ei]['orphan'] = True
                            cell_by_id[ci]['edges'][ei]['reciprocal_missing'] = True
                    continue

                if not check_line_mismatch:
                    continue

                reciprocal_locations = edge_map[recip]
                T = s[0] * avec + s[1] * bvec
                segments1 = [_edge_segment(ci, ei) for ci, ei in locations]
                segments2 = [
                    _edge_segment(cj, ej, translate=T)
                    for cj, ej in reciprocal_locations
                ]
                if any(segment is None for segment in segments1 + segments2):
                    continue
                union1 = [segment for segment in segments1 if segment is not None]
                union2 = [segment for segment in segments2 if segment is not None]
                matching = all(
                    _segment_union_covers(
                        source, target, offset_tol=off_tol,
                        angle_tol=ang_tol, coordinate_tol=coord_tol,
                    )
                    for source, target in ((union1, union2), (union2, union1))
                )
                if not matching:
                    n_mismatch += 1
                    if len(examples_mismatch) < 10:
                        examples_mismatch.append((i, j, s))
                    if mark_edges:
                        for ci, ei in locations + reciprocal_locations:
                            cell_by_id[ci]['edges'][ei]['reciprocal_mismatch'] = True

            if n_orphan:
                issues.append(
                    TessellationIssue(
                        'MISSING_RECIPROCAL',
                        reciprocity_issue_severity(
                            required=reciprocity_required,
                        ),
                        f'{n_orphan} raw edge occurrences have no reciprocal class',
                        examples=tuple(examples_missing),
                    )
                )
            if n_mismatch:
                issues.append(
                    TessellationIssue(
                        'RECIPROCAL_MISMATCH',
                        reciprocity_issue_severity(
                            required=reciprocity_required,
                        ),
                        f'{n_mismatch} reciprocal edge class pairs have '
                        'disagreeing numerical segment unions',
                        examples=tuple(examples_mismatch),
                    )
                )

    reciprocity_requested = bool(_is_periodic_domain(domain) and check_reciprocity)
    ok_recip = bool(
        not reciprocity_requested
        or (
            reciprocity_checked
            and not invalid_adjacency
            and n_orphan == 0
            and n_mismatch == 0
        )
    )

    ok = diagnostics_ok(issues)
    if not ok and mode is not None:
        issues.append(
            TessellationIssue('MODE', 'info', f'Diagnostics produced for mode={mode!r}')
        )

    return TessellationDiagnostics(
        domain_area=float(dom_area),
        sum_cell_area=float(sum_area),
        area_ratio=float(sum_area / dom_area) if dom_area > 0 else 0.0,
        area_gap=float(gap),
        area_overlap=float(overlap),
        n_sites_expected=int(
            expected_ids_array.size
            if expected_ids_array is not None
            else len(set(present_ids))
        ),
        n_cells_returned=int(len(cells)),
        missing_ids=tuple(int(x) for x in missing_ids),
        empty_ids=tuple(int(x) for x in sorted(set(empty_ids))),
        edge_shift_available=bool(edge_shift_available),
        reciprocity_checked=bool(reciprocity_checked),
        n_edges_total=int(n_edges_total),
        n_edges_orphan=int(n_orphan),
        n_edges_mismatched=int(n_mismatch),
        issues=tuple(issues),
        ok_area=bool(ok_area),
        ok_reciprocity=bool(ok_recip),
        ok=bool(ok),
    )


def validate_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Domain2D,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    level: Literal['basic', 'strict'] = 'basic',
    require_reciprocity: bool | None = None,
    area_tol_rel: float = 1e-8,
    area_tol_abs: float = 1e-12,
    line_offset_tol: float | None = None,
    line_angle_tol: float | None = None,
    mark_edges: bool | None = None,
) -> TessellationDiagnostics:
    """Validate planar tessellation sanity using the final diagnostic policy.

    Optional reciprocity inspection emits warning/info findings, while required
    reciprocity emits errors. Strict validation raises exactly when the returned
    diagnostics have ``ok=False``. Like :func:`analyze_tessellation`, this checks
    raw-record and numerical consistency, not exact N/E/S contact or native
    collapse. Repeated labels do not require equal occurrence counts.
    """

    level = require_string_choice(
        level,
        name='level',
        choices=('basic', 'strict'),
    )
    if mode is not None:
        mode = require_string_choice(
            mode,
            name='mode',
            choices=('standard', 'power'),
        )

    require_reciprocity = require_optional_bool(
        require_reciprocity,
        name='require_reciprocity',
    )
    mark_edges = require_optional_bool(mark_edges, name='mark_edges')

    periodic = _is_periodic_domain(domain)
    if require_reciprocity is None:
        require_reciprocity = bool(periodic)
    if mark_edges is None:
        mark_edges = bool(periodic)

    diag = _analyze_tessellation(
        cells,
        domain,
        expected_ids=expected_ids,
        mode=mode,
        area_tol_rel=area_tol_rel,
        area_tol_abs=area_tol_abs,
        check_reciprocity=bool(periodic),
        reciprocity_required=bool(require_reciprocity),
        check_line_mismatch=bool(periodic),
        line_offset_tol=line_offset_tol,
        line_angle_tol=line_angle_tol,
        mark_edges=mark_edges,
    )

    if level == 'strict' and not diag.ok:
        error = next(
            (issue for issue in diag.issues if issue.severity == 'error'),
            None,
        )
        if error is None:  # pragma: no cover - guarded by severity-complete policy
            message = 'Tessellation validation failed'
        else:
            message = (
                f'Tessellation validation failed ({error.code}): '
                f'{error.message}'
            )
        raise TessellationError(message, diag)

    return diag
