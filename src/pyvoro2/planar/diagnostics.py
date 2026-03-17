"""Planar tessellation diagnostics and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import warnings

import numpy as np

from ._domain_geometry import geometry2d
from .domains import Box, RectangularCell


Domain2D = Box | RectangularCell


@dataclass(frozen=True, slots=True)
class TessellationIssue:
    code: str
    severity: Literal['info', 'warning', 'error']
    message: str
    examples: tuple[Any, ...] = ()


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


def _line_from_vertices(v: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return (unit normal, d) for the line n·x = d, or None if degenerate."""

    if v.shape[0] < 2:
        return None
    dv = v[1] - v[0]
    nn = float(np.linalg.norm(dv))
    if nn == 0.0:
        return None
    tangent = dv / nn
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    d = float(np.mean(v @ normal))
    return normal, d


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
    """Analyze planar tessellation sanity and optionally annotate edges."""

    issues: list[TessellationIssue] = []

    dom_area = _domain_area(domain)
    sum_area = 0.0
    empty_ids: list[int] = []
    present_ids: list[int] = []
    for cell in cells:
        cid = int(cell.get('id', -1))
        if cid >= 0:
            present_ids.append(cid)
        if bool(cell.get('empty', False)):
            if cid >= 0:
                empty_ids.append(cid)
            continue
        try:
            sum_area += float(cell.get('area', 0.0))
        except Exception:
            pass

    if dom_area <= 0.0:
        issues.append(
            TessellationIssue('DOMAIN_AREA', 'error', 'Domain area is non-positive')
        )
        dom_area = max(dom_area, 0.0)

    area_tol = max(float(area_tol_abs), float(area_tol_rel) * dom_area)
    diff = sum_area - dom_area
    ok_area = abs(diff) <= area_tol
    gap = max(0.0, dom_area - sum_area)
    overlap = max(0.0, sum_area - dom_area)
    if not ok_area:
        if gap > area_tol:
            issues.append(
                TessellationIssue(
                    'GAP',
                    'warning',
                    f'Sum of cell areas is smaller than domain area by {gap:g}',
                )
            )
        if overlap > area_tol:
            issues.append(
                TessellationIssue(
                    'OVERLAP',
                    'warning',
                    f'Sum of cell areas exceeds domain area by {overlap:g}',
                )
            )

    missing_ids: list[int] = []
    if expected_ids is not None:
        exp = {int(x) for x in expected_ids}
        missing_ids = sorted(exp - set(present_ids))
        if missing_ids:
            issues.append(
                TessellationIssue(
                    'MISSING_IDS',
                    'warning',
                    f'{len(missing_ids)} expected ids are missing from output',
                    examples=tuple(missing_ids[:10]),
                )
            )

    edge_shift_available = False
    reciprocity_checked = False
    n_edges_total = 0
    n_orphan = 0
    n_mismatch = 0

    if _is_periodic_domain(domain) and check_reciprocity:
        for cell in cells:
            for edge in cell.get('edges') or []:
                if (
                    int(edge.get('adjacent_cell', -999999)) >= 0
                    and 'adjacent_shift' in edge
                ):
                    edge_shift_available = True
                    break
            if edge_shift_available:
                break

        if not edge_shift_available:
            issues.append(
                TessellationIssue(
                    'NO_EDGE_SHIFTS',
                    'info',
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
            size_tol = float(max(1000.0 * off_tol, 128.0 * eps_f * L))

            def _skey(s: Any) -> tuple[int, int]:
                return int(s[0]), int(s[1])

            edge_map: dict[tuple[int, int, tuple[int, int]], tuple[int, int]] = {}
            for cell in cells:
                i = int(cell.get('id', -1))
                if i < 0:
                    continue
                verts = np.asarray(cell.get('vertices', []), dtype=np.float64)
                if verts.size == 0:
                    verts = verts.reshape((0, 2))
                edges = cell.get('edges') or []
                for ei, edge in enumerate(edges):
                    j = int(edge.get('adjacent_cell', -999999))
                    if j < 0:
                        continue
                    s = _skey(edge.get('adjacent_shift', (0, 0)))
                    n_edges_total += 1

                    idx = np.asarray(edge.get('vertices', []), dtype=np.int64)
                    if idx.shape != (2,) or verts.size == 0:
                        continue
                    vv = verts[idx]
                    size = float(np.linalg.norm(vv[1] - vv[0]))
                    if size < size_tol:
                        continue

                    key = (i, j, s)
                    if key in edge_map:
                        issues.append(
                            TessellationIssue(
                                'DUPLICATE_DIRECTED_EDGE',
                                'error',
                                f'Duplicate directed edge key encountered: {key}',
                            )
                        )
                    else:
                        edge_map[key] = (i, ei)

                    if mark_edges:
                        edge.setdefault('orphan', False)
                        edge.setdefault('reciprocal_mismatch', False)
                        edge.setdefault('reciprocal_missing', False)

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

            for (i, j, s), loc in list(edge_map.items()):
                if (i, j, s) in checked:
                    continue
                recip = (j, i, (-s[0], -s[1]))
                checked.add((i, j, s))
                checked.add(recip)
                if recip not in edge_map:
                    n_orphan += 1
                    if len(examples_missing) < 10:
                        examples_missing.append((i, j, s))
                    if mark_edges:
                        ci, ei = loc
                        try:
                            cell_by_id[ci]['edges'][ei]['orphan'] = True
                            cell_by_id[ci]['edges'][ei]['reciprocal_missing'] = True
                        except Exception:
                            pass
                    continue

                if not check_line_mismatch:
                    continue

                (ci, ei) = loc
                (cj, ej) = edge_map[recip]
                T = s[0] * avec + s[1] * bvec
                seg1 = _edge_segment(ci, ei)
                seg2 = _edge_segment(cj, ej, translate=T)
                if seg1 is None or seg2 is None:
                    continue
                line1 = _line_from_vertices(seg1)
                line2 = _line_from_vertices(seg2)
                if line1 is None or line2 is None:
                    continue

                n1, d1 = line1
                n2, d2 = line2
                dot = float(np.dot(n1, n2))
                if dot < 0.0:
                    n2 = -n2
                    d2 = -d2
                    dot = -dot
                dot = max(-1.0, min(1.0, dot))
                ang = float(np.arccos(dot))
                off = float(abs(d1 - d2))
                dist_same = max(
                    float(np.linalg.norm(seg1[0] - seg2[0])),
                    float(np.linalg.norm(seg1[1] - seg2[1])),
                )
                dist_flip = max(
                    float(np.linalg.norm(seg1[0] - seg2[1])),
                    float(np.linalg.norm(seg1[1] - seg2[0])),
                )
                coord_mismatch = min(dist_same, dist_flip)

                if ang > ang_tol or off > off_tol or coord_mismatch > size_tol:
                    n_mismatch += 1
                    if len(examples_mismatch) < 10:
                        examples_mismatch.append((i, j, s))
                    if mark_edges:
                        try:
                            cell_by_id[ci]['edges'][ei]['reciprocal_mismatch'] = True
                            cell_by_id[cj]['edges'][ej]['reciprocal_mismatch'] = True
                        except Exception:
                            pass

            if n_orphan:
                issues.append(
                    TessellationIssue(
                        'MISSING_RECIPROCAL',
                        'warning',
                        f'{n_orphan} edges are missing a reciprocal',
                        examples=tuple(examples_missing),
                    )
                )
            if n_mismatch:
                issues.append(
                    TessellationIssue(
                        'RECIPROCAL_MISMATCH',
                        'warning',
                        f'{n_mismatch} reciprocal edge pairs disagree geometrically',
                        examples=tuple(examples_mismatch),
                    )
                )

    ok_recip = True
    if reciprocity_checked:
        ok_recip = (n_orphan == 0) and (n_mismatch == 0)

    ok = ok_area and (ok_recip if reciprocity_checked else True)
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
            len(expected_ids) if expected_ids is not None else len(set(present_ids))
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
    """Validate planar tessellation sanity, optionally raising in strict mode."""

    if level not in ('basic', 'strict'):
        raise ValueError("level must be 'basic' or 'strict'")

    periodic = _is_periodic_domain(domain)
    if require_reciprocity is None:
        require_reciprocity = bool(periodic)
    if mark_edges is None:
        mark_edges = bool(periodic)

    diag = analyze_tessellation(
        cells,
        domain,
        expected_ids=expected_ids,
        mode=mode,
        area_tol_rel=float(area_tol_rel),
        area_tol_abs=float(area_tol_abs),
        check_reciprocity=bool(periodic),
        check_line_mismatch=bool(periodic),
        line_offset_tol=line_offset_tol,
        line_angle_tol=line_angle_tol,
        mark_edges=bool(mark_edges),
    )

    if level == 'strict':
        ok = bool(diag.ok_area) and (
            bool(diag.ok_reciprocity)
            if bool(require_reciprocity) and bool(diag.reciprocity_checked)
            else True
        )
        if not ok:
            raise TessellationError(
                'Tessellation validation failed: '
                f'area_ratio={diag.area_ratio:g}, '
                f'orphan_edges={diag.n_edges_orphan}, '
                f'mismatched_edges={diag.n_edges_mismatched}',
                diag,
            )

    return diag
