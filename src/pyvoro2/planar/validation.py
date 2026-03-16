"""Strict validation utilities for planar normalization outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ._domain_geometry import geometry2d
from .domains import Box, RectangularCell
from .normalize import NormalizedTopology, NormalizedVertices


Domain2D = Box | RectangularCell


@dataclass(frozen=True, slots=True)
class NormalizationIssue:
    code: str
    severity: Literal['info', 'warning', 'error']
    message: str
    examples: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizationDiagnostics:
    n_cells: int
    n_global_vertices: int
    n_global_edges: int | None
    is_periodic_domain: bool
    fully_periodic_domain: bool
    has_wall_edges: bool

    n_vertex_edge_shift_mismatch: int
    n_edge_vertex_set_mismatch: int
    n_vertices_low_incidence: int
    n_cells_bad_polygon: int

    issues: tuple[NormalizationIssue, ...]

    ok_vertex_edge_shift: bool
    ok_edge_vertex_sets: bool
    ok_incidence: bool
    ok_polygon: bool
    ok: bool


class NormalizationError(ValueError):
    """Raised when strict planar normalization validation fails."""

    def __init__(self, message: str, diagnostics: NormalizationDiagnostics):
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


def _as_shift(s: Any) -> tuple[int, int]:
    return int(s[0]), int(s[1])


def _is_periodic_domain(domain: Domain2D) -> bool:
    return bool(geometry2d(domain).has_any_periodic_axis)


def _fully_periodic(domain: Domain2D) -> bool:
    geom = geometry2d(domain)
    return bool(all(geom.periodic_axes))


def _iter_edge_vertex_indices(edge: dict[str, Any]) -> list[int]:
    idx = edge.get('vertices')
    if idx is None:
        return []
    return [int(x) for x in idx]


def validate_normalized_topology(
    normalized: NormalizedVertices | NormalizedTopology,
    domain: Domain2D,
    *,
    level: Literal['basic', 'strict'] = 'basic',
    check_vertex_edge_shift: bool = True,
    check_edge_vertex_sets: bool = True,
    check_incidence: bool = True,
    check_polygon: bool = True,
    max_examples: int = 10,
) -> NormalizationDiagnostics:
    """Validate periodic shift and topology consistency after normalization."""

    if level not in ('basic', 'strict'):
        raise ValueError("level must be 'basic' or 'strict'")

    cells = list(normalized.cells)
    n_cells = len(cells)
    n_global_vertices = int(normalized.global_vertices.shape[0])
    n_global_edges: int | None = None
    if isinstance(normalized, NormalizedTopology):
        n_global_edges = len(normalized.global_edges)

    periodic = _is_periodic_domain(domain)
    fully_periodic = _fully_periodic(domain)

    has_wall_edges = False
    for cell in cells:
        for edge in cell.get('edges') or []:
            if int(edge.get('adjacent_cell', -1)) < 0:
                has_wall_edges = True
                break
        if has_wall_edges:
            break

    issues: list[NormalizationIssue] = []

    cell_by_id: dict[int, dict[str, Any]] = {}
    gid_shift_by_cell: dict[int, dict[int, set[tuple[int, int]]]] = {}

    for cell in cells:
        cid = int(cell.get('id', -1))
        if cid < 0:
            continue
        cell_by_id[cid] = cell

        gids = cell.get('vertex_global_id')
        vsh = cell.get('vertex_shift')
        if gids is None or vsh is None:
            continue
        mapping: dict[int, set[tuple[int, int]]] = {}
        for k, gid in enumerate(gids):
            g = int(gid)
            s = _as_shift(vsh[k])
            mapping.setdefault(g, set()).add(s)
        gid_shift_by_cell[cid] = mapping

    n_ves_mismatch = 0
    if periodic and check_vertex_edge_shift:
        examples: list[
            tuple[
                int,
                int,
                tuple[int, int],
                int,
                tuple[tuple[int, int], ...],
                tuple[int, int],
            ]
        ] = []
        missing_neighbor_cells: list[tuple[int, int, tuple[int, int]]] = []
        missing_shared_vertex: list[tuple[int, int, tuple[int, int], int]] = []

        for cell in cells:
            cid = int(cell.get('id', -1))
            if cid < 0 or bool(cell.get('empty', False)):
                continue
            edges = cell.get('edges') or []
            gids = cell.get('vertex_global_id')
            vsh = cell.get('vertex_shift')
            if gids is None or vsh is None:
                continue

            gids_list = [int(x) for x in gids]
            vsh_list = [_as_shift(x) for x in vsh]

            for edge in edges:
                j = int(edge.get('adjacent_cell', -1))
                if j < 0:
                    continue
                if 'adjacent_shift' not in edge:
                    issues.append(
                        NormalizationIssue(
                            code='EDGE_MISSING_ADJACENT_SHIFT',
                            severity='error',
                            message=(
                                'A periodic neighbor edge is missing adjacent_shift. '
                                'Ensure compute(..., return_edge_shifts=True) was used.'
                            ),
                            examples=((cid, j),),
                        )
                    )
                    continue

                s = _as_shift(edge.get('adjacent_shift', (0, 0)))
                cj = cell_by_id.get(j)
                if cj is None:
                    if len(missing_neighbor_cells) < max_examples:
                        missing_neighbor_cells.append((cid, j, s))
                    continue
                map_j = gid_shift_by_cell.get(j)
                if map_j is None:
                    if len(missing_neighbor_cells) < max_examples:
                        missing_neighbor_cells.append((cid, j, s))
                    continue

                for lv in _iter_edge_vertex_indices(edge):
                    if lv < 0 or lv >= len(gids_list):
                        continue
                    gid = gids_list[lv]
                    si = vsh_list[lv]
                    sj_set = map_j.get(gid)
                    if not sj_set:
                        n_ves_mismatch += 1
                        if len(missing_shared_vertex) < max_examples:
                            missing_shared_vertex.append((cid, j, s, gid))
                        continue
                    expected_set = {(sj[0] + s[0], sj[1] + s[1]) for sj in sj_set}
                    if si not in expected_set:
                        n_ves_mismatch += 1
                        if len(examples) < max_examples:
                            examples.append((cid, gid, si, j, tuple(sorted(sj_set)), s))

        if missing_neighbor_cells:
            issues.append(
                NormalizationIssue(
                    code='MISSING_NEIGHBOR_CELL',
                    severity='warning',
                    message=(
                        'Some reciprocal neighbor cells are missing from the '
                        'cell list.'
                    ),
                    examples=tuple(missing_neighbor_cells),
                )
            )
        if missing_shared_vertex:
            issues.append(
                NormalizationIssue(
                    code='MISSING_SHARED_VERTEX',
                    severity='error',
                    message=(
                        'A reciprocal neighboring cell does not contain a shared '
                        'global vertex referenced by a periodic edge.'
                    ),
                    examples=tuple(missing_shared_vertex),
                )
            )
        if examples:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_EDGE_SHIFT_MISMATCH',
                    severity='error',
                    message=(
                        'vertex_shift values disagree with edge adjacent_shift across '
                        'reciprocal neighboring cells.'
                    ),
                    examples=tuple(examples),
                )
            )

    n_evt_mismatch = 0
    if periodic and check_edge_vertex_sets:
        examples: list[tuple[int, int, tuple[int, int]]] = []
        for cell in cells:
            cid = int(cell.get('id', -1))
            if cid < 0 or bool(cell.get('empty', False)):
                continue
            gids = cell.get('vertex_global_id')
            if gids is None:
                continue
            edges = cell.get('edges') or []
            for edge in edges:
                j = int(edge.get('adjacent_cell', -1))
                if j < 0 or 'adjacent_shift' not in edge:
                    continue
                s = _as_shift(edge.get('adjacent_shift', (0, 0)))
                cj = cell_by_id.get(j)
                if cj is None:
                    continue
                gids_here = tuple(
                    sorted(int(gids[v]) for v in _iter_edge_vertex_indices(edge))
                )
                found = False
                for edge_j in cj.get('edges') or []:
                    if int(edge_j.get('adjacent_cell', -1)) != cid:
                        continue
                    if _as_shift(edge_j.get('adjacent_shift', (0, 0))) != (
                        -s[0],
                        -s[1],
                    ):
                        continue
                    gids_j = cj.get('vertex_global_id')
                    if gids_j is None:
                        continue
                    peer = tuple(
                        sorted(
                            int(gids_j[v])
                            for v in _iter_edge_vertex_indices(edge_j)
                        )
                    )
                    if peer == gids_here:
                        found = True
                        break
                if not found:
                    n_evt_mismatch += 1
                    if len(examples) < max_examples:
                        examples.append((cid, j, s))
        if examples:
            issues.append(
                NormalizationIssue(
                    code='EDGE_VERTEX_SET_MISMATCH',
                    severity='error',
                    message=(
                        'Reciprocal periodic edges do not reference the same set '
                        'of global vertex ids.'
                    ),
                    examples=tuple(examples),
                )
            )

    n_vertices_low_incidence = 0
    if (
        isinstance(normalized, NormalizedTopology)
        and check_incidence
        and fully_periodic
        and not has_wall_edges
    ):
        inc: dict[int, set[int]] = {i: set() for i in range(n_global_vertices)}
        for eid, edge in enumerate(normalized.global_edges):
            for gid in edge.get('vertices', ()):
                inc[int(gid)].add(eid)
        examples: list[tuple[int, int]] = []
        for gid, eids in inc.items():
            if len(eids) < 3:
                n_vertices_low_incidence += 1
                if len(examples) < max_examples:
                    examples.append((gid, len(eids)))
        if examples:
            issues.append(
                NormalizationIssue(
                    code='LOW_VERTEX_INCIDENCE',
                    severity='warning',
                    message=(
                        'Some global vertices have low edge incidence in a fully '
                        'periodic planar tessellation.'
                    ),
                    examples=tuple(examples),
                )
            )

    n_cells_bad_polygon = 0
    if check_polygon:
        examples: list[tuple[int, int, int]] = []
        for cell in cells:
            cid = int(cell.get('id', -1))
            if cid < 0 or bool(cell.get('empty', False)):
                continue
            verts = cell.get('vertices') or []
            edges = cell.get('edges') or []
            nv = len(verts)
            ne = len(edges)
            if nv != ne:
                n_cells_bad_polygon += 1
                if len(examples) < max_examples:
                    examples.append((cid, nv, ne))
        if examples:
            issues.append(
                NormalizationIssue(
                    code='BAD_POLYGON_COUNT',
                    severity='warning',
                    message=(
                        'Some cells do not satisfy the expected planar polygon '
                        'count V == E.'
                    ),
                    examples=tuple(examples),
                )
            )

    ok_vertex_edge_shift = n_ves_mismatch == 0
    ok_edge_vertex_sets = n_evt_mismatch == 0
    ok_incidence = n_vertices_low_incidence == 0
    ok_polygon = n_cells_bad_polygon == 0
    ok = ok_vertex_edge_shift and ok_edge_vertex_sets and ok_incidence and ok_polygon

    diag = NormalizationDiagnostics(
        n_cells=int(n_cells),
        n_global_vertices=int(n_global_vertices),
        n_global_edges=(int(n_global_edges) if n_global_edges is not None else None),
        is_periodic_domain=bool(periodic),
        fully_periodic_domain=bool(fully_periodic),
        has_wall_edges=bool(has_wall_edges),
        n_vertex_edge_shift_mismatch=int(n_ves_mismatch),
        n_edge_vertex_set_mismatch=int(n_evt_mismatch),
        n_vertices_low_incidence=int(n_vertices_low_incidence),
        n_cells_bad_polygon=int(n_cells_bad_polygon),
        issues=tuple(issues),
        ok_vertex_edge_shift=bool(ok_vertex_edge_shift),
        ok_edge_vertex_sets=bool(ok_edge_vertex_sets),
        ok_incidence=bool(ok_incidence),
        ok_polygon=bool(ok_polygon),
        ok=bool(ok),
    )

    if level == 'strict' and not diag.ok:
        raise NormalizationError(
            'Normalized planar topology validation failed: '
            f'vertex_edge_shift_mismatch={diag.n_vertex_edge_shift_mismatch}, '
            f'edge_vertex_set_mismatch={diag.n_edge_vertex_set_mismatch}, '
            f'low_incidence_vertices={diag.n_vertices_low_incidence}, '
            f'bad_polygon_cells={diag.n_cells_bad_polygon}',
            diag,
        )

    return diag
