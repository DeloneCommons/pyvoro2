"""Strict validation utilities for planar normalization outputs."""

from __future__ import annotations

from .._internal.ghost import reject_ghost_records

from dataclasses import dataclass
import sys
from typing import Any, Literal

from .._internal.planar.domain_geometry import geometry2d
from .._internal.planar.normalization_context import context_for
from .._internal.normalization_proof import ProofFailure
from .._internal.normalization import require_global_vertex_ids
from .._internal.tessellation_diagnostics import diagnostics_ok
from .._internal.validation import (
    require_bool,
    require_nonnegative_index,
    require_string,
    require_string_choice,
)
from .domains import Box, RectangularCell
from .normalize import (
    NormalizedTopology, NormalizedVertices, _prepare_topology_cells,
    _prepare_vertex_mappings, _canon_edge, _canon_cell_pair,
)


Domain2D = Box | RectangularCell


@dataclass(frozen=True, slots=True)
class NormalizationIssue:
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


def _precondition_error(normalized, domain, level, code, message):
    """A failed precondition checks no representation or semantic obligation."""
    diag = NormalizationDiagnostics(
        n_cells=len(normalized.cells),
        n_global_vertices=len(normalized.global_vertices),
        n_global_edges=(len(normalized.global_edges)
                        if isinstance(normalized, NormalizedTopology) else None),
        is_periodic_domain=_is_periodic_domain(domain),
        fully_periodic_domain=_fully_periodic(domain), has_wall_edges=False,
        n_vertex_edge_shift_mismatch=0, n_edge_vertex_set_mismatch=0,
        n_vertices_low_incidence=0, n_cells_bad_polygon=0,
        issues=(NormalizationIssue(code, 'error', message),),
        ok_vertex_edge_shift=False, ok_edge_vertex_sets=False,
        ok_incidence=False, ok_polygon=False, ok=False)
    if level == 'strict':
        raise NormalizationError(f'{code}: {message}', diag)
    return diag


def _check_global_mappings(normalized, prepared):
    if not isinstance(normalized, NormalizedTopology):
        return
    for item, cell in zip(prepared, normalized.cells):
        ids = require_global_vertex_ids(
            cell.get('edge_global_id'), name='edge_global_id',
            n_vertices=len(item['edges']),
            n_global_vertices=len(normalized.global_edges))
        for edge, eid in zip(item['edges'], ids):
            pair = _canon_cell_pair(item['id'], edge['adjacent'], edge['shift'])
            u, v = edge['vertices']
            _key, rep = _canon_edge((item['gids'][u], item['vertex_shifts'][u]),
                                    (item['gids'][v], item['vertex_shifts'][v]))
            expected = dict(cells=(pair[0], pair[3]),
                            cell_shifts=((0, 0), (pair[4], pair[5])),
                            vertices=(rep[0][0], rep[1][0]),
                            vertex_shifts=((0, 0), (rep[1][1], rep[1][2])))
            actual = normalized.global_edges[eid]
            if not isinstance(actual, dict) or any(
                    tuple(tuple(v) if isinstance(v, (list, tuple)) else v
                          for v in actual.get(key, ())) != value
                    for key, value in expected.items()):
                raise ValueError('edge_global_id does not preserve raw provenance '
                                 'and endpoints')


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
    """Validate enabled, applicable raw representation checks.

    Live compute-owned views additionally require their bound audit, identity
    lifts, protected distinctions and occurrence obligations. Eligible retained
    artifacts are exempt only from positive reciprocity; their records remain
    structurally checked. Standalone/copied views have no such exemptions.
    Stale retained authority is an explicit error, never a numerical fallback.
    Strict success does not certify exact S reconstruction or artifact support.
    """

    level = require_string_choice(
        level,
        name='level',
        choices=('basic', 'strict'),
    )

    check_vertex_edge_shift = require_bool(
        check_vertex_edge_shift,
        name='check_vertex_edge_shift',
    )
    check_edge_vertex_sets = require_bool(
        check_edge_vertex_sets,
        name='check_edge_vertex_sets',
    )
    check_incidence = require_bool(check_incidence, name='check_incidence')
    check_polygon = require_bool(check_polygon, name='check_polygon')
    max_examples = require_nonnegative_index(
        max_examples,
        name='max_examples',
        maximum=sys.maxsize,
    )
    example_probe_limit = max(max_examples, 1)
    reject_ghost_records(normalized.cells)

    try:
        context = context_for(normalized, domain)
        periodic = _is_periodic_domain(domain)
        if (isinstance(normalized, NormalizedTopology) or check_polygon
                or (periodic and (check_vertex_edge_shift or check_edge_vertex_sets))):
            _vertices, prepared = _prepare_topology_cells(
                normalized, domain=domain, periodic=periodic)
        else:
            _vertices, prepared = _prepare_vertex_mappings(normalized, domain=domain)
        _check_global_mappings(normalized, prepared)
    except ProofFailure as exc:
        return _precondition_error(normalized, domain, level, exc.code, str(exc))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return _precondition_error(normalized, domain, level,
                                   'INVALID_NORMALIZED_MAPPING', str(exc))

    cells = list(normalized.cells)
    reject_ghost_records(cells)
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

        for position, cell in enumerate(cells):
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

            for slot, edge in enumerate(edges):
                if context is not None and (position, slot) in context.artifacts:
                    continue
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
                            examples=((cid, j),)[:max_examples],
                        )
                    )
                    continue

                s = _as_shift(edge.get('adjacent_shift', (0, 0)))
                cj = cell_by_id.get(j)
                if cj is None:
                    if len(missing_neighbor_cells) < example_probe_limit:
                        missing_neighbor_cells.append((cid, j, s))
                    continue
                map_j = gid_shift_by_cell.get(j)
                if map_j is None:
                    if len(missing_neighbor_cells) < example_probe_limit:
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
                        if len(missing_shared_vertex) < example_probe_limit:
                            missing_shared_vertex.append((cid, j, s, gid))
                        continue
                    expected_set = {(sj[0] + s[0], sj[1] + s[1]) for sj in sj_set}
                    if si not in expected_set:
                        n_ves_mismatch += 1
                        if len(examples) < example_probe_limit:
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
                    examples=tuple(missing_neighbor_cells[:max_examples]),
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
                    examples=tuple(missing_shared_vertex[:max_examples]),
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
                    examples=tuple(examples[:max_examples]),
                )
            )

    n_evt_mismatch = 0
    if periodic and check_edge_vertex_sets:
        examples: list[tuple[int, int, tuple[int, int]]] = []
        classes = {}
        for position, cell in enumerate(cells):
            cid = int(cell.get('id', -1))
            if cid < 0 or bool(cell.get('empty', False)):
                continue
            gids = cell.get('vertex_global_id')
            if gids is None:
                continue
            edges = cell.get('edges') or []
            for slot, edge in enumerate(edges):
                if context is not None and (position, slot) in context.artifacts:
                    continue
                j = int(edge.get('adjacent_cell', -1))
                if j < 0 or 'adjacent_shift' not in edge:
                    continue
                s = _as_shift(edge.get('adjacent_shift', (0, 0)))
                points = classes.setdefault((cid, j, s), set())
                points.update((int(gids[v]), _as_shift(cell['vertex_shift'][v]))
                              for v in _iter_edge_vertex_indices(edge))
        for (cid, j, s), points in classes.items():
            if j not in cell_by_id:
                continue
            peer = classes.get((j, cid, (-s[0], -s[1])), set())
            transported = {(gid, (shift[0] + s[0], shift[1] + s[1]))
                           for gid, shift in peer}
            if points != transported:
                n_evt_mismatch += 1
                if len(examples) < example_probe_limit:
                    examples.append((cid, j, s))
        if examples:
            issues.append(
                NormalizationIssue(
                    code='EDGE_VERTEX_SET_MISMATCH',
                    severity='error',
                    message=(
                        'Reciprocal periodic edges do not reference the same set '
                        'of image-qualified global vertices.'
                    ),
                    examples=tuple(examples[:max_examples]),
                )
            )

    n_vertices_low_incidence = 0
    if (
        isinstance(normalized, NormalizedTopology)
        and check_incidence
        and fully_periodic
        and not has_wall_edges
    ):
        inc: dict[int, set[tuple]] = {i: set() for i in range(n_global_vertices)}
        for eid, edge in enumerate(normalized.global_edges):
            for gid, shift in zip(edge.get('vertices', ()),
                                  edge.get('vertex_shifts', ())):
                # A periodic loop meets the vertex through both endpoint
                # images; distinct lifts must not collapse to one bare eid.
                inc[int(gid)].add((eid, tuple(-int(value) for value in shift)))
        examples: list[tuple[int, int]] = []
        for gid, eids in inc.items():
            if len(eids) < 3:
                n_vertices_low_incidence += 1
                if len(examples) < example_probe_limit:
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
                    examples=tuple(examples[:max_examples]),
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
                if len(examples) < example_probe_limit:
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
                    examples=tuple(examples[:max_examples]),
                )
            )

    ok_vertex_edge_shift = n_ves_mismatch == 0
    ok_edge_vertex_sets = n_evt_mismatch == 0
    ok_incidence = n_vertices_low_incidence == 0
    ok_polygon = n_cells_bad_polygon == 0
    ok = diagnostics_ok(issues)

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
