"""Strict validation utilities.

This module provides *post-hoc* validators that can be used to sanity check:

1) tessellation outputs (via :func:`pyvoro2.validate_tessellation`, implemented
   as a thin wrapper around :func:`pyvoro2.analyze_tessellation`), and
2) topology/normalization outputs (via :func:`validate_normalized_topology`).

The core goal is to turn subtle periodic bookkeeping mistakes into explicit,
actionable errors.
"""

from __future__ import annotations

from ._internal.ghost import reject_ghost_records

from dataclasses import dataclass
import sys
from typing import Any, Literal

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.native_runtime import checked_call, checked_tuple
from ._internal.spatial.domain_utils import is_periodic_domain
from ._internal.tessellation_diagnostics import diagnostics_ok
from ._internal.normalization import (
    require_global_vertex_ids, require_local_vertex_indices,
)
from ._internal.validation import (
    require_bool,
    require_nonnegative_index,
    require_string,
    require_string_choice,
)
from .normalize import (
    NormalizedVertices, NormalizedTopology, _prepare_topology_cells,
    _canon_edge, _canon_face_pair,
)


Domain = Box | OrthorhombicCell | PeriodicCell


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
    n_global_faces: int | None
    is_periodic_domain: bool
    fully_periodic_domain: bool
    has_wall_faces: bool

    n_vertex_face_shift_mismatch: int
    n_face_vertex_set_mismatch: int
    n_vertices_low_incidence: int
    n_edges_low_incidence: int
    n_cells_bad_euler: int

    issues: tuple[NormalizationIssue, ...]

    ok_vertex_face_shift: bool
    ok_face_vertex_sets: bool
    ok_incidence: bool
    ok_euler: bool
    ok: bool


class NormalizationError(ValueError):
    """Raised when strict normalization validation fails."""

    def __init__(self, message: str, diagnostics: NormalizationDiagnostics):
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


def _as_shift(s: Any) -> tuple[int, int, int]:
    return int(s[0]), int(s[1]), int(s[2])


def _fully_periodic(domain: Domain) -> bool:
    if isinstance(domain, PeriodicCell):
        return True
    if isinstance(domain, OrthorhombicCell):
        return all(checked_call(bool, value) for value in checked_tuple(
            checked_call(getattr, domain, 'periodic')))
    return False


def _iter_face_vertex_indices(face: dict[str, Any]) -> list[int]:
    idx = face.get('vertices')
    if idx is None:
        return []
    # Keep as Python ints for safe dict/set use.
    return [int(x) for x in idx]


def _check_consumed_mappings(normalized, domain, *, need_faces):
    """Check numerical operands, without any ideal or proof-context consumer."""
    periodic = is_periodic_domain(domain)
    candidate = normalized
    if not need_faces:
        # Vertex-only nonperiodic validation does not consume face metadata.
        candidate = NormalizedVertices(normalized.global_vertices,
                                       [dict(c, faces=[]) for c in normalized.cells])
    _vertices, prepared = _prepare_topology_cells(candidate, periodic=periodic)
    if isinstance(domain, PeriodicCell):
        axes = (True,) * 3
    elif isinstance(domain, OrthorhombicCell):
        axes = checked_tuple(checked_call(getattr, domain, 'periodic'))
    else:
        axes = (False,) * 3
    for item, cell in zip(prepared, normalized.cells):
        shifts = item['vertex_shifts']
        if any(value and not flag for shift in shifts
               for value, flag in zip(shift, axes)):
            raise ValueError('vertex_shift must be zero along nonperiodic axes')
        if any(value and not flag for face in item['faces']
               for value, flag in zip(face['shift'], axes)):
            raise ValueError('adjacent_shift must be zero along nonperiodic axes')
        if not isinstance(normalized, NormalizedTopology):
            continue
        edges = cell.get('edges')
        if edges is None:
            raise ValueError('normalized cells must include edges')
        edge_ids = require_global_vertex_ids(
            cell.get('edge_global_id'), name='edge_global_id',
            n_vertices=len(edges), n_global_vertices=len(normalized.global_edges))
        for edge, eid in zip(edges, edge_ids):
            u, v = require_local_vertex_indices(
                edge, name='edge.vertices', n_vertices=len(item['gids']))
            _, rep = _canon_edge((item['gids'][u], shifts[u]),
                                 (item['gids'][v], shifts[v]))
            actual = normalized.global_edges[eid]
            if (tuple(actual['vertices']) != (rep[0][0], rep[1][0])
                    or tuple(map(tuple, actual['vertex_shifts']))
                    != ((0, 0, 0), tuple(rep[1][1:]))):
                raise ValueError('edge_global_id loses raw endpoints/images')
        face_ids = require_global_vertex_ids(
            cell.get('face_global_id'), name='face_global_id',
            n_vertices=len(item['faces']),
            n_global_vertices=len(normalized.global_faces))
        for face, fid in zip(item['faces'], face_ids):
            shift = (face['shift'] if periodic and face['adjacent'] >= 0
                     else (0, 0, 0))
            pair = _canon_face_pair(item['id'], face['adjacent'], shift)
            actual = normalized.global_faces[fid]
            if (tuple(actual['cells']) != (pair[0], pair[4])
                    or tuple(map(tuple, actual['cell_shifts']))
                    != ((0, 0, 0), tuple(pair[5:]))):
                raise ValueError('face_global_id loses raw ownership/images')


def _precondition_error(normalized, domain, level, code, message):
    """Unavailable operands do not count as checked representation success."""
    diag = NormalizationDiagnostics(
        n_cells=len(normalized.cells),
        n_global_vertices=len(normalized.global_vertices),
        n_global_edges=(len(normalized.global_edges)
                        if isinstance(normalized, NormalizedTopology) else None),
        n_global_faces=(len(normalized.global_faces)
                        if isinstance(normalized, NormalizedTopology) else None),
        is_periodic_domain=is_periodic_domain(domain),
        fully_periodic_domain=_fully_periodic(domain), has_wall_faces=False,
        n_vertex_face_shift_mismatch=0, n_face_vertex_set_mismatch=0,
        n_vertices_low_incidence=0, n_edges_low_incidence=0, n_cells_bad_euler=0,
        issues=(NormalizationIssue(code, 'error', message),),
        ok_vertex_face_shift=False, ok_face_vertex_sets=False,
        ok_incidence=False, ok_euler=False, ok=False)
    if level == 'strict':
        raise NormalizationError(f'{code}: {message}', diag)
    return diag


def validate_normalized_topology(
    normalized: NormalizedVertices | NormalizedTopology,
    domain: Domain,
    *,
    level: Literal['basic', 'strict'] = 'basic',
    check_vertex_face_shift: bool = True,
    check_face_vertex_sets: bool = True,
    check_incidence: bool = True,
    check_euler: bool = True,
    max_examples: int = 10,
) -> NormalizationDiagnostics:
    """Validate periodic shift/topology consistency after normalization.

    The most important invariant (and the one most likely to detect subtle
    periodic bookkeeping bugs) is:

        For every periodic face i -> j with adjacent_shift = s,
        every local face vertex occurrence (gid, t_i) must have a peer
        occurrence (gid, t_j) satisfying:

            t_i == t_j + s

    Here t_i and t_j are the per-cell lattice-image shifts returned by
    :func:`pyvoro2.normalize_vertices`; one gid may have several local images.

    Spatial normalized vertices, edges and faces organize raw representations
    numerically. Strict success covers enabled/applicable representation checks,
    not exact public-semantic S topology or an N-to-S vertex bijection. Exact
    scientific contact incidence belongs to the separate WP5 audit.

    Args:
        normalized: Output of :func:`pyvoro2.normalize_vertices` or
            :func:`pyvoro2.normalize_topology`.
        domain: Domain used to compute the tessellation.
        level: 'basic' returns diagnostics; 'strict' raises
            :class:`NormalizationError` on any error-level issue.
        check_vertex_face_shift: Check the key vertex/face shift invariant.
        check_face_vertex_sets: Check that reciprocal face classes reference
            the same image-qualified vertices after shift transport.
        check_incidence: In fully periodic domains, check minimal incidence
            counts of cell images for vertices (>=4) and edges (>=3). Runs when
            `normalized` includes edges (i.e., is a NormalizedTopology).
        check_euler: Check Euler characteristic per cell (V - E + F == 2)
            as a warning-level sanity check.
        max_examples: Non-negative maximum number of example tuples to attach
            per issue.

    Returns:
        NormalizationDiagnostics
    """

    level = require_string_choice(
        level,
        name='level',
        choices=('basic', 'strict'),
    )

    check_vertex_face_shift = require_bool(
        check_vertex_face_shift,
        name='check_vertex_face_shift',
    )
    check_face_vertex_sets = require_bool(
        check_face_vertex_sets,
        name='check_face_vertex_sets',
    )
    check_incidence = require_bool(check_incidence, name='check_incidence')
    check_euler = require_bool(check_euler, name='check_euler')
    max_examples = require_nonnegative_index(
        max_examples,
        name='max_examples',
        maximum=sys.maxsize,
    )
    example_probe_limit = max(max_examples, 1)

    reject_ghost_records(normalized.cells)
    try:
        need_faces = (isinstance(normalized, NormalizedTopology) or check_euler
                      or (is_periodic_domain(domain)
                          and (check_vertex_face_shift or check_face_vertex_sets)))
        _check_consumed_mappings(
            normalized, domain, need_faces=need_faces)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        code = ('FACE_MISSING_ADJACENT_SHIFT'
                if 'Periodic domain face missing adjacent_shift' in str(exc)
                else 'INVALID_NORMALIZED_MAPPING')
        return _precondition_error(normalized, domain, level, code, str(exc))

    cells = list(normalized.cells)
    reject_ghost_records(cells)
    n_cells = len(cells)
    n_global_vertices = int(normalized.global_vertices.shape[0])

    n_global_edges: int | None = None
    n_global_faces: int | None = None
    if isinstance(normalized, NormalizedTopology):
        n_global_edges = len(normalized.global_edges)
        n_global_faces = len(normalized.global_faces)

    periodic = bool(is_periodic_domain(domain))
    fully_periodic = bool(_fully_periodic(domain))

    # Detect wall faces (adjacent_cell < 0). This matters for incidence checks.
    has_wall_faces = False
    for c in cells:
        faces = c.get('faces') or []
        for f in faces:
            if int(f.get('adjacent_cell', -1)) < 0:
                has_wall_faces = True
                break
        if has_wall_faces:
            break

    issues: list[NormalizationIssue] = []

    # ------------------------------------------------------------------
    # Build id->cell mapping and per-cell gid->shift mapping
    # ------------------------------------------------------------------
    cell_by_id: dict[int, dict[str, Any]] = {}
    gid_shift_by_cell: dict[int, dict[int, set[tuple[int, int, int]]]] = {}

    for c in cells:
        cid = int(c.get('id', -1))
        if cid < 0:
            continue
        cell_by_id[cid] = c

        gids = c.get('vertex_global_id')
        vsh = c.get('vertex_shift')
        if gids is None or vsh is None:
            continue
        # One quotient vertex may occur in several images of the same cell.
        m: dict[int, set[tuple[int, int, int]]] = {}
        for k, gid in enumerate(gids):
            m.setdefault(int(gid), set()).add(_as_shift(vsh[k]))
        gid_shift_by_cell[cid] = m

    # ------------------------------------------------------------------
    # Check 1: vertex_shift <-> face adjacent_shift consistency
    # ------------------------------------------------------------------
    n_vfs_mismatch = 0
    if periodic and check_vertex_face_shift:
        examples: list[
            tuple[
                int,
                int,
                tuple[int, int, int],
                int,
                tuple[int, int, int],
                tuple[tuple[int, int, int], ...],
            ]
        ] = []
        missing_neighbor_cells: list[tuple[int, int, tuple[int, int, int]]] = []
        missing_shared_vertex: list[tuple[int, int, tuple[int, int, int], int]] = []

        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            gids = c.get('vertex_global_id')
            vsh = c.get('vertex_shift')
            if gids is None or vsh is None:
                continue

            gids_list = [int(x) for x in gids]
            vsh_list = [_as_shift(x) for x in vsh]

            for f in faces:
                j = int(f.get('adjacent_cell', -1))
                if j < 0:
                    continue
                if 'adjacent_shift' not in f:
                    issues.append(
                        NormalizationIssue(
                            code='FACE_MISSING_ADJACENT_SHIFT',
                            severity='error',
                            message=(
                                'A periodic neighbor face is missing adjacent_shift. '
                                'Ensure compute(..., return_face_shifts=True) was used.'
                            ),
                            examples=((cid, j),)[:max_examples],
                        )
                    )
                    continue

                s = _as_shift(f.get('adjacent_shift', (0, 0, 0)))
                cj = cell_by_id.get(j)
                if cj is None:
                    if len(missing_neighbor_cells) < example_probe_limit:
                        missing_neighbor_cells.append((cid, j, s))
                    continue
                map_j = gid_shift_by_cell.get(j)
                if map_j is None:
                    continue

                idx = _iter_face_vertex_indices(f)
                for vk in idx:
                    if vk < 0 or vk >= len(gids_list):
                        continue
                    gid = gids_list[vk]
                    ti = vsh_list[vk]
                    tj_set = map_j.get(gid)
                    if not tj_set:
                        if len(missing_shared_vertex) < example_probe_limit:
                            missing_shared_vertex.append((cid, j, s, gid))
                        continue
                    expected = tuple(ti[axis] - s[axis] for axis in range(3))
                    if expected not in tj_set:
                        n_vfs_mismatch += 1
                        if len(examples) < example_probe_limit:
                            examples.append((cid, j, s, gid, ti, tuple(sorted(tj_set))))

        if missing_neighbor_cells:
            issues.append(
                NormalizationIssue(
                    code='NEIGHBOR_CELL_MISSING',
                    severity='error',
                    message=(
                        'A face references a neighbor cell id that is not present '
                        'in the normalized output.'
                    ),
                    examples=tuple(missing_neighbor_cells[:max_examples]),
                )
            )
        if missing_shared_vertex:
            issues.append(
                NormalizationIssue(
                    code='SHARED_VERTEX_MISSING_IN_NEIGHBOR',
                    severity='error',
                    message=(
                        'A periodic neighbor face references a global vertex id '
                        'that is not present in the neighbor cell. '
                        'This suggests inconsistent vertex normalization.'
                    ),
                    examples=tuple(missing_shared_vertex[:max_examples]),
                )
            )
        if examples:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_FACE_SHIFT_MISMATCH',
                    severity='error',
                    message=(
                        'vertex_shift and adjacent_shift are inconsistent across '
                        'a periodic face: expected vertex_shift_i == '
                        'vertex_shift_j + adjacent_shift.'
                    ),
                    examples=tuple(examples[:max_examples]),
                )
            )

    # ------------------------------------------------------------------
    # Check 2: reciprocal face classes have matching image-qualified vertices
    # ------------------------------------------------------------------
    n_face_set_mismatch = 0
    if periodic and check_face_vertex_sets:
        face_map: dict[
            tuple[int, int, tuple[int, int, int]], list[dict[str, Any]]
        ] = {}
        for c in cells:
            i = int(c.get('id', -1))
            if i < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            for f in faces:
                j = int(f.get('adjacent_cell', -1))
                if j < 0:
                    continue
                if 'adjacent_shift' not in f:
                    continue
                s = _as_shift(f.get('adjacent_shift', (0, 0, 0)))
                face_map.setdefault((i, j, s), []).append(f)

        checked: set[tuple[int, int, tuple[int, int, int]]] = set()
        examples: list[
            tuple[int, int, tuple[int, int, int], tuple[Any, ...], tuple[Any, ...]]
        ] = []

        def _face_vertices(cid, faces, translation):
            c = cell_by_id[cid]
            gids, shifts = c.get('vertex_global_id'), c.get('vertex_shift')
            if gids is None or shifts is None:
                return set()
            # Keep all fragments in a directed class; never overwrite one
            # occurrence or guess a one-to-one reciprocal fragment pairing.
            return {
                (int(gids[v]), tuple(
                    int(shifts[v][axis]) + translation[axis] for axis in range(3)))
                for face in faces for v in _iter_face_vertex_indices(face)
                if 0 <= v < len(gids)
            }

        for (i, j, s), faces in face_map.items():
            if (i, j, s) in checked:
                continue
            r = (j, i, tuple(-value for value in s))
            checked.update(((i, j, s), r))
            reverse = face_map.get(r)
            if reverse is None:
                continue
            here = _face_vertices(i, faces, (0, 0, 0))
            there = _face_vertices(j, reverse, s)
            if here != there:
                n_face_set_mismatch += 1
                if len(examples) < example_probe_limit:
                    examples.append(
                        (i, j, s, tuple(sorted(here)), tuple(sorted(there))))

        if examples:
            issues.append(
                NormalizationIssue(
                    code='RECIPROCAL_FACE_VERTEX_SET_MISMATCH',
                    severity='error',
                    message=(
                        'Reciprocal periodic faces do not reference the same set of '
                        'image-qualified global vertices.'
                    ),
                    examples=tuple(examples[:max_examples]),
                )
            )

    # ------------------------------------------------------------------
    # Check 3: incidence sanity (fully periodic domains)
    # ------------------------------------------------------------------
    n_vertices_low = 0
    n_edges_low = 0
    if (
        check_incidence
        and fully_periodic
        and periodic
        and (not has_wall_faces)
        and isinstance(normalized, NormalizedTopology)
    ):
        vertex_to_cells: dict[int, set[tuple]] = {}
        edge_to_cells: dict[int, set[tuple]] = {}

        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            gids, shifts = c.get('vertex_global_id'), c.get('vertex_shift')
            if gids is None or shifts is None:
                continue
            for gid, shift in zip(gids, shifts):
                image = tuple(-int(value) for value in shift)
                vertex_to_cells.setdefault(int(gid), set()).add((cid, image))
            # Anchor each local edge at its canonical first endpoint. This
            # distinguishes the cell images meeting one lift of a pooled edge.
            for edge, eid in zip(c.get('edges') or [], c.get('edge_global_id') or []):
                anchor = min(edge, key=lambda v: (int(gids[v]), _as_shift(shifts[v])))
                image = tuple(-int(value) for value in shifts[anchor])
                edge_to_cells.setdefault(int(eid), set()).add((cid, image))

        v_warn, v_err = 4, 3
        e_warn, e_err = 3, 2

        bad_v_warn: list[tuple[int, int]] = []
        bad_v_err: list[tuple[int, int]] = []
        for gid, ss in vertex_to_cells.items():
            deg = len(ss)
            if deg < v_err:
                n_vertices_low += 1
                if len(bad_v_err) < example_probe_limit:
                    bad_v_err.append((gid, deg))
            elif deg < v_warn:
                n_vertices_low += 1
                if len(bad_v_warn) < example_probe_limit:
                    bad_v_warn.append((gid, deg))

        bad_e_warn: list[tuple[int, int]] = []
        bad_e_err: list[tuple[int, int]] = []
        for eid, ss in edge_to_cells.items():
            deg = len(ss)
            if deg < e_err:
                n_edges_low += 1
                if len(bad_e_err) < example_probe_limit:
                    bad_e_err.append((eid, deg))
            elif deg < e_warn:
                n_edges_low += 1
                if len(bad_e_warn) < example_probe_limit:
                    bad_e_warn.append((eid, deg))

        if bad_v_err:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_INCIDENCE_TOO_LOW',
                    severity='error',
                    message=(
                        'In a fully periodic tessellation, some global vertices are '
                        'incident to fewer than 3 cell images.'
                    ),
                    examples=tuple(bad_v_err[:max_examples]),
                )
            )
        if bad_v_warn:
            issues.append(
                NormalizationIssue(
                    code='VERTEX_INCIDENCE_LOW',
                    severity='warning',
                    message=(
                        'Some global vertices are incident to fewer than 4 cell images '
                        '(may indicate degeneracy or issues).'
                    ),
                    examples=tuple(bad_v_warn[:max_examples]),
                )
            )
        if bad_e_err:
            issues.append(
                NormalizationIssue(
                    code='EDGE_INCIDENCE_TOO_LOW',
                    severity='error',
                    message=(
                        'In a fully periodic tessellation, some global edges are '
                        'incident to fewer than 2 cell images.'
                    ),
                    examples=tuple(bad_e_err[:max_examples]),
                )
            )
        if bad_e_warn:
            issues.append(
                NormalizationIssue(
                    code='EDGE_INCIDENCE_LOW',
                    severity='warning',
                    message=(
                        'Some global edges are incident to fewer than 3 cell images '
                        '(may indicate degeneracy or issues).'
                    ),
                    examples=tuple(bad_e_warn[:max_examples]),
                )
            )

    # ------------------------------------------------------------------
    # Check 4: Euler characteristic per cell (warning-level)
    # ------------------------------------------------------------------
    n_bad_euler = 0
    if check_euler:
        examples: list[tuple[int, int, int, int, int]] = []
        for c in cells:
            cid = int(c.get('id', -1))
            if cid < 0 or bool(c.get('empty', False)):
                continue
            faces = c.get('faces') or []
            face_cycles: list[list[int]] = []
            for f in faces:
                idx = _iter_face_vertex_indices(f)
                if len(idx) >= 3:
                    face_cycles.append(idx)
            if not face_cycles:
                continue

            vset: set[int] = set()
            eset: set[tuple[int, int]] = set()
            for cyc in face_cycles:
                for v in cyc:
                    vset.add(int(v))
                m = len(cyc)
                for k in range(m):
                    a = int(cyc[k])
                    b = int(cyc[(k + 1) % m])
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    eset.add((a, b))
            V = len(vset)
            E = len(eset)
            F = len(face_cycles)
            chi = V - E + F
            if chi != 2:
                n_bad_euler += 1
                if len(examples) < example_probe_limit:
                    examples.append((cid, chi, V, E, F))

        if examples:
            issues.append(
                NormalizationIssue(
                    code='EULER_CHARACTERISTIC_MISMATCH',
                    severity='warning',
                    message=(
                        'Some cells do not satisfy Euler characteristic V - E + F == 2 '
                        '(may indicate degeneracy).'
                    ),
                    examples=tuple(examples[:max_examples]),
                )
            )

    ok_vertex_face = not any(
        i.severity == 'error'
        and i.code
        in (
            'DUPLICATE_GID_DIFFERENT_SHIFT',
            'FACE_MISSING_ADJACENT_SHIFT',
            'NEIGHBOR_CELL_MISSING',
            'SHARED_VERTEX_MISSING_IN_NEIGHBOR',
            'VERTEX_FACE_SHIFT_MISMATCH',
        )
        for i in issues
    )
    ok_face_sets = n_face_set_mismatch == 0
    ok_inc = not any(
        i.severity == 'error'
        and i.code
        in (
            'VERTEX_INCIDENCE_TOO_LOW',
            'EDGE_INCIDENCE_TOO_LOW',
        )
        for i in issues
    )
    ok_euler = n_bad_euler == 0
    ok = diagnostics_ok(issues)

    diag = NormalizationDiagnostics(
        n_cells=int(n_cells),
        n_global_vertices=int(n_global_vertices),
        n_global_edges=(int(n_global_edges) if n_global_edges is not None else None),
        n_global_faces=(int(n_global_faces) if n_global_faces is not None else None),
        is_periodic_domain=bool(periodic),
        fully_periodic_domain=bool(fully_periodic),
        has_wall_faces=bool(has_wall_faces),
        n_vertex_face_shift_mismatch=int(n_vfs_mismatch),
        n_face_vertex_set_mismatch=int(n_face_set_mismatch),
        n_vertices_low_incidence=int(n_vertices_low),
        n_edges_low_incidence=int(n_edges_low),
        n_cells_bad_euler=int(n_bad_euler),
        issues=tuple(issues),
        ok_vertex_face_shift=bool(ok_vertex_face),
        ok_face_vertex_sets=bool(ok_face_sets),
        ok_incidence=bool(ok_inc),
        ok_euler=bool(ok_euler),
        ok=bool(ok),
    )

    if level == 'strict' and not diag.ok:
        err = next((x for x in diag.issues if x.severity == 'error'), None)
        msg = err.message if err is not None else 'Normalization validation failed'
        raise NormalizationError(msg, diag)
    return diag
