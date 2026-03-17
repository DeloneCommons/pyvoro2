"""Planar topology-level post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import warnings

import numpy as np

from ._domain_geometry import geometry2d
from .domains import Box, RectangularCell


Domain2D = Box | RectangularCell


@dataclass(frozen=True)
class NormalizedVertices:
    """Result of :func:`normalize_vertices` for planar tessellations.

    Attributes:
        global_vertices: Array of unique planar vertices in Cartesian coordinates,
            remapped into the primary cell for periodic domains.
        cells: Per-cell dictionaries augmented with:
            - vertex_global_id: list[int] aligned with local vertices
            - vertex_shift: list[tuple[int, int]] aligned with local vertices
    """

    global_vertices: np.ndarray
    cells: list[dict[str, Any]]


@dataclass(frozen=True)
class NormalizedTopology:
    """Result of :func:`normalize_topology` for planar tessellations.

    Attributes:
        global_vertices: Unique planar vertices in Cartesian coordinates.
        global_edges: Unique geometric edges. Each edge dict contains:
            - cells: (cid0, cid1)
            - cell_shifts: ((0, 0), (sx, sy))
            - vertices: (gid0, gid1)
            - vertex_shifts: ((0, 0), (sx, sy))
        cells: Per-cell dictionaries including ``vertex_global_id``,
            ``vertex_shift``, and ``edge_global_id`` aligned with local edges.
    """

    global_vertices: np.ndarray
    global_edges: list[dict[str, Any]]
    cells: list[dict[str, Any]]


def _domain_length_scale(domain: Domain2D) -> float:
    (lx, ly), _area = geometry2d(domain)._lengths_and_area()
    L = float(max(lx, ly))
    return L if np.isfinite(L) else 0.0


def _is_periodic_domain(domain: Domain2D) -> bool:
    return bool(geometry2d(domain).has_any_periodic_axis)


def _quant_key(coord: np.ndarray, tol: float) -> tuple[int, int]:
    q = np.rint(coord / tol).astype(np.int64)
    return int(q[0]), int(q[1])


def _canonical_incident_key(
    incident: Sequence[tuple[int, tuple[int, int]]]
) -> tuple[tuple[int, int, int], ...]:
    """Canonicalize an incident cell-image set up to global translation."""

    uniq = sorted(set((int(cid), (int(s[0]), int(s[1]))) for cid, s in incident))
    if not uniq:
        return tuple()

    best: tuple[tuple[int, int, int], ...] | None = None
    for _cid_a, s_a in uniq:
        sa = np.array(s_a, dtype=np.int64)
        rep = []
        for cid, s in uniq:
            ss = np.array(s, dtype=np.int64) - sa
            rep.append((cid, int(ss[0]), int(ss[1])))
        rep_sorted = tuple(sorted(rep))
        if best is None or rep_sorted < best:
            best = rep_sorted
    assert best is not None
    return best


def normalize_vertices(
    cells: list[dict[str, Any]],
    *,
    domain: Domain2D,
    tol: float | None = None,
    require_edge_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedVertices:
    """Build a global planar vertex pool and per-cell vertex mappings."""

    L = _domain_length_scale(domain)
    periodic = _is_periodic_domain(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        if float(L) < 1e-3 or float(L) > 1e9:
            warnings.warn(
                'normalize_vertices is using a default tolerance proportional to '
                'the planar domain length scale '
                f'(L≈{float(L):.3g}). For very small/large units this may be '
                'too strict/too loose. Consider rescaling your coordinates or '
                'passing an explicit tol=... .',
                RuntimeWarning,
                stacklevel=2,
            )
    if tol <= 0:
        raise ValueError('tol must be positive')
    if not isinstance(cells, list):
        raise ValueError('cells must be a list of dicts')

    out_cells = [dict(c) for c in cells] if copy_cells else cells
    global_vertices: list[np.ndarray] = []
    key_to_gid: dict[tuple[Any, ...], int] = {}

    if not periodic:
        for cell in out_cells:
            verts = np.asarray(cell.get('vertices', []), dtype=float)
            if verts.size == 0:
                verts = verts.reshape((0, 2))
            if verts.ndim != 2 or verts.shape[1] != 2:
                raise ValueError('cells must include vertices with shape (m, 2)')

            gids: list[int] = []
            shifts: list[tuple[int, int]] = []
            for v in verts:
                key = ('box',) + _quant_key(v, tol)
                gid = key_to_gid.get(key)
                if gid is None:
                    gid = len(global_vertices)
                    key_to_gid[key] = gid
                    global_vertices.append(v.astype(np.float64))
                gids.append(gid)
                shifts.append((0, 0))
            cell['vertex_global_id'] = gids
            cell['vertex_shift'] = shifts

        return NormalizedVertices(
            global_vertices=(
                np.stack(global_vertices, axis=0)
                if global_vertices
                else np.zeros((0, 2), dtype=np.float64)
            ),
            cells=out_cells,
        )

    if require_edge_shifts:
        for cell in out_cells:
            edges = cell.get('edges')
            if edges is None:
                raise ValueError('cells must include edges for periodic normalization')
            for edge in edges:
                if 'adjacent_shift' not in edge:
                    raise ValueError(
                        'cells must include edge adjacent_shift '
                        '(compute with return_edge_shifts=True)'
                    )

    sorted_cells = sorted(out_cells, key=lambda cc: int(cc.get('id', 0)))

    for cell in sorted_cells:
        verts = np.asarray(cell.get('vertices', []), dtype=float)
        if verts.size == 0:
            verts = verts.reshape((0, 2))
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError('cells must include vertices with shape (m, 2)')
        edges = cell.get('edges')
        if edges is None:
            raise ValueError('cells must include edges for periodic normalization')

        v_edges: list[list[dict[str, Any]]] = [[] for _ in range(int(verts.shape[0]))]
        for edge in edges:
            idx = edge.get('vertices')
            if idx is None:
                continue
            for vid in idx:
                iv = int(vid)
                if 0 <= iv < len(v_edges):
                    v_edges[iv].append(edge)

        gids: list[int] = []
        shifts: list[tuple[int, int]] = []

        if not isinstance(domain, RectangularCell):
            raise ValueError('periodic planar normalization requires RectangularCell')
        remapped, rem_shifts = domain.remap_cart(verts, return_shifts=True)
        for _ in range(2):
            remapped2, extra = domain.remap_cart(remapped, return_shifts=True)
            remapped = remapped2
            rem_shifts = rem_shifts + extra
            if not np.any(extra):
                break

        for k in range(int(verts.shape[0])):
            v0 = remapped[k]
            s0 = (int(rem_shifts[k, 0]), int(rem_shifts[k, 1]))
            incident: list[tuple[int, tuple[int, int]]] = []
            cid_here = int(cell.get('id', 0))
            incident.append((cid_here, (0, 0)))
            for edge in v_edges[k]:
                adj = int(edge.get('adjacent_cell', -999999))
                sh = edge.get('adjacent_shift', (0, 0))
                sh_t = (int(sh[0]), int(sh[1]))
                incident.append((adj, sh_t))

            topo_key = _canonical_incident_key(incident)
            coord_key = _quant_key(v0, tol)
            key: tuple[Any, ...] = ('pbc',) + topo_key + ('@',) + coord_key
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(global_vertices)
                key_to_gid[key] = gid
                global_vertices.append(v0.astype(np.float64))
            else:
                dv = float(np.linalg.norm(global_vertices[gid] - v0))
                if dv > 10 * tol:
                    raise ValueError(
                        'vertex key collision: same topology key but significantly '
                        'different coordinates; '
                        f'gid={gid}, dv={dv}'
                    )
            gids.append(gid)
            shifts.append(s0)

        cell['vertex_global_id'] = gids
        cell['vertex_shift'] = shifts

    return NormalizedVertices(
        global_vertices=(
            np.stack(global_vertices, axis=0)
            if global_vertices
            else np.zeros((0, 2), dtype=np.float64)
        ),
        cells=out_cells,
    )


def _as_shift(s: Any) -> tuple[int, int]:
    return int(s[0]), int(s[1])


def _canon_edge(
    a: tuple[int, tuple[int, int]],
    b: tuple[int, tuple[int, int]],
) -> tuple[tuple[Any, ...], tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Canonicalize an edge up to translation and orientation."""

    gid0, s0 = a
    gid1, s1 = b
    s0a = np.array(s0, dtype=np.int64)
    s1a = np.array(s1, dtype=np.int64)

    candidates = []
    for ga, sa, gb, sb in ((gid0, s0a, gid1, s1a), (gid1, s1a, gid0, s0a)):
        d = sb - sa
        recs = ((int(ga), 0, 0), (int(gb), int(d[0]), int(d[1])))
        candidates.append(tuple(sorted(recs)))
    best = min(candidates)

    g0, x0, y0 = best[0]
    g1, x1, y1 = best[1]
    rep = ((int(g0), 0, 0), (int(g1), int(x1 - x0), int(y1 - y0)))
    key = ('e', int(rep[0][0]), int(rep[1][0]), int(rep[1][1]), int(rep[1][2]))
    return key, rep


def _canon_cell_pair(
    cid_here: int,
    adj: int,
    adj_shift: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    sx, sy = int(adj_shift[0]), int(adj_shift[1])
    rep1 = (int(cid_here), 0, 0, int(adj), sx, sy)
    rep2 = (int(adj), 0, 0, int(cid_here), -sx, -sy)
    return rep2 if rep2 < rep1 else rep1


def normalize_edges(
    nv: NormalizedVertices,
    *,
    domain: Domain2D,
    tol: float | None = None,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Build a global edge pool based on an existing planar normalization."""

    L = _domain_length_scale(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        if float(L) < 1e-3 or float(L) > 1e9:
            warnings.warn(
                'normalize_edges is using a default tolerance proportional to '
                'the planar domain length scale '
                f'(L≈{float(L):.3g}). For very small/large units this may be '
                'too strict/too loose. Consider rescaling your coordinates or '
                'passing an explicit tol=... .',
                RuntimeWarning,
                stacklevel=2,
            )
    if tol <= 0:
        raise ValueError('tol must be positive')

    cells = [dict(c) for c in nv.cells] if copy_cells else nv.cells
    global_edges: list[dict[str, Any]] = []
    edge_key_to_id: dict[tuple[Any, ...], int] = {}
    periodic = _is_periodic_domain(domain)
    sorted_cells = sorted(cells, key=lambda cc: int(cc.get('id', 0)))

    for cell in sorted_cells:
        edges = cell.get('edges')
        if edges is None:
            raise ValueError('cells must include edges')
        gids = cell.get('vertex_global_id')
        vsh = cell.get('vertex_shift')
        if gids is None or vsh is None:
            raise ValueError(
                'cells must include vertex_global_id and vertex_shift '
                '(call normalize_vertices first)'
            )

        edge_ids: list[int] = []
        cid_here = int(cell.get('id', 0))
        for edge in edges:
            adj = int(edge.get('adjacent_cell', -999999))
            if periodic and adj >= 0:
                if 'adjacent_shift' not in edge:
                    raise ValueError(
                        'Periodic domain edge missing adjacent_shift; compute '
                        'with return_edge_shifts=True'
                    )
                adj_shift = _as_shift(edge.get('adjacent_shift'))
            else:
                adj_shift = (0, 0)

            idx = np.asarray(edge.get('vertices', []), dtype=np.int64)
            if idx.shape != (2,):
                raise ValueError('edge vertices must have shape (2,)')
            u = int(idx[0])
            v = int(idx[1])
            if u < 0 or v < 0 or u >= len(gids) or v >= len(gids):
                raise ValueError('edge references an out-of-range local vertex index')

            ekey, erep = _canon_edge(
                (int(gids[u]), _as_shift(vsh[u])),
                (int(gids[v]), _as_shift(vsh[v])),
            )
            eid = edge_key_to_id.get(ekey)
            if eid is None:
                eid = len(global_edges)
                edge_key_to_id[ekey] = eid
                pair = _canon_cell_pair(cid_here, adj, adj_shift)
                global_edges.append(
                    {
                        'cells': (int(pair[0]), int(pair[3])),
                        'cell_shifts': ((0, 0), (int(pair[4]), int(pair[5]))),
                        'vertices': (int(erep[0][0]), int(erep[1][0])),
                        'vertex_shifts': (
                            (0, 0),
                            (int(erep[1][1]), int(erep[1][2])),
                        ),
                    }
                )
            edge_ids.append(eid)
        cell['edge_global_id'] = edge_ids

    return NormalizedTopology(
        global_vertices=nv.global_vertices,
        global_edges=global_edges,
        cells=cells,
    )


def normalize_topology(
    cells: list[dict[str, Any]],
    *,
    domain: Domain2D,
    tol: float | None = None,
    require_edge_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Convenience wrapper: normalize vertices, then deduplicate edges."""

    nv = normalize_vertices(
        cells,
        domain=domain,
        tol=tol,
        require_edge_shifts=require_edge_shifts,
        copy_cells=copy_cells,
    )
    return normalize_edges(nv, domain=domain, tol=tol, copy_cells=False)
