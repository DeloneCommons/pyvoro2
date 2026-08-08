"""Planar topology-level post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import warnings

import numpy as np

from .._internal.normalization import (
    checked_add_shift_arrays,
    checked_shift_difference,
    coerce_normalization_vertices,
    quantize_coordinates,
    quantized_key_matches,
    require_adjacent_cell_id,
    require_cell_id,
    require_global_vertex_ids,
    require_local_vertex_indices,
    require_shift,
    require_shift_rows,
    validate_pairwise_shift_differences,
)
from .._internal.planar.domain_geometry import geometry2d
from .._internal.validation import require_bool, require_positive_finite_real
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


def _prepare_vertex_cells(
    cells: list[dict[str, Any]],
    *,
    domain: Domain2D,
    tol: float,
    periodic: bool,
    require_edge_shifts: bool,
) -> list[dict[str, Any]]:
    """Validate all consumed raw records before constructing or mutating output."""

    if not isinstance(cells, list):
        raise ValueError('cells must be a list of dicts')
    if periodic and not isinstance(domain, RectangularCell):
        raise ValueError('periodic planar normalization requires RectangularCell')

    prepared: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f'cells[{cell_index}] must be a dict')
        if 'id' not in cell:
            raise ValueError(f'cells[{cell_index}].id is required')
        cid = require_cell_id(cell['id'], name=f'cells[{cell_index}].id')
        if cid in seen_ids:
            raise ValueError('cell IDs must be unique')
        seen_ids.add(cid)

        vertices = coerce_normalization_vertices(
            cell.get('vertices', []),
            name=f'cells[{cell_index}].vertices',
            dim=2,
        )
        edge_data: list[dict[str, Any]] = []
        remapped = vertices
        remap_shifts = np.zeros(vertices.shape, dtype=np.int64)

        if periodic:
            edges = cell.get('edges')
            if edges is None:
                raise ValueError(
                    'cells must include edges for periodic normalization'
                )
            try:
                edge_records = tuple(edges)
            except TypeError:
                raise ValueError(
                    f'cells[{cell_index}].edges must be a sequence of dicts'
                ) from None
            for edge_index, edge in enumerate(edge_records):
                prefix = f'cells[{cell_index}].edges[{edge_index}]'
                if not isinstance(edge, dict):
                    raise ValueError(f'{prefix} must be a dict')
                vertices_local = (
                    tuple()
                    if edge.get('vertices') is None
                    else require_local_vertex_indices(
                        edge['vertices'],
                        name=f'{prefix}.vertices',
                        n_vertices=int(vertices.shape[0]),
                        length=2,
                    )
                )
                if 'adjacent_cell' not in edge:
                    raise ValueError(f'{prefix}.adjacent_cell is required')
                adjacent = require_adjacent_cell_id(
                    edge['adjacent_cell'],
                    name=f'{prefix}.adjacent_cell',
                )
                if 'adjacent_shift' in edge:
                    adjacent_shift = require_shift(
                        edge['adjacent_shift'],
                        name=f'{prefix}.adjacent_shift',
                        dim=2,
                    )
                elif require_edge_shifts:
                    raise ValueError(
                        'cells must include edge adjacent_shift '
                        '(compute with return_edge_shifts=True)'
                    )
                else:
                    adjacent_shift = (0, 0)
                edge_data.append(
                    {
                        'vertices': vertices_local,
                        'adjacent': adjacent,
                        'shift': adjacent_shift,
                    }
                )

            for vertex_index in range(int(vertices.shape[0])):
                incident_shifts = [(0, 0)] + [
                    edge['shift']
                    for edge in edge_data
                    if vertex_index in edge['vertices']
                ]
                validate_pairwise_shift_differences(
                    incident_shifts,
                    name=(
                        f'cells[{cell_index}].vertex[{vertex_index}] '
                        'incident shift'
                    ),
                )

            remapped, remap_shifts = domain.remap_cart(
                vertices,
                return_shifts=True,
            )
            for _ in range(2):
                remapped2, extra = domain.remap_cart(
                    remapped,
                    return_shifts=True,
                )
                remapped = remapped2
                remap_shifts = checked_add_shift_arrays(
                    remap_shifts,
                    extra,
                    name=f'cells[{cell_index}].vertex_shift',
                )
                if not np.any(extra):
                    break

        quantized = quantize_coordinates(
            remapped,
            tol=tol,
            name=f'cells[{cell_index}].vertices',
        )
        prepared.append(
            {
                'position': cell_index,
                'id': cid,
                'vertices': vertices,
                'edges': tuple(edge_data),
                'remapped': remapped,
                'remap_shifts': remap_shifts,
                'quantized': quantized,
            }
        )
    return prepared


def _canonical_incident_key(
    incident: Sequence[tuple[int, tuple[int, int]]]
) -> tuple[tuple[int, int, int], ...]:
    """Canonicalize an incident cell-image set up to global translation."""

    uniq = sorted(set((cid, tuple(s)) for cid, s in incident))
    if not uniq:
        return tuple()

    best: tuple[tuple[int, int, int], ...] | None = None
    for _cid_a, s_a in uniq:
        rep = []
        for cid, s in uniq:
            ss = checked_shift_difference(
                s,
                s_a,
                name='incident lattice shift',
            )
            rep.append((cid, ss[0], ss[1]))
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

    require_edge_shifts = require_bool(
        require_edge_shifts,
        name='require_edge_shifts',
    )
    copy_cells = require_bool(copy_cells, name='copy_cells')
    if tol is not None:
        tol = require_positive_finite_real(tol, name='tol')

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
    tol = require_positive_finite_real(tol, name='tol')
    prepared = _prepare_vertex_cells(
        cells,
        domain=domain,
        tol=tol,
        periodic=periodic,
        require_edge_shifts=require_edge_shifts,
    )
    global_vertices: list[np.ndarray] = []
    key_to_gid: dict[tuple[Any, ...], int] = {}
    mappings: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}

    if not periodic:
        for item in prepared:
            verts = item['vertices']
            gids: list[int] = []
            shifts: list[tuple[int, int]] = []
            for v, coord_key in zip(verts, item['quantized']):
                key = ('box',) + coord_key
                gid = key_to_gid.get(key)
                if gid is None:
                    gid = len(global_vertices)
                    key_to_gid[key] = gid
                    global_vertices.append(v.astype(np.float64))
                elif not quantized_key_matches(
                    global_vertices[gid],
                    v,
                    tol=tol,
                ):
                    raise ValueError(
                        'vertex quantization key collision for coordinates '
                        'farther apart than tol'
                    )
                gids.append(gid)
                shifts.append((0, 0))
            mappings[item['position']] = (gids, shifts)

        out_cells = [dict(cell) for cell in cells] if copy_cells else cells
        for position, (gids, shifts) in mappings.items():
            out_cells[position]['vertex_global_id'] = gids
            out_cells[position]['vertex_shift'] = shifts

        return NormalizedVertices(
            global_vertices=(
                np.stack(global_vertices, axis=0)
                if global_vertices
                else np.zeros((0, 2), dtype=np.float64)
            ),
            cells=out_cells,
        )

    for item in sorted(prepared, key=lambda record: record['id']):
        verts = item['vertices']
        edges = item['edges']

        v_edges: list[list[dict[str, Any]]] = [
            [] for _ in range(int(verts.shape[0]))
        ]
        for edge in edges:
            for vertex_index in edge['vertices']:
                v_edges[vertex_index].append(edge)

        gids: list[int] = []
        shifts: list[tuple[int, int]] = []

        remapped = item['remapped']
        rem_shifts = item['remap_shifts']

        for k in range(int(verts.shape[0])):
            v0 = remapped[k]
            s0 = (int(rem_shifts[k, 0]), int(rem_shifts[k, 1]))
            incident: list[tuple[int, tuple[int, int]]] = []
            cid_here = item['id']
            incident.append((cid_here, (0, 0)))
            for edge in v_edges[k]:
                incident.append((edge['adjacent'], edge['shift']))

            topo_key = _canonical_incident_key(incident)
            coord_key = item['quantized'][k]
            key: tuple[Any, ...] = ('pbc',) + topo_key + ('@',) + coord_key
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(global_vertices)
                key_to_gid[key] = gid
                global_vertices.append(v0.astype(np.float64))
            elif not quantized_key_matches(
                global_vertices[gid],
                v0,
                tol=tol,
            ):
                raise ValueError(
                    'vertex quantization key collision for coordinates '
                    'farther apart than tol'
                )
            gids.append(gid)
            shifts.append(s0)
        mappings[item['position']] = (gids, shifts)

    out_cells = [dict(cell) for cell in cells] if copy_cells else cells
    for position, (gids, shifts) in mappings.items():
        out_cells[position]['vertex_global_id'] = gids
        out_cells[position]['vertex_shift'] = shifts

    return NormalizedVertices(
        global_vertices=(
            np.stack(global_vertices, axis=0)
            if global_vertices
            else np.zeros((0, 2), dtype=np.float64)
        ),
        cells=out_cells,
    )


def _canon_edge(
    a: tuple[int, tuple[int, int]],
    b: tuple[int, tuple[int, int]],
) -> tuple[tuple[Any, ...], tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Canonicalize an edge up to translation and orientation."""

    gid0, s0 = a
    gid1, s1 = b
    candidates = []
    for ga, sa, gb, sb in ((gid0, s0, gid1, s1), (gid1, s1, gid0, s0)):
        d = checked_shift_difference(sb, sa, name='edge vertex shift')
        recs = ((int(ga), 0, 0), (int(gb), d[0], d[1]))
        candidates.append(tuple(sorted(recs)))
    best = min(candidates)

    g0, x0, y0 = best[0]
    g1, x1, y1 = best[1]
    relative = checked_shift_difference(
        (x1, y1),
        (x0, y0),
        name='canonical edge vertex shift',
    )
    rep = ((int(g0), 0, 0), (int(g1), *relative))
    key = ('e', int(rep[0][0]), int(rep[1][0]), int(rep[1][1]), int(rep[1][2]))
    return key, rep


def _canon_cell_pair(
    cid_here: int,
    adj: int,
    adj_shift: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    sx, sy = int(adj_shift[0]), int(adj_shift[1])
    rep1 = (int(cid_here), 0, 0, int(adj), sx, sy)
    reverse = checked_shift_difference(
        (0, 0),
        (sx, sy),
        name='adjacent edge shift',
    )
    rep2 = (int(adj), 0, 0, int(cid_here), *reverse)
    return rep2 if rep2 < rep1 else rep1


def _prepare_topology_cells(
    nv: NormalizedVertices,
    *,
    periodic: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Validate every record consumed by planar edge construction."""

    global_vertices = coerce_normalization_vertices(
        nv.global_vertices,
        name='normalized.global_vertices',
        dim=2,
    )
    if not isinstance(nv.cells, list):
        raise ValueError('normalized.cells must be a list of dicts')

    prepared: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for cell_index, cell in enumerate(nv.cells):
        if not isinstance(cell, dict):
            raise ValueError(f'normalized.cells[{cell_index}] must be a dict')
        prefix = f'normalized.cells[{cell_index}]'
        if 'id' not in cell:
            raise ValueError(f'{prefix}.id is required')
        cid = require_cell_id(cell['id'], name=f'{prefix}.id')
        if cid in seen_ids:
            raise ValueError('cell IDs must be unique')
        seen_ids.add(cid)

        vertices = coerce_normalization_vertices(
            cell.get('vertices', []),
            name=f'{prefix}.vertices',
            dim=2,
        )
        edges = cell.get('edges')
        if edges is None:
            raise ValueError('cells must include edges')
        try:
            edge_records = tuple(edges)
        except TypeError:
            raise ValueError(f'{prefix}.edges must be a sequence of dicts') from None

        gids_raw = cell.get('vertex_global_id')
        shifts_raw = cell.get('vertex_shift')
        if gids_raw is None or shifts_raw is None:
            raise ValueError(
                'cells must include vertex_global_id and vertex_shift '
                '(call normalize_vertices first)'
            )
        gids = require_global_vertex_ids(
            gids_raw,
            name=f'{prefix}.vertex_global_id',
            n_vertices=int(vertices.shape[0]),
            n_global_vertices=int(global_vertices.shape[0]),
        )
        vertex_shifts = require_shift_rows(
            shifts_raw,
            name=f'{prefix}.vertex_shift',
            rows=int(vertices.shape[0]),
            dim=2,
        )

        edge_data: list[dict[str, Any]] = []
        for edge_index, edge in enumerate(edge_records):
            edge_prefix = f'{prefix}.edges[{edge_index}]'
            if not isinstance(edge, dict):
                raise ValueError(f'{edge_prefix} must be a dict')
            vertices_local = require_local_vertex_indices(
                edge.get('vertices', []),
                name=f'{edge_prefix}.vertices',
                n_vertices=int(vertices.shape[0]),
                length=2,
            )
            if 'adjacent_cell' not in edge:
                raise ValueError(f'{edge_prefix}.adjacent_cell is required')
            adjacent = require_adjacent_cell_id(
                edge['adjacent_cell'],
                name=f'{edge_prefix}.adjacent_cell',
            )
            if 'adjacent_shift' in edge:
                adjacent_shift = require_shift(
                    edge['adjacent_shift'],
                    name=f'{edge_prefix}.adjacent_shift',
                    dim=2,
                )
            elif periodic and adjacent >= 0:
                raise ValueError(
                    'Periodic domain edge missing adjacent_shift; compute '
                    'with return_edge_shifts=True'
                )
            else:
                adjacent_shift = (0, 0)
            effective_shift = (
                adjacent_shift
                if periodic and adjacent >= 0
                else (0, 0)
            )
            checked_shift_difference(
                (0, 0),
                effective_shift,
                name=f'{edge_prefix}.adjacent_shift',
            )
            validate_pairwise_shift_differences(
                [vertex_shifts[value] for value in vertices_local],
                name=f'{edge_prefix}.vertex_shift',
            )
            edge_data.append(
                {
                    'vertices': vertices_local,
                    'adjacent': adjacent,
                    'shift': adjacent_shift,
                }
            )

        prepared.append(
            {
                'position': cell_index,
                'id': cid,
                'gids': gids,
                'vertex_shifts': vertex_shifts,
                'edges': tuple(edge_data),
            }
        )
    return global_vertices, prepared


def normalize_edges(
    nv: NormalizedVertices,
    *,
    domain: Domain2D,
    tol: float | None = None,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Build a global edge pool based on an existing planar normalization."""

    copy_cells = require_bool(copy_cells, name='copy_cells')
    if tol is not None:
        tol = require_positive_finite_real(tol, name='tol')

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
    tol = require_positive_finite_real(tol, name='tol')

    global_edges: list[dict[str, Any]] = []
    edge_key_to_id: dict[tuple[Any, ...], int] = {}
    periodic = _is_periodic_domain(domain)
    _global_vertices, prepared = _prepare_topology_cells(
        nv,
        periodic=periodic,
    )
    sorted_cells = sorted(prepared, key=lambda item: item['id'])
    annotations: dict[int, list[int]] = {}

    for item in sorted_cells:
        edges = item['edges']
        gids = item['gids']
        vsh = item['vertex_shifts']

        edge_ids: list[int] = []
        cid_here = item['id']
        for edge in edges:
            adj = edge['adjacent']
            adj_shift = (
                edge['shift'] if periodic and adj >= 0 else (0, 0)
            )
            u, v = edge['vertices']

            ekey, erep = _canon_edge(
                (gids[u], vsh[u]),
                (gids[v], vsh[v]),
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
        annotations[item['position']] = edge_ids

    cells = [dict(cell) for cell in nv.cells] if copy_cells else nv.cells
    for position, edge_ids in annotations.items():
        cells[position]['edge_global_id'] = edge_ids

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

    copy_cells = require_bool(copy_cells, name='copy_cells')
    nv = normalize_vertices(
        cells,
        domain=domain,
        tol=tol,
        require_edge_shifts=require_edge_shifts,
        copy_cells=True,
    )
    normalized = normalize_edges(
        nv,
        domain=domain,
        tol=tol,
        copy_cells=False,
    )
    if copy_cells:
        return normalized

    annotation_fields = (
        'vertex_global_id',
        'vertex_shift',
        'edge_global_id',
    )
    for source, result in zip(cells, normalized.cells):
        for field in annotation_fields:
            source[field] = result[field]
    return NormalizedTopology(
        global_vertices=normalized.global_vertices,
        global_edges=normalized.global_edges,
        cells=cells,
    )
