"""Topology-level post-processing utilities.

Voro++ returns each Voronoi cell with its own *local* vertex list. In periodic
systems, many of those local vertices represent the same geometric vertex but in
different periodic images.

This module provides a correctness-first normalisation routine that builds a
global vertex pool in the 000 cell (primary periodic domain) and, for each cell,
maps local vertices to:
    - a global vertex index, and
    - an integer lattice shift (na, nb, nc) such that:

        v_local ~= v_global + na*a + nb*b + nc*c

All coordinates exposed by the public API are Cartesian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.spatial.domain_utils import (
    domain_length_scale,
    is_periodic_domain,
)
from ._internal.normalization import (
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
from ._internal.validation import require_bool, require_positive_finite_real


@dataclass(frozen=True)
class NormalizedVertices:
    """Result of :func:`normalize_vertices`.

    Attributes:
        global_vertices: Array of unique vertices in Cartesian coordinates,
            remapped into the primary cell for periodic domains.
        cells: A list of per-cell dictionaries. Each dictionary contains the
            original fields returned by :func:`pyvoro2.compute` plus:
                - vertex_global_id: list[int] of length n_local_vertices
                - vertex_shift: list[tuple[int,int,int]] aligned with vertices
    """

    global_vertices: np.ndarray
    cells: List[Dict[str, Any]]


@dataclass(frozen=True)
class NormalizedTopology:
    """Result of :func:`normalize_topology`.

    This extends :class:`NormalizedVertices` with globally deduplicated edges
    and faces. These are useful for building periodic Voronoi graphs.

    All coordinates exposed here are Cartesian. Periodicity is represented via
    integer lattice shifts (na, nb, nc) relative to the domain lattice vectors.

    Attributes:
        global_vertices: Array of unique vertices in Cartesian coordinates,
            remapped into the primary cell for periodic domains.
        global_edges: List of unique edges. Each edge dictionary contains:
            - vertices: (gid0, gid1) global vertex indices
            - vertex_shifts: ((0,0,0), (na,nb,nc)) such that the second endpoint
              is V[gid1] + na*a + nb*b + nc*c, while the first is V[gid0].
        global_faces: List of unique faces. Each face dictionary contains:
            - cells: (cid0, cid1) particle ids (first is the canonical anchor)
            - cell_shifts: ((0,0,0), (na,nb,nc)) shift of cid1 relative to cid0
            - vertices: list[int] global vertex ids in canonical cyclic order
            - vertex_shifts: list[(na,nb,nc)] shifts aligned with vertices,
              with the first vertex shift always (0,0,0).
        cells: Per-cell dictionaries (copies by default) including the
            vertex mapping fields plus:
            - edges: list[(u,v)] local vertex index pairs (u<v)
            - edge_global_id: list[int] aligned with edges
            - face_global_id: list[int] aligned with faces
    """

    global_vertices: np.ndarray
    global_edges: List[Dict[str, Any]]
    global_faces: List[Dict[str, Any]]
    cells: List[Dict[str, Any]]


def _prepare_vertex_cells(
    cells: List[Dict[str, Any]],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float,
    periodic: bool,
    require_face_shifts: bool,
) -> List[Dict[str, Any]]:
    """Validate all consumed raw records before constructing or mutating output."""

    if not isinstance(cells, list):
        raise ValueError('cells must be a list of dicts')

    prepared: List[Dict[str, Any]] = []
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
            dim=3,
        )
        face_data: List[Dict[str, Any]] = []
        remapped = vertices
        remap_shifts = np.zeros(vertices.shape, dtype=np.int64)

        if periodic:
            faces = cell.get('faces')
            if faces is None:
                raise ValueError(
                    'cells must include faces for periodic normalization'
                )
            try:
                face_records = tuple(faces)
            except TypeError:
                raise ValueError(
                    f'cells[{cell_index}].faces must be a sequence of dicts'
                ) from None
            for face_index, face in enumerate(face_records):
                prefix = f'cells[{cell_index}].faces[{face_index}]'
                if not isinstance(face, dict):
                    raise ValueError(f'{prefix} must be a dict')
                vertices_local = (
                    tuple()
                    if face.get('vertices') is None
                    else require_local_vertex_indices(
                        face['vertices'],
                        name=f'{prefix}.vertices',
                        n_vertices=int(vertices.shape[0]),
                    )
                )
                if 'adjacent_cell' not in face:
                    raise ValueError(f'{prefix}.adjacent_cell is required')
                adjacent = require_adjacent_cell_id(
                    face['adjacent_cell'],
                    name=f'{prefix}.adjacent_cell',
                )
                if 'adjacent_shift' in face:
                    adjacent_shift = require_shift(
                        face['adjacent_shift'],
                        name=f'{prefix}.adjacent_shift',
                        dim=3,
                    )
                elif require_face_shifts:
                    raise ValueError(
                        'cells must include face adjacent_shift '
                        '(compute with return_face_shifts=True)'
                    )
                else:
                    adjacent_shift = (0, 0, 0)
                face_data.append(
                    {
                        'record': face,
                        'vertices': vertices_local,
                        'adjacent': adjacent,
                        'shift': adjacent_shift,
                    }
                )

            for vertex_index in range(int(vertices.shape[0])):
                incident_shifts = [(0, 0, 0)] + [
                    face['shift']
                    for face in face_data
                    if vertex_index in face['vertices']
                ]
                validate_pairwise_shift_differences(
                    incident_shifts,
                    name=(
                        f'cells[{cell_index}].vertex[{vertex_index}] '
                        'incident shift'
                    ),
                )

            # type: ignore[arg-type]
            remapped, remap_shifts = domain.remap_cart(
                vertices,
                return_shifts=True,
            )
            for _ in range(2):
                # type: ignore[arg-type]
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
                'faces': tuple(face_data),
                'remapped': remapped,
                'remap_shifts': remap_shifts,
                'quantized': quantized,
            }
        )
    return prepared


def _canonical_incident_key(
    incident: Sequence[Tuple[int, Tuple[int, int, int]]]
) -> Tuple[Any, ...]:
    """Canonicalize an incident cell-image set up to global lattice translation.

    `incident` is a collection of (cell_id, shift) pairs expressed relative to
    some reference cell (i.e., shifts are only defined up to adding a constant
    vector to *all* shifts).

    We canonicalize by considering all anchors in the set: for each anchor
    shift s_a, subtract s_a from all shifts and take the lexicographically
    minimal resulting representation.

    This makes the key invariant to adding a constant shift to all elements.
    """
    # Deduplicate exactly identical tuples first
    uniq = sorted(set((cid, tuple(s)) for cid, s in incident))
    if not uniq:
        return tuple()

    best: Tuple[Any, ...] | None = None
    for _cid_a, s_a in uniq:
        rep = []
        for cid, s in uniq:
            ss = checked_shift_difference(
                s,
                s_a,
                name='incident lattice shift',
            )
            rep.append((cid, ss[0], ss[1], ss[2]))
        rep_sorted = tuple(sorted(rep))
        if best is None or rep_sorted < best:
            best = rep_sorted
    assert best is not None
    return best


def normalize_vertices(
    cells: List[Dict[str, Any]],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    require_face_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedVertices:
    """Build a global vertex pool and per-cell vertex mappings.

    Args:
        cells: Output list from :func:`pyvoro2.compute`. Must include local
            vertices (`return_vertices=True`). For periodic domains, faces and
            face shifts are required unless `require_face_shifts=False`.
        domain: The domain used for the computation.
        tol: Quantization tolerance used for coordinate keys and residual
            verification. If None, defaults to 1e-8 * L where L is a domain
            length scale.
        require_face_shifts: If True and domain is PeriodicCell, require
            face-level `adjacent_shift` entries to build robust topology keys.
        copy_cells: If True, return shallow copies of the cell dicts with
            added mapping fields. If False, mutate the input dictionaries.

    Returns:
        NormalizedVertices with global_vertices and augmented cell dicts.

    Raises:
        ValueError: if required fields are missing.
    """
    require_face_shifts = require_bool(
        require_face_shifts,
        name='require_face_shifts',
    )
    copy_cells = require_bool(copy_cells, name='copy_cells')
    if tol is not None:
        tol = require_positive_finite_real(tol, name='tol')

    L = domain_length_scale(domain)
    is_periodic = is_periodic_domain(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        # If the user relies on defaults under a suspicious unit system,
        # highlight that pyvoro2 expects explicit rescaling.
        if float(L) < 1e-3 or float(L) > 1e9:
            warnings.warn(
                'normalize_vertices is using a default tolerance proportional to the '
                f'domain length scale (L≈{float(L):.3g}). For very small/large units '
                'this may be too strict/too loose. Consider rescaling your coordinates '
                'or passing an explicit tol=... .',
                RuntimeWarning,
                stacklevel=2,
            )
    tol = require_positive_finite_real(tol, name='tol')

    prepared = _prepare_vertex_cells(
        cells,
        domain=domain,
        tol=tol,
        periodic=is_periodic,
        require_face_shifts=require_face_shifts,
    )

    # Global storage
    global_vertices: List[np.ndarray] = []
    key_to_gid: Dict[Tuple[Any, ...], int] = {}
    mappings: Dict[int, Tuple[List[int], List[Tuple[int, int, int]]]] = {}

    if not is_periodic:
        # For non-periodic boxes, coordinate-based deduplication is sufficient.
        for item in prepared:
            verts = item['vertices']
            gids: List[int] = []
            shifts: List[Tuple[int, int, int]] = []
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
                shifts.append((0, 0, 0))
            mappings[item['position']] = (gids, shifts)

        out_cells = [dict(cell) for cell in cells] if copy_cells else cells
        for position, (gids, shifts) in mappings.items():
            out_cells[position]['vertex_global_id'] = gids
            out_cells[position]['vertex_shift'] = shifts

        return NormalizedVertices(
            global_vertices=(
                np.stack(global_vertices, axis=0)
                if global_vertices
                else np.zeros((0, 3))
            ),
            cells=out_cells,
        )

    # Build per-cell local->global mapping
    # Process deterministically by sorted cell id then vertex index.
    for item in sorted(prepared, key=lambda record: record['id']):
        verts = item['vertices']
        faces = item['faces']

        # Build vertex -> incident faces list
        v_faces: List[List[Dict[str, Any]]] = [
            [] for _ in range(int(verts.shape[0]))
        ]
        for face in faces:
            for vertex_index in face['vertices']:
                v_faces[vertex_index].append(face)

        gids: List[int] = []
        shifts: List[Tuple[int, int, int]] = []
        remapped = item['remapped']
        rem_shifts = item['remap_shifts']

        for k in range(int(verts.shape[0])):
            v0 = remapped[k]
            s0 = tuple(int(x) for x in rem_shifts[k])

            # Build incident set: include this cell (id, shift=(0,0,0)) plus
            # each adjacent cell image meeting at this vertex.
            incident: List[Tuple[int, Tuple[int, int, int]]] = []
            cid_here = item['id']
            incident.append((cid_here, (0, 0, 0)))
            for face in v_faces[k]:
                incident.append((face['adjacent'], face['shift']))

            topo_key = _canonical_incident_key(incident)
            coord_key = item['quantized'][k]
            key: Tuple[Any, ...] = ('pbc',) + topo_key + ('@',) + coord_key

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

    gv = np.stack(global_vertices, axis=0) if global_vertices else np.zeros((0, 3))
    return NormalizedVertices(global_vertices=gv, cells=out_cells)


def _canon_edge(
    a: Tuple[int, Tuple[int, int, int]],
    b: Tuple[int, Tuple[int, int, int]],
) -> Tuple[
    Tuple[Any, ...], Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]
]:
    """Canonicalize an edge defined by two (gid, shift) endpoints.

    Returns:
        key: a hashable canonical key.
        rep: two endpoint records (gid, sx, sy, sz) where the first endpoint
            is guaranteed to have (0,0,0) shift.
    """
    gid0, s0 = a
    gid1, s1 = b
    # Two translation anchors: subtract s0 or subtract s1.
    candidates = []
    for ga, sa, gb, sb in ((gid0, s0, gid1, s1), (gid1, s1, gid0, s0)):
        d = checked_shift_difference(sb, sa, name='edge vertex shift')
        recs = (
            (int(ga), 0, 0, 0),
            (int(gb), d[0], d[1], d[2]),
        )
        candidates.append(tuple(sorted(recs)))
    best = min(candidates)

    # Normalize so the first record has zero shift.
    g0, x0, y0, z0 = best[0]
    g1, x1, y1, z1 = best[1]
    relative = checked_shift_difference(
        (x1, y1, z1),
        (x0, y0, z0),
        name='canonical edge vertex shift',
    )
    rep = ((int(g0), 0, 0, 0), (int(g1), *relative))
    key = (
        'e',
        int(rep[0][0]),
        int(rep[1][0]),
        int(rep[1][1]),
        int(rep[1][2]),
        int(rep[1][3]),
    )
    return key, rep


def _rotate(seq: Sequence[Any], k: int) -> List[Any]:
    k = int(k) % len(seq)
    return list(seq[k:]) + list(seq[:k])


def _canon_polygon(
    verts: Sequence[Tuple[int, Tuple[int, int, int]]]
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Canonicalize a face vertex cycle up to translation, rotation, and reversal.

    Each input vertex is (gid, shift).
    Output is a tuple of (gid, sx, sy, sz) records with the first shift (0,0,0).
    """
    if not verts:
        return tuple()
    vv = [(int(g), (int(s[0]), int(s[1]), int(s[2]))) for g, s in verts]
    n = len(vv)

    # Consider both orientations.
    candidates: List[Tuple[Tuple[int, int, int, int], ...]] = []
    for base in (vv, list(reversed(vv))):
        for r in range(n):
            seq = _rotate(base, r)
            rep = []
            for g, s in seq:
                ss = checked_shift_difference(
                    s,
                    seq[0][1],
                    name='face vertex shift',
                )
                rep.append((int(g), ss[0], ss[1], ss[2]))
            candidates.append(tuple(rep))
    return min(candidates)


def _canon_face_pair(
    cid_here: int,
    adj: int,
    adj_shift: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int, int, int]:
    sx, sy, sz = int(adj_shift[0]), int(adj_shift[1]), int(adj_shift[2])
    rep1 = (int(cid_here), 0, 0, 0, int(adj), sx, sy, sz)
    reverse = checked_shift_difference(
        (0, 0, 0),
        (sx, sy, sz),
        name='adjacent face shift',
    )
    rep2 = (int(adj), 0, 0, 0, int(cid_here), *reverse)
    return rep2 if rep2 < rep1 else rep1


def _prepare_topology_cells(
    nv: NormalizedVertices,
    *,
    periodic: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Validate every record consumed by edge/face topology construction."""

    global_vertices = coerce_normalization_vertices(
        nv.global_vertices,
        name='normalized.global_vertices',
        dim=3,
    )
    if not isinstance(nv.cells, list):
        raise ValueError('normalized.cells must be a list of dicts')

    prepared: List[Dict[str, Any]] = []
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
            dim=3,
        )
        faces = cell.get('faces')
        if faces is None:
            raise ValueError('cells must include faces')
        try:
            face_records = tuple(faces)
        except TypeError:
            raise ValueError(f'{prefix}.faces must be a sequence of dicts') from None

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
            dim=3,
        )

        face_data: List[Dict[str, Any]] = []
        for face_index, face in enumerate(face_records):
            face_prefix = f'{prefix}.faces[{face_index}]'
            if not isinstance(face, dict):
                raise ValueError(f'{face_prefix} must be a dict')
            vertices_local = (
                tuple()
                if face.get('vertices') is None
                else require_local_vertex_indices(
                    face['vertices'],
                    name=f'{face_prefix}.vertices',
                    n_vertices=int(vertices.shape[0]),
                )
            )
            if 'adjacent_cell' not in face:
                raise ValueError(f'{face_prefix}.adjacent_cell is required')
            adjacent = require_adjacent_cell_id(
                face['adjacent_cell'],
                name=f'{face_prefix}.adjacent_cell',
            )
            if 'adjacent_shift' in face:
                adjacent_shift = require_shift(
                    face['adjacent_shift'],
                    name=f'{face_prefix}.adjacent_shift',
                    dim=3,
                )
            elif periodic and adjacent >= 0:
                raise ValueError(
                    'Periodic domain face missing adjacent_shift; '
                    'compute with return_face_shifts=True'
                )
            else:
                adjacent_shift = (0, 0, 0)
            effective_shift = (
                adjacent_shift
                if periodic and adjacent >= 0
                else (0, 0, 0)
            )
            checked_shift_difference(
                (0, 0, 0),
                effective_shift,
                name=f'{face_prefix}.adjacent_shift',
            )
            validate_pairwise_shift_differences(
                [vertex_shifts[value] for value in vertices_local],
                name=f'{face_prefix}.vertex_shift',
            )
            face_data.append(
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
                'faces': tuple(face_data),
            }
        )
    return global_vertices, prepared


def normalize_edges_faces(
    nv: NormalizedVertices,
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Build global edge and face pools based on an existing vertex normalization."""
    copy_cells = require_bool(copy_cells, name='copy_cells')
    if tol is not None:
        tol = require_positive_finite_real(tol, name='tol')

    L = domain_length_scale(domain)
    if tol is None:
        if not np.isfinite(L) or float(L) <= 0.0:
            raise ValueError('domain has an invalid length scale; pass tol explicitly')
        tol = 1e-8 * float(L)
        if float(L) < 1e-3 or float(L) > 1e9:
            msg = (
                'normalize_edges_faces is using a default tolerance proportional '
                'to the '
                'domain length scale '
                f'(L≈{float(L):.3g}). '
                'For very small/large units this may be too strict/too loose. '
                'Consider rescaling your coordinates or passing an explicit tol=... .'
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    tol = require_positive_finite_real(tol, name='tol')

    global_edges: List[Dict[str, Any]] = []
    edge_key_to_id: Dict[Tuple[Any, ...], int] = {}

    global_faces: List[Dict[str, Any]] = []
    face_key_to_id: Dict[Tuple[Any, ...], int] = {}

    domain_periodic = is_periodic_domain(domain)
    _global_vertices, prepared = _prepare_topology_cells(
        nv,
        periodic=domain_periodic,
    )
    annotations: Dict[int, Dict[str, Any]] = {}

    # Deterministic processing order
    cells_sorted = sorted(prepared, key=lambda item: item['id'])

    # Build edges and faces
    for item in cells_sorted:
        faces = item['faces']
        gids = item['gids']
        vsh = item['vertex_shifts']

        # Extract local edges from face vertex cycles.
        edge_set: set[Tuple[int, int]] = set()
        for face in faces:
            vv = list(face['vertices'])
            if len(vv) < 2:
                continue
            for u, v in zip(vv, vv[1:] + vv[:1]):
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_set.add((a, b))

        edges_local = sorted(edge_set)

        # Map edges to global ids
        edge_ids: List[int] = []
        for u, v in edges_local:
            ea = (gids[u], vsh[u])
            eb = (gids[v], vsh[v])
            ekey, erep = _canon_edge(ea, eb)
            eid = edge_key_to_id.get(ekey)
            if eid is None:
                eid = len(global_edges)
                edge_key_to_id[ekey] = eid
                global_edges.append(
                    {
                        'vertices': (int(erep[0][0]), int(erep[1][0])),
                        'vertex_shifts': (
                            (0, 0, 0),
                            (int(erep[1][1]), int(erep[1][2]), int(erep[1][3])),
                        ),
                    }
                )
            edge_ids.append(eid)
        # Faces -> global ids
        face_ids: List[int] = []
        cid_here = item['id']
        for face in faces:
            adj = face['adjacent']
            adj_shift = (
                face['shift']
                if domain_periodic and adj >= 0
                else (0, 0, 0)
            )

            pair = _canon_face_pair(cid_here, adj, adj_shift)
            vids = face['vertices']
            if not vids:
                poly = tuple()
            else:
                desc = [(gids[v], vsh[v]) for v in vids]
                poly = _canon_polygon(desc)

            if domain_periodic and adj >= 0:
                # In periodic tessellations, a given (cell_id, neighbor_id,
                # neighbor_shift) pair uniquely identifies a face (two convex
                # polyhedra share at most one face). Use only the canonical
                # cell-pair key for deduplication to avoid sensitivity to
                # boundary-vertex remapping noise in polygon keys.
                fkey: Tuple[Any, ...] = ('f',) + pair
            else:
                # For non-periodic domains *or* wall faces in partially periodic
                # orthorhombic domains, adjacent_cell may be a shared wall id.
                # Include the canonical polygon to distinguish distinct boundary faces.
                fkey: Tuple[Any, ...] = ('f',) + pair + ('|',) + poly
            fid = face_key_to_id.get(fkey)
            if fid is None:
                fid = len(global_faces)
                face_key_to_id[fkey] = fid
                # Convert canonical poly back to lists
                gv_ids = [int(t[0]) for t in poly]
                gv_sh = [(int(t[1]), int(t[2]), int(t[3])) for t in poly]
                global_faces.append(
                    {
                        'cells': (int(pair[0]), int(pair[4])),
                        'cell_shifts': (
                            (0, 0, 0),
                            (int(pair[5]), int(pair[6]), int(pair[7])),
                        ),
                        'vertices': gv_ids,
                        'vertex_shifts': gv_sh,
                    }
                )
            else:
                # Keep the stored polygon deterministic across both (cid->adj)
                # and (adj->cid) occurrences by choosing the lexicographically
                # smallest canonical polygon.
                if poly:
                    old = global_faces[fid]
                    old_poly = tuple(
                        (int(gid), int(sh[0]), int(sh[1]), int(sh[2]))
                        for gid, sh in zip(
                            old.get('vertices', []), old.get('vertex_shifts', [])
                        )
                    )
                    if (not old_poly) or poly < old_poly:
                        old['vertices'] = [int(t[0]) for t in poly]
                        old['vertex_shifts'] = [
                            (int(t[1]), int(t[2]), int(t[3])) for t in poly
                        ]
            face_ids.append(fid)
        annotations[item['position']] = {
            'edges': [(int(u), int(v)) for u, v in edges_local],
            'edge_global_id': edge_ids,
            'face_global_id': face_ids,
        }

    cells = [dict(cell) for cell in nv.cells] if copy_cells else nv.cells
    for position, values in annotations.items():
        cells[position].update(values)

    return NormalizedTopology(
        global_vertices=nv.global_vertices,
        global_edges=global_edges,
        global_faces=global_faces,
        cells=cells,
    )


def normalize_topology(
    cells: List[Dict[str, Any]],
    *,
    domain: Box | OrthorhombicCell | PeriodicCell,
    tol: float | None = None,
    require_face_shifts: bool = True,
    copy_cells: bool = True,
) -> NormalizedTopology:
    """Convenience wrapper: normalize vertices, then edges/faces."""
    copy_cells = require_bool(copy_cells, name='copy_cells')
    nv = normalize_vertices(
        cells,
        domain=domain,
        tol=tol,
        require_face_shifts=require_face_shifts,
        copy_cells=True,
    )
    normalized = normalize_edges_faces(
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
        'edges',
        'edge_global_id',
        'face_global_id',
    )
    for source, result in zip(cells, normalized.cells):
        for field in annotation_fields:
            source[field] = result[field]
    return NormalizedTopology(
        global_vertices=normalized.global_vertices,
        global_edges=normalized.global_edges,
        global_faces=normalized.global_faces,
        cells=cells,
    )
