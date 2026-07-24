"""Shared helpers for raw cell-dictionary post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np


def remap_ids_inplace(
    cells: list[dict[str, Any]],
    ids_user: np.ndarray,
    *,
    boundary_key: str,
) -> None:
    """Remap internal IDs (``0..n-1``) to user IDs in-place.

    Args:
        cells: Raw cell dictionaries returned by the C++ layer.
        ids_user: User-supplied IDs aligned with internal indices.
        boundary_key: Name of the neighbor-bearing boundary list, e.g.
            ``"faces"`` in 3D or ``"edges"`` in 2D.
    """

    for cell in cells:
        pid = int(cell.get('id', -1))
        if 0 <= pid < ids_user.size:
            cell['id'] = int(ids_user[pid])

        boundaries = cell.get(boundary_key)
        if boundaries is None:
            continue

        for item in boundaries:
            adj = int(item.get('adjacent_cell', -999999))
            if 0 <= adj < ids_user.size:
                item['adjacent_cell'] = int(ids_user[adj])


def add_empty_cells_inplace(
    cells: list[dict[str, Any]],
    *,
    n: int,
    sites: np.ndarray,
    opts: tuple[bool, bool, bool],
    measure_key: str,
    boundary_key: str,
) -> None:
    """Insert explicit empty-cell records for missing particle IDs.

    In power (Laguerre) diagrams, some sites may have empty cells and the core
    backend can omit them from iteration. This helper restores a full
    length-``n`` output (IDs ``0..n-1``), marking missing entries as empty.

    Args:
        cells: List of per-cell dictionaries returned by the C++ layer.
        n: Total number of input sites.
        sites: Site positions aligned with internal IDs (shape ``(n, d)``).
        opts: ``(return_vertices, return_adjacency, return_boundaries)``.
        measure_key: Cell measure key, e.g. ``"volume"`` or ``"area"``.
        boundary_key: Boundary list key, e.g. ``"faces"`` or ``"edges"``.
    """

    if n <= 0:
        return

    present = {int(cell.get('id', -1)) for cell in cells}
    missing = [i for i in range(n) if i not in present]
    if not missing:
        return

    ret_vertices, ret_adjacency, ret_boundaries = opts
    d = int(np.asarray(sites).shape[1])
    for i in missing:
        rec: dict[str, Any] = {
            'id': int(i),
            'empty': True,
            measure_key: 0.0,
            'site': np.asarray(sites[i], dtype=np.float64).reshape(d).tolist(),
        }
        if ret_vertices:
            rec['vertices'] = []
        if ret_adjacency:
            rec['adjacency'] = []
        if ret_boundaries:
            rec[boundary_key] = []
        cells.append(rec)

    cells.sort(key=lambda cell: int(cell.get('id', 0)))
