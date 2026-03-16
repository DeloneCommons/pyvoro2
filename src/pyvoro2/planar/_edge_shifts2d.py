"""2D periodic edge-shift reconstruction helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import numpy as np


def _add_periodic_edge_shifts_inplace(
    cells: list[dict[str, Any]],
    *,
    lattice_vectors: tuple[np.ndarray, np.ndarray],
    periodic_mask: tuple[bool, bool] = (True, True),
    mode: Literal['standard', 'power'] = 'standard',
    radii: np.ndarray | None = None,
    search: int = 2,
    tol: float | None = None,
    validate: bool = True,
    repair: bool = False,
) -> None:
    """Annotate periodic edges with integer neighbor-image shifts.

    The shift for an edge is the integer lattice vector ``(na, nb)`` such that
    the adjacent cell on that edge corresponds to the neighbor site translated
    by ``na * a + nb * b``, where ``(a, b)`` are the domain lattice vectors in
    the same coordinate system as the returned vertices.
    """

    if search < 0:
        raise ValueError('search must be >= 0')
    _ = repair  # accepted for API symmetry with the 3D helper

    a = np.asarray(lattice_vectors[0], dtype=np.float64).reshape(2)
    b = np.asarray(lattice_vectors[1], dtype=np.float64).reshape(2)
    px, py = bool(periodic_mask[0]), bool(periodic_mask[1])
    if not (px or py):
        raise ValueError('periodic_mask has no periodic axes (all False)')

    basis = np.stack([a, b], axis=1)
    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError as exc:
        raise ValueError('cell lattice vectors are singular') from exc

    lcand: list[float] = []
    if px:
        lcand.append(float(np.linalg.norm(a)))
    if py:
        lcand.append(float(np.linalg.norm(b)))
    length_scale = float(max(lcand)) if lcand else 0.0

    tol_line = (1e-6 * length_scale) if tol is None else float(tol)
    if tol_line < 0.0:
        raise ValueError('tol must be >= 0')

    sites: dict[int, np.ndarray] = {}
    for cell in cells:
        pid = int(cell.get('id', -1))
        if pid < 0:
            continue
        site = np.asarray(cell.get('site', []), dtype=np.float64)
        if site.size == 2:
            sites[pid] = site.reshape(2)

    rx = range(-search, search + 1) if px else range(0, 1)
    ry = range(-search, search + 1) if py else range(0, 1)
    shifts: list[tuple[int, int]] = []
    trans: list[np.ndarray] = []
    for sx in rx:
        for sy in ry:
            shifts.append((int(sx), int(sy)))
            trans.append(sx * a + sy * b)

    trans_arr = np.stack(trans, axis=0) if trans else np.zeros((0, 2), dtype=float)
    shift_to_idx = {shift: i for i, shift in enumerate(shifts)}
    l1 = np.asarray([abs(sx) + abs(sy) for sx, sy in shifts], dtype=np.int64)

    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        weights = np.asarray(radii, dtype=np.float64) ** 2
    else:
        weights = None

    def _residual_for_trans(
        *,
        pid: int,
        nid: int,
        p_i: np.ndarray,
        p_j: np.ndarray,
        trans_subset: np.ndarray,
        verts: np.ndarray,
    ) -> np.ndarray:
        p_img = p_j.reshape(1, 2) + trans_subset
        d = p_img - p_i.reshape(1, 2)
        dn = np.linalg.norm(d, axis=1)
        dn = np.where(dn == 0.0, 1.0, dn)

        proj = np.einsum('mk,nk->mn', d, verts)
        if mode == 'standard':
            rhs = 0.5 * (
                np.sum(p_img * p_img, axis=1) - np.dot(p_i, p_i)
            )
        elif mode == 'power':
            assert weights is not None
            wi = float(weights[pid])
            wj = float(weights[nid])
            rhs = 0.5 * (
                (np.sum(p_img * p_img, axis=1) - wj)
                - (np.dot(p_i, p_i) - wi)
            )
        else:  # pragma: no cover
            raise ValueError(f'unknown mode: {mode}')

        dist = np.abs(proj - rhs[:, None]) / dn[:, None]
        return np.max(dist, axis=1)

    residuals_by_edge: dict[tuple[int, int], float] = {}

    for cell in cells:
        pid = int(cell.get('id', -1))
        if pid < 0:
            continue
        p_i = sites.get(pid)
        if p_i is None:
            continue
        vertices = np.asarray(cell.get('vertices', []), dtype=np.float64)
        if vertices.size == 0:
            vertices = vertices.reshape((0, 2))
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                'return_edge_shifts requires vertex coordinates for each cell'
            )

        edges = cell.get('edges') or []
        for ei, edge in enumerate(edges):
            nid = int(edge.get('adjacent_cell', -999999))
            if nid < 0:
                edge['adjacent_shift'] = (0, 0)
                residuals_by_edge[(pid, ei)] = 0.0
                continue

            p_j = sites.get(nid)
            if p_j is None:
                raise ValueError(f'missing site for adjacent_cell={nid}')

            idx = np.asarray(edge.get('vertices', []), dtype=np.int64)
            if idx.shape != (2,):
                edge['adjacent_shift'] = (0, 0)
                residuals_by_edge[(pid, ei)] = 0.0
                continue
            verts = vertices[idx]

            self_neighbor = nid == pid
            if self_neighbor and search == 0:
                raise ValueError(
                    'search=0 cannot resolve edges against periodic images '
                    'of the same site; increase search'
                )

            frac = basis_inv @ (p_j - p_i)
            base = (-np.rint(frac)).astype(np.int64)
            if not px:
                base[0] = 0
            if not py:
                base[1] = 0

            dx_rng = (-1, 0, 1) if px else (0,)
            dy_rng = (-1, 0, 1) if py else (0,)
            seed_idx: list[int] = []
            for dx in dx_rng:
                for dy in dy_rng:
                    shift = (int(base[0] + dx), int(base[1] + dy))
                    if max(abs(shift[0]), abs(shift[1])) > search:
                        continue
                    ii = shift_to_idx.get(shift)
                    if ii is not None:
                        seed_idx.append(ii)

            idx0 = shift_to_idx.get((0, 0))
            if self_neighbor and idx0 is not None:
                seed_idx = [ii for ii in seed_idx if ii != idx0]
            if not seed_idx:
                if self_neighbor:
                    raise ValueError(
                        'unable to seed edge shift candidates for self-neighbor '
                        'edge; increase search'
                    )
                if idx0 is None:
                    raise ValueError('internal error: missing (0, 0) shift candidate')
                seed_idx = [idx0]

            seen: set[int] = set()
            seed_idx = [ii for ii in seed_idx if not (ii in seen or seen.add(ii))]

            resid_seed = _residual_for_trans(
                pid=pid,
                nid=nid,
                p_i=p_i,
                p_j=p_j,
                trans_subset=trans_arr[seed_idx],
                verts=verts,
            )
            best_local = int(np.argmin(resid_seed))
            best_idx = int(seed_idx[best_local])
            best_resid = float(resid_seed[best_local])

            if best_resid > tol_line and len(shifts) > len(seed_idx):
                resid_full = _residual_for_trans(
                    pid=pid,
                    nid=nid,
                    p_i=p_i,
                    p_j=p_j,
                    trans_subset=trans_arr,
                    verts=verts,
                )
                if self_neighbor and idx0 is not None and idx0 < resid_full.shape[0]:
                    resid_full[idx0] = np.inf
                best_idx = int(np.argmin(resid_full))
                best_resid = float(resid_full[best_idx])
                resid_for_tie = resid_full
                cand_idx = list(range(len(shifts)))
            else:
                resid_for_tie = resid_seed
                cand_idx = seed_idx

            if best_resid > tol_line:
                raise ValueError(
                    'unable to determine adjacent_shift within tolerance; '
                    f'pid={pid}, nid={nid}, best_resid={best_resid:g}, '
                    f'tol={tol_line:g}. Consider increasing search.'
                )

            scale = max(
                float(np.linalg.norm(p_i)),
                float(np.linalg.norm(p_j)),
                length_scale,
                1e-30,
            )
            eps_tie = max(1e-12 * scale, 64.0 * np.finfo(float).eps * scale)
            near = [
                cand_idx[k]
                for k, rr in enumerate(resid_for_tie)
                if float(rr) <= best_resid + eps_tie
            ]
            if len(near) > 1:
                near.sort(key=lambda ii: (int(l1[ii]), shifts[ii]))
                best_idx = int(near[0])

            edge['adjacent_shift'] = shifts[best_idx]
            residuals_by_edge[(pid, ei)] = best_resid

    if not validate:
        return

    directed_counts: dict[tuple[int, int], Counter[tuple[int, int]]] = {}
    for cell in cells:
        pid = int(cell.get('id', -1))
        if pid < 0:
            continue
        for edge in cell.get('edges') or []:
            nid = int(edge.get('adjacent_cell', -999999))
            if nid < 0:
                continue
            shift = tuple(int(v) for v in edge.get('adjacent_shift', (0, 0)))
            directed_counts.setdefault((pid, nid), Counter())[shift] += 1

    for (pid, nid), counts in directed_counts.items():
        rev = directed_counts.get((nid, pid), Counter())
        expected = Counter({(-sx, -sy): c for (sx, sy), c in counts.items()})
        if rev != expected:
            raise ValueError(
                'edge-shift reciprocity validation failed for '
                f'({pid}, {nid}); expected {expected}, got {rev}'
            )
