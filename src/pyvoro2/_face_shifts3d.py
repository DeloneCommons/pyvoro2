"""3D periodic face-shift reconstruction helpers.

These helpers remain intentionally 3D-specific. They isolate the current
face-shift logic from the main API wrapper so the later planar implementation
can add a parallel edge-shift path without re-entangling ``api.py``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def _add_periodic_face_shifts_inplace(
    cells: list[dict[str, Any]],
    *,
    lattice_vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_mask: tuple[bool, bool, bool] = (True, True, True),
    mode: Literal['standard', 'power'] = 'standard',
    radii: np.ndarray | None = None,
    search: int = 2,
    tol: float | None = None,
    validate: bool = True,
    repair: bool = False,
) -> None:
    """Annotate periodic faces with integer neighbor-image shifts.

    This is a Python reference implementation used for correctness and testing.
    A future C++ fast-path can be added to match these results.

    The shift for a face is defined as the integer lattice vector (na, nb, nc)
    such that the adjacent cell on that face corresponds to the neighbor site
    translated by:

        p_neighbor_image = p_neighbor + na*a + nb*b + nc*c

    where (a, b, c) are lattice translation vectors in the coordinate system of
    the cell dictionaries.

    For partially periodic orthorhombic domains, `periodic_mask` can be used to
    restrict shifts to periodic axes; non-periodic axes are forced to shift=0.

    Args:
        cells: Cell dicts returned by the C++ layer.
        lattice_vectors: Tuple (a, b, c) lattice vectors.
        periodic_mask: Tuple (pa, pb, pc) of booleans. If False for an axis,
            the corresponding shift component is forced to 0.
        mode: 'standard' or 'power'.
        radii: Radii array for power mode.
        search: Search radius S; candidates in [-S..S]^3 are evaluated (with
            non-periodic axes restricted to 0).
        tol: Maximum allowed plane residual (absolute distance). If None, a
            conservative default based on the periodic length scale is used.
        validate: If True, validate plane residuals and reciprocity of shifts.
        repair: If True, attempt to repair rare reciprocity mismatches by
            enforcing opposite shifts on reciprocal faces.

    Raises:
        ValueError: if a consistent shift cannot be determined within the search
            radius, or if reciprocity validation fails.
    """
    if search < 0:
        raise ValueError('search must be >= 0')

    a = np.asarray(lattice_vectors[0], dtype=np.float64).reshape(3)
    b = np.asarray(lattice_vectors[1], dtype=np.float64).reshape(3)
    cvec = np.asarray(lattice_vectors[2], dtype=np.float64).reshape(3)

    pa, pb, pc = bool(periodic_mask[0]), bool(periodic_mask[1]), bool(periodic_mask[2])
    if not (pa or pb or pc):
        raise ValueError('periodic_mask has no periodic axes (all False)')

    # Lattice basis (columns) and inverse for nearest-image seeding.
    A = np.stack([a, b, cvec], axis=1)  # shape (3,3)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        raise ValueError('cell lattice vectors are singular') from e

    # Characteristic length for tolerance scaling (periodic axes only).
    #
    # NOTE:
    # We intentionally do **not** clamp this scale to 1.0. For very small or very
    # large coordinate systems the user should rescale inputs explicitly.
    Lcand: list[float] = []
    if pa:
        Lcand.append(float(np.linalg.norm(a)))
    if pb:
        Lcand.append(float(np.linalg.norm(b)))
    if pc:
        Lcand.append(float(np.linalg.norm(cvec)))
    L = float(max(Lcand)) if Lcand else 0.0

    tol_plane = (1e-6 * L) if tol is None else float(tol)
    if tol_plane < 0:
        raise ValueError('tol must be >= 0')

    # Map particle id -> site position (in the same coordinates as vertices).
    sites: dict[int, np.ndarray] = {}
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        s = np.asarray(c.get('site', []), dtype=np.float64)
        if s.size == 3:
            sites[pid] = s.reshape(3)

    # Precompute candidate shifts and their translation vectors.
    ra = range(-search, search + 1) if pa else range(0, 1)
    rb = range(-search, search + 1) if pb else range(0, 1)
    rc = range(-search, search + 1) if pc else range(0, 1)

    shifts: list[tuple[int, int, int]] = []
    trans: list[np.ndarray] = []
    for na in ra:
        for nb in rb:
            for nc in rc:
                shifts.append((int(na), int(nb), int(nc)))
                trans.append(na * a + nb * b + nc * cvec)

    trans_arr = np.stack(trans, axis=0) if trans else np.zeros((0, 3), dtype=np.float64)
    shift_to_idx = {s: i for i, s in enumerate(shifts)}
    l1 = np.asarray([abs(s[0]) + abs(s[1]) + abs(s[2]) for s in shifts], dtype=np.int64)

    # Weights for power mode (Laguerre diagram)
    if mode == 'power':
        if radii is None:
            raise ValueError('radii is required for mode="power"')
        w = np.asarray(radii, dtype=np.float64) ** 2
    else:
        w = None

    def _residual_for_trans(
        *,
        pid: int,
        nid: int,
        p_i: np.ndarray,
        p_j: np.ndarray,
        trans_subset: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Compute plane residuals for each candidate translation in trans_subset.

        Residual is the max absolute signed distance of face vertices to the
        expected bisector plane (midplane for standard Voronoi, or the power
        bisector for Laguerre diagrams).
        """
        pj = p_j.reshape(1, 3) + trans_subset  # (m,3)
        d = pj - p_i.reshape(1, 3)  # (m,3)
        dn = np.linalg.norm(d, axis=1)  # (m,)
        dn = np.where(dn == 0.0, 1.0, dn)

        # Project vertices along the direction vector for each candidate.
        # v: (k,3) -> proj: (m,k)
        proj = np.einsum('mk,nk->mn', d, v)

        if mode == 'standard':
            mid = 0.5 * (p_i.reshape(1, 3) + pj)  # (m,3)
            proj_mid = np.einsum('mk,mk->m', d, mid)  # (m,)
            dist = np.abs(proj - proj_mid[:, None]) / dn[:, None]
            return np.max(dist, axis=1)

        if mode == 'power':
            assert w is not None
            wi = float(w[pid])
            wj = float(w[nid])
            # Radical plane: d·x = (|pj|^2 - wj - (|pi|^2 - wi)) / 2
            rhs = 0.5 * (
                (np.sum(pj * pj, axis=1) - wj) - (np.dot(p_i, p_i) - wi)
            )  # (m,)
            dist = np.abs(proj - rhs[:, None]) / dn[:, None]
            return np.max(dist, axis=1)

        raise ValueError(f'unknown mode: {mode}')

    # Cache per-face residuals for potential debug / repair decisions.
    resid_by_face: dict[tuple[int, int], float] = {}

    # Solve shifts face-by-face.
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        faces = c.get('faces')
        if faces is None:
            continue

        p_i = sites.get(pid)
        if p_i is None:
            continue

        verts = np.asarray(c.get('vertices', []), dtype=np.float64)
        if verts.size == 0:
            verts = verts.reshape((0, 3))
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                'return_face_shifts requires vertex coordinates for each cell'
            )

        for fi, f in enumerate(faces):
            nid = int(f.get('adjacent_cell', -999999))
            if nid < 0:
                # Wall / invalid neighbor.
                f['adjacent_shift'] = (0, 0, 0)
                resid_by_face[(pid, fi)] = 0.0
                continue

            p_j = sites.get(nid)
            if p_j is None:
                raise ValueError(f'missing site for adjacent_cell={nid}')

            idx = np.asarray(f.get('vertices', []), dtype=np.int64)
            if idx.size == 0 or verts.shape[0] == 0:
                f['adjacent_shift'] = (0, 0, 0)
                resid_by_face[(pid, fi)] = 0.0
                continue
            v = verts[idx]

            # Periodic domains can have faces against *images of itself*.
            self_neighbor = nid == pid
            if self_neighbor and search == 0:
                raise ValueError(
                    'face_shift_search=0 cannot resolve faces against periodic images '
                    'of the same site; increase face_shift_search'
                )

            # Nearest-image seed: pick shift that brings p_j closest to p_i.
            frac = A_inv @ (p_j - p_i)
            base = (-np.rint(frac)).astype(np.int64)
            if not pa:
                base[0] = 0
            if not pb:
                base[1] = 0
            if not pc:
                base[2] = 0

            da_rng = (-1, 0, 1) if pa else (0,)
            db_rng = (-1, 0, 1) if pb else (0,)
            dc_rng = (-1, 0, 1) if pc else (0,)

            seed_idx: list[int] = []
            for da in da_rng:
                for db in db_rng:
                    for dc in dc_rng:
                        s = (int(base[0] + da), int(base[1] + db), int(base[2] + dc))
                        # max() bounds check is still correct even if some
                        # axes are restricted.
                        if max(abs(s[0]), abs(s[1]), abs(s[2])) > search:
                            continue
                        ii = shift_to_idx.get(s)
                        if ii is not None:
                            seed_idx.append(ii)

            # Exclude the zero shift for self-neighbor faces.
            idx0 = shift_to_idx.get((0, 0, 0))
            if self_neighbor and idx0 is not None:
                seed_idx = [ii for ii in seed_idx if ii != idx0]
            if not seed_idx:
                if self_neighbor:
                    raise ValueError(
                        'unable to seed face shift candidates for self-neighbor face; '
                        'increase face_shift_search'
                    )
                # Fall back to zero shift (may be the only allowed candidate when
                # periodic axes are restricted).
                if idx0 is None:
                    raise ValueError('internal error: missing (0,0,0) shift candidate')
                seed_idx = [idx0]

            # Deduplicate while preserving order
            seen: set[int] = set()
            seed_idx = [x for x in seed_idx if not (x in seen or seen.add(x))]

            resid_seed = _residual_for_trans(
                pid=pid,
                nid=nid,
                p_i=p_i,
                p_j=p_j,
                trans_subset=trans_arr[seed_idx],
                v=v,
            )
            best_local = int(np.argmin(resid_seed))
            best_idx = int(seed_idx[best_local])
            best_resid = float(resid_seed[best_local])

            if best_resid > tol_plane and len(shifts) > len(seed_idx):
                # Fall back to full candidate cube.
                resid_full = _residual_for_trans(
                    pid=pid, nid=nid, p_i=p_i, p_j=p_j, trans_subset=trans_arr, v=v
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

            if best_resid > tol_plane:
                raise ValueError(
                    'unable to determine adjacent_shift within tolerance; '
                    f'pid={pid}, nid={nid}, best_resid={best_resid:g}, '
                    f'tol={tol_plane:g}. Consider increasing face_shift_search.'
                )

            # Tie-break deterministically among *numerically indistinguishable*
            # candidates.
            #
            # Important: do NOT use a tolerance proportional to `tol_plane` here.
            # `tol_plane` is a permissive validation threshold; using it for
            # tie-breaking can incorrectly prefer a smaller-|shift| candidate even
            # when it has a clearly worse residual.
            scale = max(
                float(np.linalg.norm(p_i)),
                float(np.linalg.norm(p_j)),
                L,
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

            f['adjacent_shift'] = shifts[best_idx]
            resid_by_face[(pid, fi)] = best_resid

    if not validate and not repair:
        return

    # Build fast lookup of directed faces by (pid, nid, shift).
    def _skey(s: Any) -> tuple[int, int, int]:
        return int(s[0]), int(s[1]), int(s[2])

    face_key_to_loc: dict[tuple[int, int, tuple[int, int, int]], tuple[int, int]] = {}
    for c in cells:
        pid = int(c.get('id', -1))
        if pid < 0:
            continue
        faces = c.get('faces') or []
        for fi, f in enumerate(faces):
            nid = int(f.get('adjacent_cell', -999999))
            if nid < 0:
                continue
            s = _skey(f.get('adjacent_shift', (0, 0, 0)))
            key = (pid, nid, s)
            if key in face_key_to_loc:
                raise ValueError(f'duplicate directed face key: {key}')
            face_key_to_loc[key] = (pid, fi)

    def _missing_reciprocals() -> list[tuple[int, int, tuple[int, int, int]]]:
        missing: list[tuple[int, int, tuple[int, int, int]]] = []
        for pid, nid, s in face_key_to_loc.keys():
            recip = (nid, pid, (-s[0], -s[1], -s[2]))
            if recip not in face_key_to_loc:
                missing.append((pid, nid, s))
        return missing

    missing = _missing_reciprocals()

    # Reciprocity is a strict invariant for periodic standard Voronoi and
    # power diagrams.
    if missing and not repair:
        raise ValueError(
            f'face shift reciprocity check failed for {len(missing)} faces; '
            'set repair_face_shifts=True to attempt repair, '
            'or inspect face_shift_search/tolerance.'
        )

    if missing and repair:
        cell_by_id: dict[int, dict[str, Any]] = {
            int(c.get('id', -1)): c for c in cells if int(c.get('id', -1)) >= 0
        }

        # (cell_id, face_index) already modified
        used_faces: set[tuple[int, int]] = set()

        def _force_shift_on_neighbor_face(
            pid: int, nid: int, s: tuple[int, int, int]
        ) -> None:
            """Force the reciprocal face in nid to have shift -s.

            The reciprocal face is chosen by minimal plane residual.
            """
            target = (-s[0], -s[1], -s[2])
            cc = cell_by_id.get(nid)
            if cc is None:
                raise ValueError(f'cannot repair: missing cell dict for nid={nid}')
            faces_n = cc.get('faces') or []
            verts_n = np.asarray(cc.get('vertices', []), dtype=np.float64)
            if verts_n.size == 0:
                verts_n = verts_n.reshape((0, 3))
            if verts_n.ndim != 2 or verts_n.shape[1] != 3:
                raise ValueError('cannot repair: neighbor cell missing vertices')

            p_n = sites.get(nid)
            p_p = sites.get(pid)
            if p_n is None or p_p is None:
                raise ValueError('cannot repair: missing site positions')

            cand: list[tuple[float, int]] = []
            for fi2, f2 in enumerate(faces_n):
                if int(f2.get('adjacent_cell', -999999)) != pid:
                    continue
                if (nid, fi2) in used_faces:
                    continue
                idx2 = np.asarray(f2.get('vertices', []), dtype=np.int64)
                if idx2.size == 0 or verts_n.shape[0] == 0:
                    continue
                v2 = verts_n[idx2]
                # Evaluate residual for forcing target shift on this candidate face.
                trans_force = (
                    float(target[0]) * a
                    + float(target[1]) * b
                    + float(target[2]) * cvec
                )
                rr = _residual_for_trans(
                    pid=nid,
                    nid=pid,
                    p_i=p_n,
                    p_j=p_p,
                    trans_subset=trans_force.reshape(1, 3),
                    v=v2,
                )
                cand.append((float(rr[0]), fi2))

            if not cand:
                raise ValueError(
                    f'cannot repair: no candidate faces in cell {nid} pointing to {pid}'
                )

            cand.sort(key=lambda x: x[0])
            best_r, best_fi = cand[0]
            if best_r > tol_plane:
                raise ValueError(
                    f'cannot repair: best residual {best_r:g} exceeds tol '
                    f'{tol_plane:g} for reciprocal face nid={nid} -> pid={pid}'
                )

            faces_n[best_fi]['adjacent_shift'] = target
            used_faces.add((nid, best_fi))

        # Only repair faces in one direction (pid < nid) to avoid oscillations.
        for pid, nid, s in missing:
            if pid >= nid:
                continue
            _force_shift_on_neighbor_face(pid, nid, s)

        # Rebuild lookup after modifications.
        face_key_to_loc.clear()
        for c in cells:
            pid = int(c.get('id', -1))
            if pid < 0:
                continue
            faces = c.get('faces') or []
            for fi, f in enumerate(faces):
                nid = int(f.get('adjacent_cell', -999999))
                if nid < 0:
                    continue
                s = _skey(f.get('adjacent_shift', (0, 0, 0)))
                key = (pid, nid, s)
                if key in face_key_to_loc:
                    raise ValueError(f'duplicate directed face key after repair: {key}')
                face_key_to_loc[key] = (pid, fi)

        missing2 = _missing_reciprocals()
        if missing2 and mode in ('standard', 'power'):
            raise ValueError(
                f'face shift reciprocity repair failed for {len(missing2)} faces'
            )
