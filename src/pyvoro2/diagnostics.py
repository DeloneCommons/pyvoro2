"""Tessellation diagnostics and sanity checks.

These utilities help detect when the computed collection of cells may not form
an expected partition of the domain (e.g. due to numerical issues or missing
cells in the output when empty cells are omitted).

Key ideas:
  - Required invariant failures are explicit error-severity issues and make
    ``diagnostics.ok`` false.
  - Warning- and info-only findings do not fail ``diagnostics.ok``.
  - For periodic domains, faces should be reciprocal: if cell i has a face to
    (j, s), then cell j should have a face to (i, -s).

The public entry point is :func:`analyze_tessellation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import warnings

import numpy as np

from .domains import Box, OrthorhombicCell, PeriodicCell
from ._internal.inputs import coerce_external_id_array
from ._internal.tessellation_diagnostics import (
    classify_expected_ids,
    diagnostics_ok,
    reciprocity_issue_severity,
    reset_owned_annotations,
    stable_measure_sum,
    validate_cell_measure,
)
from ._internal.validation import (
    require_bool,
    require_nonnegative_finite_real,
    require_optional_bool,
    require_optional_nonnegative_finite_real,
    require_string,
    require_string_choice,
)


@dataclass(frozen=True, slots=True)
class TessellationIssue:
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
class TessellationDiagnostics:
    domain_volume: float
    sum_cell_volume: float
    volume_ratio: float
    volume_gap: float
    volume_overlap: float
    n_sites_expected: int
    n_cells_returned: int
    missing_ids: tuple[int, ...]
    empty_ids: tuple[int, ...]
    face_shift_available: bool
    reciprocity_checked: bool
    n_faces_total: int
    n_faces_orphan: int
    n_faces_mismatched: int
    issues: tuple[TessellationIssue, ...]
    ok_volume: bool
    ok_reciprocity: bool
    ok: bool


class TessellationError(ValueError):
    """Raised when tessellation sanity checks fail under strict settings."""

    def __init__(self, message: str, diagnostics: TessellationDiagnostics):
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


def _domain_volume(domain: Box | OrthorhombicCell | PeriodicCell) -> float:
    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
        return float((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
    vec = np.asarray(domain.vectors, dtype=np.float64)
    # vectors are rows -> det of rows
    return float(abs(np.linalg.det(vec)))


def _characteristic_length(domain: Box | OrthorhombicCell | PeriodicCell) -> float:
    if isinstance(domain, (Box, OrthorhombicCell)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = domain.bounds
        L = float(max(xmax - xmin, ymax - ymin, zmax - zmin))
        return L if np.isfinite(L) else 0.0
    vec = np.asarray(domain.vectors, dtype=np.float64)
    L = float(
        max(
            np.linalg.norm(vec[0]),
            np.linalg.norm(vec[1]),
            np.linalg.norm(vec[2]),
        )
    )
    return L if np.isfinite(L) else 0.0


def _is_periodic_domain(domain: Box | OrthorhombicCell | PeriodicCell) -> bool:
    """Return True if the domain has any periodicity."""
    if isinstance(domain, PeriodicCell):
        return True
    if isinstance(domain, OrthorhombicCell):
        return bool(any(bool(x) for x in domain.periodic))
    return False


def _lattice_vectors_cart(
    domain: Box | OrthorhombicCell | PeriodicCell,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lattice vectors (a,b,c) in Cartesian coordinates.

    For :class:`~pyvoro2.domains.PeriodicCell`, these are the user-provided
    triclinic vectors.

    For :class:`~pyvoro2.domains.OrthorhombicCell`, these are axis-aligned
    vectors of lengths (Lx,Ly,Lz) derived from the bounds.

    For :class:`~pyvoro2.domains.Box`, the same axis-aligned vectors are
    returned, but they are only meaningful if face shifts are provided.
    """
    if isinstance(domain, PeriodicCell):
        vec = np.asarray(domain.vectors, dtype=np.float64)
        return vec[0], vec[1], vec[2]

    bounds = domain.bounds  # type: ignore[attr-defined]
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    a = np.array([xmax - xmin, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, ymax - ymin, 0.0], dtype=np.float64)
    c = np.array([0.0, 0.0, zmax - zmin], dtype=np.float64)
    return a, b, c


def _plane_from_vertices(v: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return (unit normal, d) for plane n·x = d, or None if degenerate."""
    if v.shape[0] < 3:
        return None
    # Newell's method
    n = np.zeros(3, dtype=np.float64)
    for i in range(v.shape[0]):
        j = (i + 1) % v.shape[0]
        n[0] += (v[i, 1] - v[j, 1]) * (v[i, 2] + v[j, 2])
        n[1] += (v[i, 2] - v[j, 2]) * (v[i, 0] + v[j, 0])
        n[2] += (v[i, 0] - v[j, 0]) * (v[i, 1] + v[j, 1])
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        return None
    n = n / nn
    d = float(np.mean(v @ n))
    return n, d


def analyze_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Box | OrthorhombicCell | PeriodicCell,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    volume_tol_rel: float = 1e-8,
    volume_tol_abs: float = 1e-12,
    check_reciprocity: bool = True,
    check_plane_mismatch: bool = True,
    plane_offset_tol: float | None = None,
    plane_angle_tol: float | None = None,
    mark_faces: bool = True,
) -> TessellationDiagnostics:
    """Analyze tessellation sanity and (optionally) annotate faces.

    This function is designed to be conservative: it reports issues but does not
    modify geometry. If `mark_faces=True`, it will annotate face dicts with
    local flags such as `orphan` and `reciprocal_mismatch`.

    Args:
        cells: Output of :func:`pyvoro2.compute`.
        domain: Domain used for computation.
        expected_ids: Optional list of expected cell ids (useful when ids were
            remapped by the user). If provided, missing ids are reported.
        mode: Optional mode string ('standard'|'power') selecting expected-ID
            semantics. Missing standard IDs are errors, missing power IDs are
            informational hidden cells, and ``None`` leaves them as warnings.
        volume_tol_rel: Relative tolerance for domain volume comparison.
        volume_tol_abs: Absolute tolerance for domain volume comparison.
        check_reciprocity: Whether to check face reciprocity (periodic domains
            only). An explicitly requested public check is required, so missing
            shifts, reciprocals, or geometric agreement are errors.
        check_plane_mismatch: Whether to check that reciprocal faces represent the
            same geometric plane (periodic domains only).
        plane_offset_tol: Absolute tolerance for reciprocal plane offset mismatch.
            If None, a conservative default based on domain length is used.
        plane_angle_tol: Tolerance for reciprocal plane normal mismatch (radians).
            If None, a conservative default is used.
        mark_faces: If True, reset and then annotate the analyzer-owned face
            flags in place. If False, existing caller annotations are untouched.

    Returns:
        TessellationDiagnostics
    """
    return _analyze_tessellation(
        cells,
        domain,
        expected_ids=expected_ids,
        mode=mode,
        volume_tol_rel=volume_tol_rel,
        volume_tol_abs=volume_tol_abs,
        check_reciprocity=check_reciprocity,
        reciprocity_required=True,
        check_plane_mismatch=check_plane_mismatch,
        plane_offset_tol=plane_offset_tol,
        plane_angle_tol=plane_angle_tol,
        mark_faces=mark_faces,
    )


def _analyze_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Box | OrthorhombicCell | PeriodicCell,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    volume_tol_rel: float = 1e-8,
    volume_tol_abs: float = 1e-12,
    check_reciprocity: bool = True,
    reciprocity_required: bool,
    check_plane_mismatch: bool = True,
    plane_offset_tol: float | None = None,
    plane_angle_tol: float | None = None,
    mark_faces: bool = True,
) -> TessellationDiagnostics:
    """Private analyzer with an explicit required/optional reciprocity policy."""

    if mode is not None:
        mode = require_string_choice(
            mode,
            name='mode',
            choices=('standard', 'power'),
        )
    volume_tol_rel = require_nonnegative_finite_real(
        volume_tol_rel,
        name='volume_tol_rel',
    )
    volume_tol_abs = require_nonnegative_finite_real(
        volume_tol_abs,
        name='volume_tol_abs',
    )
    check_reciprocity = require_bool(
        check_reciprocity,
        name='check_reciprocity',
    )
    check_plane_mismatch = require_bool(
        check_plane_mismatch,
        name='check_plane_mismatch',
    )
    plane_offset_tol = require_optional_nonnegative_finite_real(
        plane_offset_tol,
        name='plane_offset_tol',
    )
    plane_angle_tol = require_optional_nonnegative_finite_real(
        plane_angle_tol,
        name='plane_angle_tol',
    )
    mark_faces = require_bool(mark_faces, name='mark_faces')
    expected_ids_array = (
        None
        if expected_ids is None
        else coerce_external_id_array(expected_ids, name='expected_ids')
    )

    issues: list[TessellationIssue] = []

    # --- Volume sanity ---
    dom_vol = _domain_volume(domain)
    valid_volumes: list[float] = []
    measures_valid = True
    empty_ids: list[int] = []
    present_ids: list[int] = []
    for c in cells:
        cid = int(c.get('id', -1))
        if cid >= 0:
            present_ids.append(cid)
        empty = bool(c.get('empty', False))
        if empty:
            if cid >= 0:
                empty_ids.append(cid)
        measure = validate_cell_measure(c, field='volume', empty=empty)
        if not measure.valid:
            measures_valid = False
            for code in measure.issue_codes:
                if code == 'MISSING_CELL_MEASURE':
                    message = f'Non-empty cell {cid} is missing its volume'
                elif code == 'INVALID_CELL_MEASURE':
                    message = f'Cell {cid} has an invalid non-real volume'
                elif code == 'NONFINITE_CELL_MEASURE':
                    message = f'Cell {cid} has a non-finite volume'
                elif code == 'NEGATIVE_CELL_MEASURE':
                    message = f'Cell {cid} has a negative volume'
                else:
                    message = f'Empty cell {cid} has a nonzero volume'
                issues.append(
                    TessellationIssue(
                        code,
                        'error',
                        message,
                        examples=((cid,) if cid >= 0 else ()),
                    )
                )
        elif not empty:
            assert measure.value is not None
            valid_volumes.append(measure.value)

    sum_vol = stable_measure_sum(valid_volumes)
    domain_volume_valid = bool(np.isfinite(dom_vol) and dom_vol > 0.0)
    if not domain_volume_valid:
        # Degenerate domain; treat as error.
        issues.append(
            TessellationIssue('DOMAIN_VOLUME', 'error', 'Domain volume is non-positive')
        )
        dom_vol = max(dom_vol, 0.0)

    vol_tol = max(float(volume_tol_abs), float(volume_tol_rel) * dom_vol)
    diff = sum_vol - dom_vol
    aggregate_overflow = bool(np.isposinf(sum_vol))
    ok_volume = bool(
        measures_valid
        and domain_volume_valid
        and not aggregate_overflow
        and abs(diff) <= vol_tol
    )
    gap = max(0.0, dom_vol - sum_vol)
    overlap = max(0.0, sum_vol - dom_vol)
    if measures_valid and domain_volume_valid:
        if aggregate_overflow:
            issues.append(
                TessellationIssue(
                    'OVERLAP',
                    'error',
                    'Sum of cell volumes exceeds the finite representable range',
                )
            )
        elif not ok_volume:
            if gap > vol_tol:
                issues.append(
                    TessellationIssue(
                        'GAP',
                        'error',
                        f'Sum of cell volumes is smaller than domain volume by {gap:g}',
                    )
                )
            if overlap > vol_tol:
                issues.append(
                    TessellationIssue(
                        'OVERLAP',
                        'error',
                        f'Sum of cell volumes exceeds domain volume by {overlap:g}',
                    )
                )

    # --- Missing ids (optional) ---
    missing_ids: list[int] = []
    if expected_ids_array is not None:
        classification = classify_expected_ids(
            expected_ids_array.tolist(),
            present_ids,
            mode=mode,
        )
        missing_ids = list(classification.missing_ids)
        empty_ids.extend(classification.hidden_ids)
        if classification.issue_code is not None:
            subject = (
                'hidden power'
                if classification.issue_code == 'HIDDEN_IDS'
                else 'expected'
            )
            issues.append(
                TessellationIssue(
                    classification.issue_code,
                    classification.severity,
                    f'{len(missing_ids)} {subject} ids are absent from output',
                    examples=tuple(missing_ids[:10]),
                )
            )

    if mark_faces:
        reset_owned_annotations(
            (
                face
                for cell in cells
                for face in (cell.get('faces') or [])
            ),
            fields=('orphan', 'reciprocal_missing', 'reciprocal_mismatch'),
        )

    # --- Reciprocity + plane mismatch (Periodic only) ---
    face_shift_available = False
    reciprocity_checked = False
    n_faces_total = 0
    n_orphan = 0
    n_mismatch = 0

    if _is_periodic_domain(domain) and check_reciprocity:
        all_faces = [
            face
            for cell in cells
            for face in (cell.get('faces') or [])
        ]
        relevant_faces = [
            face
            for face in all_faces
            if int(face.get('adjacent_cell', -999999)) >= 0
        ]
        if relevant_faces:
            face_shift_available = all(
                'adjacent_shift' in face for face in relevant_faces
            )
        else:
            # Preserve the historical no-relevant-boundary behavior.
            face_shift_available = any(
                'adjacent_shift' in face for face in all_faces
            )

        if not face_shift_available:
            issues.append(
                TessellationIssue(
                    'NO_FACE_SHIFTS',
                    reciprocity_issue_severity(
                        required=reciprocity_required,
                        missing_shifts=True,
                    ),
                    'Face shifts are not available; '
                    'set return_face_shifts=True to enable reciprocity diagnostics',
                )
            )
        else:
            reciprocity_checked = True

            def _polygon_area(v: np.ndarray) -> float:
                """Return polygon area using the vector-area (Newell) formula."""
                if v.shape[0] < 3:
                    return 0.0
                vv = np.asarray(v, dtype=np.float64)
                area_vec = 0.5 * np.sum(
                    np.cross(vv, np.roll(vv, -1, axis=0)), axis=0
                )
                return float(np.linalg.norm(area_vec))

            # Precompute lattice translation vectors (Cartesian)
            avec, bvec, cvec = _lattice_vectors_cart(domain)

            # Map id -> cell dict for quick lookup
            cell_by_id: dict[int, dict[str, Any]] = {}
            for c in cells:
                cid = int(c.get('id', -1))
                if cid >= 0:
                    cell_by_id[cid] = c

            # Defaults for plane mismatch tolerances
            L = _characteristic_length(domain)
            if (plane_offset_tol is None or plane_angle_tol is None) and (
                float(L) < 1e-3 or float(L) > 1e9
            ):
                msg = (
                    'analyze_tessellation is using default periodic plane-mismatch '
                    'tolerances derived from the domain length scale '
                    f'(L≈{float(L):.3g}). '
                    'For very small/large units this may be too strict/too loose. '
                    'Consider rescaling inputs or passing plane_offset_tol=... '
                    'and/or plane_angle_tol=... explicitly.'
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            off_tol = (
                (1e-6 * L) if plane_offset_tol is None else float(plane_offset_tol)
            )
            ang_tol = (1e-6) if plane_angle_tol is None else float(plane_angle_tol)

            # Faces whose polygon area is extremely small can be reported
            # non-reciprocally by Voro++ on some platforms, especially in power
            # mode. Such faces are numerically degenerate and are ignored for
            # reciprocity/plane-mismatch diagnostics.
            eps_f = float(np.finfo(float).eps)
            area_tol = float(max(100.0 * off_tol * off_tol, 1024.0 * eps_f * L * L))
            size_tol = float(max(1000.0 * off_tol, 128.0 * eps_f * L))

            # Build a directed-face map: (i, j, s) -> (cell_id, face_index)
            def _skey(s: Any) -> tuple[int, int, int]:
                return int(s[0]), int(s[1]), int(s[2])

            face_map: dict[tuple[int, int, tuple[int, int, int]], tuple[int, int]] = {}
            for c in cells:
                i = int(c.get('id', -1))
                if i < 0:
                    continue
                verts = np.asarray(c.get('vertices', []), dtype=np.float64)
                faces = c.get('faces') or []
                for fi, f in enumerate(faces):
                    j = int(f.get('adjacent_cell', -999999))
                    if j < 0:
                        continue
                    s = _skey(f['adjacent_shift'])
                    n_faces_total += 1

                    idx = np.asarray(f.get('vertices', []), dtype=np.int64)
                    if idx.size < 3 or verts.size == 0:
                        continue
                    vv = verts[idx]
                    area = _polygon_area(vv)
                    size = float(np.linalg.norm(np.ptp(vv, axis=0)))
                    if float(area) < area_tol or float(size) < size_tol:
                        continue

                    key = (i, j, s)
                    # Duplicates indicate a shift-solver issue or a degenerate case.
                    if key in face_map:
                        issues.append(
                            TessellationIssue(
                                'DUPLICATE_DIRECTED_FACE',
                                'error',
                                f'Duplicate directed face key encountered: {key}',
                            )
                        )
                    else:
                        face_map[key] = (i, fi)

            def _face_plane(
                cell_id: int,
                face_index: int,
                *,
                translate: np.ndarray | None = None,
            ) -> tuple[np.ndarray, float] | None:
                c = cell_by_id.get(cell_id)
                if c is None:
                    return None
                verts = np.asarray(c.get('vertices', []), dtype=np.float64)
                faces = c.get('faces') or []
                if face_index < 0 or face_index >= len(faces):
                    return None
                idx = np.asarray(faces[face_index].get('vertices', []), dtype=np.int64)
                if idx.size < 3 or verts.size == 0:
                    return None
                vv = verts[idx]
                if translate is not None:
                    vv = vv + translate.reshape(1, 3)
                return _plane_from_vertices(vv)

            # Check reciprocity and (optionally) plane mismatch
            checked_pairs: set[tuple[int, int, tuple[int, int, int]]] = set()
            examples_missing: list[tuple[int, int, tuple[int, int, int]]] = []
            examples_mismatch: list[tuple[int, int, tuple[int, int, int]]] = []

            for (i, j, s), loc in list(face_map.items()):
                if (i, j, s) in checked_pairs:
                    continue

                recip = (j, i, (-s[0], -s[1], -s[2]))
                # Avoid double-counting by marking both directions as checked.
                checked_pairs.add((i, j, s))
                checked_pairs.add(recip)
                if recip not in face_map:
                    n_orphan += 1
                    if len(examples_missing) < 10:
                        examples_missing.append((i, j, s))
                    if mark_faces:
                        ci, fi = loc
                        try:
                            cell_by_id[ci]['faces'][fi]['orphan'] = True
                            cell_by_id[ci]['faces'][fi]['reciprocal_missing'] = True
                        except Exception:
                            pass
                    continue

                # Reciprocal exists
                if not check_plane_mismatch:
                    continue

                (ci, fi) = loc
                (cj, fj) = face_map[recip]

                # Translation that maps the reciprocal face into the same periodic image
                # as the i->j(s) face.
                T = s[0] * avec + s[1] * bvec + s[2] * cvec

                p1 = _face_plane(ci, fi, translate=None)
                p2 = _face_plane(cj, fj, translate=T)
                if p1 is None or p2 is None:
                    continue
                n1, d1 = p1
                n2, d2 = p2
                # Align normals
                dot = float(np.dot(n1, n2))
                if dot < 0.0:
                    n2 = -n2
                    d2 = -d2
                    dot = -dot
                dot = max(-1.0, min(1.0, dot))
                ang = float(np.arccos(dot))
                off = float(abs(d1 - d2))

                if ang > ang_tol or off > off_tol:
                    n_mismatch += 1
                    if len(examples_mismatch) < 10:
                        examples_mismatch.append((i, j, s))
                    if mark_faces:
                        try:
                            cell_by_id[ci]['faces'][fi]['reciprocal_mismatch'] = True
                            cell_by_id[cj]['faces'][fj]['reciprocal_mismatch'] = True
                        except Exception:
                            pass

            if n_orphan:
                issues.append(
                    TessellationIssue(
                        'MISSING_RECIPROCAL',
                        reciprocity_issue_severity(
                            required=reciprocity_required,
                        ),
                        f'{n_orphan} faces are missing a reciprocal',
                        examples=tuple(examples_missing),
                    )
                )
            if n_mismatch:
                issues.append(
                    TessellationIssue(
                        'RECIPROCAL_MISMATCH',
                        reciprocity_issue_severity(
                            required=reciprocity_required,
                        ),
                        f'{n_mismatch} reciprocal face pairs disagree geometrically',
                        examples=tuple(examples_mismatch),
                    )
                )

    reciprocity_requested = bool(_is_periodic_domain(domain) and check_reciprocity)
    ok_recip = bool(
        not reciprocity_requested
        or (
            reciprocity_checked
            and n_orphan == 0
            and n_mismatch == 0
        )
    )

    ok = diagnostics_ok(issues)
    if not ok and mode is not None:
        issues.append(
            TessellationIssue('MODE', 'info', f'Diagnostics produced for mode={mode!r}')
        )

    diag = TessellationDiagnostics(
        domain_volume=float(dom_vol),
        sum_cell_volume=float(sum_vol),
        volume_ratio=float(sum_vol / dom_vol) if dom_vol > 0 else 0.0,
        volume_gap=float(gap),
        volume_overlap=float(overlap),
        n_sites_expected=int(
            expected_ids_array.size
            if expected_ids_array is not None
            else len(set(present_ids))
        ),
        n_cells_returned=int(len(cells)),
        missing_ids=tuple(int(x) for x in missing_ids),
        empty_ids=tuple(int(x) for x in sorted(set(empty_ids))),
        face_shift_available=bool(face_shift_available),
        reciprocity_checked=bool(reciprocity_checked),
        n_faces_total=int(n_faces_total),
        n_faces_orphan=int(n_orphan),
        n_faces_mismatched=int(n_mismatch),
        issues=tuple(issues),
        ok_volume=bool(ok_volume),
        ok_reciprocity=bool(ok_recip),
        ok=bool(ok),
    )

    return diag


def validate_tessellation(
    cells: Sequence[dict[str, Any]],
    domain: Box | OrthorhombicCell | PeriodicCell,
    *,
    expected_ids: Sequence[int] | None = None,
    mode: str | None = None,
    level: Literal['basic', 'strict'] = 'basic',
    require_reciprocity: bool | None = None,
    volume_tol_rel: float = 1e-8,
    volume_tol_abs: float = 1e-12,
    plane_offset_tol: float | None = None,
    plane_angle_tol: float | None = None,
    mark_faces: bool | None = None,
) -> TessellationDiagnostics:
    """Validate tessellation sanity, optionally raising in strict mode.

    This is a convenience wrapper around :func:`analyze_tessellation`.

    Args:
        cells: Output of :func:`pyvoro2.compute`.
        domain: Domain used for computation.
        expected_ids: Optional list of expected ids.
        mode: Optional mode selecting standard, power, or undeclared expected-ID
            semantics.
        level: 'basic' returns diagnostics; 'strict' raises
            :class:`TessellationError` when validation fails.
        require_reciprocity: If True, periodic reciprocity findings are errors.
            If False, they are inspected as warning/info findings. If None,
            defaults to True for periodic domains and False otherwise.
        volume_tol_rel: Relative tolerance for volume closure.
        volume_tol_abs: Absolute tolerance for volume closure.
        plane_offset_tol: Absolute tolerance for reciprocal plane offset mismatch.
        plane_angle_tol: Tolerance for reciprocal plane normal mismatch (radians).
        mark_faces: Whether to annotate faces with local flags.

    Returns:
        TessellationDiagnostics
    """

    level = require_string_choice(
        level,
        name='level',
        choices=('basic', 'strict'),
    )
    if mode is not None:
        mode = require_string_choice(
            mode,
            name='mode',
            choices=('standard', 'power'),
        )

    require_reciprocity = require_optional_bool(
        require_reciprocity,
        name='require_reciprocity',
    )
    mark_faces = require_optional_bool(mark_faces, name='mark_faces')

    periodic = _is_periodic_domain(domain)
    if require_reciprocity is None:
        require_reciprocity = bool(periodic)
    if mark_faces is None:
        mark_faces = bool(periodic)

    diag = _analyze_tessellation(
        cells,
        domain,
        expected_ids=expected_ids,
        mode=mode,
        volume_tol_rel=volume_tol_rel,
        volume_tol_abs=volume_tol_abs,
        check_reciprocity=bool(periodic),
        reciprocity_required=bool(require_reciprocity),
        check_plane_mismatch=bool(periodic),
        plane_offset_tol=plane_offset_tol,
        plane_angle_tol=plane_angle_tol,
        mark_faces=mark_faces,
    )

    if level == 'strict' and not diag.ok:
        error = next(
            (issue for issue in diag.issues if issue.severity == 'error'),
            None,
        )
        if error is None:  # pragma: no cover - guarded by severity-complete policy
            message = 'Tessellation validation failed'
        else:
            message = (
                f'Tessellation validation failed ({error.code}): '
                f'{error.message}'
            )
        raise TessellationError(message, diag)

    return diag
