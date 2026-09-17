"""Sign-normalized nonpivoting orthogonal frame for periodic native calls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exact_lattice import exact_basis_3d


@dataclass(frozen=True, slots=True)
class BackendFrame:
    """Detached backend frame with ``A @ q`` lower triangular."""

    q: np.ndarray
    lower: np.ndarray
    params: tuple[float, float, float, float, float, float]
    parity: int


def _scaled_tolerance(scale: float, *, factor: float = 512.0) -> float:
    tiny = float.fromhex('0x0.0000000000001p-1022')
    return factor * max(float(np.finfo(np.float64).eps) * scale, tiny)


def prepare_backend_frame(vectors: np.ndarray) -> BackendFrame:
    """Prepare and validate the one canonical backend frame convention."""

    matrix = np.asarray(vectors, dtype=np.float64)
    exact_sign = exact_basis_3d(matrix).determinant_sign
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            q_raw, upper_raw = np.linalg.qr(matrix.T, mode='reduced')
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError('periodic backend frame preparation failed') from exc
    if not np.all(np.isfinite(q_raw)) or not np.all(np.isfinite(upper_raw)):
        raise ValueError('periodic backend frame preparation produced non-finite data')

    diagonal = np.diag(upper_raw)
    if np.any(diagonal == 0.0):
        raise ValueError('periodic backend frame has a zero numerical diagonal')
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q = q_raw * signs[None, :]
    upper = signs[:, None] * upper_raw
    lower = upper.T
    if np.any(np.diag(lower) <= 0.0):
        raise ValueError('periodic backend frame diagonal is not positive')

    orthogonality = q.T @ q
    ortho_error = float(np.max(np.abs(orthogonality - np.eye(3))))
    if not np.isfinite(ortho_error) or ortho_error > _scaled_tolerance(1.0):
        raise ValueError('periodic backend frame failed orthogonality validation')

    with np.errstate(over='ignore', invalid='ignore'):
        reconstructed = matrix @ q
    scale = max(float(np.max(np.abs(matrix))),
                float(np.max(np.abs(lower))))
    residual = float(np.max(np.abs(reconstructed - lower)))
    if not np.isfinite(residual) or residual > _scaled_tolerance(scale):
        raise ValueError('periodic backend frame failed reconstruction validation')

    determinant_q = float(np.linalg.det(q))
    numerical_parity = 1 if determinant_q > 0.0 else -1
    if (
        not np.isfinite(determinant_q)
        or abs(abs(determinant_q) - 1.0) > _scaled_tolerance(1.0)
        or numerical_parity != exact_sign
    ):
        raise ValueError('periodic backend frame failed parity validation')

    params = (
        float(lower[0, 0]),
        float(lower[1, 0]),
        float(lower[1, 1]),
        float(lower[2, 0]),
        float(lower[2, 1]),
        float(lower[2, 2]),
    )
    q_result = np.array(q, dtype=np.float64, copy=True, order='C')
    lower_result = np.array(lower, dtype=np.float64, copy=True, order='C')
    q_result.setflags(write=False)
    lower_result.setflags(write=False)
    return BackendFrame(q=q_result, lower=lower_result, params=params,
                        parity=exact_sign)
