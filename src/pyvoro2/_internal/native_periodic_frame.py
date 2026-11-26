"""Qualified numerical frame for the 3D periodic native adapter.

Source rows remain authoritative. Equal integer image labels in the source
and triangular frames do not imply exact binary64 translation equivalence.
This leaf module owns numerical admission; it does not change the A1 lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from . import exact_user_basis
from .inputs import coerce_finite_matrix, owned_readonly_array


_FRAME_TOL = 60 * 2**-53


@dataclass(frozen=True, slots=True)
class NativePeriodicFrame:
    """Owned orthogonal Cartesian-to-native transform (possibly improper)."""

    rotation_to_internal: np.ndarray  # C = Q.T, row-stacked frame
    params: tuple[float, float, float, float, float, float]
    parity: int
    length_scale: float
    volume: float


def _require_native_periodic_basis(vectors) -> tuple[int, float]:
    """Preserve pre-A1 numerical safeguards, allowing either exact sign."""
    canonical = tuple(tuple(float(value) for value in row) for row in vectors)
    parity = exact_user_basis.determinant_sign(canonical)
    if parity == 0:
        raise ValueError('PeriodicCell native adapter needs a nonsingular basis')
    vec = np.asarray(canonical, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        norms = np.linalg.norm(vec, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError(
                'PeriodicCell native adapter needs positive finite lengths')
        det = float(np.linalg.det(vec))
        if not np.isfinite(det) or det == 0.0 or np.sign(det) != parity:
            raise ValueError('PeriodicCell native adapter needs a finite nonzero '
                             'floating determinant with the exact source sign')
        try:
            s = np.linalg.svd(vec, compute_uv=False)
        except np.linalg.LinAlgError:
            raise ValueError('PeriodicCell native conditioning check failed') from None
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError('PeriodicCell native conditioning check is ill-defined')
        cond = float(np.max(s) / np.min(s))
        norm_product = float(norms[0] * norms[1] * norms[2])
        if not np.isfinite(norm_product) or norm_product <= 0.0:
            raise ValueError('PeriodicCell native adapter needs a positive finite '
                             'vector-length product for relative volume')
        relvol = float(abs(det) / norm_product)
    if not np.isfinite(relvol) or relvol < 1e-12 or cond > 1e15:
        raise ValueError('PeriodicCell native adapter: nearly degenerate '
                         f'(poorly conditioned). relvol={relvol:.3g}, cond={cond:.3g}')
    if cond > 1e10:
        warnings.warn(
            'PeriodicCell lattice vectors are very ill-conditioned '
            f'(cond≈{cond:.3g}) for the native adapter. Numerical accuracy and '
            'periodic image bookkeeping may be unstable; consider rescaling '
            'or using a better-conditioned basis.', RuntimeWarning, stacklevel=3,
        )
    return parity, float(np.max(norms))


def prepare_native_periodic_frame(vectors: object) -> NativePeriodicFrame:
    """Prepare and qualify nonpivoted Householder QR of the source transpose.

    Failures are native-adapter operation errors, not source singularity
    claims. Orthogonality and row-relative backward error are bounded by 60u,
    u=2**-53; this is not a relative-error certificate for short translations.
    """
    a = coerce_finite_matrix(vectors, name='PeriodicCell.vectors', shape=(3, 3))
    parity, length_scale = _require_native_periodic_basis(a)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        try:
            q0, r0 = np.linalg.qr(a.T)
        except np.linalg.LinAlgError:
            raise ValueError('PeriodicCell native frame QR failed') from None
        if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(r0))):
            raise ValueError('PeriodicCell native frame QR factors must be finite')
        diagonal = np.diag(r0)
        if np.any(diagonal == 0):
            raise ValueError('PeriodicCell native frame QR has a zero diagonal')
        signs = np.sign(diagonal)
        q = q0 * signs[None, :]
        r = signs[:, None] * r0
        if np.any(np.diag(r) <= 0) or np.any(np.tril(r, -1) != 0):
            raise ValueError(
                'PeriodicCell native frame needs positive triangular factors')
        params = tuple(float(r[i, j]) for i, j in
                       ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)))
        volume = params[0] * params[2] * params[5]
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError(
                'PeriodicCell native frame volume must be positive and finite')
        c = q.T
        frame_det = float(np.linalg.det(c))
        if not np.isfinite(frame_det) or np.sign(frame_det) != parity:
            raise ValueError('PeriodicCell native frame parity disagrees with source')
        orthogonality = float(np.linalg.norm(c @ c.T - np.eye(3), ord=np.inf))
        if not np.isfinite(orthogonality) or orthogonality > _FRAME_TOL:
            raise ValueError('PeriodicCell native frame orthogonality check failed: '
                             f'{orthogonality:.3g} > {_FRAME_TOL:.3g}')
        # R is triangular, so R.T is exactly the lattice emitted by params.
        residual = float(np.max(np.sum(abs(r.T - a @ q), axis=1)
                                / np.sum(abs(a), axis=1)))
        if not np.isfinite(residual) or residual > _FRAME_TOL:
            raise ValueError('PeriodicCell native frame backward consistency failed: '
                             f'{residual:.3g} > {_FRAME_TOL:.3g}')
    return NativePeriodicFrame(
        rotation_to_internal=owned_readonly_array(c, dtype=np.float64),
        params=params,
        parity=parity,
        length_scale=length_scale,
        volume=volume,
    )
