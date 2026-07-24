"""Shared transforms between mathematical power weights and backend radii."""

from __future__ import annotations

import numpy as np


def radii_to_weights(radii: np.ndarray) -> np.ndarray:
    """Convert radii to finite power weights (``w = r^2``).

    Raises ``ValueError`` when the input or its squared result is non-finite.
    """

    r = np.asarray(radii, dtype=float)
    if r.ndim != 1:
        raise ValueError('radii must be 1D')
    if not np.all(np.isfinite(r)):
        raise ValueError('radii must contain only finite values')
    if np.any(r < 0):
        raise ValueError('radii must be non-negative')
    with np.errstate(over='ignore', invalid='ignore'):
        weights = r * r
    if not np.all(np.isfinite(weights)):
        raise ValueError('radii produced non-finite weights')
    return weights


def weights_to_radii(
    weights: np.ndarray,
    *,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert power weights to finite radii using one global shift.

    Raises ``ValueError`` when an input, intermediate value, or result is
    non-finite.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError('weights must be 1D')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must contain only finite values')

    r_min = float(r_min)
    if not np.isfinite(r_min):
        raise ValueError('r_min must be finite')
    if r_min < 0:
        raise ValueError('r_min must be >= 0')

    if weight_shift is not None:
        if r_min != 0.0:
            raise ValueError('specify at most one of r_min and weight_shift')
        C = float(weight_shift)
        if not np.isfinite(C):
            raise ValueError('weight_shift must be finite')
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            r_min_squared = r_min * r_min
        if not np.isfinite(r_min_squared):
            raise ValueError('r_min squared must be finite')
        w_min = float(np.min(w)) if w.size else 0.0
        with np.errstate(over='ignore', invalid='ignore'):
            C = r_min_squared - w_min
        if not np.isfinite(C):
            raise ValueError('derived weight shift must be finite')

    with np.errstate(over='ignore', invalid='ignore'):
        w_shifted = w + C
    if not np.all(np.isfinite(w_shifted)):
        raise ValueError('weight shift produced non-finite values')
    if np.any(w_shifted < -1e-14):
        raise ValueError('weight shift produced negative values (numerical issue)')
    w_shifted = np.maximum(w_shifted, 0.0)
    with np.errstate(over='ignore', invalid='ignore'):
        radii = np.sqrt(w_shifted)
    if not np.all(np.isfinite(radii)) or not np.isfinite(C):
        raise ValueError('weight-to-radius transform produced non-finite results')
    return radii, float(C)
