"""Public scalar transforms for power-fit weights and radii."""

from __future__ import annotations

import numpy as np


def radii_to_weights(radii: np.ndarray) -> np.ndarray:
    """Convert radii to power weights (``w = r^2``)."""

    r = np.asarray(radii, dtype=float)
    if r.ndim != 1:
        raise ValueError('radii must be 1D')
    if not np.all(np.isfinite(r)):
        raise ValueError('radii must contain only finite values')
    if np.any(r < 0):
        raise ValueError('radii must be non-negative')
    return r * r


def weights_to_radii(
    weights: np.ndarray,
    *,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert power weights to radii using an explicit global shift."""

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError('weights must be 1D')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must contain only finite values')

    if weight_shift is not None:
        if r_min != 0.0:
            raise ValueError('specify at most one of r_min and weight_shift')
        C = float(weight_shift)
        if not np.isfinite(C):
            raise ValueError('weight_shift must be finite')
    else:
        r_min = float(r_min)
        if r_min < 0:
            raise ValueError('r_min must be >= 0')
        w_min = float(np.min(w)) if w.size else 0.0
        C = (r_min * r_min) - w_min

    w_shifted = w + C
    if np.any(w_shifted < -1e-14):
        raise ValueError('weight shift produced negative values (numerical issue)')
    w_shifted = np.maximum(w_shifted, 0.0)
    return np.sqrt(w_shifted), float(C)
