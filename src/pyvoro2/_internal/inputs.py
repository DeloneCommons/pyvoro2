"""Internal coercion helpers for public Python entry points.

These helpers intentionally keep error messages stable so the public API can be
refactored without changing its validation surface.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def coerce_point_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    dim: int,
) -> np.ndarray:
    """Return a finite ``(n, dim)`` float64 array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f'{name} must have shape (n, {dim})')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    return arr


def coerce_id_array(
    ids: Sequence[int] | np.ndarray | None,
    *,
    n: int,
) -> np.ndarray | None:
    """Return validated non-negative unique IDs or ``None``."""

    if ids is None:
        return None
    if len(ids) != n:
        raise ValueError('ids must have length n')
    ids_arr = np.asarray(ids, dtype=np.int64)
    if ids_arr.shape != (n,):
        raise ValueError('ids must be a 1D sequence of length n')
    if np.any(ids_arr < 0):
        raise ValueError('ids must be non-negative')
    if np.unique(ids_arr).size != n:
        raise ValueError('ids must be unique')
    return ids_arr


def coerce_nonnegative_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
) -> np.ndarray:
    """Return a finite non-negative float64 vector with shape ``(n,)``."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f'{name} must have shape (n,)')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    if np.any(arr < 0):
        raise ValueError(f'{name} must be non-negative')
    return arr


def coerce_nonnegative_scalar_or_vector(
    values: float | Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
    length_name: str,
) -> np.ndarray:
    """Return a finite non-negative float64 vector.

    Scalars are broadcast to shape ``(n,)``. Vector inputs must already have
    shape ``(n,)``.
    """

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((n,), float(arr), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(
            f'{name} must be a scalar or have shape ({length_name},)'
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    if np.any(arr < 0):
        raise ValueError(f'{name} must be non-negative')
    return arr


def validate_duplicate_check_mode(mode: str) -> None:
    """Validate the public duplicate-check mode string."""

    if mode not in ('off', 'warn', 'raise'):
        raise ValueError("duplicate_check must be one of: 'off', 'warn', 'raise'")
