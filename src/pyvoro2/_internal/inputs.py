"""Internal coercion helpers for public Python entry points.

These helpers intentionally keep error messages stable so the public API can be
refactored without changing its validation surface.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .validation import (
    CPP_INT_MAX,
    _as_original_array,
    _finite_float64_array,
    require_positive_finite_real,
    require_positive_index,
)


def coerce_point_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    dim: int,
) -> np.ndarray:
    """Return a finite ``(n, dim)`` float64 array."""

    arr = _as_original_array(values, name=name)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f'{name} must have shape (n, {dim})')
    return _finite_float64_array(arr, name=name)


def coerce_finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    n: int,
) -> np.ndarray:
    """Return a finite real float64 vector with shape ``(n,)``."""

    arr = _as_original_array(values, name=name)
    if arr.shape != (n,):
        raise ValueError(f'{name} must have shape (n,)')
    return _finite_float64_array(arr, name=name)


def coerce_finite_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a finite real float64 matrix with an exact shape."""

    arr = _as_original_array(values, name=name)
    if arr.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return _finite_float64_array(arr, name=name)


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

    arr = coerce_finite_vector(values, name=name, n=n)
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

    arr = _as_original_array(values, name=name)
    if arr.ndim == 0:
        scalar = _finite_float64_array(arr, name=name)
        arr = np.full((n,), float(scalar), dtype=np.float64)
    elif arr.shape == (n,):
        arr = _finite_float64_array(arr, name=name)
    else:
        raise ValueError(
            f'{name} must be a scalar or have shape ({length_name},)'
        )
    if np.any(arr < 0):
        raise ValueError(f'{name} must be non-negative')
    return arr


def coerce_native_block_parameters(
    *,
    blocks: Sequence[object] | None,
    block_size: object | None,
    dim: int,
) -> tuple[tuple[int, ...] | None, float | None]:
    """Validate explicit native block counts and optional block size."""

    block_size_value = (
        None
        if block_size is None
        else require_positive_finite_real(block_size, name='block_size')
    )
    if blocks is None:
        return None, block_size_value

    try:
        items = tuple(blocks)
    except TypeError:
        raise ValueError(
            f'blocks must be a length-{dim} sequence of positive exact integers'
        ) from None
    if len(items) != dim:
        raise ValueError(f'blocks must have length {dim}')
    counts = tuple(
        require_positive_index(
            item,
            name=f'blocks[{axis}]',
            maximum=CPP_INT_MAX,
        )
        for axis, item in enumerate(items)
    )
    return counts, block_size_value


def require_internal_id_range(n: int, *, name: str = 'points') -> None:
    """Require generated internal IDs ``0..n-1`` to fit C++ ``int``."""

    if n < 0 or n - 1 > CPP_INT_MAX:
        raise ValueError(
            f'{name} length is too large; internal IDs must fit C++ int range'
        )


def require_query_index_range(m: int, *, name: str = 'queries') -> None:
    """Require generated ghost query indices ``0..m-1`` to fit C++ ``int``."""

    if m < 0 or m - 1 > CPP_INT_MAX:
        raise ValueError(
            f'{name} length is too large; generated query indices must fit '
            'the C++ int destination range'
        )


def require_planar_ghost_site_id_range(
    n: int,
    *,
    name: str = 'points',
) -> None:
    """Keep planar generated site IDs below the reserved native ghost ID."""

    if n < 0 or n > CPP_INT_MAX:
        raise ValueError(
            f'{name}/site length is too large for planar ghost construction; '
            'generated site IDs must fit the C++ int destination range with '
            'CPP_INT_MAX reserved as the native ghost ID'
        )


def owned_readonly_array(
    values: Sequence[object] | np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Return a C-contiguous owned array whose write flag is disabled."""

    result = np.array(values, dtype=dtype, copy=True, order='C')
    result.setflags(write=False)
    return result


def validate_forward_mode(mode: object) -> None:
    """Validate the two native forward construction modes."""

    if not isinstance(mode, str) or mode not in ('standard', 'power'):
        raise ValueError(f'unknown mode: {mode}')


def validate_duplicate_check_mode(mode: str) -> None:
    """Validate the public duplicate-check mode string."""

    if mode not in ('off', 'warn', 'raise'):
        raise ValueError("duplicate_check must be one of: 'off', 'warn', 'raise'")
