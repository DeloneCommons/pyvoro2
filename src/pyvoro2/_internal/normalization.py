"""Strict input preparation for spatial and planar normalization."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .validation import (
    INT64_MAX,
    INT64_MIN,
    _as_original_array,
    _finite_float64_array,
    require_index,
    require_index_array,
    require_nonnegative_index,
)


def coerce_normalization_vertices(
    values: Any,
    *,
    name: str,
    dim: int,
) -> np.ndarray:
    """Return finite real vertex coordinates with shape ``(m, dim)``."""

    raw = _as_original_array(values, name=name)
    if raw.shape == (0,):
        raw = raw.reshape((0, dim))
    if raw.ndim != 2 or raw.shape[1] != dim:
        raise ValueError(f'{name} must have shape (m, {dim})')
    return _finite_float64_array(raw, name=name)


def quantize_coordinates(
    coordinates: np.ndarray,
    *,
    tol: float,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    """Return warning-free signed-int64 keys for finite coordinates.

    The division and rounding remain in float64 to preserve existing valid
    normalization values. Conversion happens only after every rounded quotient
    is known to be finite and within the signed-int64 destination range.
    """

    with np.errstate(all='ignore'):
        quotient = coordinates / tol
        rounded = np.rint(quotient)
    if not np.all(np.isfinite(quotient)) or not np.all(np.isfinite(rounded)):
        raise ValueError(
            f'{name} divided by tol must produce finite quantization values '
            'representable as signed int64'
        )

    keys: list[tuple[int, ...]] = []
    for row in rounded:
        key = tuple(int(value) for value in row)
        if any(value < INT64_MIN or value > INT64_MAX for value in key):
            raise ValueError(
                f'{name} divided by tol must produce finite quantization '
                'values representable as signed int64'
            )
        keys.append(key)
    return tuple(keys)


def require_cell_id(value: object, *, name: str) -> int:
    """Return a non-negative external cell ID in the public result range."""

    return require_nonnegative_index(value, name=name, maximum=INT64_MAX)


def require_adjacent_cell_id(value: object, *, name: str) -> int:
    """Return a signed-int64 adjacent cell or wall ID."""

    return require_index(
        value,
        name=name,
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )


def require_local_vertex_indices(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    n_vertices: int,
    length: int | None = None,
) -> tuple[int, ...]:
    """Return exact local vertex indices proven safe for indexing."""

    raw = _as_original_array(values, name=name)
    if raw.ndim != 1 or (length is not None and raw.shape != (length,)):
        if length is None:
            raise ValueError(
                f'{name} must be a 1D sequence of exact integer indices'
            )
        raise ValueError(
            f'{name} must have shape ({length},) and contain exact integer '
            'indices'
        )
    result = require_index_array(
        raw,
        name=name,
        shape=raw.shape,
        minimum=0,
        maximum=INT64_MAX,
    )
    indices = tuple(int(value) for value in result)
    if any(value >= n_vertices for value in indices):
        raise ValueError(
            f'{name} must contain local vertex indices in [0, {n_vertices})'
        )
    return indices


def require_global_vertex_ids(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    n_vertices: int,
    n_global_vertices: int,
) -> tuple[int, ...]:
    """Return exact global IDs aligned with a cell's local vertices."""

    raw = _as_original_array(values, name=name)
    if raw.shape != (n_vertices,):
        raise ValueError(f'{name} must have shape ({n_vertices},)')
    result = require_index_array(
        raw,
        name=name,
        shape=raw.shape,
        minimum=0,
        maximum=INT64_MAX,
    )
    ids = tuple(int(value) for value in result)
    if any(value >= n_global_vertices for value in ids):
        raise ValueError(
            f'{name} must contain global vertex IDs in '
            f'[0, {n_global_vertices})'
        )
    return ids


def require_shift(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    dim: int,
) -> tuple[int, ...]:
    """Return one exact signed-int64 lattice shift."""

    result = require_index_array(
        values,
        name=name,
        shape=(dim,),
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )
    return tuple(int(value) for value in result)


def require_shift_rows(
    values: Sequence[Sequence[object]] | np.ndarray,
    *,
    name: str,
    rows: int,
    dim: int,
) -> tuple[tuple[int, ...], ...]:
    """Return exact signed-int64 lattice shifts aligned with local vertices."""

    raw = _as_original_array(values, name=name)
    if rows == 0 and raw.shape == (0,):
        raw = raw.reshape((0, dim))
    result = require_index_array(
        raw,
        name=name,
        shape=(rows, dim),
        minimum=INT64_MIN,
        maximum=INT64_MAX,
    )
    return tuple(tuple(int(value) for value in row) for row in result)


def checked_shift_difference(
    value: Sequence[int],
    anchor: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    """Subtract validated shifts and require a signed-int64 result."""

    result = tuple(int(left) - int(right) for left, right in zip(value, anchor))
    if any(component < INT64_MIN or component > INT64_MAX for component in result):
        raise ValueError(
            f'{name} difference must be representable as signed int64'
        )
    return result


def validate_pairwise_shift_differences(
    shifts: Sequence[Sequence[int]],
    *,
    name: str,
) -> None:
    """Require every relative shift used by canonicalization to fit int64."""

    for anchor in shifts:
        for value in shifts:
            checked_shift_difference(value, anchor, name=name)


def checked_add_shift_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Add signed-int64 shift arrays without allowing integer wraparound."""

    if left.shape != right.shape:
        raise ValueError(f'{name} shift arrays must have matching shapes')
    values = np.asarray(left, dtype=object) + np.asarray(right, dtype=object)
    if any(
        value < INT64_MIN or value > INT64_MAX
        for value in values.flat
    ):
        raise ValueError(f'{name} must be representable as signed int64')
    return np.asarray(values, dtype=np.int64)


def quantized_key_matches(
    first: np.ndarray,
    second: np.ndarray,
    *,
    tol: float,
) -> bool:
    """Return whether equal quantization keys remain coordinate-consistent."""

    with np.errstate(all='ignore'):
        delta = np.abs(first - second)
    return bool(np.all(np.isfinite(delta)) and np.all(delta <= tol))
