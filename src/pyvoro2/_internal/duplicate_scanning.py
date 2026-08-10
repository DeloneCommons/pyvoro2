"""Complete private candidate scanning and duplicate-distance classification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Iterator

import numpy as np

from .periodic_images import (
    _exact_triclinic_bucket_layout,
    exact_distance_less_than,
    exact_distance_squared_less_equal,
)


_MAX_CANDIDATE_COMPARISONS = 10_000_000
_CLASSIFICATION_BATCH = 1024


class DuplicateScanResourceError(ValueError):
    """Raised when a complete scan would exceed its private work budget."""


@dataclass(frozen=True, slots=True)
class PairScan:
    """Private close-pair scan result."""

    pairs: tuple[tuple[int, int, float], ...]
    truncated: bool
    candidate_count: int
    minimum_image_used: bool


@dataclass(frozen=True, slots=True)
class _BucketLayout:
    keys: tuple[tuple[int, ...], ...]
    bins: tuple[int | None, ...]
    periodic: tuple[bool, ...]

    def neighboring_keys(self, key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        neighbors: set[tuple[int, ...]] = set()
        for offset in product((-1, 0, 1), repeat=len(key)):
            values: list[int] = []
            valid = True
            for axis, delta in enumerate(offset):
                value = key[axis] + delta
                count = self.bins[axis]
                if self.periodic[axis]:
                    assert count is not None
                    value %= count
                elif count is not None and not (0 <= value < count):
                    valid = False
                    break
                values.append(value)
            if valid:
                neighbors.add(tuple(values))
        return tuple(sorted(neighbors))


def _exact_span_ratio(lo: float, hi: float) -> tuple[int, int]:
    lo_num, lo_den = float(lo).as_integer_ratio()
    hi_num, hi_den = float(hi).as_integer_ratio()
    span_num = hi_num * lo_den - lo_num * hi_den
    span_den = hi_den * lo_den
    if span_num <= 0:
        raise DuplicateScanResourceError(
            'duplicate candidate layout requires strictly ordered bounds'
        )
    return span_num, span_den


def _rectangular_bin_count(lo: float, hi: float, radius: float) -> int:
    """Return the exact bin count for a binary64 half-open interval."""

    if not (math.isfinite(radius) and radius > 0.0):
        raise DuplicateScanResourceError(
            'duplicate candidate layout requires a positive finite radius'
        )
    span_num, span_den = _exact_span_ratio(lo, hi)
    radius_num, radius_den = float(radius).as_integer_ratio()
    return max(1, (span_num * radius_den) // (span_den * radius_num))


def _exact_rectangular_key(
    value: float,
    lo: float,
    hi: float,
    bins: int,
) -> int:
    """Map one binary64 coordinate to an exact sparse primary-bin key."""

    value_num, value_den = float(value).as_integer_ratio()
    lo_num, lo_den = float(lo).as_integer_ratio()
    offset_num = value_num * lo_den - lo_num * value_den
    offset_den = value_den * lo_den
    span_num, span_den = _exact_span_ratio(lo, hi)
    raw = (offset_num * span_den * bins) // (offset_den * span_num)
    return min(max(raw, 0), bins - 1)


def _unbounded_key(value: float, radius: float) -> int:
    quotient = value / radius
    if math.isfinite(quotient):
        return math.floor(quotient)
    value_num, value_den = float(value).as_integer_ratio()
    radius_num, radius_den = float(radius).as_integer_ratio()
    return (value_num * radius_den) // (value_den * radius_num)


def _unbounded_layout(points: np.ndarray, radius: float) -> _BucketLayout:
    keys = tuple(
        tuple(_unbounded_key(float(value), radius) for value in row)
        for row in points
    )
    dim = int(points.shape[1])
    return _BucketLayout(
        keys=keys,
        bins=(None,) * dim,
        periodic=(False,) * dim,
    )


def _rectangular_layout(points: np.ndarray, radius: float, geometry) -> _BucketLayout:
    bounds = geometry.native_bounds
    periodic = tuple(bool(value) for value in geometry.periodic_axes)
    bins = tuple(
        _rectangular_bin_count(float(lo), float(hi), radius)
        for lo, hi in bounds
    )
    keys: list[tuple[int, ...]] = []
    for row in points:
        key = []
        for axis, value in enumerate(row):
            lo = float(bounds[axis][0])
            hi = float(bounds[axis][1])
            key.append(
                _exact_rectangular_key(float(value), lo, hi, bins[axis])
            )
        keys.append(tuple(key))
    return _BucketLayout(
        keys=tuple(keys),
        bins=bins,
        periodic=periodic,
    )


def _fractional_layout(points: np.ndarray, radius: float, geometry) -> _BucketLayout:
    """Build the R5-SC-001 triclinic layout with exact rational keys.

    If an image displacement has Cartesian norm at most ``radius``, each exact
    fractional-coordinate change is at most ``radius * ||A^-1[:, k]||_1``.
    The exact bin counts below make every bin at least that wide.  Exact modulo
    and floor assignment therefore put such endpoints in the same or adjacent
    periodic bins on every axis, including at bucket boundaries and seams.
    """

    lattice = np.asarray(geometry.lattice_vectors_cart, dtype=np.float64)
    origin = np.asarray(
        getattr(geometry.domain, 'origin', (0.0,) * points.shape[1]),
        dtype=np.float64,
    )
    try:
        exact = _exact_triclinic_bucket_layout(
            points,
            origin=origin,
            lattice_vectors=lattice,
            radius=radius,
        )
    except (ValueError, ArithmeticError) as exc:
        raise DuplicateScanResourceError(
            'periodic duplicate candidate layout could not be certified: '
            f'{exc}'
        ) from None
    return _BucketLayout(
        keys=exact.keys,
        bins=exact.bins,
        periodic=(True,) * int(points.shape[1]),
    )


def _layout(points: np.ndarray, radius: float, geometry) -> _BucketLayout:
    if geometry is None or not geometry.has_any_periodic_axis:
        return _unbounded_layout(points, radius)
    if getattr(geometry, 'is_triclinic', False):
        return _fractional_layout(points, radius, geometry)
    return _rectangular_layout(points, radius, geometry)


def candidate_pairs(
    points: np.ndarray,
    *,
    radius: float,
    geometry=None,
) -> Iterator[tuple[int, int]]:
    """Yield each complete candidate pair once for the requested radius."""

    pts = np.asarray(points, dtype=np.float64)
    layout = _layout(pts, float(radius), geometry)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(layout.keys):
        for neighbor in layout.neighboring_keys(key):
            for j in buckets.get(neighbor, ()):
                yield int(j), int(i)
        buckets.setdefault(key, []).append(i)


def cross_candidate_pairs(
    reference_points: np.ndarray,
    query_points: np.ndarray,
    *,
    radius: float,
    geometry=None,
) -> Iterator[tuple[int, int]]:
    """Yield complete reference/query candidates without query/query pairs."""

    reference = np.asarray(reference_points, dtype=np.float64)
    queries = np.asarray(query_points, dtype=np.float64)
    combined = np.concatenate((reference, queries), axis=0)
    layout = _layout(combined, float(radius), geometry)
    split = int(reference.shape[0])
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(layout.keys[:split]):
        buckets.setdefault(key, []).append(i)
    for query_index, key in enumerate(layout.keys[split:]):
        for neighbor in layout.neighboring_keys(key):
            for reference_index in buckets.get(neighbor, ()):
                yield int(reference_index), int(query_index)


def _chunks(
    pairs: Iterable[tuple[int, int]],
    size: int,
) -> Iterator[tuple[tuple[int, int], ...]]:
    chunk: list[tuple[int, int]] = []
    for pair in pairs:
        chunk.append(pair)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def _unwrapped_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    return math.dist(
        tuple(float(value) for value in left),
        tuple(float(value) for value in right),
    )


def _exact_binary64_difference_squared_sum_less_equal(
    left: np.ndarray,
    right: np.ndarray,
    threshold_squared: float,
) -> bool:
    """Compare the exact dyadic source-coordinate distance to a limit."""

    parts: list[tuple[int, int]] = []
    common_exponent = 0
    for left_value, right_value in zip(left, right):
        left_numerator, left_denominator = (
            float(left_value).as_integer_ratio()
        )
        right_numerator, right_denominator = (
            float(right_value).as_integer_ratio()
        )
        left_exponent = left_denominator.bit_length() - 1
        right_exponent = right_denominator.bit_length() - 1
        coordinate_exponent = max(left_exponent, right_exponent)
        difference_numerator = (
            left_numerator << (coordinate_exponent - left_exponent)
        ) - (
            right_numerator << (coordinate_exponent - right_exponent)
        )
        squared_exponent = 2 * coordinate_exponent
        parts.append(
            (
                difference_numerator * difference_numerator,
                squared_exponent,
            )
        )
        common_exponent = max(common_exponent, squared_exponent)
    squared_numerator = sum(
        numerator << (common_exponent - exponent)
        for numerator, exponent in parts
    )
    threshold_numerator, threshold_denominator = (
        float(threshold_squared).as_integer_ratio()
    )
    return (
        squared_numerator * threshold_denominator
        <= threshold_numerator * (1 << common_exponent)
    )


def _scan(
    left_points: np.ndarray,
    right_points: np.ndarray,
    candidates: Iterable[tuple[int, int]],
    *,
    radius: float,
    geometry,
    inclusive_squared: float | None,
    max_pairs: int,
) -> PairScan:
    found: list[tuple[int, int, float]] = []
    candidate_count = 0
    minimum_image_used = bool(
        geometry is not None and geometry.has_any_periodic_axis
    )
    for chunk in _chunks(candidates, _CLASSIFICATION_BATCH):
        candidate_count += len(chunk)
        if candidate_count > _MAX_CANDIDATE_COMPARISONS:
            raise DuplicateScanResourceError(
                'complete duplicate scanning exceeds the private candidate '
                f'budget of {_MAX_CANDIDATE_COMPARISONS}'
            )
        minimum = None
        if minimum_image_used:
            pi = np.asarray([left_points[i] for i, _ in chunk], dtype=np.float64)
            pj = np.asarray([right_points[j] for _, j in chunk], dtype=np.float64)
            minimum = geometry.minimum_image_displacements(
                pi,
                pj,
                tie_orientation=np.ones(len(chunk), dtype=np.int8),
                image_search=1,
            )
        for offset, (i, j) in enumerate(chunk):
            if minimum is None:
                distance = _unwrapped_distance(left_points[i], right_points[j])
                if inclusive_squared is None:
                    close = distance < radius
                else:
                    close = (
                        _exact_binary64_difference_squared_sum_less_equal(
                            left_points[i],
                            right_points[j],
                            inclusive_squared,
                        )
                    )
            else:
                key = minimum.exact_distance_key[offset]
                if inclusive_squared is None:
                    close = exact_distance_less_than(key, radius)
                else:
                    close = exact_distance_squared_less_equal(
                        key,
                        inclusive_squared,
                    )
                distance = math.sqrt(float(minimum.distance_squared[offset]))
            if close:
                found.append((int(i), int(j), float(distance)))
                if len(found) >= max_pairs:
                    return PairScan(
                        pairs=tuple(found),
                        truncated=True,
                        candidate_count=candidate_count,
                        minimum_image_used=minimum_image_used,
                    )
    return PairScan(
        pairs=tuple(found),
        truncated=False,
        candidate_count=candidate_count,
        minimum_image_used=minimum_image_used,
    )


def scan_close_pairs(
    points: np.ndarray,
    *,
    radius: float,
    geometry=None,
    inclusive_squared: float | None = None,
    max_pairs: int,
) -> PairScan:
    """Classify within-set candidates using exact periodic final geometry."""

    pts = np.asarray(points, dtype=np.float64)
    return _scan(
        pts,
        pts,
        candidate_pairs(pts, radius=radius, geometry=geometry),
        radius=float(radius),
        geometry=geometry,
        inclusive_squared=inclusive_squared,
        max_pairs=max_pairs,
    )


def scan_cross_close_pairs(
    reference_points: np.ndarray,
    query_points: np.ndarray,
    *,
    radius: float,
    geometry=None,
    inclusive_squared: float | None = None,
    max_pairs: int,
) -> PairScan:
    """Classify reference/query candidates using exact periodic geometry."""

    reference = np.asarray(reference_points, dtype=np.float64)
    queries = np.asarray(query_points, dtype=np.float64)
    return _scan(
        reference,
        queries,
        cross_candidate_pairs(
            reference,
            queries,
            radius=radius,
            geometry=geometry,
        ),
        radius=float(radius),
        geometry=geometry,
        inclusive_squared=inclusive_squared,
        max_pairs=max_pairs,
    )


def candidate_pair_count(
    points: np.ndarray,
    *,
    radius: float,
    geometry=None,
    limit: int = _MAX_CANDIDATE_COMPARISONS,
) -> int:
    """Return bounded private scanner work evidence for tests/benchmarks."""

    count = 0
    for _pair in candidate_pairs(points, radius=radius, geometry=geometry):
        count += 1
        if count > limit:
            raise DuplicateScanResourceError(
                f'candidate count exceeds the requested evidence limit {limit}'
            )
    return count
