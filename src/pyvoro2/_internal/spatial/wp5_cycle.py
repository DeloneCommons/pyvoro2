"""Exact projected native-support observation audit fixed by ADR 0021.

The returned private cycle is an observation proof, not replacement native
geometry and not a measure or polygon of either exact E/S ideal.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

from .wp5_common import WP5Budget, WP5Failure


Point = tuple[Fraction, Fraction, Fraction]
PlanePoint = tuple[Fraction, Fraction]
_STAGE = 'native cycle'


def _failure(code: str, message: str) -> None:
    raise WP5Failure(code, message)


def _checked(value: Fraction, budget: WP5Budget) -> Fraction:
    budget.fraction(value, stage=_STAGE)
    return value


def _exact(value: object, budget: WP5Budget) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise WP5Failure(
            'WP5_SOURCE_PROFILE_MISMATCH',
            'native cycle support and coordinates must be finite numbers',
        ) from error
    return _checked(result, budget)


def _point(values: Iterable[object], budget: WP5Budget) -> Point:
    try:
        values = tuple(values)
    except TypeError as error:
        raise WP5Failure(
            'WP5_SOURCE_PROFILE_MISMATCH',
            'native cycle requires three-dimensional coordinates',
        ) from error
    if len(values) != 3:
        _failure('WP5_SOURCE_PROFILE_MISMATCH',
                 'native cycle requires three-dimensional coordinates')
    return tuple(_exact(value, budget) for value in values)


def _dot(left: Point, right: Point, budget: WP5Budget) -> Fraction:
    total = Fraction(0)
    for x, y in zip(left, right):
        total = _checked(total + _checked(x * y, budget), budget)
    return total


def _orient(a: PlanePoint, b: PlanePoint, c: PlanePoint,
            budget: WP5Budget) -> Fraction:
    budget.charge(stage=_STAGE)
    ab = tuple(_checked(b[k] - a[k], budget) for k in range(2))
    ac = tuple(_checked(c[k] - a[k], budget) for k in range(2))
    return _checked(_checked(ab[0] * ac[1], budget)
                    - _checked(ab[1] * ac[0], budget), budget)


def _on_segment(a: PlanePoint, point: PlanePoint, b: PlanePoint) -> bool:
    """Bounding-box membership after a caller has proved collinearity."""
    return all(min(a[k], b[k]) <= point[k] <= max(a[k], b[k])
               for k in range(2))


def _segments_touch(a: PlanePoint, b: PlanePoint, c: PlanePoint,
                    d: PlanePoint, budget: WP5Budget) -> bool:
    abc = _orient(a, b, c, budget)
    abd = _orient(a, b, d, budget)
    cda = _orient(c, d, a, budget)
    cdb = _orient(c, d, b, budget)
    if ((abc == 0 and _on_segment(a, c, b))
            or (abd == 0 and _on_segment(a, d, b))
            or (cda == 0 and _on_segment(c, a, d))
            or (cdb == 0 and _on_segment(c, b, d))):
        return True
    return ((abc > 0 and abd < 0) or (abc < 0 and abd > 0)) and (
        (cda > 0 and cdb < 0) or (cda < 0 and cdb > 0))


def audit_cycle(vertices_doubled: Iterable[Iterable[object]],
                normal: Iterable[object], offset: object, *,
                budget: WP5Budget | None = None) -> tuple[Point, ...]:
    """Project exactly and accept only an ordered simple convex 2D cycle.

    Consecutive equal projected points and strictly between-collinear points
    may be removed. Repeated nonconsecutive vertices, crossings, touches,
    overlaps and opposed turns fail without reordering or numerical tolerances.
    Rank-three original observations may pass after projection onto support.
    """
    budget = WP5Budget() if budget is None else budget
    n = _point(normal, budget)
    h = _exact(offset, budget)
    norm_squared = _dot(n, n, budget)
    if norm_squared == 0:
        _failure('WP5_SOURCE_PROFILE_MISMATCH',
                 'native cycle support must have a nonzero normal')
    dropped = next(k for k, value in enumerate(n) if value != 0)
    kept = tuple(k for k in range(3) if k != dropped)

    projected = []
    for raw_point in vertices_doubled:
        budget.charge(stage=_STAGE)
        point = _point(raw_point, budget)
        residual = _checked(_dot(n, point, budget) - h, budget)
        factor = _checked(residual / norm_squared, budget)
        value = tuple(_checked(point[k] - _checked(n[k] * factor, budget),
                               budget) for k in range(3))
        if not projected or value != projected[-1]:
            projected.append(value)
    if len(projected) > 1 and projected[-1] == projected[0]:
        projected.pop()
    plane = [tuple(point[k] for k in kept) for point in projected]
    unique = set(plane)
    if len(unique) < 3 or not any(
            _orient(plane[0], plane[1], point, budget) != 0
            for point in plane[2:]):
        _failure('WP5_NATIVE_CYCLE_COLLAPSED',
                 'projected native cycle has affine dimension below two')
    # Reject these before removing a collinear point that could hide a repeat.
    if len(unique) != len(plane):
        _failure('WP5_NATIVE_CYCLE_INVALID',
                 'projected native cycle repeats a nonconsecutive vertex')

    while True:
        for index in range(len(plane)):
            previous = plane[index - 1]
            current = plane[index]
            following = plane[(index + 1) % len(plane)]
            if _orient(previous, current, following, budget) != 0:
                continue
            if not _on_segment(previous, current, following):
                _failure('WP5_NATIVE_CYCLE_INVALID',
                         'projected native cycle backtracks on an edge')
            # All vertices are distinct, so closed membership is strict here.
            del plane[index]
            del projected[index]
            break
        else:
            break

    count = len(plane)
    # This is the complete nonadjacent edge-pair set, not a searched prefix.
    budget.charge(count * (count - 3) // 2, stage=_STAGE)
    for first in range(count):
        a, b = plane[first], plane[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = plane[second], plane[(second + 1) % count]
            if _segments_touch(a, b, c, d, budget):
                _failure('WP5_NATIVE_CYCLE_INVALID',
                         'projected native nonadjacent edges cross or touch')

    turns = [_orient(plane[index - 1], plane[index],
                     plane[(index + 1) % count], budget)
             for index in range(count)]
    if not (all(turn > 0 for turn in turns)
            or all(turn < 0 for turn in turns)):
        _failure('WP5_NATIVE_CYCLE_INVALID',
                 'projected native cycle is not a convex ordered boundary')
    area_twice = Fraction(0)
    for index in range(1, count - 1):
        area_twice = _checked(
            area_twice + _orient(plane[0], plane[index], plane[index + 1],
                                 budget), budget)
    if area_twice == 0:
        _failure('WP5_NATIVE_CYCLE_INVALID',
                 'projected native cycle has zero oriented area')
    return tuple(projected)
