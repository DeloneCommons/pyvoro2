"""Independent exact planar power geometry for the WP6 E and S audits.

This module has no native witness input and never selects native provenance.
Each source is bounded by its exact axial self images and its real walls. For
each owner, exact nearest-period centering followed by three adjacent image
coefficients per periodic axis gives a complete finite family: every omitted
image is strictly farther than a retained image throughout that rectangle.
The same-owner weight cancels in this domination argument, including at ties.

Convex polygon clipping constructs the complete bounded cell with rational
arithmetic. Contacts are classified against that final cell; labels are never
merged when their lines coincide. Coordinates are physical source-local
``z = X - point[source]``, not doubled native coordinates or rounded views.

The caller supplies each ideal's weights: mathematical weights for S, exact
squares of separately exactified supplied radii for radius S, or exact squares
of actual stored backend radii for E. Periods are explicit operands, not exact
differences silently recomputed from rounded rectangular bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
import math
from numbers import Integral, Real
import operator
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


ExactPoint = tuple[Fraction, Fraction]
Shift = tuple[int, int]
BoundaryLabel = tuple[int, Shift] | int


class ExactAuditRefusal(RuntimeError):
    """A requested exact audit or numerical measure could not be completed.

    ``reason`` distinguishes proof resource exhaustion from an unavailable
    required binary64 view. Neither reason is an attribution failure or a
    mathematical assertion that an ideal contact is absent.
    """

    def __init__(self, reason, message, *, stage, resource=None,
                 observed=None, limit=None):
        super().__init__(message)
        self.reason = reason
        self.stage = stage
        self.resource = resource
        self.observed = observed
        self.limit = limit


@dataclass(slots=True)
class ExactAuditBudget:
    """Cumulative planar work limits, shareable across independent E/S ideals.

    A source has ``n * 3**p - 1`` generator-image constraints. The candidate
    ceiling bounds retained labeled contacts, while arithmetic work bounds
    clipping and classification even for cells with many vertices. Each
    normalized rational numerator and denominator is bit bounded; a primitive
    rational operation therefore has bounded raw integer factors as well.
    These are private fail-closed resource limits, not geometric tolerances.
    """

    max_candidates: int = 250_000
    max_work: int = 10_000_000
    max_bits: int = 16_384
    candidate_count: int = field(default=0, init=False)
    work: int = field(default=0, init=False)

    def __post_init__(self):
        for name in ('max_candidates', 'max_work', 'max_bits'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f'{name} must be an integer')
            value = operator.index(value)
            if value < 0:
                raise ValueError(f'{name} must be nonnegative')
            setattr(self, name, value)

    def _limit(self, observed, limit, resource, stage):
        if observed > limit:
            raise ExactAuditRefusal(
                'resource', f'exact planar {resource} budget exceeded',
                stage=stage, resource=resource, observed=observed, limit=limit,
            )

    def preflight(self, count: int) -> None:
        self._limit(self.candidate_count + count, self.max_candidates,
                    'candidates', 'ideal_candidate_family')

    def candidates(self, count: int) -> None:
        self.preflight(count)
        self.candidate_count += count

    def charge(self, count: int = 1, *, stage='ideal_arithmetic') -> None:
        self._limit(self.work + count, self.max_work, 'work', stage)
        self.work += count

    def integer(self, value: int, *, stage='ideal_arithmetic') -> None:
        self._limit(abs(value).bit_length(), self.max_bits, 'bits', stage)

    def fraction(self, value: Fraction, *, stage='ideal_arithmetic') -> None:
        self.integer(value.numerator, stage=stage)
        self.integer(value.denominator, stage=stage)


class _Arithmetic:
    def __init__(self, budget):
        self.budget = budget

    def exact(self, value) -> Fraction:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError('exact geometry scalars must not be Boolean')
        if isinstance(value, Fraction):
            result = Fraction(int(value.numerator), int(value.denominator))
        elif isinstance(value, Integral):
            result = Fraction(int(value))
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError('exact geometry scalars must be finite')
            result = Fraction.from_float(numeric)
        else:
            raise TypeError('exact geometry needs rational or real scalar operands')
        self.budget.fraction(result, stage='ideal_input')
        return result

    def check(self, value):
        self.budget.charge()
        self.budget.fraction(value)
        return value

    def add(self, left, right):
        return self.check(left + right)

    def sub(self, left, right):
        return self.check(left - right)

    def mul(self, left, right):
        return self.check(left * right)

    def div(self, left, right):
        return self.check(left / right)

    def dot(self, left, right):
        return self.add(self.mul(left[0], right[0]), self.mul(left[1], right[1]))

    def difference(self, left, right):
        return self.sub(left[0], right[0]), self.sub(left[1], right[1])

    def cross(self, left, right):
        return self.sub(self.mul(left[0], right[1]), self.mul(left[1], right[0]))


@dataclass(frozen=True, slots=True)
class ExactContact:
    """Exact full-cell contact; a line segment's endpoints are sorted.

    Status is ``positive``, ``point``, ``lower-dimensional``, ``identical`` or
    ``absent``. Only ``positive`` denotes a one-dimensional edge of a full
    two-dimensional cell. An identical power function has no unique support
    line and retains the entire cell's vertices and dimension, even if empty.
    """

    status: str
    dimension: int
    endpoints: tuple[ExactPoint, ...]
    length_squared: Fraction


_ABSENT = ExactContact('absent', -1, (), Fraction())


@dataclass(frozen=True, slots=True)
class _Cut:
    label: BoundaryLabel
    normal: ExactPoint
    offset: Fraction


@dataclass(frozen=True, slots=True)
class IdealCell:
    """Complete exact cell; dimension -1 is empty, 0/1 is degenerate.

    Generator labels are ``(owner, (sx, sy))``. Wall labels are actual native
    side codes -1 x-low, -2 x-high, -3 y-low, -4 y-high. ``positive`` retains
    every distinct positive provenance label exactly once, including labels
    sharing an identical geometric segment.
    """

    source: int
    dimension: int
    vertices: tuple[ExactPoint, ...]
    area: Fraction
    contacts: Mapping[BoundaryLabel, ExactContact]
    positive: Mapping[BoundaryLabel, ExactContact]
    _ideal: ExactIdeal = field(repr=False, compare=False)
    _contacts: dict[BoundaryLabel, ExactContact] = field(repr=False, compare=False)

    def contact(self, owner: int, shift: Sequence[int]) -> ExactContact:
        """Classify a specified image, including a far public representative.

        No int64 conversion is required for this exact semantic question.
        The source's identity comparison and shifts on nonperiodic axes are
        not boundary candidates and are reported absent.
        """

        ideal = self._ideal
        owner, shift = ideal._index(owner), ideal._shift(shift)
        if (owner == self.source and shift == (0, 0)) or any(
            value and not flag for value, flag in zip(shift, ideal.periodic)
        ):
            return _ABSENT
        label = owner, shift
        if label not in self._contacts:
            ideal.budget.candidates(1)
            cut = ideal._cut(self.source, owner, shift)
            self._contacts[label] = ideal._contact(self.vertices, self.dimension, cut)
        return self._contacts[label]

    def wall(self, side: int) -> ExactContact:
        """Return a real wall contact; a periodic initialization is not a wall."""

        side = operator.index(side)
        if side not in (-1, -2, -3, -4):
            raise ValueError('wall side must be -1, -2, -3 or -4')
        axis = (-side - 1) // 2
        if self._ideal.periodic[axis]:
            return _ABSENT
        return self._contacts[side]


class ExactIdeal:
    """Immutable rational input snapshot with lazily constructed source cells.

    ``bounds`` has two increasing axis pairs; ``periods`` has the two actual
    positive represented spans. Nonperiodic directions use their real bounds,
    and all image coefficients in those directions are zero. Callers own the
    E/S input distinction; no backend conversion is performed here.
    """

    def __init__(self, points, weights, bounds, periods, periodic, *, budget=None):
        self.budget = ExactAuditBudget() if budget is None else budget
        self._arithmetic = a = _Arithmetic(self.budget)
        self.points = tuple(self._point(point) for point in points)
        self.weights = tuple(a.exact(value) for value in weights)
        if len(self.points) != len(self.weights):
            raise ValueError('exact points and weights must have equal lengths')
        self.bounds = tuple(self._point(pair) for pair in bounds)
        if len(self.bounds) != 2 or any(low >= high for low, high in self.bounds):
            raise ValueError('bounds need two increasing axis intervals')
        self.periods = self._point(periods)
        if any(length <= 0 for length in self.periods):
            raise ValueError('periods must be positive')
        self.periodic = tuple(periodic)
        if len(self.periodic) != 2 or not all(
            isinstance(value, (bool, np.bool_)) for value in self.periodic
        ):
            raise ValueError('periodic needs two Boolean flags')
        self.periodic = tuple(bool(value) for value in self.periodic)
        self._image_count = len(self.points) * 3 ** sum(self.periodic) - 1
        self._cells: dict[int, IdealCell] = {}

    def _point(self, values) -> ExactPoint:
        result = tuple(self._arithmetic.exact(value) for value in values)
        if len(result) != 2:
            raise ValueError('exact planar coordinates need two components')
        return result

    def _index(self, source):
        if isinstance(source, (bool, np.bool_)):
            raise TypeError('source index must be an integer')
        source = operator.index(source)
        if source < 0 or source >= len(self.points):
            raise ValueError('source index is out of range')
        return source

    def _shift(self, values) -> Shift:
        result = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError('image shifts must be integers')
            value = operator.index(value)
            self.budget.integer(value, stage='ideal_shift')
            result.append(value)
        if len(result) != 2:
            raise ValueError('planar image shifts need two integers')
        return tuple(result)

    def _cut(self, source, owner, shift):
        a = self._arithmetic
        displacement = tuple(
            a.sub(a.add(self.points[owner][axis],
                        a.mul(self.periods[axis], shift[axis])),
                  self.points[source][axis]) for axis in range(2)
        )
        normal = tuple(a.mul(value, 2) for value in displacement)
        offset = a.add(a.dot(displacement, displacement),
                       a.sub(self.weights[source], self.weights[owner]))
        return _Cut((owner, shift), normal, offset)

    def _outer(self, source):
        a = self._arithmetic
        intervals, walls = [], []
        for axis in range(2):
            if self.periodic[axis]:
                half = a.div(self.periods[axis], 2)
                low, high = -half, half
            else:
                low, high = tuple(a.sub(v, self.points[source][axis])
                                  for v in self.bounds[axis])
                for side, (sign, value) in enumerate(((-1, -low), (1, high))):
                    normal = ((Fraction(sign), Fraction()) if axis == 0
                              else (Fraction(), Fraction(sign)))
                    walls.append(_Cut(-(2 * axis + side + 1), normal, value))
            intervals.append((low, high))
        (left, right), (bottom, top) = intervals
        return ((left, bottom), (right, bottom), (right, top), (left, top)), walls

    def _cuts(self, source):
        a = self._arithmetic
        for owner, point in enumerate(self.points):
            ranges = []
            for axis in range(2):
                if self.periodic[axis]:
                    ratio = a.div(a.sub(self.points[source][axis], point[axis]),
                                  self.periods[axis])
                    center = math.floor(a.add(ratio, Fraction(1, 2)))
                    self.budget.integer(center, stage='ideal_image_center')
                    self.budget.integer(center - 1, stage='ideal_image_center')
                    self.budget.integer(center + 1, stage='ideal_image_center')
                    ranges.append(range(center - 1, center + 2))
                else:
                    ranges.append((0,))
            for shift in product(*ranges):
                if owner != source or shift != (0, 0):
                    yield self._cut(source, owner, shift)

    def _clip(self, vertices, cut):
        if not vertices:
            return ()
        a = self._arithmetic
        gaps = tuple(a.sub(a.dot(cut.normal, p), cut.offset) for p in vertices)
        if all(gap <= 0 for gap in gaps):
            return vertices
        if all(gap > 0 for gap in gaps):
            return ()
        result = []
        for index, left in enumerate(vertices):
            next_index = (index + 1) % len(vertices)
            right = vertices[next_index]
            left_gap, right_gap = gaps[index], gaps[next_index]
            if left_gap <= 0:
                result.append(left)
            if (left_gap < 0 < right_gap) or (right_gap < 0 < left_gap):
                fraction = a.div(left_gap, a.sub(left_gap, right_gap))
                point = tuple(a.add(x, a.mul(fraction, a.sub(y, x)))
                              for x, y in zip(left, right))
                result.append(point)
        return tuple(dict.fromkeys(result))

    def _hull(self, vertices):
        ordered = sorted(set(vertices))
        if len(ordered) < 2:
            return tuple(ordered)
        a = self._arithmetic
        halves = []
        for sequence in (ordered, reversed(ordered)):
            half = []
            for point in sequence:
                while len(half) >= 2:
                    left = a.difference(half[-1], half[-2])
                    right = a.difference(point, half[-2])
                    if a.cross(left, right) > 0:
                        break
                    half.pop()
                half.append(point)
            halves.extend(half[:-1])
        return tuple(halves)

    def _contact(self, vertices, dimension, cut):
        a = self._arithmetic
        if not any(cut.normal):
            if cut.offset == 0:
                return ExactContact('identical', dimension, tuple(sorted(vertices)),
                                    Fraction())
            return _ABSENT
        endpoints = []
        for point in vertices:
            value = a.dot(cut.normal, point)
            if value > cut.offset:
                raise RuntimeError('complete exact planar cell violates an image cut')
            if value == cut.offset:
                endpoints.append(point)
        if not endpoints:
            return _ABSENT
        endpoints = tuple(sorted(endpoints))
        contact_dimension = min(1, len(endpoints) - 1)
        status = ('lower-dimensional' if dimension < 2 else 'point'
                  if contact_dimension == 0 else 'positive')
        delta = a.difference(endpoints[-1], endpoints[0])
        return ExactContact(status, contact_dimension, endpoints, a.dot(delta, delta))

    def cell(self, source: int) -> IdealCell:
        """Publish a cell only after its complete construction/classification."""

        source = self._index(source)
        if source in self._cells:
            return self._cells[source]
        # The complete family size is known before any image is generated.
        self.budget.candidates(self._image_count)
        vertices, walls = self._outer(source)
        cuts = walls + list(self._cuts(source))
        for cut in cuts:
            vertices = self._clip(vertices, cut)
        vertices = self._hull(vertices)
        dimension = min(2, len(vertices) - 1)
        a = self._arithmetic
        area = Fraction()
        if dimension == 2:
            for left, right in zip(vertices, vertices[1:] + vertices[:1]):
                area = a.add(area, a.cross(left, right))
            area = a.div(abs(area), 2)
        contacts = {cut.label: self._contact(vertices, dimension, cut) for cut in cuts}
        positive = {label: contact for label, contact in contacts.items()
                    if contact.status == 'positive'}
        result = IdealCell(source, dimension, vertices, area,
                           MappingProxyType(contacts), MappingProxyType(positive),
                           self, contacts)
        self._cells[source] = result
        return result


def ideal_cells(points, weights, bounds, periods, periodic, *, budget=None):
    """Atomically construct all sources after preflighting the full image count."""

    ideal = ExactIdeal(points, weights, bounds, periods, periodic, budget=budget)
    if ideal.points:
        ideal.budget.preflight(len(ideal.points) * ideal._image_count)
    return tuple(ideal.cell(source) for source in range(len(ideal.points)))


def length_from_squared(value, *, budget=None) -> float:
    """Correctly round sqrt(exact squared length) to a finite binary64 view.

    Rounding is performed on exact scaled integers with ties to even. This
    also handles a squared length outside binary64's range whose square root
    is representable. A positive length rounding to zero or infinity raises
    a representation refusal; no threshold defines semantic positivity.
    """

    budget = ExactAuditBudget() if budget is None else budget
    a = _Arithmetic(budget)
    value = a.exact(value)
    if value < 0:
        raise ValueError('squared length must be nonnegative')
    if not value:
        return 0.0
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator < denominator << exponent if exponent >= 0
            else numerator << -exponent < denominator):
        exponent -= 1
    root_exponent = exponent // 2
    if root_exponent > 1023:
        raise ExactAuditRefusal('representation', 'exact length exceeds binary64 range',
                                stage='ideal_length')
    unit_exponent = max(root_exponent - 52, -1074)
    scale = (Fraction(1 << (2 * unit_exponent)) if unit_exponent >= 0
             else Fraction(1, 1 << (-2 * unit_exponent)))
    budget.fraction(scale, stage='ideal_length')
    scaled = a.div(value, scale)
    lower = math.isqrt(scaled.numerator // scaled.denominator)
    budget.charge(stage='ideal_length')
    midpoint_square_times_four = (2 * lower + 1) ** 2
    budget.integer(midpoint_square_times_four, stage='ideal_length')
    four_scaled = a.mul(scaled, 4)
    if four_scaled > midpoint_square_times_four or (
        four_scaled == midpoint_square_times_four and lower % 2
    ):
        lower += 1
    try:
        result = math.ldexp(float(lower), unit_exponent)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result) or result == 0:
        raise ExactAuditRefusal('representation',
                                'positive exact length has no finite positive view',
                                stage='ideal_length')
    return result
