"""Bounded exact 3D power cells for the independent WP5 E and S ideals.

Every coordinate is an exact rational; vertices use doubled source-local
coordinates. Self-image slabs and real walls enclose the full cell. An exact
outward square root of their maximum vertex squared norm bounds its radius,
giving a complete finite image region before enumeration. Triclinic regions
use the accepted exact reduced basis, inverse columns and integer map to the
input row basis.

The polytope starts with eight exact vertices. Clipping retains old vertices
and intersects crossing edges; two distinct vertices share an edge exactly
when their common active normals have rank two. This remains true for bounded
lower-dimensional cells. Geometry constraints may coincide, but semantic
labels remain distinct. Final contacts are classified only on the full cell.

An area need not be rational. ``area_squared`` is the exact square of the
physical facet area; this module does not materialize a floating measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
import math
import operator
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..exact_lattice import (
    ExactLatticeReductionInvariantError,
    ExactLatticeReductionResourceError,
)
from ..periodic_images import _basis_key, _prepare_basis
from .wp5_common import WP5Budget, WP5Failure


ExactPoint = tuple[Fraction, Fraction, Fraction]
Shift = tuple[int, int, int]
BoundaryLabel = tuple[int, Shift] | tuple[str, int]
_ZERO = (Fraction(), Fraction(), Fraction())


class _Arithmetic:
    """Observe semantic operands/results before later cancellation."""

    def __init__(self, budget: WP5Budget) -> None:
        self.budget = budget

    def exact(self, value: object) -> Fraction:
        raw = Fraction(value)
        # Fraction(np.int64(...)) retains a NumPy numerator. Normalize once
        # before any geometry so all subsequent integer arithmetic is unbounded.
        result = Fraction(int(raw.numerator), int(raw.denominator))
        self.budget.fraction(result, stage='ideal_input')
        return result

    def check(self, value: Fraction) -> Fraction:
        self.budget.fraction(value, stage='ideal_arithmetic')
        self.budget.charge(stage='ideal_arithmetic')
        return value

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        return self.check(a + b)

    def sub(self, a: Fraction, b: Fraction) -> Fraction:
        return self.check(a - b)

    def mul(self, a: Fraction, b: Fraction | int) -> Fraction:
        return self.check(a * b)

    def div(self, a: Fraction, b: Fraction | int) -> Fraction:
        return self.check(a / b)

    def sum(self, values) -> Fraction:
        result = Fraction()
        for value in values:
            result = self.add(result, value)
        return result

    def dot(self, a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
        return self.sum(self.mul(x, y) for x, y in zip(a, b) if x and y)

    def difference(self, a: ExactPoint, b: ExactPoint) -> ExactPoint:
        return tuple(self.sub(x, y) for x, y in zip(a, b))

    def cross(self, a: ExactPoint, b: ExactPoint) -> ExactPoint:
        return tuple(self.sub(self.mul(a[j], b[k]), self.mul(a[k], b[j]))
                     for j, k in ((1, 2), (2, 0), (0, 1)))

    def affine_dimension(self, points: Sequence[ExactPoint]) -> int:
        if not points:
            return -1
        first = next((self.difference(p, points[0]) for p in points[1:]
                      if p != points[0]), None)
        if first is None:
            return 0
        normal = None
        for point in points[1:]:
            cross = self.cross(first, self.difference(point, points[0]))
            if any(cross):
                normal = cross
                break
        if normal is None:
            return 1
        if any(self.dot(normal, self.difference(p, points[0])) for p in points[1:]):
            return 3
        return 2


@dataclass(frozen=True, slots=True)
class ExactContact:
    """One full-cell contact, with physical area squared kept exact."""

    status: str
    dimension: int
    vertices: tuple[ExactPoint, ...]
    area_squared: Fraction


_ABSENT = ExactContact('absent', -1, (), Fraction())


@dataclass(frozen=True, slots=True)
class _Cut:
    label: BoundaryLabel
    normal: ExactPoint
    offset: Fraction


@dataclass(frozen=True, slots=True)
class ExactCell:
    """A fully reconstructed bounded cell; dimension -1 denotes emptiness."""

    source: int
    dimension: int
    vertices: tuple[ExactPoint, ...]
    facets: Mapping[BoundaryLabel, ExactContact]
    _ideal: ExactIdeal = field(repr=False, compare=False)
    _contacts: dict[BoundaryLabel, ExactContact] = field(repr=False, compare=False)

    def contact(self, owner: int, shift: Sequence[int]) -> ExactContact:
        """Classify any allowed image, including images outside the proof box."""

        owner = self._ideal._index(owner)
        shift = self._ideal._shift(shift)
        if (owner == self.source and not any(shift)) or any(
            s and not periodic for s, periodic in zip(shift, self._ideal.periodic)
        ):
            return _ABSENT
        label = (owner, shift)
        if label not in self._contacts:
            cut = self._ideal._cut(self.source, owner, shift)
            self._contacts[label] = self._ideal._contact(self.vertices, cut)
        return self._contacts[label]

    def wall(self, side_code: int) -> ExactContact:
        """Classify a real rectangular wall, retaining its native -1..-6 ID."""

        side_code = operator.index(side_code)
        if side_code not in range(-6, 0):
            raise ValueError('wall side code must be in -6..-1')
        label = ('wall', side_code)
        if label not in self._contacts:
            axis, upper = divmod(-side_code - 1, 2)
            if self._ideal.periodic[axis]:
                return _ABSENT
            cut = self._ideal._wall_cut(self.source, axis, upper)
            self._contacts[label] = self._ideal._contact(self.vertices, cut)
        return self._contacts[label]


class _Polytope:
    def __init__(self, cuts: Sequence[_Cut], vertices, arithmetic: _Arithmetic):
        self.cuts = list(cuts)
        self.vertices = vertices
        self.arithmetic = arithmetic

    def _edge(self, left: frozenset[int], right: frozenset[int]) -> bool:
        common = iter(sorted(left & right))
        first = next(common, None)
        if first is None:
            return False
        a = self.cuts[first].normal
        return any(any(self.arithmetic.cross(a, self.cuts[index].normal))
                   for index in common)

    def clip(self, cut: _Cut) -> None:
        """Intersect using true edges; never substitute all vertex pairs."""

        if not self.vertices:
            return
        arithmetic = self.arithmetic
        gaps = {point: arithmetic.sub(arithmetic.dot(cut.normal, point), cut.offset)
                for point in self.vertices}
        if all(gap < 0 for gap in gaps.values()):
            return
        index = len(self.cuts)
        self.cuts.append(cut)
        retained = {
            point: active | {index} if gaps[point] == 0 else active
            for point, active in self.vertices.items() if gaps[point] <= 0
        }
        inside = [p for p, gap in gaps.items() if gap < 0]
        outside = [p for p, gap in gaps.items() if gap > 0]
        arithmetic.budget.charge(len(inside) * len(outside), stage='ideal_edges')
        for left in inside:
            for right in outside:
                left_active, right_active = self.vertices[left], self.vertices[right]
                if not self._edge(left_active, right_active):
                    continue
                difference = arithmetic.sub(gaps[left], gaps[right])
                amount = arithmetic.div(gaps[left], difference)
                point = tuple(arithmetic.add(a, arithmetic.mul(
                    amount, arithmetic.sub(b, a)))
                              for a, b in zip(left, right))
                active = (left_active & right_active) | {index}
                retained[point] = retained.get(point, frozenset()) | active
        self.vertices = retained


@dataclass(frozen=True, slots=True)
class _Outer:
    cuts: tuple[_Cut, ...]
    vertices: Mapping[ExactPoint, frozenset[int]]
    inverse: tuple[ExactPoint, ExactPoint, ExactPoint]
    intervals: tuple[tuple[Fraction, Fraction], ...]
    radius: Fraction

    def support(self, normal: ExactPoint, arithmetic: _Arithmetic) -> Fraction:
        # Y = inverse @ t for a column t in the three defining intervals.
        values = []
        for axis, (lower, upper) in enumerate(self.intervals):
            column = tuple(row[axis] for row in self.inverse)
            coefficient = arithmetic.dot(normal, column)
            endpoint = upper if coefficient >= 0 else lower
            values.append(arithmetic.mul(coefficient, endpoint))
        return arithmetic.sum(values)


class ExactIdeal:
    """One exact E or S input snapshot; instances never share cells/weights.

    Sites/weights accept exact fractions or binary64 operands, exactified
    separately. Lattice entries are the supplied binary64 source values (an
    exact Fraction spelling is also accepted). Only Cartesian diagonal partial
    periodicity and fully periodic rank-three lattices are supported. Bounds
    are axis pairs ``((xmin,xmax), (ymin,ymax), (zmin,zmax))``.
    """

    def __init__(self, sites, lattice, weights, periodic, bounds=None, budget=None):
        self.budget = WP5Budget() if budget is None else budget
        self._arithmetic = arithmetic = _Arithmetic(self.budget)
        self.sites = tuple(self._point(row) for row in sites)
        self.lattice = tuple(self._point(row) for row in lattice)
        self.weights = tuple(arithmetic.exact(value) for value in weights)
        self.periodic = tuple(periodic)
        if len(self.lattice) != 3 or len(self.periodic) != 3:
            raise ValueError('exact ideal requires three-dimensional geometry')
        if len(self.weights) != len(self.sites):
            raise ValueError('exact ideal weights and sites must have equal lengths')
        if not all(isinstance(value, (bool, np.bool_)) for value in self.periodic):
            raise ValueError('periodic axes must be Boolean')
        self.bounds = None
        if bounds is not None:
            self.bounds = tuple(tuple(arithmetic.exact(value) for value in pair)
                                for pair in bounds)
            if len(self.bounds) != 3 or any(
                len(pair) != 2 or pair[0] >= pair[1] for pair in self.bounds
            ):
                raise ValueError('bounds must contain three increasing axis intervals')
        if not all(self.periodic) and self.bounds is None:
            raise ValueError('nonperiodic directions require exact real wall bounds')
        try:
            lattice_float = np.asarray(self.lattice, dtype=np.float64)
            if not np.isfinite(lattice_float).all() or any(
                Fraction.from_float(float(value)) != exact
                for row, exact_row in zip(lattice_float, self.lattice)
                for value, exact in zip(row, exact_row)
            ):
                raise ValueError(
                    'lattice must represent its original finite binary64 operands'
                )
            self._basis = _prepare_basis(*_basis_key(lattice_float, self.periodic))
        except ExactLatticeReductionResourceError as exc:
            raise WP5Failure('WP5_RESOURCE_LIMIT', str(exc), stage=exc.stage,
                             resource=exc.resource, required=exc.observed,
                             limit=exc.configured_limit) from exc
        except ExactLatticeReductionInvariantError as exc:
            raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH', str(exc),
                             stage='ideal_reduced_basis') from exc
        for row in self._basis.fractions:
            for value in row:
                arithmetic.check(value)
        if self._basis.inverse is not None:
            for row in self._basis.inverse:
                for value in row:
                    arithmetic.check(value)
        self._cells: dict[int, ExactCell] = {}

    def _point(self, values) -> ExactPoint:
        row = tuple(self._arithmetic.exact(value) for value in values)
        if len(row) != 3:
            raise ValueError('exact coordinates must have three components')
        return row

    def _index(self, index: int) -> int:
        index = operator.index(index)
        if not 0 <= index < len(self.sites):
            raise ValueError('ideal generator index is out of range')
        return index

    def _shift(self, values: Sequence[int]) -> Shift:
        result = tuple(operator.index(value) for value in values)
        if len(result) != 3:
            raise ValueError('ideal image shift must have three integers')
        for value in result:
            self.budget.integer(value, stage='ideal_shift')
        return result

    def _map_shift(self, reduced: Shift) -> Shift:
        if self._basis.reduction is None:
            return self._shift(reduced)
        transform = self._basis.reduction.transform
        result = []
        for axis in range(3):
            total = 0
            for coefficient, row in zip(reduced, transform):
                self.budget.integer(row[axis], stage='ideal_shift_map')
                term = coefficient * row[axis]
                self.budget.integer(term, stage='ideal_shift_map')
                total += term
                self.budget.integer(total, stage='ideal_shift_map')
            result.append(total)
        self.budget.charge(9, stage='ideal_shift_map')
        return tuple(result)

    def _displacement(self, source: int, owner: int, shift: Shift, rows) -> ExactPoint:
        a = self._arithmetic
        return tuple(a.add(a.sub(self.sites[owner][axis], self.sites[source][axis]),
                           a.sum(a.mul(row[axis], coefficient)
                                 for coefficient, row in zip(shift, rows)
                                 if coefficient and row[axis]))
                     for axis in range(3))

    def _cut(self, source: int, owner: int, shift: Shift) -> _Cut:
        a = self._arithmetic
        normal = self._displacement(source, owner, shift, self.lattice)
        delta = a.sub(self.weights[source], self.weights[owner])
        offset = a.add(a.dot(normal, normal), delta)
        return _Cut((owner, shift), normal, offset)

    def _wall_cut(self, source: int, axis: int, upper: int) -> _Cut:
        a = self._arithmetic
        sign = 1 if upper else -1
        normal = tuple(Fraction(sign if k == axis else 0) for k in range(3))
        distance = a.sub(self.bounds[axis][upper], self.sites[source][axis])
        offset = a.mul(distance, 2 * sign)
        return _Cut(('wall', -(2 * axis + upper + 1)), normal, offset)

    def _outer(self, source: int) -> _Outer:
        a = self._arithmetic
        rows, intervals, cuts = [], [], []
        for axis in range(3):
            if self.periodic[axis]:
                if self._basis.orthogonal:
                    row = tuple(
                        abs(self.lattice[axis][axis]) if k == axis else Fraction()
                        for k in range(3)
                    )
                    sign = 1 if self.lattice[axis][axis] > 0 else -1
                    shift = tuple(sign if k == axis else 0 for k in range(3))
                else:
                    row = self._basis.fractions[axis]
                    shift = self._map_shift(tuple(int(k == axis) for k in range(3)))
                offset = a.dot(row, row)
                intervals.append((-offset, offset))
                negative = _Cut((source, tuple(-s for s in shift)),
                                tuple(-v for v in row), offset)
                cuts.extend((negative,
                             _Cut((source, shift), row, offset)))
            else:
                row = tuple(Fraction(int(k == axis)) for k in range(3))
                lower, upper = (self._wall_cut(source, axis, side) for side in (0, 1))
                intervals.append((-lower.offset, upper.offset))
                cuts.extend((lower, upper))
            rows.append(row)
        if self._basis.orthogonal:
            inverse = tuple(tuple(
                a.div(Fraction(1), rows[k][k]) if k == j else Fraction()
                for j in range(3)
            ) for k in range(3))
        else:
            inverse = self._basis.inverse
        vertices = {}
        for sides in product((0, 1), repeat=3):
            values = tuple(intervals[k][sides[k]] for k in range(3))
            point = tuple(a.dot(row, values) for row in inverse)
            vertices[point] = frozenset(2 * k + sides[k] for k in range(3))
        radius_squared = max(a.dot(point, point) for point in vertices)
        radius = self._sqrt_upper(radius_squared, extra_bits=32)
        return _Outer(tuple(cuts), vertices, inverse, tuple(intervals), radius)

    def _sqrt_upper(self, value: Fraction, *, extra_bits: int = 0) -> Fraction:
        """Enclose sqrt(value) rationally; additional bits only tighten work.

        ceil_sqrt(n*d*2**(2*k)) / (d*2**k) is an exact upper bound for
        sqrt(n/d). Perfect rational squares retain their exact root. The
        outer norm uses k=32 so unit-scale irrational radii are tight; no
        floating square root participates in a coefficient bound.
        """

        a = self._arithmetic
        numerator_root = math.isqrt(value.numerator)
        denominator_root = math.isqrt(value.denominator)
        self.budget.charge(2, stage='ideal_radius')
        if (numerator_root * numerator_root == value.numerator
                and denominator_root * denominator_root == value.denominator):
            return a.check(Fraction(numerator_root, denominator_root))
        product_value = value.numerator * value.denominator
        self.budget.integer(product_value, stage='ideal_radius')
        product_value <<= 2 * extra_bits
        self.budget.integer(product_value, stage='ideal_radius')
        root = math.isqrt(product_value)
        self.budget.integer(root * root, stage='ideal_radius')
        if root * root < product_value:
            root += 1
        self.budget.integer(root, stage='ideal_radius')
        denominator = value.denominator << extra_bits
        self.budget.integer(denominator, stage='ideal_radius')
        return a.check(Fraction(root, denominator))

    def _radius(self, outer_radius: Fraction, delta: Fraction) -> Fraction:
        a = self._arithmetic
        discriminant = a.add(a.mul(outer_radius, outer_radius), a.mul(abs(delta), 4))
        upper_root = self._sqrt_upper(discriminant)
        return a.div(a.add(outer_radius, upper_root), 2)

    def _candidate_plans(self, source: int, outer: _Outer):
        a = self._arithmetic
        plans = []
        total = 0
        for owner in range(len(self.sites)):
            delta = a.sub(self.weights[source], self.weights[owner])
            radius = self._radius(outer.radius, delta)
            displacement = a.difference(self.sites[owner], self.sites[source])
            intervals = []
            for axis in range(3):
                if not self.periodic[axis]:
                    lower = upper = 0
                elif self._basis.orthogonal:
                    length = self.lattice[axis][axis]
                    endpoints = (a.div(a.sub(-radius, displacement[axis]), length),
                                 a.div(a.sub(radius, displacement[axis]), length))
                    lower, upper = math.ceil(min(endpoints)), math.floor(max(endpoints))
                else:
                    inverse_column = tuple(row[axis] for row in self._basis.inverse)
                    center = -a.dot(displacement, inverse_column)
                    bound = a.mul(radius, self._basis.inverse_column_l1[axis])
                    lower = math.ceil(a.sub(center, bound))
                    upper = math.floor(a.add(center, bound))
                self.budget.integer(lower, stage='ideal_coefficients')
                self.budget.integer(upper, stage='ideal_coefficients')
                intervals.append((lower, upper))
            count = 1
            for lower, upper in intervals:
                width = max(0, upper - lower + 1)
                self.budget.integer(width, stage='ideal_candidate_count')
                count *= width
                self.budget.integer(count, stage='ideal_candidate_count')
            total += count
            self.budget.integer(total, stage='ideal_candidate_count')
            plans.append((owner, tuple(intervals)))
        # The entire cell's region is known before the first image is visited.
        self.budget.candidates(total, stage='ideal_candidate_region')
        return plans

    def _candidate_cuts(self, source: int, outer: _Outer, plans):
        a = self._arithmetic
        result = []
        seed_labels = {cut.label for cut in outer.cuts}
        for owner, intervals in plans:
            delta = a.sub(self.weights[source], self.weights[owner])
            ranges = (range(lower, upper + 1) for lower, upper in intervals)
            for reduced in product(*ranges):
                self.budget.charge(stage='ideal_candidate')
                if owner == source and not any(reduced):
                    continue
                normal = self._displacement(
                    source, owner, reduced, self._basis.fractions,
                )
                offset = a.add(a.dot(normal, normal), delta)
                if offset > outer.support(normal, a):
                    continue
                shift = self._map_shift(reduced)
                label = (owner, shift)
                if label not in seed_labels:
                    result.append(_Cut(label, normal, offset))
        return result

    def _contact(self, vertices: tuple[ExactPoint, ...], cut: _Cut) -> ExactContact:
        if not vertices or not any(cut.normal):
            return _ABSENT
        a = self._arithmetic
        points = []
        for point in vertices:
            value = a.dot(cut.normal, point)
            if value > cut.offset:
                raise WP5Failure('WP5_SOURCE_PROFILE_MISMATCH',
                                 'reconstructed ideal cell violates an ideal image cut',
                                 stage='ideal_contact', label=cut.label)
            if value == cut.offset:
                points.append(point)
        dimension = a.affine_dimension(points)
        if dimension < 0:
            return _ABSENT
        if dimension < 2:
            return ExactContact('zero', dimension, tuple(points), Fraction())
        drop = next(k for k, value in enumerate(cut.normal) if value)
        axes = tuple(k for k in range(3) if k != drop)
        ordered = sorted(points, key=lambda point: tuple(point[k] for k in axes))

        def turn(origin, left, right):
            u, v = (a.difference(point, origin) for point in (left, right))
            return a.sub(a.mul(u[axes[0]], v[axes[1]]),
                         a.mul(u[axes[1]], v[axes[0]]))

        halves = []
        for sequence in (ordered, reversed(ordered)):
            half = []
            for point in sequence:
                while len(half) >= 2 and turn(half[-2], half[-1], point) <= 0:
                    half.pop()
                half.append(point)
            halves.append(half[:-1])
        polygon = tuple(halves[0] + halves[1])
        area_vector = _ZERO
        for left, right in zip(polygon[1:-1], polygon[2:]):
            cross = a.cross(a.difference(left, polygon[0]),
                            a.difference(right, polygon[0]))
            area_vector = tuple(a.add(x, y) for x, y in zip(area_vector, cross))
        # Sum of crosses is twice area in Y; physical x=site+Y/2 scales area by 1/4.
        area_squared = a.div(a.dot(area_vector, area_vector), 64)
        return ExactContact('positive', 2, polygon, area_squared)

    def cell(self, source: int) -> ExactCell:
        """Reconstruct once, publishing a cell only after all exact work succeeds."""

        source = self._index(source)
        if source in self._cells:
            return self._cells[source]
        outer = self._outer(source)
        plans = self._candidate_plans(source, outer)
        cuts = self._candidate_cuts(source, outer, plans)
        polytope = _Polytope(outer.cuts, dict(outer.vertices), self._arithmetic)
        for cut in cuts:
            polytope.clip(cut)
        vertices = tuple(sorted(polytope.vertices))
        dimension = self._arithmetic.affine_dimension(vertices)
        contacts = {cut.label: self._contact(vertices, cut) for cut in polytope.cuts}
        facets = MappingProxyType({label: contact for label, contact in contacts.items()
                                   if contact.status == 'positive'})
        result = ExactCell(source, dimension, vertices, facets, self, contacts)
        self._cells[source] = result
        return result
