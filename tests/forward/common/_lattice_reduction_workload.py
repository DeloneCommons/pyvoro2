"""Independent WP3 workload formulas and frozen qualification fixtures.

This module is test-only.  It intentionally reconstructs source binary64
values with :class:`fractions.Fraction` and does not import production exact
lattice, reduction, periodic-image, or duplicate-scanning helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from typing import Iterable, Sequence


ExactRow = tuple[Fraction, Fraction, Fraction]
ExactMatrix = tuple[ExactRow, ExactRow, ExactRow]
IntMatrix = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class WorkloadFixture:
    name: str
    basis: tuple[tuple[float, float, float], ...]
    pi: tuple[float, float, float]
    pj: tuple[float, float, float]
    cohort: str


@dataclass(frozen=True, slots=True)
class ProofWorkload:
    image_search: int
    exponent: int
    interval_widths: tuple[int, int, int]
    box_count: int
    inverse_column_l1: tuple[Fraction, Fraction, Fraction]
    inverse_bound_product: Fraction
    bucket_bins: tuple[int, int, int]
    bucket_bin_product: int
    seed_count: int
    incumbent_squared: Fraction
    fixed_interval_widths: tuple[int, int, int]
    fixed_box_count: int
    gram_diagonal: tuple[Fraction, Fraction, Fraction]
    gram_off_diagonal_l1: Fraction
    hadamard_defect_squared: Fraction


def exact_matrix(values: Sequence[Sequence[float]]) -> ExactMatrix:
    """Interpret one binary64 matrix directly, independently of production."""

    return tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in values
    )  # type: ignore[return-value]


def determinant(rows: ExactMatrix) -> Fraction:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse(rows: ExactMatrix) -> ExactMatrix:
    """Return the direct cofactor inverse, without production helpers."""

    det = determinant(rows)
    if det == 0:
        raise ValueError('fixture basis is exactly singular')
    a, b, c = rows
    adjugate = (
        (
            b[1] * c[2] - b[2] * c[1],
            a[2] * c[1] - a[1] * c[2],
            a[1] * b[2] - a[2] * b[1],
        ),
        (
            b[2] * c[0] - b[0] * c[2],
            a[0] * c[2] - a[2] * c[0],
            a[2] * b[0] - a[0] * b[2],
        ),
        (
            b[0] * c[1] - b[1] * c[0],
            a[1] * c[0] - a[0] * c[1],
            a[0] * b[1] - a[1] * b[0],
        ),
    )
    return tuple(
        tuple(value / det for value in row) for row in adjugate
    )  # type: ignore[return-value]


def _dyadic_exponent(values: Iterable[Fraction]) -> int:
    return max(
        (value.denominator.bit_length() - 1 for value in values),
        default=0,
    )


def common_alignment_exponent(
    *matrices_or_rows: Sequence[Sequence[float]] | Sequence[float],
) -> int:
    values: list[Fraction] = []
    for item in matrices_or_rows:
        for row in item:
            if isinstance(row, (float, int)):
                values.append(Fraction.from_float(float(row)))
            else:
                values.extend(Fraction.from_float(float(value)) for value in row)
    return _dyadic_exponent(values)


def _aligned_integer(value: Fraction, exponent: int) -> int:
    value_exponent = value.denominator.bit_length() - 1
    if value.denominator != 1 << value_exponent:
        raise ValueError('workload alignment requires exact dyadics')
    return value.numerator << (exponent - value_exponent)


def _nearest_ties_lower(value: Fraction) -> int:
    lower = math.floor(value)
    upper = math.ceil(value)
    return lower if value - lower <= upper - value else upper


def _candidate_distance(
    displacement: tuple[int, int, int],
    lattice: tuple[tuple[int, int, int], ...],
    shift: tuple[int, int, int],
) -> int:
    return sum(
        (
            displacement[column]
            + sum(shift[row] * lattice[row][column] for row in range(3))
        ) ** 2
        for column in range(3)
    )


def _upper_norm_from_squared(value: Fraction, exponent: int) -> Fraction:
    scaled = value * (1 << (2 * exponent))
    scaled_ceiling = math.ceil(scaled)
    root = math.isqrt(scaled_ceiling)
    if root * root < scaled_ceiling:
        root += 1
    return Fraction(root, 1 << exponent)


def _intervals(
    q: tuple[Fraction, Fraction, Fraction],
    inverse_l1: tuple[Fraction, Fraction, Fraction],
    upper_norm: Fraction,
) -> tuple[tuple[int, int, int], int]:
    lower = tuple(
        math.ceil(-q[axis] - upper_norm * inverse_l1[axis])
        for axis in range(3)
    )
    upper = tuple(
        math.floor(-q[axis] + upper_norm * inverse_l1[axis])
        for axis in range(3)
    )
    widths = tuple(upper[index] - lower[index] + 1 for index in range(3))
    return widths, math.prod(widths)  # type: ignore[return-value]


def evaluate_proof_workload(
    basis_values: Sequence[Sequence[float | Fraction]],
    *,
    pi: Sequence[float],
    pj: Sequence[float],
    image_search: int,
    common_exponent: int | None = None,
    fixed_incumbent_squared: Fraction | None = None,
    bucket_radius: float = 1e-5,
) -> ProofWorkload:
    """Evaluate the current proof-box and R5 bucket formulas exactly."""

    basis = tuple(
        tuple(
            value if isinstance(value, Fraction)
            else Fraction.from_float(float(value))
            for value in row
        )
        for row in basis_values
    )
    basis = basis  # type: ignore[assignment]
    basis_inverse = inverse(basis)  # type: ignore[arg-type]
    delta = tuple(
        Fraction.from_float(float(pj[index]))
        - Fraction.from_float(float(pi[index]))
        for index in range(3)
    )
    exponent = common_exponent
    if exponent is None:
        exponent = _dyadic_exponent(
            value for row in (*basis, (delta,)) for value in row
        )
    displacement = tuple(_aligned_integer(value, exponent) for value in delta)
    lattice = tuple(
        tuple(_aligned_integer(value, exponent) for value in row)
        for row in basis
    )
    q = tuple(
        sum(delta[row] * basis_inverse[row][column] for row in range(3))
        for column in range(3)
    )
    center = tuple(_nearest_ties_lower(-value) for value in q)
    offsets = product(
        range(-image_search, image_search + 1), repeat=3
    )
    seeds = tuple(
        tuple(center[index] + offset[index] for index in range(3))
        for offset in offsets
    )
    incumbent_numerator = min(
        _candidate_distance(displacement, lattice, shift) for shift in seeds
    )
    incumbent_squared = Fraction(
        incumbent_numerator, 1 << (2 * exponent)
    )
    upper_norm = _upper_norm_from_squared(incumbent_squared, exponent)
    inverse_l1 = tuple(
        sum((abs(basis_inverse[row][column]) for row in range(3)), Fraction())
        for column in range(3)
    )
    widths, count = _intervals(q, inverse_l1, upper_norm)

    if fixed_incumbent_squared is None:
        fixed_incumbent_squared = incumbent_squared
    fixed_upper_norm = _upper_norm_from_squared(
        fixed_incumbent_squared, exponent
    )
    fixed_widths, fixed_count = _intervals(q, inverse_l1, fixed_upper_norm)

    radius = Fraction.from_float(float(bucket_radius))
    coefficient_bounds = tuple(radius * value for value in inverse_l1)
    bins = tuple(
        1 if bound >= 1 else bound.denominator // bound.numerator
        for bound in coefficient_bounds
    )
    gram = tuple(
        tuple(sum(left * right for left, right in zip(row, other))
              for other in basis)
        for row in basis
    )
    diagonal = tuple(gram[index][index] for index in range(3))
    off_l1 = sum(
        (abs(gram[row][column]) for row in range(3)
         for column in range(row)),
        Fraction(),
    )
    det = determinant(basis)  # type: ignore[arg-type]
    hadamard = math.prod(diagonal) / (det * det)
    return ProofWorkload(
        image_search=image_search,
        exponent=exponent,
        interval_widths=widths,
        box_count=count,
        inverse_column_l1=inverse_l1,
        inverse_bound_product=math.prod(inverse_l1),
        bucket_bins=bins,
        bucket_bin_product=math.prod(bins),
        seed_count=len(seeds),
        incumbent_squared=incumbent_squared,
        fixed_interval_widths=fixed_widths,
        fixed_box_count=fixed_count,
        gram_diagonal=diagonal,
        gram_off_diagonal_l1=off_l1,
        hadamard_defect_squared=hadamard,
    )


def _hex(value: str) -> float:
    return float.fromhex(value)


def seeded_unimodular_pair(seed: int) -> tuple[IntMatrix, IntMatrix]:
    """Return a deterministic elementary-row composition and known inverse."""

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    state = (int(seed) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

    def next_value() -> int:
        nonlocal state
        state ^= state >> 12
        state ^= (state << 25) & ((1 << 64) - 1)
        state ^= state >> 27
        state &= (1 << 64) - 1
        return (state * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

    transform = [list(row) for row in identity]
    inverse_transform = [list(row) for row in identity]
    for _ in range(18):
        operation = next_value() % 3
        first = next_value() % 3
        second = next_value() % 2
        if second >= first:
            second += 1
        if operation == 0:
            transform[first], transform[second] = (
                transform[second], transform[first]
            )
            for row in inverse_transform:
                row[first], row[second] = row[second], row[first]
        elif operation == 1:
            transform[first] = [-value for value in transform[first]]
            for row in inverse_transform:
                row[first] = -row[first]
        else:
            multiplier = (-7, -5, -3, 3, 5, 7)[next_value() % 6]
            transform[first] = [
                transform[first][column]
                + multiplier * transform[second][column]
                for column in range(3)
            ]
            for row in inverse_transform:
                row[second] -= multiplier * row[first]
    return (
        tuple(tuple(row) for row in transform),  # type: ignore[return-value]
        tuple(tuple(row) for row in inverse_transform),  # type: ignore[return-value]
    )


FROZEN_WORKLOAD_FIXTURES = (
    WorkloadFixture(
        name='thin-3e-4',
        basis=((1.0, 0.0, 0.0), (1.0, 0.0003, 0.0), (0.0, 0.0, 1.0)),
        pi=(0.0, 0.0, 0.0),
        pj=(
            _hex('0x1.f5c28f5c28f5cp-1'),
            _hex('0x1.344806290eed0p-13'),
            _hex('0x1.f5c28f5c28f5cp-2'),
        ),
        cohort='repository-regression',
    ),
    WorkloadFixture(
        name='thin-1e-3',
        basis=((1.0, 0.0, 0.0), (1.0, 0.001, 0.0), (0.0, 0.0, 1.0)),
        pi=(0.0, 0.0, 0.0),
        pj=(
            _hex('0x1.f5c28f5c28f5cp-1'),
            _hex('0x1.00e6afcce1c58p-11'),
            _hex('0x1.f5c28f5c28f5cp-2'),
        ),
        cohort='repository-regression',
    ),
    *(
        WorkloadFixture(
            name=f'cubic-shear-2p{power}',
            basis=(
                (1.0, 0.0, 0.0),
                (float(2**power), 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            pi=(0.0, 0.0, 0.0),
            pj=(0.25, -0.375, 0.125),
            cohort='exact-large-shear',
        )
        for power in (8, 32, 53, 80)
    ),
    WorkloadFixture(
        name='equivalent-composed-a',
        basis=((1.0, 0.0, 0.0), (32.0, 1.0, 0.0), (-21.0, 17.0, 1.0)),
        pi=(0.125, -0.25, 0.375),
        pj=(0.5, 0.125, -0.25),
        cohort='equivalent-well-conditioned',
    ),
    WorkloadFixture(
        name='equivalent-composed-b',
        basis=((0.0, 1.0, 0.0), (1.0, 48.0, 0.0), (13.0, -29.0, -1.0)),
        pi=(-0.25, 0.5, 0.125),
        pj=(0.375, -0.125, 0.5),
        cohort='equivalent-well-conditioned',
    ),
    WorkloadFixture(
        name='equivalent-composed-c',
        basis=((1.0, 0.0, 0.0), (-64.0, -1.0, 0.0), (37.0, 11.0, 1.0)),
        pi=(0.0, 0.0, 0.0),
        pj=(-0.375, 0.25, 0.125),
        cohort='equivalent-well-conditioned',
    ),
    WorkloadFixture(
        name='intrinsic-anisotropy',
        basis=((1.0, 0.0, 0.0), (16.0, 1.0, 0.0), (0.0, 0.0, 2.0**-16)),
        pi=(0.0, 0.0, 0.0),
        pj=(0.25, 0.25, 2.0**-18),
        cohort='intrinsically-anisotropic',
    ),
    WorkloadFixture(
        name='r5-sc-001',
        basis=(
            (0.05185516747129633, 0.0, 0.0),
            (1721.8706815591584, 2.1100027038765874, 0.0),
            (4310.293148429306, -3510.8020392421427, 5986.1025926131315),
        ),
        pi=(0.0, 0.0, 0.0),
        pj=(0.0125, -0.025, 0.0375),
        cohort='bucket-regression',
    ),
)


FROZEN_RANDOM_UNIMODULAR_FIXTURES = tuple(
    WorkloadFixture(
        name=f'random-unimodular-seed-{seed}',
        basis=tuple(
            tuple(float(value) for value in row)
            for row in seeded_unimodular_pair(seed)[0]
        ),
        pi=pi,
        pj=pj,
        cohort='seeded-random-unimodular',
    )
    for seed, pi, pj in (
        (0, (0.125, -0.25, 0.375), (0.5, 0.125, -0.25)),
        (1, (-0.25, 0.5, 0.125), (0.375, -0.125, 0.5)),
        (56, (0.0, 0.0, 0.0), (-0.375, 0.25, 0.125)),
    )
)
