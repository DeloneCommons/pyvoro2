"""Certify lattice translations against an explicit exact Cartesian box.

This private consumer supplies no numerical error model for native producers.
A producer must choose its semantic reference and justify its own finite box.
For a reduced row basis B with exact inverse V, every compatible t = s @ B
satisfies s = t @ V. Extrema of each linear form over the Cartesian box give
a complete finite integer coefficient box. Every admitted candidate is then
filtered against the original Cartesian box; outer-box membership alone is
not compatibility. Diagonal partial-periodic geometry uses interval division.

Resource guards observe semantic integers and normalized rational operands and
results, including factors and partial sums before cancellation. They do not
model temporary integers internal to Python's Fraction normalization. Native
limits are separate from reduction, minimum-image, and duplicate-scan limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Sequence

import numpy as np

from .exact_lattice import (
    DEFAULT_REDUCTION_LIMITS,
    ExactLatticeReductionDiagnostics,
    ExactLatticeReductionInvariantError,
    ExactLatticeReductionLimits,
    ExactLatticeReductionResourceError,
)
from .inputs import coerce_finite_matrix, coerce_finite_vector
from .periodic_images import _BasisData, _basis_key, _prepare_basis
from .validation import require_bool_tuple, require_index, require_positive_index


_MAX_NATIVE_TRANSLATION_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class NativeTranslationLimits:
    """Private recovery resources; the million-candidate ceiling may be lowered."""

    max_candidates: int = _MAX_NATIVE_TRANSLATION_CANDIDATES
    max_integer_bits: int = 32_768
    max_rational_bits: int = 65_536
    max_witnesses: int = 4

    def __post_init__(self) -> None:
        for name in (
            'max_candidates', 'max_integer_bits', 'max_rational_bits',
            'max_witnesses',
        ):
            value = require_positive_index(
                getattr(self, name), name=name,
                maximum=(_MAX_NATIVE_TRANSLATION_CANDIDATES
                         if name == 'max_candidates' else None),
            )
            object.__setattr__(self, name, value)


DEFAULT_NATIVE_TRANSLATION_LIMITS = NativeTranslationLimits()


def _exact_endpoint_row(values: object, *, name: str) -> tuple[Fraction, ...]:
    try:
        items = tuple(values)
    except TypeError:
        raise ValueError(f'{name} must be an exact rational coordinate row') from None
    result = []
    for item in items:
        if isinstance(item, Fraction):
            result.append(item)
        else:
            result.append(Fraction(require_index(item, name=name)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CartesianCompatibilityBox:
    """Finite closed Cartesian box with exact Fraction/integer endpoints.

    Binary64 coordinates belong in ``translation_box_from_observation`` or an
    explicit caller-side exactification, so no hidden rounding defines a box.
    """

    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        lower = _exact_endpoint_row(self.lower, name='lower')
        upper = _exact_endpoint_row(self.upper, name='upper')
        if len(lower) not in (2, 3) or len(upper) != len(lower):
            raise ValueError('a compatibility box must have dimension two or three')
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError('compatibility box lower endpoints must not exceed upper')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)


@dataclass(frozen=True, slots=True)
class NativeTranslationWitnessResource:
    """One optional witness-mapping limit after ambiguity is already proved."""

    stage: str
    resource: str
    observed: int
    configured_limit: int


@dataclass(frozen=True, slots=True)
class NativeTranslationDiagnostics:
    """Exact proof workload and bounded user-basis witnesses for one call."""

    reason: str
    stage: str
    basis_summary: dict[str, object] | None
    envelope_summary: CartesianCompatibilityBox
    coefficient_widths: tuple[int, ...] | None
    candidate_bound: int | None
    examined_count: int
    compatible_count: int | None
    limits: NativeTranslationLimits
    reduction_limits: ExactLatticeReductionLimits | None
    witnesses: tuple[tuple[int, ...], ...]
    witness_resource: NativeTranslationWitnessResource | None
    max_integer_bits: int
    max_rational_bits: int
    max_reduced_coefficient_bits: int
    max_mapped_coefficient_bits: int
    reduction_diagnostics: ExactLatticeReductionDiagnostics | None


@dataclass(frozen=True, slots=True)
class NativeTranslationResult:
    """One uniquely certified user-basis shift, without fixed-width coercion."""

    shift: tuple[int, ...]
    translation: tuple[Fraction, ...]
    diagnostics: NativeTranslationDiagnostics
    certified: bool = True


class NativeTranslationCertificationError(RuntimeError):
    """Base for structured native compatibility certification failures."""

    def __init__(
        self, message: str, diagnostics: NativeTranslationDiagnostics,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.reason = diagnostics.reason
        self.stage = diagnostics.stage


class NativeTranslationInconsistencyError(NativeTranslationCertificationError):
    """No exact lattice translation is compatible with the declared box."""


class NativeTranslationAmbiguityError(NativeTranslationCertificationError):
    """Several translations are compatible; no residual-based winner exists."""


class NativeTranslationInvariantError(NativeTranslationCertificationError):
    """An exact proof/reduction invariant failed, independently of resources."""


class NativeTranslationResourceError(NativeTranslationCertificationError):
    """A structural limit prevented certification or required shift mapping."""

    def __init__(
        self, message: str, diagnostics: NativeTranslationDiagnostics, *,
        resource: str, observed: int, configured_limit: int,
    ) -> None:
        super().__init__(message, diagnostics)
        self.resource = resource
        self.observed = observed
        self.configured_limit = configured_limit


@dataclass(slots=True)
class _RecoveryState:
    box: CartesianCompatibilityBox
    limits: NativeTranslationLimits
    reduction_limits: ExactLatticeReductionLimits | None = None
    stage: str = 'envelope'
    basis_summary: dict[str, object] | None = None
    coefficient_widths: tuple[int, ...] | None = None
    candidate_bound: int | None = None
    examined_count: int = 0
    compatible_count: int | None = None
    witnesses: list[tuple[int, ...]] = field(default_factory=list)
    witness_resource: NativeTranslationWitnessResource | None = None
    max_integer_bits: int = 0
    max_rational_bits: int = 0
    max_reduced_coefficient_bits: int = 0
    max_mapped_coefficient_bits: int = 0
    reduction_diagnostics: ExactLatticeReductionDiagnostics | None = None

    def diagnostics(self, reason: str) -> NativeTranslationDiagnostics:
        return NativeTranslationDiagnostics(
            reason=reason, stage=self.stage,
            basis_summary=(None if self.basis_summary is None
                           else dict(self.basis_summary)),
            envelope_summary=self.box,
            coefficient_widths=self.coefficient_widths,
            candidate_bound=self.candidate_bound,
            examined_count=self.examined_count,
            compatible_count=self.compatible_count,
            limits=self.limits, reduction_limits=self.reduction_limits,
            witnesses=tuple(self.witnesses),
            witness_resource=self.witness_resource,
            max_integer_bits=self.max_integer_bits,
            max_rational_bits=self.max_rational_bits,
            max_reduced_coefficient_bits=self.max_reduced_coefficient_bits,
            max_mapped_coefficient_bits=self.max_mapped_coefficient_bits,
            reduction_diagnostics=self.reduction_diagnostics,
        )

    def resource(self, resource: str, observed: int, limit: int) -> None:
        raise NativeTranslationResourceError(
            f'native translation certification exceeded {resource}',
            self.diagnostics('resource-limit'), resource=resource,
            observed=observed, configured_limit=limit,
        )

    def integer(self, value: int) -> int:
        bits = abs(value).bit_length()
        self.max_integer_bits = max(self.max_integer_bits, bits)
        if bits > self.limits.max_integer_bits:
            self.resource('integer_bits', bits, self.limits.max_integer_bits)
        return value

    def rational(self, value: Fraction) -> Fraction:
        bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        self.max_rational_bits = max(self.max_rational_bits, bits)
        if bits > self.limits.max_rational_bits:
            self.resource('rational_bits', bits, self.limits.max_rational_bits)
        return value

    def check_box(self) -> None:
        for value in self.box.lower + self.box.upper:
            self.rational(value)


def _require_limits(limits: NativeTranslationLimits) -> None:
    if not isinstance(limits, NativeTranslationLimits):
        raise ValueError('limits must be NativeTranslationLimits')


def translation_box_from_observation(
    observed: Sequence[float], reference: Sequence[float],
    error_box: CartesianCompatibilityBox, *,
    limits: NativeTranslationLimits = DEFAULT_NATIVE_TRANSLATION_LIMITS,
) -> CartesianCompatibilityBox:
    """Construct exact t bounds for ``reference + t - observed in error_box``.

    The caller explicitly chooses the reference and signed error box. Each
    binary64 operand is exactified separately, before subtraction. No default
    tolerance or native-producer arithmetic guarantee is supplied here.
    """

    _require_limits(limits)
    if not isinstance(error_box, CartesianCompatibilityBox):
        raise ValueError('error_box must be CartesianCompatibilityBox')
    dimension = len(error_box.lower)
    observed_values = coerce_finite_vector(observed, name='observed', n=dimension)
    reference_values = coerce_finite_vector(reference, name='reference', n=dimension)
    state = _RecoveryState(error_box, limits, stage='envelope-construction')
    state.check_box()
    lower, upper = [], []
    for index in range(dimension):
        observed_exact = state.rational(
            Fraction.from_float(float(observed_values[index])),
        )
        reference_exact = state.rational(
            Fraction.from_float(float(reference_values[index])),
        )
        difference = state.rational(observed_exact - reference_exact)
        lower.append(state.rational(difference + error_box.lower[index]))
        upper.append(state.rational(difference + error_box.upper[index]))
    return CartesianCompatibilityBox(tuple(lower), tuple(upper))


def _coefficient_intervals(
    basis: _BasisData, state: _RecoveryState,
) -> tuple[tuple[int, int], ...]:
    """Enclose all compatible integer coefficients without search heuristics."""

    state.stage = 'coefficient-bounds'
    intervals = []
    for column in range(basis.dimension):
        if not basis.periodic_axes[column]:
            intervals.append((0, 0))
            continue
        if basis.orthogonal:
            length = state.rational(basis.fractions[column][column])
            left = state.rational(state.box.lower[column] / length)
            right = state.rational(state.box.upper[column] / length)
            low, high = min(left, right), max(left, right)
        else:
            assert basis.inverse is not None
            low, high = Fraction(), Fraction()
            for row in range(basis.dimension):
                inverse = state.rational(basis.inverse[row][column])
                left = state.rational(state.box.lower[row] * inverse)
                right = state.rational(state.box.upper[row] * inverse)
                low = state.rational(low + min(left, right))
                high = state.rational(high + max(left, right))
        intervals.append((
            state.integer(math.ceil(low)), state.integer(math.floor(high)),
        ))
    return tuple(intervals)


def _exact_translation(
    coefficients: tuple[int, ...], basis: _BasisData, state: _RecoveryState,
) -> tuple[Fraction, ...]:
    values = []
    for column in range(basis.dimension):
        value = Fraction()
        for row in range(basis.dimension):
            coefficient = state.rational(Fraction(coefficients[row]))
            entry = state.rational(basis.fractions[row][column])
            term = state.rational(coefficient * entry)
            value = state.rational(value + term)
        values.append(value)
    return tuple(values)


def _map_user_shift(
    coefficients: tuple[int, ...], basis: _BasisData, state: _RecoveryState,
) -> tuple[int, ...]:
    state.stage = 'shift-mapping'
    if basis.reduction is None:
        mapped = coefficients
    else:
        # The accepted WP3 mapper owns the row convention. Observe every
        # semantic multiplication and partial sum before invoking that mapper;
        # observing only the final tuple would miss cancellation growth.
        transform = basis.reduction.transform
        for column in range(basis.dimension):
            total = 0
            for row in range(basis.dimension):
                left = state.integer(coefficients[row])
                right = state.integer(transform[row][column])
                term = state.integer(left * right)
                total = state.integer(total + term)
        mapped = basis.reduction.map_reduced_to_user(coefficients)
    for value in mapped:
        state.integer(value)
        state.max_mapped_coefficient_bits = max(
            state.max_mapped_coefficient_bits, abs(value).bit_length(),
        )
    return mapped


def certify_native_translation(
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    compatibility_box: CartesianCompatibilityBox, *,
    periodic_axes: Sequence[bool] | None = None,
    limits: NativeTranslationLimits = DEFAULT_NATIVE_TRANSLATION_LIMITS,
    reduction_limits: ExactLatticeReductionLimits = DEFAULT_REDUCTION_LIMITS,
) -> NativeTranslationResult:
    """Certify the unique user-basis integer shift whose translation is in C.

    Zero solutions raise inconsistency; several raise ambiguity with an exact
    total and bounded witnesses. Candidate/bit limits raise resources, never a
    partial answer. After completed ambiguity certification, optional witness
    mapping limits are diagnostic and do not change the proved outcome.
    Supported geometries are Cartesian-diagonal 2D/3D with any periodic axes
    and exactly nonsingular fully periodic 3D row lattices.
    """

    _require_limits(limits)
    if not isinstance(reduction_limits, ExactLatticeReductionLimits):
        raise ValueError('reduction_limits must be ExactLatticeReductionLimits')
    if not isinstance(compatibility_box, CartesianCompatibilityBox):
        raise ValueError('compatibility_box must be CartesianCompatibilityBox')
    dimension = len(compatibility_box.lower)
    lattice = coerce_finite_matrix(
        lattice_vectors, name='lattice_vectors', shape=(dimension, dimension),
    )
    axes = ((True,) * dimension if periodic_axes is None else require_bool_tuple(
        periodic_axes, name='periodic_axes', length=dimension,
    ))
    key = _basis_key(lattice, axes)
    state = _RecoveryState(compatibility_box, limits, reduction_limits)
    state.basis_summary = {
        'dimension': dimension, 'periodic_axes': axes,
        'lattice_bits': tuple(f'{value:016x}' for value in key[2]),
    }
    state.check_box()
    state.stage = 'basis-reduction'
    try:
        basis = _prepare_basis(*key, reduction_limits=reduction_limits)
    except ExactLatticeReductionResourceError as error:
        raise NativeTranslationResourceError(
            'native translation proof basis exceeded reduction resources',
            state.diagnostics('reduction-resource'), resource=error.resource,
            observed=error.observed, configured_limit=error.configured_limit,
        ) from error
    except ExactLatticeReductionInvariantError as error:
        raise NativeTranslationInvariantError(
            'native translation proof basis failed its exact certificate',
            state.diagnostics('reduction-invariant'),
        ) from error
    state.basis_summary = basis.summary
    if basis.reduction is not None:
        state.reduction_diagnostics = basis.reduction.diagnostics
    state.stage = 'basis-operands'
    for row in basis.fractions:
        for value in row:
            state.rational(value)
    if basis.orthogonal and any(
        not axes[index]
        and not compatibility_box.lower[index] <= 0 <= compatibility_box.upper[index]
        for index in range(dimension)
    ):
        state.compatible_count = 0
        raise NativeTranslationInconsistencyError(
            'a nonperiodic Cartesian translation component cannot be nonzero',
            state.diagnostics('nonperiodic-incompatibility'),
        )

    intervals = _coefficient_intervals(basis, state)
    widths = []
    for low, high in intervals:
        difference = state.integer(high - low)
        widths.append(max(0, state.integer(difference + 1)))
    state.coefficient_widths = tuple(widths)
    state.stage = 'candidate-preflight'
    bound = 1
    for width in widths:
        bound = state.integer(bound * width)
    state.candidate_bound = bound
    if bound > limits.max_candidates:
        state.resource('candidates', bound, limits.max_candidates)

    state.stage = 'cartesian-filter'
    compatible_count = 0
    reduced_witnesses = []
    unique_translation = None
    if bound:
        ranges = tuple(range(low, state.integer(high + 1)) for low, high in intervals)
        # Nested generators keep only the current coefficient tuple. In
        # contrast, itertools.product pools every integer in each range,
        # multiplying potentially large coefficient storage by interval width.
        if dimension == 2:
            coefficient_rows = ((a, b) for a in ranges[0] for b in ranges[1])
        else:
            coefficient_rows = (
                (a, b, c) for a in ranges[0] for b in ranges[1] for c in ranges[2]
            )
        for coefficients in coefficient_rows:
            for coefficient in coefficients:
                state.integer(coefficient)
                state.max_reduced_coefficient_bits = max(
                    state.max_reduced_coefficient_bits, abs(coefficient).bit_length(),
                )
            translation = _exact_translation(coefficients, basis, state)
            state.examined_count = state.integer(state.examined_count + 1)
            if all(lo <= value <= hi for lo, value, hi in zip(
                compatibility_box.lower, translation, compatibility_box.upper,
            )):
                compatible_count = state.integer(compatible_count + 1)
                if compatible_count == 1:
                    unique_translation = translation
                if len(reduced_witnesses) < limits.max_witnesses:
                    reduced_witnesses.append(coefficients)
    state.compatible_count = compatible_count
    for coefficients in reduced_witnesses:
        try:
            mapped = _map_user_shift(coefficients, basis, state)
        except NativeTranslationResourceError as error:
            if compatible_count == 1:
                raise
            # Enumeration already proved ambiguity. Optional examples cannot
            # turn that exact outcome into a resource failure. Keep earlier
            # examples and one bounded explanation for stopping their mapping.
            state.witness_resource = NativeTranslationWitnessResource(
                stage=error.stage, resource=error.resource,
                observed=error.observed, configured_limit=error.configured_limit,
            )
            break
        state.witnesses.append(mapped)
    state.stage = 'complete'
    if not compatible_count:
        raise NativeTranslationInconsistencyError(
            'no lattice translation is compatible with the declared Cartesian box',
            state.diagnostics('no-compatible-translation'),
        )
    if compatible_count > 1:
        raise NativeTranslationAmbiguityError(
            'multiple lattice translations are compatible with the Cartesian box',
            state.diagnostics('multiple-compatible-translations'),
        )
    assert unique_translation is not None
    return NativeTranslationResult(
        shift=state.witnesses[0], translation=unique_translation,
        diagnostics=state.diagnostics('unique-compatible-translation'),
    )
