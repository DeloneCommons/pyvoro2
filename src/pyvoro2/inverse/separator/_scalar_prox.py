"""Certified private scalar proximal solver for separator ADMM rows."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal, DecimalException, InvalidOperation
from fractions import Fraction

import numpy as np

from ._numerics import (
    _dd_add,
    _DoubleDouble,
    _fraction,
    _fraction_float_neighbors,
    _float_to_ordered_int,
    _ordered_float_midpoint,
    _ordered_floats_adjacent,
    _scaled_physical_float,
    _ScaledEnclosure,
    _stable_ratio_product_scalar,
    _stable_sum_products_sign,
    _stable_weighted_average,
)
from ._objective import (
    _array_scalar_derivative_terms,
    _array_scalar_objective_difference,
    _compile_scalar_objective,
    _CompiledScalarObjective,
    _exact_expression_interval,
    _KernelDerivativeEnclosures,
    _scalar_derivative_exact_expression,
    _scalar_derivative_terms,
    _scalar_objective_difference,
    _scalar_objective_difference_exact_expression,
    _scalar_objective_value,
)
from .model import FitModel


_FLOAT_MAX = np.finfo(np.float64).max
_MAX_EXPANSIONS = 128
_MAX_SCALAR_ITERATIONS = 128
_EXPANSION_EXPONENT_STEP = 8
_FALLBACK_PRECISIONS = (80, 160)
_MAX_FALLBACK_DECISIONS = 4


@dataclass(frozen=True, slots=True)
class _ScalarProxSpec:
    """Compiled static mismatch and positive-strength penalty information."""

    objective: _CompiledScalarObjective


@dataclass(frozen=True, slots=True)
class _DerivativeCertificateEvidence:
    """One point's ordinary enclosures and any rigorously resolved signs."""

    y: float
    minus_enclosure: tuple[float, float]
    plus_enclosure: tuple[float, float]
    minus_resolved_sign: int | None
    plus_resolved_sign: int | None
    minus_ordinary_resolved_sign: int | None = None
    plus_ordinary_resolved_sign: int | None = None
    minus_fallback_enclosure: tuple[Decimal, Decimal] | None = None
    plus_fallback_enclosure: tuple[Decimal, Decimal] | None = None
    minus_fallback_resolved_sign: int | None = None
    plus_fallback_resolved_sign: int | None = None
    minus_fallback_exact: bool = False
    plus_fallback_exact: bool = False


@dataclass(frozen=True, slots=True)
class _TerminalDifferenceCertificateEvidence:
    """Reconstructable endpoint-selection evidence for an adjacent bracket."""

    lower_endpoint: float
    upper_endpoint: float
    ordinary_enclosure: tuple[float, float]
    ordinary_resolved_sign: int | None
    fallback_enclosure: tuple[Decimal, Decimal] | None
    fallback_resolved_sign: int | None
    fallback_exact: bool
    selected_endpoint: float
    selection_reason: str


@dataclass(frozen=True, slots=True)
class _ScalarProxResult:
    """A binary64 coordinate together with its private certificate."""

    value: float
    certificate_kind: str
    kkt_residual: float
    scalar_iterations: int
    expansion_count: int
    final_bracket: tuple[float, float]
    objective: float
    point_derivative_evidence: _DerivativeCertificateEvidence | None = None
    adjacent_bracket_evidence: (
        tuple[_DerivativeCertificateEvidence, _DerivativeCertificateEvidence]
        | None
    ) = None
    terminal_difference_evidence: (
        _TerminalDifferenceCertificateEvidence | None
    ) = None
    localization_bound: float | None = None
    fallback_count: int = 0
    execution_path: str = 'scalar_ordinary'
    batch_probe_count: int = 0

    @property
    def n_iter(self) -> int:
        return self.scalar_iterations

    @property
    def n_expansions(self) -> int:
        return self.expansion_count

    @property
    def high_precision_fallback_count(self) -> int:
        return self.fallback_count


@dataclass(frozen=True, slots=True)
class _ScalarProxBatchOutcome:
    """Per-lane ordinary batch certificates and explicit routing metadata."""

    results: tuple[_ScalarProxResult | None, ...]
    eligible: np.ndarray
    certified: np.ndarray
    probe_counts: np.ndarray
    arithmetic_exception: bool = False


@dataclass(frozen=True, slots=True)
class _ScalarProxFailure:
    """Inspectable evidence retained when scalar certification fails."""

    reason: str
    scalar_iterations: int
    expansion_count: int
    last_candidate: float | None
    last_finite_bracket: tuple[float, float] | None
    last_derivative_enclosure: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    localization_bound: float | None = None
    fallback_count: int = 0


class _ScalarProxError(RuntimeError):
    """Carry a structured scalar proximal failure to ADMM integration."""

    def __init__(self, failure: _ScalarProxFailure) -> None:
        self.failure = failure
        super().__init__(
            f'{failure.reason}; scalar_iterations='
            f'{failure.scalar_iterations}; expansions='
            f'{failure.expansion_count}; last_candidate='
            f'{failure.last_candidate!r}; last_finite_bracket='
            f'{failure.last_finite_bracket!r}; last_derivative_enclosure='
            f'{failure.last_derivative_enclosure!r}; localization_bound='
            f'{failure.localization_bound!r}; fallback_count='
            f'{failure.fallback_count}'
        )


@dataclass(frozen=True, slots=True)
class _DerivativeEvaluation:
    """Rigorous ordinary enclosures plus optional fallback-resolved signs."""

    y: float
    minus: _ScaledEnclosure
    plus: _ScaledEnclosure
    curvature_enclosure: _ScaledEnclosure
    smooth: bool
    fallback_minus_sign: int | None = None
    fallback_plus_sign: int | None = None
    fallback_minus_value: float | None = None
    fallback_plus_value: float | None = None
    fallback_minus_interval: tuple[Decimal, Decimal] | None = None
    fallback_plus_interval: tuple[Decimal, Decimal] | None = None
    fallback_minus_exact: bool = False
    fallback_plus_exact: bool = False

    @property
    def g_minus(self) -> float:
        lower, upper = self.minus.physical_bounds()
        return 0.5 * (lower + upper)

    @property
    def g_plus(self) -> float:
        lower, upper = self.plus.physical_bounds()
        return 0.5 * (lower + upper)

    @property
    def curvature(self) -> float:
        lower, upper = self.curvature_enclosure.physical_bounds()
        return 0.5 * (lower + upper)

    @property
    def scale_log(self) -> float:
        return max(self.minus.log_scale, self.plus.log_scale)

    @property
    def widened_minus(self) -> float:
        return self.minus.physical_bounds()[0]

    @property
    def widened_plus(self) -> float:
        return self.plus.physical_bounds()[1]

    @property
    def certified_negative(self) -> bool:
        if self.fallback_plus_sign is not None:
            return self.fallback_plus_sign < 0
        return self.plus.strictly_negative

    @property
    def certified_positive(self) -> bool:
        if self.fallback_minus_sign is not None:
            return self.fallback_minus_sign > 0
        return self.minus.strictly_positive

    def physical_enclosures(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.minus.physical_bounds(), self.plus.physical_bounds()

    def certificate_evidence(self) -> _DerivativeCertificateEvidence:
        minus, plus = self.physical_enclosures()

        def ordinary_sign(enclosure: _ScaledEnclosure) -> int | None:
            if enclosure.strictly_negative:
                return -1
            if enclosure.strictly_positive:
                return 1
            if enclosure.lower == 0.0 and enclosure.upper == 0.0:
                return 0
            return None

        minus_ordinary = ordinary_sign(self.minus)
        plus_ordinary = ordinary_sign(self.plus)

        return _DerivativeCertificateEvidence(
            y=self.y,
            minus_enclosure=minus,
            plus_enclosure=plus,
            minus_resolved_sign=(
                self.fallback_minus_sign
                if self.fallback_minus_sign is not None
                else minus_ordinary
            ),
            plus_resolved_sign=(
                self.fallback_plus_sign
                if self.fallback_plus_sign is not None
                else plus_ordinary
            ),
            minus_ordinary_resolved_sign=minus_ordinary,
            plus_ordinary_resolved_sign=plus_ordinary,
            minus_fallback_enclosure=self.fallback_minus_interval,
            plus_fallback_enclosure=self.fallback_plus_interval,
            minus_fallback_resolved_sign=self.fallback_minus_sign,
            plus_fallback_resolved_sign=self.fallback_plus_sign,
            minus_fallback_exact=self.fallback_minus_exact,
            plus_fallback_exact=self.fallback_plus_exact,
        )


def _compile_scalar_prox_spec(model: FitModel) -> _ScalarProxSpec:
    """Compile one model for repeated coordinate solves."""

    return _ScalarProxSpec(
        objective=_compile_scalar_objective(model.mismatch, model.penalties)
    )


def _physical_float(value: float | Decimal, scale_log: float | Decimal) -> float:
    """Compatibility adapter for scaled private diagnostics."""

    return _scaled_physical_float(float(value), float(scale_log))


def _evaluate_derivative(
    spec: _ScalarProxSpec,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _DerivativeEvaluation:
    terms: _KernelDerivativeEnclosures = _scalar_derivative_terms(
        spec.objective,
        y=float(y),
        target=float(target),
        confidence=float(confidence),
        v=float(v),
        rho=float(rho),
    )
    return _DerivativeEvaluation(
        y=float(y),
        minus=terms.minus,
        plus=terms.plus,
        curvature_enclosure=terms.curvature,
        smooth=terms.smooth,
    )


def _coordinate_breakpoints(
    spec: _ScalarProxSpec,
    *,
    target: float,
    lower: float,
    upper: float,
) -> tuple[Fraction, ...]:
    """Compile coordinate-specific exact dyadic search locations once."""

    points = set(spec.objective.breakpoints)
    if spec.objective.huber_delta is not None:
        target_fraction = _fraction(target)
        points.add(target_fraction - spec.objective.huber_delta)
        points.add(target_fraction + spec.objective.huber_delta)
    if math.isfinite(lower):
        points.add(_fraction(lower))
    if math.isfinite(upper):
        points.add(_fraction(upper))
    return tuple(sorted(points))


def _mismatch_only_candidate(
    spec: _ScalarProxSpec,
    *,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> float:
    # Form the convex weighted average by normalized weights before any raw
    # weighted numerator can overflow at finite binary64 extrema.
    quadratic = float(
        _stable_weighted_average(v, rho, target, confidence)
    )
    if spec.objective.mismatch_kind == 'squared':
        return quadratic
    assert spec.objective.huber_delta_value is not None
    delta = spec.objective.huber_delta_value
    lower_test = float(
        _stable_sum_products_sign(
            (
                (rho, v),
                (-rho, target),
                (rho, delta),
                (confidence, delta),
            )
        )
    )
    upper_test = float(
        _stable_sum_products_sign(
            (
                (rho, v),
                (-rho, target),
                (-rho, delta),
                (-confidence, delta),
            )
        )
    )
    if lower_test < 0.0:
        return _dd_add(
            _DoubleDouble(v),
            _DoubleDouble(
                _stable_ratio_product_scalar(
                    (confidence, delta),
                    (rho,),
                )
            ),
        ).value
    if upper_test > 0.0:
        return _dd_add(
            _DoubleDouble(v),
            _DoubleDouble(
                -_stable_ratio_product_scalar(
                    (confidence, delta),
                    (rho,),
                )
            ),
        ).value
    return quadratic


def _proposal_derivative_curvature(
    spec: _ScalarProxSpec,
    *,
    y: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> tuple[float, float] | None:
    """Return a fast central Newton model used only to propose a point.

    This path deliberately has no certificate role.  Every proposed point is
    re-evaluated by the twofold-ball kernel before it can move a bracket or
    authorize success.
    """

    residual = y - target
    if spec.objective.mismatch_kind == 'squared':
        derivative_terms = [confidence * residual, rho * (y - v)]
        curvature_terms = [confidence, rho]
    else:
        assert spec.objective.huber_delta_value is not None
        delta = spec.objective.huber_delta_value
        mismatch_derivative = confidence * max(-delta, min(delta, residual))
        derivative_terms = [mismatch_derivative, rho * (y - v)]
        curvature_terms = [
            confidence if abs(residual) <= delta else 0.0,
            rho,
        ]
    try:
        for penalty in spec.objective.penalties:
            if penalty.kind == 'soft':
                if y < penalty.lower_value:
                    derivative_terms.append(
                        2.0 * penalty.strength_value
                        * (y - penalty.lower_value)
                    )
                    curvature_terms.append(2.0 * penalty.strength_value)
                elif y > penalty.upper_value:
                    derivative_terms.append(
                        2.0 * penalty.strength_value
                        * (y - penalty.upper_value)
                    )
                    curvature_terms.append(2.0 * penalty.strength_value)
                continue
            if penalty.kind == 'exponential':
                assert penalty.margin_value is not None
                assert penalty.tau_value is not None
                lower_argument = math.fsum((
                    penalty.lower_value,
                    penalty.margin_value,
                    -y,
                )) / penalty.tau_value
                upper_argument = math.fsum((
                    y,
                    -penalty.upper_value,
                    penalty.margin_value,
                )) / penalty.tau_value
                lower_exponential = math.exp(lower_argument)
                upper_exponential = math.exp(upper_argument)
                derivative_terms.append(
                    penalty.strength_value / penalty.tau_value
                    * (upper_exponential - lower_exponential)
                )
                curvature_terms.append(
                    penalty.strength_value / (penalty.tau_value**2)
                    * (upper_exponential + lower_exponential)
                )
                continue
            assert penalty.epsilon_value is not None
            assert penalty.margin_value is not None
            lower_distance = y - penalty.lower_value
            upper_distance = penalty.upper_value - y
            if lower_distance < penalty.margin_value:
                distance = max(lower_distance, penalty.epsilon_value)
                derivative_terms.append(
                    -penalty.strength_value / (distance**2)
                )
                if lower_distance > penalty.epsilon_value:
                    curvature_terms.append(
                        2.0 * penalty.strength_value / (distance**3)
                    )
            if upper_distance < penalty.margin_value:
                distance = max(upper_distance, penalty.epsilon_value)
                derivative_terms.append(
                    penalty.strength_value / (distance**2)
                )
                if upper_distance > penalty.epsilon_value:
                    curvature_terms.append(
                        2.0 * penalty.strength_value / (distance**3)
                    )
        derivative = math.fsum(derivative_terms)
        curvature = math.fsum(curvature_terms)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(derivative) or not math.isfinite(curvature):
        return None
    if not curvature > 0.0:
        return None
    return derivative, curvature


def _untrusted_initial_proposal(
    spec: _ScalarProxSpec,
    *,
    initial: float,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    lower: float,
    upper: float,
) -> float:
    """Run a cheap uncertified Newton prepass, then bias one ulp left."""

    value = float(initial)
    for _ in range(24):
        model = _proposal_derivative_curvature(
            spec,
            y=value,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        )
        if model is None:
            break
        derivative, curvature = model
        proposal = value - derivative / curvature
        if not math.isfinite(proposal):
            break
        proposal = min(max(proposal, lower), upper)
        if proposal == value:
            break
        value = proposal
    if lower < value < upper:
        biased = math.nextafter(value, -math.inf)
        if lower < biased < upper:
            value = biased
    return value


def _untrusted_batch_proposal(
    spec: _ScalarProxSpec,
    *,
    initial: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    v: np.ndarray,
    rho: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Vectorized proposal-only Newton prepass for heterogeneous rows."""

    value = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(24):
        residual = value - target
        if spec.objective.mismatch_kind == 'squared':
            derivative = confidence * residual + rho * (value - v)
            curvature = confidence + rho
        else:
            assert spec.objective.huber_delta_value is not None
            delta = spec.objective.huber_delta_value
            derivative = (
                confidence * np.clip(residual, -delta, delta)
                + rho * (value - v)
            )
            curvature = (
                np.where(np.abs(residual) <= delta, confidence, 0.0)
                + rho
            )
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            for penalty in spec.objective.penalties:
                if penalty.kind == 'soft':
                    derivative += np.where(
                        value < penalty.lower_value,
                        2.0 * penalty.strength_value
                        * (value - penalty.lower_value),
                        np.where(
                            value > penalty.upper_value,
                            2.0 * penalty.strength_value
                            * (value - penalty.upper_value),
                            0.0,
                        ),
                    )
                    curvature += np.where(
                        (value < penalty.lower_value)
                        | (value > penalty.upper_value),
                        2.0 * penalty.strength_value,
                        0.0,
                    )
                    continue
                if penalty.kind == 'exponential':
                    assert penalty.margin_value is not None
                    assert penalty.tau_value is not None
                    left = np.exp(
                        (penalty.lower_value + penalty.margin_value - value)
                        / penalty.tau_value
                    )
                    right = np.exp(
                        (value - penalty.upper_value + penalty.margin_value)
                        / penalty.tau_value
                    )
                    derivative += (
                        penalty.strength_value / penalty.tau_value
                        * (right - left)
                    )
                    curvature += (
                        penalty.strength_value / (penalty.tau_value**2)
                        * (right + left)
                    )
                    continue
                # Reciprocal rows are excluded before this proposal helper is
                # entered.  Keep a finite proposal if called defensively.
                return np.asarray(initial, dtype=np.float64).copy()
        proposal = value - derivative / curvature
        proposal = np.minimum(np.maximum(proposal, lower), upper)
        valid = np.isfinite(proposal)
        changed = valid & (proposal != value)
        if not np.any(changed):
            break
        value = np.where(changed, proposal, value)
    interior = (value > lower) & (value < upper)
    biased = np.nextafter(value, -np.inf)
    return np.where(interior & (biased > lower), biased, value)


def _batch_spec_supported(spec: _ScalarProxSpec) -> bool:
    """Return whether every active penalty has an array-ball implementation."""

    return all(
        penalty.kind in ('soft', 'exponential')
        for penalty in spec.objective.penalties
    )


def _solve_scalar_prox_batch_ordinary(
    *,
    spec: _ScalarProxSpec,
    initial: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    v: np.ndarray,
    rho: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> _ScalarProxBatchOutcome:
    """Return evidence-bearing certificates for supported ordinary lanes."""

    initial = np.asarray(initial, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    shape = initial.shape
    empty_results: tuple[_ScalarProxResult | None, ...] = tuple(
        None for _ in range(initial.size)
    )
    eligible = (
        np.isfinite(initial)
        & np.isfinite(target)
        & np.isfinite(confidence)
        & (confidence >= 0.0)
        & np.isfinite(v)
        & ~np.isnan(lower)
        & ~np.isnan(upper)
        & (lower < upper)
    )
    if not _batch_spec_supported(spec) or not math.isfinite(rho) or rho <= 0.0:
        return _ScalarProxBatchOutcome(
            results=empty_results,
            eligible=np.zeros(shape, dtype=bool),
            certified=np.zeros(shape, dtype=bool),
            probe_counts=np.zeros(shape, dtype=np.int64),
        )

    try:
        # Array arithmetic is a proof accelerator.  Exceptional floating
        # results are consumed only through explicit per-lane resolved masks;
        # Python, type, and shape errors remain visible programming errors.
        with np.errstate(all='ignore'):
            proposal = _untrusted_batch_proposal(
                spec,
                initial=initial,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
                lower=lower,
                upper=upper,
            )
            evaluation = _array_scalar_derivative_terms(
                spec.objective,
                y=proposal,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            derivative_lower, derivative_upper = (
                evaluation.derivative.physical_bounds()
            )
            negative = (
                eligible
                & evaluation.derivative.resolved
                & (derivative_upper < 0.0)
            )
            positive = (
                eligible
                & evaluation.derivative.resolved
                & (derivative_lower > 0.0)
            )
            left = np.where(negative, proposal, np.nan)
            right = np.where(positive, proposal, np.nan)
            current = proposal.copy()
            probe_counts = np.zeros(shape, dtype=np.int64)

            for _ in range(8):
                missing = eligible & (np.isnan(left) | np.isnan(right))
                if not np.any(missing):
                    break
                direction = np.where(np.isnan(right), np.inf, -np.inf)
                probe = np.nextafter(current, direction)
                feasible_probe = (
                    missing
                    & np.isfinite(probe)
                    & (probe >= lower)
                    & (probe <= upper)
                )
                probe_counts += feasible_probe.astype(np.int64)
                probe = np.where(feasible_probe, probe, current)
                probe_evaluation = _array_scalar_derivative_terms(
                    spec.objective,
                    y=probe,
                    target=target,
                    confidence=confidence,
                    v=v,
                    rho=rho,
                )
                probe_lower, probe_upper = (
                    probe_evaluation.derivative.physical_bounds()
                )
                probe_negative = (
                    probe_evaluation.derivative.resolved
                    & (probe_upper < 0.0)
                )
                probe_positive = (
                    probe_evaluation.derivative.resolved
                    & (probe_lower > 0.0)
                )
                left = np.where(
                    feasible_probe & probe_negative,
                    probe,
                    left,
                )
                right = np.where(
                    feasible_probe & probe_positive,
                    probe,
                    right,
                )
                current = np.where(feasible_probe, probe, current)

            bracketed = (
                eligible
                & np.isfinite(left)
                & np.isfinite(right)
                & (left < right)
                & (np.nextafter(left, np.inf) == right)
            )
            safe_left = np.where(bracketed, left, proposal)
            safe_right = np.where(bracketed, right, proposal)
            left_evaluation = _array_scalar_derivative_terms(
                spec.objective,
                y=safe_left,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            right_evaluation = _array_scalar_derivative_terms(
                spec.objective,
                y=safe_right,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            left_lower, left_upper = (
                left_evaluation.derivative.physical_bounds()
            )
            right_lower, right_upper = (
                right_evaluation.derivative.physical_bounds()
            )
            derivative_certified = (
                left_evaluation.derivative.resolved
                & right_evaluation.derivative.resolved
                & (left_upper < 0.0)
                & (right_lower > 0.0)
            )
            difference = _array_scalar_objective_difference(
                spec.objective,
                lower=safe_left,
                upper=safe_right,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            difference_lower, difference_upper = difference.physical_bounds()
            difference_sign = np.where(
                difference_lower > 0.0,
                1,
                np.where(difference_upper < 0.0, -1, 0),
            )
            certified = (
                bracketed
                & derivative_certified
                & difference.resolved
                & (difference_sign != 0)
            )
            selected = np.where(difference_sign > 0, left, right)
    except FloatingPointError:
        return _ScalarProxBatchOutcome(
            results=empty_results,
            eligible=eligible,
            certified=np.zeros(shape, dtype=bool),
            probe_counts=np.zeros(shape, dtype=np.int64),
            arithmetic_exception=True,
        )

    results: list[_ScalarProxResult | None] = [None] * initial.size
    for index in np.flatnonzero(certified):
        lane = int(index)
        value = float(selected[lane])
        try:
            objective = _objective_at(
                spec,
                value,
                target=float(target[lane]),
                confidence=float(confidence[lane]),
                v=float(v[lane]),
                rho=float(rho),
            )
        except ArithmeticError:
            certified[lane] = False
            continue
        if not math.isfinite(objective):
            certified[lane] = False
            continue
        left_value = float(left[lane])
        right_value = float(right[lane])
        left_bounds = (float(left_lower[lane]), float(left_upper[lane]))
        right_bounds = (float(right_lower[lane]), float(right_upper[lane]))
        ordinary_sign = int(difference_sign[lane])
        terminal = _TerminalDifferenceCertificateEvidence(
            lower_endpoint=left_value,
            upper_endpoint=right_value,
            ordinary_enclosure=(
                float(difference_lower[lane]),
                float(difference_upper[lane]),
            ),
            ordinary_resolved_sign=ordinary_sign,
            fallback_enclosure=None,
            fallback_resolved_sign=None,
            fallback_exact=False,
            selected_endpoint=value,
            selection_reason=(
                'ordinary_positive'
                if ordinary_sign > 0
                else 'ordinary_negative'
            ),
        )
        results[lane] = _ScalarProxResult(
            value=value,
            certificate_kind='adjacent_bracket',
            kkt_residual=0.0,
            scalar_iterations=0,
            expansion_count=0,
            final_bracket=(left_value, right_value),
            objective=objective,
            adjacent_bracket_evidence=(
                _DerivativeCertificateEvidence(
                    y=left_value,
                    minus_enclosure=left_bounds,
                    plus_enclosure=left_bounds,
                    minus_resolved_sign=-1,
                    plus_resolved_sign=-1,
                    minus_ordinary_resolved_sign=-1,
                    plus_ordinary_resolved_sign=-1,
                ),
                _DerivativeCertificateEvidence(
                    y=right_value,
                    minus_enclosure=right_bounds,
                    plus_enclosure=right_bounds,
                    minus_resolved_sign=1,
                    plus_resolved_sign=1,
                    minus_ordinary_resolved_sign=1,
                    plus_ordinary_resolved_sign=1,
                ),
            ),
            terminal_difference_evidence=terminal,
            localization_bound=right_value - left_value,
            fallback_count=0,
            execution_path='batch_ordinary',
            batch_probe_count=int(probe_counts[lane]),
        )
    return _ScalarProxBatchOutcome(
        results=tuple(results),
        eligible=eligible,
        certified=certified,
        probe_counts=probe_counts,
    )


def _penalties_inactive_at(spec: _ScalarProxSpec, value: float) -> bool:
    """Return whether all compiled penalties vanish with zero subgradient."""

    y = float(value)
    for penalty in spec.objective.penalties:
        if penalty.kind == 'exponential':
            return False
        if penalty.kind == 'soft':
            if not penalty.lower_value <= y <= penalty.upper_value:
                return False
            continue
        assert penalty.lower_margin is not None
        assert penalty.upper_margin is not None
        if not (
            y >= penalty.lower_margin.above
            and y <= penalty.upper_margin.below
        ):
            return False
    return True


def _point_residual(
    evaluation: _DerivativeEvaluation,
    *,
    at_lower: bool,
    at_upper: bool,
) -> float:
    """Return a diagnostic residual upper bound, never a success tolerance."""

    minus = evaluation.minus.physical_bounds()
    plus = evaluation.plus.physical_bounds()
    if at_lower:
        return max(-plus[0], 0.0)
    if at_upper:
        return max(minus[1], 0.0)
    return max(abs(minus[0]), abs(minus[1]), abs(plus[0]), abs(plus[1]))


def _candidate_values(
    breakpoints: tuple[Fraction, ...],
    *,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    candidates: set[float] = set()
    for breakpoint in breakpoints:
        below, above = _fraction_float_neighbors(breakpoint)
        for value in (below, above):
            if math.isfinite(value) and lower <= value <= upper:
                candidates.add(value)
    return tuple(sorted(candidates))


def _objective_at(
    spec: _ScalarProxSpec,
    value: float,
    *,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> float:
    return _scalar_objective_value(
        spec.objective,
        y=value,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
    )


def _point_condition(
    evaluation: _DerivativeEvaluation,
    *,
    at_lower: bool,
    at_upper: bool,
) -> bool:
    minus_sign = evaluation.fallback_minus_sign
    plus_sign = evaluation.fallback_plus_sign
    if at_lower:
        if plus_sign is not None:
            return plus_sign >= 0
        return evaluation.plus.lower >= 0.0
    if at_upper:
        if minus_sign is not None:
            return minus_sign <= 0
        return evaluation.minus.upper <= 0.0
    minus_ok = (
        minus_sign <= 0
        if minus_sign is not None
        else evaluation.minus.upper <= 0.0
    )
    plus_ok = (
        plus_sign >= 0
        if plus_sign is not None
        else evaluation.plus.lower >= 0.0
    )
    return minus_ok and plus_ok


def _needs_sign_fallback(evaluation: _DerivativeEvaluation) -> bool:
    if (
        evaluation.fallback_minus_sign is not None
        and evaluation.fallback_plus_sign is not None
    ):
        return False
    return not (
        evaluation.plus.strictly_negative
        or evaluation.minus.strictly_positive
        or _point_condition(evaluation, at_lower=False, at_upper=False)
    )


def _fallback_resolve_derivative(
    evaluation: _DerivativeEvaluation,
    *,
    spec: _ScalarProxSpec,
    target: float,
    confidence: float,
    v: float,
    rho: float,
) -> _DerivativeEvaluation | None:
    """Resolve an ambiguous derivative by exact signs or outward intervals."""

    def ordinary_sign(enclosure: _ScaledEnclosure) -> int | None:
        if enclosure.strictly_negative:
            return -1
        if enclosure.strictly_positive:
            return 1
        if enclosure.lower == 0.0 and enclosure.upper == 0.0:
            return 0
        return None

    ordinary_signs = [
        ordinary_sign(evaluation.minus),
        ordinary_sign(evaluation.plus),
    ]
    needs_fallback = [sign is None for sign in ordinary_signs]
    expressions = tuple(
        _scalar_derivative_exact_expression(
            spec.objective,
            y=evaluation.y,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
            side=side,
        )
        for side in ('minus', 'plus')
    )
    signs: list[int | None] = [
        (
            expression.algebraic_sign
            if needs_fallback[index]
            else ordinary_signs[index]
        )
        for index, expression in enumerate(expressions)
    ]
    intervals = [None, None]
    for precision in _FALLBACK_PRECISIONS:
        for index, expression in enumerate(expressions):
            if signs[index] is not None:
                continue
            try:
                interval = _exact_expression_interval(
                    expression,
                    precision=precision,
                )
            except (ArithmeticError, DecimalException, OverflowError, ValueError):
                return None
            intervals[index] = interval
            interval_sign = interval.sign
            if interval_sign in (-1, 1):
                signs[index] = interval_sign
        if all(sign is not None for sign in signs):
            values: list[float] = []
            for expression, interval in zip(expressions, intervals):
                if interval is None:
                    try:
                        values.append(float(expression.rational))
                    except OverflowError:
                        values.append(
                            -math.inf if expression.rational < 0 else math.inf
                        )
                else:
                    values.append(float(
                        (interval.lower + interval.upper) / Decimal(2)
                    ))
            return _DerivativeEvaluation(
                y=evaluation.y,
                minus=evaluation.minus,
                plus=evaluation.plus,
                curvature_enclosure=evaluation.curvature_enclosure,
                smooth=evaluation.smooth,
                fallback_minus_sign=(signs[0] if needs_fallback[0] else None),
                fallback_plus_sign=(signs[1] if needs_fallback[1] else None),
                fallback_minus_value=(
                    values[0] if needs_fallback[0] else None
                ),
                fallback_plus_value=(
                    values[1] if needs_fallback[1] else None
                ),
                fallback_minus_interval=(
                    None
                    if intervals[0] is None or not needs_fallback[0]
                    else (intervals[0].lower, intervals[0].upper)
                ),
                fallback_plus_interval=(
                    None
                    if intervals[1] is None or not needs_fallback[1]
                    else (intervals[1].lower, intervals[1].upper)
                ),
                fallback_minus_exact=(
                    needs_fallback[0] and intervals[0] is None
                ),
                fallback_plus_exact=(
                    needs_fallback[1] and intervals[1] is None
                ),
            )
    return None


def _newton_proposal(
    evaluation: _DerivativeEvaluation,
    *,
    lower: float,
    upper: float,
    breakpoint_neighbors: tuple[tuple[float, float], ...],
) -> float | None:
    if not evaluation.smooth or evaluation.curvature_enclosure.lower <= 0.0:
        return None
    if (
        evaluation.fallback_minus_value is not None
        and evaluation.fallback_plus_value is not None
    ):
        derivative = 0.5 * (
            evaluation.fallback_minus_value
            + evaluation.fallback_plus_value
        )
        curvature = evaluation.curvature
        if derivative == 0.0 or not math.isfinite(curvature) or curvature <= 0.0:
            return None
        step = derivative / curvature
        proposal = evaluation.y - step
        derivative_center = None
        derivative_log_abs = math.log(abs(derivative))
    else:
        derivative_center = 0.5 * (
            evaluation.minus.lower + evaluation.plus.upper
        )
        curvature_center = 0.5 * (
            evaluation.curvature_enclosure.lower
            + evaluation.curvature_enclosure.upper
        )
        if derivative_center == 0.0 or curvature_center <= 0.0:
            return None
        log_step = (
            math.log(abs(derivative_center))
            - math.log(curvature_center)
            + evaluation.minus.log_scale
            - evaluation.curvature_enclosure.log_scale
        )
        if log_step > math.log(_FLOAT_MAX):
            return None
        if log_step < math.log(math.ulp(0.0)) - 2.0:
            return None
        step = math.copysign(math.exp(log_step), derivative_center)
        proposal = evaluation.y - step
        derivative_log_abs = (
            math.log(abs(derivative_center)) + evaluation.minus.log_scale
        )
    if (
        evaluation.y in (lower, upper)
        and derivative_log_abs > math.log(2.0**40)
    ):
        return None
    if proposal == evaluation.y:
        direction = upper if (
            (
                evaluation.fallback_minus_value
                if evaluation.fallback_minus_value is not None
                else derivative_center
            )
            < 0.0
        ) else lower
        proposal = math.nextafter(evaluation.y, direction)
    if not math.isfinite(proposal) or not lower < proposal < upper:
        return None
    for below, above in breakpoint_neighbors:
        if below == above:
            current_side = (
                int(evaluation.y > below) - int(evaluation.y < below)
            )
            proposal_side = int(proposal > below) - int(proposal < below)
        else:
            current_side = -1 if evaluation.y <= below else 1
            proposal_side = -1 if proposal <= below else 1
        if current_side == 0:
            return None
        if proposal_side == 0 or proposal_side != current_side:
            return None
    lower_key = _float_to_ordered_int(lower)
    upper_key = _float_to_ordered_int(upper)
    proposal_key = _float_to_ordered_int(proposal)
    contraction_floor = 1
    if (
        proposal_key - lower_key < contraction_floor
        or upper_key - proposal_key < contraction_floor
    ):
        return None
    return proposal


def _ties_to_even_endpoint(lower: float, upper: float) -> float:
    """Choose the adjacent endpoint whose binary significand has even LSB."""

    for value in (lower, upper):
        bits = struct.unpack('>Q', struct.pack('>d', float(value)))[0]
        if bits & 1 == 0:
            return float(value)
    return float(lower)


def _solve_scalar_prox_coordinate(
    *,
    spec: _ScalarProxSpec,
    target: float,
    confidence: float,
    v: float,
    rho: float,
    lower: float = float('-inf'),
    upper: float = float('inf'),
) -> _ScalarProxResult:
    """Solve one coordinate by exact point signs or an adjacent sign bracket."""

    scalar_iterations = 0
    expansion_count = 0
    fallback_count = 0
    last_candidate: float | None = None
    last_bracket: tuple[float, float] | None = None
    last_enclosure = None
    localization_bound: float | None = None

    def fail(reason: str) -> None:
        raise _ScalarProxError(
            _ScalarProxFailure(
                reason=reason,
                scalar_iterations=scalar_iterations,
                expansion_count=expansion_count,
                last_candidate=last_candidate,
                last_finite_bracket=last_bracket,
                last_derivative_enclosure=last_enclosure,
                localization_bound=localization_bound,
                fallback_count=fallback_count,
            )
        )

    target = float(target)
    confidence = float(confidence)
    v = float(v)
    rho = float(rho)
    lower = float(lower)
    upper = float(upper)
    if math.isnan(lower) or math.isnan(upper):
        fail('invalid_interval_nan_endpoint')
    if lower > upper:
        fail('invalid_interval_order')
    if not math.isfinite(rho) or rho <= 0.0:
        fail('invalid_rho')
    if not (
        math.isfinite(target)
        and math.isfinite(confidence)
        and confidence >= 0.0
        and math.isfinite(v)
    ):
        fail('invalid_required_finite_data')

    breakpoints = _coordinate_breakpoints(
        spec,
        target=target,
        lower=lower,
        upper=upper,
    )
    breakpoint_neighbors = tuple(
        _fraction_float_neighbors(breakpoint) for breakpoint in breakpoints
    )

    def success(
        value: float,
        *,
        kind: str,
        bracket: tuple[float, float],
        point_evaluation: _DerivativeEvaluation | None = None,
        adjacent_evaluations: (
            tuple[_DerivativeEvaluation, _DerivativeEvaluation] | None
        ) = None,
        terminal_difference_evidence: (
            _TerminalDifferenceCertificateEvidence | None
        ) = None,
    ) -> _ScalarProxResult:
        objective = _objective_at(
            spec,
            value,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        )
        if not math.isfinite(objective):
            fail('nonfinite_authoritative_objective')
        return _ScalarProxResult(
            value=value,
            certificate_kind=kind,
            kkt_residual=0.0,
            scalar_iterations=scalar_iterations,
            expansion_count=expansion_count,
            final_bracket=bracket,
            objective=objective,
            point_derivative_evidence=(
                None
                if point_evaluation is None
                else point_evaluation.certificate_evidence()
            ),
            adjacent_bracket_evidence=(
                None
                if adjacent_evaluations is None
                else tuple(
                    evaluation.certificate_evidence()
                    for evaluation in adjacent_evaluations
                )
            ),
            terminal_difference_evidence=terminal_difference_evidence,
            localization_bound=localization_bound,
            fallback_count=fallback_count,
            execution_path=(
                'scalar_fallback'
                if fallback_count
                else 'scalar_ordinary'
            ),
        )

    if lower == upper:
        if not math.isfinite(lower):
            fail('nonfinite_equality_domain')
        last_candidate = lower
        return success(
            lower,
            kind='equality',
            bracket=(lower, upper),
        )

    initial = _mismatch_only_candidate(
        spec,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
    )
    initial = min(max(initial, lower), upper)
    initial = _untrusted_initial_proposal(
        spec,
        initial=initial,
        target=target,
        confidence=confidence,
        v=v,
        rho=rho,
        lower=lower,
        upper=upper,
    )
    if not math.isfinite(initial):
        fail('nonfinite_initial_candidate')

    evaluations: dict[float, _DerivativeEvaluation] = {}

    def evaluate(value: float, *, resolve: bool = True) -> _DerivativeEvaluation:
        nonlocal last_candidate, last_enclosure, fallback_count
        last_candidate = value
        result = evaluations.get(value)
        if result is None:
            try:
                result = _evaluate_derivative(
                    spec,
                    y=value,
                    target=target,
                    confidence=confidence,
                    v=v,
                    rho=rho,
                )
            except (ArithmeticError, InvalidOperation, OverflowError, ValueError):
                fail('nonfinite_or_unsupported_derivative_evaluation')
                raise AssertionError('unreachable')
        if resolve and _needs_sign_fallback(result):
            if fallback_count >= _MAX_FALLBACK_DECISIONS:
                fail('high_precision_fallback_budget_exhausted')
            fallback_count += 1
            resolved = _fallback_resolve_derivative(
                result,
                spec=spec,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            if resolved is None:
                fail('high_precision_derivative_sign_unresolved')
            result = resolved
        evaluations[value] = result
        last_enclosure = result.physical_enclosures()
        return result

    def try_point(value: float) -> _ScalarProxResult | None:
        evaluation = evaluate(value)
        if _point_condition(
            evaluation,
            at_lower=value == lower,
            at_upper=value == upper,
        ):
            return success(
                value,
                kind='point_kkt',
                bracket=(value, value),
                point_evaluation=evaluation,
            )
        return None

    left: float | None = None
    right: float | None = None
    initial_success = try_point(initial)
    if initial_success is not None:
        return initial_success
    initial_evaluation = evaluate(initial)
    if initial_evaluation.certified_negative:
        left = initial
    elif initial_evaluation.certified_positive:
        right = initial
    else:
        fail('derivative_sign_not_certified')

    # A proposal prepass normally lands within a few lattice points of the
    # root.  Probe only in the certified monotone direction before resorting
    # to geometric expansion; every probe still consumes a fresh ball sign.
    local_probe = initial
    for _ in range(8):
        if left is not None and right is not None:
            break
        direction = math.inf if left is not None else -math.inf
        probe = math.nextafter(local_probe, direction)
        if not math.isfinite(probe) or not lower <= probe <= upper:
            break
        probe_success = try_point(probe)
        if probe_success is not None:
            return probe_success
        probe_evaluation = evaluate(probe)
        if probe_evaluation.certified_negative:
            left = probe if left is None else max(left, probe)
        elif probe_evaluation.certified_positive:
            right = probe if right is None else min(right, probe)
        else:
            fail('derivative_sign_not_certified')
        local_probe = probe

    for endpoint in (lower, upper):
        if not math.isfinite(endpoint) or endpoint == initial:
            continue
        endpoint_success = try_point(endpoint)
        if endpoint_success is not None:
            return endpoint_success
        endpoint_evaluation = evaluate(endpoint)
        if endpoint_evaluation.certified_negative:
            left = endpoint if left is None else max(left, endpoint)
        if endpoint_evaluation.certified_positive:
            right = endpoint if right is None else min(right, endpoint)

    initial_scale = max(
        1.0,
        abs(target),
        abs(v),
        abs(initial),
        *(abs(value) for value in (lower, upper) if math.isfinite(value)),
    )
    while (left is None or right is None) and expansion_count < _MAX_EXPANSIONS:
        exponent = _EXPANSION_EXPONENT_STEP * (expansion_count + 1)
        try:
            magnitude = math.ldexp(initial_scale, exponent)
        except OverflowError:
            magnitude = math.inf
        if left is None:
            probe = max(
                -_FLOAT_MAX
                if not math.isfinite(initial - magnitude)
                else initial - magnitude,
                lower,
            )
        else:
            probe = min(
                _FLOAT_MAX
                if not math.isfinite(initial + magnitude)
                else initial + magnitude,
                upper,
            )
        expansion_count += 1
        if not math.isfinite(probe) or probe in evaluations:
            continue
        probe_success = try_point(probe)
        if probe_success is not None:
            return probe_success
        probe_evaluation = evaluate(probe)
        if probe_evaluation.certified_negative:
            left = probe if left is None else max(left, probe)
        if probe_evaluation.certified_positive:
            right = probe if right is None else min(right, probe)

    if left is None or right is None or not left < right:
        fail('bracketing_expansion_limit')
    last_bracket = (left, right)
    current = initial if left <= initial <= right else None

    def adjacent_success() -> _ScalarProxResult | None:
        nonlocal fallback_count, localization_bound, last_enclosure
        if not _ordered_floats_adjacent(left, right):
            return None
        left_evaluation = evaluate(left)
        right_evaluation = evaluate(right)
        if not (
            left_evaluation.certified_negative
            and right_evaluation.certified_positive
        ):
            fail('adjacent_bracket_not_certified')
        difference = _scalar_objective_difference(
            spec.objective,
            lower=left,
            upper=right,
            target=target,
            confidence=confidence,
            v=v,
            rho=rho,
        )
        ordinary_enclosure = difference.physical_bounds()
        ordinary_sign: int | None = None
        fallback_enclosure: tuple[Decimal, Decimal] | None = None
        fallback_sign: int | None = None
        fallback_exact = False
        if difference.lower > 0.0:
            selected = left
            ordinary_sign = 1
            selection_reason = 'ordinary_positive'
        elif difference.upper < 0.0:
            selected = right
            ordinary_sign = -1
            selection_reason = 'ordinary_negative'
        else:
            if fallback_count >= _MAX_FALLBACK_DECISIONS:
                fail('high_precision_fallback_budget_exhausted')
            fallback_count += 1
            expression = _scalar_objective_difference_exact_expression(
                spec.objective,
                lower=left,
                upper=right,
                target=target,
                confidence=confidence,
                v=v,
                rho=rho,
            )
            sign = expression.algebraic_sign
            fallback_exact = sign is not None
            if sign is None:
                for precision in _FALLBACK_PRECISIONS:
                    try:
                        interval = _exact_expression_interval(
                            expression,
                            precision=precision,
                        )
                    except (
                        ArithmeticError,
                        DecimalException,
                        OverflowError,
                        ValueError,
                    ):
                        break
                    interval_sign = interval.sign
                    if interval_sign in (-1, 1):
                        sign = interval_sign
                        fallback_enclosure = (interval.lower, interval.upper)
                        break
            if sign is None:
                fail('high_precision_objective_difference_unresolved')
            fallback_sign = sign
            selected = (
                left
                if sign > 0
                else right
                if sign < 0
                else _ties_to_even_endpoint(left, right)
            )
            selection_reason = (
                'exact_zero_ties_to_even'
                if sign == 0
                else 'exact_positive'
                if fallback_exact and sign > 0
                else 'exact_negative'
                if fallback_exact
                else 'decimal_positive'
                if sign > 0
                else 'decimal_negative'
            )
        localization_bound = right - left
        last_enclosure = (
            left_evaluation.physical_enclosures()[1],
            right_evaluation.physical_enclosures()[0],
        )
        return success(
            selected,
            kind='adjacent_bracket',
            bracket=(left, right),
            adjacent_evaluations=(left_evaluation, right_evaluation),
            terminal_difference_evidence=(
                _TerminalDifferenceCertificateEvidence(
                    lower_endpoint=left,
                    upper_endpoint=right,
                    ordinary_enclosure=ordinary_enclosure,
                    ordinary_resolved_sign=ordinary_sign,
                    fallback_enclosure=fallback_enclosure,
                    fallback_resolved_sign=fallback_sign,
                    fallback_exact=fallback_exact,
                    selected_endpoint=selected,
                    selection_reason=selection_reason,
                )
            ),
        )

    for scalar_iterations in range(1, _MAX_SCALAR_ITERATIONS + 1):
        adjacent = adjacent_success()
        if adjacent is not None:
            return adjacent
        proposal = None
        if current is not None:
            proposal = _newton_proposal(
                evaluate(current),
                lower=left,
                upper=right,
                breakpoint_neighbors=breakpoint_neighbors,
            )
        if proposal is None or proposal in evaluations:
            arithmetic = left + 0.5 * (right - left)
            if (
                math.isfinite(arithmetic)
                and left < arithmetic < right
                and arithmetic not in evaluations
            ):
                proposal = arithmetic
            else:
                proposal = _ordered_float_midpoint(left, right)
        point = try_point(proposal)
        if point is not None:
            return point
        proposal_evaluation = evaluate(proposal)
        if proposal_evaluation.certified_negative:
            left = proposal
        elif proposal_evaluation.certified_positive:
            right = proposal
        else:
            fail('derivative_sign_not_certified')
        last_bracket = (left, right)
        current = proposal

    # Fresh endpoint decisions at the resource limit; stale metadata is not a
    # certificate.
    evaluations.pop(left, None)
    evaluations.pop(right, None)
    adjacent = adjacent_success()
    if adjacent is not None:
        return adjacent
    fail('scalar_iteration_limit')
    raise AssertionError('unreachable')
