"""Public power-fit problem construction, evaluation, and result packaging."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal
import sys

import numpy as np

from ..._internal.inputs import coerce_finite_vector, coerce_real_vector
from ..._internal.validation import (
    require_bool,
    require_bool_mask,
    require_nonnegative_finite_real,
    require_nonnegative_index,
    require_optional_string,
    require_string,
    require_string_tuple,
)
from ..._internal.weight_transforms import (
    validate_weight_representation_options,
    weights_to_radii,
)
from ._objective import (
    _active_scalar_penalties,
    _hard_accepted_measurement_bounds,
    _hard_row_status,
    _l2_value,
    _mismatch_terms as _objective_mismatch_terms,
    _mismatch_values as _objective_mismatch_values,
    _mismatch_values_from_affine,
    _penalty_terms,
    _penalty_value,
    _penalty_value_from_affine,
    _quadratic_row_data,
)
from ._numerics import (
    _stable_affine_difference,
    _stable_affine_residual,
    _stable_incidence_accumulate,
    _stable_mean_abs,
    _stable_norm,
    _stable_product,
    _stable_ratio_difference,
    _stable_rms,
    _stable_scaled_difference,
    _stable_sum_products,
    _stable_sum_scalar,
)
from .constraints import SeparatorObservations
from ._identity import _bind_originating_observations
from .model import (
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HardConstraint,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)
from .operators import (
    SeparatorObservationGraphView,
    SeparatorQuadraticOperatorView,
)
from .types import (
    AlgebraicEdgeDiagnostics,
    ConnectivityDiagnostics,
    ConstraintGraphDiagnostics,
    HardConstraintConflict,
    HardConstraintConflictTerm,
    PowerFitBounds,
    PowerFitObjectiveBreakdown,
    PowerFitPredictions,
    SeparatorFitResult,
    _readonly_array,
)


class _NonFiniteOptimalObjectiveError(ValueError):
    """Raised when result packaging would claim success for a non-finite objective."""


_ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES: ContextVar[bool] = ContextVar(
    '_ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES',
    default=False,
)


@dataclass(frozen=True, slots=True)
class _MeasurementGeometry:
    alpha: np.ndarray
    beta: np.ndarray
    target: np.ndarray
    target_fraction: np.ndarray
    target_position: np.ndarray


@dataclass(frozen=True, slots=True)
class _DifferenceEdge:
    source: int
    target: int
    weight: float
    constraint_index: int
    site_i: int
    site_j: int
    relation: Literal['<=', '>=']
    bound_value: float


@dataclass(frozen=True, slots=True)
class SeparatorFitProblem:
    """Resolved fixed-observation separator fit problem.

    ``offset_identifying_constraint_mask`` is the historical public name for
    the row mask used to decompose the numerical problem.  It includes rows
    touched by hard restrictions or positive-strength penalties because those
    terms can couple solver variables.  Zero-strength penalties are absent.
    The mask does not claim that those rows identify offsets from separator
    data or select them uniquely.
    """

    constraints: SeparatorObservations
    model: FitModel
    alpha: np.ndarray
    beta: np.ndarray
    z_obs: np.ndarray
    edge_weight: np.ndarray
    regularization_strength: float
    regularization_reference: np.ndarray
    offset_identifying_constraint_mask: np.ndarray
    bounds: PowerFitBounds
    connectivity: ConnectivityDiagnostics
    hard_feasible: bool
    hard_conflict: HardConstraintConflict | None

    def __post_init__(self) -> None:
        m = int(self.constraints.n_constraints)
        n = int(self.constraints.n_points)
        # R1/R2 deliberately support finite source data whose scaled derived
        # row representation overflows. Only the builder may retain those
        # derived infinities; direct construction remains finite-strict.
        derived_vector = (
            coerce_real_vector
            if _ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES.get()
            else coerce_finite_vector
        )
        alpha = _readonly_array(
            derived_vector(self.alpha, name='alpha', n=m),
            dtype=np.float64,
        )
        beta = _readonly_array(
            derived_vector(self.beta, name='beta', n=m),
            dtype=np.float64,
        )
        z_obs = _readonly_array(
            derived_vector(self.z_obs, name='z_obs', n=m),
            dtype=np.float64,
        )
        edge_weight = _readonly_array(
            derived_vector(self.edge_weight, name='edge_weight', n=m),
            dtype=np.float64,
        )
        regularization_reference = _readonly_array(
            coerce_finite_vector(
                self.regularization_reference,
                name='regularization_reference',
                n=n,
            ),
            dtype=np.float64,
        )
        offset_mask = require_bool_mask(
            self.offset_identifying_constraint_mask,
            name='offset_identifying_constraint_mask',
            length=m,
        )
        regularization_strength = require_nonnegative_finite_real(
            self.regularization_strength,
            name='regularization_strength',
        )
        hard_feasible = require_bool(
            self.hard_feasible,
            name='hard_feasible',
        )

        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'beta', beta)
        object.__setattr__(self, 'z_obs', z_obs)
        object.__setattr__(
            self,
            'edge_weight',
            edge_weight,
        )
        object.__setattr__(
            self,
            'regularization_reference',
            regularization_reference,
        )
        object.__setattr__(
            self,
            'offset_identifying_constraint_mask',
            offset_mask,
        )
        object.__setattr__(
            self,
            'regularization_strength',
            regularization_strength,
        )
        object.__setattr__(self, 'hard_feasible', hard_feasible)

    @property
    def measurement(self) -> str:
        return self.constraints.measurement

    @property
    def measurement_target(self) -> np.ndarray:
        target = (
            self.constraints.target_fraction
            if self.constraints.measurement == 'fraction'
            else self.constraints.target_position
        )
        return np.asarray(target, dtype=np.float64)

    @property
    def confidence(self) -> np.ndarray:
        return np.asarray(self.constraints.confidence, dtype=np.float64)

    @property
    def observation_graph(self) -> SeparatorObservationGraphView:
        """Return the oriented observation multigraph and its row data.

        Every resolved observation remains a distinct incidence column.  The
        view shares the problem's established read-only arrays; only the
        derived positive-confidence mask is newly allocated.
        """

        informative_mask = _readonly_array(
            self.constraints.confidence > 0.0,
            dtype=bool,
        )
        return SeparatorObservationGraphView(
            n_sites=int(self.constraints.n_points),
            site_i=self.constraints.i,
            site_j=self.constraints.j,
            observation_indices=self.constraints.input_index,
            requested_shifts=self.constraints.shifts,
            alpha=self.alpha,
            beta=self.beta,
            z_obs=self.z_obs,
            rho=self.edge_weight,
            informative_mask=informative_mask,
            connectivity=self.connectivity,
        )

    @property
    def quadratic_operator(self) -> SeparatorQuadraticOperatorView:
        """Return the exact fixed least-squares normal operator.

        This view is intentionally limited to ``SquaredLoss`` models without
        positive-strength scalar penalties.  Zero-strength penalties are
        absent from the objective.  Hard restrictions may coexist, but they
        remain in ``bounds`` and are not folded into the unconstrained normal
        equation.
        """

        if not isinstance(self.model.mismatch, SquaredLoss):
            raise ValueError(
                'quadratic_operator is available only for SquaredLoss models'
            )
        if _active_scalar_penalties(self.model.penalties):
            raise ValueError(
                'quadratic_operator is unavailable when positive-strength '
                'scalar penalties are present because one fixed normal system '
                'does not represent the full objective'
            )

        graph = self.observation_graph
        rows = _quadratic_row_data(
            self.alpha,
            self.beta,
            self.measurement_target,
            self.confidence,
        )
        observation_rhs = _stable_incidence_accumulate(
            int(self.constraints.n_points),
            self.constraints.i,
            self.constraints.j,
            rows.rhs,
        )
        regularized_rhs = _stable_sum_products(
            (
                (observation_rhs,),
                (
                    float(self.regularization_strength),
                    self.regularization_reference,
                ),
            )
        )
        return SeparatorQuadraticOperatorView(
            observation_graph=graph,
            observation_rhs=_readonly_array(
                observation_rhs,
                dtype=np.float64,
            ),
            regularized_normal_rhs=_readonly_array(
                regularized_rhs,
                dtype=np.float64,
            ),
            regularization_strength=float(self.regularization_strength),
            regularization_reference=self.regularization_reference,
            bounds=self.bounds,
            has_hard_constraints=self.model.feasible is not None,
        )

    def _model_coupling_components(self) -> list[list[int]]:
        mask = np.asarray(self.offset_identifying_constraint_mask, dtype=bool)
        return _connected_components(
            int(self.constraints.n_points),
            self.constraints.i[mask],
            self.constraints.j[mask],
        )

    @property
    def suggested_anchor_indices(self) -> tuple[int, ...]:
        components = self._model_coupling_components()
        if (
            float(self.regularization_strength) > 0.0
            or int(self.constraints.n_points) <= 1
            or len(components) == 1
        ):
            return tuple()
        return tuple(
            int(component[0])
            for component in components
            if component
        )

    def predict(self, weights: np.ndarray) -> PowerFitPredictions:
        w = _validated_weight_vector(self, weights)
        return _predict_all(self, w)

    def predict_difference(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(weights).difference, dtype=np.float64)

    def predict_fraction(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(weights).fraction, dtype=np.float64)

    def predict_position(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(weights).position, dtype=np.float64)

    def predict_measurement(self, weights: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(weights).measurement, dtype=np.float64)

    def edge_diagnostics(self, weights: np.ndarray) -> AlgebraicEdgeDiagnostics:
        w = _validated_weight_vector(self, weights)
        predictions = _predict_all(self, w)
        return _compute_edge_diagnostics(
            self.constraints,
            weights=w,
            predictions=predictions,
        )

    def objective_breakdown(
        self,
        weights: np.ndarray,
    ) -> PowerFitObjectiveBreakdown:
        w = _validated_weight_vector(self, weights)
        predictions = _predict_all(self, w)
        return _objective_breakdown(self, predictions, w)

    def evaluate_objective(self, weights: np.ndarray) -> float:
        parts = self.objective_breakdown(weights)
        if not parts.hard_constraints_satisfied:
            return float('inf')
        return float(parts.total)

    def canonicalize_gauge(self, weights: np.ndarray) -> np.ndarray:
        """Apply the standalone component gauge convention to candidate weights."""

        w = _validated_weight_vector(self, weights)
        components = self._model_coupling_components()
        if (
            float(self.regularization_strength) > 0.0
            or int(self.constraints.n_points) <= 1
            or len(components) == 1
        ):
            return w.copy()
        return _apply_component_mean_gauge(
            w,
            components,
            reference=(
                None
                if self.model.regularization.reference is None
                else self.regularization_reference
            ),
        )


def build_power_fit_problem(
    constraints: SeparatorObservations,
    *,
    model: FitModel | None = None,
) -> SeparatorFitProblem:
    """Build a public separator-fit problem from resolved observations."""

    if model is None:
        model = FitModel()
    geom = _measurement_geometry(constraints)
    reg_ref = _regularization_reference(model.regularization, constraints.n_points)
    hard_measurement = _hard_constraint_measurement_bounds(
        model.feasible,
        constraints.n_constraints,
    )
    hard_diff = (
        None
        if hard_measurement is None
        else _hard_constraint_bounds(
            hard_measurement[0],
            hard_measurement[1],
            geom.alpha,
            geom.beta,
        )
    )
    bounds = PowerFitBounds(
        measurement_lower=None if hard_measurement is None else hard_measurement[0],
        measurement_upper=None if hard_measurement is None else hard_measurement[1],
        difference_lower=None if hard_diff is None else hard_diff[0],
        difference_upper=None if hard_diff is None else hard_diff[1],
    )
    hard_feasible = True
    conflict = None
    if hard_diff is not None:
        hard_feasible, conflict = _check_hard_feasibility(
            int(constraints.n_points),
            constraints.i,
            constraints.j,
            hard_diff[0],
            hard_diff[1],
        )
    connectivity = _build_fit_connectivity_diagnostics(
        constraints,
        model=model,
        gauge_policy=_standalone_gauge_policy_description(model),
    )
    alpha = np.asarray(geom.alpha, dtype=np.float64)
    beta = np.asarray(geom.beta, dtype=np.float64)
    target = np.asarray(geom.target, dtype=np.float64)
    quadratic_rows = _quadratic_row_data(
        alpha,
        beta,
        target,
        constraints.confidence,
    )
    token = _ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES.set(True)
    try:
        return SeparatorFitProblem(
            constraints=constraints,
            model=model,
            alpha=alpha,
            beta=beta,
            z_obs=quadratic_rows.z_obs,
            edge_weight=quadratic_rows.rho,
            regularization_strength=float(model.regularization.strength),
            regularization_reference=reg_ref,
            offset_identifying_constraint_mask=_model_coupling_constraint_mask(
                constraints,
                model,
            ),
            bounds=bounds,
            connectivity=connectivity,
            hard_feasible=bool(hard_feasible),
            hard_conflict=conflict,
        )
    finally:
        _ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES.reset(token)


def build_power_fit_result(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
    *,
    solver: str = 'external',
    linear_backend: str | None = None,
    status: str = 'optimal',
    status_detail: str | None = None,
    converged: bool = True,
    n_iter: int = 0,
    warnings: tuple[str, ...] = (),
    canonicalize_gauge: bool = True,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> SeparatorFitResult:
    """Package candidate weights into a standard power-fit result object.

    An optimal or converged request is rejected when any reported
    soft-objective component or total is non-finite.
    """

    solver = require_string(solver, name='solver')
    linear_backend = require_optional_string(
        linear_backend,
        name='linear_backend',
    )
    status = require_string(status, name='status')
    status_detail = require_optional_string(
        status_detail,
        name='status_detail',
    )
    warnings = require_string_tuple(warnings, name='warnings')
    converged_value = require_bool(converged, name='converged')
    canonicalize_gauge_value = require_bool(
        canonicalize_gauge,
        name='canonicalize_gauge',
    )
    n_iter_value = require_nonnegative_index(
        n_iter,
        name='n_iter',
        maximum=sys.maxsize,
    )
    r_min_value, weight_shift_value = validate_weight_representation_options(
        r_min,
        weight_shift,
    )
    w = _validated_weight_vector(problem, weights)
    if canonicalize_gauge_value:
        w = problem.canonicalize_gauge(w)
    predictions = _predict_all(problem, w)
    residuals = _measurement_residuals(problem, w)
    edge_diagnostics = _compute_edge_diagnostics(
        problem.constraints,
        weights=w,
        predictions=predictions,
    )
    objective_breakdown = _objective_breakdown(problem, predictions, w)
    if (
        (status == 'optimal' or converged_value)
        and not _soft_objective_is_finite(objective_breakdown)
    ):
        raise _NonFiniteOptimalObjectiveError(
            'cannot package an optimal or converged result with a non-finite '
            'soft objective'
        )
    warnings_list = list(warnings)
    if not objective_breakdown.hard_constraints_satisfied:
        warnings_list.append(
            'candidate weights violate hard measurement bounds'
        )
    radii, shift = weights_to_radii(
        w,
        r_min=r_min_value,
        weight_shift=weight_shift_value,
    )
    rms = _stable_rms(residuals)
    mx = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    result = SeparatorFitResult(
        status=status,
        status_detail=status_detail,
        hard_feasible=bool(problem.hard_feasible),
        weights=np.asarray(w, dtype=np.float64),
        radii=radii,
        weight_shift=shift,
        measurement=problem.constraints.measurement,
        target=np.asarray(problem.measurement_target, dtype=np.float64),
        predicted=np.asarray(predictions.measurement, dtype=np.float64),
        predicted_fraction=np.asarray(predictions.fraction, dtype=np.float64),
        predicted_position=np.asarray(predictions.position, dtype=np.float64),
        residuals=residuals,
        rms_residual=rms,
        max_residual=mx,
        used_shifts=np.asarray(problem.constraints.shifts),
        solver=solver,
        linear_backend=linear_backend,
        n_iter=n_iter_value,
        converged=converged_value,
        conflict=problem.hard_conflict,
        warnings=tuple(warnings_list),
        connectivity=problem.connectivity,
        edge_diagnostics=edge_diagnostics,
        objective_breakdown=objective_breakdown,
    )
    return _bind_originating_observations(result, problem.constraints)


def _validated_weight_vector(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
) -> np.ndarray:
    return coerce_finite_vector(
        weights,
        name='weights',
        n=int(problem.constraints.n_points),
    )


def _measurement_geometry(constraints: SeparatorObservations) -> _MeasurementGeometry:
    d = constraints.distance
    d2 = constraints.distance2
    if constraints.measurement == 'fraction':
        alpha = _stable_ratio_difference(0.5, 0.0, d2)
        beta = np.full_like(alpha, 0.5)
        target = constraints.target_fraction
    else:
        alpha = _stable_ratio_difference(0.5, 0.0, d)
        beta = 0.5 * d
        target = constraints.target_position
    return _MeasurementGeometry(
        alpha=np.asarray(alpha, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
        target_fraction=np.asarray(constraints.target_fraction, dtype=np.float64),
        target_position=np.asarray(constraints.target_position, dtype=np.float64),
    )


def _predict_all(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
) -> PowerFitPredictions:
    left = weights[problem.constraints.i]
    right = weights[problem.constraints.j]
    z_pred = _stable_scaled_difference(left, right, 1.0)
    fraction = _stable_affine_difference(
        0.5,
        _stable_ratio_difference(
            0.5,
            0.0,
            problem.constraints.distance2,
        ),
        left,
        right,
    )
    position = _stable_affine_difference(
        0.5 * problem.constraints.distance,
        _stable_ratio_difference(
            0.5,
            0.0,
            problem.constraints.distance,
        ),
        left,
        right,
    )
    measurement = (
        fraction if problem.constraints.measurement == 'fraction' else position
    )
    return PowerFitPredictions(
        difference=np.asarray(z_pred, dtype=np.float64),
        fraction=np.asarray(fraction, dtype=np.float64),
        position=np.asarray(position, dtype=np.float64),
        measurement=np.asarray(measurement, dtype=np.float64),
    )


def _measurement_residuals(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
    *,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Return direct affine residuals without materializing predictions."""

    return _stable_affine_residual(
        problem.beta,
        problem.alpha,
        weights[problem.constraints.i],
        weights[problem.constraints.j],
        problem.measurement_target,
        active=active,
    )


def _compute_edge_diagnostics(
    constraints: SeparatorObservations,
    *,
    weights: np.ndarray | None,
    predictions: PowerFitPredictions | None = None,
    geom: _MeasurementGeometry | None = None,
) -> AlgebraicEdgeDiagnostics:
    if geom is None:
        geom = _measurement_geometry(constraints)
    alpha = np.asarray(geom.alpha, dtype=np.float64)
    beta = np.asarray(geom.beta, dtype=np.float64)
    target = np.asarray(geom.target, dtype=np.float64)
    quadratic_rows = _quadratic_row_data(
        alpha,
        beta,
        target,
        constraints.confidence,
    )
    z_obs = quadratic_rows.z_obs
    edge_weight = quadratic_rows.rho
    if weights is None:
        return AlgebraicEdgeDiagnostics(
            alpha=alpha,
            beta=beta,
            z_obs=z_obs,
            z_fit=None,
            residual=None,
            edge_weight=edge_weight,
            weighted_l2=None,
            weighted_rmse=None,
            rmse=None,
            mae=None,
        )
    if predictions is None:
        z_fit = _stable_scaled_difference(
            weights[constraints.i],
            weights[constraints.j],
            1.0,
        )
    else:
        z_fit = np.asarray(predictions.difference, dtype=np.float64)
    residual = _stable_scaled_difference(z_obs, z_fit, 1.0)
    if residual.size:
        measurement_residual = -_stable_affine_residual(
            beta,
            alpha,
            weights[constraints.i],
            weights[constraints.j],
            target,
        )
        weighted_residual = _stable_product(
            np.sqrt(np.asarray(constraints.confidence, dtype=np.float64)),
            measurement_residual,
        )
        weighted_l2 = _stable_norm(weighted_residual)
        weighted_rmse = _stable_rms(weighted_residual)
        rmse = _stable_rms(residual)
        mae = _stable_mean_abs(residual)
    else:
        weighted_l2 = 0.0
        weighted_rmse = 0.0
        rmse = 0.0
        mae = 0.0
    return AlgebraicEdgeDiagnostics(
        alpha=alpha,
        beta=beta,
        z_obs=z_obs,
        z_fit=z_fit,
        residual=np.asarray(residual, dtype=np.float64),
        edge_weight=edge_weight,
        weighted_l2=weighted_l2,
        weighted_rmse=weighted_rmse,
        rmse=rmse,
        mae=mae,
    )


def _edge_diagnostics_for_result(
    result: SeparatorFitResult,
    constraints: SeparatorObservations,
) -> AlgebraicEdgeDiagnostics:
    if result.edge_diagnostics is not None:
        return result.edge_diagnostics
    return _compute_edge_diagnostics(constraints, weights=result.weights)


def _mismatch_values(
    measurement: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> np.ndarray:
    return _objective_mismatch_values(
        measurement,
        target,
        confidence,
        mismatch,
    )


def _penalty_values(
    measurement: np.ndarray,
    penalty: SoftIntervalPenalty
    | ExponentialBoundaryPenalty
    | ReciprocalBoundaryPenalty,
) -> np.ndarray:
    return _penalty_value(measurement, penalty)


def _hard_constraint_status(
    problem: SeparatorFitProblem,
    predictions: PowerFitPredictions,
) -> tuple[bool, float, float]:
    lower = problem.bounds.measurement_lower
    upper = problem.bounds.measurement_upper
    if lower is None or upper is None:
        return True, 0.0, 0.0
    y = np.asarray(predictions.measurement, dtype=np.float64)
    satisfied, violation, tolerance = _hard_row_status(lower, y, upper)
    if violation.size == 0:
        return True, 0.0, 0.0
    max_violation = float(np.max(violation))
    max_tolerance = float(np.max(tolerance))
    return bool(np.all(satisfied)), max_violation, max_tolerance


def _objective_breakdown(
    problem: SeparatorFitProblem,
    predictions: PowerFitPredictions,
    weights: np.ndarray,
) -> PowerFitObjectiveBreakdown:
    confidence = np.asarray(problem.constraints.confidence, dtype=np.float64)
    measurement = np.asarray(predictions.measurement, dtype=np.float64)
    left = weights[problem.constraints.i]
    right = weights[problem.constraints.j]
    mismatch_values = _mismatch_values_from_affine(
        problem.beta,
        problem.alpha,
        left,
        right,
        problem.measurement_target,
        confidence,
        problem.model.mismatch,
    )
    mismatch = _stable_sum_scalar(*mismatch_values.tolist())
    penalty_terms_list: list[tuple[str, float]] = []
    penalties_total = 0.0
    for penalty in problem.model.penalties:
        value = _stable_sum_scalar(
            *_penalty_value_from_affine(
                problem.beta,
                problem.alpha,
                left,
                right,
                measurement,
                penalty,
            ).tolist()
        )
        penalty_terms_list.append((type(penalty).__name__, value))
        penalties_total = _stable_sum_scalar(penalties_total, value)
    reg = _l2_value(
        weights,
        problem.regularization_reference,
        problem.regularization_strength,
    )
    (
        hard_satisfied,
        hard_max_violation,
        hard_max_tolerance,
    ) = _hard_constraint_status(problem, predictions)
    total = _stable_sum_scalar(mismatch, penalties_total, reg)
    return PowerFitObjectiveBreakdown(
        total=float(total),
        mismatch=float(mismatch),
        penalties_total=float(penalties_total),
        penalty_terms=tuple(penalty_terms_list),
        regularization=float(reg),
        hard_constraints_satisfied=bool(hard_satisfied),
        hard_max_violation=float(hard_max_violation),
        hard_max_tolerance=float(hard_max_tolerance),
    )


def _soft_objective_is_finite(
    breakdown: PowerFitObjectiveBreakdown,
) -> bool:
    """Return whether every reported soft-objective value is finite."""

    values = (
        breakdown.total,
        breakdown.mismatch,
        breakdown.penalties_total,
        breakdown.regularization,
        *(value for _, value in breakdown.penalty_terms),
    )
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def _regularization_reference(reg: L2Regularization, n: int) -> np.ndarray:
    if reg.reference is None:
        return np.zeros(n, dtype=np.float64)
    w0 = np.asarray(reg.reference, dtype=float)
    if w0.shape != (n,):
        raise ValueError('regularization.reference must have shape (n,)')
    return np.asarray(w0, dtype=np.float64)


def _informative_observation_mask(
    constraints: SeparatorObservations,
) -> np.ndarray:
    return np.asarray(constraints.confidence > 0.0, dtype=bool)


def _model_coupling_constraint_mask(
    constraints: SeparatorObservations,
    model: FitModel,
) -> np.ndarray:
    mask = _informative_observation_mask(constraints)
    if (
        model.feasible is not None
        or _active_scalar_penalties(model.penalties)
    ):
        mask = np.ones(constraints.n_constraints, dtype=bool)
    return mask


def _component_offsets_selected_by_objective(model: FitModel) -> bool:
    """Return whether a supported extra objective guarantees offset selection.

    Positive L2 regularization is strictly convex in every site weight.  The
    supported scalar penalties are deliberately not classified as uniquely
    selecting component offsets: they may have zero strength or flat regions.
    """

    return float(model.regularization.strength) > 0.0


def _apply_component_mean_gauge(
    weights: np.ndarray,
    comps: list[list[int]],
    *,
    reference: np.ndarray | None,
) -> np.ndarray:
    aligned = np.asarray(weights, dtype=np.float64).copy()
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)
    for comp in comps:
        idx = np.asarray(comp, dtype=np.int64)
        if idx.size == 0:
            continue
        if ref is None:
            target_mean = 0.0
        else:
            target_mean = float(np.mean(ref[idx]))
        current_mean = float(np.mean(aligned[idx]))
        aligned[idx] += target_mean - current_mean
    return aligned


def _standalone_gauge_policy_description(model: FitModel) -> str:
    reg = model.regularization
    if float(reg.strength) > 0.0:
        if reg.reference is not None:
            return (
                'positive L2 regularization selects weights relative to the '
                'supplied reference'
            )
        return 'positive L2 regularization selects weights relative to zero'
    if reg.reference is not None:
        return (
            'disconnected model-coupling components are shifted so each mean '
            'matches its reference mean; within-component solver conventions '
            'are retained'
        )
    return (
        'disconnected model-coupling components are centered to mean zero; '
        'within-component solver conventions are retained'
    )


def _connected_components(
    n: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        adj[i].append(j)
        adj[j].append(i)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        if len(adj[start]) == 0:
            seen[start] = True
            comps.append([start])
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comps.append(sorted(comp))
    return comps


def _graph_diagnostics(
    n: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    *,
    n_constraints: int,
) -> ConstraintGraphDiagnostics:
    ii = np.asarray(i_idx, dtype=np.int64)
    jj = np.asarray(j_idx, dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    if ii.size:
        np.add.at(degree, ii, 1)
        np.add.at(degree, jj, 1)
    isolated = tuple(np.flatnonzero(degree == 0).tolist())
    components = tuple(
        tuple(int(node) for node in comp)
        for comp in _connected_components(n, ii, jj)
    )
    edges = {
        (int(min(i, j)), int(max(i, j)))
        for i, j in zip(ii.tolist(), jj.tolist())
    }
    return ConstraintGraphDiagnostics(
        n_points=int(n),
        n_constraints=int(n_constraints),
        n_edges=int(len(edges)),
        isolated_points=isolated,
        connected_components=components,
        fully_connected=bool((n <= 1) or len(components) == 1),
    )


def _format_component_counts(graph: ConstraintGraphDiagnostics) -> str:
    n_components = graph.n_components
    if n_components == 1:
        return '1 connected component'
    return f'{n_components} connected components'


def _format_point_list(points: tuple[int, ...]) -> str:
    return '[' + ', '.join(str(int(v)) for v in points) + ']'


def _build_fit_connectivity_diagnostics(
    constraints: SeparatorObservations,
    *,
    model: FitModel,
    gauge_policy: str,
) -> ConnectivityDiagnostics:
    n = int(constraints.n_points)
    candidate_graph = _graph_diagnostics(
        n,
        constraints.i,
        constraints.j,
        n_constraints=constraints.n_constraints,
    )
    effective_mask = _informative_observation_mask(constraints)
    effective_graph = _graph_diagnostics(
        n,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
        n_constraints=int(np.count_nonzero(effective_mask)),
    )

    messages: list[str] = []
    if candidate_graph.isolated_points:
        messages.append(
            'candidate graph leaves unconstrained points '
            f'{_format_point_list(candidate_graph.isolated_points)}'
        )
    if candidate_graph.n_components > 1:
        messages.append(
            'candidate graph has ' f'{_format_component_counts(candidate_graph)}'
        )
    if np.any(~effective_mask):
        messages.append(
            'zero-confidence candidate rows are excluded from the informative '
            'observation graph and data-identification diagnostics; model '
            'restrictions and penalties are assessed separately'
        )
    if effective_graph.n_components > 1:
        messages.append(
            'pairwise data identify only '
            f'{_format_component_counts(effective_graph)}; relative component '
            'offsets are not identified by the data'
        )

    return ConnectivityDiagnostics(
        unconstrained_points=candidate_graph.isolated_points,
        candidate_graph=candidate_graph,
        effective_graph=effective_graph,
        candidate_offsets_identified_by_data=bool(effective_graph.fully_connected),
        active_offsets_identified_by_data=None,
        offsets_identified_in_objective=bool(
            effective_graph.fully_connected
            or _component_offsets_selected_by_objective(model)
        ),
        gauge_policy=gauge_policy,
        messages=tuple(messages),
    )


def _build_active_set_connectivity_diagnostics(
    constraints: SeparatorObservations,
    active_mask: np.ndarray,
    *,
    model: FitModel,
    gauge_policy: str,
) -> ConnectivityDiagnostics:
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (constraints.n_constraints,):
        raise ValueError('active_mask must have shape (m,)')

    n = int(constraints.n_points)
    candidate_graph = _graph_diagnostics(
        n,
        constraints.i,
        constraints.j,
        n_constraints=constraints.n_constraints,
    )
    effective_mask = _informative_observation_mask(constraints)
    effective_graph = _graph_diagnostics(
        n,
        constraints.i[effective_mask],
        constraints.j[effective_mask],
        n_constraints=int(np.count_nonzero(effective_mask)),
    )

    active_constraints = constraints.subset(mask)
    active_graph = _graph_diagnostics(
        n,
        active_constraints.i,
        active_constraints.j,
        n_constraints=active_constraints.n_constraints,
    )
    active_effective_mask = _informative_observation_mask(active_constraints)
    active_effective_graph = _graph_diagnostics(
        n,
        active_constraints.i[active_effective_mask],
        active_constraints.j[active_effective_mask],
        n_constraints=int(np.count_nonzero(active_effective_mask)),
    )

    messages: list[str] = []
    if candidate_graph.isolated_points:
        messages.append(
            'candidate graph leaves unconstrained points '
            f'{_format_point_list(candidate_graph.isolated_points)}'
        )
    if candidate_graph.n_components > 1:
        messages.append(
            'candidate graph has ' f'{_format_component_counts(candidate_graph)}'
        )
    if np.any(~effective_mask):
        messages.append(
            'zero-confidence candidate rows are excluded from the informative '
            'observation graph and data-identification diagnostics; model '
            'restrictions and penalties are assessed separately'
        )
    if effective_graph.n_components > 1:
        messages.append(
            'candidate pairwise data identify only '
            f'{_format_component_counts(effective_graph)}; relative component '
            'offsets are not identified by the data'
        )
    if active_graph.n_components > 1:
        messages.append(
            'final active graph has ' f'{_format_component_counts(active_graph)}'
        )
    if np.any(mask) and np.any(~active_effective_mask):
        messages.append(
            'zero-confidence active rows are excluded from the active '
            'informative observation graph and data-identification '
            'diagnostics; model restrictions and penalties are assessed '
            'separately'
        )
    if active_effective_graph.n_components > 1:
        if _component_offsets_selected_by_objective(model):
            messages.append(
                'final active pairwise data identify only '
                f'{_format_component_counts(active_effective_graph)}; relative '
                'component offsets are selected by the objective rather than '
                'identified by the data'
            )
        else:
            messages.append(
                'final active pairwise data identify only '
                f'{_format_component_counts(active_effective_graph)}; relative '
                'component offsets are preserved by the self-consistent gauge '
                'policy rather than identified by the data'
            )

    return ConnectivityDiagnostics(
        unconstrained_points=candidate_graph.isolated_points,
        candidate_graph=candidate_graph,
        effective_graph=effective_graph,
        active_graph=active_graph,
        active_effective_graph=active_effective_graph,
        candidate_offsets_identified_by_data=bool(effective_graph.fully_connected),
        active_offsets_identified_by_data=bool(
            active_effective_graph.fully_connected
        ),
        offsets_identified_in_objective=bool(
            active_effective_graph.fully_connected
            or _component_offsets_selected_by_objective(model)
        ),
        gauge_policy=gauge_policy,
        messages=tuple(messages),
    )


def _hard_constraint_measurement_bounds(
    feasible: HardConstraint | None,
    n_constraints: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feasible is None:
        return None
    if isinstance(feasible, Interval):
        lower = np.full(n_constraints, float(feasible.lower), dtype=np.float64)
        upper = np.full(n_constraints, float(feasible.upper), dtype=np.float64)
        return lower, upper
    if isinstance(feasible, FixedValue):
        lower = np.full(n_constraints, float(feasible.value), dtype=np.float64)
        return lower, lower.copy()
    raise TypeError(f'unsupported hard constraint: {type(feasible)!r}')


def _hard_constraint_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accepted_lower, accepted_upper = _hard_accepted_measurement_bounds(
        lower,
        upper,
    )
    z_lo = _stable_ratio_difference(accepted_lower, beta, alpha)
    z_hi = _stable_ratio_difference(accepted_upper, beta, alpha)
    lo = np.minimum(z_lo, z_hi)
    hi = np.maximum(z_lo, z_hi)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _check_hard_feasibility(
    n: int,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
) -> tuple[bool, HardConstraintConflict | None]:
    edges: list[_DifferenceEdge] = []
    for k, (i, j, lo, hi) in enumerate(
        zip(i_idx.tolist(), j_idx.tolist(), z_lo.tolist(), z_hi.tolist())
    ):
        edges.append(
            _DifferenceEdge(
                source=int(j),
                target=int(i),
                weight=float(hi),
                constraint_index=int(k),
                site_i=int(i),
                site_j=int(j),
                relation='<=',
                bound_value=float(hi),
            )
        )
        edges.append(
            _DifferenceEdge(
                source=int(i),
                target=int(j),
                weight=float(-lo),
                constraint_index=int(k),
                site_i=int(i),
                site_j=int(j),
                relation='>=',
                bound_value=float(lo),
            )
        )

    dist = np.zeros(n, dtype=np.float64)
    pred_node = np.full(n, -1, dtype=np.int64)
    pred_edge = np.full(n, -1, dtype=np.int64)
    last_updated = -1

    for _ in range(n):
        updated = False
        last_updated = -1
        for edge_index, edge in enumerate(edges):
            cand = _stable_sum_scalar(dist[edge.source], edge.weight)
            if cand < dist[edge.target]:
                dist[edge.target] = cand
                pred_node[edge.target] = edge.source
                pred_edge[edge.target] = edge_index
                updated = True
                last_updated = edge.target
        if not updated:
            return True, None

    if last_updated < 0:
        return True, None

    y = int(last_updated)
    for _ in range(n):
        y = int(pred_node[y])
        if y < 0:
            return False, None

    cycle_edges_rev: list[_DifferenceEdge] = []
    cur = y
    while True:
        edge_index = int(pred_edge[cur])
        if edge_index < 0:
            return False, None
        edge = edges[edge_index]
        cycle_edges_rev.append(edge)
        cur = edge.source
        if cur == y:
            break

    cycle_edges = tuple(reversed(cycle_edges_rev))
    cycle_nodes_list: list[int] = []
    if cycle_edges:
        cycle_nodes_list.append(cycle_edges[0].source)
        cycle_nodes_list.extend(edge.target for edge in cycle_edges)
        if len(cycle_nodes_list) >= 2 and cycle_nodes_list[0] == cycle_nodes_list[-1]:
            cycle_nodes_list.pop()

    cycle_node_set = set(cycle_nodes_list)
    component_nodes: tuple[int, ...] = ()
    for comp in _connected_components(n, i_idx, j_idx):
        if any(node in cycle_node_set for node in comp):
            component_nodes = tuple(int(node) for node in comp)
            break

    terms = tuple(
        HardConstraintConflictTerm(
            constraint_index=edge.constraint_index,
            site_i=edge.site_i,
            site_j=edge.site_j,
            relation=edge.relation,
            bound_value=edge.bound_value,
        )
        for edge in cycle_edges
    )
    unique_constraints = tuple(sorted({term.constraint_index for term in terms}))
    component_label = (
        '[' + ', '.join(str(v) for v in component_nodes) + ']'
        if component_nodes
        else '[]'
    )
    cycle_label = '[' + ', '.join(str(v) for v in unique_constraints) + ']'
    conflict = HardConstraintConflict(
        component_nodes=component_nodes,
        cycle_nodes=tuple(int(v) for v in cycle_nodes_list),
        terms=terms,
        message=(
            'inconsistent hard separator restrictions on connected component '
            f'{component_label}; contradiction cycle uses constraint rows {cycle_label}'
        ),
    )
    return False, conflict


def _requires_admm(model: FitModel) -> bool:
    if model.feasible is not None:
        return True
    if _active_scalar_penalties(model.penalties):
        return True
    return not isinstance(model.mismatch, SquaredLoss)


def _mismatch_derivatives(
    y: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> tuple[np.ndarray, np.ndarray]:
    _, first, second = _objective_mismatch_terms(
        y,
        target,
        confidence,
        mismatch,
        evaluate_value=False,
    )
    return first, second


def _penalty_derivatives(
    y: np.ndarray,
    penalty: SoftIntervalPenalty
    | ExponentialBoundaryPenalty
    | ReciprocalBoundaryPenalty,
) -> tuple[np.ndarray, np.ndarray]:
    _, first, second = _penalty_terms(
        y,
        penalty,
        evaluate_value=False,
    )
    return first, second
