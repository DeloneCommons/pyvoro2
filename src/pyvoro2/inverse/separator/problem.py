"""Public power-fit problem construction, evaluation, and result packaging."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import InitVar, KW_ONLY, dataclass
from fractions import Fraction
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
    _stable_incidence_accumulate,
    _stable_mean_abs,
    _stable_product,
    _stable_ratio_difference,
    _stable_rms,
    _stable_scaled_difference,
    _stable_sum_products,
    _stable_sum_scalar,
)
from .constraints import SeparatorObservations
from ._diagnostics import (
    _affine_diagnostic, _affine_rms,
    _derived, _has_dependency, _max_diagnostic, _norm_diagnostic,
)
from ._identity import (
    _bind_originating_observations, _require_observation_association,
)
from ._policy import (
    _BoundPolicy, _PolicyBindingInit, _PolicyStorage, _bind_policy,
    _bind_result_policy, _expand, _row_model, _policy_getstate, _policy_setstate,
)
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
class SeparatorFitProblem(_PolicyStorage):
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
    _: KW_ONLY
    _bound_policy_init: InitVar[_BoundPolicy | None] = _PolicyBindingInit()
    # Python 3.10 typing rejects postponed dataclass pseudo-types.
    __annotations__['_'] = KW_ONLY
    __annotations__['_bound_policy_init'] = InitVar[_BoundPolicy | None]

    def __post_init__(self, _bound_policy_init) -> None:
        if _bound_policy_init is not None:
            _require_observation_association(
                _bound_policy_init.observations, self.constraints,
                context='problem resolved policy',
            )
        policy = _bind_policy(self.constraints, self.model)
        if _bound_policy_init is not None and _bound_policy_init.view != policy.view:
            raise ValueError('problem resolved policy does not match its model')
        object.__setattr__(self, '_bound_policy', policy)
        object.__setattr__(self, 'model', policy.model)
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

        For nonempty observations this view requires ``SquaredLoss`` without
        positive-strength scalar penalties. Empty observations have only the
        site regularizer. Zero-strength penalties are absent from the
        objective. Hard restrictions may coexist, but they
        remain in ``bounds`` and are not folded into the unconstrained normal
        equation.
        """

        has_rows = self.constraints.n_constraints > 0
        if has_rows and not isinstance(self.model.mismatch, SquaredLoss):
            raise ValueError(
                'quadratic_operator is available only for SquaredLoss models'
            )
        if has_rows and _active_scalar_penalties(self.model.penalties):
            raise ValueError(
                'quadratic_operator is unavailable when positive-strength '
                'scalar penalties are present because one fixed normal system '
                'does not represent the full objective'
            )

        graph = self.observation_graph
        rows = _quadratic_row_data(
            self.alpha,
            self.beta,
            self.mismatch_target,
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
            has_hard_constraints=bool(np.any(self.bounds.applicable)),
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

    return _build_power_fit_problem(constraints, model=model)


def _build_power_fit_problem(constraints, *, model=None, compile_hard=True):
    """Prepare full candidate predictions without enforcing unselected hard rows."""

    if model is None:
        model = FitModel()
    policy = _bind_policy(constraints, model)
    model = policy.model
    geom = _measurement_geometry(constraints, policy.mismatch_space)
    reg_ref = _regularization_reference(model.regularization, constraints.n_points)
    hard_measurement = _hard_constraint_measurement_bounds(
        model.feasible,
        constraints.n_constraints,
    )
    applicable = policy.applicable
    hard_diff = None
    if hard_measurement is not None:
        hard_diff = (np.full(constraints.n_constraints, np.nan),
                     np.full(constraints.n_constraints, np.nan))
        if compile_hard and np.any(applicable):
            hard_geom = _measurement_geometry(constraints, policy.hard_constraint_space)
            converted = _hard_constraint_bounds(
                hard_measurement[0][applicable], hard_measurement[1][applicable],
                hard_geom.alpha[applicable], hard_geom.beta[applicable],
            )
            hard_diff[0][applicable], hard_diff[1][applicable] = converted
    bounds = PowerFitBounds(
        measurement_lower=None if hard_measurement is None else hard_measurement[0],
        measurement_upper=None if hard_measurement is None else hard_measurement[1],
        difference_lower=None if hard_diff is None else hard_diff[0],
        difference_upper=None if hard_diff is None else hard_diff[1],
        space=policy.hard_constraint_space,
        applicable=applicable,
    )
    hard_feasible = True
    conflict = None
    if hard_diff is not None and compile_hard and np.any(applicable):
        hard_feasible, conflict = _check_hard_feasibility(
            int(constraints.n_points),
            constraints.i[applicable],
            constraints.j[applicable],
            hard_diff[0][applicable],
            hard_diff[1][applicable],
            row_indices=np.flatnonzero(applicable),
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
            _bound_policy_init=policy,
        )
    finally:
        _ALLOW_DERIVED_NONFINITE_PROBLEM_VALUES.reset(token)


SeparatorFitProblem.__getstate__ = _policy_getstate
SeparatorFitProblem.__setstate__ = _policy_setstate


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

    An optimal or converged request requires a finite soft objective and every
    applicable hard row to satisfy its authoritative per-row predicate on the
    exact returned representative, after any requested gauge selection.
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
    source_values = _source_diagnostic_values(problem.constraints, w)
    predictions = _predictions_from_diagnostics(source_values)
    residuals = source_values['residuals'].value
    edge_diagnostics = _compute_edge_diagnostics(
        problem.constraints,
        weights=w,
        predictions=predictions,
    )
    objective_breakdown = _objective_breakdown(problem, predictions, w)
    _require_final_success(objective_breakdown, status, converged_value)
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
    rms = source_values['rms_residual'].value
    mx = source_values['max_residual'].value
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
    _bind_originating_observations(result, problem.constraints)
    return _bind_result_policy(result, problem._bound_policy)


def _validated_weight_vector(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
) -> np.ndarray:
    return coerce_finite_vector(
        weights,
        name='weights',
        n=int(problem.constraints.n_points),
    )


def _measurement_geometry(
    constraints: SeparatorObservations, space: str | None = None,
) -> _MeasurementGeometry:
    d = constraints.distance
    d2 = constraints.distance2
    if (space or constraints.measurement) == 'fraction':
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


def _penalty_affines(policy):
    """Exact term affine maps in the mismatch coordinate, in local row order."""
    observations = policy.observations
    mismatch = _measurement_geometry(observations, policy.mismatch_space)
    result = [[] for _ in range(observations.n_constraints)]
    for term, space in zip(policy.model.penalties, policy.penalty_spaces):
        if space == policy.mismatch_space:
            maps = [(Fraction(1), Fraction(0))] * observations.n_constraints
        else:
            geometry = _measurement_geometry(observations, space)
            active = _expand(term.strength, observations.n_constraints) > 0.
            maps = []
            for index in range(observations.n_constraints):
                if not active[index]:
                    maps.append((Fraction(1), Fraction(0)))
                    continue
                scale = (Fraction.from_float(float(geometry.alpha[index])) /
                         Fraction.from_float(float(mismatch.alpha[index])))
                offset = (Fraction.from_float(float(geometry.beta[index])) -
                          scale * Fraction.from_float(float(mismatch.beta[index])))
                maps.append((scale, offset))
        for row, affine in zip(result, maps):
            row.append(affine)
    return tuple(tuple(row) for row in result)


def _hard_prox_bounds(problem):
    """Private accepted hard domains in mismatch units; absence stays private."""
    applicable = problem.bounds.applicable
    if not np.any(applicable):
        return None
    lower = np.full(len(applicable), -np.inf)
    upper = np.full(len(applicable), np.inf)
    accepted = _hard_accepted_measurement_bounds(
        problem.bounds.measurement_lower[applicable],
        problem.bounds.measurement_upper[applicable],
    )
    if problem.hard_constraint_space == problem.mismatch_space:
        lower[applicable], upper[applicable] = accepted
    else:
        hard = _measurement_geometry(problem.constraints,
                                     problem.hard_constraint_space)
        largest = Fraction.from_float(float(np.finfo(np.float64).max))
        for local, row in enumerate(np.flatnonzero(applicable)):
            operands = (problem.alpha[row], hard.alpha[row],
                        problem.beta[row], hard.beta[row])
            if (not np.all(np.isfinite(operands)) or
                    problem.alpha[row] <= 0. or hard.alpha[row] <= 0.):
                raise ValueError('mixed hard affine coefficients are not representable')
            # Map the accepted original hard endpoints directly. The public
            # difference arrays are rounded diagnostics, not affine operands.
            scale = (Fraction.from_float(float(problem.alpha[row])) /
                     Fraction.from_float(float(hard.alpha[row])))
            offset = (Fraction.from_float(float(problem.beta[row])) -
                      scale * Fraction.from_float(float(hard.beta[row])))
            exact_lower = (
                scale * Fraction.from_float(float(accepted[0][local])) + offset
            )
            exact_upper = (
                scale * Fraction.from_float(float(accepted[1][local])) + offset
            )
            if exact_lower > largest or exact_upper < -largest:
                raise ValueError(
                    'mixed hard domain has no representable mismatch value')
            # A domain extending past the finite lattice still admits every
            # finite value on that side. Saturate private endpoints only;
            # configured finite public bounds remain unchanged.
            exact_lower = max(exact_lower, -largest)
            exact_upper = min(exact_upper, largest)
            mapped_lower = float(exact_lower)
            mapped_upper = float(exact_upper)
            # Nearest rounding can enlarge the domain beyond the authoritative
            # hard predicate. Choose the adjacent inward endpoint instead.
            if Fraction.from_float(mapped_lower) < exact_lower:
                mapped_lower = np.nextafter(mapped_lower, np.inf)
            if Fraction.from_float(mapped_upper) > exact_upper:
                mapped_upper = np.nextafter(mapped_upper, -np.inf)
            if mapped_lower > mapped_upper:
                raise ValueError(
                    'mixed hard domain has no representable mismatch value')
            lower[row], upper[row] = mapped_lower, mapped_upper
    return lower, upper


def _predict_all(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
) -> PowerFitPredictions:
    return _predictions_from_diagnostics(
        _source_diagnostic_values(problem.constraints, weights),
    )


def _predictions_from_diagnostics(values):
    return PowerFitPredictions(
        difference=values['difference'].value,
        fraction=values['predicted_fraction'].value,
        position=values['predicted_position'].value,
        measurement=values['predicted'].value,
    )


def _residual_diagnostic(constraints, weights, *, space=None, active=None):
    geom = _measurement_geometry(constraints, space)
    scales = () if active is None else (np.asarray(active, dtype=float),)
    return _affine_diagnostic(
        geom.beta, geom.alpha, weights[constraints.i], weights[constraints.j],
        geom.target, *scales,
    )


def _source_diagnostic_values(constraints, weights):
    """Evaluate complete source rows, predictions and reductions together."""
    left, right = weights[constraints.i], weights[constraints.j]
    fraction_geom = _measurement_geometry(constraints, 'fraction')
    position_geom = _measurement_geometry(constraints, 'position')
    fraction = _affine_diagnostic(
        fraction_geom.beta, fraction_geom.alpha, left, right, 0.,
    )
    position = _affine_diagnostic(
        position_geom.beta, position_geom.alpha, left, right, 0.,
    )
    geom = (fraction_geom if constraints.measurement == 'fraction'
            else position_geom)
    residuals = _affine_diagnostic(
        geom.beta, geom.alpha, left, right, geom.target,
    )
    return {
        'difference': _derived(_stable_scaled_difference(left, right, 1.)),
        'predicted_fraction': fraction,
        'predicted_position': position,
        'predicted': (fraction if constraints.measurement == 'fraction'
                      else position),
        'residuals': residuals,
        'rms_residual': _affine_rms(
            geom.beta, geom.alpha, left, right, geom.target, residuals,
        ),
        'max_residual': _max_diagnostic(residuals),
    }


def _measurement_residuals(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
    *,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Return direct affine residuals without materializing predictions."""

    return _residual_diagnostic(
        problem.constraints, weights, active=active,
    ).value


def _algebraic_residual_diagnostic(geom, left, right, z_obs, z_fit):
    normal = np.isfinite(z_obs.value) & np.isfinite(z_fit.value)
    values = np.zeros(normal.shape, dtype=float)
    values[normal] = _stable_scaled_difference(
        z_obs.value[normal], z_fit.value[normal], 1.,
    )
    available = normal.copy()
    recover = ~normal & np.isfinite(geom.alpha)
    large_alpha = recover & (geom.alpha >= 1.)
    if np.any(large_alpha):
        evaluated = _affine_diagnostic(
            geom.beta[large_alpha], geom.alpha[large_alpha],
            left[large_alpha], right[large_alpha], geom.target[large_alpha],
            -_stable_ratio_difference(1., 0., geom.alpha[large_alpha]),
        )
        values[large_alpha] = evaluated.value
        available[large_alpha] = True
    small_alpha = recover & ~large_alpha
    if np.any(small_alpha):
        residual = _affine_diagnostic(
            geom.beta[small_alpha], geom.alpha[small_alpha],
            left[small_alpha], right[small_alpha], geom.target[small_alpha],
        )
        # Dividing a proven outside-range row by 0 < alpha < 1 cannot
        # restore representability. Finite rows use the existing ratio owner.
        values[small_alpha] = _stable_ratio_difference(
            0., residual.value, geom.alpha[small_alpha],
        )
        available[small_alpha] = True
    return _derived(values, operands_available=available)


def _algebraic_reduction(geom, left, right, z_obs, z_fit, rows, *, mean_abs):
    if _has_dependency(rows):
        return _derived(np.nan, operands_available=False)
    if np.all(np.isfinite(rows.value)):
        reduce = _stable_mean_abs if mean_abs else _stable_rms
        return _derived(reduce(rows.value))
    count = rows.value.size
    scale = 1./count if mean_abs else 1./np.sqrt(count)
    normal = np.isfinite(z_obs.value) & np.isfinite(z_fit.value)
    values = np.zeros(normal.shape, dtype=float)
    values[normal] = _stable_scaled_difference(
        z_obs.value[normal], z_fit.value[normal], scale,
    )
    recover = ~normal & np.isfinite(geom.alpha) & (geom.alpha != 0.)
    available = normal | recover
    if np.any(recover):
        # Keep the reciprocal as normal factors, including subnormal alpha.
        # Materializing 1/alpha can overflow; scale/alpha can underflow before
        # the complete affine products restore a representable aggregate row.
        part, exponent = np.frexp(geom.alpha[recover])
        half_exponent = exponent // 2
        values[recover] = _affine_diagnostic(
            geom.beta[recover], geom.alpha[recover], left[recover],
            right[recover], geom.target[recover],
            -1./part, np.ldexp(1., -half_exponent),
            np.ldexp(1., half_exponent - exponent), scale,
        ).value
    scaled = _derived(values, operands_available=available)
    if _has_dependency(scaled):
        return _derived(np.nan, operands_available=False)
    if mean_abs:
        return _derived(_stable_sum_scalar(*np.abs(scaled.value)))
    return _norm_diagnostic(scaled)


def _edge_diagnostic_values(constraints, weights, *, geom=None):
    geom = _measurement_geometry(constraints) if geom is None else geom
    alpha, beta, target = geom.alpha, geom.beta, geom.target
    coefficient_available = np.isfinite(alpha) & np.isfinite(beta)
    confidence = np.asarray(constraints.confidence, dtype=float)
    absent = confidence == 0.
    work = ~absent & coefficient_available
    rho = np.zeros(alpha.shape, dtype=float)
    rho[work] = _stable_product(confidence[work], alpha[work], alpha[work])
    values = {
        'alpha': _derived(alpha),
        'beta': _derived(beta),
        'z_obs': _derived(
            _stable_ratio_difference(target, beta, alpha),
            operands_available=coefficient_available,
        ),
        'edge_weight': _derived(
            rho, operands_available=coefficient_available | absent,
        ),
    }
    if weights is None:
        values.update({name: _derived(None) for name in (
            'z_fit', 'residual', 'weighted_l2', 'weighted_rmse', 'rmse', 'mae',
        )})
        return values
    left, right = weights[constraints.i], weights[constraints.j]
    z_fit = _derived(_stable_scaled_difference(left, right, 1.))
    residual = _algebraic_residual_diagnostic(
        geom, left, right, values['z_obs'], z_fit,
    )
    weighted = _affine_diagnostic(
        beta, alpha, left, right, target, np.sqrt(confidence),
    )
    values.update({
        'z_fit': z_fit,
        'residual': residual,
        'weighted_l2': _norm_diagnostic(weighted),
        'weighted_rmse': _affine_rms(
            beta, alpha, left, right, target, weighted, np.sqrt(confidence),
        ),
        'rmse': _algebraic_reduction(
            geom, left, right, values['z_obs'], z_fit, residual, mean_abs=False,
        ),
        'mae': _algebraic_reduction(
            geom, left, right, values['z_obs'], z_fit, residual, mean_abs=True,
        ),
    })
    return values


def _compute_edge_diagnostics(
    constraints: SeparatorObservations,
    *,
    weights: np.ndarray | None,
    predictions: PowerFitPredictions | None = None,
    geom: _MeasurementGeometry | None = None,
) -> AlgebraicEdgeDiagnostics:
    values = _edge_diagnostic_values(constraints, weights, geom=geom)
    return AlgebraicEdgeDiagnostics(**{
        name: diagnostic.value for name, diagnostic in values.items()
    })


def _fit_diagnostic_values(result, constraints):
    """Recompute availability from bound rows/policy and exact final weights."""
    result.observation_view(constraints)
    policy = result._require_policy()
    _require_observation_association(
        policy.observations, constraints, context='fit diagnostic policy',
    )
    if result.weights is None:
        if result.status == 'optimal' or result.converged:
            raise ValueError('a successful final result requires finite weights')
        source = {name: _derived(None) for name in (
            'predicted', 'predicted_fraction', 'predicted_position', 'residuals',
            'rms_residual', 'max_residual',
        )}
        mismatch_predicted = mismatch_residual = _derived(None)
        weights = None
    else:
        weights = coerce_finite_vector(
            result.weights, name='fit result weights', n=constraints.n_points,
        )
        source = _source_diagnostic_values(constraints, weights)
        if result.status == 'optimal' or result.converged:
            # Replacements and other supported reconstructions must not turn
            # an honest unsuccessful candidate into a false success claim.
            # Only this fit's selected policy is authoritative here. Hard
            # compilation/precheck is unnecessary for its per-row predicate.
            problem = _build_power_fit_problem(
                constraints, model=policy.model, compile_hard=False,
            )
            breakdown = _objective_breakdown(
                problem, _predictions_from_diagnostics(source), weights,
            )
            _require_final_success(breakdown, result.status, result.converged)
            if result.objective_breakdown is not None:
                _require_final_success(
                    result.objective_breakdown, result.status, result.converged,
                )
        geom = _measurement_geometry(constraints, policy.mismatch_space)
        mismatch_predicted = _affine_diagnostic(
            geom.beta, geom.alpha, weights[constraints.i],
            weights[constraints.j], 0.,
        )
        mismatch_residual = _residual_diagnostic(
            constraints, weights, space=policy.mismatch_space,
        )
    for name in (
        'predicted', 'predicted_fraction', 'predicted_position', 'residuals',
        'rms_residual', 'max_residual',
    ):
        source[name].require(getattr(result, name), context='fit ' + name)
    mismatch_predicted.require(result.mismatch_predicted,
                               context='fit mismatch prediction')
    mismatch_residual.require(result.mismatch_residuals,
                              context='fit mismatch residual')
    edge = _edge_diagnostic_values(constraints, weights)
    if result.edge_diagnostics is not None:
        for name, expected in edge.items():
            expected.require(getattr(result.edge_diagnostics, name),
                             context='edge diagnostic ' + name)
    record = {
        'predicted': source['predicted'],
        'predicted_fraction': source['predicted_fraction'],
        'predicted_position': source['predicted_position'],
        'residual': source['residuals'],
        'mismatch_predicted': mismatch_predicted,
        'mismatch_residual': mismatch_residual,
        **{name: edge[name] for name in (
            'alpha', 'beta', 'z_obs', 'z_fit', 'edge_weight',
        )},
        'algebraic_residual': edge['residual'],
    }
    return {'record': record, 'edge': edge, 'summary': {
        name: source[name] for name in ('rms_residual', 'max_residual')
    }}


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
    applicable = problem.bounds.applicable
    if not np.any(applicable):
        return True, 0.0, 0.0
    y = (predictions.fraction if problem.hard_constraint_space == 'fraction'
         else predictions.position)
    satisfied, violation, tolerance = _hard_row_status(
        lower[applicable], y[applicable], upper[applicable],
    )
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
    left = weights[problem.constraints.i]
    right = weights[problem.constraints.j]
    mismatch_values = _mismatch_values_from_affine(
        problem.beta,
        problem.alpha,
        left,
        right,
        problem.mismatch_target,
        confidence,
        problem.model.mismatch,
    )
    mismatch = _stable_sum_scalar(*mismatch_values.tolist())
    penalty_terms_list: list[tuple[str, float]] = []
    penalties_total = 0.0
    for term_index, penalty in enumerate(problem.model.penalties):
        active = _expand(penalty.strength, problem.constraints.n_constraints) > 0.
        value = 0.
        if np.any(active):
            space = problem.penalty_spaces[term_index]
            geometry = _measurement_geometry(problem.constraints, space)
            measurement = (predictions.fraction if space == 'fraction'
                           else predictions.position)
            # Row templates are scalarized at the owned policy seam. No absent
            # lane reaches reciprocal or exponential arithmetic.
            if any(isinstance(getattr(penalty, name), np.ndarray)
                   for name in ('lower', 'upper', 'strength')):
                values = [
                    _penalty_value_from_affine(
                        geometry.beta[k:k+1], geometry.alpha[k:k+1],
                        left[k:k+1], right[k:k+1], measurement[k:k+1],
                        _row_model(problem.model, k).penalties[term_index],
                    )[0] for k in np.flatnonzero(active)
                ]
            else:
                values = _penalty_value_from_affine(
                    geometry.beta[active], geometry.alpha[active],
                    left[active], right[active], measurement[active], penalty,
                ).tolist()
            value = _stable_sum_scalar(*values)
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


def _require_final_success(breakdown, status, converged):
    """Apply the shared success predicate to a final or reconstructed state."""
    if status != 'optimal' and not converged:
        return
    if not _soft_objective_is_finite(breakdown):
        raise _NonFiniteOptimalObjectiveError(
            'cannot accept an optimal or converged result with a non-finite '
            'soft objective'
        )
    if not breakdown.hard_constraints_satisfied:
        raise ValueError(
            'cannot accept an optimal or converged result whose final weights '
            'violate hard measurement bounds'
        )


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
    if model.feasible is not None:
        mask = mask | _expand(model.feasible.applicable, constraints.n_constraints,
                              dtype=bool)
    for penalty in model.penalties:
        mask = mask | (_expand(penalty.strength, constraints.n_constraints) > 0.)
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
        if idx.size == 1:
            aligned[idx[0]] = 0.0 if ref is None else ref[idx[0]]
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
        lower = _expand(feasible.lower, n_constraints)
        upper = _expand(feasible.upper, n_constraints)
        return lower, upper
    if isinstance(feasible, FixedValue):
        lower = _expand(feasible.value, n_constraints)
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
    *,
    row_indices: np.ndarray | None = None,
) -> tuple[bool, HardConstraintConflict | None]:
    edges: list[_DifferenceEdge] = []
    if row_indices is None:
        row_indices = np.arange(len(i_idx))
    for k, (i, j, lo, hi) in enumerate(
        zip(i_idx.tolist(), j_idx.tolist(), z_lo.tolist(), z_hi.tolist())
    ):
        edges.append(
            _DifferenceEdge(
                source=int(j),
                target=int(i),
                weight=float(hi),
                constraint_index=int(row_indices[k]),
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
                constraint_index=int(row_indices[k]),
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


def _requires_admm(model: FitModel, *, n_rows: int | None = None) -> bool:
    if n_rows == 0:
        return False
    if model.feasible is not None and np.any(model.feasible.applicable):
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
