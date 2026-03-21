"""Public power-fit problem construction, evaluation, and result packaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constraints import PairBisectorConstraints
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
from .transforms import weights_to_radii
from .types import (
    AlgebraicEdgeDiagnostics,
    ConnectivityDiagnostics,
    ConstraintGraphDiagnostics,
    HardConstraintConflict,
    HardConstraintConflictTerm,
    PowerFitBounds,
    PowerFitObjectiveBreakdown,
    PowerFitPredictions,
    PowerWeightFitResult,
    _readonly_array,
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
class PowerFitProblem:
    """Stable public export of the resolved inverse power-fit problem."""

    constraints: PairBisectorConstraints
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
        object.__setattr__(self, 'alpha', _readonly_array(self.alpha, dtype=np.float64))
        object.__setattr__(self, 'beta', _readonly_array(self.beta, dtype=np.float64))
        object.__setattr__(self, 'z_obs', _readonly_array(self.z_obs, dtype=np.float64))
        object.__setattr__(
            self,
            'edge_weight',
            _readonly_array(self.edge_weight, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'regularization_reference',
            _readonly_array(self.regularization_reference, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'offset_identifying_constraint_mask',
            _readonly_array(self.offset_identifying_constraint_mask, dtype=bool),
        )
        if self.alpha.shape != (m,):
            raise ValueError('alpha must have shape (m,)')
        if self.beta.shape != (m,):
            raise ValueError('beta must have shape (m,)')
        if self.z_obs.shape != (m,):
            raise ValueError('z_obs must have shape (m,)')
        if self.edge_weight.shape != (m,):
            raise ValueError('edge_weight must have shape (m,)')
        if self.regularization_reference.shape != (n,):
            raise ValueError('regularization_reference must have shape (n_points,)')
        if self.offset_identifying_constraint_mask.shape != (m,):
            raise ValueError(
                'offset_identifying_constraint_mask must have shape (m,)'
            )

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
    def suggested_anchor_indices(self) -> tuple[int, ...]:
        if self.connectivity.offsets_identified_in_objective:
            return tuple()
        return tuple(
            int(component[0])
            for component in self.connectivity.effective_graph.connected_components
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
        if self.connectivity.offsets_identified_in_objective:
            return w.copy()
        comps = [
            list(component)
            for component in self.connectivity.effective_graph.connected_components
        ]
        return _apply_component_mean_gauge(
            w,
            comps,
            reference=(
                None
                if self.model.regularization.reference is None
                else self.regularization_reference
            ),
        )


def build_power_fit_problem(
    constraints: PairBisectorConstraints,
    *,
    model: FitModel | None = None,
) -> PowerFitProblem:
    """Build a stable public power-fit problem from resolved constraints."""

    if model is None:
        model = FitModel()
    geom = _measurement_geometry(constraints)
    reg_ref = _regularization_reference(model.regularization, constraints.n_points)
    hard_diff = _hard_constraint_bounds(model.feasible, geom.alpha, geom.beta)
    hard_measurement = _hard_constraint_measurement_bounds(
        model.feasible,
        constraints.n_constraints,
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
        gauge_policy=_standalone_gauge_policy_description(model.regularization),
    )
    alpha = np.asarray(geom.alpha, dtype=np.float64)
    beta = np.asarray(geom.beta, dtype=np.float64)
    target = np.asarray(geom.target, dtype=np.float64)
    z_obs = (target - beta) / alpha
    edge_weight = np.asarray(constraints.confidence, dtype=np.float64) * (alpha * alpha)
    return PowerFitProblem(
        constraints=constraints,
        model=model,
        alpha=alpha,
        beta=beta,
        z_obs=z_obs,
        edge_weight=edge_weight,
        regularization_strength=float(model.regularization.strength),
        regularization_reference=reg_ref,
        offset_identifying_constraint_mask=_offset_identifying_constraint_mask(
            constraints,
            model,
        ),
        bounds=bounds,
        connectivity=connectivity,
        hard_feasible=bool(hard_feasible),
        hard_conflict=conflict,
    )


def build_power_fit_result(
    problem: PowerFitProblem,
    weights: np.ndarray,
    *,
    solver: str = 'external',
    status: str = 'optimal',
    status_detail: str | None = None,
    converged: bool = True,
    n_iter: int = 0,
    warnings: tuple[str, ...] = (),
    canonicalize_gauge: bool = True,
    r_min: float = 0.0,
    weight_shift: float | None = None,
) -> PowerWeightFitResult:
    """Package candidate weights into a standard power-fit result object."""

    w = _validated_weight_vector(problem, weights)
    if canonicalize_gauge:
        w = problem.canonicalize_gauge(w)
    predictions = _predict_all(problem, w)
    residuals = np.asarray(
        predictions.measurement - problem.measurement_target,
        dtype=np.float64,
    )
    edge_diagnostics = _compute_edge_diagnostics(
        problem.constraints,
        weights=w,
        predictions=predictions,
    )
    objective_breakdown = _objective_breakdown(problem, predictions, w)
    warnings_list = list(warnings)
    if not objective_breakdown.hard_constraints_satisfied:
        warnings_list.append(
            'candidate weights violate hard measurement bounds'
        )
    radii, shift = weights_to_radii(
        w,
        r_min=r_min,
        weight_shift=weight_shift,
    )
    rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    mx = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    return PowerWeightFitResult(
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
        n_iter=int(n_iter),
        converged=bool(converged),
        conflict=problem.hard_conflict,
        warnings=tuple(warnings_list),
        connectivity=problem.connectivity,
        edge_diagnostics=edge_diagnostics,
        objective_breakdown=objective_breakdown,
    )


def _validated_weight_vector(
    problem: PowerFitProblem,
    weights: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape != (int(problem.constraints.n_points),):
        raise ValueError('weights must have shape (n_points,)')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must contain only finite values')
    return np.asarray(w, dtype=np.float64)


def _measurement_geometry(constraints: PairBisectorConstraints) -> _MeasurementGeometry:
    d = constraints.distance
    d2 = constraints.distance2
    if constraints.measurement == 'fraction':
        alpha = 1.0 / (2.0 * d2)
        beta = np.full_like(alpha, 0.5)
        target = constraints.target_fraction
    else:
        alpha = 1.0 / (2.0 * d)
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
    problem: PowerFitProblem,
    weights: np.ndarray,
) -> PowerFitPredictions:
    z_pred = weights[problem.constraints.i] - weights[problem.constraints.j]
    fraction = 0.5 + z_pred / (2.0 * problem.constraints.distance2)
    position = problem.constraints.distance * fraction
    measurement = (
        fraction if problem.constraints.measurement == 'fraction' else position
    )
    return PowerFitPredictions(
        difference=np.asarray(z_pred, dtype=np.float64),
        fraction=np.asarray(fraction, dtype=np.float64),
        position=np.asarray(position, dtype=np.float64),
        measurement=np.asarray(measurement, dtype=np.float64),
    )


def _compute_edge_diagnostics(
    constraints: PairBisectorConstraints,
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
    z_obs = (target - beta) / alpha
    edge_weight = np.asarray(constraints.confidence, dtype=np.float64) * (alpha * alpha)
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
        z_fit = np.asarray(
            weights[constraints.i] - weights[constraints.j],
            dtype=np.float64,
        )
    else:
        z_fit = np.asarray(predictions.difference, dtype=np.float64)
    residual = z_obs - z_fit
    weighted_sq = edge_weight * residual * residual
    if residual.size:
        weighted_l2 = float(np.linalg.norm(np.sqrt(edge_weight) * residual))
        weighted_rmse = float(np.sqrt(np.mean(weighted_sq)))
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
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
    result: PowerWeightFitResult,
    constraints: PairBisectorConstraints,
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
    residual = measurement - target
    if isinstance(mismatch, SquaredLoss):
        return confidence * residual * residual
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        abs_r = np.abs(residual)
        return confidence * np.where(
            abs_r <= delta,
            0.5 * residual * residual,
            delta * (abs_r - 0.5 * delta),
        )
    raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')


def _penalty_values(
    measurement: np.ndarray,
    penalty: SoftIntervalPenalty
    | ExponentialBoundaryPenalty
    | ReciprocalBoundaryPenalty,
) -> np.ndarray:
    y = np.asarray(measurement, dtype=np.float64)
    if isinstance(penalty, SoftIntervalPenalty):
        out = np.zeros_like(y)
        lo_mask = y < float(penalty.lower)
        hi_mask = y > float(penalty.upper)
        if np.any(lo_mask):
            out[lo_mask] = float(penalty.strength) * (
                y[lo_mask] - float(penalty.lower)
            ) ** 2
        if np.any(hi_mask):
            out[hi_mask] = float(penalty.strength) * (
                y[hi_mask] - float(penalty.upper)
            ) ** 2
        return out
    if isinstance(penalty, ExponentialBoundaryPenalty):
        left = float(penalty.lower) + float(penalty.margin)
        right = float(penalty.upper) - float(penalty.margin)
        tau = float(penalty.tau)
        strength = float(penalty.strength)
        A = np.exp((left - y) / tau)
        B = np.exp((y - right) / tau)
        return strength * (A + B)
    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        left = lower + float(penalty.margin)
        right = upper - float(penalty.margin)
        eps = float(penalty.epsilon)
        strength = float(penalty.strength)
        out = np.zeros_like(y)
        lo_mask = y < left
        hi_mask = y > right
        if np.any(lo_mask):
            denom = np.maximum(y[lo_mask] - lower, eps)
            base = max(left - lower, eps)
            out[lo_mask] = strength * ((1.0 / denom) - (1.0 / base))
        if np.any(hi_mask):
            denom = np.maximum(upper - y[hi_mask], eps)
            base = max(upper - right, eps)
            out[hi_mask] = strength * ((1.0 / denom) - (1.0 / base))
        return out
    raise TypeError(f'unsupported penalty: {type(penalty)!r}')


def _hard_constraint_status(
    problem: PowerFitProblem,
    predictions: PowerFitPredictions,
) -> tuple[bool, float]:
    lower = problem.bounds.measurement_lower
    upper = problem.bounds.measurement_upper
    if lower is None or upper is None:
        return True, 0.0
    y = np.asarray(predictions.measurement, dtype=np.float64)
    lo_violation = np.maximum(lower - y, 0.0)
    hi_violation = np.maximum(y - upper, 0.0)
    violation = np.maximum(lo_violation, hi_violation)
    if violation.size == 0:
        return True, 0.0
    max_violation = float(np.max(violation))
    return max_violation <= 1e-12, max_violation


def _objective_breakdown(
    problem: PowerFitProblem,
    predictions: PowerFitPredictions,
    weights: np.ndarray,
) -> PowerFitObjectiveBreakdown:
    target = np.asarray(problem.measurement_target, dtype=np.float64)
    confidence = np.asarray(problem.constraints.confidence, dtype=np.float64)
    measurement = np.asarray(predictions.measurement, dtype=np.float64)
    mismatch = float(
        np.sum(
            _mismatch_values(
                measurement,
                target,
                confidence,
                problem.model.mismatch,
            )
        )
    )
    penalty_terms_list: list[tuple[str, float]] = []
    penalties_total = 0.0
    for penalty in problem.model.penalties:
        value = float(np.sum(_penalty_values(measurement, penalty)))
        penalty_terms_list.append((type(penalty).__name__, value))
        penalties_total += value
    reg = problem.regularization_strength * float(
        np.sum((weights - problem.regularization_reference) ** 2)
    )
    hard_satisfied, hard_max_violation = _hard_constraint_status(problem, predictions)
    total = mismatch + penalties_total + reg
    return PowerFitObjectiveBreakdown(
        total=float(total),
        mismatch=float(mismatch),
        penalties_total=float(penalties_total),
        penalty_terms=tuple(penalty_terms_list),
        regularization=float(reg),
        hard_constraints_satisfied=bool(hard_satisfied),
        hard_max_violation=float(hard_max_violation),
    )


def _regularization_reference(reg: L2Regularization, n: int) -> np.ndarray:
    if reg.reference is None:
        return np.zeros(n, dtype=np.float64)
    w0 = np.asarray(reg.reference, dtype=float)
    if w0.shape != (n,):
        raise ValueError('regularization.reference must have shape (n,)')
    return np.asarray(w0, dtype=np.float64)


def _offset_identifying_constraint_mask(
    constraints: PairBisectorConstraints,
    model: FitModel,
) -> np.ndarray:
    mask = np.asarray(constraints.confidence > 0.0, dtype=bool)
    if model.feasible is not None or len(model.penalties) > 0:
        mask = np.ones(constraints.n_constraints, dtype=bool)
    return mask


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


def _standalone_gauge_policy_description(reg: L2Regularization) -> str:
    if reg.reference is not None:
        return (
            'each effective component is shifted so its mean matches the '
            'reference mean on that component'
        )
    return 'each effective component is centered to mean zero'


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
    constraints: PairBisectorConstraints,
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
    effective_mask = _offset_identifying_constraint_mask(constraints, model)
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
            'zero-confidence candidate rows do not identify pair differences '
            'in the current objective and are ignored for '
            'connectivity/gauge diagnostics'
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
            or float(model.regularization.strength) > 0.0
        ),
        gauge_policy=gauge_policy,
        messages=tuple(messages),
    )


def _build_active_set_connectivity_diagnostics(
    constraints: PairBisectorConstraints,
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
    effective_mask = _offset_identifying_constraint_mask(constraints, model)
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
    active_effective_mask = _offset_identifying_constraint_mask(
        active_constraints,
        model,
    )
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
            'zero-confidence candidate rows do not identify pair differences '
            'in the current objective and are ignored for '
            'connectivity/gauge diagnostics'
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
            'zero-confidence active rows do not identify pair differences in '
            'the current objective and are ignored for active-component gauge '
            'alignment'
        )
    if active_effective_graph.n_components > 1:
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
            or float(model.regularization.strength) > 0.0
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
    feasible: HardConstraint | None,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feasible is None:
        return None
    if isinstance(feasible, Interval):
        lower = np.full_like(alpha, float(feasible.lower))
        upper = np.full_like(alpha, float(feasible.upper))
    elif isinstance(feasible, FixedValue):
        lower = np.full_like(alpha, float(feasible.value))
        upper = lower.copy()
    else:
        raise TypeError(f'unsupported hard constraint: {type(feasible)!r}')
    z_lo = (lower - beta) / alpha
    z_hi = (upper - beta) / alpha
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
    tol = 1e-12

    for _ in range(n):
        updated = False
        last_updated = -1
        for edge_index, edge in enumerate(edges):
            cand = dist[edge.source] + edge.weight
            if cand < dist[edge.target] - tol:
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
    if model.penalties:
        return True
    return not isinstance(model.mismatch, SquaredLoss)


def _mismatch_derivatives(
    y: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
) -> tuple[np.ndarray, np.ndarray]:
    residual = y - target
    if isinstance(mismatch, SquaredLoss):
        fp_y = 2.0 * confidence * residual
        fpp_y = 2.0 * confidence
        return fp_y, fpp_y
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        abs_r = np.abs(residual)
        quad = abs_r <= delta
        fp_y = np.where(
            quad,
            confidence * residual,
            confidence * delta * np.sign(residual),
        )
        fpp_y = np.where(quad, confidence, 0.0)
        return fp_y, fpp_y
    raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')


def _penalty_derivatives(
    y: np.ndarray,
    penalty: SoftIntervalPenalty
    | ExponentialBoundaryPenalty
    | ReciprocalBoundaryPenalty,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(penalty, SoftIntervalPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        strength = float(penalty.strength)
        fp = np.zeros_like(y)
        fpp = np.zeros_like(y)
        lo_mask = y < lower
        hi_mask = y > upper
        if np.any(lo_mask):
            fp[lo_mask] += 2.0 * strength * (y[lo_mask] - lower)
            fpp[lo_mask] += 2.0 * strength
        if np.any(hi_mask):
            fp[hi_mask] += 2.0 * strength * (y[hi_mask] - upper)
            fpp[hi_mask] += 2.0 * strength
        return fp, fpp

    if isinstance(penalty, ExponentialBoundaryPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        margin = float(penalty.margin)
        strength = float(penalty.strength)
        tau = float(penalty.tau)
        left = lower + margin
        right = upper - margin
        A = np.exp((left - y) / tau)
        B = np.exp((y - right) / tau)
        fp = strength * (-A + B) / tau
        fpp = strength * (A + B) / (tau * tau)
        return fp, fpp

    if isinstance(penalty, ReciprocalBoundaryPenalty):
        lower = float(penalty.lower)
        upper = float(penalty.upper)
        margin = float(penalty.margin)
        strength = float(penalty.strength)
        eps = float(penalty.epsilon)
        left = lower + margin
        right = upper - margin
        fp = np.zeros_like(y)
        fpp = np.zeros_like(y)
        lo_mask = y < left
        if np.any(lo_mask):
            denom = np.maximum(y[lo_mask] - lower, eps)
            fp[lo_mask] += -strength / (denom**2)
            fpp[lo_mask] += 2.0 * strength / (denom**3)
        hi_mask = y > right
        if np.any(hi_mask):
            denom = np.maximum(upper - y[hi_mask], eps)
            fp[hi_mask] += strength / (denom**2)
            fpp[hi_mask] += 2.0 * strength / (denom**3)
        return fp, fpp

    raise TypeError(f'unsupported penalty: {type(penalty)!r}')
