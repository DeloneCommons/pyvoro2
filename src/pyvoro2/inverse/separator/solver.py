"""Native numerical solvers for fitting power weights."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Sequence

import numpy as np

from ._objective import (
    _active_scalar_penalties,
    _hard_accepted_measurement_bounds,
    _hard_row_status,
    _l2_value,
    _mismatch_values_from_affine,
)
from ._numerics import (
    _stable_affine_difference,
    _stable_affine_residual,
    _stable_incidence_accumulate,
    _stable_norm,
    _stable_product,
    _stable_product_scalar,
    _stable_ratio_product,
    _stable_scaled_difference,
    _stable_sum,
    _stable_sum_products_sign,
    _stable_sum_scalar,
    _stable_weighted_average,
)
from ._scalar_prox import (
    _batch_spec_supported,
    _compile_scalar_prox_spec,
    _penalties_inactive_at,
    _ScalarProxError,
    _ScalarProxSpec,
    _solve_scalar_prox_batch_ordinary,
    _solve_scalar_prox_coordinate,
)
from ._quadratic import (
    QuadraticNumericalError,
    QuadraticWeightSystem,
    _require_scipy_sparse,
    _shift_to_required_mean,
    certify_quadratic_component_candidate,
    solve_quadratic_component,
)
from .constraints import (
    DomainAny,
    SeparatorObservations,
    resolve_separator_observations,
)
from .model import FitModel, HuberLoss, SquaredLoss
from .problem import (
    _compute_edge_diagnostics,
    _connected_components,
    _NonFiniteOptimalObjectiveError,
    _requires_admm,
    _soft_objective_is_finite,
    SeparatorFitProblem,
    build_power_fit_problem,
    build_power_fit_result,
)
from .types import (
    ConnectivityDiagnostics,
    SeparatorFitResult,
    _bind_originating_observations,
)


class ConnectivityDiagnosticsError(ValueError):
    """Raised when connectivity_check='raise' detects a graph issue."""

    def __init__(
        self,
        message: str,
        diagnostics: ConnectivityDiagnostics,
    ) -> None:
        super().__init__(message, diagnostics)
        self.diagnostics = diagnostics

    def __str__(self) -> str:
        return str(self.args[0])


class _NumericalFailure(RuntimeError):
    """Raised when the numerical backend fails before producing a result."""


class _IterativeNumericalFailure(_NumericalFailure):
    """Carry completed ADMM work through structured numerical failure."""

    def __init__(self, message: str, *, n_iter: int) -> None:
        super().__init__(message)
        self.n_iter = int(n_iter)


def _numerical_failure_result(
    problem: SeparatorFitProblem,
    constraints: SeparatorObservations,
    *,
    solver: str,
    linear_backend: str | None,
    n_iter: int,
    warnings_list: list[str],
    connectivity: ConnectivityDiagnostics | None,
    error: Exception,
) -> SeparatorFitResult:
    """Build the existing structured result for a numerical solver failure."""

    warnings_list.append(f'numerical solver failure: {error}')
    result = SeparatorFitResult(
        status='numerical_failure',
        status_detail=str(error),
        hard_feasible=True,
        weights=None,
        radii=None,
        weight_shift=None,
        measurement=constraints.measurement,
        target=np.asarray(problem.measurement_target, dtype=np.float64),
        predicted=None,
        predicted_fraction=None,
        predicted_position=None,
        residuals=None,
        rms_residual=None,
        max_residual=None,
        used_shifts=np.asarray(constraints.shifts),
        solver=solver,
        linear_backend=linear_backend,
        n_iter=int(n_iter),
        converged=False,
        conflict=problem.hard_conflict,
        warnings=tuple(warnings_list),
        connectivity=connectivity,
        edge_diagnostics=_compute_edge_diagnostics(
            problem.constraints,
            weights=None,
        ),
        objective_breakdown=None,
    )
    return _bind_originating_observations(result, constraints)


def _certify_final_quadratic_weights(
    problem: SeparatorFitProblem,
    weights: np.ndarray,
    *,
    backend: Literal['dense', 'sparse'],
    components_already_certified: bool,
) -> np.ndarray:
    """Certify the assembled candidate after public gauge canonicalization."""

    constraints = problem.constraints
    candidate = problem.canonicalize_gauge(weights)
    if (
        components_already_certified
        and np.array_equal(
            candidate,
            np.asarray(weights, dtype=np.float64),
        )
    ):
        # Each component solver already certified these exact source-unit
        # binary64 values.  Canonicalization returned the same bit pattern, so
        # that certificate applies to the final public vector itself.
        return candidate
    components = problem._model_coupling_components()
    align_component_means = (
        problem.regularization_strength == 0.0
        and len(components) > 1
    )

    def certify_components(
        values: np.ndarray,
        *,
        allow_recovery: bool,
    ) -> np.ndarray:
        certified = np.asarray(values, dtype=np.float64).copy()
        for nodes in components:
            idx_nodes = np.asarray(nodes, dtype=np.int64)
            if idx_nodes.size <= 1:
                if (
                    problem.regularization_strength > 0.0
                    and idx_nodes.size == 1
                    and certified[idx_nodes[0]]
                    != problem.regularization_reference[idx_nodes[0]]
                ):
                    raise QuadraticNumericalError(
                        'final singleton L2 coordinate is not exactly equal '
                        'to its reference'
                    )
                continue
            node_set = set(nodes)
            mask = (
                problem.offset_identifying_constraint_mask
                & np.fromiter(
                    (
                        (int(i) in node_set) and (int(j) in node_set)
                        for i, j in zip(constraints.i, constraints.j)
                    ),
                    dtype=bool,
                    count=constraints.n_constraints,
                )
            )
            local_index = {int(node): k for k, node in enumerate(nodes)}
            required_mean = None
            if align_component_means:
                reference = problem.model.regularization.reference
                required_mean = (
                    0.0
                    if reference is None
                    else float(
                        np.mean(
                            problem.regularization_reference[idx_nodes]
                        )
                    )
                )
            ii = np.asarray(
                [local_index[int(i)] for i in constraints.i[mask]],
                dtype=np.int64,
            )
            jj = np.asarray(
                [local_index[int(j)] for j in constraints.j[mask]],
                dtype=np.int64,
            )
            certified[idx_nodes] = certify_quadratic_component_candidate(
                ii,
                jj,
                problem.alpha[mask],
                problem.beta[mask],
                problem.measurement_target[mask],
                constraints.confidence[mask],
                problem.regularization_reference[idx_nodes],
                problem.regularization_strength,
                certified[idx_nodes],
                backend=backend,
                allow_recovery=allow_recovery,
                required_mean=required_mean,
            )
        return certified

    certified_candidate = certify_components(
        candidate,
        allow_recovery=True,
    )
    if np.array_equal(certified_candidate, candidate):
        return candidate
    candidate = certified_candidate
    final_candidate = problem.canonicalize_gauge(candidate)
    recertified = certify_components(
        final_candidate,
        allow_recovery=False,
    )
    if not np.array_equal(recertified, final_candidate):
        raise QuadraticNumericalError(
            'quadratic recovery changed the final gauge-canonical candidate'
        )
    return final_candidate


def fit_weights_from_separators(
    points: np.ndarray,
    constraints: SeparatorObservations | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: DomainAny | None = None,
    ids: Sequence[int | np.integer] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    solver: Literal['direct', 'admm'] = 'direct',
    linear_backend: Literal['dense', 'sparse'] = 'dense',
    admm_max_iter: int = 2000,
    admm_rho: float = 1.0,
    admm_abs_tol: float = 1e-6,
    admm_rel_tol: float = 1e-5,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> SeparatorFitResult:
    """Fit power weights from resolved pairwise separator observations.

    ``solver='direct'`` solves a purely quadratic model directly.
    ``solver='admm'`` executes ADMM whenever a component solve is required and
    is required for Huber mismatch, active scalar penalties, or hard
    restrictions.  A no-work fit reports ``solver='none'`` and
    ``linear_backend=None``.  ``linear_backend`` selects dense NumPy or
    explicitly requested sparse SciPy linear algebra without size-based
    switching.  Zero-strength penalties are absent and do not force ADMM.
    Supported numerical failure of the optional direct ADMM warm start falls
    back to the reference or zero initialization.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')

    if model is None:
        model = FitModel()

    if isinstance(constraints, SeparatorObservations):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
        if resolved.dim != pts.shape[1]:
            raise ValueError('resolved constraints do not match the point dimension')
        if resolved.measurement != measurement:
            measurement = resolved.measurement
    else:
        resolved = resolve_separator_observations(
            pts,
            constraints,
            measurement=measurement,
            domain=domain,
            ids=ids,
            index_mode=index_mode,
            image=image,
            image_search=image_search,
            confidence=confidence,
            allow_empty=True,
        )
        measurement = resolved.measurement

    return _fit_power_weights_resolved(
        resolved,
        model=model,
        r_min=r_min,
        weight_shift=weight_shift,
        solver=solver,
        linear_backend=linear_backend,
        admm_max_iter=admm_max_iter,
        admm_rho=admm_rho,
        admm_abs_tol=admm_abs_tol,
        admm_rel_tol=admm_rel_tol,
        connectivity_check=connectivity_check,
    )


def _fit_power_weights_resolved(
    constraints: SeparatorObservations,
    *,
    model: FitModel,
    r_min: float,
    weight_shift: float | None,
    solver: Literal['direct', 'admm'],
    linear_backend: Literal['dense', 'sparse'],
    admm_max_iter: int,
    admm_rho: float,
    admm_abs_tol: float,
    admm_rel_tol: float,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'],
) -> SeparatorFitResult:
    n = int(constraints.n_points)
    m = int(constraints.n_constraints)
    warnings_list = list(constraints.warnings)

    if solver not in ('direct', 'admm'):
        raise ValueError("solver must be 'direct' or 'admm'")
    if linear_backend not in ('dense', 'sparse'):
        raise ValueError("linear_backend must be 'dense' or 'sparse'")
    if admm_max_iter <= 0:
        raise ValueError('admm_max_iter must be > 0')
    if admm_rho <= 0:
        raise ValueError('admm_rho must be > 0')
    if admm_abs_tol <= 0 or admm_rel_tol <= 0:
        raise ValueError('admm_abs_tol and admm_rel_tol must be > 0')
    if linear_backend == 'sparse':
        _require_scipy_sparse()
    if connectivity_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'connectivity_check must be none, diagnose, warn, or raise'
        )

    problem = build_power_fit_problem(constraints, model=model)
    accepted_hard_bounds = None
    if (
        problem.bounds.measurement_lower is not None
        and problem.bounds.measurement_upper is not None
    ):
        accepted_hard_bounds = _hard_accepted_measurement_bounds(
            problem.bounds.measurement_lower,
            problem.bounds.measurement_upper,
        )
    lam = float(problem.regularization_strength)
    reference = (
        None
        if problem.model.regularization.reference is None
        else problem.regularization_reference
    )

    nonquadratic = _requires_admm(model)
    if solver == 'direct' and nonquadratic:
        raise ValueError(
            "solver='direct' cannot be used with hard constraints, "
            "non-quadratic mismatch, or active scalar penalties; select "
            "solver='admm'"
        )

    connectivity = None if connectivity_check == 'none' else problem.connectivity
    if connectivity is not None:
        _apply_connectivity_policy(connectivity_check, connectivity, warnings_list)

    if not problem.hard_feasible:
        warnings_list.append('hard feasibility check failed before optimization')
        if problem.hard_conflict is not None:
            warnings_list.append(problem.hard_conflict.message)
        result = SeparatorFitResult(
            status='infeasible_hard_constraints',
            status_detail=(
                None
                if problem.hard_conflict is None
                else problem.hard_conflict.message
            ),
            hard_feasible=False,
            weights=None,
            radii=None,
            weight_shift=None,
            measurement=constraints.measurement,
            target=np.asarray(problem.measurement_target, dtype=np.float64),
            predicted=None,
            predicted_fraction=None,
            predicted_position=None,
            residuals=None,
            rms_residual=None,
            max_residual=None,
            used_shifts=np.asarray(constraints.shifts),
            solver='none',
            linear_backend=None,
            n_iter=0,
            converged=False,
            conflict=problem.hard_conflict,
            warnings=tuple(warnings_list),
            connectivity=connectivity,
            edge_diagnostics=_compute_edge_diagnostics(
                problem.constraints,
                weights=None,
            ),
            objective_breakdown=None,
        )
        return _bind_originating_observations(result, constraints)

    if m == 0:
        if lam > 0.0:
            weights = problem.regularization_reference.copy()
            warnings_list.append(
                'empty constraint set; using the regularization-only solution'
            )
        elif reference is not None:
            weights = reference.copy()
            warnings_list.append(
                'empty constraint set; no pair data are present, so weights '
                'follow the zero-strength reference gauge convention'
            )
        else:
            weights = np.zeros(n, dtype=np.float64)
            warnings_list.append(
                'empty constraint set; returning the mean-zero gauge solution'
            )
        try:
            result = build_power_fit_result(
                problem,
                weights,
                solver='none',
                linear_backend=None,
                status='optimal',
                converged=True,
                n_iter=0,
                warnings=tuple(warnings_list),
                canonicalize_gauge=True,
                r_min=r_min,
                weight_shift=weight_shift,
            )
        except _NonFiniteOptimalObjectiveError as exc:
            return _numerical_failure_result(
                problem,
                constraints,
                solver='none',
                linear_backend=None,
                n_iter=0,
                warnings_list=warnings_list,
                connectivity=connectivity,
                error=exc,
            )
        if connectivity is None:
            result = replace(result, connectivity=None)
            return _bind_originating_observations(result, constraints)
        return result

    weights = np.zeros(n, dtype=np.float64)
    converged_all = True
    n_iter_max = 0
    solver_ran = False
    comps = _connected_components(
        n,
        constraints.i[problem.offset_identifying_constraint_mask],
        constraints.j[problem.offset_identifying_constraint_mask],
    )
    align_component_means = lam == 0.0 and len(comps) > 1

    try:
        for nodes in comps:
            idx_nodes = np.asarray(nodes, dtype=np.int64)
            if idx_nodes.size <= 1:
                if lam > 0.0 and idx_nodes.size == 1:
                    weights[idx_nodes[0]] = (
                        problem.regularization_reference[idx_nodes[0]]
                    )
                continue

            node_set = set(nodes)
            mask = problem.offset_identifying_constraint_mask & np.fromiter(
                (
                    (int(i) in node_set) and (int(j) in node_set)
                    for i, j in zip(constraints.i, constraints.j)
                ),
                dtype=bool,
                count=m,
            )
            local_index = {int(node): k for k, node in enumerate(nodes)}
            ii = np.array(
                [local_index[int(i)] for i in constraints.i[mask]],
                dtype=np.int64,
            )
            jj = np.array(
                [local_index[int(j)] for j in constraints.j[mask]],
                dtype=np.int64,
            )
            alpha_c = problem.alpha[mask]
            beta_c = problem.beta[mask]
            target_c = problem.measurement_target[mask]
            conf_c = constraints.confidence[mask]
            w0_c = problem.regularization_reference[idx_nodes]
            required_mean = None
            if align_component_means:
                required_mean = (
                    0.0
                    if problem.model.regularization.reference is None
                    else float(np.mean(w0_c))
                )
            if solver == 'direct':
                solver_ran = True
                w_c = _solve_component_direct(
                    ii,
                    jj,
                    alpha_c,
                    beta_c,
                    target_c,
                    conf_c,
                    w0_c,
                    lam,
                    backend=linear_backend,
                    required_mean=required_mean,
                )
                iters = 1
                conv = True
            else:
                solver_ran = True
                try:
                    w_c, iters, conv = _solve_component_admm(
                        ii,
                        jj,
                        alpha_c,
                        beta_c,
                        target_c,
                        conf_c,
                        w0_c,
                        model=model,
                        lambda_regularize=lam,
                        backend=linear_backend,
                        required_mean=required_mean,
                        rho=admm_rho,
                        max_iter=admm_max_iter,
                        tol_abs=admm_abs_tol,
                        tol_rel=admm_rel_tol,
                        y_lo=(
                            None
                            if problem.bounds.measurement_lower is None
                            else problem.bounds.measurement_lower[mask]
                        ),
                        y_hi=(
                            None
                            if problem.bounds.measurement_upper is None
                            else problem.bounds.measurement_upper[mask]
                        ),
                        accepted_y_lo=(
                            None
                            if accepted_hard_bounds is None
                            else accepted_hard_bounds[0][mask]
                        ),
                        accepted_y_hi=(
                            None
                            if accepted_hard_bounds is None
                            else accepted_hard_bounds[1][mask]
                        ),
                        row_indices=np.flatnonzero(mask),
                    )
                except _IterativeNumericalFailure as exc:
                    n_iter_max = max(n_iter_max, exc.n_iter)
                    raise
            if not np.all(np.isfinite(w_c)):
                raise _NumericalFailure(
                    'component solver returned non-finite weights'
                )
            weights[idx_nodes] = w_c
            converged_all = converged_all and conv
            n_iter_max = max(n_iter_max, iters)

        if not np.all(np.isfinite(weights)):
            raise _NumericalFailure('assembled weight vector is non-finite')
        if converged_all and not nonquadratic:
            weights = _certify_final_quadratic_weights(
                problem,
                weights,
                backend=linear_backend,
                components_already_certified=(solver == 'direct'),
            )
        result = build_power_fit_result(
            problem,
            weights,
            solver=solver if solver_ran else 'none',
            linear_backend=linear_backend if solver_ran else None,
            status='optimal' if converged_all else 'max_iter',
            status_detail=(
                None
                if converged_all
                else 'iterative solver reached max_iter before convergence'
            ),
            converged=bool(converged_all),
            n_iter=int(n_iter_max),
            warnings=tuple(warnings_list) + (
                ()
                if converged_all
                else ('iterative solver reached max_iter before convergence',)
            ),
            canonicalize_gauge=False,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        if (
            result.objective_breakdown is None
            or not _soft_objective_is_finite(result.objective_breakdown)
        ):
            raise _NumericalFailure(
                'solver produced a non-finite soft objective'
            )
        if connectivity is None:
            result = replace(result, connectivity=None)
            return _bind_originating_observations(result, constraints)
        return result
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        QuadraticNumericalError,
        _NumericalFailure,
        _NonFiniteOptimalObjectiveError,
    ) as exc:
        return _numerical_failure_result(
            problem,
            constraints,
            solver=solver if solver_ran else 'none',
            linear_backend=linear_backend if solver_ran else None,
            n_iter=n_iter_max,
            warnings_list=warnings_list,
            connectivity=connectivity,
            error=exc,
        )


def _solve_component_direct(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
    *,
    backend: Literal['dense', 'sparse'],
    required_mean: float | None = None,
) -> np.ndarray:
    """Solve one component through the requested certified backend."""

    return solve_quadratic_component(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize,
        backend=backend,
        required_mean=required_mean,
    )


def _solve_component_analytic(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray:
    """Private dense-direct helper retained for focused internal tests."""

    return _solve_component_direct(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize,
        backend='dense',
    )


def _solve_component_sparse(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray:
    """Solve one component through certified sparse least squares."""

    return _solve_component_direct(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize,
        backend='sparse',
    )


def _positive_confidence_connects_component(
    n_c: int,
    I: np.ndarray,
    J: np.ndarray,
    confidence: np.ndarray,
) -> bool:
    mask = np.asarray(confidence, dtype=np.float64) > 0.0
    if not np.any(mask):
        return False
    comps = _connected_components(n_c, I[mask], J[mask])
    return len(comps) == 1


def _admm_warm_start_weights(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    *,
    lambda_regularize: float,
    backend: Literal['dense', 'sparse'],
) -> np.ndarray:
    """Return an optional direct warm start or the safe reference/zero seed."""

    n_c = int(np.max(np.maximum(I, J))) + 1
    lam = float(lambda_regularize)
    if lam > 0.0 or _positive_confidence_connects_component(
        n_c,
        I,
        J,
        confidence,
    ):
        try:
            return _solve_component_direct(
                I,
                J,
                alpha,
                beta,
                target,
                confidence,
                w0,
                lam,
                backend=backend,
            )
        except (
            np.linalg.LinAlgError,
            FloatingPointError,
            OverflowError,
            QuadraticNumericalError,
            _NumericalFailure,
        ):
            # The warm start is acceleration only.  Its supported
            # linear-algebra failure must not prevent ADMM from continuing.
            pass
    if lam > 0.0:
        return np.asarray(w0, dtype=np.float64).copy()
    return np.zeros(n_c, dtype=np.float64)


def _mismatch_component_objective(
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    weights: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    reference: np.ndarray,
    lambda_regularize: float,
    mismatch: SquaredLoss | HuberLoss,
) -> float:
    """Return the authoritative local mismatch-plus-L2 objective."""

    rows = _mismatch_values_from_affine(
        beta,
        alpha,
        weights[I],
        weights[J],
        target,
        confidence,
        mismatch,
    )
    regularization = _l2_value(
        weights,
        reference,
        lambda_regularize,
    )
    return _stable_sum_scalar(*rows.tolist(), regularization)


def _solve_component_admm(
    I: np.ndarray,
    J: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    w0: np.ndarray,
    *,
    model: FitModel,
    lambda_regularize: float,
    backend: Literal['dense', 'sparse'],
    required_mean: float | None,
    rho: float,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
    accepted_y_lo: np.ndarray | None,
    accepted_y_hi: np.ndarray | None,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    n_c = int(np.max(np.maximum(I, J))) + 1
    m_c = I.shape[0]
    lam = float(lambda_regularize)

    hard_lower = y_lo
    hard_upper = y_hi
    prox_lower = accepted_y_lo
    prox_upper = accepted_y_hi
    hints: list[np.ndarray] = []
    for bound in (prox_lower, prox_upper):
        if bound is None:
            continue
        bound_array = np.asarray(bound, dtype=np.float64)
        hints.append(
            np.where(np.isfinite(bound_array), bound_array, target)
        )
    weight_system = QuadraticWeightSystem.build(
        I,
        J,
        alpha,
        beta,
        target,
        np.full(alpha.shape, rho, dtype=np.float64),
        w0,
        lam,
        backend=backend,
        rhs_hints=tuple(hints),
    )

    w = _admm_warm_start_weights(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize=lam,
        backend=backend,
    )
    if not np.all(np.isfinite(w)):
        raise _NumericalFailure('ADMM warm start produced non-finite values')

    track_huber_candidates = (
        isinstance(model.mismatch, HuberLoss)
        and not _active_scalar_penalties(model.penalties)
        and hard_lower is None
        and hard_upper is None
    )
    best_huber_w = None
    best_huber_objective = float('inf')
    if track_huber_candidates:
        best_huber_w = w.copy()
        best_huber_objective = _mismatch_component_objective(
            alpha,
            beta,
            target,
            confidence,
            w,
            I,
            J,
            w0,
            lam,
            model.mismatch,
        )

    y = _stable_affine_difference(beta, alpha, w[I], w[J])
    if prox_lower is not None:
        y = np.maximum(y, prox_lower)
    if prox_upper is not None:
        y = np.minimum(y, prox_upper)
    u = np.zeros(m_c, dtype=np.float64)
    converged = False
    iteration = 0
    completed_iterations = 0
    prox_spec = None
    if _active_scalar_penalties(model.penalties):
        try:
            prox_spec = _compile_scalar_prox_spec(model)
        except (TypeError, ValueError) as exc:
            raise _NumericalFailure(
                f'invalid compiled scalar proximal objective: {exc}'
            ) from exc

    try:
        for iteration in range(1, max_iter + 1):
            target_w = _stable_scaled_difference(y, u, 1.0)
            w = weight_system.solve(target_w)
            if not np.all(np.isfinite(w)):
                raise _NumericalFailure(
                    'ADMM primal iterate became non-finite'
                )

            if track_huber_candidates:
                candidate_objective = _mismatch_component_objective(
                    alpha,
                    beta,
                    target,
                    confidence,
                    w,
                    I,
                    J,
                    w0,
                    lam,
                    model.mismatch,
                )
                if candidate_objective < best_huber_objective:
                    best_huber_objective = candidate_objective
                    best_huber_w = w.copy()

            predicted_y = _stable_affine_difference(
                beta,
                alpha,
                w[I],
                w[J],
            )
            y_prev = y.copy()
            prox_input = _stable_affine_residual(
                beta,
                alpha,
                w[I],
                w[J],
                -u,
            )
            y = _prox_measurement_objective(
                prox_input,
                target,
                confidence,
                model=model,
                rho=rho,
                y_lo=prox_lower,
                y_hi=prox_upper,
                spec=prox_spec,
                row_indices=row_indices,
            )
            r = _stable_affine_residual(
                beta,
                alpha,
                w[I],
                w[J],
                y,
            )
            u = _stable_sum(u, r)
            if not (
                np.all(np.isfinite(predicted_y))
                and np.all(np.isfinite(y))
                and np.all(np.isfinite(r))
                and np.all(np.isfinite(u))
            ):
                raise _NumericalFailure('ADMM iterates became non-finite')

            r_norm = _stable_norm(r)
            predicted_norm = _stable_norm(predicted_y)
            y_norm = _stable_norm(y)
            eps_pri = (
                np.sqrt(m_c) * tol_abs
                + tol_rel * max(predicted_norm, y_norm)
            )

            scaled_dy = _stable_scaled_difference(
                y,
                y_prev,
                rho,
                alpha,
            )
            s_vec = _stable_incidence_accumulate(n_c, I, J, scaled_dy)
            s_mean = _stable_sum_scalar(*s_vec.tolist()) / n_c
            s_vec = _stable_scaled_difference(
                s_vec,
                s_mean,
                1.0,
            )
            s_norm = _stable_norm(s_vec)

            scaled_dual = _stable_product(rho, alpha, u)
            dual_vec = _stable_incidence_accumulate(
                n_c,
                I,
                J,
                scaled_dual,
            )
            dual_mean = _stable_sum_scalar(*dual_vec.tolist()) / n_c
            dual_vec = _stable_scaled_difference(
                dual_vec,
                dual_mean,
                1.0,
            )
            dual_norm = _stable_norm(dual_vec)
            eps_dual = np.sqrt(n_c) * tol_abs + tol_rel * dual_norm

            hard_satisfied = True
            if hard_lower is not None and hard_upper is not None:
                hard_satisfied = bool(
                    np.all(
                        _hard_row_status(
                            hard_lower,
                            predicted_y,
                            hard_upper,
                        )[0]
                    )
                )
            if (
                r_norm <= eps_pri
                and s_norm <= eps_dual
                and hard_satisfied
            ):
                completed_iterations = iteration
                converged = True
                break
            completed_iterations = iteration

        if best_huber_w is not None:
            final_huber_objective = _mismatch_component_objective(
                alpha,
                beta,
                target,
                confidence,
                w,
                I,
                J,
                w0,
                lam,
                model.mismatch,
            )
            comparison_scale = max(
                abs(best_huber_objective),
                abs(final_huber_objective),
            )
            material_improvement = _stable_product_scalar(
                64.0 * np.finfo(np.float64).eps,
                comparison_scale,
            )
            if (
                best_huber_objective == 0.0
                or (
                    final_huber_objective - best_huber_objective
                    > material_improvement
                )
            ):
                w = best_huber_w
        if lam == 0.0:
            w = _stable_scaled_difference(w, w[0], 1.0)
            if required_mean is not None:
                w = _shift_to_required_mean(w, required_mean)
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        QuadraticNumericalError,
        _NumericalFailure,
    ) as exc:
        raise _IterativeNumericalFailure(
            str(exc),
            n_iter=completed_iterations,
        ) from exc
    return w, completed_iterations, converged


def _prox_measurement_mismatch_only(
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
    rho: float,
) -> np.ndarray:
    if isinstance(mismatch, SquaredLoss):
        return _stable_weighted_average(v, rho, target, confidence)
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        y_quad = _stable_weighted_average(v, rho, target, confidence)
        shift = _stable_ratio_product(
            (confidence, delta),
            (rho,),
        )
        y_lower = _stable_sum(v, shift)
        y_upper = _stable_sum(v, -shift)
        lower_sign = _stable_sum_products_sign(
            (
                (rho, v),
                (-rho, target),
                (rho, delta),
                (confidence, delta),
            )
        )
        upper_sign = _stable_sum_products_sign(
            (
                (rho, v),
                (-rho, target),
                (-rho, delta),
                (-1.0, confidence, delta),
            )
        )
        return np.where(
            lower_sign < 0,
            y_lower,
            np.where(upper_sign > 0, y_upper, y_quad),
        )
    raise TypeError(f'unsupported mismatch: {type(mismatch)!r}')


def _prox_measurement_objective(
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    model: FitModel,
    rho: float,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
    spec: _ScalarProxSpec | None = None,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    y = _prox_measurement_mismatch_only(
        v,
        target,
        confidence,
        model.mismatch,
        rho,
    )
    if y_lo is not None:
        y = np.maximum(y, y_lo)
    if y_hi is not None:
        y = np.minimum(y, y_hi)
    active_penalties = _active_scalar_penalties(model.penalties)
    if not active_penalties:
        return y
    if spec is None:
        try:
            spec = _compile_scalar_prox_spec(model)
        except (TypeError, ValueError) as exc:
            raise _NumericalFailure(
                f'invalid compiled scalar proximal objective: {exc}'
            ) from exc
    original_rows = (
        np.arange(y.shape[0], dtype=np.int64)
        if row_indices is None
        else np.asarray(row_indices, dtype=np.int64)
    )
    if original_rows.shape != y.shape:
        raise _NumericalFailure('scalar proximal row metadata shape mismatch')

    batch_certified = np.zeros(y.shape, dtype=bool)
    needs_general = np.fromiter(
        (
            not _penalties_inactive_at(spec, float(candidate))
            for candidate in y
        ),
        dtype=bool,
        count=y.shape[0],
    )
    general_indices = np.flatnonzero(needs_general)
    general_keys = {
        (
            float(target[index]),
            float(confidence[index]),
            float(v[index]),
            float('-inf') if y_lo is None else float(y_lo[index]),
            float('inf') if y_hi is None else float(y_hi[index]),
        )
        for index in general_indices
    }
    if len(general_keys) >= 8 and _batch_spec_supported(spec):
        batch_lower = (
            np.full(general_indices.size, -np.inf, dtype=np.float64)
            if y_lo is None
            else np.asarray(y_lo[general_indices], dtype=np.float64)
        )
        batch_upper = (
            np.full(general_indices.size, np.inf, dtype=np.float64)
            if y_hi is None
            else np.asarray(y_hi[general_indices], dtype=np.float64)
        )
        batch_outcome = _solve_scalar_prox_batch_ordinary(
            spec=spec,
            initial=y[general_indices],
            target=target[general_indices],
            confidence=confidence[general_indices],
            v=v[general_indices],
            rho=float(rho),
            lower=batch_lower,
            upper=batch_upper,
        )
        for batch_index, result in enumerate(batch_outcome.results):
            if result is None:
                continue
            local_index = int(general_indices[batch_index])
            y[local_index] = result.value
            batch_certified[local_index] = True

    coordinate_cache: dict[
        tuple[float, float, float, float, float],
        float,
    ] = {}
    for local_index in range(y.shape[0]):
        if batch_certified[local_index]:
            continue
        candidate = float(y[local_index])
        if _penalties_inactive_at(spec, candidate):
            continue
        lower = (
            float('-inf')
            if y_lo is None
            else float(y_lo[local_index])
        )
        upper = (
            float('inf')
            if y_hi is None
            else float(y_hi[local_index])
        )
        cache_key = (
            float(target[local_index]),
            float(confidence[local_index]),
            float(v[local_index]),
            lower,
            upper,
        )
        cached = coordinate_cache.get(cache_key)
        if cached is not None:
            y[local_index] = cached
            continue
        try:
            result = _solve_scalar_prox_coordinate(
                spec=spec,
                target=float(target[local_index]),
                confidence=float(confidence[local_index]),
                v=float(v[local_index]),
                rho=float(rho),
                lower=lower,
                upper=upper,
            )
        except _ScalarProxError as exc:
            failure = exc.failure
            raise _NumericalFailure(
                'scalar proximal failure for observation row '
                f'{int(original_rows[local_index])} '
                f'(component-local row {local_index}): '
                f'reason={failure.reason}; scalar_iterations='
                f'{failure.scalar_iterations}; expansions='
                f'{failure.expansion_count}; last_candidate='
                f'{failure.last_candidate!r}; last_finite_bracket='
                f'{failure.last_finite_bracket!r}; '
                f'last_derivative_enclosure='
                f'{failure.last_derivative_enclosure!r}; '
                f'localization_bound={failure.localization_bound!r}; '
                f'fallback_count={failure.fallback_count}'
            ) from exc
        y[local_index] = result.value
        coordinate_cache[cache_key] = result.value
    return y


def _apply_connectivity_policy(
    policy: Literal['none', 'diagnose', 'warn', 'raise'],
    diagnostics: ConnectivityDiagnostics,
    warnings_list: list[str],
) -> None:
    if policy in ('none', 'diagnose') or not diagnostics.messages:
        return
    if policy == 'warn':
        warnings_list.extend(diagnostics.messages)
        return
    if policy == 'raise':
        raise ConnectivityDiagnosticsError(
            '; '.join(diagnostics.messages),
            diagnostics,
        )
    raise ValueError('unsupported connectivity policy')


__all__ = [
    'fit_weights_from_separators',
    'ConnectivityDiagnosticsError',
]
