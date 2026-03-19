"""Native numerical solvers for fitting power weights."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from .constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from .model import FitModel, HuberLoss, SquaredLoss
from .problem import (
    _compute_edge_diagnostics,
    _connected_components,
    _mismatch_derivatives,
    _penalty_derivatives,
    _requires_admm,
    build_power_fit_problem,
    build_power_fit_result,
)
from .types import ConnectivityDiagnostics, PowerWeightFitResult
from ..domains import Box, OrthorhombicCell, PeriodicCell


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


def fit_power_weights(
    points: np.ndarray,
    constraints: PairBisectorConstraints | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box | OrthorhombicCell | PeriodicCell | None = None,
    ids: list[int] | tuple[int, ...] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    solver: Literal['auto', 'analytic', 'admm'] = 'auto',
    max_iter: int = 2000,
    rho: float = 1.0,
    tol_abs: float = 1e-6,
    tol_rel: float = 1e-5,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> PowerWeightFitResult:
    """Fit power weights from resolved pairwise separator constraints."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    if not np.all(np.isfinite(pts)):
        raise ValueError('points must contain only finite values')

    if model is None:
        model = FitModel()

    if isinstance(constraints, PairBisectorConstraints):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
        if resolved.dim != pts.shape[1]:
            raise ValueError('resolved constraints do not match the point dimension')
        if resolved.measurement != measurement:
            measurement = resolved.measurement
    else:
        resolved = resolve_pair_bisector_constraints(
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
        max_iter=max_iter,
        rho=rho,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        connectivity_check=connectivity_check,
    )


def _fit_power_weights_resolved(
    constraints: PairBisectorConstraints,
    *,
    model: FitModel,
    r_min: float,
    weight_shift: float | None,
    solver: Literal['auto', 'analytic', 'admm'],
    max_iter: int,
    rho: float,
    tol_abs: float,
    tol_rel: float,
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'],
) -> PowerWeightFitResult:
    n = int(constraints.n_points)
    m = int(constraints.n_constraints)
    warnings_list = list(constraints.warnings)

    if max_iter <= 0:
        raise ValueError('max_iter must be > 0')
    if rho <= 0:
        raise ValueError('rho must be > 0')
    if tol_abs <= 0 or tol_rel <= 0:
        raise ValueError('tol_abs and tol_rel must be > 0')
    if connectivity_check not in ('none', 'diagnose', 'warn', 'raise'):
        raise ValueError(
            'connectivity_check must be none, diagnose, warn, or raise'
        )

    problem = build_power_fit_problem(constraints, model=model)
    lam = float(problem.regularization_strength)
    reference = (
        None
        if problem.model.regularization.reference is None
        else problem.regularization_reference
    )

    nonquadratic = _requires_admm(model)
    if solver == 'auto':
        solver_eff = 'analytic' if not nonquadratic else 'admm'
    else:
        solver_eff = solver
    if solver_eff not in ('analytic', 'admm'):
        raise ValueError('solver must be auto, analytic, or admm')
    if solver_eff == 'analytic' and nonquadratic:
        raise ValueError(
            'analytic solver cannot be used with hard constraints '
            'or non-quadratic penalties'
        )

    connectivity = None if connectivity_check == 'none' else problem.connectivity
    if connectivity is not None:
        _apply_connectivity_policy(connectivity_check, connectivity, warnings_list)

    if not problem.hard_feasible:
        warnings_list.append('hard feasibility check failed before optimization')
        if problem.hard_conflict is not None:
            warnings_list.append(problem.hard_conflict.message)
        result = PowerWeightFitResult(
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
        return result

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
        result = build_power_fit_result(
            problem,
            weights,
            solver='analytic',
            status='optimal',
            converged=True,
            n_iter=0,
            warnings=tuple(warnings_list),
            canonicalize_gauge=True,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        if connectivity is None:
            return replace(result, connectivity=None)
        return result

    weights = np.zeros(n, dtype=np.float64)
    converged_all = True
    n_iter_max = 0
    comps = _connected_components(
        n,
        constraints.i[problem.offset_identifying_constraint_mask],
        constraints.j[problem.offset_identifying_constraint_mask],
    )

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
            if solver_eff == 'analytic':
                w_c = _solve_component_analytic(
                    ii,
                    jj,
                    problem.edge_weight[mask],
                    problem.z_obs[mask],
                    w0_c,
                    lam,
                )
                iters = 1
                conv = True
            else:
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
                    rho=rho,
                    max_iter=max_iter,
                    tol_abs=tol_abs,
                    tol_rel=tol_rel,
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
                )
            if not np.all(np.isfinite(w_c)):
                raise _NumericalFailure('component solver returned non-finite weights')
            weights[idx_nodes] = w_c
            converged_all = converged_all and conv
            n_iter_max = max(n_iter_max, iters)

        if not np.all(np.isfinite(weights)):
            raise _NumericalFailure('assembled weight vector is non-finite')
        result = build_power_fit_result(
            problem,
            weights,
            solver=solver_eff,
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
            canonicalize_gauge=True,
            r_min=r_min,
            weight_shift=weight_shift,
        )
        if connectivity is None:
            return replace(result, connectivity=None)
        return result
    except (np.linalg.LinAlgError, FloatingPointError, _NumericalFailure) as exc:
        warnings_list.append(f'numerical solver failure: {exc}')
        return PowerWeightFitResult(
            status='numerical_failure',
            status_detail=str(exc),
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
            solver=solver_eff,
            n_iter=int(n_iter_max),
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


def _solve_component_analytic(
    I: np.ndarray,
    J: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    lambda_regularize: float,
) -> np.ndarray:
    n_c = int(np.max(np.maximum(I, J))) + 1
    if w0.shape != (n_c,):
        w0 = np.asarray(w0, dtype=float).reshape(n_c)
    lam = float(lambda_regularize)
    L = np.zeros((n_c, n_c), dtype=np.float64)
    rhs = np.zeros(n_c, dtype=np.float64)
    for i, j, ak, bk in zip(I.tolist(), J.tolist(), a.tolist(), b.tolist()):
        L[i, i] += ak
        L[j, j] += ak
        L[i, j] -= ak
        L[j, i] -= ak
        rhs[i] += ak * bk
        rhs[j] -= ak * bk
    if lam > 0:
        L += lam * np.eye(n_c)
        rhs += lam * w0

    if n_c == 1:
        if lam > 0:
            return w0.astype(np.float64, copy=True)
        return np.zeros(1, dtype=np.float64)

    if lam > 0:
        return np.linalg.solve(L, rhs).astype(np.float64)

    free = np.arange(1, n_c, dtype=np.int64)
    Lf = L[np.ix_(free, free)]
    rhsf = rhs[free]
    wf = np.linalg.solve(Lf, rhsf)
    w = np.zeros(n_c, dtype=np.float64)
    w[free] = wf
    return w


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
) -> np.ndarray:
    n_c = int(np.max(np.maximum(I, J))) + 1
    lam = float(lambda_regularize)
    if lam > 0.0 or _positive_confidence_connects_component(
        n_c,
        I,
        J,
        confidence,
    ):
        try:
            return _solve_component_analytic(
                I,
                J,
                np.asarray(confidence, dtype=np.float64) * (alpha * alpha),
                (target - beta) / alpha,
                w0,
                lam,
            )
        except np.linalg.LinAlgError:
            pass
    if lam > 0.0:
        return np.asarray(w0, dtype=np.float64).copy()
    return np.zeros(n_c, dtype=np.float64)


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
    rho: float,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
) -> tuple[np.ndarray, int, bool]:
    n_c = int(np.max(np.maximum(I, J))) + 1
    m_c = I.shape[0]
    lam = float(lambda_regularize)

    if lam > 0.0:
        anchor: int | None = None
        free = np.arange(n_c, dtype=np.int64)
    else:
        anchor = 0
        free = np.arange(1, n_c, dtype=np.int64)

    edge_scale = alpha * alpha
    L = np.zeros((n_c, n_c), dtype=np.float64)
    for i, j, scale in zip(I.tolist(), J.tolist(), edge_scale.tolist()):
        L[i, i] += scale
        L[j, j] += scale
        L[i, j] -= scale
        L[j, i] -= scale

    M = rho * L + lam * np.eye(n_c)
    Mf = M[np.ix_(free, free)]
    if free.size and not np.all(np.isfinite(Mf)):
        raise _NumericalFailure('ADMM system matrix contains non-finite values')
    try:
        chol = (
            np.linalg.cholesky(Mf)
            if free.size
            else np.zeros((0, 0), dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        Mf2 = Mf + 1e-12 * np.eye(Mf.shape[0])
        try:
            chol = np.linalg.cholesky(Mf2)
        except np.linalg.LinAlgError as exc:
            raise _NumericalFailure(
                'ADMM system matrix is not numerically positive definite'
            ) from exc
        Mf = Mf2

    def solve_M(rhs_free: np.ndarray) -> np.ndarray:
        if rhs_free.size == 0:
            return np.zeros(0, dtype=np.float64)
        y = np.linalg.solve(chol, rhs_free)
        x = np.linalg.solve(chol.T, y)
        if not np.all(np.isfinite(x)):
            raise _NumericalFailure('ADMM linear solve produced non-finite values')
        return x

    w = _admm_warm_start_weights(
        I,
        J,
        alpha,
        beta,
        target,
        confidence,
        w0,
        lambda_regularize=lam,
    )
    if not np.all(np.isfinite(w)):
        raise _NumericalFailure('ADMM warm start produced non-finite values')
    y = beta + alpha * (w[I] - w[J])
    if y_lo is not None:
        y = np.maximum(y, y_lo)
    if y_hi is not None:
        y = np.minimum(y, y_hi)
    u = np.zeros(m_c, dtype=np.float64)
    converged = False

    for _it in range(1, max_iter + 1):
        rhs = np.zeros(n_c, dtype=np.float64)
        edge_rhs = rho * alpha * (y - u - beta)
        np.add.at(rhs, I, edge_rhs)
        np.add.at(rhs, J, -edge_rhs)
        if lam > 0.0:
            rhs += lam * w0

        w_free = solve_M(rhs[free])
        if anchor is not None:
            w[anchor] = 0.0
        w[free] = w_free
        if not np.all(np.isfinite(w)):
            raise _NumericalFailure('ADMM primal iterate became non-finite')

        predicted_y = beta + alpha * (w[I] - w[J])
        y_prev = y.copy()
        y = _prox_measurement_objective(
            predicted_y + u,
            target,
            confidence,
            model=model,
            rho=rho,
            y_lo=y_lo,
            y_hi=y_hi,
        )
        r = predicted_y - y
        u = u + r
        if not (
            np.all(np.isfinite(predicted_y))
            and np.all(np.isfinite(y))
            and np.all(np.isfinite(r))
            and np.all(np.isfinite(u))
        ):
            raise _NumericalFailure('ADMM iterates became non-finite')

        r_norm = float(np.linalg.norm(r))
        predicted_norm = float(np.linalg.norm(predicted_y))
        y_norm = float(np.linalg.norm(y))
        eps_pri = np.sqrt(m_c) * tol_abs + tol_rel * max(predicted_norm, y_norm)

        dy = y - y_prev
        s_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(s_vec, I, rho * alpha * dy)
        np.add.at(s_vec, J, -rho * alpha * dy)
        s_norm = float(np.linalg.norm(s_vec[free])) if free.size else 0.0

        dual_vec = np.zeros(n_c, dtype=np.float64)
        np.add.at(dual_vec, I, rho * alpha * u)
        np.add.at(dual_vec, J, -rho * alpha * u)
        dual_norm = float(np.linalg.norm(dual_vec[free])) if free.size else 0.0
        eps_dual = np.sqrt(len(free)) * tol_abs + tol_rel * dual_norm

        if r_norm <= eps_pri and s_norm <= eps_dual:
            converged = True
            break

    return w, _it, converged


def _prox_measurement_mismatch_only(
    v: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    mismatch: SquaredLoss | HuberLoss,
    rho: float,
) -> np.ndarray:
    if isinstance(mismatch, SquaredLoss):
        denom = rho + (2.0 * confidence)
        return (rho * v + (2.0 * confidence * target)) / denom
    if isinstance(mismatch, HuberLoss):
        delta = float(mismatch.delta)
        y_quad = (rho * v + confidence * target) / (rho + confidence)
        lower = target - delta
        upper = target + delta
        y_lower = v + (confidence * delta) / rho
        y_upper = v - (confidence * delta) / rho
        return np.where(
            y_quad < lower,
            y_lower,
            np.where(y_quad > upper, y_upper, y_quad),
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
    if not model.penalties:
        return y

    for _ in range(60):
        fp_y, fpp_y = _mismatch_derivatives(y, target, confidence, model.mismatch)
        for penalty in model.penalties:
            p_fp_y, p_fpp_y = _penalty_derivatives(y, penalty)
            fp_y = fp_y + p_fp_y
            fpp_y = fpp_y + p_fpp_y

        g = fp_y + rho * (y - v)
        gp = fpp_y + rho
        if not np.all(np.isfinite(gp)) or np.any(np.abs(gp) < 1e-18):
            raise _NumericalFailure(
                'prox Newton derivative became singular or non-finite'
            )
        step = g / gp
        if not np.all(np.isfinite(step)):
            raise _NumericalFailure('prox Newton step became non-finite')
        y_new = y - step
        if y_lo is not None:
            y_new = np.maximum(y_new, y_lo)
        if y_hi is not None:
            y_new = np.minimum(y_new, y_hi)
        if float(np.max(np.abs(step))) < 1e-12:
            y = y_new
            break
        y = y_new
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


__all__ = ['fit_power_weights', 'ConnectivityDiagnosticsError']
