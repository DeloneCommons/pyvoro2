"""Independent regression oracles for the separator objective contract."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

import pyvoro2.inverse.separator as separator
import pyvoro2.inverse.separator._quadratic as separator_quadratic
import pyvoro2.inverse.separator.solver as separator_solver
from pyvoro2.inverse.separator._objective import (
    HARD_ATOL,
    HARD_RTOL,
    _hard_accepted_measurement_bounds,
    _hard_row_status,
    _l2_value,
)
from pyvoro2.inverse.separator._numerics import (
    _stable_affine_residual,
    _stable_incidence_accumulate,
    _stable_norm,
    _stable_product,
    _stable_sum,
    _stable_sum_scalar,
    _stable_weighted_average,
)
from pyvoro2.inverse.separator.problem import (
    _check_hard_feasibility,
    _mismatch_derivatives,
    _mismatch_values,
    _penalty_derivatives,
    _penalty_values,
)
from pyvoro2.inverse.separator.solver import (
    _prox_measurement_mismatch_only,
)


def _solver_kwargs(name: str) -> dict[str, str]:
    if name == 'analytic':
        return {'solver': 'direct', 'linear_backend': 'dense'}
    if name == 'sparse':
        return {'solver': 'direct', 'linear_backend': 'sparse'}
    if name == 'admm':
        return {'solver': 'admm', 'linear_backend': 'dense'}
    raise ValueError(f'unsupported test solver label: {name!r}')


def _regularized_oracle(
    points: np.ndarray,
    rows: list[tuple[int, int, float]],
    confidence: np.ndarray,
    strength: float,
    reference: np.ndarray,
) -> np.ndarray:
    """Assemble the approved normal equations independently of pyvoro2."""

    n_sites = int(points.shape[0])
    design = np.zeros((len(rows), n_sites), dtype=np.float64)
    target_offset = np.zeros(len(rows), dtype=np.float64)
    for row_index, (site_i, site_j, target) in enumerate(rows):
        delta = points[site_j] - points[site_i]
        alpha = 1.0 / (2.0 * float(np.dot(delta, delta)))
        design[row_index, site_i] = alpha
        design[row_index, site_j] = -alpha
        target_offset[row_index] = float(target) - 0.5
    normal = design.T @ (confidence[:, None] * design)
    rhs = design.T @ (confidence * target_offset)
    normal += strength * np.eye(n_sites, dtype=np.float64)
    rhs += strength * reference
    return np.linalg.solve(normal, rhs)


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
@pytest.mark.parametrize('disconnected', [False, True])
def test_squared_l2_solvers_match_independent_normal_oracle(
    solver_name: str,
    disconnected: bool,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')

    if disconnected:
        points = np.array(
            [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [13.0, 0.0]],
            dtype=np.float64,
        )
        rows = [(0, 1, 0.20), (0, 1, 0.35), (2, 3, 0.70)]
        confidence = np.array([1.0, 0.5, 2.0], dtype=np.float64)
        reference = np.array([1.0, -0.5, 2.0, 3.0], dtype=np.float64)
    else:
        points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        rows = [(0, 1, 0.20), (0, 1, 0.35)]
        confidence = np.array([1.0, 0.5], dtype=np.float64)
        reference = np.array([1.0, -0.5], dtype=np.float64)

    strength = 0.2
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=strength,
            reference=reference,
        )
    )
    expected = _regularized_oracle(
        points,
        rows,
        confidence,
        strength,
        reference,
    )

    result = separator.fit_weights_from_separators(
        points,
        rows,
        confidence=confidence,
        model=model,
        **_solver_kwargs(solver_name),
        admm_max_iter=20000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(result.weights, expected, rtol=2e-8, atol=2e-9)


def _reciprocal_q(
    distance: np.ndarray,
    *,
    margin: float,
    epsilon: float,
    strength: float,
) -> np.ndarray:
    value = np.zeros_like(distance)
    continuation = distance <= epsilon
    reciprocal = (distance > epsilon) & (distance < margin)
    value[continuation] = strength * (
        (1.0 / epsilon - 1.0 / margin)
        - (distance[continuation] - epsilon) / epsilon**2
    )
    value[reciprocal] = strength * (
        1.0 / distance[reciprocal] - 1.0 / margin
    )
    return value


def test_direct_objective_result_and_report_agree_with_manual_recomputation() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [20.0, 0.0],
            [21.0, 0.0],
        ],
        dtype=np.float64,
    )
    rows = [(0, 1, 0.20), (2, 3, 0.60), (4, 5, 0.80)]
    confidence = np.array([2.0, 0.5, 0.0], dtype=np.float64)
    predicted = np.array([0.01, 0.50, 0.99], dtype=np.float64)
    differences = 2.0 * (predicted - 0.5)
    weights = np.zeros(6, dtype=np.float64)
    weights[0::2] = 0.5 * differences
    weights[1::2] = -0.5 * differences
    reference = np.linspace(-0.3, 0.2, 6)

    soft = separator.SoftIntervalPenalty(0.1, 0.9, 0.7)
    exponential = separator.ExponentialBoundaryPenalty(
        lower=0.0,
        upper=1.0,
        margin=0.1,
        strength=0.2,
        tau=0.08,
    )
    reciprocal = separator.ReciprocalBoundaryPenalty(
        lower=0.0,
        upper=1.0,
        margin=0.1,
        strength=0.3,
        epsilon=0.02,
    )
    regularization_strength = 0.4
    model = separator.FitModel(
        feasible=separator.Interval(0.0, 1.0),
        penalties=(soft, exponential, reciprocal),
        regularization=separator.L2Regularization(
            regularization_strength,
            reference,
        ),
    )
    observations = separator.resolve_separator_observations(
        points,
        rows,
        confidence=confidence,
    )
    problem = separator.build_power_fit_problem(observations, model=model)
    result = separator.build_power_fit_result(
        problem,
        weights,
        solver='external',
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )

    target = np.array([row[2] for row in rows], dtype=np.float64)
    residual = predicted - target
    mismatch = 0.5 * float(np.sum(confidence * residual**2))
    soft_value = float(
        np.sum(
            soft.strength
            * (
                np.maximum(soft.lower - predicted, 0.0) ** 2
                + np.maximum(predicted - soft.upper, 0.0) ** 2
            )
        )
    )
    lower_term = np.exp(
        (exponential.lower + exponential.margin - predicted)
        / exponential.tau
    )
    upper_term = np.exp(
        (predicted - (exponential.upper - exponential.margin))
        / exponential.tau
    )
    exponential_value = float(
        np.sum(exponential.strength * (lower_term + upper_term))
    )
    reciprocal_value = float(
        np.sum(
            _reciprocal_q(
                predicted - reciprocal.lower,
                margin=reciprocal.margin,
                epsilon=reciprocal.epsilon,
                strength=reciprocal.strength,
            )
            + _reciprocal_q(
                reciprocal.upper - predicted,
                margin=reciprocal.margin,
                epsilon=reciprocal.epsilon,
                strength=reciprocal.strength,
            )
        )
    )
    regularization = 0.5 * regularization_strength * float(
        np.sum((weights - reference) ** 2)
    )
    penalty_values = (soft_value, exponential_value, reciprocal_value)
    expected_total = mismatch + sum(penalty_values) + regularization

    breakdown = result.objective_breakdown
    assert breakdown is not None
    assert breakdown.mismatch == pytest.approx(mismatch)
    assert breakdown.regularization == pytest.approx(regularization)
    assert breakdown.penalties_total == pytest.approx(sum(penalty_values))
    assert tuple(value for _, value in breakdown.penalty_terms) == pytest.approx(
        penalty_values
    )
    assert breakdown.total == pytest.approx(expected_total)
    assert problem.evaluate_objective(weights) == pytest.approx(expected_total)

    report = separator.build_fit_report(result, observations)
    record = report['objective_breakdown']
    assert record is not None
    assert record['mismatch'] == pytest.approx(mismatch)
    assert record['regularization'] == pytest.approx(regularization)
    assert record['penalties_total'] == pytest.approx(sum(penalty_values))
    assert record['total'] == pytest.approx(expected_total)
    assert [item['value'] for item in record['penalty_terms']] == pytest.approx(
        penalty_values
    )
    assert record['hard_max_tolerance'] == pytest.approx(
        breakdown.hard_max_tolerance
    )


def test_large_delta_huber_is_squared_loss_in_values_derivatives_and_reports() -> None:
    y = np.array([-0.4, 0.0, 0.7], dtype=np.float64)
    target = np.array([0.1, -0.2, 0.4], dtype=np.float64)
    confidence = np.array([0.0, 0.5, 3.0], dtype=np.float64)
    squared = separator.SquaredLoss()
    huber = separator.HuberLoss(delta=1e6)

    np.testing.assert_allclose(
        _mismatch_values(y, target, confidence, squared),
        _mismatch_values(y, target, confidence, huber),
    )
    squared_first, squared_second = _mismatch_derivatives(
        y,
        target,
        confidence,
        squared,
    )
    huber_first, huber_second = _mismatch_derivatives(
        y,
        target,
        confidence,
        huber,
    )
    np.testing.assert_allclose(squared_first, huber_first)
    np.testing.assert_allclose(squared_second, huber_second)

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.5, 0.0]],
        dtype=np.float64,
    )
    rows = [(0, 1, 0.2), (1, 2, 0.7), (0, 2, 0.4)]
    reference = np.array([0.4, -0.2, 0.8])

    def fit(mismatch):
        return separator.fit_weights_from_separators(
            points,
            rows,
            confidence=[1.0, 2.0, 0.5],
            model=separator.FitModel(
                mismatch=mismatch,
                regularization=separator.L2Regularization(0.1, reference),
            ),
            solver='admm',
            admm_max_iter=20000,
            admm_abs_tol=1e-10,
            admm_rel_tol=1e-10,
            connectivity_check='diagnose',
        )

    squared_result = fit(squared)
    huber_result = fit(huber)
    assert squared_result.status == huber_result.status == 'optimal'
    np.testing.assert_allclose(
        squared_result.weights,
        huber_result.weights,
        rtol=1e-9,
        atol=1e-10,
    )
    assert squared_result.objective_breakdown is not None
    assert huber_result.objective_breakdown is not None
    assert huber_result.objective_breakdown.total == pytest.approx(
        squared_result.objective_breakdown.total
    )

    squared_report = separator.build_fit_report(
        squared_result,
        separator.resolve_separator_observations(
            points,
            rows,
            confidence=[1.0, 2.0, 0.5],
        ),
    )
    huber_report = separator.build_fit_report(
        huber_result,
        separator.resolve_separator_observations(
            points,
            rows,
            confidence=[1.0, 2.0, 0.5],
        ),
    )
    assert huber_report['objective_breakdown'] == pytest.approx(
        squared_report['objective_breakdown']
    )


def _manual_penalty_terms(
    penalty,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.zeros_like(y)
    first = np.zeros_like(y)
    second = np.zeros_like(y)
    strength = float(penalty.strength)

    if isinstance(penalty, separator.SoftIntervalPenalty):
        lower = y < penalty.lower
        upper = y > penalty.upper
        for mask, bound in ((lower, penalty.lower), (upper, penalty.upper)):
            displacement = y[mask] - bound
            value[mask] = strength * displacement**2
            first[mask] = 2.0 * strength * displacement
            second[mask] = 2.0 * strength
        return value, first, second

    if isinstance(penalty, separator.ExponentialBoundaryPenalty):
        lower_term = np.exp(
            (penalty.lower + penalty.margin - y) / penalty.tau
        )
        upper_term = np.exp(
            (y - (penalty.upper - penalty.margin)) / penalty.tau
        )
        value = strength * (lower_term + upper_term)
        first = strength * (-lower_term + upper_term) / penalty.tau
        second = (
            strength
            * (lower_term + upper_term)
            / penalty.tau**2
        )
        return value, first, second

    if isinstance(penalty, separator.ReciprocalBoundaryPenalty):
        for distance, direction in (
            (y - penalty.lower, 1.0),
            (penalty.upper - y, -1.0),
        ):
            continuation = distance <= penalty.epsilon
            reciprocal = (
                (distance > penalty.epsilon)
                & (distance < penalty.margin)
            )
            d = distance[continuation]
            value[continuation] += strength * (
                (1.0 / penalty.epsilon - 1.0 / penalty.margin)
                - (d - penalty.epsilon) / penalty.epsilon**2
            )
            first[continuation] += (
                -direction * strength / penalty.epsilon**2
            )
            d = distance[reciprocal]
            value[reciprocal] += strength * (
                1.0 / d - 1.0 / penalty.margin
            )
            first[reciprocal] += -direction * strength / d**2
            second[reciprocal] += 2.0 * strength / d**3
        return value, first, second

    raise AssertionError('unhandled test penalty')


@pytest.mark.parametrize(
    ('penalty', 'points'),
    [
        (
            separator.SoftIntervalPenalty(0.0, 1.0, 0.7),
            [-0.2, 0.5, 1.2],
        ),
        (
            separator.ExponentialBoundaryPenalty(
                0.0,
                1.0,
                0.1,
                0.3,
                0.2,
            ),
            [0.2, 0.5, 0.8],
        ),
        (
            separator.ReciprocalBoundaryPenalty(
                0.0,
                1.0,
                0.1,
                0.4,
                0.02,
            ),
            [0.01, 0.05, 0.5, 0.95, 0.99],
        ),
    ],
)
def test_scalar_penalty_values_and_derivatives_match_independent_formulas(
    penalty,
    points: list[float],
) -> None:
    y = np.asarray(points, dtype=np.float64)
    expected_value, expected_first, expected_second = _manual_penalty_terms(
        penalty,
        y,
    )
    actual_value = _penalty_values(y, penalty)
    actual_first, actual_second = _penalty_derivatives(y, penalty)
    np.testing.assert_allclose(actual_value, expected_value)
    np.testing.assert_allclose(actual_first, expected_first)
    np.testing.assert_allclose(actual_second, expected_second)

    for index, coordinate in enumerate(y):
        h = 1e-6
        samples = np.array([coordinate - h, coordinate, coordinate + h])
        values = _penalty_values(samples, penalty)
        finite_difference_first = (values[2] - values[0]) / (2.0 * h)
        assert actual_first[index] == pytest.approx(
            finite_difference_first,
            rel=2e-5,
            abs=2e-7,
        )
        if not (
            isinstance(penalty, separator.ReciprocalBoundaryPenalty)
            and (
                coordinate <= penalty.epsilon
                or coordinate >= penalty.upper - penalty.epsilon
            )
        ):
            finite_difference_second = (
                values[2] - 2.0 * values[1] + values[0]
            ) / h**2
            assert actual_second[index] == pytest.approx(
                finite_difference_second,
                rel=3e-4,
                abs=3e-4,
            )


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_reciprocal_branch_continuity_and_margin_derivative_jump(
    side: str,
) -> None:
    penalty = separator.ReciprocalBoundaryPenalty(
        0.0,
        1.0,
        margin=0.25,
        strength=0.4,
        epsilon=0.125,
    )
    sign = 1.0 if side == 'lower' else -1.0
    epsilon_point = (
        penalty.lower + penalty.epsilon
        if side == 'lower'
        else penalty.upper - penalty.epsilon
    )
    margin_point = (
        penalty.lower + penalty.margin
        if side == 'lower'
        else penalty.upper - penalty.margin
    )
    inward = sign
    eta = 1e-9

    epsilon_value = _penalty_values(
        np.array([epsilon_point]),
        penalty,
    )[0]
    epsilon_first, epsilon_second = _penalty_derivatives(
        np.array([epsilon_point]),
        penalty,
    )
    assert epsilon_value == pytest.approx(
        penalty.strength
        * (1.0 / penalty.epsilon - 1.0 / penalty.margin)
    )
    assert epsilon_first[0] == pytest.approx(
        -sign * penalty.strength / penalty.epsilon**2
    )
    assert epsilon_second[0] == 0.0
    for point in (
        epsilon_point - inward * eta,
        epsilon_point + inward * eta,
    ):
        value = _penalty_values(np.array([point]), penalty)[0]
        first = _penalty_derivatives(np.array([point]), penalty)[0][0]
        assert value == pytest.approx(epsilon_value, rel=2e-6)
        assert first == pytest.approx(epsilon_first[0], rel=2e-6)

    margin_value = _penalty_values(np.array([margin_point]), penalty)[0]
    margin_first, margin_second = _penalty_derivatives(
        np.array([margin_point]),
        penalty,
    )
    assert margin_value == 0.0
    assert margin_first[0] == 0.0
    assert margin_second[0] == 0.0
    active_point = margin_point - inward * eta
    active_value = _penalty_values(np.array([active_point]), penalty)[0]
    active_first = _penalty_derivatives(
        np.array([active_point]),
        penalty,
    )[0][0]
    assert active_value == pytest.approx(0.0, abs=1e-6)
    assert active_first == pytest.approx(
        -sign * penalty.strength / penalty.margin**2,
        rel=2e-7,
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'lower': 0.0, 'upper': 0.0},
        {'lower': 0.0, 'upper': 1.0, 'margin': 0.0},
        {'lower': 0.0, 'upper': 1.0, 'margin': 0.1, 'epsilon': 0.1},
        {'lower': 0.0, 'upper': 1.0, 'margin': 0.1, 'epsilon': 0.2},
        {'lower': 0.0, 'upper': 1.0, 'margin': 0.6},
        {'lower': 0.0, 'upper': 1.0, 'strength': -1.0},
    ],
)
def test_reciprocal_penalty_rejects_invalid_relational_parameters(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        separator.ReciprocalBoundaryPenalty(**kwargs)


@pytest.mark.parametrize(
    'penalty',
    [
        separator.SoftIntervalPenalty(-1.0, 1.0, 0.0),
        separator.ExponentialBoundaryPenalty(
            -1.0,
            1.0,
            0.1,
            0.0,
            np.nextafter(0.0, 1.0),
        ),
        separator.ReciprocalBoundaryPenalty(
            -1.0,
            1.0,
            0.5,
            0.0,
            np.nextafter(0.0, 1.0),
        ),
    ],
)
def test_zero_strength_penalties_are_exact_backend_invariant_noops(
    penalty,
) -> None:
    extreme = np.array(
        [-np.finfo(np.float64).max, np.finfo(np.float64).max],
        dtype=np.float64,
    )
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(
            _penalty_values(extreme, penalty),
            np.zeros_like(extreme),
        )
        first, second = _penalty_derivatives(extreme, penalty)
    np.testing.assert_array_equal(first, np.zeros_like(extreme))
    np.testing.assert_array_equal(second, np.zeros_like(extreme))

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=np.float64,
    )
    rows = [(0, 1, 0.2), (1, 2, 0.7)]
    reference = np.array([0.4, -0.2, 0.8])
    baseline_model = separator.FitModel(
        regularization=separator.L2Regularization(0.1, reference)
    )
    penalty_model = separator.FitModel(
        penalties=(penalty,),
        regularization=separator.L2Regularization(0.1, reference),
    )
    observations = separator.resolve_separator_observations(points, rows)
    baseline_problem = separator.build_power_fit_problem(
        observations,
        model=baseline_model,
    )
    penalty_problem = separator.build_power_fit_problem(
        observations,
        model=penalty_model,
    )
    np.testing.assert_array_equal(
        penalty_problem.offset_identifying_constraint_mask,
        baseline_problem.offset_identifying_constraint_mask,
    )
    assert (
        penalty_problem.connectivity.effective_graph
        == baseline_problem.connectivity.effective_graph
    )
    penalty_problem.quadratic_operator

    baseline = separator.fit_weights_from_separators(
        points,
        rows,
        model=baseline_model,
        solver='direct',
        connectivity_check='diagnose',
    )
    for solver_name in ('analytic', 'admm', 'sparse'):
        if solver_name == 'sparse':
            pytest.importorskip('scipy.sparse.linalg')
        result = separator.fit_weights_from_separators(
            points,
            rows,
            model=penalty_model,
            **_solver_kwargs(solver_name),
            admm_max_iter=20000,
            admm_abs_tol=1e-10,
            admm_rel_tol=1e-10,
            connectivity_check='diagnose',
        )
        assert result.status == 'optimal'
        np.testing.assert_allclose(
            result.weights,
            baseline.weights,
            rtol=2e-8,
            atol=2e-9,
        )
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.penalty_terms == (
            (type(penalty).__name__, 0.0),
        )
        assert result.objective_breakdown.total == pytest.approx(
            baseline.objective_breakdown.total,
            rel=2e-9,
            abs=2e-10,
        )


def _hard_result(
    lower: float,
    upper: float,
    prediction: float,
):
    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, lower)],
        measurement='position',
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            feasible=separator.Interval(lower, upper)
        ),
    )
    difference = 4.0 * (prediction - 1.0)
    weights = np.array([0.5 * difference, -0.5 * difference])
    result = separator.build_power_fit_result(
        problem,
        weights,
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )
    return problem, result


@pytest.mark.parametrize(
    ('lower', 'upper'),
    [
        (1e-9, 1.0),
        (1.0, 2.0),
        (1e12, 1e12 + 1e6),
    ],
)
def test_hard_bound_classification_is_scale_aware(
    lower: float,
    upper: float,
) -> None:
    nominal_tolerance = HARD_ATOL + HARD_RTOL * max(
        abs(lower),
        abs(lower),
        abs(upper),
    )
    inside_problem, inside = _hard_result(
        lower,
        upper,
        lower - 0.5 * nominal_tolerance,
    )
    outside_problem, outside = _hard_result(
        lower,
        upper,
        lower - 2.0 * nominal_tolerance,
    )

    assert inside.objective_breakdown.hard_constraints_satisfied is True
    assert outside.objective_breakdown.hard_constraints_satisfied is False
    for result in (inside, outside):
        breakdown = result.objective_breakdown
        expected_violation = max(lower - result.predicted[0], 0.0)
        expected_tolerance = HARD_ATOL + HARD_RTOL * max(
            abs(lower),
            abs(result.predicted[0]),
            abs(upper),
        )
        assert breakdown.hard_max_violation == pytest.approx(
            expected_violation
        )
        assert breakdown.hard_max_tolerance == pytest.approx(
            expected_tolerance
        )
    assert np.isfinite(inside_problem.evaluate_objective(inside.weights))
    assert outside_problem.evaluate_objective(outside.weights) == float('inf')

    report = separator.build_fit_report(
        outside,
        outside_problem.constraints,
    )
    assert report['objective_breakdown']['hard_max_tolerance'] == pytest.approx(
        outside.objective_breakdown.hard_max_tolerance
    )


@pytest.mark.parametrize('scale', [1e-9, 1.0, 1e12])
def test_difference_relaxation_does_not_reapply_measurement_tolerance(
    scale: float,
) -> None:
    site_i = np.array([0, 1, 2], dtype=np.int64)
    site_j = np.array([1, 2, 0], dtype=np.int64)
    tolerance = HARD_ATOL + HARD_RTOL * abs(scale)
    exact = np.array([scale, -scale, 0.0])
    contradictory = np.array([scale, -scale, 0.25 * tolerance])

    exact_feasible, exact_conflict = _check_hard_feasibility(
        3,
        site_i,
        site_j,
        exact,
        exact,
    )
    contradictory_feasible, contradictory_conflict = _check_hard_feasibility(
        3,
        site_i,
        site_j,
        contradictory,
        contradictory,
    )
    assert exact_feasible is True
    assert exact_conflict is None
    assert contradictory_feasible is False
    assert contradictory_conflict is not None


def test_no_hard_bounds_report_zero_maximum_tolerance() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
    )
    problem = separator.build_power_fit_problem(observations)
    breakdown = problem.objective_breakdown(np.array([-1.0, 1.0]))
    assert breakdown.hard_max_violation == 0.0
    assert breakdown.hard_max_tolerance == 0.0


def test_zero_strength_l2_reference_is_absent_from_objective_identification() -> None:
    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]],
        dtype=np.float64,
    )
    reference = np.array([10.0, 20.0, 30.0, 40.0])
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.25), (2, 3, 0.75)],
        model=separator.FitModel(
            regularization=separator.L2Regularization(0.0, reference)
        ),
        connectivity_check='diagnose',
    )
    assert result.objective_breakdown.regularization == 0.0
    assert result.connectivity.offsets_identified_in_objective is False
    assert np.mean(result.weights[:2]) == pytest.approx(np.mean(reference[:2]))
    assert np.mean(result.weights[2:]) == pytest.approx(np.mean(reference[2:]))


def test_public_builder_rejects_falsely_optimal_nonfinite_objective() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array(
        [np.finfo(np.float64).max, -np.finfo(np.float64).max]
    )
    with np.errstate(all='ignore'):
        with pytest.raises(ValueError, match='non-finite soft objective'):
            separator.build_power_fit_result(
                problem,
                weights,
                status='optimal',
                converged=True,
                canonicalize_gauge=False,
            )


def test_solver_converts_nonfinite_final_objective_to_numerical_failure(
    monkeypatch,
) -> None:
    import pyvoro2.inverse.separator.problem as problem_module

    original = problem_module._objective_breakdown

    def nonfinite_breakdown(*args, **kwargs):
        return replace(original(*args, **kwargs), total=float('inf'))

    monkeypatch.setattr(
        problem_module,
        '_objective_breakdown',
        nonfinite_breakdown,
    )
    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.25)],
        solver='direct',
    )
    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.objective_breakdown is None
    assert result.weights is None


def test_admm_warm_start_linear_algebra_failure_falls_back(
    monkeypatch,
) -> None:
    import pyvoro2.inverse.separator.solver as solver_module

    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    rows = [(0, 1, 0.2), (0, 1, 0.35)]
    confidence = np.array([1.0, 0.5])
    reference = np.array([1.0, -0.5])
    strength = 0.2
    expected = _regularized_oracle(
        points,
        rows,
        confidence,
        strength,
        reference,
    )

    warm_start_calls = 0

    def fail_warm_start(*args, **kwargs):
        nonlocal warm_start_calls
        warm_start_calls += 1
        raise np.linalg.LinAlgError('synthetic warm-start failure')

    monkeypatch.setattr(
        solver_module,
        '_solve_component_direct',
        fail_warm_start,
    )
    result = separator.fit_weights_from_separators(
        points,
        rows,
        confidence=confidence,
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                strength,
                reference,
            )
        ),
        solver='admm',
        admm_max_iter=20000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )
    assert warm_start_calls == 1
    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(result.weights, expected, rtol=2e-8, atol=2e-9)


def test_ordinary_default_squared_solution_remains_compatible() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    assert result.status == 'optimal'
    assert result.solver == 'direct'
    assert result.linear_backend == 'dense'
    np.testing.assert_allclose(result.weights, np.array([0.0, 2.0]))
    np.testing.assert_allclose(result.predicted, np.array([0.25]))


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def test_squared_mismatch_preserves_finite_weighted_extreme_values() -> None:
    value = _mismatch_values(
        np.array([1e200]),
        np.array([0.0]),
        np.array([1e-300]),
        separator.SquaredLoss(),
    )
    first, second = _mismatch_derivatives(
        np.array([1e200]),
        np.array([0.0]),
        np.array([1e-300]),
        separator.SquaredLoss(),
    )
    assert value[0] == pytest.approx(5e99, rel=2e-15)
    assert first[0] == pytest.approx(1e-100, rel=2e-15)
    assert second[0] == 1e-300


def test_squared_mismatch_scales_before_opposite_extreme_subtraction() -> None:
    maximum = np.finfo(np.float64).max
    confidence = 1e-320
    expected_value = float(
        Decimal('0.5')
        * _decimal(confidence)
        * (_decimal(maximum) - _decimal(-maximum)) ** 2
    )
    expected_first = float(
        _decimal(confidence)
        * (_decimal(maximum) - _decimal(-maximum))
    )
    with np.errstate(all='raise'):
        value = _mismatch_values(
            np.array([maximum]),
            np.array([-maximum]),
            np.array([confidence]),
            separator.SquaredLoss(),
        )
        first, second = _mismatch_derivatives(
            np.array([maximum]),
            np.array([-maximum]),
            np.array([confidence]),
            separator.SquaredLoss(),
        )
    assert value[0] == pytest.approx(expected_value, rel=3e-15)
    assert first[0] == pytest.approx(expected_first, rel=3e-15)
    assert second[0] == confidence


def test_zero_confidence_mismatch_never_evaluates_extreme_residual() -> None:
    maximum = np.finfo(np.float64).max
    for mismatch in (separator.SquaredLoss(), separator.HuberLoss(1.0)):
        with np.errstate(all='raise'):
            value = _mismatch_values(
                np.array([maximum]),
                np.array([-maximum]),
                np.array([0.0]),
                mismatch,
            )
            first, second = _mismatch_derivatives(
                np.array([maximum]),
                np.array([-maximum]),
                np.array([0.0]),
                mismatch,
            )
        np.testing.assert_array_equal(value, np.zeros(1))
        np.testing.assert_array_equal(first, np.zeros(1))
        np.testing.assert_array_equal(second, np.zeros(1))


def test_huber_linear_branch_preserves_finite_weighted_value() -> None:
    residual = 2e200
    delta = 1e200
    confidence = 1e-300
    expected = float(
        _decimal(confidence)
        * _decimal(delta)
        * (_decimal(residual) - Decimal('0.5') * _decimal(delta))
    )
    with np.errstate(all='raise'):
        value = _mismatch_values(
            np.array([residual]),
            np.array([0.0]),
            np.array([confidence]),
            separator.HuberLoss(delta),
        )
    assert value[0] == pytest.approx(expected, rel=3e-15)


def test_huber_linear_branch_does_not_evaluate_unused_square() -> None:
    with np.errstate(all='raise'):
        value = _mismatch_values(
            np.array([1e200]),
            np.array([0.0]),
            np.array([1.0]),
            separator.HuberLoss(1.0),
        )
    assert value[0] == pytest.approx(1e200)


def test_zero_and_tiny_l2_use_scale_safe_differences() -> None:
    maximum = np.finfo(np.float64).max
    with np.errstate(all='raise'):
        assert _l2_value(
            np.array([maximum]),
            np.array([-maximum]),
            0.0,
        ) == 0.0

    strength = 1e-300
    displacement = 1e200
    expected = float(
        Decimal('0.5')
        * _decimal(strength)
        * _decimal(displacement) ** 2
    )
    with np.errstate(all='raise'):
        actual = _l2_value(
            np.array([displacement]),
            np.array([0.0]),
            strength,
        )
    assert actual == pytest.approx(expected, rel=3e-15)


def test_true_nonrepresentable_objective_values_remain_nonfinite() -> None:
    mismatch = _mismatch_values(
        np.array([1e308]),
        np.array([0.0]),
        np.array([1.0]),
        separator.SquaredLoss(),
    )
    regularization = _l2_value(
        np.array([np.finfo(np.float64).max]),
        np.array([0.0]),
        1.0,
    )
    assert np.isinf(mismatch[0])
    assert np.isinf(regularization)


def test_zero_l2_problem_breakdown_skips_extreme_reference_difference() -> None:
    maximum = np.finfo(np.float64).max
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [],
        allow_empty=True,
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                0.0,
                np.array([-maximum, -maximum]),
            )
        ),
    )
    with np.errstate(all='raise'):
        breakdown = problem.objective_breakdown(
            np.array([maximum, maximum])
        )
    assert breakdown.regularization == 0.0


def test_extreme_contradictory_rows_have_finite_optimal_objective() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 1e200), (0, 1, -1e200)],
        confidence=np.array([1e-300, 1e-300]),
        solver='direct',
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    assert result.converged is True
    assert result.objective_breakdown.total == pytest.approx(1e100, rel=3e-15)
    assert result.rms_residual == pytest.approx(1e200, rel=3e-15)


@pytest.mark.parametrize(
    ('alpha', 'confidence', 'rho'),
    [
        (1e200, 1e-300, 1e100),
        (1e-200, 1e100, 1e-300),
    ],
)
def test_quadratic_rows_preserve_extreme_curvature_and_rhs(
    alpha: float,
    confidence: float,
    rho: float,
) -> None:
    distance2 = 0.5 / alpha
    distance = np.sqrt(distance2)
    points = np.array([[0.0, 0.0], [distance, 0.0]])
    target = 0.8
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, target)],
        confidence=np.array([confidence]),
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                rho,
                np.zeros(2),
            )
        ),
    )
    expected_row_rhs = confidence * alpha * (target - 0.5)
    expected_weight = (target - 0.5) / (3.0 * alpha)

    np.testing.assert_allclose(problem.alpha, [alpha], rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(
        problem.observation_graph.rho,
        [rho],
        rtol=4e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        problem.quadratic_operator.observation_rhs,
        [expected_row_rhs, -expected_row_rhs],
        rtol=4e-15,
        atol=0.0,
    )

    result = separator.fit_weights_from_separators(
        points,
        observations,
        model=problem.model,
        solver='direct',
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    np.testing.assert_allclose(
        result.weights,
        [expected_weight, -expected_weight],
        rtol=5e-15,
        atol=0.0,
    )


def test_large_finite_distance_keeps_alpha_prediction_and_operator_rhs() -> None:
    points = np.array([[0.0, 0.0], [1e154, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.75)],
    )
    with np.errstate(all='raise'):
        problem = separator.build_power_fit_problem(observations)
        prediction = problem.predict(np.array([5e307, -5e307]))
    np.testing.assert_allclose(problem.alpha, [5e-309], rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(prediction.fraction, [1.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        problem.quadratic_operator.observation_rhs,
        [1.25e-309, -1.25e-309],
        rtol=2e-15,
        atol=0.0,
    )


def test_quadratic_rhs_does_not_depend_on_nonfinite_z_obs() -> None:
    alpha = 1e-200
    distance = np.sqrt(0.5 / alpha)
    confidence = 1e100
    target = 1e200
    points = np.array([[0.0, 0.0], [distance, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, target)],
        confidence=np.array([confidence]),
    )
    problem = separator.build_power_fit_problem(observations)
    assert np.isinf(problem.z_obs[0])
    expected_rhs = confidence * alpha * (target - 0.5)
    np.testing.assert_allclose(
        problem.quadratic_operator.observation_rhs,
        [expected_rhs, -expected_rhs],
        rtol=4e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ('side', 'target', 'expected_feasible'),
    [
        (1e6, 0.5 + 5e-13, True),
        (1e-6, 0.55, False),
    ],
)
def test_measurement_accepted_set_is_mapped_into_hard_precheck(
    side: float,
    target: float,
    expected_feasible: bool,
) -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [side, 0.0],
            [0.5 * side, np.sqrt(3.0) * 0.5 * side],
        ]
    )
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, target), (1, 2, target), (2, 0, target)],
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            feasible=separator.FixedValue(target)
        ),
    )
    breakdown = problem.objective_breakdown(np.zeros(3))
    assert problem.hard_feasible is expected_feasible
    assert breakdown.hard_constraints_satisfied is expected_feasible

    if expected_feasible:
        result = separator.build_power_fit_result(
            problem,
            np.zeros(3),
            solver='external',
        )
        assert result.status == 'optimal'
        assert result.objective_breakdown.hard_constraints_satisfied is True
    else:
        result = separator.fit_weights_from_separators(
            points,
            observations,
            model=problem.model,
            solver='admm',
            connectivity_check='diagnose',
        )
        assert result.status == 'infeasible_hard_constraints'
        assert result.converged is False


def test_nonfinite_hard_rows_are_never_satisfied() -> None:
    satisfied, _, _ = _hard_row_status(
        np.array([0.0, np.inf]),
        np.array([np.inf, np.inf]),
        np.array([1.0, np.inf]),
    )
    np.testing.assert_array_equal(satisfied, np.array([False, False]))


def test_tiny_reciprocal_value_and_derivatives_are_scale_safe() -> None:
    penalty = separator.ReciprocalBoundaryPenalty(
        lower=0.0,
        upper=1.0,
        margin=1e-100,
        epsilon=1e-200,
        strength=1e-300,
    )
    measurement = np.array([0.0])
    with np.errstate(all='raise'):
        value = _penalty_values(measurement, penalty)
    assert value[0] == pytest.approx(2e-100, rel=3e-15)

    with np.errstate(all='raise'):
        first, second = _penalty_derivatives(measurement, penalty)
    assert first[0] == pytest.approx(-1e100, rel=3e-15)
    assert second[0] == 0.0


def test_admm_success_requires_authoritative_hard_satisfaction() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = separator.FitModel(
        feasible=separator.FixedValue(0.9),
        regularization=separator.L2Regularization(
            1.0,
            np.zeros(2),
        ),
    )
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.1)],
        model=model,
        solver='admm',
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    assert result.converged is True
    assert result.objective_breakdown.hard_constraints_satisfied is True
    assert (
        result.objective_breakdown.hard_max_violation
        <= result.objective_breakdown.hard_max_tolerance
    )

    limited = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.1)],
        model=model,
        solver='admm',
        admm_max_iter=30,
        connectivity_check='diagnose',
    )
    assert limited.status == 'max_iter'
    assert limited.converged is False
    assert limited.objective_breakdown.hard_constraints_satisfied is False


def test_stable_reductions_preserve_large_finite_norms_and_edge_metrics() -> None:
    assert _stable_norm(np.array([1e160])) == pytest.approx(1e160)

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 1e200), (0, 1, -1e200)],
        confidence=np.array([1e-300, 1e-300]),
        solver='direct',
        connectivity_check='diagnose',
    )
    assert result.rms_residual == pytest.approx(1e200, rel=3e-15)
    assert np.isfinite(result.edge_diagnostics.weighted_l2)
    assert np.isfinite(result.edge_diagnostics.weighted_rmse)
    assert result.edge_diagnostics.rmse == pytest.approx(2e200, rel=3e-15)
    assert result.edge_diagnostics.mae == pytest.approx(2e200, rel=3e-15)


def test_huber_linear_value_combines_terms_before_range_conversion() -> None:
    delta = 1.4e154
    residual = np.nextafter(delta, np.inf)
    maximum = np.finfo(np.float64).max
    measurements = np.array([residual, maximum])
    targets = np.array([0.0, -maximum])
    confidence = np.array([1.0, 1e-320])
    with localcontext() as context:
        context.prec = 1000
        expected = np.array(
            [
                float(
                    Decimal.from_float(float(row_confidence))
                    * Decimal.from_float(delta)
                    * (
                        abs(
                            Decimal.from_float(float(measurement))
                            - Decimal.from_float(float(target))
                        )
                        - Decimal('0.5') * Decimal.from_float(delta)
                    )
                )
                for measurement, target, row_confidence in zip(
                    measurements,
                    targets,
                    confidence,
                )
            ]
        )

    with np.errstate(all='raise'):
        actual = _mismatch_values(
            measurements,
            targets,
            confidence,
            separator.HuberLoss(delta),
        )

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=0.0)


def test_positive_soft_interval_uses_scaled_displacement() -> None:
    maximum = np.finfo(np.float64).max
    strength = 1e-320
    upper = -maximum / 2.0
    penalty = separator.SoftIntervalPenalty(
        lower=-maximum,
        upper=upper,
        strength=strength,
    )
    with localcontext() as context:
        context.prec = 1000
        displacement = (
            Decimal.from_float(maximum)
            - Decimal.from_float(upper)
        )
        expected_value = float(
            Decimal.from_float(strength) * displacement * displacement
        )
        expected_first = float(
            Decimal(2) * Decimal.from_float(strength) * displacement
        )

    with np.errstate(all='raise'):
        value = _penalty_values(np.array([maximum]), penalty)
        first, second = _penalty_derivatives(np.array([maximum]), penalty)

    assert value[0] == pytest.approx(expected_value, rel=4e-15)
    assert first[0] == pytest.approx(expected_first, rel=4e-15)
    assert second[0] == 2.0 * strength


def test_reciprocal_combines_overflowing_quotient_difference() -> None:
    maximum = np.finfo(np.float64).max
    epsilon = 0.5
    margin = np.nextafter(epsilon, np.inf)
    penalty = separator.ReciprocalBoundaryPenalty(
        lower=0.0,
        upper=2.0,
        margin=margin,
        epsilon=epsilon,
        strength=maximum,
    )
    with localcontext() as context:
        context.prec = 1000
        expected = float(
            Decimal.from_float(maximum)
            * (
                Decimal.from_float(margin)
                - Decimal.from_float(epsilon)
            )
            / Decimal.from_float(epsilon)
            / Decimal.from_float(margin)
        )

    with np.errstate(all='raise'):
        actual = _penalty_values(np.array([epsilon]), penalty)

    assert np.isfinite(actual[0])
    assert actual[0] == pytest.approx(expected, rel=4e-15)

    continuation_penalty = separator.ReciprocalBoundaryPenalty(
        lower=-maximum,
        upper=-maximum / 2.0,
        margin=0.5,
        epsilon=0.25,
        strength=1e-320,
    )
    with localcontext() as context:
        context.prec = 1000
        distance = (
            Decimal.from_float(-maximum / 2.0)
            - Decimal.from_float(maximum)
        )
        expected_continuation = float(
            Decimal.from_float(1e-320)
            * (
                Decimal(1) / Decimal.from_float(0.25)
                - Decimal(1) / Decimal.from_float(0.5)
                - (
                    distance - Decimal.from_float(0.25)
                )
                / Decimal.from_float(0.25) ** 2
            )
        )
    with np.errstate(all='raise'):
        continuation = _penalty_values(
            np.array([maximum]),
            continuation_penalty,
        )
    assert continuation[0] == pytest.approx(
        expected_continuation,
        rel=4e-15,
    )


def test_direct_affine_residual_survives_nonrepresentable_prediction() -> None:
    maximum = np.finfo(np.float64).max
    distance = np.nextafter(np.sqrt(1.0 / maximum), 0.0)
    points = np.array([[0.0, 0.0], [distance, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, maximum)],
        measurement='fraction',
        confidence=np.array([1e-320]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([2.0, 0.0])
    with localcontext() as context:
        context.prec = 1000
        expected_residual = float(
            Decimal.from_float(float(problem.beta[0]))
            + Decimal.from_float(float(problem.alpha[0]))
            * Decimal.from_float(float(weights[0]))
            - Decimal.from_float(float(problem.alpha[0]))
            * Decimal.from_float(float(weights[1]))
            - Decimal.from_float(maximum)
        )
        expected_mismatch = float(
            Decimal('0.5')
            * Decimal.from_float(1e-320)
            * Decimal.from_float(expected_residual)
            * Decimal.from_float(expected_residual)
        )

    with np.errstate(all='raise'):
        prediction = problem.predict(weights)
        breakdown = problem.objective_breakdown(weights)
        result = separator.build_power_fit_result(
            problem,
            weights,
            canonicalize_gauge=False,
        )

    assert np.isinf(prediction.measurement[0])
    assert np.isfinite(result.residuals[0])
    assert result.residuals[0] == pytest.approx(expected_residual, rel=4e-15)
    assert breakdown.mismatch == pytest.approx(expected_mismatch, rel=8e-15)
    assert result.objective_breakdown.mismatch == breakdown.mismatch
    assert np.isfinite(result.edge_diagnostics.weighted_l2)


def test_squared_proximal_weighted_average_preserves_float64_max() -> None:
    maximum = np.finfo(np.float64).max
    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            np.array([maximum]),
            np.array([maximum]),
            np.array([1.0]),
            separator.SquaredLoss(),
            1.0,
        )
    assert actual[0] == maximum


def test_quadratic_operator_matvec_scales_before_subtracting() -> None:
    maximum = np.finfo(np.float64).max
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, 0.5)],
        confidence=np.array([4e-320]),
    )
    operator = separator.build_power_fit_problem(
        observations
    ).quadratic_operator
    vector = np.array([maximum, -maximum])
    dense = operator.observation_laplacian_dense()
    with np.errstate(all='raise'):
        expected = dense @ vector
        actual = operator.observation_laplacian_matvec(vector)
        regularized = operator.regularized_normal_matvec(vector)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.0)
    np.testing.assert_array_equal(regularized, actual)

    scipy = pytest.importorskip('scipy')
    assert scipy is not None
    with np.errstate(all='raise'):
        sparse_actual = operator.observation_laplacian_sparse() @ vector
    np.testing.assert_allclose(
        sparse_actual,
        expected,
        rtol=3e-15,
        atol=0.0,
    )

    regularized_operator = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                1e-320,
                np.zeros(2),
            )
        ),
    ).quadratic_operator
    with np.errstate(all='raise'):
        regularized_expected = (
            regularized_operator.regularized_normal_matrix_dense() @ vector
        )
        regularized_actual = (
            regularized_operator.regularized_normal_matvec(vector)
        )
    np.testing.assert_allclose(
        regularized_actual,
        regularized_expected,
        rtol=3e-15,
        atol=0.0,
    )


def test_admm_projects_onto_authoritative_accepted_hard_interval() -> None:
    side = 1e6
    target = 0.5 + 5e-13
    points = np.array(
        [
            [0.0, 0.0],
            [side, 0.0],
            [0.5 * side, np.sqrt(3.0) * 0.5 * side],
        ]
    )
    rows = [(0, 1, target), (1, 2, target), (2, 0, target)]
    model = separator.FitModel(feasible=separator.FixedValue(target))
    problem = separator.build_power_fit_problem(
        separator.resolve_separator_observations(points, rows),
        model=model,
    )
    assert problem.hard_feasible is True
    assert problem.objective_breakdown(
        np.zeros(3)
    ).hard_constraints_satisfied is True

    result = separator.fit_weights_from_separators(
        points,
        rows,
        model=model,
        solver='admm',
        admm_abs_tol=1e-13,
        admm_rel_tol=1e-13,
        admm_max_iter=5000,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    assert result.objective_breakdown.hard_constraints_satisfied is True
    np.testing.assert_allclose(result.weights, np.zeros(3), atol=1e-12)


def test_vectorized_ordinary_mismatch_matches_manual_row_oracle() -> None:
    measurement = np.linspace(-2.0, 3.0, 1000)
    target = np.linspace(1.0, -1.0, 1000)
    confidence = np.linspace(0.0, 2.0, 1000)
    residual = measurement - target
    expected_squared = 0.5 * confidence * residual**2
    delta = 0.7
    expected_huber = confidence * np.where(
        np.abs(residual) <= delta,
        0.5 * residual**2,
        delta * (np.abs(residual) - 0.5 * delta),
    )

    squared = _mismatch_values(
        measurement,
        target,
        confidence,
        separator.SquaredLoss(),
    )
    huber = _mismatch_values(
        measurement,
        target,
        confidence,
        separator.HuberLoss(delta),
    )

    np.testing.assert_allclose(squared, expected_squared, rtol=2e-15)
    np.testing.assert_allclose(huber, expected_huber, rtol=2e-15)


def test_direct_affine_residual_detects_finite_destructive_cancellation() -> None:
    distance = 1e150
    points = np.array([[0.0, 0.0], [distance, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.0)],
        measurement='position',
        confidence=np.array([1e-300]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([1.0, 1e300])
    with localcontext() as context:
        context.prec = 1000
        residual = (
            _decimal(problem.beta[0])
            + _decimal(problem.alpha[0]) * _decimal(weights[0])
            - _decimal(problem.alpha[0]) * _decimal(weights[1])
            - _decimal(problem.measurement_target[0])
        )
        expected_residual = float(residual)
        expected_mismatch = float(
            Decimal('0.5') * _decimal(1e-300) * residual * residual
        )

    with np.errstate(all='raise'):
        result = separator.build_power_fit_result(
            problem,
            weights,
            canonicalize_gauge=False,
        )

    assert result.residuals[0] == pytest.approx(
        expected_residual,
        rel=4e-15,
    )
    assert result.objective_breakdown.mismatch == pytest.approx(
        expected_mismatch,
        rel=8e-15,
    )


def test_scalar_penalties_use_direct_affine_boundary_distances() -> None:
    maximum = np.finfo(np.float64).max
    distance = np.nextafter(np.sqrt(1.0 / maximum), 0.0)
    points = np.array([[0.0, 0.0], [distance, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        measurement='fraction',
        confidence=np.array([0.0]),
    )
    weights = np.array([2.0, 0.0])
    base_problem = separator.build_power_fit_problem(observations)
    with localcontext() as context:
        context.prec = 1000
        prediction = (
            _decimal(base_problem.beta[0])
            + _decimal(base_problem.alpha[0]) * _decimal(weights[0])
            - _decimal(base_problem.alpha[0]) * _decimal(weights[1])
        )
        upper_distance = _decimal(maximum) - prediction
        soft_expected = float(
            _decimal(1e-300)
            * (prediction - _decimal(maximum))
            * (prediction - _decimal(maximum))
        )
        reciprocal_expected = float(
            _decimal(1e-300)
            * (
                Decimal(1) / _decimal(0.25)
                - Decimal(1) / _decimal(0.5)
                - (upper_distance - _decimal(0.25))
                / (_decimal(0.25) * _decimal(0.25))
            )
        )

    penalties = (
        (
            separator.SoftIntervalPenalty(
                lower=-maximum,
                upper=maximum,
                strength=1e-300,
            ),
            soft_expected,
        ),
        (
            separator.ReciprocalBoundaryPenalty(
                lower=-maximum,
                upper=maximum,
                margin=0.5,
                epsilon=0.25,
                strength=1e-300,
            ),
            reciprocal_expected,
        ),
    )
    for penalty, expected in penalties:
        problem = separator.build_power_fit_problem(
            observations,
            model=separator.FitModel(penalties=(penalty,)),
        )
        with np.errstate(all='raise'):
            prediction_record = problem.predict(weights)
            breakdown = problem.objective_breakdown(weights)
        assert np.isinf(prediction_record.measurement[0])
        assert breakdown.penalties_total == pytest.approx(
            expected,
            rel=8e-15,
        )
        assert breakdown.total == breakdown.penalties_total


@pytest.mark.parametrize(
    ('value', 'target', 'rho', 'confidence'),
    [
        (
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            1.0,
            np.nextafter(1.0, 0.0),
        ),
        (
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            1.0,
            np.nextafter(1.0, np.inf),
        ),
        (
            -np.finfo(np.float64).max,
            np.finfo(np.float64).max,
            np.nextafter(1.0, 0.0),
            1.0,
        ),
        (
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            np.nextafter(1.0, np.inf),
            1.0,
        ),
    ],
)
def test_squared_proximal_preserves_near_balanced_extreme_average(
    value: float,
    target: float,
    rho: float,
    confidence: float,
) -> None:
    exact = (
        Fraction.from_float(rho) * Fraction.from_float(value)
        + Fraction.from_float(confidence) * Fraction.from_float(target)
    ) / (
        Fraction.from_float(rho) + Fraction.from_float(confidence)
    )
    expected = float(exact)

    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            np.array([value]),
            np.array([target]),
            np.array([confidence]),
            separator.SquaredLoss(),
            rho,
        )

    assert actual[0] == expected


def test_incidence_and_normal_rhs_preserve_small_cancellation_residual() -> None:
    radius = np.sqrt(0.5)
    points = np.array(
        [
            [0.0, 0.0],
            [radius, 0.0],
            [0.0, radius],
            [-radius, 0.0],
        ]
    )
    rows = [(0, 1, 0.5), (0, 2, 0.5), (0, 3, 0.5)]
    observations = separator.resolve_separator_observations(
        points,
        rows,
        confidence=np.ones(3),
    )
    problem = separator.build_power_fit_problem(observations)
    vector = np.array([0.0, -1e300, -1.0, 1e300])
    graph = problem.observation_graph
    exact_edge_values = [
        Fraction.from_float(float(rho))
        * (
            Fraction.from_float(float(vector[site_i]))
            - Fraction.from_float(float(vector[site_j]))
        )
        for site_i, site_j, rho in zip(
            graph.site_i,
            graph.site_j,
            graph.rho,
        )
    ]
    exact_sites = [Fraction(0) for _ in range(4)]
    for site_i, site_j, value in zip(
        graph.site_i,
        graph.site_j,
        exact_edge_values,
    ):
        exact_sites[int(site_i)] += value
        exact_sites[int(site_j)] -= value
    expected = np.array([float(value) for value in exact_sites])
    with np.errstate(all='raise'):
        actual = problem.quadratic_operator.observation_laplacian_matvec(
            vector
        )
        accumulated = _stable_incidence_accumulate(
            4,
            np.array([0, 0, 0]),
            np.array([1, 2, 3]),
            np.array([1e300, 1.0, -1e300]),
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        accumulated,
        np.array([1.0, -1e300, -1.0, 1e300]),
    )

    rhs_observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 1e300), (0, 2, 1.5), (0, 3, -1e300)],
        confidence=np.ones(3),
    )
    with np.errstate(all='raise'):
        rhs_problem = separator.build_power_fit_problem(rhs_observations)
        rhs = rhs_problem.quadratic_operator.observation_rhs
    exact_rhs_rows = [
        Fraction.from_float(
            float(
                Fraction.from_float(float(confidence))
                * Fraction.from_float(float(alpha))
                * (
                    Fraction.from_float(float(target))
                    - Fraction.from_float(float(beta))
                )
            )
        )
        for confidence, alpha, target, beta in zip(
            rhs_problem.constraints.confidence,
            rhs_problem.alpha,
            rhs_problem.measurement_target,
            rhs_problem.beta,
        )
    ]
    exact_rhs_sites = [Fraction(0) for _ in range(4)]
    for site_i, site_j, value in zip(
        rhs_problem.constraints.i,
        rhs_problem.constraints.j,
        exact_rhs_rows,
    ):
        exact_rhs_sites[int(site_i)] += value
        exact_rhs_sites[int(site_j)] -= value
    expected_rhs = np.array([float(value) for value in exact_rhs_sites])
    np.testing.assert_array_equal(rhs, expected_rhs)
    assert rhs[0] != 0.0


def test_exceptional_finite_sums_preserve_final_small_term() -> None:
    maximum = np.finfo(np.float64).max
    values = (maximum, maximum, -maximum, -maximum, 1.0)
    with np.errstate(all='raise'):
        scalar = _stable_sum_scalar(*values)
        vectorized = _stable_sum(
            *(np.array([value]) for value in values)
        )
    assert scalar == 1.0
    assert vectorized[0] == 1.0


@pytest.mark.parametrize(
    'bound',
    [np.finfo(np.float64).max, -np.finfo(np.float64).max],
)
def test_extreme_hard_accepted_bounds_do_not_overflow(bound: float) -> None:
    original = np.array([bound])
    with np.errstate(all='raise'):
        accepted_lower, accepted_upper = (
            _hard_accepted_measurement_bounds(original, original)
        )
        lower_satisfied = _hard_row_status(
            original,
            accepted_lower,
            original,
        )[0]
        upper_satisfied = _hard_row_status(
            original,
            accepted_upper,
            original,
        )[0]

    assert np.all(np.isfinite(accepted_lower))
    assert np.all(np.isfinite(accepted_upper))
    assert accepted_lower[0] <= accepted_upper[0]
    assert lower_satisfied[0]
    assert upper_satisfied[0]


@pytest.mark.parametrize(
    'mismatch',
    [
        separator.SquaredLoss(),
        separator.HuberLoss(np.finfo(np.float64).max),
    ],
)
def test_minimum_positive_confidence_preserves_weighted_quadratic_value(
    mismatch,
) -> None:
    maximum = np.finfo(np.float64).max
    confidence = np.nextafter(0.0, 1.0)
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * Fraction.from_float(maximum)
        * Fraction.from_float(maximum)
    )

    with np.errstate(all='raise'):
        actual = _mismatch_values(
            np.array([maximum]),
            np.array([0.0]),
            np.array([confidence]),
            mismatch,
        )

    assert actual[0] == expected


def test_minimum_confidence_public_objective_and_external_result_are_nonzero(
) -> None:
    maximum = np.finfo(np.float64).max
    confidence = np.nextafter(0.0, 1.0)
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, -maximum)],
        measurement='fraction',
        confidence=np.array([confidence]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.zeros(2)
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        - Fraction.from_float(float(problem.measurement_target[0]))
    )
    expected = float(
        Fraction(1, 2)
        * Fraction.from_float(confidence)
        * exact_residual
        * exact_residual
    )

    with np.errstate(all='raise'):
        breakdown = problem.objective_breakdown(weights)
        result = separator.build_power_fit_result(
            problem,
            weights,
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert breakdown.mismatch == expected
    assert breakdown.total == expected
    assert result.objective_breakdown.mismatch == expected
    assert result.objective_breakdown.total == expected
    assert expected > 0.0


@pytest.mark.parametrize(
    'mismatch',
    [
        separator.SquaredLoss(),
        separator.HuberLoss(1.4e154),
    ],
)
def test_direct_affine_mismatch_value_survives_nonrepresentable_residual(
    mismatch,
) -> None:
    maximum = np.finfo(np.float64).max
    confidence = 1e-320
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, -maximum)],
        measurement='fraction',
        confidence=np.array([confidence]),
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(mismatch=mismatch),
    )
    weights = np.array([maximum, -maximum])
    with localcontext() as context:
        context.prec = 1000
        residual = (
            _decimal(problem.beta[0])
            + _decimal(problem.alpha[0]) * _decimal(weights[0])
            - _decimal(problem.alpha[0]) * _decimal(weights[1])
            - _decimal(problem.measurement_target[0])
        )
        if isinstance(mismatch, separator.SquaredLoss):
            expected = float(
                Decimal('0.5')
                * _decimal(confidence)
                * residual
                * residual
            )
        else:
            delta = _decimal(mismatch.delta)
            expected = float(
                _decimal(confidence)
                * delta
                * (abs(residual) - Decimal('0.5') * delta)
            )

    with np.errstate(all='raise'):
        prediction = problem.predict(weights)
        breakdown = problem.objective_breakdown(weights)

    assert np.isfinite(prediction.measurement[0])
    assert breakdown.mismatch == pytest.approx(expected, rel=8e-15)
    assert np.isfinite(breakdown.mismatch)


def test_analytic_fit_with_nonrepresentable_residual_has_backend_radii() -> None:
    maximum = np.finfo(np.float64).max
    points = np.array([[0.0, 0.0], [0.5, 0.0]])
    reference = np.array([maximum / 4.0, -maximum / 4.0])
    confidence = np.array([1e-320])
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=1.0,
            reference=reference,
        )
    )
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, -maximum)],
        measurement='fraction',
        confidence=confidence,
    )
    problem = separator.build_power_fit_problem(observations, model=model)
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(reference[0]))
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(reference[1]))
        - Fraction.from_float(-maximum)
    )
    expected_objective = float(
        Fraction(1, 2)
        * Fraction.from_float(float(confidence[0]))
        * exact_residual
        * exact_residual
    )

    with np.errstate(all='raise'):
        result = separator.fit_weights_from_separators(
            points,
            [(0, 1, -maximum)],
            measurement='fraction',
            confidence=confidence,
            model=model,
            solver='direct',
        )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_array_equal(result.weights, reference)
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(
        expected_objective,
        rel=8e-15,
    )
    assert result.radii is not None
    assert result.weight_shift is not None
    assert np.all(np.isfinite(result.radii))
    assert np.isfinite(result.weight_shift)


@pytest.mark.parametrize(
    ('left', 'left_weight', 'right', 'right_weight'),
    [
        (1e20, 1e-20, 1.0, 1e20),
        (1.0, 1e20, 1e20, 1e-20),
        (
            0.0,
            1e308,
            np.finfo(np.float64).max,
            np.nextafter(0.0, 1.0),
        ),
        (
            np.finfo(np.float64).max,
            np.nextafter(0.0, 1.0),
            0.0,
            1e308,
        ),
        (1e300, 3e-15, 0.0, 1.0),
        (0.0, 1.0, 1e300, 3e-15),
        (1e300, np.nextafter(3e-15, 0.0), 0.0, 1.0),
        (0.0, 1.0, 1e300, np.nextafter(3e-15, np.inf)),
    ],
)
def test_weighted_average_preserves_complete_same_sign_expression(
    left: float,
    left_weight: float,
    right: float,
    right_weight: float,
) -> None:
    expected = float(
        (
            Fraction.from_float(left_weight) * Fraction.from_float(left)
            + Fraction.from_float(right_weight) * Fraction.from_float(right)
        )
        / (
            Fraction.from_float(left_weight)
            + Fraction.from_float(right_weight)
        )
    )

    with np.errstate(all='raise'):
        actual = _stable_weighted_average(
            np.array([left]),
            np.array([left_weight]),
            np.array([right]),
            np.array([right_weight]),
        )

    assert actual[0] == expected


@pytest.mark.parametrize(
    'mismatch',
    [
        separator.SquaredLoss(),
        separator.HuberLoss(np.finfo(np.float64).max),
    ],
)
@pytest.mark.parametrize(
    ('value', 'rho', 'target', 'confidence'),
    [
        (1e20, 1e-20, 1.0, 1e20),
        (
            0.0,
            1e308,
            np.finfo(np.float64).max,
            np.nextafter(0.0, 1.0),
        ),
        (1e300, 3e-15, 0.0, 1.0),
        (0.0, 1.0, 1e300, 3e-15),
        (1e300, np.nextafter(3e-15, 0.0), 0.0, 1.0),
        (0.0, 1.0, 1e300, np.nextafter(3e-15, np.inf)),
    ],
)
def test_mismatch_proximal_uses_complete_weighted_average(
    mismatch,
    value: float,
    rho: float,
    target: float,
    confidence: float,
) -> None:
    expected = float(
        (
            Fraction.from_float(rho) * Fraction.from_float(value)
            + Fraction.from_float(confidence) * Fraction.from_float(target)
        )
        / (
            Fraction.from_float(rho)
            + Fraction.from_float(confidence)
        )
    )

    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            np.array([value]),
            np.array([target]),
            np.array([confidence]),
            mismatch,
            rho,
        )

    assert actual[0] == expected


def test_direct_affine_reciprocal_objective_overflow_is_not_saturated() -> None:
    maximum = np.finfo(np.float64).max
    distance = np.nextafter(np.sqrt(1.0 / maximum), 0.0)
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [distance, 0.0]]),
        [(0, 1, 0.5)],
        measurement='fraction',
        confidence=np.array([0.0]),
    )
    penalty = separator.ReciprocalBoundaryPenalty(
        lower=-maximum,
        upper=maximum,
        margin=0.5,
        epsilon=0.25,
        strength=maximum,
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(penalties=(penalty,)),
    )
    weights = np.array([4.0, 0.0])

    with np.errstate(all='raise'):
        breakdown = problem.objective_breakdown(weights)

    assert breakdown.penalties_total == float('inf')
    assert breakdown.total == float('inf')
    with pytest.raises(
        ValueError,
        match='non-finite soft objective',
    ):
        separator.build_power_fit_result(
            problem,
            weights,
            status='optimal',
            converged=True,
            canonicalize_gauge=False,
        )


def test_subnormal_products_and_public_objective_do_not_raise() -> None:
    alpha_target = 4.185525585196824e-309
    weight = 2.7854831801759516
    distance = np.sqrt(0.5 / alpha_target)
    maximum = np.finfo(np.float64).max
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [distance, 0.0]]),
        [(0, 1, 0.5)],
        measurement='fraction',
        confidence=np.array([maximum]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([weight, 0.0])
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(weight)
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(0.0)
        - Fraction.from_float(float(problem.measurement_target[0]))
    )
    expected_residual = float(exact_residual)
    expected_objective = float(
        Fraction(1, 2)
        * Fraction.from_float(maximum)
        * exact_residual
        * exact_residual
    )

    with np.errstate(all='raise'):
        product = _stable_product(
            np.array([problem.alpha[0]]),
            np.array([weight]),
        )
        result = separator.build_power_fit_result(
            problem,
            weights,
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert product[0] == expected_residual
    assert result.residuals[0] == expected_residual
    assert result.objective_breakdown.mismatch == expected_objective


def test_subnormal_quadratic_rhs_and_cancellation_bound_do_not_raise() -> None:
    alpha_target = 4.185525585196824e-309
    factor = 2.7854831801759516
    distance = np.sqrt(0.5 / alpha_target)
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [distance, 0.0]]),
        [(0, 1, 1.5)],
        measurement='fraction',
        confidence=np.array([factor]),
    )
    expected_row_rhs = float(
        Fraction.from_float(factor)
        * Fraction.from_float(alpha_target)
    )
    values = (1e-308, -5e-309, 1e-323)
    expected_sum = float(sum(Fraction.from_float(value) for value in values))

    with np.errstate(all='raise'):
        problem = separator.build_power_fit_problem(observations)
        rhs = problem.quadratic_operator.observation_rhs
        summed = _stable_sum(
            *(np.array([value]) for value in values)
        )

    np.testing.assert_array_equal(
        rhs,
        np.array([expected_row_rhs, -expected_row_rhs]),
    )
    assert summed[0] == expected_sum


def test_material_affine_cancellation_uses_exact_exceptional_result() -> None:
    points = np.array([[0.0, 0.0], [0.5, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 1e150)],
        measurement='fraction',
        confidence=np.array([1.0]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([5e149, -5e135])
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[0]))
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[1]))
        - Fraction.from_float(float(problem.measurement_target[0]))
    )
    expected_residual = float(exact_residual)
    expected_mismatch = float(Fraction(1, 2) * exact_residual * exact_residual)

    with np.errstate(all='raise'):
        breakdown = problem.objective_breakdown(weights)
        result = separator.build_power_fit_result(
            problem,
            weights,
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert result.residuals is not None
    assert result.residuals[0] == expected_residual
    assert breakdown.mismatch == expected_mismatch
    assert breakdown.mismatch == 0.5 * result.residuals[0] ** 2


def test_material_position_cancellation_uses_exact_exceptional_result() -> None:
    points = np.array([[0.0, 0.0], [0.5, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 1e150)],
        measurement='position',
        confidence=np.array([1.0]),
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([1e150, -1e136])
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[0]))
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[1]))
        - Fraction.from_float(float(problem.measurement_target[0]))
    )
    expected_residual = float(exact_residual)
    expected_mismatch = float(Fraction(1, 2) * exact_residual * exact_residual)

    with np.errstate(all='raise'):
        result = separator.build_power_fit_result(
            problem,
            weights,
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert result.residuals is not None
    assert result.objective_breakdown is not None
    assert result.residuals[0] == expected_residual
    assert result.objective_breakdown.mismatch == expected_mismatch
    assert (
        result.objective_breakdown.mismatch
        == 0.5 * result.residuals[0] ** 2
    )


def test_material_incidence_cancellation_uses_exact_site_sum() -> None:
    radius = np.sqrt(0.5)
    points = np.array(
        [
            [0.0, 0.0],
            [radius, 0.0],
            [0.0, radius],
            [-radius, 0.0],
        ]
    )
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.5), (0, 2, 0.5), (0, 3, 0.5)],
        confidence=np.ones(3),
    )
    problem = separator.build_power_fit_problem(observations)
    x = 2.3049249849969992e285
    vector = np.array([0.0, -1e300, -x, 1e300])
    graph = problem.observation_graph
    exact_rows = [
        Fraction.from_float(float(rho))
        * (
            Fraction.from_float(float(vector[site_i]))
            - Fraction.from_float(float(vector[site_j]))
        )
        for site_i, site_j, rho in zip(
            graph.site_i,
            graph.site_j,
            graph.rho,
        )
    ]
    exact_sites = [Fraction(0) for _ in range(4)]
    for site_i, site_j, row in zip(
        graph.site_i,
        graph.site_j,
        exact_rows,
    ):
        exact_sites[int(site_i)] += row
        exact_sites[int(site_j)] -= row
    expected = np.array([float(value) for value in exact_sites])
    expected_accumulated = np.array([x, -1e300, -x, 1e300])

    with np.errstate(all='raise'):
        actual = problem.quadratic_operator.observation_laplacian_matvec(
            vector
        )
        accumulated = _stable_incidence_accumulate(
            4,
            np.array([0, 0, 0]),
            np.array([1, 2, 3]),
            np.array([1e300, x, -1e300]),
        )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(accumulated, expected_accumulated)


def test_huber_affine_value_preserves_aggregate_subnormal_product() -> None:
    minimum = np.nextafter(0.0, 1.0)
    observations = separator.resolve_separator_observations(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, 0.5)],
        measurement='fraction',
        confidence=np.array([1e-300]),
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            mismatch=separator.HuberLoss(minimum),
        ),
    )

    with np.errstate(all='raise'):
        breakdown = problem.objective_breakdown(
            np.array([0.0, 1e300])
        )

    assert breakdown.mismatch == minimum
    assert breakdown.total == minimum


def test_huber_proximal_shift_uses_complete_ratio_product() -> None:
    minimum = np.nextafter(0.0, 1.0)

    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            np.array([0.0]),
            np.array([10.0]),
            np.array([minimum]),
            separator.HuberLoss(1.0),
            minimum,
        )

    assert actual[0] == 1.0


def test_admm_huber_minimum_confidence_keeps_finite_iterates() -> None:
    minimum = np.nextafter(0.0, 1.0)
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = separator.FitModel(
        mismatch=separator.HuberLoss(1.0),
        regularization=separator.L2Regularization(
            strength=1.0,
            reference=np.zeros(2),
        ),
    )

    with np.errstate(all='raise'):
        result = separator.fit_weights_from_separators(
            points,
            [(0, 1, 10.0)],
            measurement='fraction',
            confidence=np.array([minimum]),
            model=model,
            solver='admm',
            admm_rho=minimum,
            admm_max_iter=1000,
        )

    assert result.status == 'optimal'
    assert result.converged is True
    assert result.weights is not None
    assert np.all(np.isfinite(result.weights))
    assert result.objective_breakdown is not None
    assert np.isfinite(result.objective_breakdown.total)


def test_minimum_positive_hard_bound_tolerance_does_not_underflow() -> None:
    minimum = np.nextafter(0.0, 1.0)
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = separator.FitModel(
        feasible=separator.Interval(0.0, minimum),
    )

    with np.errstate(all='raise'):
        observations = separator.resolve_separator_observations(
            points,
            [(0, 1, 0.5)],
            measurement='fraction',
        )
        problem = separator.build_power_fit_problem(
            observations,
            model=model,
        )
        weights = np.array([-1.0, 0.0])
        prediction = problem.predict_measurement(weights)
        breakdown = problem.objective_breakdown(weights)

    assert prediction[0] == 0.0
    assert problem.hard_feasible is True
    assert breakdown.hard_constraints_satisfied is True
    assert breakdown.hard_max_violation == 0.0
    assert breakdown.hard_max_tolerance == HARD_ATOL


def test_huber_proximal_classifies_region_without_rounded_boundaries() -> None:
    v = np.array([0.0])
    target = np.array([1e20])
    confidence = np.array([1e18])
    rho = 1.0
    delta = 1.0
    expected = float(
        Fraction.from_float(float(v[0]))
        + (
            Fraction.from_float(float(confidence[0]))
            * Fraction.from_float(delta)
            / Fraction.from_float(rho)
        )
    )

    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            v,
            target,
            confidence,
            separator.HuberLoss(delta),
            rho,
        )

    assert actual[0] == expected


def test_admm_huber_uses_unrounded_proximal_region_oracle() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    target = 1e20
    confidence = 1e18
    model = separator.FitModel(
        mismatch=separator.HuberLoss(1.0),
        regularization=separator.L2Regularization(
            strength=1.0,
            reference=np.zeros(2),
        ),
    )
    expected_weights = np.array([5e17, -5e17])
    beta = Fraction(1, 2)
    alpha = Fraction(1, 2)
    difference = (
        Fraction.from_float(float(expected_weights[0]))
        - Fraction.from_float(float(expected_weights[1]))
    )
    residual = (
        beta
        + alpha * difference
        - Fraction.from_float(target)
    )
    expected_mismatch = Fraction.from_float(confidence) * (
        abs(residual) - Fraction(1, 2)
    )
    expected_l2 = Fraction(1, 2) * sum(
        (
            Fraction.from_float(float(weight))
            * Fraction.from_float(float(weight))
        )
        for weight in expected_weights
    )
    expected_total = float(expected_mismatch + expected_l2)

    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, target)],
        measurement='fraction',
        confidence=np.array([confidence]),
        model=model,
        solver='admm',
        admm_rho=1.0,
        admm_max_iter=10000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-12,
    )

    assert result.status == 'optimal'
    assert result.converged is True
    assert result.weights is not None
    assert result.objective_breakdown is not None
    np.testing.assert_allclose(
        result.weights,
        expected_weights,
        rtol=1e-12,
        atol=0.0,
    )
    assert result.objective_breakdown.total == pytest.approx(
        expected_total,
        rel=2e-15,
    )


def test_reciprocal_branch_uses_exact_operand_distance_comparison() -> None:
    lower = 2.0**-100
    strength = 1e100
    penalty = separator.ReciprocalBoundaryPenalty(
        lower=lower,
        upper=3.0,
        margin=1.0,
        epsilon=0.25,
        strength=strength,
    )
    exact_distance = Fraction(1) - Fraction.from_float(lower)
    strength_fraction = Fraction.from_float(strength)
    expected_value = float(
        strength_fraction * (Fraction(1, 1) / exact_distance - 1)
    )
    expected_first = float(
        -strength_fraction / (exact_distance * exact_distance)
    )
    expected_second = float(
        2 * strength_fraction
        / (exact_distance * exact_distance * exact_distance)
    )

    with np.errstate(all='raise'):
        value = _penalty_values(np.array([1.0]), penalty)
        first, second = _penalty_derivatives(np.array([1.0]), penalty)

    assert value[0] == expected_value
    assert first[0] == expected_first
    assert second[0] == expected_second


def test_direct_affine_reciprocal_uses_exact_branch_distance() -> None:
    points = np.array([[0.0, 0.0], [2.0, 0.0]])
    weights = np.array([-2.0**-99, 2.0**-99])
    strength = 1e100
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.5)],
        measurement='position',
        confidence=np.array([0.0]),
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            penalties=(
                separator.ReciprocalBoundaryPenalty(
                    lower=0.0,
                    upper=2.0,
                    margin=1.0,
                    epsilon=0.25,
                    strength=strength,
                ),
            ),
        ),
    )
    exact_distance = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[0]))
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[1]))
    )
    expected = float(
        Fraction.from_float(strength)
        * (Fraction(1, 1) / exact_distance - 1)
    )

    with np.errstate(all='raise'):
        prediction = problem.predict_measurement(weights)
        breakdown = problem.objective_breakdown(weights)

    assert prediction[0] == 1.0
    assert breakdown.penalties_total == expected
    assert breakdown.total == expected


def test_hard_status_uses_conditioned_complete_affine_prediction() -> None:
    points = np.array([[0.0, 0.0], [1e140, 0.0]])
    weights = np.array(
        [-4.99999850000001e279, 4.99999850000001e279]
    )
    seed = separator.build_power_fit_problem(
        separator.resolve_separator_observations(
            points,
            [(0, 1, 0.0)],
            measurement='position',
        )
    )
    exact_prediction = (
        Fraction.from_float(float(seed.beta[0]))
        + Fraction.from_float(float(seed.alpha[0]))
        * Fraction.from_float(float(weights[0]))
        - Fraction.from_float(float(seed.alpha[0]))
        * Fraction.from_float(float(weights[1]))
    )
    target = float(exact_prediction)
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, target)],
        measurement='position',
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            feasible=separator.FixedValue(target),
        ),
    )
    exact_residual = (
        exact_prediction - Fraction.from_float(target)
    )
    expected_residual = float(exact_residual)
    expected_mismatch = float(
        Fraction(1, 2) * exact_residual * exact_residual
    )

    with np.errstate(all='raise'):
        result = separator.build_power_fit_result(
            problem,
            weights,
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert result.predicted is not None
    assert result.residuals is not None
    assert result.objective_breakdown is not None
    assert result.predicted[0] == target
    assert result.residuals[0] == expected_residual
    assert result.objective_breakdown.mismatch == expected_mismatch
    assert result.objective_breakdown.mismatch == (
        0.5 * result.residuals[0] ** 2
    )
    assert result.objective_breakdown.hard_constraints_satisfied is True


def test_public_edge_mae_omits_irrelevant_underflowing_ratio() -> None:
    maximum = np.finfo(np.float64).max
    alpha = 1e10
    short_distance = np.sqrt(0.5 / alpha)
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [short_distance, 0.0]]
    )
    observations = separator.resolve_separator_observations(
        points,
        [
            (0, 1, maximum / 2.0),
            (0, 2, 0.5000000001),
        ],
        measurement='fraction',
        confidence=np.zeros(2),
    )
    problem = separator.build_power_fit_problem(observations)
    exact_residuals = np.asarray(
        problem.observation_graph.z_obs,
        dtype=np.float64,
    )
    expected = float(
        (
            Fraction.from_float(float(abs(exact_residuals[0])))
            + Fraction.from_float(float(abs(exact_residuals[1])))
        )
        / 2
    )

    with np.errstate(all='raise'):
        result = separator.build_power_fit_result(
            problem,
            np.zeros(3),
            status='external_candidate',
            converged=False,
            canonicalize_gauge=False,
        )

    assert result.edge_diagnostics is not None
    assert result.edge_diagnostics.residual is not None
    np.testing.assert_array_equal(
        result.edge_diagnostics.residual,
        exact_residuals,
    )
    assert result.edge_diagnostics.mae == expected


def test_moderate_affine_cancellation_recomputes_squared_mismatch() -> None:
    points = np.array([[0.0, 0.0], [3.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.0)],
        measurement='position',
    )
    problem = separator.build_power_fit_problem(observations)
    weights = np.array([1_000_001_000_000.0, 1_000_000_000_000.0])
    exact_residual = (
        Fraction.from_float(float(problem.beta[0]))
        + Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[0]))
        - Fraction.from_float(float(problem.alpha[0]))
        * Fraction.from_float(float(weights[1]))
        - Fraction.from_float(float(problem.measurement_target[0]))
    )
    expected_residual = float(exact_residual)
    expected_mismatch = float(
        Fraction(1, 2) * exact_residual * exact_residual
    )

    result = separator.build_power_fit_result(
        problem,
        weights,
        status='external_candidate',
        converged=False,
        canonicalize_gauge=False,
    )

    assert result.residuals is not None
    assert result.objective_breakdown is not None
    assert result.residuals[0] == expected_residual
    assert result.objective_breakdown.mismatch == expected_mismatch
    assert result.objective_breakdown.mismatch == (
        0.5 * result.residuals[0] ** 2
    )


def test_subnormal_compensation_candidate_uses_exact_fallback() -> None:
    beta = float.fromhex('0x1.b6e3d22865634p-1005')
    alpha = beta
    target = float.fromhex('0x1.b6e3cef532300p-1004')
    exact = (
        Fraction.from_float(beta)
        + Fraction.from_float(alpha)
        - Fraction.from_float(target)
    )

    with np.errstate(all='raise'):
        actual = _stable_affine_residual(
            np.array([beta]),
            np.array([alpha]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([target]),
        )

    assert actual[0] == float(exact)
    assert actual[0].hex() == '0x0.0ccccccd00000p-1022'


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_scaled_quadratic_solve_preserves_l2_component_mean(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    reference = np.ones(2)
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.5)],
        confidence=np.array([4e30]),
        model=separator.FitModel(
            regularization=separator.L2Regularization(1.0, reference)
        ),
        **_solver_kwargs(solver_name),
        admm_max_iter=20000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_array_equal(result.weights, reference)
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == 0.0


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_scaled_quadratic_solve_preserves_subnormal_curvature(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    minimum = np.nextafter(0.0, 1.0)
    target = 1e308
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, target)],
        confidence=np.array([minimum]),
    )
    problem = separator.build_power_fit_problem(observations)
    alpha = Fraction.from_float(float(problem.alpha[0]))
    beta = Fraction.from_float(float(problem.beta[0]))
    target_exact = Fraction.from_float(
        float(problem.measurement_target[0])
    )
    displacement = target_exact - beta
    difference = (
        alpha * displacement
        / (alpha * alpha + Fraction(1, 2))
    )
    expected = np.array(
        [float(difference / 2), float(-difference / 2)]
    )
    residual = beta + alpha * Fraction.from_float(expected[0])
    residual -= alpha * Fraction.from_float(expected[1])
    residual -= target_exact
    objective = Fraction.from_float(minimum) * (
        Fraction(1, 2) * residual * residual
        + Fraction(1, 2)
        * (
            Fraction.from_float(expected[0]) ** 2
            + Fraction.from_float(expected[1]) ** 2
        )
    )

    result = separator.fit_weights_from_separators(
        points,
        observations,
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                minimum,
                np.zeros(2),
            )
        ),
        **_solver_kwargs(solver_name),
        admm_max_iter=20000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(result.weights, expected, rtol=2e-15)
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(
        float(objective),
        rel=2e-15,
    )


def test_huber_proximal_uses_exact_sign_below_float_range() -> None:
    minimum = np.nextafter(0.0, 1.0)
    with np.errstate(all='raise'):
        actual = _prox_measurement_mismatch_only(
            np.array([0.5]),
            np.array([-0.5]),
            np.array([minimum]),
            separator.HuberLoss(0.25),
            minimum,
        )
    np.testing.assert_array_equal(actual, np.array([0.25]))


def test_huber_derivatives_classify_exact_residual_boundary() -> None:
    measurement = np.array([1.0])
    target = np.array([-(2.0 ** -100)])
    first, second = _mismatch_derivatives(
        measurement,
        target,
        np.ones(1),
        separator.HuberLoss(1.0),
    )
    np.testing.assert_array_equal(first, np.ones(1))
    np.testing.assert_array_equal(second, np.zeros(1))


def test_huber_admm_uses_unrounded_proximal_branch_tests() -> None:
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, 1e20)],
        confidence=np.array([1e18]),
        model=separator.FitModel(
            mismatch=separator.HuberLoss(1.0),
            regularization=separator.L2Regularization(1.0, np.zeros(2)),
        ),
        solver='admm',
        admm_rho=1.0,
        admm_max_iter=20000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )
    expected = np.array([5e17, -5e17])
    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(result.weights, expected, rtol=2e-10)
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(9.975e37)


@pytest.mark.parametrize(
    ('left', 'left_weight', 'right', 'right_weight'),
    [
        (1.0, 1.6711764316676128e-7, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.6711764316676128e-7),
    ],
)
def test_weighted_average_uses_shared_strict_conditioning_policy(
    left: float,
    left_weight: float,
    right: float,
    right_weight: float,
) -> None:
    exact = (
        Fraction.from_float(left_weight) * Fraction.from_float(left)
        + Fraction.from_float(right_weight) * Fraction.from_float(right)
    ) / (
        Fraction.from_float(left_weight)
        + Fraction.from_float(right_weight)
    )
    expected = float(exact)
    with np.errstate(all='raise'):
        actual = _stable_weighted_average(
            np.array([left]),
            np.array([left_weight]),
            np.array([right]),
            np.array([right_weight]),
        )
        squared = _prox_measurement_mismatch_only(
            np.array([left]),
            np.array([right]),
            np.array([right_weight]),
            separator.SquaredLoss(),
            left_weight,
        )
        huber = _prox_measurement_mismatch_only(
            np.array([left]),
            np.array([right]),
            np.array([right_weight]),
            separator.HuberLoss(2.0),
            left_weight,
        )
    np.testing.assert_allclose(
        (actual[0], squared[0], huber[0]),
        expected,
        rtol=64.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )


def test_operator_and_rhs_use_shared_strict_incidence_policy() -> None:
    radius = np.sqrt(0.5)
    points = np.array(
        [
            [0.0, 0.0],
            [radius, 0.0],
            [0.0, radius],
            [-radius, 0.0],
        ]
    )
    x = 1.4980197934072807e293
    ordinary_rows = [(0, 1, 0.5), (0, 2, 0.5), (0, 3, 0.5)]
    problem = separator.build_power_fit_problem(
        separator.resolve_separator_observations(
            points,
            ordinary_rows,
            confidence=np.ones(3),
        )
    )
    vector = np.array([0.0, -1e300, -x, 1e300])
    row_values = tuple(
        float(
            Fraction.from_float(float(curvature))
            * (
                Fraction.from_float(float(vector[site_i]))
                - Fraction.from_float(float(vector[site_j]))
            )
        )
        for site_i, site_j, curvature in zip(
            problem.constraints.i,
            problem.constraints.j,
            problem.edge_weight,
        )
    )
    expected_center = float(
        sum(
            (Fraction.from_float(value) for value in row_values),
            Fraction(0),
        )
    )
    with np.errstate(all='raise'):
        actual = problem.quadratic_operator.observation_laplacian_matvec(
            vector
        )
    assert actual[0] == expected_center

    rhs_problem = separator.build_power_fit_problem(
        separator.resolve_separator_observations(
            points,
            [(0, 1, 1e300), (0, 2, x), (0, 3, -1e300)],
            confidence=np.ones(3),
        )
    )
    row_rhs = tuple(
        float(
            Fraction.from_float(float(confidence))
            * Fraction.from_float(float(alpha))
            * (
                Fraction.from_float(float(target))
                - Fraction.from_float(float(beta))
            )
        )
        for confidence, alpha, target, beta in zip(
            rhs_problem.confidence,
            rhs_problem.alpha,
            rhs_problem.measurement_target,
            rhs_problem.beta,
        )
    )
    expected_rhs = float(
        sum((Fraction.from_float(float(value)) for value in row_rhs))
    )
    with np.errstate(all='raise'):
        rhs = rhs_problem.quadratic_operator.observation_rhs
    assert rhs[0] == expected_rhs


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_multiscale_quadratic_component_preserves_weaker_terms(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    model = separator.FitModel(
        regularization=separator.L2Regularization(1.0, np.zeros(3))
    )

    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.5), (1, 2, 1.0)],
        confidence=np.array([4e30, 4.0]),
        model=model,
        **_solver_kwargs(solver_name),
        admm_max_iter=5000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(
        result.weights,
        np.array([0.2, 0.2, -0.4]),
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(
        0.2,
        rel=8.0 * np.finfo(np.float64).eps,
        abs=0.0,
    )


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse'])
def test_augmented_rhs_preserves_low_parts_before_row_cancellation(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, 1e16), (0, 1, -1e16)],
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(
        result.weights,
        np.array([0.0, 1.0]),
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_resolution_limited_component_mean_is_structured_failure(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    reference = np.array([0.0, -3e-21, 0.0])
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        [(0, 1, 1e20), (1, 2, -5e19)],
        confidence=np.array([4e30, 4e30]),
        model=separator.FitModel(
            regularization=separator.L2Regularization(1.0, reference)
        ),
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.weights is None
    assert result.status_detail is not None
    assert 'binary64 output resolution' in result.status_detail


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_weak_l2_mean_direction_is_not_hidden_by_graph_conditioning(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    n_sites = 9
    points = np.column_stack(
        (np.arange(n_sites, dtype=np.float64), np.zeros(n_sites))
    )
    target = 0.5 + 2.0**59
    result = separator.fit_weights_from_separators(
        points,
        [(site, site + 1, target) for site in range(n_sites - 1)],
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                2.0**-100,
                np.ones(n_sites),
            )
        ),
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.weights is None
    assert result.status_detail is not None
    assert 'binary64 output resolution' in result.status_detail


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_resolution_limited_two_site_optimum_is_not_false_optimal(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    distance = 2.0**-160
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [distance, 0.0]]),
        [(0, 1, -(2.0**206))],
        measurement='position',
        confidence=np.array([2.0**827]),
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                2.0**300,
                np.array([-(2.0**-132), 2.0**-132]),
            )
        ),
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.weights is None
    assert result.status_detail is not None


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_extreme_finite_position_problem_has_no_raw_overflow(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    tiny = np.nextafter(0.0, 1.0)
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [1e154, 0.0]]),
        [(0, 1, 1e308)],
        measurement='position',
        confidence=np.array([tiny]),
        model=separator.FitModel(
            regularization=separator.L2Regularization(
                tiny,
                np.zeros(2),
            )
        ),
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    assert result.weights is not None
    if solver_name != 'admm':
        np.testing.assert_allclose(
            result.weights,
            np.array([5e153, -5e153]),
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    assert result.objective_breakdown is not None
    assert np.isfinite(result.objective_breakdown.total)
    assert result.objective_breakdown.total == pytest.approx(
        2.470328229206233e292,
        rel=8.0 * np.finfo(np.float64).eps,
    )


@pytest.mark.parametrize('solver_name', ['analytic', 'sparse', 'admm'])
def test_large_weight_contrast_has_no_absolute_magnitude_cutoff(
    solver_name: str,
) -> None:
    if solver_name == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    distance = 2.0**40
    target = 2.0**39 + 2.0**59
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [distance, 0.0]]),
        [(0, 1, target)],
        measurement='position',
        **_solver_kwargs(solver_name),
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_array_equal(
        result.weights,
        np.array([0.0, -(2.0**100)]),
    )
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == 0.0


def test_admm_warm_start_numerical_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm_start_calls = 0

    def reject_warm_start(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal warm_start_calls
        warm_start_calls += 1
        raise separator_solver._NumericalFailure('forced warm-start failure')

    monkeypatch.setattr(
        separator_solver,
        '_solve_component_direct',
        reject_warm_start,
    )
    result = separator.fit_weights_from_separators(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        [(0, 1, 0.75)],
        model=separator.FitModel(mismatch=separator.HuberLoss(1.0)),
        solver='admm',
        admm_max_iter=10000,
        admm_abs_tol=1e-12,
        admm_rel_tol=1e-12,
        connectivity_check='diagnose',
    )

    assert warm_start_calls == 1
    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(result.residuals, np.zeros(1), atol=1e-14)


def test_huber_admm_preserves_multiscale_quadratic_warm_start() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0 + 2.0**30, 0.0]]
    )
    rows = [(0, 1, 0.5), (1, 2, 2.0**29 + 0.5)]
    expected = np.array([0.0, 0.0, -(2.0**30)])

    results = []
    for mismatch in (
        separator.SquaredLoss(),
        separator.HuberLoss(1e20),
    ):
        results.append(
            separator.fit_weights_from_separators(
                points,
                rows,
                measurement='position',
                model=separator.FitModel(mismatch=mismatch),
                solver='admm',
                admm_rho=1.0,
                admm_max_iter=5000,
                admm_abs_tol=1e-10,
                admm_rel_tol=1e-10,
                connectivity_check='diagnose',
            )
        )

    for result in results:
        assert result.status == 'optimal'
        assert result.converged is True
        np.testing.assert_array_equal(result.weights, expected)
        np.testing.assert_array_equal(result.residuals, np.zeros(2))
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.total == 0.0


def test_huber_admm_multiscale_quadratic_matches_nonzero_oracle() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0 + 2.0**30, 0.0]]
    )
    rows = [(0, 1, 0.5), (1, 2, 2.0**29 + 0.5)]
    expected = np.array(
        [0.2 * 2.0**30, 0.2 * 2.0**30, -0.4 * 2.0**30]
    )
    model = separator.FitModel(
        mismatch=separator.HuberLoss(1e20),
        regularization=separator.L2Regularization(
            2.0**-62,
            np.zeros(3),
        ),
    )

    result = separator.fit_weights_from_separators(
        points,
        rows,
        measurement='position',
        model=model,
        solver='admm',
        admm_rho=1.0,
        admm_max_iter=20000,
        admm_abs_tol=1e-12,
        admm_rel_tol=1e-12,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_allclose(
        result.weights,
        expected,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.residuals,
        np.array([0.0, -0.19999999999999998]),
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(0.05)


def test_large_sparse_component_does_not_construct_dense_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip('scipy.sparse.linalg')
    n_sites = 640
    points = np.column_stack(
        (np.arange(n_sites, dtype=np.float64), np.zeros(n_sites))
    )
    rows = [
        (site, site + 1, 0.5)
        for site in range(n_sites - 1)
    ]
    rows.append((0, 1, 0.6))

    class RejectDenseFactor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError('sparse component constructed a dense factor')

    monkeypatch.setattr(
        separator_quadratic,
        '_DenseLeastSquaresFactor',
        RejectDenseFactor,
    )
    result = separator.fit_weights_from_separators(
        points,
        rows,
        solver='direct',
        linear_backend='sparse',
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == pytest.approx(0.0025)
    assert np.max(np.abs(result.residuals)) == pytest.approx(0.05)


def _nonzero_large_multiscale_chain() -> tuple[
    np.ndarray,
    list[tuple[int, int, float]],
    np.ndarray,
]:
    n_sites = 65
    steps = np.ones(n_sites - 1)
    steps[0] = 2.0**30
    steps[2] = 2.0**40
    positions = np.concatenate(([0.0], np.cumsum(steps)))
    points = np.column_stack((positions, np.zeros(n_sites)))
    rows = [
        (
            site,
            site + 1,
            0.5 * distance + (0.5 if site in (0, 2) else 0.0),
        )
        for site, distance in enumerate(steps)
    ]
    expected = np.zeros(n_sites)
    expected[1:3] = -(2.0**30)
    expected[3:] = -(2.0**30) - 2.0**40
    return points, rows, expected


def test_large_sparse_multiscale_consistent_chain_is_exact() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows, expected = _nonzero_large_multiscale_chain()

    result = separator.fit_weights_from_separators(
        points,
        rows,
        measurement='position',
        solver='direct',
        linear_backend='sparse',
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_array_equal(result.weights, expected)
    np.testing.assert_array_equal(result.residuals, np.zeros(len(rows)))
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == 0.0


@pytest.mark.parametrize(
    'mismatch',
    [separator.SquaredLoss(), separator.HuberLoss(1e20)],
)
def test_large_multiscale_admm_nonzero_system_has_functional_parity(
    mismatch: separator.SquaredLoss | separator.HuberLoss,
) -> None:
    points, rows, expected = _nonzero_large_multiscale_chain()
    result = separator.fit_weights_from_separators(
        points,
        rows,
        measurement='position',
        model=separator.FitModel(mismatch=mismatch),
        solver='admm',
        admm_rho=1.0,
        admm_max_iter=5000,
        admm_abs_tol=1e-10,
        admm_rel_tol=1e-10,
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert result.converged is True
    np.testing.assert_array_equal(result.weights, expected)
    np.testing.assert_array_equal(result.residuals, np.zeros(len(rows)))
    assert result.objective_breakdown is not None
    assert result.objective_breakdown.total == 0.0


def test_large_sparse_multiscale_positive_l2_exact_reference_is_optimal() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows, expected = _nonzero_large_multiscale_chain()
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            2.0**-80,
            expected,
        )
    )

    results = [
        separator.fit_weights_from_separators(
            points,
            rows,
            measurement='position',
            model=model,
            **_solver_kwargs(solver_name),
            connectivity_check='diagnose',
        )
        for solver_name in ('analytic', 'sparse', 'admm')
    ]

    for result in results:
        assert result.status == 'optimal'
        assert result.converged is True
        np.testing.assert_array_equal(result.weights, expected)
        np.testing.assert_array_equal(result.residuals, np.zeros(len(rows)))
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.total == 0.0


def test_large_multiscale_inconsistent_parallel_rows_match_objective() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows, _ = _nonzero_large_multiscale_chain()
    first_distance = 2.0**30
    rows.append((0, 1, 0.5 * first_distance - 0.5))

    results = [
        separator.fit_weights_from_separators(
            points,
            rows,
            measurement='position',
            **_solver_kwargs(solver_name),
            connectivity_check='diagnose',
        )
        for solver_name in ('analytic', 'sparse', 'admm')
    ]

    for result in results:
        assert result.status == 'optimal'
        assert result.converged is True
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.total == pytest.approx(
            0.25,
            rel=5e-12,
            abs=0.0,
        )
        assert np.max(np.abs(result.residuals)) == pytest.approx(
            0.5,
            rel=5e-10,
            abs=0.0,
        )
    np.testing.assert_allclose(
        results[0].predicted,
        results[1].predicted,
        rtol=0.0,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        results[2].predicted,
        results[0].predicted,
        rtol=0.0,
        atol=2e-9,
    )


def _large_multiscale_cycle(
    n_sites: int,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, float]],
    np.ndarray,
    float,
]:
    steps = np.ones(n_sites - 1)
    steps[2] = 2.0**30
    positions = np.concatenate(([0.0], np.cumsum(steps)))
    points = np.column_stack((positions, np.zeros(n_sites)))
    cycle_target = 1.0 + 0.05
    rows = [
        (site, site + 1, 0.5 * distance)
        for site, distance in enumerate(steps)
    ]
    rows.append((0, 2, cycle_target))

    cycle_difference = (
        Fraction.from_float(cycle_target) - Fraction(1)
    ) / Fraction.from_float(0.25)
    expected = np.full(
        n_sites,
        float(-cycle_difference / 3),
    )
    expected[0] = 0.0
    expected[1] = float(-cycle_difference / 6)
    objective = float(cycle_difference * cycle_difference / 48)
    return points, rows, expected, objective


@pytest.mark.parametrize('n_sites', [64, 65])
def test_large_multiscale_cycle_has_backend_parity(n_sites: int) -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows, expected, objective = _large_multiscale_cycle(n_sites)
    results = [
        separator.fit_weights_from_separators(
            points,
            rows,
            measurement='position',
            **_solver_kwargs(solver_name),
            connectivity_check='diagnose',
        )
        for solver_name in ('analytic', 'sparse', 'admm')
    ]

    for result in results:
        assert result.status in ('optimal', 'numerical_failure')
        if result.status == 'optimal':
            assert result.converged is True
            np.testing.assert_allclose(
                result.weights,
                expected,
                rtol=4.0 * np.finfo(np.float64).eps,
                atol=0.0,
            )
            assert result.objective_breakdown is not None
            assert result.objective_breakdown.total == pytest.approx(
                objective,
                rel=4e-15,
                abs=0.0,
            )


def _fraction_solve(
    matrix: list[list[Fraction]],
    rhs: list[Fraction],
) -> list[Fraction]:
    size = len(rhs)
    for column in range(size):
        pivot = next(
            row
            for row in range(column, size)
            if matrix[row][column] != 0
        )
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        for row in range(column + 1, size):
            if matrix[row][column] == 0:
                continue
            factor = matrix[row][column] / matrix[column][column]
            for inner in range(column, size):
                matrix[row][inner] -= factor * matrix[column][inner]
            rhs[row] -= factor * rhs[column]
    solution = [Fraction(0) for _ in range(size)]
    for row in range(size - 1, -1, -1):
        remainder = rhs[row] - sum(
            (
                matrix[row][column] * solution[column]
                for column in range(row + 1, size)
            ),
            Fraction(0),
        )
        solution[row] = remainder / matrix[row][row]
    return solution


def test_guarded_normal_candidate_requires_forward_gap_certificate() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    site_i = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 7, 2, 2, 5, 4],
        dtype=np.int64,
    )
    site_j = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 4, 6, 6, 6],
        dtype=np.int64,
    )
    exponents = np.array(
        [-9, -3, -1, 9, -5, 4, 5, 1, 3, -10, 4, -5, -10, 10, -7]
    )
    confidence = np.ldexp(np.ones(exponents.shape), 2 * exponents)
    target = np.array(
        [
            0.5,
            -512.0,
            32768.0,
            0.03125,
            -0.001953125,
            8.0,
            -16384.0,
            -6.103515625e-05,
            16384.0,
            2048.0,
            6.103515625e-05,
            32768.0,
            -32768.0,
            -0.0001220703125,
            -0.25,
        ]
    )
    reference = np.array(
        [
            0.0625,
            16.0,
            16.0,
            -4096.0,
            3.0517578125e-05,
            -8192.0,
            512.0,
            -16.0,
            -6.103515625e-05,
            -0.000244140625,
        ]
    )
    alpha = np.ones(target.shape)
    beta = np.zeros(target.shape)
    expected_objective = 10238.480408242358  # Exact Fraction oracle.

    prepared = separator_quadratic._prepare_quadratic(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
    )
    weights = [
        separator_quadratic.solve_quadratic_component(
            site_i,
            site_j,
            alpha,
            beta,
            target,
            confidence,
            reference,
            0.0,
            backend=backend,
        )
        for backend in ('dense', 'sparse')
    ]

    np.testing.assert_array_equal(weights[0], weights[1])
    for candidate in weights:
        assert separator_quadratic._quadratic_objective(
            prepared,
            candidate,
        ) == pytest.approx(expected_objective, rel=2e-15, abs=0.0)


def test_small_sparse_augmented_fallback_avoids_inaccurate_kkt_result() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    site_i = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 6, 2, 0, 3],
        dtype=np.int64,
    )
    site_j = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 9, 9, 3, 9],
        dtype=np.int64,
    )
    exponents = np.array(
        [17, -12, -12, 6, 22, -3, -9, 12, -13, -22, 24, 22, -13, -2, -23]
    )
    confidence = np.ldexp(np.ones(exponents.shape), 2 * exponents)
    target = np.array(
        [
            32768.0,
            -4096.0,
            -8.0,
            -4096.0,
            6.103515625e-05,
            -0.00390625,
            -3.814697265625e-06,
            -0.015625,
            -0.25,
            1024.0,
            -262144.0,
            0.0625,
            8192.0,
            524288.0,
            1.0,
        ]
    )
    reference = np.zeros(10)
    alpha = np.ones(target.shape)
    beta = np.zeros(target.shape)
    expected_objective = 536874817.0628462  # Exact Fraction oracle.

    prepared = separator_quadratic._prepare_quadratic(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
    )
    dense = separator_quadratic.solve_quadratic_component(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
        backend='dense',
    )
    sparse = separator_quadratic.solve_quadratic_component(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
        backend='sparse',
    )

    np.testing.assert_array_equal(sparse, dense)
    assert separator_quadratic._quadratic_objective(
        prepared,
        sparse,
    ) == pytest.approx(expected_objective, rel=2e-15, abs=0.0)


def test_dense_augmented_candidate_uses_bounded_exact_certificate() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    site_i = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 4, 7, 2, 3, 2],
        dtype=np.int64,
    )
    site_j = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 9, 4, 5, 5],
        dtype=np.int64,
    )
    exponents = np.array(
        [-15, -23, 19, 10, 13, -17, 16, -12, -21, 20, -5, 3, -19, -20, 13]
    )
    confidence = np.ldexp(np.ones(exponents.shape), 2 * exponents)
    target = np.array(
        [
            -32768.0,
            0.25,
            4096.0,
            -3.814697265625e-06,
            0.0625,
            0.0001220703125,
            -0.125,
            -64.0,
            -6.103515625e-05,
            -3.814697265625e-06,
            0.001953125,
            128.0,
            -128.0,
            2.0,
            -0.015625,
        ]
    )
    reference = np.array(
        [
            0.5,
            -0.0009765625,
            -32768.0,
            4096.0,
            -8192.0,
            131072.0,
            2048.0,
            -16384.0,
            -2048.0,
            -0.03125,
        ]
    )
    alpha = np.ones(target.shape)
    beta = np.zeros(target.shape)
    expected_objective = 8529838559604.227  # Exact Fraction oracle.

    prepared = separator_quadratic._prepare_quadratic(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
    )
    weights = [
        separator_quadratic.solve_quadratic_component(
            site_i,
            site_j,
            alpha,
            beta,
            target,
            confidence,
            reference,
            0.0,
            backend=backend,
        )
        for backend in ('dense', 'sparse')
    ]

    np.testing.assert_array_equal(weights[0], weights[1])
    for candidate in weights:
        assert separator_quadratic._quadratic_objective(
            prepared,
            candidate,
        ) == pytest.approx(expected_objective, rel=2e-15, abs=0.0)


def test_large_high_condition_component_uses_stable_gauge_and_rank_candidate() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    site_i = np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 5,
            5, 8, 2, 11, 6, 14, 5, 12, 12, 8,
        ],
        dtype=np.int64,
    )
    site_j = np.array(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 14,
            7, 17, 17, 15, 14, 17, 18, 15, 14, 12,
        ],
        dtype=np.int64,
    )
    exponents = np.array(
        [
            -17, -12, -10, -15, 17, 21, 0, -12, -11, 8,
            3, -24, 17, -2, 22, 6, -8, -24, 23, 11,
            17, -7, -23, -21, 0, -8, -23, 22, -20, -5,
        ]
    )
    confidence = np.ldexp(np.ones(exponents.shape), 2 * exponents)
    target = np.array(
        [
            -0.015625, -2048.0, -262144.0, 1024.0, -0.03125,
            -512.0, 1024.0, -2048.0, -0.015625, -0.001953125,
            -128.0, 2048.0, 64.0, -64.0, -131072.0,
            7.62939453125e-06, -8192.0, 4096.0, 0.03125,
            0.0001220703125, -32768.0, 2.0, -0.001953125,
            0.00390625, -4.0, 1024.0, -1024.0, 1024.0,
            1.52587890625e-05, 131072.0,
        ]
    )
    reference = np.zeros(20)
    alpha = np.ones(target.shape)
    beta = np.zeros(target.shape)
    expected_objective = 1099656410.2512417  # Exact Fraction oracle.

    prepared = separator_quadratic._prepare_quadratic(
        site_i,
        site_j,
        alpha,
        beta,
        target,
        confidence,
        reference,
        0.0,
    )
    assert prepared.n_sites > separator_quadratic._TINY_EXACT_ORACLE_MAX_SITES
    weights = []
    for backend in ('dense', 'sparse'):
        try:
            candidate = separator_quadratic.solve_quadratic_component(
                site_i,
                site_j,
                alpha,
                beta,
                target,
                confidence,
                reference,
                0.0,
                backend=backend,
            )
        except separator_quadratic.QuadraticNumericalError:
            continue
        assert separator_quadratic._quadratic_objective(
            prepared,
            candidate,
        ) == pytest.approx(expected_objective, rel=2e-14, abs=0.0)
        weights.append(candidate)
    if len(weights) == 2:
        np.testing.assert_allclose(
            weights[1],
            weights[0],
            rtol=5e-10,
            atol=2e-3,
        )


def _fraction_quadratic_oracle(
    problem: separator.SeparatorFitProblem,
    *,
    active_sites: int | None = None,
) -> tuple[np.ndarray, float]:
    n_sites = (
        problem.constraints.n_points
        if active_sites is None
        else active_sites
    )
    matrix = [
        [Fraction(0) for _ in range(n_sites)]
        for _ in range(n_sites)
    ]
    rhs = [Fraction(0) for _ in range(n_sites)]
    for row, (site_i, site_j) in enumerate(
        zip(problem.constraints.i, problem.constraints.j)
    ):
        i = int(site_i)
        j = int(site_j)
        if i >= n_sites or j >= n_sites:
            continue
        confidence = Fraction.from_float(float(problem.confidence[row]))
        alpha = Fraction.from_float(float(problem.alpha[row]))
        curvature = confidence * alpha * alpha
        row_rhs = (
            confidence
            * alpha
            * (
                Fraction.from_float(
                    float(problem.measurement_target[row])
                )
                - Fraction.from_float(float(problem.beta[row]))
            )
        )
        matrix[i][i] += curvature
        matrix[j][j] += curvature
        matrix[i][j] -= curvature
        matrix[j][i] -= curvature
        rhs[i] += row_rhs
        rhs[j] -= row_rhs

    regularization = Fraction.from_float(
        float(problem.regularization_strength)
    )
    if regularization > 0:
        for site in range(n_sites):
            matrix[site][site] += regularization
            rhs[site] += regularization * Fraction.from_float(
                float(problem.regularization_reference[site])
            )
        exact = _fraction_solve(matrix, rhs)
    else:
        reduced = [row[1:] for row in matrix[1:]]
        exact = [Fraction(0)] + _fraction_solve(reduced, rhs[1:])

    objective = Fraction(0)
    for row, (site_i, site_j) in enumerate(
        zip(problem.constraints.i, problem.constraints.j)
    ):
        i = int(site_i)
        j = int(site_j)
        if i >= n_sites or j >= n_sites:
            continue
        residual = (
            Fraction.from_float(float(problem.beta[row]))
            + Fraction.from_float(float(problem.alpha[row]))
            * (exact[i] - exact[j])
            - Fraction.from_float(float(problem.measurement_target[row]))
        )
        objective += (
            Fraction(1, 2)
            * Fraction.from_float(float(problem.confidence[row]))
            * residual
            * residual
        )
    if regularization > 0:
        for site, value in enumerate(exact):
            displacement = value - Fraction.from_float(
                float(problem.regularization_reference[site])
            )
            objective += (
                Fraction(1, 2)
                * regularization
                * displacement
                * displacement
            )
    return np.asarray([float(value) for value in exact]), float(objective)


def test_large_multiscale_positive_l2_nonzero_optimum_has_backend_parity() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows, _ = _nonzero_large_multiscale_chain()
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            2.0**-80,
            np.zeros(points.shape[0]),
        )
    )
    observations = separator.resolve_separator_observations(
        points,
        rows,
        measurement='position',
    )
    problem = separator.build_power_fit_problem(
        observations,
        model=model,
    )
    oracle_weights, objective = _fraction_quadratic_oracle(problem)

    results = [
        separator.fit_weights_from_separators(
            points,
            rows,
            measurement='position',
            model=model,
            **_solver_kwargs(solver_name),
            connectivity_check='diagnose',
        )
        for solver_name in ('analytic', 'sparse', 'admm')
    ]

    for result in results:
        assert result.status in ('optimal', 'numerical_failure')
        if result.status == 'optimal':
            assert result.converged is True
            assert result.objective_breakdown is not None
            assert result.objective_breakdown.total == pytest.approx(
                objective,
                rel=5e-10,
                abs=0.0,
            )
            np.testing.assert_allclose(
                result.weights,
                oracle_weights,
                rtol=5e-14,
                atol=1e-2,
            )


def _large_multiscale_ranked_cycle(
    cycle_rank: int,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    n_sites = 65
    steps = np.ones(n_sites - 1)
    steps[12] = 2.0**30
    positions = np.concatenate(([0.0], np.cumsum(steps)))
    points = np.column_stack((positions, np.zeros(n_sites)))
    rows = [
        (site, site + 1, 0.5 * distance)
        for site, distance in enumerate(steps)
    ]
    adjacent = {(site, site + 1) for site in range(12)}
    chords = [
        pair
        for pair in combinations(range(13), 2)
        if pair not in adjacent
    ]
    rows.extend(
        (
            site_i,
            site_j,
            0.5 * (positions[site_j] - positions[site_i])
            + (0.05 if chord == 0 else 0.0),
        )
        for chord, (site_i, site_j) in enumerate(chords[:cycle_rank])
    )
    return points, rows


@pytest.mark.parametrize('cycle_rank', [64, 65])
def test_large_ranked_cycle_has_no_architecture_cutoff(
    cycle_rank: int,
) -> None:
    pytest.importorskip('scipy.sparse.linalg')
    points, rows = _large_multiscale_ranked_cycle(cycle_rank)
    observations = separator.resolve_separator_observations(
        points,
        rows,
        measurement='position',
    )
    problem = separator.build_power_fit_problem(observations)
    oracle_weights, objective = _fraction_quadratic_oracle(
        problem,
        active_sites=13,
    )
    results = [
        separator.fit_weights_from_separators(
            points,
            rows,
            measurement='position',
            **_solver_kwargs(solver_name),
            connectivity_check='diagnose',
        )
        for solver_name in ('analytic', 'sparse', 'admm')
    ]

    for result in results:
        assert result.status in ('optimal', 'numerical_failure')
        if result.status == 'optimal':
            assert result.converged is True
            assert result.objective_breakdown is not None
            assert result.objective_breakdown.total == pytest.approx(
                objective,
                rel=5e-13,
                abs=0.0,
            )
            np.testing.assert_allclose(
                result.weights[:13],
                oracle_weights,
                rtol=2e-13,
                atol=1e-14,
            )
            np.testing.assert_array_equal(
                result.weights[13:],
                np.full(52, result.weights[13]),
            )


def _quadratic_certification_threshold_case(
    n_sites: int,
) -> tuple[np.ndarray, list[tuple[int, int, float]], np.ndarray]:
    points = np.column_stack(
        (
            0.5 * np.arange(n_sites, dtype=np.float64),
            np.zeros(n_sites),
        )
    )
    rows: list[tuple[int, int, float]] = []
    confidence: list[float] = []
    for site in range(n_sites - 1):
        if site == 8:
            rows.append((site, site + 1, 0.25))
            confidence.append(2.0**-150)
            rows.append((site, site + 1, 0.25 + 2.0**-20))
            confidence.append(2.0**12)
        else:
            rows.append((site, site + 1, 0.25))
            confidence.append(1.0)
    return points, rows, np.asarray(confidence, dtype=np.float64)


@pytest.mark.parametrize('n_sites', [16, 17])
def test_quadratic_certificate_does_not_change_at_tiny_oracle_boundary(
    n_sites: int,
) -> None:
    points, rows, confidence = _quadratic_certification_threshold_case(
        n_sites
    )
    result = separator.fit_weights_from_separators(
        points,
        rows,
        confidence=confidence,
        measurement='position',
        connectivity_check='diagnose',
    )

    assert result.status in ('optimal', 'numerical_failure')
    if result.status == 'optimal':
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.total <= (
            3.1861838222649046e-58 * (1.0 + 1.0e-12)
        )


@pytest.mark.parametrize('n_sites', [16, 17])
@pytest.mark.parametrize(
    ('solver_name', 'linear_backend'),
    [
        ('direct', 'dense'),
        ('direct', 'sparse'),
        ('admm', 'dense'),
        ('admm', 'sparse'),
    ],
)
def test_nonrepresentable_zero_optimum_is_independent_of_exact_oracle_limit(
    n_sites: int,
    solver_name: str,
    linear_backend: str,
) -> None:
    if linear_backend == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')

    points = np.column_stack(
        (
            3.0 * np.arange(n_sites, dtype=np.float64),
            np.zeros(n_sites),
        )
    )
    rows = [(0, 1, 0.6)] + [
        (site, site + 1, 0.5)
        for site in range(1, n_sites - 1)
    ]

    result = separator.fit_weights_from_separators(
        points,
        rows,
        solver=solver_name,
        linear_backend=linear_backend,
        connectivity_check='diagnose',
    )

    assert result.status == 'numerical_failure'
    assert result.converged is False
    assert result.weights is None
    assert result.objective_breakdown is None
    assert result.status_detail is not None
    assert (
        'binary64 output resolution' in result.status_detail
        or 'objective-gap certificate' in result.status_detail
    )
    if solver_name == 'admm':
        assert result.n_iter > 0


@pytest.mark.parametrize('n_sites', [256, 257])
def test_exact_zero_certificate_does_not_change_at_helper_boundary(
    n_sites: int,
) -> None:
    points = np.column_stack(
        (
            0.5 * np.arange(n_sites, dtype=np.float64),
            np.zeros(n_sites),
        )
    )
    rows = [
        (site, site + 1, 0.25 + 2.0**-20)
        for site in range(n_sites - 1)
    ]
    confidence = np.full(n_sites - 1, 2.0**-900)
    result = separator.fit_weights_from_separators(
        points,
        rows,
        confidence=confidence,
        measurement='position',
        connectivity_check='diagnose',
    )

    assert result.status in ('optimal', 'numerical_failure')
    if result.status == 'optimal':
        assert result.objective_breakdown is not None
        assert result.objective_breakdown.total == 0.0
        np.testing.assert_array_equal(result.residuals, 0.0)


def test_objective_underflow_is_not_an_exact_zero_proof() -> None:
    prepared = separator_quadratic._prepare_quadratic(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.ones(1),
        np.zeros(1),
        np.array([1.0e-100]),
        np.array([np.nextafter(0.0, 1.0)]),
        np.zeros(2),
        0.0,
    )
    candidate = np.zeros(2)

    assert separator_quadratic._quadratic_objective(
        prepared,
        candidate,
    ) == 0.0
    assert not separator_quadratic._is_exact_zero_objective(
        prepared,
        candidate,
    )


def test_exact_zero_proof_survives_overflowing_ordinary_difference() -> None:
    maximum = np.finfo(np.float64).max
    prepared = separator_quadratic._prepare_quadratic(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0.5]),
        np.array([-maximum]),
        np.zeros(1),
        np.ones(1),
        np.zeros(2),
        0.0,
    )
    candidate = np.array([maximum, -maximum])

    assert separator_quadratic._is_exact_zero_objective(
        prepared,
        candidate,
    )


def test_all_quadratic_factor_paths_enforce_forward_gap(
    monkeypatch,
) -> None:
    pytest.importorskip('scipy.sparse.linalg')
    prepared = separator_quadratic._prepare_quadratic(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        np.ones(2),
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.ones(2),
        np.zeros(2),
        0.0,
    )
    monkeypatch.setattr(
        separator_quadratic,
        '_can_use_tiny_exact_oracle',
        lambda _prepared: False,
    )
    factors = (
        separator_quadratic._make_normal_factor(prepared, 'dense'),
        separator_quadratic._make_factor(prepared, 'dense'),
        separator_quadratic._make_factor(prepared, 'sparse'),
    )
    assert all(factor is not None for factor in factors)
    for factor in factors:
        with pytest.raises(
            separator_quadratic.QuadraticNumericalError,
            match='forward objective-gap',
        ):
            separator_quadratic._certify(
                prepared,
                factor,
                np.zeros(2),
            )


def test_factor_without_singular_value_bound_cannot_certify() -> None:
    prepared = separator_quadratic._prepare_quadratic(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        np.ones(2),
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.ones(2),
        np.zeros(2),
        0.0,
    )

    class MissingBoundFactor:
        condition = 1.0
        largest_singular = 1.0
        smallest_singular_lower_bound = None

    with pytest.raises(
        separator_quadratic.QuadraticNumericalError,
        match='no defensible smallest-singular-value bound',
    ):
        separator_quadratic._certify(
            prepared,
            MissingBoundFactor(),
            np.array([0.0, -0.5]),
        )


def test_spanning_tree_singular_bound_is_conservative_and_topology_aware() -> None:
    n_sites = 32
    prepared = separator_quadratic._prepare_quadratic(
        np.zeros(n_sites - 1, dtype=np.int64),
        np.arange(1, n_sites, dtype=np.int64),
        np.ones(n_sites - 1),
        np.zeros(n_sites - 1),
        np.zeros(n_sites - 1),
        np.ones(n_sites - 1),
        np.zeros(n_sites),
        0.0,
    )

    lower = prepared.smallest_singular_lower_bound()
    actual = float(
        np.linalg.svd(
            prepared.dense_design(),
            compute_uv=False,
        )[-1]
    )

    assert lower > 1.0 / n_sites
    assert lower <= np.nextafter(actual, np.inf)


def test_extended_gradient_bound_dominates_exact_dyadic_oracle() -> None:
    rng = np.random.default_rng(20260731)
    checked = 0
    for _ in range(20):
        n_sites = 24
        site_i = np.concatenate(
            (
                np.arange(n_sites - 1, dtype=np.int64),
                rng.integers(0, n_sites, size=48, dtype=np.int64),
            )
        )
        site_j = np.concatenate(
            (
                np.arange(1, n_sites, dtype=np.int64),
                rng.integers(0, n_sites, size=48, dtype=np.int64),
            )
        )
        keep = site_i != site_j
        site_i = site_i[keep]
        site_j = site_j[keep]
        prepared = separator_quadratic._prepare_quadratic(
            site_i,
            site_j,
            rng.uniform(0.25, 2.0, size=site_i.size),
            rng.normal(size=site_i.size),
            rng.normal(size=site_i.size),
            rng.uniform(0.1, 3.0, size=site_i.size),
            rng.normal(size=n_sites),
            0.125,
        )
        candidate = rng.normal(size=n_sites)

        extended = separator_quadratic._extended_gradient_norm_upper(
            prepared,
            candidate,
        )
        exact = separator_quadratic._exact_gradient_norm_upper(
            prepared,
            candidate,
        )

        if extended is None:
            pytest.skip('platform longdouble is not wider than binary64')
        assert exact is not None
        assert extended >= exact
        checked += 1
    assert checked == 20


def test_final_quadratic_certificate_sees_public_component_gauge(
    monkeypatch,
) -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
        dtype=np.float64,
    )
    reference = np.array([10.0, 20.0, 30.0, 50.0])
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=0.0,
            reference=reference,
        )
    )
    recorded: list[np.ndarray] = []
    original = separator_quadratic._certify

    def recording_certificate(prepared, factor, weights, **kwargs):
        certified = original(prepared, factor, weights, **kwargs)
        if kwargs.get('required_mean') is not None:
            recorded.append(np.asarray(certified, dtype=np.float64).copy())
        return certified

    monkeypatch.setattr(
        separator_quadratic,
        '_certify',
        recording_certificate,
    )
    result = separator.fit_weights_from_separators(
        points,
        [(0, 1, 0.25), (2, 3, 0.75)],
        model=model,
        solver='direct',
        linear_backend='dense',
        connectivity_check='diagnose',
    )

    assert result.status == 'optimal'
    assert any(np.array_equal(values, result.weights[:2]) for values in recorded)
    assert any(np.array_equal(values, result.weights[2:]) for values in recorded)


def _coupled_binary64_objective(
    weights: tuple[Fraction, Fraction],
    *,
    target_difference: Fraction,
    regularization: Fraction,
) -> Fraction:
    reference = Fraction(3, 2)
    left, right = weights
    residual = left - right - target_difference
    return (
        Fraction(1, 2) * residual * residual
        + Fraction(1, 2)
        * regularization
        * (
            (left - reference) * (left - reference)
            + (right - reference) * (right - reference)
        )
    )


def test_coordinatewise_rounding_is_not_a_discrete_quadratic_certificate() -> None:
    regularization_float = 2.0**-60
    target_difference_float = 2.0**-52
    regularization = Fraction.from_float(regularization_float)
    target_difference = Fraction.from_float(target_difference_float)
    reference = Fraction(3, 2)
    offset = target_difference / (2 + regularization)
    exact_optimum = (reference + offset, reference - offset)
    optimum_objective = _coupled_binary64_objective(
        exact_optimum,
        target_difference=target_difference,
        regularization=regularization,
    )

    coordinatewise = tuple(
        Fraction.from_float(float(value))
        for value in exact_optimum
    )
    assert coordinatewise == (reference, reference)
    coordinatewise_gap = (
        _coupled_binary64_objective(
            coordinatewise,
            target_difference=target_difference,
            regularization=regularization,
        )
        - optimum_objective
    )

    center = 1.5
    nearby = (
        np.nextafter(center, -np.inf),
        center,
        np.nextafter(center, np.inf),
    )
    local_gaps = []
    for left in nearby:
        for right in nearby:
            candidate = (
                Fraction.from_float(float(left)),
                Fraction.from_float(float(right)),
            )
            local_gaps.append(
                _coupled_binary64_objective(
                    candidate,
                    target_difference=target_difference,
                    regularization=regularization,
                )
                - optimum_objective
            )
    best_local_gap = min(local_gaps)

    assert best_local_gap > 0
    assert coordinatewise_gap > 10**12 * best_local_gap


@pytest.mark.parametrize(
    ('solver_name', 'linear_backend'),
    [
        ('direct', 'dense'),
        ('direct', 'sparse'),
        ('admm', 'dense'),
        ('admm', 'sparse'),
    ],
)
def test_coupled_nonrepresentable_optimum_is_not_reported_optimal(
    solver_name: str,
    linear_backend: str,
) -> None:
    if linear_backend == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')

    points = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
    regularization = 2.0**-60
    target_difference = 2.0**-52
    reference = np.array([1.5, 1.5], dtype=np.float64)
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            regularization,
            reference,
        )
    )
    observations = separator.resolve_separator_observations(
        points,
        [(0, 1, 0.25 + target_difference)],
        measurement='position',
    )

    result = separator.fit_weights_from_separators(
        points,
        observations,
        model=model,
        solver=solver_name,
        linear_backend=linear_backend,
        admm_abs_tol=1.0e-14,
        admm_rel_tol=1.0e-14,
        connectivity_check='diagnose',
    )

    assert result.status == 'numerical_failure'
    assert result.status_detail is not None
    assert 'binary64 output resolution' in result.status_detail
    assert result.weights is None
    if solver_name == 'admm':
        assert result.n_iter > 0
        assert result.solver_termination.n_iter == result.n_iter
        report = separator.build_fit_report(result, observations)
        assert report['summary']['n_iter'] == result.n_iter
