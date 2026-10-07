"""Certify the returned representative, including no-work components."""

from fractions import Fraction

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel,
    FixedValue,
    HuberLoss,
    L2Regularization,
    SquaredLoss,
    build_power_fit_problem,
    build_power_fit_result,
    fit_weights_from_separators,
    resolve_separator_observations,
)


@pytest.mark.parametrize('confidence', [0.0, 1.0])
def test_final_reference_shift_cannot_return_false_hard_success(confidence):
    points = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    obs = resolve_separator_observations(
        points,
        [(0, 1, 0.8)],
        confidence=[confidence],
    )
    model = FitModel(
        feasible=FixedValue(0.3),
        regularization=L2Regularization(0.0, [2.0**20, 2.0**20, 0.0]),
    )
    result = fit_weights_from_separators(
        points,
        obs,
        model=model,
        solver='admm',
        admm_abs_tol=1e-11,
        admm_rel_tol=1e-11,
    )
    if result.status == 'optimal' or result.converged:
        prediction = (
            Fraction(1, 2)
            + (Fraction(float(result.weights[0])) - Fraction(float(result.weights[1])))
            / 2
        )
        tolerance = 1e-12 + 64 * np.finfo(float).eps * max(
            0.3,
            abs(float(prediction)),
        )
        assert abs(prediction - Fraction(0.3)) <= tolerance
        assert result.objective.hard_constraints_satisfied
    else:
        assert result.status == 'numerical_failure'
        assert not result.converged
        assert result.hard_feasible and result.conflict is None
        assert result.solver == 'admm' and result.n_iter > 0
        assert result.linear_backend == 'dense'
        assert result.edge_diagnostics is not None
        for name in (
            'weights',
            'radii',
            'weight_shift',
            'predicted',
            'predicted_fraction',
            'predicted_position',
            'residuals',
            'rms_residual',
            'max_residual',
            'objective_breakdown',
        ):
            assert getattr(result, name) is None


def test_representable_reference_keeps_certified_hard_success():
    points = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    model = FitModel(
        feasible=FixedValue(0.3),
        regularization=L2Regularization(0.0, [0.125, 0.125, 9.0]),
    )
    result = fit_weights_from_separators(
        points,
        [(0, 1, 0.8)],
        confidence=[0.0],
        model=model,
        solver='admm',
    )
    assert result.status == 'optimal' and result.converged
    assert result.objective.hard_constraints_satisfied
    assert result.weights[2] == 9.0


@pytest.mark.parametrize('canonicalize', [False, True])
@pytest.mark.parametrize(
    'status,converged',
    [
        ('optimal', False),
        ('max_iter', True),
        ('external', True),
    ],
)
def test_builder_rejects_every_false_success_claim(canonicalize, status, converged):
    obs = resolve_separator_observations([[0.0, 0.0], [1.0, 0.0]], [(0, 1, 0.3)])
    problem = build_power_fit_problem(obs, model=FitModel(feasible=FixedValue(0.3)))
    with pytest.raises(ValueError, match='hard'):
        build_power_fit_result(
            problem,
            np.zeros(2),
            status=status,
            converged=converged,
            canonicalize_gauge=canonicalize,
        )


def test_honest_unsuccessful_external_candidate_remains_inspectable():
    obs = resolve_separator_observations([[0.0, 0.0], [1.0, 0.0]], [(0, 1, 0.3)])
    problem = build_power_fit_problem(obs, model=FitModel(feasible=FixedValue(0.3)))
    result = build_power_fit_result(
        problem,
        np.zeros(2),
        status='max_iter',
        converged=False,
    )
    np.testing.assert_array_equal(result.weights, [0.0, 0.0])
    assert not result.objective.hard_constraints_satisfied
    assert any('hard' in warning for warning in result.warnings)


def test_final_guard_uses_each_rows_tolerance():
    obs = resolve_separator_observations(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [(0, 1, 0.3), (1, 2, 1e8)],
    )
    problem = build_power_fit_problem(
        obs,
        model=FitModel(feasible=FixedValue([0.3, 1e8])),
    )
    weights = np.array([-0.4 + 4e-12, 0.0, -199999999.0])
    breakdown = problem.objective_breakdown(weights)
    assert not breakdown.hard_constraints_satisfied
    assert breakdown.hard_max_violation < breakdown.hard_max_tolerance
    with pytest.raises(ValueError, match='hard'):
        build_power_fit_result(problem, weights, canonicalize_gauge=False)


@pytest.mark.parametrize('mean', [2.0**20, 0.125])
def test_native_facade_preserves_final_hard_success_or_refusal(mean):
    from pyvoro2.inverse import fit_self_consistent_weights_from_separators
    from pyvoro2.planar import Box

    result = fit_self_consistent_weights_from_separators(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        [(0, 1, 0.8)],
        confidence=[0.0],
        domain=Box([[-1.0, 4.0], [-1.0, 1.0]]),
        fit_solver='admm',
        fit_admm_abs_tol=1e-11,
        fit_admm_rel_tol=1e-11,
        model=FitModel(
            feasible=FixedValue(0.3),
            regularization=L2Regularization(0.0, [mean, mean, 0.0]),
        ),
    )
    if mean == 0.125:
        assert result.fit.status == 'optimal' and result.final_state_available
        assert result.active_mask.tolist() == [True]
    if result.fit.status == 'optimal' or result.fit.converged:
        assert result.fit.objective.hard_constraints_satisfied
    else:
        assert result.fit.status == 'numerical_failure'
        assert not result.final_state_available
        assert result.fit.weights is None and result.diagnostics is None
    result.to_report()


@pytest.mark.parametrize('loss', [SquaredLoss(), HuberLoss()])
@pytest.mark.parametrize('reference', [None, [7.0, 9.0]])
def test_no_work_components_use_exact_reference_entries(loss, reference):
    points = [[0.0, 0.0], [1.0, 0.0]]
    obs = resolve_separator_observations(points, [(0, 1, 0.25)], confidence=[0.0])
    result = fit_weights_from_separators(
        points,
        obs,
        solver='admm',
        model=FitModel(mismatch=loss, regularization=L2Regularization(0.0, reference)),
    )
    assert result.status == 'optimal' and result.converged
    assert result.solver == 'none' and result.linear_backend is None
    assert result.n_iter == 0 and result.objective.total == 0.0
    np.testing.assert_array_equal(
        result.weights,
        [0.0, 0.0] if reference is None else reference,
    )


@pytest.mark.parametrize('feasible', [None, FixedValue(0.25)])
def test_isolated_reference_survives_hard_dispatch(feasible):
    points = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
    result = fit_weights_from_separators(
        points,
        [(0, 1, 0.25)],
        solver='admm',
        model=FitModel(
            feasible=feasible, regularization=L2Regularization(0.0, [0.0, 0.0, 9.0])
        ),
    )
    assert result.status == 'optimal' and result.converged
    assert result.objective.hard_constraints_satisfied
    np.testing.assert_array_equal(result.weights, [-0.25, 0.25, 9.0])


@pytest.mark.parametrize('n', [0, 1, 3])
@pytest.mark.parametrize('strength', [0.0, 1.0])
def test_empty_native_solution_preserves_supplied_reference(n, strength):
    points = np.zeros((n, 2))
    reference = np.arange(n, dtype=float) + 7.0
    result = fit_weights_from_separators(
        points,
        [],
        model=FitModel(regularization=L2Regularization(strength, reference)),
    )
    assert result.solver == 'none' and result.n_iter == 0
    np.testing.assert_array_equal(result.weights, reference)
