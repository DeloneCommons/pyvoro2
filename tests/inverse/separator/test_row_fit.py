"""Independent source/mismatch/constraint units and row solver policy."""

import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from pyvoro2.inverse.separator import (
    FitModel, Interval, L2Regularization, SoftIntervalPenalty, SquaredLoss,
    build_power_fit_problem, build_power_fit_result,
    fit_weights_from_separators, resolve_separator_observations,
)


def test_mixed_space_hand_computable_anchor():
    points = np.array([[0., 0.], [2., 0.]])
    model = FitModel(
        mismatch=SquaredLoss(space='position'),
        feasible=Interval(0., 1., space='fraction'),
        penalties=(SoftIntervalPenalty(3/8, 5/8, 8., space='fraction'),),
    )
    result = fit_weights_from_separators(
        points, [(0, 1, 1/4)], model=model, solver='admm',
        admm_abs_tol=1e-10, admm_rel_tol=1e-10, admm_max_iter=5000,
    )
    assert result.status == 'optimal', result.status_detail
    assert result.predicted[0] == pytest.approx(7/20, abs=1e-9)
    assert result.mismatch_predicted[0] == pytest.approx(7/10, abs=1e-9)
    assert result.weights[0] - result.weights[1] == pytest.approx(-6/5, abs=1e-8)
    assert result.objective.total == pytest.approx(1/40, abs=1e-10)
    np.testing.assert_array_equal(result.mismatch_target, [.5])
    assert result.mismatch_residuals[0] == pytest.approx(.2, abs=1e-9)
    assert result.edge_diagnostics.alpha[0] == .125
    assert result.mismatch_space == 'position'
    assert result.hard_constraint_space == 'fraction'
    assert result.penalty_spaces == ('fraction',)


def test_problem_operator_uses_mismatch_geometry_and_result_keeps_source():
    observations = resolve_separator_observations(
        [[0., 0.], [2., 0.]], [(0, 1, .25)],
    )
    problem = build_power_fit_problem(
        observations, model=FitModel(mismatch=SquaredLoss(space='position')),
    )
    assert problem.alpha[0] == .25
    assert problem.z_obs[0] == -2.
    assert problem.edge_weight[0] == .0625
    np.testing.assert_array_equal(problem.quadratic_operator.observation_rhs,
                                  [-.125, .125])
    result = build_power_fit_result(problem, [-1., 1.])
    assert result.predicted[0] == .25
    assert result.mismatch_predicted[0] == .5
    for restored in (replace(result, warnings=('copied',)),
                     copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        assert restored.resolved_policy == problem.resolved_policy
        assert not restored.mismatch_target.flags.writeable
        assert not restored.mismatch_predicted.flags.writeable
    for restored in (copy.deepcopy(problem), pickle.loads(pickle.dumps(problem))):
        assert restored.resolved_policy == problem.resolved_policy


def test_inapplicable_hard_rows_are_absent_before_conversion_and_statistics():
    observations = resolve_separator_observations(
        [[0., 0.], [2., 0.], [4., 0.]],
        [(0, 1, .25), (1, 2, .25)], confidence=[1., 0.],
    )
    model = FitModel(feasible=Interval(
        [0., -1e308], [1., 1e308], applicable=[True, False], space='position',
    ))
    problem = build_power_fit_problem(observations, model=model)
    assert problem.hard_feasible
    assert problem.bounds.space == 'position'
    np.testing.assert_array_equal(problem.bounds.applicable, [True, False])
    assert np.isnan(problem.bounds.difference_lower[1])
    assert np.isnan(problem.bounds.difference_upper[1])
    assert problem.bounds.measurement_lower[1] == -1e308
    for restored in (copy.deepcopy(problem.bounds),
                     pickle.loads(pickle.dumps(problem.bounds))):
        assert not restored.applicable.flags.writeable
    np.testing.assert_array_equal(problem.offset_identifying_constraint_mask,
                                  [True, False])
    result = build_power_fit_result(problem, [0., 0., 0.])
    assert result.objective.hard_max_violation == 0.
    assert result.objective.hard_max_tolerance < 1e-6


def test_all_false_hard_and_zero_penalty_allow_direct_and_retain_policy():
    model = FitModel(
        feasible=Interval(0., 1., applicable=[False, False]),
        penalties=(SoftIntervalPenalty(0., 1., [0., 0.]),),
        regularization=L2Regularization(1.),
    )
    result = fit_weights_from_separators(
        [[0., 0.], [2., 0.], [4., 0.]], [(0, 1, .2), (1, 2, .3)],
        model=model, solver='direct',
    )
    assert result.status == 'optimal'
    assert result.resolved_policy['model_spaces']['hard_constraint'] == 'fraction'


def test_component_row_projection_preserves_strength_and_site_reference():
    model = FitModel(
        penalties=(SoftIntervalPenalty(.4, .6, [1., 4.]),),
        regularization=L2Regularization(0., [10., 20., 30., 40.]),
    )
    result = fit_weights_from_separators(
        [[0., 0.], [1., 0.], [0., 2.], [1., 2.]],
        [(0, 1, .2), (2, 3, .2)], model=model, solver='admm',
        admm_abs_tol=1e-10, admm_rel_tol=1e-10, admm_max_iter=5000,
    )
    assert result.status == 'optimal', result.status_detail
    np.testing.assert_allclose(result.predicted, [1/3, 17/45], atol=1e-9)
    np.testing.assert_allclose([np.mean(result.weights[:2]),
                               np.mean(result.weights[2:])], [15., 35.])


@pytest.mark.parametrize('source', ['fraction', 'position'])
@pytest.mark.parametrize('mismatch', ['fraction', 'position'])
@pytest.mark.parametrize('backend', ['dense', 'sparse'])
@pytest.mark.parametrize('solver,restricted',
                         [('direct', False), ('admm', False), ('admm', True)])
def test_scalar_and_uniform_rows_match_on_legal_solver_paths(
    source, mismatch, backend, solver, restricted,
):
    if backend == 'sparse':
        pytest.importorskip('scipy')
    points = np.array([[0., 0.], [2., 0.], [0., 3.], [1., 3.]])
    rows = [(0, 1, .25 if source == 'fraction' else .5),
            (2, 3, .25)]
    models = [FitModel(
        mismatch=SquaredLoss(space=mismatch),
        feasible=(Interval([0., 0.] if vector else 0.,
                           [1., 1.] if vector else 1., space='fraction')
                  if restricted else None),
        penalties=(SoftIntervalPenalty(0., 1., [0., 0.] if vector else 0.),),
    ) for vector in (False, True)]
    results = [fit_weights_from_separators(
        points, rows, measurement=source, model=model, solver=solver,
        linear_backend=backend, admm_abs_tol=1e-9, admm_rel_tol=1e-9,
    ) for model in models]
    assert all(result.status == 'optimal' for result in results)
    np.testing.assert_array_equal(results[0].weights, results[1].weights)
    assert results[0].objective == results[1].objective


def test_mixed_hard_rows_conflict_indices_and_zero_hazard_sentries(monkeypatch):
    import pyvoro2.inverse.separator.problem as implementation
    from pyvoro2.inverse.separator import (
        ExponentialBoundaryPenalty, ReciprocalBoundaryPenalty,
    )

    observed = []
    original = implementation._hard_constraint_bounds

    def conversion(lower, upper, alpha, beta):
        observed.extend(lower.tolist())
        assert np.all(np.abs(lower) < 100.)
        return original(lower, upper, alpha, beta)

    monkeypatch.setattr(implementation, '_hard_constraint_bounds', conversion)
    observations = resolve_separator_observations(
        [[0., 0.], [1., 0.], [2., 0.]],
        [(0, 1, .2), (1, 2, .4), (0, 2, .8)], confidence=[0., 0., 0.],
    )
    model = FitModel(
        feasible=Interval([.2, -1e308, .3], [.6, 1e308, .3],
                          applicable=[True, False, True]),
        penalties=(ExponentialBoundaryPenalty(strength=[0., 0., 0.]),
                   ReciprocalBoundaryPenalty(strength=[0., 0., 0.])),
    )
    problem = build_power_fit_problem(observations, model=model)
    assert observed == [.2, .3]
    zero_value_coupling = build_power_fit_problem(observations, model=FitModel(
        penalties=(SoftIntervalPenalty(0., 1., [2., 0., 0.]),),
    ))
    np.testing.assert_array_equal(
        zero_value_coupling.offset_identifying_constraint_mask, [True, False, False],
    )
    assert zero_value_coupling.objective_breakdown([0., 0., 0.]).penalties_total == 0.
    # No zero-strength family can reach its compiled/evaluation branch.

    def forbidden(*args, **kwargs):
        raise AssertionError('absent penalty was evaluated')
    monkeypatch.setattr(implementation, '_penalty_value_from_affine', forbidden)
    assert problem.objective_breakdown([0., 0., 0.]).penalties_total == 0.
    np.testing.assert_array_equal(problem.offset_identifying_constraint_mask,
                                  [True, False, True])
    conflicting = build_power_fit_problem(observations, model=FitModel(
        feasible=Interval([.5, .5, .1], [.5, .5, .1]),
    ))
    assert not conflicting.hard_feasible
    assert set(conflicting.hard_conflict.constraint_indices) == {0, 1, 2}


def test_bound_policy_resists_mutation_of_the_original_term_owner():
    observations = resolve_separator_observations([[0., 0.], [2., 0.]], [(0, 1, .5)])
    term = SoftIntervalPenalty([0.], [1.], [2.])
    problem = build_power_fit_problem(observations, model=FitModel(penalties=(term,)))
    term.strength.setflags(write=True)
    term.strength[:] = 99.
    assert problem.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength']['values'] == (2.,)
