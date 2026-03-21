import numpy as np


def test_build_power_fit_problem_exposes_resolved_numeric_problem():
    from pyvoro2.powerfit import (
        build_power_fit_problem,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    problem = build_power_fit_problem(constraints)

    assert np.allclose(problem.alpha, np.array([0.125]))
    assert np.allclose(problem.beta, np.array([0.5]))
    assert np.allclose(problem.z_obs, np.array([-2.0]))
    assert np.allclose(problem.edge_weight, np.array([0.015625]))
    assert np.allclose(problem.measurement_target, np.array([0.25]))
    assert problem.suggested_anchor_indices == tuple()


def test_build_power_fit_problem_reports_advisory_anchors_for_disconnected_case():
    from pyvoro2.powerfit import (
        build_power_fit_problem,
        resolve_pair_bisector_constraints,
    )

    pts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=float,
    )
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25), (2, 3, 0.75)],
        measurement='fraction',
    )
    problem = build_power_fit_problem(constraints)

    assert problem.connectivity.effective_graph.n_components == 2
    assert problem.suggested_anchor_indices == (0, 2)


def test_problem_definition_objects_are_read_only():
    from pyvoro2.powerfit import (
        FitModel,
        Interval,
        build_power_fit_problem,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    problem = build_power_fit_problem(
        constraints,
        model=FitModel(feasible=Interval(0.0, 1.0)),
    )
    predictions = problem.predict(np.array([0.0, 2.0]))

    assert not constraints.i.flags.writeable
    assert not problem.alpha.flags.writeable
    assert not problem.bounds.measurement_lower.flags.writeable
    assert not predictions.measurement.flags.writeable


def test_build_power_fit_result_round_trips_native_weights_and_reports_objective():
    from pyvoro2.powerfit import (
        build_power_fit_problem,
        build_power_fit_result,
        fit_power_weights,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    fit = fit_power_weights(pts, constraints)
    problem = build_power_fit_problem(constraints)
    rebuilt = build_power_fit_result(problem, fit.weights, solver='external-lbfgsb')

    assert np.allclose(rebuilt.weights, fit.weights)
    assert np.allclose(rebuilt.predicted, fit.predicted)
    assert rebuilt.objective_breakdown is not None
    assert rebuilt.objective_breakdown.hard_constraints_satisfied is True
    assert rebuilt.objective_breakdown.total == 0.0


def test_build_power_fit_result_can_package_imperfect_external_weights():
    from pyvoro2.powerfit import (
        FitModel,
        Interval,
        build_power_fit_problem,
        build_power_fit_result,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    problem = build_power_fit_problem(
        constraints,
        model=FitModel(feasible=Interval(0.0, 1.0)),
    )
    result = build_power_fit_result(
        problem,
        np.array([0.0, 8.0]),
        solver='external',
        status='external_failure',
        status_detail='line search failed',
    )

    assert result.objective_breakdown is not None
    assert result.objective_breakdown.hard_constraints_satisfied is False
    assert result.status == 'external_failure'
    assert result.status_detail == 'line search failed'
    assert any('hard measurement bounds' in warning for warning in result.warnings)


def test_fit_report_includes_objective_breakdown_and_status_detail():
    from pyvoro2.powerfit import (
        FitModel,
        Interval,
        build_fit_report,
        build_power_fit_problem,
        build_power_fit_result,
        resolve_pair_bisector_constraints,
    )

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    constraints = resolve_pair_bisector_constraints(
        pts,
        [(0, 1, 0.25)],
        measurement='fraction',
    )
    problem = build_power_fit_problem(
        constraints,
        model=FitModel(feasible=Interval(0.0, 1.0)),
    )
    result = build_power_fit_result(
        problem,
        np.array([0.0, 2.0]),
        solver='external',
        status='external_failure',
        status_detail='test detail',
    )

    report = build_fit_report(result, constraints)

    assert report['summary']['status_detail'] == 'test detail'
    assert report['objective_breakdown'] is not None
    assert report['objective_breakdown']['hard_constraints_satisfied'] is True
