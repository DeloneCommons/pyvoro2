"""Graph and quadratic-operator views introduced for v0.7 issue #14."""

from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
import warnings

import numpy as np
import pytest


def _noisy_problem(*, model=None):
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.20), (1, 2, 0.70), (0, 2, 0.40), (0, 1, 0.30)],
        confidence=[1.0, 0.5, 2.0, 0.75],
    )
    problem = separator.build_power_fit_problem(observations, model=model)
    return points, observations, problem


def test_orientation_and_public_quadratic_reconstruction() -> None:
    _, observations, problem = _noisy_problem()
    graph = problem.observation_graph
    operator = problem.quadratic_operator

    assert graph.n_sites == 3
    assert graph.n_observations == 4
    assert graph.site_i is observations.i
    assert graph.site_j is observations.j
    assert graph.observation_indices is observations.input_index
    assert graph.requested_shifts is observations.shifts
    assert graph.alpha is problem.alpha
    assert graph.beta is problem.beta
    assert graph.z_obs is problem.z_obs
    assert graph.rho is problem.edge_weight
    assert not graph.informative_mask.flags.writeable
    np.testing.assert_array_equal(graph.informative_mask, [True, True, True, True])

    incidence = graph.incidence_dense()
    expected_incidence = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, -1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(incidence, expected_incidence)
    weights = np.array([1.25, -0.5, 2.0])
    np.testing.assert_allclose(
        incidence.T @ weights,
        weights[graph.site_i] - weights[graph.site_j],
    )

    expected_z = (observations.target - problem.beta) / problem.alpha
    expected_rho = observations.confidence * problem.alpha**2
    expected_laplacian = incidence @ np.diag(expected_rho) @ incidence.T
    expected_rhs = incidence @ (expected_rho * expected_z)
    np.testing.assert_allclose(graph.z_obs, expected_z)
    np.testing.assert_allclose(graph.rho, expected_rho)
    np.testing.assert_allclose(
        operator.observation_laplacian_dense(),
        expected_laplacian,
    )
    np.testing.assert_allclose(operator.observation_rhs, expected_rhs)
    np.testing.assert_allclose(
        operator.observation_laplacian_matvec(weights),
        expected_laplacian @ weights,
    )
    np.testing.assert_allclose(
        operator.regularized_normal_matrix_dense(),
        expected_laplacian,
    )
    np.testing.assert_allclose(operator.regularized_normal_rhs, expected_rhs)


def test_noisy_analytic_fit_satisfies_normal_equations_and_predictions() -> None:
    import pyvoro2.inverse as inverse

    points, observations, problem = _noisy_problem()
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        connectivity_check='diagnose',
    )
    operator = problem.quadratic_operator
    graph = problem.observation_graph
    incidence = graph.incidence_dense()

    assert fit.solver == 'analytic'
    assert fit.weights is not None
    np.testing.assert_allclose(
        operator.regularized_normal_matvec(fit.weights),
        operator.regularized_normal_rhs,
        rtol=1e-11,
        atol=1e-12,
    )
    fitted_differences = incidence.T @ fit.weights
    np.testing.assert_allclose(
        fitted_differences,
        problem.predict_difference(fit.weights),
    )
    np.testing.assert_allclose(
        graph.beta + graph.alpha * fitted_differences,
        fit.predicted,
    )
    assert fit.edge_diagnostics is not None
    np.testing.assert_allclose(fit.edge_diagnostics.z_fit, fitted_differences)
    np.testing.assert_allclose(
        fit.edge_diagnostics.residual,
        graph.z_obs - fitted_differences,
    )


def test_l2_regularized_matrix_rhs_and_analytic_fit() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    strength = 0.4
    reference = np.array([2.0, -1.0, 4.0])
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=strength,
            reference=reference,
        )
    )
    points, observations, problem = _noisy_problem(model=model)
    operator = problem.quadratic_operator
    laplacian = operator.observation_laplacian_dense()

    assert operator.regularization_reference is problem.regularization_reference
    assert operator.bounds is problem.bounds
    assert not operator.observation_rhs.flags.writeable
    assert not operator.regularized_normal_rhs.flags.writeable
    np.testing.assert_allclose(
        operator.regularized_normal_matrix_dense(),
        laplacian + strength * np.eye(3),
    )
    np.testing.assert_allclose(
        operator.regularized_normal_rhs,
        operator.observation_rhs + strength * reference,
    )
    assert operator.observation_nullity == 1
    assert operator.regularized_normal_nullity == 0
    assert operator.regularization_removes_observation_nullspace is True

    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        model=model,
        connectivity_check='diagnose',
    )
    assert fit.weights is not None
    np.testing.assert_allclose(
        operator.regularized_normal_matvec(fit.weights),
        operator.regularized_normal_rhs,
        rtol=1e-11,
        atol=1e-12,
    )


def test_zero_confidence_rows_remain_but_do_not_connect_or_contribute() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25, (0, 0)), (1, 2, 0.90, (0, 0))],
        confidence=[1.0, 0.0],
    )
    problem = separator.build_power_fit_problem(observations)
    graph = problem.observation_graph
    operator = problem.quadratic_operator

    assert graph.incidence_dense().shape == (3, 2)
    np.testing.assert_array_equal(graph.requested_shifts, observations.shifts)
    np.testing.assert_array_equal(graph.informative_mask, [True, False])
    assert graph.rho[1] == 0.0
    assert graph.informative_components == ((0, 1), (2,))
    assert graph.isolated_sites == (2,)
    assert graph.relative_component_offsets_identified_by_data is False

    positive_only = graph.incidence_dense()[:, :1]
    expected_laplacian = (
        positive_only @ np.diag(graph.rho[:1]) @ positive_only.T
    )
    expected_rhs = positive_only @ (graph.rho[:1] * graph.z_obs[:1])
    np.testing.assert_allclose(
        operator.observation_laplacian_dense(),
        expected_laplacian,
    )
    np.testing.assert_allclose(operator.observation_rhs, expected_rhs)


def test_disconnected_graph_nullity_is_per_informative_component() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [7.0, 0.0], [10.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25), (2, 3, 0.75)],
    )
    problem = separator.build_power_fit_problem(observations)
    graph = problem.observation_graph
    operator = problem.quadratic_operator

    assert graph.informative_components == ((0, 1), (2, 3), (4,))
    assert graph.isolated_sites == (4,)
    assert operator.observation_nullity == 3
    assert operator.regularized_normal_nullity == 3
    assert operator.relative_component_offsets_identified_by_data is False
    assert operator.regularization_removes_observation_nullspace is False
    eigenvalues = np.linalg.eigvalsh(operator.observation_laplacian_dense())
    assert np.count_nonzero(np.isclose(eigenvalues, 0.0, atol=1e-12)) == 3

    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        connectivity_check='diagnose',
    )
    assert fit.weights is not None
    np.testing.assert_allclose(
        operator.regularized_normal_matvec(fit.weights),
        operator.regularized_normal_rhs,
        atol=1e-12,
    )
    assert fit.identification.relative_component_offsets_identified_by_data is False


def test_repeated_periodic_parallel_rows_preserve_identity_and_geometry() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator
    import pyvoro2.planar as planar

    points = np.array([[0.1, 0.5], [0.9, 0.5]], dtype=float)
    domain = planar.RectangularCell(((0.0, 1.0), (0.0, 1.0)))
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.40, (0, 0)),
            (0, 1, 0.60, (-1, 0)),
            (0, 1, 0.55, (0, 0)),
        ],
        domain=domain,
        image='given_only',
        confidence=[1.0, 2.0, 0.5],
    )
    graph = separator.build_power_fit_problem(observations).observation_graph

    np.testing.assert_array_equal(graph.site_i, [0, 0, 0])
    np.testing.assert_array_equal(graph.site_j, [1, 1, 1])
    np.testing.assert_array_equal(graph.observation_indices, [0, 1, 2])
    np.testing.assert_array_equal(
        graph.requested_shifts,
        [[0, 0], [-1, 0], [0, 0]],
    )
    incidence = graph.incidence_dense()
    assert incidence.shape == (2, 3)
    np.testing.assert_array_equal(incidence[:, 0], incidence[:, 1])
    np.testing.assert_array_equal(incidence[:, 0], incidence[:, 2])
    assert graph.alpha[0] != graph.alpha[1]
    assert graph.z_obs[0] != graph.z_obs[1]
    assert graph.rho[0] != graph.rho[1]


def test_empty_observations_have_zero_operators_and_singleton_components() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [],
        allow_empty=True,
    )
    problem = separator.build_power_fit_problem(observations)
    graph = problem.observation_graph
    operator = problem.quadratic_operator

    assert graph.incidence_dense().shape == (3, 0)
    assert graph.informative_components == ((0,), (1,), (2,))
    assert operator.observation_laplacian_dense().shape == (3, 3)
    np.testing.assert_array_equal(operator.observation_laplacian_dense(), 0.0)
    assert operator.observation_rhs.shape == (3,)
    np.testing.assert_array_equal(operator.observation_rhs, 0.0)
    assert operator.observation_nullity == 3

    zero_site_observations = inverse.resolve_separator_observations(
        np.empty((0, 2), dtype=float),
        [],
        allow_empty=True,
    )
    zero_site_problem = separator.build_power_fit_problem(zero_site_observations)
    zero_graph = zero_site_problem.observation_graph
    zero_operator = zero_site_problem.quadratic_operator
    assert zero_graph.incidence_dense().shape == (0, 0)
    assert zero_graph.informative_components == ()
    assert zero_operator.observation_laplacian_dense().shape == (0, 0)
    assert zero_operator.observation_rhs.shape == (0,)
    assert zero_operator.observation_nullity == 0


def test_nonquadratic_rejection_and_constrained_metadata() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, -0.20)],
    )

    huber_problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(mismatch=separator.HuberLoss(delta=0.1)),
    )
    assert huber_problem.observation_graph.n_observations == 1
    with pytest.raises(ValueError, match='only for SquaredLoss'):
        _ = huber_problem.quadratic_operator

    penalty_problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 0.0),)
        ),
    )
    assert penalty_problem.observation_graph.n_observations == 1
    with pytest.raises(ValueError, match='scalar penalties'):
        _ = penalty_problem.quadratic_operator

    constrained_problem = separator.build_power_fit_problem(
        observations,
        model=separator.FitModel(feasible=separator.Interval(0.0, 1.0)),
    )
    constrained_operator = constrained_problem.quadratic_operator
    assert constrained_operator.has_hard_constraints is True
    assert constrained_operator.represents_full_quadratic_objective is True
    assert constrained_operator.normal_equations_characterize_fit is False
    assert constrained_operator.bounds is constrained_problem.bounds

    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        model=constrained_problem.model,
        solver='admm',
        max_iter=5000,
    )
    assert fit.weights is not None
    assert not np.allclose(
        constrained_operator.regularized_normal_matvec(fit.weights),
        constrained_operator.regularized_normal_rhs,
    )


def test_scipy_is_lazy_and_missing_dependency_error_is_actionable(monkeypatch) -> None:
    _, _, problem = _noisy_problem()
    graph = problem.observation_graph
    operator = problem.quadratic_operator

    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'scipy' or name.startswith('scipy.'):
            raise ImportError('blocked optional dependency')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    with pytest.raises(ImportError, match='install scipy.*dense NumPy'):
        graph.incidence_sparse()
    with pytest.raises(ImportError, match='install scipy.*dense NumPy'):
        operator.observation_laplacian_sparse()
    with pytest.raises(ImportError, match='install scipy.*dense NumPy'):
        operator.regularized_normal_matrix_sparse()


def test_dense_only_subprocess_does_not_import_scipy() -> None:
    code = """
import json
import sys
import numpy as np
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

points = np.array([[0.0, 0.0], [2.0, 0.0]])
observations = inverse.resolve_separator_observations(points, [(0, 1, 0.25)])
problem = separator.build_power_fit_problem(observations)
problem.observation_graph.incidence_dense()
problem.quadratic_operator.observation_laplacian_dense()
print(json.dumps({'scipy': 'scipy' in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '{"scipy": false}'


def test_optional_scipy_conversions_match_dense_operators() -> None:
    pytest.importorskip('scipy.sparse')
    import pyvoro2.inverse.separator as separator

    _, _, problem = _noisy_problem(
        model=separator.FitModel(
            regularization=separator.L2Regularization(strength=0.25)
        )
    )
    graph = problem.observation_graph
    operator = problem.quadratic_operator
    vector = np.array([1.0, -2.0, 0.5])

    incidence_sparse = graph.incidence_sparse(format='csc')
    laplacian_sparse = operator.observation_laplacian_sparse(format='coo')
    normal_sparse = operator.regularized_normal_matrix_sparse(format='csr')
    np.testing.assert_allclose(incidence_sparse.toarray(), graph.incidence_dense())
    np.testing.assert_allclose(
        laplacian_sparse.toarray(),
        operator.observation_laplacian_dense(),
    )
    np.testing.assert_allclose(
        normal_sparse.toarray(),
        operator.regularized_normal_matrix_dense(),
    )
    np.testing.assert_allclose(
        laplacian_sparse @ vector,
        operator.observation_laplacian_matvec(vector),
    )
    np.testing.assert_allclose(
        normal_sparse @ vector,
        operator.regularized_normal_matvec(vector),
    )


def test_operator_views_are_canonical_only_and_aliases_gain_properties() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    assert separator.PowerFitProblem is separator.SeparatorFitProblem
    assert {
        'SeparatorObservationGraphView',
        'SeparatorQuadraticOperatorView',
    } <= set(separator.__all__)
    assert tuple(inverse.__all__) == (
        'SeparatorObservations',
        'resolve_separator_observations',
        'SeparatorFitResult',
        'fit_weights_from_separators',
        'weights_to_radii',
        'radii_to_weights',
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        powerfit = importlib.import_module('pyvoro2.powerfit')
    assert 'SeparatorObservationGraphView' not in powerfit.__all__
    assert 'SeparatorQuadraticOperatorView' not in powerfit.__all__
    assert not hasattr(powerfit, 'SeparatorObservationGraphView')
    assert not hasattr(powerfit, 'SeparatorQuadraticOperatorView')

    _, _, problem = _noisy_problem()
    assert isinstance(
        problem.observation_graph,
        separator.SeparatorObservationGraphView,
    )
    assert isinstance(
        problem.quadratic_operator,
        separator.SeparatorQuadraticOperatorView,
    )
