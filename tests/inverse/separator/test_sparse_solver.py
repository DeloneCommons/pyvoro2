"""Sparse-direct quadratic separator solving for v0.7 issue #17."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys

import numpy as np
import pytest


def _fit_both(points, observations, *, model=None):
    pytest.importorskip('scipy.sparse.linalg')
    import pyvoro2.inverse as inverse

    common = {
        'model': model,
        'connectivity_check': 'diagnose',
    }
    dense = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='analytic',
        **common,
    )
    sparse = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='sparse',
        **common,
    )
    return dense, sparse


def _assert_gauge_invariant_agreement(dense, sparse) -> None:
    assert dense.status == sparse.status == 'optimal'
    assert dense.converged is sparse.converged is True
    assert dense.solver == 'analytic'
    assert sparse.solver == 'sparse'
    assert dense.solver_termination.backend == 'analytic'
    assert sparse.solver_termination.backend == 'sparse'
    assert dense.edge_diagnostics is not None
    assert sparse.edge_diagnostics is not None
    assert dense.objective_breakdown is not None
    assert sparse.objective_breakdown is not None

    np.testing.assert_allclose(
        dense.edge_diagnostics.z_fit,
        sparse.edge_diagnostics.z_fit,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense.predicted,
        sparse.predicted,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense.residuals,
        sparse.residuals,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense.edge_diagnostics.residual,
        sparse.edge_diagnostics.residual,
        rtol=1e-10,
        atol=1e-11,
    )
    assert sparse.objective_breakdown.total == pytest.approx(
        dense.objective_breakdown.total,
        rel=1e-10,
        abs=1e-12,
    )
    assert sparse.objective_breakdown.mismatch == pytest.approx(
        dense.objective_breakdown.mismatch,
        rel=1e-10,
        abs=1e-12,
    )
    assert sparse.objective_breakdown.regularization == pytest.approx(
        dense.objective_breakdown.regularization,
        rel=1e-10,
        abs=1e-12,
    )


def test_connected_repeated_and_zero_confidence_rows_match_dense() -> None:
    import pyvoro2.inverse as inverse

    points = np.array(
        [[0.0, 0.0], [1.2, 0.1], [2.3, -0.2], [3.1, 0.4]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.30),
            (1, 2, 0.65),
            (0, 1, 0.45),
            (2, 3, 0.20),
            (0, 3, 0.90),
        ],
        confidence=[1.0, 2.0, 0.5, 1.5, 0.0],
    )

    dense, sparse = _fit_both(points, observations)

    _assert_gauge_invariant_agreement(dense, sparse)
    assert dense.connectivity.effective_graph.n_components == 1
    assert sparse.connectivity.effective_graph.n_components == 1
    assert sparse.edge_diagnostics.edge_weight[-1] == 0.0


def test_disconnected_component_gauges_match_dense() -> None:
    import pyvoro2.inverse as inverse

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [11.5, 0.0],
            [30.0, 0.0],
        ],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.25),
            (0, 1, 0.40),
            (2, 3, 0.70),
            (1, 2, 0.50),
        ],
        confidence=[1.0, 0.25, 2.0, 0.0],
    )

    dense, sparse = _fit_both(points, observations)

    _assert_gauge_invariant_agreement(dense, sparse)
    expected_components = ((0, 1), (2, 3), (4,))
    assert dense.identification.effective_observation_components == (
        expected_components
    )
    assert sparse.identification.effective_observation_components == (
        expected_components
    )
    for component in expected_components:
        assert np.mean(dense.weights[list(component)]) == pytest.approx(0.0)
        assert np.mean(sparse.weights[list(component)]) == pytest.approx(0.0)


def test_regularized_disconnected_problem_matches_dense() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.2), (2, 3, 0.8)],
    )
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=0.15,
            reference=np.array([1.0, 2.0, -2.0, -1.0]),
        )
    )

    dense, sparse = _fit_both(points, observations, model=model)

    _assert_gauge_invariant_agreement(dense, sparse)


def test_zero_strength_reference_component_alignment_matches_dense() -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.2), (2, 3, 0.8)],
    )
    reference = np.array([10.0, 20.0, 30.0, 50.0])
    model = separator.FitModel(
        regularization=separator.L2Regularization(
            strength=0.0,
            reference=reference,
        )
    )

    dense, sparse = _fit_both(points, observations, model=model)

    _assert_gauge_invariant_agreement(dense, sparse)
    assert np.mean(dense.weights[:2]) == pytest.approx(np.mean(reference[:2]))
    assert np.mean(sparse.weights[:2]) == pytest.approx(np.mean(reference[:2]))
    assert np.mean(dense.weights[2:]) == pytest.approx(np.mean(reference[2:]))
    assert np.mean(sparse.weights[2:]) == pytest.approx(np.mean(reference[2:]))


def test_repeated_periodic_parallel_observations_match_dense() -> None:
    import pyvoro2.inverse as inverse
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

    dense, sparse = _fit_both(points, observations)

    _assert_gauge_invariant_agreement(dense, sparse)
    np.testing.assert_array_equal(dense.used_shifts, sparse.used_shifts)
    np.testing.assert_array_equal(
        sparse.used_shifts,
        [[0, 0], [-1, 0], [0, 0]],
    )


def test_position_measurement_in_three_dimensions_matches_dense() -> None:
    import pyvoro2.inverse as inverse

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [1.8, 1.1, 0.3],
            [0.4, 1.5, 0.8],
        ],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.35),
            (1, 2, 0.80),
            (2, 3, 0.55),
            (0, 3, 0.65),
            (0, 2, 1.10),
        ],
        measurement='position',
        confidence=[1.0, 0.5, 2.0, 1.5, 0.75],
    )

    dense, sparse = _fit_both(points, observations)

    _assert_gauge_invariant_agreement(dense, sparse)
    assert dense.measurement == sparse.measurement == 'position'


@pytest.mark.parametrize(
    'model',
    [
        pytest.param('huber', id='nonquadratic-mismatch'),
        pytest.param('hard', id='hard-constraints'),
        pytest.param('penalty', id='scalar-penalty'),
    ],
)
def test_sparse_backend_rejects_unsupported_solver_branches(model) -> None:
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
    )
    if model == 'huber':
        fit_model = separator.FitModel(
            mismatch=separator.HuberLoss(delta=0.1)
        )
    elif model == 'hard':
        fit_model = separator.FitModel(
            feasible=separator.Interval(0.0, 1.0)
        )
    else:
        fit_model = separator.FitModel(
            penalties=(separator.SoftIntervalPenalty(0.0, 1.0, 1.0),)
        )

    with pytest.raises(ValueError, match='sparse solver cannot be used'):
        inverse.fit_weights_from_separators(
            points,
            observations,
            model=fit_model,
            solver='sparse',
        )


def test_missing_scipy_error_is_actionable_and_dense_still_works(monkeypatch) -> None:
    import pyvoro2.inverse as inverse

    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
    )
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'scipy' or name.startswith('scipy.'):
            raise ImportError('blocked optional dependency')
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    with pytest.raises(ImportError, match=r"solver='sparse'.*pyvoro2\[sparse\]"):
        inverse.fit_weights_from_separators(
            points,
            observations,
            solver='sparse',
        )

    dense = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='analytic',
    )
    assert dense.status == 'optimal'
    assert dense.solver == 'analytic'


def test_default_dense_fit_does_not_import_scipy() -> None:
    code = """
import json
import sys
import numpy as np
import pyvoro2.inverse as inverse

points = np.array([[0.0, 0.0], [1.0, 0.0]])
fit = inverse.fit_weights_from_separators(points, [(0, 1, 0.25)])
print(json.dumps({'solver': fit.solver, 'scipy': 'scipy' in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        'solver': 'analytic',
        'scipy': False,
    }
