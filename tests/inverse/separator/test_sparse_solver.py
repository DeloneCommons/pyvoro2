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
        solver='direct',
        linear_backend='dense',
        **common,
    )
    sparse = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='direct',
        linear_backend='sparse',
        **common,
    )
    return dense, sparse


def _assert_gauge_invariant_agreement(dense, sparse) -> None:
    assert dense.status == sparse.status == 'optimal'
    assert dense.converged is sparse.converged is True
    assert dense.solver == sparse.solver == 'direct'
    assert dense.linear_backend == 'dense'
    assert sparse.linear_backend == 'sparse'
    assert dense.solver_termination.solver == 'direct'
    assert sparse.solver_termination.solver == 'direct'
    assert dense.solver_termination.linear_backend == 'dense'
    assert sparse.solver_termination.linear_backend == 'sparse'
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
            [11.0, 0.0],
            [30.0, 0.0],
        ],
        dtype=float,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.25),
            (0, 1, 0.50),
            (2, 3, 0.75),
            (1, 2, 0.50),
        ],
        confidence=[1.0, 1.0, 2.0, 0.0],
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
        [(0, 1, 0.25), (2, 3, 0.75)],
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
def test_direct_solver_rejects_admm_required_models(model) -> None:
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

    with pytest.raises(ValueError, match="select solver='admm'"):
        inverse.fit_weights_from_separators(
            points,
            observations,
            model=fit_model,
            solver='direct',
            linear_backend='dense',
        )


@pytest.mark.parametrize('solver_name', ['direct', 'admm'])
def test_missing_scipy_error_is_actionable_and_dense_still_works(
    monkeypatch,
    solver_name: str,
) -> None:
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
    with pytest.raises(
        ImportError,
        match=r"linear_backend='sparse'.*pyvoro2\[sparse\]",
    ):
        inverse.fit_weights_from_separators(
            points,
            observations,
            solver=solver_name,
            linear_backend='sparse',
        )

    dense = inverse.fit_weights_from_separators(
        points,
        observations,
        solver=solver_name,
        linear_backend='dense',
    )
    assert dense.status == 'optimal'
    assert dense.solver == solver_name
    assert dense.linear_backend == 'dense'


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
        'solver': 'direct',
        'scipy': False,
    }


@pytest.mark.parametrize('n_sites', [512, 513])
@pytest.mark.parametrize('solver_name', ['direct', 'admm'])
def test_explicit_dense_solver_does_not_import_scipy_at_size_boundary(
    monkeypatch,
    n_sites: int,
    solver_name: str,
) -> None:
    from pyvoro2.inverse import separator

    points = np.column_stack(
        (
            0.5 * np.arange(n_sites, dtype=np.float64),
            np.zeros(n_sites),
        )
    )
    rows = [
        (site, site + 1, 0.25)
        for site in range(n_sites - 1)
    ]
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == 'scipy' or name.startswith('scipy.'):
            raise AssertionError('dense solver imported SciPy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    result = separator.fit_weights_from_separators(
        points,
        rows,
        measurement='position',
        model=(
            None
            if solver_name == 'direct'
            else separator.FitModel(
                mismatch=separator.HuberLoss(delta=1.0)
            )
        ),
        solver=solver_name,
        linear_backend='dense',
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    assert result.solver == solver_name
    assert result.linear_backend == 'dense'


def test_explicit_quadratic_admm_enters_admm_path(monkeypatch) -> None:
    import pyvoro2.inverse.separator.solver as solver_mod

    calls = 0
    original = solver_mod._solve_component_admm

    def traced(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_mod, '_solve_component_admm', traced)
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = solver_mod.fit_weights_from_separators(
        points,
        [(0, 1, 0.4), (0, 1, 0.6)],
        solver='admm',
        linear_backend='dense',
        connectivity_check='diagnose',
    )
    assert calls == 1
    assert result.status == 'optimal'
    assert result.solver == 'admm'


@pytest.mark.parametrize('backend', ['dense', 'sparse'])
def test_admm_warm_start_and_weight_system_use_requested_backend(
    monkeypatch,
    backend: str,
) -> None:
    if backend == 'sparse':
        pytest.importorskip('scipy.sparse.linalg')
    import pyvoro2.inverse.separator.solver as solver_mod

    warm_backends: list[str] = []
    system_backends: list[str] = []
    original_direct = solver_mod._solve_component_direct
    original_build = solver_mod.QuadraticWeightSystem.build.__func__

    def traced_direct(*args, **kwargs):
        warm_backends.append(kwargs['backend'])
        return original_direct(*args, **kwargs)

    def traced_build(cls, *args, **kwargs):
        system_backends.append(kwargs['backend'])
        return original_build(cls, *args, **kwargs)

    monkeypatch.setattr(
        solver_mod,
        '_solve_component_direct',
        traced_direct,
    )
    monkeypatch.setattr(
        solver_mod.QuadraticWeightSystem,
        'build',
        classmethod(traced_build),
    )
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = solver_mod.fit_weights_from_separators(
        points,
        [(0, 1, 0.4), (0, 1, 0.6)],
        model=solver_mod.FitModel(
            mismatch=solver_mod.HuberLoss(delta=0.2)
        ),
        solver='admm',
        linear_backend=backend,
        connectivity_check='diagnose',
    )
    assert result.status == 'optimal'
    assert warm_backends == [backend]
    assert system_backends == [backend]


def test_sparse_augmented_recovery_does_not_cross_to_dense(
    monkeypatch,
) -> None:
    pytest.importorskip('scipy.sparse.linalg')
    import pyvoro2.inverse.separator._quadratic as quadratic

    monkeypatch.setattr(
        quadratic,
        '_make_normal_factor',
        lambda prepared, backend: None,
    )

    class ForbiddenDense:
        def __init__(self, *args, **kwargs):
            raise AssertionError('sparse recovery crossed to dense')

    monkeypatch.setattr(quadratic, '_DenseLeastSquaresFactor', ForbiddenDense)
    candidate = quadratic.solve_quadratic_component(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        np.ones(2),
        np.zeros(2),
        np.array([0.4, 0.6]),
        np.ones(2),
        np.zeros(2),
        0.0,
        backend='sparse',
    )
    assert np.all(np.isfinite(candidate))
