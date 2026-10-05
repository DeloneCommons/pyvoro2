"""WP11 public workflow oracles; native cases require an admitted artifact."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import numpy as np
import pytest

import pyvoro2 as spatial
import pyvoro2.planar as planar
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as advanced


DEFAULTS = (
    ('points', inspect.Parameter.empty),
    ('constraints', inspect.Parameter.empty),
    ('measurement', 'fraction'), ('domain', inspect.Parameter.empty),
    ('ids', None), ('index_mode', 'index'), ('image', 'nearest'),
    ('image_search', 1), ('confidence', None), ('model', None),
    ('r_min', 0.0), ('weight_shift', None),
    ('fit_solver', 'direct'), ('fit_linear_backend', 'dense'),
    ('fit_admm_max_iter', 2000), ('fit_admm_rho', 1.0),
    ('fit_admm_abs_tol', 1e-6), ('fit_admm_rel_tol', 1e-5),
    ('max_outer_iter', 25), ('return_cells', False),
    ('return_boundary_measure', False), ('return_tessellation_diagnostics', False),
    ('tessellation_check', 'diagnose'), ('connectivity_check', 'warn'),
    ('unaccounted_pair_check', 'warn'),
)


def _fit(*args, **kwargs):
    return inverse.fit_self_consistent_weights_from_separators(*args, **kwargs)


def _sites(dim, count=3):
    points = np.zeros((count, dim))
    points[:, 0] = range(count)
    domain_type = planar.Box if dim == 2 else spatial.Box
    return points, domain_type(((-5., 5.),) * dim)


def _strict_report(result):
    report = result.to_report()
    assert report['schema'] == {
        'name': 'pyvoro2.inverse.separator.report', 'version': 2,
    }
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert json.loads(advanced.dumps_report_json(report)) == report
    return report


def _row_ids(observations):
    return tuple(record['row_id'] for record in observations.to_records())


def test_facade_exact_signature_class_identity_and_annotations():
    function = inverse.fit_self_consistent_weights_from_separators
    signature = inspect.signature(function)
    assert tuple((p.name, p.default) for p in signature.parameters.values()) == DEFAULTS
    for index, parameter in enumerate(signature.parameters.values()):
        assert parameter.kind is (
            inspect.Parameter.POSITIONAL_OR_KEYWORD if index < 2
            else inspect.Parameter.KEYWORD_ONLY
        )
    assert inverse.SelfConsistentPowerFitResult is advanced.SelfConsistentPowerFitResult
    assert inverse.SelfConsistentPowerFitResult.__module__ == (
        'pyvoro2.inverse.separator.active'
    )
    assert not hasattr(advanced, function.__name__)
    assert not hasattr(spatial, function.__name__)
    assert not hasattr(spatial, 'SelfConsistentPowerFitResult')
    assert get_type_hints(function)['return'] is inverse.SelfConsistentPowerFitResult
    for domain in (planar.Box(((-1., 1.),) * 2),
                   planar.RectangularCell(((0., 1.),) * 2),
                   spatial.Box(((-1., 1.),) * 3),
                   spatial.OrthorhombicCell(((0., 1.),) * 3),
                   spatial.PeriodicCell(np.eye(3))):
        assert isinstance(domain, get_type_hints(function)['domain'])
    ast.parse(Path(inspect.getfile(function)).read_text(), feature_version=(3, 10))


def test_preferred_import_keeps_geometry_and_scipy_lazy():
    code = """
import json, sys
from pyvoro2.inverse import (
    fit_self_consistent_weights_from_separators, SelfConsistentPowerFitResult,
)
print(json.dumps([name for name in sys.modules
                 if name in ('pyvoro2._core', 'pyvoro2._core2d')
                 or name == 'scipy' or name.startswith('scipy.')]))
"""
    result = subprocess.run([sys.executable, '-c', code], check=True,
                            capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_domain_is_required():
    with pytest.raises(TypeError, match='domain'):
        _fit([[0., 0.], [1., 0.]], [(0, 1, .5)])


def test_forwarding_preserves_argument_objects_and_returns_engine_result(monkeypatch):
    # Secondary delegation evidence supplements the public numerical oracles.
    from pyvoro2.inverse.separator import _facade

    arguments = {name: object() for name, _ in DEFAULTS if name != 'max_outer_iter'}
    returned = object()
    captured = {}

    def engine(points, constraints, **kwargs):
        captured.update(points=points, constraints=constraints, **kwargs)
        return returned

    monkeypatch.setattr(_facade, 'solve_self_consistent_power_weights', engine)
    assert _fit(**arguments, max_outer_iter=np.int64(3)) is returned
    assert all(captured[name] is value for name, value in arguments.items())
    assert captured['active0'] is None and captured['return_history'] is False
    assert captured['options'] == advanced.ActiveSetOptions(
        add_after=1, drop_after=2, relax=1., max_iter=3,
        cycle_window=8, weight_step_tol=1e-8,
    )


@pytest.mark.parametrize('name', (
    'options', 'active0', 'add_after', 'drop_after', 'relax', 'cycle_window',
    'weight_step_tol', 'return_history', 'solver', 'linear_backend', 'max_iter',
))
def test_research_controls_and_legacy_aliases_are_not_accepted(name):
    points, domain = _sites(2)
    with pytest.raises(TypeError, match=name):
        _fit(points, [(0, 1, .5)], domain=domain, **{name: None})


@pytest.mark.parametrize('value', (
    True, np.bool_(False), 1., 1.5, '1', None, np.array(1), np.array([1]),
    0, -1, sys.maxsize + 1,
))
def test_max_outer_iter_uses_strict_existing_option_validation(value):
    points, domain = _sites(2)
    with pytest.raises(ValueError, match='ActiveSetOptions.max_iter'):
        _fit(points, [(0, 1, .5)], domain=domain, max_outer_iter=value)


class _Index:
    def __index__(self):
        return 2


@pytest.mark.parametrize('value', (np.int64(1), _Index(), sys.maxsize))
def test_positive_index_protocol_and_maximum_are_admitted(value):
    points, domain = _sites(2, 2)
    result = _fit(points, [(0, 1, .5), (0, 1, .5)], domain=domain,
                  max_outer_iter=value, fit_solver='admm',
                  model=advanced.FitModel(feasible=advanced.FixedValue([.2, .8])))
    assert result.termination == 'infeasible_active_set'


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('limit', (1, 25))
def test_native_midpoint_selection_final_refit_and_outer_limit(dim, limit):
    points, domain = _sites(dim)
    rows = [(0, 1, .5), (1, 2, .5), (0, 2, .5)]
    result = _fit(points, rows, domain=domain, max_outer_iter=limit,
                  return_boundary_measure=True)
    expected_mask = [True, True, limit == 1]
    assert result.termination == ('max_outer_iter' if limit == 1 else 'self_consistent')
    assert result.converged is (limit != 1)
    assert result.inner_fit.status == 'optimal'
    assert result.final_state_available and result.final_refit_converged
    assert result.final_state_unavailable_reason is None
    assert result.inner_fit is result.fit
    assert result.final_realization is result.realized
    assert result.candidate_diagnostics is result.diagnostics
    assert result.outer_termination.status == result.termination
    assert not hasattr(result, 'success')
    assert type(result) is inverse.SelfConsistentPowerFitResult
    assert result.history is None and result.path_summary is not None
    np.testing.assert_array_equal(result.active_mask, expected_mask)
    np.testing.assert_array_equal(
        result.realized.realized_same_shift, [True, True, False],
    )
    np.testing.assert_allclose(np.diff(result.inner_fit.weights), 0., atol=1e-14)
    np.testing.assert_allclose(result.realized.boundary_measure,
                               [10.**(dim-1), 10.**(dim-1), np.nan])
    assert result.to_records()[2]['boundary_measure'] is None
    report = _strict_report(result)
    assert result.resolved_policy['row_ids'] == _row_ids(result.constraints)
    assert result.fit.resolved_policy['row_ids'] == tuple(
        row for row, active in zip(_row_ids(result.constraints), expected_mask)
        if active
    )
    assert report['observation_set']['row_ids'] == list(_row_ids(result.constraints))
    assert report['fit']['observation_set']['row_ids'] == list(
        result.fit.resolved_policy['row_ids'])
    equivalent = advanced.solve_self_consistent_power_weights(
        points, rows, domain=domain, options=advanced.ActiveSetOptions(max_iter=limit),
        active0=None, return_history=False, return_boundary_measure=True,
    )
    np.testing.assert_array_equal(result.active_mask, equivalent.active_mask)
    np.testing.assert_allclose(result.fit.weights, equivalent.fit.weights)
    assert report == equivalent.to_report()


@pytest.mark.parametrize('dim', (2, 3))
def test_native_default_cycle_keeps_atomic_available_final_state(dim):
    points, domain = _sites(dim)
    result = _fit(points, [(0, 1, 7/4), (1, 2, 1/8), (0, 2, 3/4)],
                  domain=domain,
                  model=advanced.FitModel(
                      regularization=advanced.L2Regularization(.01)))
    assert result.termination == 'cycle_detected' and result.cycle_length == 3
    assert not result.converged
    assert result.inner_fit.status == 'optimal'
    assert result.final_state_available and result.final_refit_converged
    assert result.history is None
    np.testing.assert_array_equal(result.active_mask, [True, True, True])
    np.testing.assert_array_equal(
        result.realized.realized_same_shift, [False, False, True],
    )
    # Fresh public realization from the accepted vector is the final-state oracle.
    realized = advanced.match_realized_pairs(
        points, domain=domain, constraints=result.constraints,
        weights=result.fit.weights,
    )
    np.testing.assert_array_equal(result.realized.realized_same_shift,
                                  realized.realized_same_shift)
    _strict_report(result)


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('cells', (False, True))
@pytest.mark.parametrize('measures', (False, True))
@pytest.mark.parametrize('diagnostics', (False, True))
def test_native_optional_output_matrix_does_not_control_realization(
    dim, cells, measures, diagnostics,
):
    points, domain = _sites(dim, 2)
    result = _fit(points, [(0, 1, .5)], domain=domain, return_cells=cells,
                  return_boundary_measure=measures,
                  return_tessellation_diagnostics=diagnostics)
    assert result.termination == 'self_consistent'
    assert result.final_state_available
    np.testing.assert_array_equal(result.realized.realized_same_shift, [True])
    assert (result.realized.cells is not None) is cells
    assert (result.realized.boundary_measure is not None) is measures
    assert (result.tessellation_diagnostics is not None) is diagnostics
    if cells:
        assert len(result.realized.cells) == 2
    if measures:
        np.testing.assert_allclose(result.realized.boundary_measure, [10.**(dim-1)])
    _strict_report(result)


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('failure', ('infeasible', 'numerical'))
@pytest.mark.parametrize('outputs', (False, True))
def test_unavailable_failures_keep_policy_without_placeholder_outputs(
    dim, failure, outputs,
):
    points, domain = _sites(dim, 2)
    points[1, 0] = 2.
    if failure == 'infeasible':
        rows = [(0, 1, .5), (0, 1, .5)]
        model = advanced.FitModel(feasible=advanced.FixedValue([.2, .8]))
        outer, inner = 'infeasible_active_set', 'infeasible_hard_constraints'
    else:
        rows = [(0, 1, .5)]
        model = advanced.FitModel(mismatch=advanced.SquaredLoss(space='position'),
                                  feasible=advanced.Interval(1e308, 1e308))
        outer = inner = 'numerical_failure'
    with np.errstate(over='ignore', invalid='ignore'):
        result = _fit(points, rows, domain=domain, model=model, fit_solver='admm',
                      return_cells=outputs, return_boundary_measure=outputs,
                      return_tessellation_diagnostics=outputs)
    assert result.termination == outer and result.inner_fit.status == inner
    assert not result.converged and not result.final_refit_converged
    assert not result.final_state_available
    assert result.final_state_unavailable_reason == inner
    assert result.fit.weights is None and result.realized is None
    assert result.diagnostics is None and result.to_records() is None
    assert result.mismatch_predicted is None and result.mismatch_residuals is None
    assert result.rms_residual_all is None and result.max_residual_all is None
    assert result.tessellation_diagnostics is None
    assert result.resolved_policy['row_ids'] == _row_ids(result.constraints)
    report = _strict_report(result)
    assert report['availability'] == {
        'weights': False, 'realization': False, 'records': False, 'reason': inner,
    }
    assert report['model_policy'] == report['fit']['model_policy']


@pytest.mark.parametrize('dim', (2, 3))
def test_native_mixed_space_independent_quadratic_oracle(dim):
    points, domain = _sites(dim, 2)
    points[1, 0] = 2.
    model = advanced.FitModel(
        mismatch=advanced.SquaredLoss(space='position'),
        feasible=advanced.Interval(0., 1., space='fraction'),
        penalties=(advanced.SoftIntervalPenalty(3/8, 5/8, 8., space='fraction'),),
    )
    result = _fit(points, [(0, 1, 1/4)], domain=domain, model=model,
                  fit_solver='admm', fit_admm_max_iter=5000,
                  fit_admm_abs_tol=1e-10, fit_admm_rel_tol=1e-10)
    # min 1/2*(2*t-1/2)^2 + 8*(t-3/8)^2 gives t=7/20.
    assert result.converged and result.final_refit_converged
    assert result.constraints.measurement == 'fraction'
    assert result.mismatch_space == 'position'
    np.testing.assert_allclose(result.diagnostics.predicted_fraction, [7/20], atol=1e-9)
    np.testing.assert_allclose(result.mismatch_predicted, [7/10], atol=1e-9)
    assert (result.fit.weights[0] - result.fit.weights[1]
            == pytest.approx(-6/5, abs=1e-8))
    assert result.fit.objective.total == pytest.approx(1/40, abs=1e-10)
    record = result.to_records()[0]
    assert record['mismatch_space'] == 'position'
    report = _strict_report(result)
    assert report['observation_set']['measurement'] == 'fraction'
    with pytest.raises(ValueError, match='admm'):
        _fit(points, [(0, 1, 1/4)], domain=domain, model=model)


@pytest.mark.parametrize('domain_kind', ('planar', 'orthorhombic', 'periodic'))
def test_native_periodic_distinct_image_bound_policy_projection(domain_kind):
    dim = 2 if domain_kind == 'planar' else 3
    domain = (planar.RectangularCell(((0., 1.),) * 2) if dim == 2 else
              spatial.OrthorhombicCell(((0., 1.),) * 3) if domain_kind == 'orthorhombic'
              else spatial.PeriodicCell(np.eye(3)))
    points = np.full((2, dim), .5)
    points[:, 0] = [1/8, 7/8]
    shifts = [(-1,) + (0,) * (dim-1), (1,) + (0,) * (dim-1)]
    model = advanced.FitModel(
        mismatch=advanced.SquaredLoss(space='position'),
        feasible=advanced.Interval([0., .25], [1., .75], applicable=[True, False]),
        penalties=(advanced.SoftIntervalPenalty(.4, .6, [1., 4.]),),
    )
    result = _fit(points, [(0, 1, .5, shift) for shift in shifts], domain=domain,
                  image='given_only', model=model, fit_solver='admm',
                  return_cells=True, return_boundary_measure=True)
    assert result.termination == 'self_consistent'
    assert result.inner_fit.status == 'optimal'
    np.testing.assert_array_equal(result.active_mask, [True, False])
    np.testing.assert_array_equal(result.realized.realized_same_shift, [True, False])
    assert result.realized.realized_other_shift[1]
    # A wrong-image row may carry the measure of another realized image.
    # Matching flags, rather than the measure alone, decide requested support.
    np.testing.assert_allclose(result.realized.boundary_measure, [1., 1.])
    np.testing.assert_allclose(
        result.fit.weights[0] - result.fit.weights[1], 0., atol=1e-12,
    )
    assert result.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength']['values'] == (1., 4.)
    assert result.fit.resolved_policy['model_policy']['penalties'][0]['parameters'][
        'strength']['values'] == (1.,)
    assert _row_ids(result.constraints)[0] != _row_ids(result.constraints)[1]
    report = _strict_report(result)
    strength = report['model_policy']['penalties'][0]['parameters']['strength']
    selected = report['fit']['model_policy']['penalties'][0]['parameters']['strength']
    assert strength['values'] == [1., 4.]
    assert selected['values'] == [1.]
    assert report['fit']['observation_set']['row_ids'] == [
        _row_ids(result.constraints)[0],
    ]


@pytest.mark.parametrize('dim', (2, 3))
def test_native_absent_policy_allows_direct_and_retains_full_site_reference(dim):
    points, domain = _sites(dim)
    model = advanced.FitModel(
        feasible=advanced.Interval(1e308, 1e308, applicable=[False] * 3),
        penalties=(advanced.SoftIntervalPenalty(1e308, 1.5e308, [0.] * 3),),
        regularization=advanced.L2Regularization(0., [7., 8., 9.]),
    )
    result = _fit(points, [(0, 1, .5), (1, 2, .5), (0, 2, .5)],
                  domain=domain, model=model)
    assert result.inner_fit.status == 'optimal'
    assert result.inner_fit.solver == 'direct'
    np.testing.assert_array_equal(result.active_mask, [True, True, False])
    assert result.fit.resolved_policy['model_policy']['regularization'][
        'reference']['values'] == (7., 8., 9.)
    _strict_report(result)


@pytest.mark.parametrize('which', ('hard', 'penalty'))
def test_invalid_row_lengths_remain_invalid_when_policy_absent(which):
    points, domain = _sites(2)
    model = (advanced.FitModel(
        feasible=advanced.Interval([0., 0.], 1., applicable=False))
             if which == 'hard' else advanced.FitModel(
                 penalties=(advanced.SoftIntervalPenalty([0., 0.], 1., 0.),)))
    with pytest.raises(ValueError, match='shape'):
        _fit(points, [(0, 1, .5)], domain=domain, model=model)


@pytest.mark.parametrize(('name', 'value'), (
    ('measurement', 'bad'), ('index_mode', 'bad'), ('image', 'bad'),
    ('image_search', True), ('fit_solver', 'auto'), ('fit_linear_backend', 'auto'),
    ('fit_admm_max_iter', 0), ('fit_admm_rho', 0.), ('fit_admm_abs_tol', 0.),
    ('fit_admm_rel_tol', np.nan), ('r_min', -1.), ('weight_shift', np.inf),
    ('return_cells', 1), ('return_boundary_measure', 1),
    ('return_tessellation_diagnostics', 1), ('tessellation_check', 'bad'),
    ('connectivity_check', 'bad'), ('unaccounted_pair_check', 'bad'),
))
def test_unconditional_validation_also_applies_to_resolved_rows(name, value):
    points, domain = _sites(2, 2)
    observations = inverse.resolve_separator_observations(
        points, [(0, 1, .5)], domain=domain,
    )
    with pytest.raises(ValueError, match=name):
        _fit(points, observations, domain=domain, **{name: value})


@pytest.mark.parametrize('dim', (2, 3))
def test_native_resolved_measurement_ids_confidence_and_images_are_authoritative(
    dim,
):
    points, domain = _sites(dim, 2)
    observations = inverse.resolve_separator_observations(
        points, [(101, 202, .25, (0,) * dim)], domain=domain,
        ids=[101, 202], index_mode='id', image='given_only', confidence=[.75],
    )
    result = _fit(points, observations, domain=domain, measurement='position',
                  ids=[9, 8], index_mode='index', image='nearest', confidence=[0.])
    assert result.constraints is observations
    assert observations.measurement == 'fraction'
    np.testing.assert_array_equal(observations.ids, [101, 202])
    np.testing.assert_array_equal(observations.confidence, [.75])
    np.testing.assert_allclose(result.diagnostics.predicted_fraction, [.25], atol=1e-12)
    assert result.to_records(use_ids=True)[0]['site_i'] == 101


@pytest.mark.parametrize('change', (
    'points', 'domain', 'no-domain', 'equivalent-basis',
))
def test_exact_source_binding_cannot_be_reinterpreted(change):
    dim = 3
    points, domain = _sites(dim, 2)
    if change == 'equivalent-basis':
        domain = spatial.PeriodicCell(np.eye(3))
        points[:, 0] = [.125, .875]
    observations = inverse.resolve_separator_observations(
        points, [(0, 1, .5)], domain=None if change == 'no-domain' else domain,
    )
    if change == 'points':
        points[1, 1] += 2.**-40
    elif change == 'domain':
        domain = spatial.Box(((-6., 5.),) * 3)
    elif change == 'equivalent-basis':
        domain = spatial.PeriodicCell([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
    with pytest.raises(ValueError, match='source|domain|points'):
        _fit(points, observations, domain=domain)


@pytest.mark.parametrize('dim', (2, 3))
@pytest.mark.parametrize('stage', ('outer', 'final'))
def test_semantic_certificate_failures_propagate_even_without_outputs(
    monkeypatch, dim, stage,
):
    # Inject a failure at the real certificate consumer, never fabricate a native state.
    from pyvoro2 import api as spatial_api
    from pyvoro2.planar import api as planar_api

    api = planar_api if dim == 2 else spatial_api
    original = api._compute_with_certificate
    error_type = planar.TessellationError if dim == 2 else spatial.TessellationError
    calls = []

    def incomplete_audit(*args, **kwargs):
        computed, certificate = original(*args, **kwargs)
        calls.append(certificate)
        if len(calls) == (1 if stage == 'outer' else 2):
            certificate = replace(certificate, audit_complete=False)
        return computed, certificate

    monkeypatch.setattr(api, '_compute_with_certificate', incomplete_audit)
    points = np.full((2, dim), .5)
    points[:, 0] = [.25, .75]
    domain = (planar.RectangularCell(((0., 1.),) * 2) if dim == 2 else
              spatial.PeriodicCell(np.eye(3)))
    with pytest.raises(error_type) as caught:
        _fit(points, [(0, 1, .5)], domain=domain, tessellation_check='none')
    expected = 'WP6_AUDIT_INCOMPLETE' if dim == 2 else 'WP5_RESOURCE_LIMIT'
    assert any(issue.code == expected for issue in caught.value.diagnostics.issues)
    assert len(calls) == (1 if stage == 'outer' else 2)


def test_fixed_solver_retains_independent_nonrealization_contract():
    points, domain = _sites(3)
    rows = [(0, 1, .5), (1, 2, .5), (0, 2, .5)]
    observations = inverse.resolve_separator_observations(points, rows, domain=domain)
    result = inverse.fit_weights_from_separators(points, observations)
    assert result.status == 'optimal'
    assert type(result) is inverse.SeparatorFitResult
    assert len(result.to_records(observations)) == 3
    assert not hasattr(result, 'outer_termination')


def test_sparse_dependency_refusal_is_not_an_outer_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, 'scipy', None)
    monkeypatch.setitem(sys.modules, 'scipy.sparse', None)
    points, domain = _sites(2, 2)
    with pytest.raises(ImportError, match='SciPy|scipy'):
        _fit(points, [(0, 1, .5)], domain=domain, fit_linear_backend='sparse')
