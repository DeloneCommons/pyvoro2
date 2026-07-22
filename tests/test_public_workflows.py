"""Public integration assets for v0.7 issue #15."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_chemvoro_shaped_workflow_preserves_ids_metadata_and_layers() -> None:
    import pyvoro2 as pv
    from examples.chemvoro_workflow import run_workflow

    workflow = run_workflow()
    fit = workflow['fit']
    tessellation = workflow['tessellation']
    realization = workflow['realization']

    assert isinstance(tessellation, pv.TessellationResult)
    np.testing.assert_array_equal(tessellation.ids, [205, 101])
    np.testing.assert_allclose(
        tessellation.input_weights,
        fit.state.mathematical_weights,
    )
    assert tessellation.representation_shift == (
        fit.state.global_representation_shift
    )
    assert fit.identification.effective_observation_components == ((0, 1),)
    assert not fit.identification.global_geometric_gauge_identified_by_data
    assert fit.solver_termination.backend == 'analytic'

    site_records = workflow['site_records']
    assert [row['site_id'] for row in site_records] == [205, 101]
    assert site_records[0]['metadata']['label'] == 'left-site'
    assert site_records[1]['metadata']['label'] == 'right-site'
    assert all(row['empty'] is False for row in site_records)
    assert all(row['cell_measure'] > 0.0 for row in site_records)
    assert [
        {neighbor['site_id'] for neighbor in row['neighbor_images']}
        for row in site_records
    ] == [{205, 101}, {205, 101}]
    assert all('metadata' not in cell for cell in tessellation.cells)

    for records in (
        workflow['observation_records'],
        workflow['fit_records'],
        workflow['realization_records'],
        workflow['fit_report']['constraints'],
        workflow['fit_report']['fit_records'],
        workflow['realization_report']['records'],
    ):
        assert [(row['site_i'], row['site_j']) for row in records] == [
            (205, 101),
        ] * 3
    matching = realization.requested_image_matching
    assert matching.same_requested_shift.tolist() == [True, True, False]
    assert matching.another_periodic_shift.tolist() == [False, False, True]
    assert workflow['fit_report']['summary']['status'] == 'optimal'
    assert workflow['realization_report']['summary']['n_other_shift'] == 1


def test_public_paper_regression_ladder_runs_without_sparse() -> None:
    from examples.paper_regressions import run_regression_ladder

    results = run_regression_ladder(include_sparse=False)
    assert set(results) == {
        'exact_connected_recovery',
        'disconnected_and_zero_confidence',
        'hard_infeasibility',
        'realization_and_periodic_images',
        'active_set_diagnostics',
        'weight_radius_forward_equivalence',
    }
    assert results['hard_infeasibility']['status'] == (
        'infeasible_hard_constraints'
    )
    assert results['realization_and_periodic_images'] == {
        'requested_shift_realized': True,
        'another_shift_realized': True,
        'pair_not_realized': True,
        'unrealized_fit_max_residual': 0.0,
    }
    assert results['active_set_diagnostics']['inner_backend'] == 'analytic'
    assert results['active_set_diagnostics']['converged'] is True
    assert results['active_set_diagnostics']['termination'] == 'self_consistent'
    assert results['active_set_diagnostics']['final_active_mask'] == [
        True,
        True,
        False,
    ]
    assert results['weight_radius_forward_equivalence'][
        'boundary_geometry_equal'
    ] is True
    assert results['weight_radius_forward_equivalence'][
        'n_boundary_faces'
    ] > 0


def test_ci_scale_static_sparse_downstream_case() -> None:
    pytest.importorskip('scipy.sparse.linalg')
    from examples.paper_regressions import dense_sparse_static_equivalence

    result = dense_sparse_static_equivalence()
    assert result['available'] is True
    assert result['n_sites'] == 32
    assert result['n_observations'] > result['n_sites']
    assert result['dense_backend'] == 'analytic'
    assert result['sparse_backend'] == 'sparse'
    assert result['max_prediction_disagreement'] < 1e-10
    assert result['objective_disagreement'] < 1e-12


def test_static_locality_inputs_are_deterministic_and_id_keyed() -> None:
    pytest.importorskip('scipy.spatial')
    from examples.static_separator_cases import molecular_locality_inputs

    first = molecular_locality_inputs(12, neighbors=3, n_components=2)
    second = molecular_locality_inputs(12, neighbors=3, n_components=2)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.ids, second.ids)
    assert first.observations == second.observations
    assert first.components == (tuple(range(6)), tuple(range(6, 12)))
    labels = {value for row in first.observations for value in row[:2]}
    assert labels.issubset(set(first.ids.tolist()))


def test_canonical_workflow_dense_path_works_when_scipy_is_blocked() -> None:
    code = r'''
import builtins
import json

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'scipy' or name.startswith('scipy.'):
        raise ImportError('blocked optional dependency')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
from examples.chemvoro_workflow import run_workflow
workflow = run_workflow(solver='analytic')
print(json.dumps({
    'solver': workflow['fit'].solver_termination.backend,
    'ids': workflow['tessellation'].ids.tolist(),
}))
'''
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        'solver': 'analytic',
        'ids': [205, 101],
    }


def test_new_public_examples_use_only_preferred_import_routes() -> None:
    sources = [
        (REPO_ROOT / 'examples' / name).read_text(encoding='utf-8')
        for name in ('chemvoro_workflow.py', 'paper_regressions.py')
    ]
    combined = '\n'.join(sources)
    assert 'pyvoro2.powerfit' not in combined
    for historical_name in (
        'PairBisectorConstraints',
        'resolve_pair_bisector_constraints',
        'PowerWeightFitResult',
        'fit_power_weights',
    ):
        assert historical_name not in combined
    chemvoro_source = sources[0]
    assert 'weights=state.mathematical_weights' in chemvoro_source
    assert 'radii=' not in chemvoro_source


def test_realization_accepts_exactly_one_power_representation() -> None:
    import pyvoro2 as pv
    import pyvoro2.inverse as inverse
    import pyvoro2.inverse.separator as separator

    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    box = pv.Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    observations = inverse.resolve_separator_observations(
        points,
        [(0, 1, 0.25)],
        domain=box,
    )
    fit = inverse.fit_weights_from_separators(points, observations)
    by_weights = separator.match_realized_pairs(
        points,
        domain=box,
        constraints=observations,
        weights=fit.state.mathematical_weights,
    )
    by_radii = separator.match_realized_pairs(
        points,
        domain=box,
        constraints=observations,
        radii=fit.state.backend_radii,
    )
    assert by_weights.to_records(observations) == by_radii.to_records(
        observations
    )

    with pytest.raises(ValueError, match='exactly one of weights or radii'):
        separator.match_realized_pairs(
            points,
            domain=box,
            constraints=observations,
        )
    with pytest.raises(ValueError, match='mutually exclusive'):
        separator.match_realized_pairs(
            points,
            domain=box,
            constraints=observations,
            weights=fit.state.mathematical_weights,
            radii=fit.state.backend_radii,
        )
