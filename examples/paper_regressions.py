#!/usr/bin/env python3
"""Compact deterministic paper-style regression ladder for public APIs."""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

import pyvoro2 as pv
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

try:
    from examples.static_separator_cases import molecular_locality_inputs
except ModuleNotFoundError as exc:
    if exc.name != 'examples':
        raise
    from static_separator_cases import molecular_locality_inputs


def exact_connected_recovery() -> dict[str, object]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [2.1, 0.8, 0.2],
            [2.8, 1.4, 0.6],
        ],
        dtype=np.float64,
    )
    expected_weights = np.array([0.30, -0.20, 0.10, -0.20])
    pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
    rows = _compatible_fraction_rows(points, expected_weights, pairs)
    observations = inverse.resolve_separator_observations(points, rows)
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='analytic',
        connectivity_check='diagnose',
    )
    observation_fit = fit.observation_view(observations)
    edge = fit.algebraic.edge_diagnostics
    assert fit.status == 'optimal' and edge is not None
    expected_differences = expected_weights[observations.i] - (
        expected_weights[observations.j]
    )
    np.testing.assert_allclose(edge.z_fit, expected_differences, atol=1e-12)
    np.testing.assert_allclose(
        observation_fit.predictions,
        observation_fit.targets,
        atol=1e-12,
    )
    assert fit.identification.relative_component_offsets_identified_by_data
    return {
        'max_difference_error': float(
            np.max(np.abs(edge.z_fit - expected_differences))
        ),
        'max_prediction_error': float(
            np.max(
                np.abs(
                    observation_fit.predictions - observation_fit.targets
                )
            )
        ),
    }


def disconnected_and_zero_confidence() -> dict[str, object]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    observations = inverse.resolve_separator_observations(
        points,
        [
            (0, 1, 0.25),
            (2, 3, 0.70),
            (1, 2, 0.99),
        ],
        confidence=[1.0, 2.0, 0.0],
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        solver='analytic',
        connectivity_check='diagnose',
    )
    identification = fit.identification
    edge = fit.algebraic.edge_diagnostics
    assert fit.status == 'optimal' and edge is not None
    assert identification.effective_observation_components == (
        (0, 1),
        (2, 3),
        (4,),
    )
    assert not identification.relative_component_offsets_identified_by_data
    assert not identification.component_offsets_selected_by_objective
    assert edge.edge_weight[2] == 0.0
    assert len(fit.to_records(observations)) == 3
    np.testing.assert_allclose(edge.z_fit[:2], edge.z_obs[:2], atol=1e-12)
    return {
        'components': identification.effective_observation_components,
        'unconstrained_sites': identification.unconstrained_sites,
        'zero_confidence_row_retained': True,
        'component_alignment_policy': (
            identification.component_alignment_policy
        ),
    }


def hard_infeasibility() -> dict[str, object]:
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    ids = np.array([10, 20, 30], dtype=np.int64)
    observations = inverse.resolve_separator_observations(
        points,
        [(10, 20, 0.0), (20, 30, 0.0), (10, 30, 0.0)],
        ids=ids,
        index_mode='id',
        measurement='position',
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        model=separator.FitModel(feasible=separator.FixedValue(0.0)),
        solver='admm',
        connectivity_check='diagnose',
    )
    termination = fit.solver_termination
    assert fit.is_infeasible and not termination.hard_feasible
    assert termination.conflict is not None
    assert fit.state.mathematical_weights is None
    witness = termination.conflict.to_records(ids=ids)
    assert set(fit.conflicting_constraint_indices) == {0, 1, 2}
    return {
        'status': termination.status,
        'constraint_indices': fit.conflicting_constraint_indices,
        'witness': witness,
    }


def realization_and_periodic_images() -> dict[str, object]:
    cell = pv.PeriodicCell(
        vectors=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    periodic_points = np.array(
        [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]],
        dtype=np.float64,
    )
    periodic_observations = inverse.resolve_separator_observations(
        periodic_points,
        [
            (0, 1, 0.5, (-1, 0, 0)),
            (0, 1, 0.5, (1, 0, 0)),
        ],
        domain=cell,
        image='given_only',
    )
    periodic_fit = inverse.fit_weights_from_separators(
        periodic_points,
        periodic_observations,
        solver='analytic',
    )
    periodic_realization = separator.match_realized_pairs(
        periodic_points,
        domain=cell,
        weights=periodic_fit.state.mathematical_weights,
        constraints=periodic_observations,
    )
    matching = periodic_realization.requested_image_matching
    assert matching.same_requested_shift.tolist() == [True, False]
    assert matching.another_periodic_shift.tolist() == [False, True]

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    box = pv.Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    unsupported_observation = inverse.resolve_separator_observations(
        points,
        [(0, 2, 0.5)],
        domain=box,
    )
    algebraic_fit = inverse.fit_weights_from_separators(
        points,
        unsupported_observation,
        solver='analytic',
    )
    fit_view = algebraic_fit.observation_view(unsupported_observation)
    assert fit_view.residuals is not None
    np.testing.assert_allclose(fit_view.residuals, 0.0, atol=1e-12)
    unsupported_realization = separator.match_realized_pairs(
        points,
        domain=box,
        weights=algebraic_fit.state.mathematical_weights,
        constraints=unsupported_observation,
    )
    assert not bool(
        unsupported_realization.requested_image_matching.any_realization[0]
    )
    return {
        'requested_shift_realized': True,
        'another_shift_realized': True,
        'pair_not_realized': True,
        'unrealized_fit_max_residual': float(
            np.max(np.abs(fit_view.residuals))
        ),
    }


def active_set_diagnostics() -> dict[str, object]:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    box = pv.Box(((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)))
    result = separator.solve_self_consistent_power_weights(
        points,
        [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
        domain=box,
        active0=np.array([False, False, False]),
        options=separator.ActiveSetOptions(
            add_after=1,
            drop_after=1,
            max_iter=6,
        ),
        fit_solver='analytic',
        return_history=True,
        connectivity_check='diagnose',
        unaccounted_pair_check='diagnose',
    )
    termination = result.outer_termination
    path = result.path
    assert path.history is not None and path.summary is not None
    assert path.summary.ever_fit_active_effective_graph_disconnected
    assert result.inner_fit.solver_termination.backend == 'analytic'
    assert termination.converged
    assert termination.status == 'self_consistent'
    np.testing.assert_array_equal(
        path.active_mask,
        np.array([True, True, False]),
    )
    return {
        'termination': termination.status,
        'converged': termination.converged,
        'n_outer_iter': termination.n_outer_iter,
        'final_active_mask': path.active_mask.tolist(),
        'ever_disconnected': (
            path.summary.ever_fit_active_effective_graph_disconnected
        ),
        'inner_backend': result.inner_fit.solver_termination.backend,
    }


def weight_radius_forward_equivalence() -> dict[str, object]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.4, 1.1, 0.3],
            [0.2, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    ids = np.array([11, 23, 47, 89], dtype=np.int64)
    weights = np.array([-0.3, 0.2, 0.1, -0.1], dtype=np.float64)
    radii, shift = inverse.weights_to_radii(weights)
    box = pv.Box(((-2.0, 3.0), (-2.0, 3.0), (-2.0, 3.0)))
    common = {
        'domain': box,
        'ids': ids,
        'mode': 'power',
        'include_empty': True,
        'return_vertices': True,
        'return_adjacency': False,
        'return_faces': True,
    }
    by_weights = pv.compute(points, weights=weights, **common)
    by_radii = pv.compute(points, radii=radii, **common)
    np.testing.assert_allclose(
        by_weights.cell_measures,
        by_radii.cell_measures,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(by_weights.empty_mask, by_radii.empty_mask)
    weight_geometry = _boundary_geometry_signature(by_weights)
    radius_geometry = _boundary_geometry_signature(by_radii)
    assert weight_geometry == radius_geometry
    assert by_weights.representation_shift == shift
    assert by_radii.representation_shift is None
    return {
        'representation_shift': float(shift),
        'boundary_geometry_equal': True,
        'n_boundary_faces': len(weight_geometry),
        'max_measure_difference': float(
            np.max(
                np.abs(
                    by_weights.cell_measures - by_radii.cell_measures
                )
            )
        ),
    }


def dense_sparse_static_equivalence() -> dict[str, object]:
    if importlib.util.find_spec('scipy') is None:
        return {'available': False}
    inputs = molecular_locality_inputs(
        32,
        neighbors=4,
        target_perturbation=2.0e-3,
    )
    observations = inverse.resolve_separator_observations(
        inputs.points,
        inputs.observations,
        ids=inputs.ids,
        index_mode='id',
        confidence=inputs.confidence,
    )
    dense = inverse.fit_weights_from_separators(
        inputs.points,
        observations,
        solver='analytic',
        connectivity_check='diagnose',
    )
    sparse = inverse.fit_weights_from_separators(
        inputs.points,
        observations,
        solver='sparse',
        connectivity_check='diagnose',
    )
    assert dense.solver_termination.backend == 'analytic'
    assert sparse.solver_termination.backend == 'sparse'
    assert dense.objective is not None and sparse.objective is not None
    assert dense.algebraic.edge_diagnostics is not None
    assert sparse.algebraic.edge_diagnostics is not None
    dense_view = dense.observation_view(observations)
    sparse_view = sparse.observation_view(observations)
    np.testing.assert_allclose(
        dense.algebraic.edge_diagnostics.z_fit,
        sparse.algebraic.edge_diagnostics.z_fit,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense_view.predictions,
        sparse_view.predictions,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense_view.residuals,
        sparse_view.residuals,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense.objective.total,
        sparse.objective.total,
        rtol=1e-10,
        atol=1e-12,
    )
    assert {row['site_i'] for row in sparse.to_records(
        observations,
        use_ids=True,
    )}.issubset(set(inputs.ids.tolist()))
    return {
        'available': True,
        'n_sites': int(inputs.points.shape[0]),
        'n_observations': observations.n_constraints,
        'dense_backend': dense.solver_termination.backend,
        'sparse_backend': sparse.solver_termination.backend,
        'max_prediction_disagreement': float(
            np.max(
                np.abs(
                    dense_view.predictions - sparse_view.predictions
                )
            )
        ),
        'objective_disagreement': abs(
            float(dense.objective.total) - float(sparse.objective.total)
        ),
    }


def run_regression_ladder(*, include_sparse: bool = True) -> dict[str, object]:
    results = {
        'exact_connected_recovery': exact_connected_recovery(),
        'disconnected_and_zero_confidence': (
            disconnected_and_zero_confidence()
        ),
        'hard_infeasibility': hard_infeasibility(),
        'realization_and_periodic_images': (
            realization_and_periodic_images()
        ),
        'active_set_diagnostics': active_set_diagnostics(),
        'weight_radius_forward_equivalence': (
            weight_radius_forward_equivalence()
        ),
    }
    if include_sparse:
        results['dense_sparse_static_equivalence'] = (
            dense_sparse_static_equivalence()
        )
    return results


def _compatible_fraction_rows(
    points: np.ndarray,
    weights: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, float], ...]:
    rows: list[tuple[int, int, float]] = []
    for i, j in pairs:
        delta = points[j] - points[i]
        distance2 = float(np.dot(delta, delta))
        fraction = 0.5 + (weights[i] - weights[j]) / (2.0 * distance2)
        rows.append((i, j, float(fraction)))
    return tuple(rows)


def _boundary_geometry_signature(
    result: pv.TessellationResult,
) -> tuple[tuple[object, ...], ...]:
    """Return an ID-keyed, boundary-order-independent face signature."""

    cells_by_id = {int(cell['id']): cell for cell in result.cells}
    signatures: list[tuple[object, ...]] = []
    for site_id_value, boundaries in zip(
        result.ids,
        result.require_boundaries(),
    ):
        site_id = int(site_id_value)
        vertices = np.round(
            np.asarray(
                cells_by_id[site_id].get('vertices', ()),
                dtype=np.float64,
            ).reshape((-1, result.dimension)),
            decimals=9,
        )
        for boundary in boundaries:
            face_vertices = tuple(
                sorted(
                    tuple(float(value) for value in vertices[int(index)])
                    for index in boundary.get('vertices', ())
                )
            )
            shift_value = boundary.get('adjacent_shift')
            shift = (
                None
                if shift_value is None
                else tuple(int(value) for value in shift_value)
            )
            signatures.append(
                (
                    site_id,
                    int(boundary.get('adjacent_cell', -1)),
                    shift,
                    face_vertices,
                )
            )
    return tuple(sorted(signatures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-sparse',
        action='store_true',
        help='run the NumPy-only ladder without the optional SciPy case',
    )
    args = parser.parse_args()
    results = run_regression_ladder(include_sparse=not args.skip_sparse)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
