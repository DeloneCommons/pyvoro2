#!/usr/bin/env python3
"""Canonical preferred-API workflow for a downstream scientific package."""

from __future__ import annotations

import json

import numpy as np

import pyvoro2 as pv
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator


def run_workflow(*, solver: str = 'analytic') -> dict[str, object]:
    """Fit and realize a small periodic system using external IDs throughout."""

    points = np.array(
        [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]],
        dtype=np.float64,
    )
    site_ids = np.array([205, 101], dtype=np.int64)
    metadata_by_id = {
        205: {'label': 'left-site', 'source_row': 0},
        101: {'label': 'right-site', 'source_row': 1},
    }
    domain = pv.PeriodicCell(
        vectors=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )

    observations = inverse.resolve_separator_observations(
        points,
        [
            (205, 101, 0.5, (-1, 0, 0)),
            (205, 101, 0.5, (0, 0, 0)),
            (205, 101, 0.5, (1, 0, 0)),
        ],
        ids=site_ids,
        index_mode='id',
        measurement='fraction',
        domain=domain,
        image='given_only',
    )
    fit = inverse.fit_weights_from_separators(
        points,
        observations,
        solver=solver,
        connectivity_check='diagnose',
    )
    state = fit.state
    if state.mathematical_weights is None:
        raise RuntimeError(f'separator fit failed with status {fit.status!r}')

    tessellation = pv.compute(
        points,
        domain=domain,
        ids=site_ids,
        mode='power',
        weights=state.mathematical_weights,
        include_empty=True,
        return_vertices=True,
        return_adjacency=False,
        return_faces=True,
        return_face_shifts=True,
        tessellation_check='diagnose',
    )
    boundaries_by_input = tessellation.require_boundaries()

    realization = separator.match_realized_pairs(
        points,
        domain=domain,
        weights=state.mathematical_weights,
        constraints=observations,
        return_boundary_measure=True,
        return_tessellation_diagnostics=True,
    )

    site_records: list[dict[str, object]] = []
    for position, site_id_value in enumerate(tessellation.ids):
        site_id = int(site_id_value)
        empty = bool(tessellation.empty_mask[position])
        neighbor_images = []
        for boundary in boundaries_by_input[position]:
            neighbor_id = int(boundary.get('adjacent_cell', -1))
            if neighbor_id < 0:
                continue
            neighbor_images.append(
                {
                    'site_id': neighbor_id,
                    'shift': tuple(
                        int(value)
                        for value in boundary['adjacent_shift']
                    ),
                }
            )
        site_records.append(
            {
                'site_id': site_id,
                'metadata': metadata_by_id[site_id],
                'mathematical_weight': float(
                    state.mathematical_weights[position]
                ),
                'empty': empty,
                'cell_measure': (
                    None
                    if empty
                    else float(tessellation.cell_measures[position])
                ),
                'neighbor_images': tuple(neighbor_images),
            }
        )

    return {
        'points': points,
        'site_ids': site_ids,
        'metadata_by_id': metadata_by_id,
        'observations': observations,
        'fit': fit,
        'tessellation': tessellation,
        'realization': realization,
        'site_records': tuple(site_records),
        'observation_records': observations.to_records(use_ids=True),
        'fit_records': fit.to_records(observations, use_ids=True),
        'realization_records': realization.to_records(
            observations,
            use_ids=True,
        ),
        'fit_report': fit.to_report(observations, use_ids=True),
        'realization_report': realization.to_report(
            observations,
            use_ids=True,
        ),
    }


def _summary(workflow: dict[str, object]) -> dict[str, object]:
    fit = workflow['fit']
    tessellation = workflow['tessellation']
    realization = workflow['realization']
    return {
        'solver': fit.solver_termination.backend,
        'site_ids': [int(value) for value in tessellation.ids],
        'mathematical_weights': fit.state.mathematical_weights.tolist(),
        'global_representation_shift': (
            fit.state.global_representation_shift
        ),
        'effective_observation_components': (
            fit.identification.effective_observation_components
        ),
        'empty_sites': [
            int(tessellation.ids[index])
            for index in np.flatnonzero(tessellation.empty_mask)
        ],
        'requested_shift_realized': (
            realization.requested_image_matching.same_requested_shift.tolist()
        ),
        'another_shift_realized': (
            realization.requested_image_matching.another_periodic_shift.tolist()
        ),
        'fit_record_count': len(workflow['fit_records']),
    }


if __name__ == '__main__':
    print(json.dumps(_summary(run_workflow()), indent=2))
