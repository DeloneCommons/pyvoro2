<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/04_powerfit.ipynb)
# Power fitting from separator observations

This notebook shows the new math-oriented inverse API in `pyvoro2`:

1. keep downstream metadata in an external-ID sidecar,
2. resolve periodic separator observations and fit mathematical weights,
3. compute and inspect a structured power tessellation from those weights,
4. match requested periodic images, and
5. run the experimental self-consistent active-set solver.
```python
import numpy as np

import pyvoro2 as pv
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator
```
## 1) Resolve and fit periodic observations by external ID

A raw constraint tuple is `(i, j, value[, shift])`, where `value` is
interpreted in either fraction-space or absolute position-space. Domain-
specific metadata remains downstream in an explicit ID-keyed sidecar.
```python
points = np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=float)
site_ids = np.array([205, 101], dtype=int)
metadata_by_id = {
    205: {'label': 'left-site', 'source_row': 0},
    101: {'label': 'right-site', 'source_row': 1},
}
cell = pv.PeriodicCell(
    vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
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
    domain=cell,
    image='given_only',
)

fit = inverse.fit_weights_from_separators(
    points, observations, connectivity_check='diagnose'
)
state = fit.state
observation_fit = fit.observation_view(observations)
identification = fit.identification

print('mathematical weights:', state.mathematical_weights)
print('predicted fractions:', observation_fit.predicted_fraction)
print('global representation shift:', state.global_representation_shift)
print('observation components:', identification.effective_observation_components)
print('component alignment:', identification.component_alignment_policy)
```
## 2) Add hard feasibility and a near-boundary penalty

The fitting model separates mismatch, hard feasibility, and soft penalties.
```python
model = separator.FitModel(
    mismatch=separator.SquaredLoss(),
    feasible=separator.Interval(0.0, 1.0),
    penalties=(
        separator.ExponentialBoundaryPenalty(
            lower=0.0,
            upper=1.0,
            margin=0.05,
            strength=1.0,
            tau=0.01,
        ),
    ),
)

simple_points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

fit_penalized = inverse.fit_weights_from_separators(
    simple_points,
    [(0, 1, 1e-3)],
    measurement='fraction',
    domain=box,
    model=model,
    solver='admm',
)

print('predicted fraction with penalty:', fit_penalized.predicted_fraction[0])
```
## 3) Compute structured cells and match requested images

The forward call consumes fitted mathematical weights directly. Cell measures,
empty state, and boundary collections are aligned with the input IDs rather
than raw backend order. Requested separator rows are then checked without
silently replacing their periodic images.
```python
tessellation = pv.compute(
    points,
    domain=cell,
    ids=site_ids,
    mode='power',
    weights=state.mathematical_weights,
    include_empty=True,
    return_vertices=True,
    return_adjacency=False,
    return_faces=True,
    return_face_shifts=True,
)
boundaries = tessellation.require_boundaries()
site_records = [
    {
        'site_id': int(site_id),
        'metadata': metadata_by_id[int(site_id)],
        'empty': bool(tessellation.empty_mask[position]),
        'cell_measure': (
            None
            if tessellation.empty_mask[position]
            else float(tessellation.cell_measures[position])
        ),
        'boundary_count': len(boundaries[position]),
    }
    for position, site_id in enumerate(tessellation.ids)
]

realized = separator.match_realized_pairs(
    points,
    domain=cell,
    weights=state.mathematical_weights,
    constraints=observations,
    return_boundary_measure=True,
    return_tessellation_diagnostics=True,
)

print('sites:', site_records)
print('same requested shift:', realized.requested_image_matching.same_requested_shift)
print('another shift:', realized.requested_image_matching.another_periodic_shift)
print('boundary measure:', realized.geometry.boundary_measure)
print('tessellation ok:', realized.geometry.tessellation_diagnostics.ok)
print('ID-labelled fit records:', fit.to_records(observations, use_ids=True))
```
## 4) Self-consistent active-set refinement

For larger candidate sets, the active-set solver repeatedly fits, tessellates,
and keeps the constraints whose requested pairs are actually realized.
```python
points3 = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box3 = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result = separator.solve_self_consistent_power_weights(
    points3,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement='fraction',
    domain=box3,
    options=separator.ActiveSetOptions(add_after=1, drop_after=2, relax=0.5),
    return_history=True,
    return_boundary_measure=True,
)
outer = result.outer_termination
path = result.path
candidate_diagnostics = result.candidate_diagnostics

print('termination:', outer.status)
print('active mask:', path.active_mask)
print('constraint status:', candidate_diagnostics.status)
print('marginal constraints:', path.marginal_constraint_indices)
print('path summary:', path.summary)
```
## Disconnected path example

The next example starts from an empty active set so the first fitted subproblem is completely disconnected, while the final active set reconnects into the expected nearest-neighbor chain. This illustrates the difference between final-state diagnostics and optimization-path diagnostics.
```python
points4 = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box4 = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result_path = separator.solve_self_consistent_power_weights(
    points4,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement='fraction',
    domain=box4,
    active0=np.array([False, False, False]),
    options=separator.ActiveSetOptions(add_after=1, drop_after=1, max_iter=6),
    return_history=True,
    connectivity_check='diagnose',
    unaccounted_pair_check='diagnose',
)
path = result_path.path

print('final active graph components:', result_path.connectivity.active_graph.n_components)
print('path summary:', path.summary)
print('first history row:', path.history[0])
```
