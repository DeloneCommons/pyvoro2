<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/08_powerfit_active_path.ipynb)
# Active-set path diagnostics

This notebook focuses on the difference between **final-state** diagnostics and **optimization-path** diagnostics in `solve_self_consistent_power_weights(...)`. The path diagnostics are especially useful when the active graph is transiently disconnected, even though the final returned solution is connected.
```python
import numpy as np
import pyvoro2 as pv
import pyvoro2.inverse.separator as separator
```
## A chain example with an initially empty active set

The candidate graph is connected through the nearest-neighbor chain, but the first fitted subproblem is completely disconnected because `active0` is empty. The final active set reconnects after the first realization pass.
```python
points = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    dtype=float,
)
box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

result = separator.solve_self_consistent_power_weights(
    points,
    [(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)],
    measurement="fraction",
    domain=box,
    active0=np.array([False, False, False]),
    options=separator.ActiveSetOptions(add_after=1, drop_after=1, max_iter=6),
    return_history=True,
    connectivity_check="diagnose",
    unaccounted_pair_check="diagnose",
)
outer = result.outer_termination
path = result.path

print("termination:", outer.status)
print("final active mask:", path.active_mask)
print("final active graph components:", result.connectivity.active_graph.n_components)
print("path summary:", path.summary)
```
**Output**

```text
termination: self_consistent
final active mask: [ True  True False]
final active graph components: 1
path summary: ActiveSetPathSummary(n_iterations=2, ever_fit_active_graph_disconnected=True, ever_fit_active_effective_graph_disconnected=True, ever_fit_active_offsets_unidentified_by_data=True, ever_unaccounted_pairs=False, max_fit_active_graph_components=3, max_fit_active_effective_graph_components=3, max_n_unaccounted_pairs=0, first_fit_active_graph_disconnected_iter=1, first_fit_active_effective_graph_disconnected_iter=1, first_unaccounted_pairs_iter=None)
```
```python
for row in path.history:
    print(row)
```
**Output**

```text
ActiveSetIteration(iteration=1, n_active=2, n_realized=2, n_added=2, n_removed=0, rms_residual_all=0.0, max_residual_all=0.0, weight_step_norm=0.0, n_active_fit=0, fit_active_graph_n_components=3, fit_active_effective_graph_n_components=3, fit_active_offsets_identified_by_data=False, n_unaccounted_pairs=0)
ActiveSetIteration(iteration=2, n_active=2, n_realized=2, n_added=0, n_removed=0, rms_residual_all=0.0, max_residual_all=0.0, weight_step_norm=0.0, n_active_fit=2, fit_active_graph_n_components=1, fit_active_effective_graph_n_components=1, fit_active_offsets_identified_by_data=True, n_unaccounted_pairs=0)
```
Notice the distinction between `n_active_fit` (the mask that actually generated the current iterate) and `n_active` (the post-toggle mask used for the next iterate). This lets downstream code say whether disconnectivity happened **during** optimization, not just in the final answer.
```python
solve_report = result.to_report()
solve_report["path_summary"]
```
**Output**

```text
{'n_iterations': 2,
 'ever_fit_active_graph_disconnected': True,
 'ever_fit_active_effective_graph_disconnected': True,
 'ever_fit_active_offsets_unidentified_by_data': True,
 'ever_unaccounted_pairs': False,
 'max_fit_active_graph_components': 3,
 'max_fit_active_effective_graph_components': 3,
 'max_n_unaccounted_pairs': 0,
 'first_fit_active_graph_disconnected_iter': 1,
 'first_fit_active_effective_graph_disconnected_iter': 1,
 'first_unaccounted_pairs_iter': None}
```
