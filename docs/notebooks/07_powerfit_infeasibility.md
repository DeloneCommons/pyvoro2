<!-- This file is generated from the matching notebook. -->
<!-- Regenerate with: python tools/export_notebooks.py -->
[Open the original notebook on GitHub](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/07_powerfit_infeasibility.ipynb)
# Hard infeasibility witnesses in power fitting

This notebook shows how the low-level inverse solver reports hard
infeasibility when the requested equalities or bounds cannot all be
satisfied at once.
```python
import numpy as np

import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator
```
## 1) Build a contradictory hard system

For three collinear sites, forcing all pairwise separator positions to be
at absolute position `0.0` is impossible.
```python
points = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ],
    dtype=float,
)
ids = np.array([10, 11, 12], dtype=int)
raw_observations = [
    (10, 11, 0.0),
    (11, 12, 0.0),
    (10, 12, 0.0),
]
observations = inverse.resolve_separator_observations(
    points,
    raw_observations,
    measurement="position",
    ids=ids,
    index_mode="id",
)
```
```python
fit = inverse.fit_weights_from_separators(
    points,
    observations,
    model=separator.FitModel(feasible=separator.FixedValue(0.0)),
    solver="admm",
)
termination = fit.solver_termination

termination.status, termination.hard_feasible, fit.is_infeasible
```
**Output**

```text
('infeasible_hard_constraints', False, True)
```
## 2) Inspect the contradiction witness
```python
fit.conflicting_constraint_indices
```
**Output**

```text
(0, 1, 2)
```
```python
termination.conflict.message
```
**Output**

```text
'inconsistent hard separator restrictions on connected component [0, 1, 2]; contradiction cycle uses constraint rows [0, 1, 2]'
```
```python
termination.conflict.to_records(ids=ids)
```
**Output**

```text
({'constraint_index': 0,
  'site_i': 10,
  'site_j': 11,
  'relation': '>=',
  'bound_value': -4.0},
 {'constraint_index': 1,
  'site_i': 11,
  'site_j': 12,
  'relation': '>=',
  'bound_value': -4.0},
 {'constraint_index': 2,
  'site_i': 10,
  'site_j': 12,
  'relation': '<=',
  'bound_value': -16.0})
```
## 3) Export the same information through the report helper
```python
fit_report = fit.to_report(observations, use_ids=True)
fit_report["conflict"]
```
**Output**

```text
{'message': 'inconsistent hard separator restrictions on connected component [0, 1, 2]; contradiction cycle uses constraint rows [0, 1, 2]',
 'component_nodes': [10, 11, 12],
 'cycle_nodes': [10, 11, 12],
 'constraint_indices': [0, 1, 2],
 'terms': [{'constraint_index': 0,
   'site_i': 10,
   'site_j': 11,
   'relation': '>=',
   'bound_value': -4.0},
  {'constraint_index': 1,
   'site_i': 11,
   'site_j': 12,
   'relation': '>=',
   'bound_value': -4.0},
  {'constraint_index': 2,
   'site_i': 10,
   'site_j': 12,
   'relation': '<=',
   'bound_value': -16.0}]}
```
The contradiction witness is intended to be compact and actionable rather
than a full proof certificate.
