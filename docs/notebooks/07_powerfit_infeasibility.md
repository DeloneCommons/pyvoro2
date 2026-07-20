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
    (0, 1, 0.0),
    (1, 2, 0.0),
    (0, 2, 0.0),
]
```
```python
fit = inverse.fit_weights_from_separators(
    points,
    raw_observations,
    measurement="position",
    ids=ids,
    model=separator.FitModel(feasible=separator.FixedValue(0.0)),
    solver="admm",
)

fit.status, fit.hard_feasible, fit.is_infeasible
```
## 2) Inspect the contradiction witness
```python
fit.conflicting_constraint_indices
```
```python
fit.conflict.message
```
```python
fit.conflict.to_records(ids=ids)
```
## 3) Export the same information through the report helper
```python
observations = inverse.resolve_separator_observations(
    points,
    raw_observations,
    measurement="position",
    ids=ids,
)
fit_report = fit.to_report(observations, use_ids=True)
fit_report["conflict"]
```
The contradiction witness is intended to be compact and actionable rather
than a full proof certificate.
