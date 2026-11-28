"""Manual representative timing check; no acceptance threshold or CI hook."""
import json
import statistics
import time

import numpy as np

from pyvoro2 import OrthorhombicCell, PeriodicCell
from pyvoro2.api import _compute_for_matching, compute
from pyvoro2.inverse.separator import (
    match_realized_pairs, resolve_separator_observations,
)


rng = np.random.default_rng(20261128)
records = []
domains = [
    ('rectangular', OrthorhombicCell(((0., 1.),) * 3)),
    ('triclinic', PeriodicCell(((1., 0., 0.), (.25, 1., 0.), (.125, .25, 1.)))),
]
for name, domain in domains:
    points = rng.uniform(.1, .9, (256, 3))
    weights = np.zeros(len(points))
    obs = resolve_separator_observations(
        points, [(i, i + 1, .5) for i in range(31)], domain=domain)
    kwargs = dict(
        domain=domain, mode='power', weights=weights,
        return_vertices=True, return_faces=True, return_adjacency=False,
        return_face_shifts=True, include_empty=True, output='cells',
    )
    calls = {
        'compute': lambda: compute(points, **kwargs),
        'capture': lambda: _compute_for_matching(points, **kwargs),
        'matching': lambda: match_realized_pairs(
            points, domain=domain, weights=np.zeros(len(points)), constraints=obs,
            tessellation_check='none', unaccounted_pair_check='diagnose'),
    }
    for run in calls.values():
        run()
    samples = {key: [] for key in calls}
    for repeat in range(7):
        order = list(calls) if repeat % 2 == 0 else list(calls)[::-1]
        for key in order:
            start = time.perf_counter()
            calls[key]()
            samples[key].append((time.perf_counter() - start) * 1000)
    records.append(dict(
        domain=name, n=256, observations=31, samples_ms=samples,
        median_ms={k: statistics.median(v) for k, v in samples.items()},
    ))
print(json.dumps(dict(seed=20261128, warmups=1, repeats=7, records=records), indent=2))
