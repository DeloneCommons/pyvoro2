"""Manual integrated preparation/compute timings, not a CI performance gate."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import timeit

import numpy as np
import pyvoro2 as pv
from pyvoro2 import _core, _core2d
from pyvoro2._internal.generator_preparation import prepare_generators
from pyvoro2._internal.spatial.domain_geometry import geometry3d


root = Path(__file__).resolve().parents[4]
origin = np.array([1., -2., .5])
measurements = []
for n in (1000, 4000):
    rng = np.random.default_rng(166 + n)
    fractions = rng.integers(1, 2**16, size=(n, 3)) / 2**16
    lifts = rng.integers(-2, 3, size=(n, 3))
    for kind in ('rectangular', 'oblique'):
        basis = (np.eye(3) * 10 if kind == 'rectangular' else
                 np.array([[10., 2.5, 0.], [0., 10., 1.25], [-1.25, 0., 10.]]))
        domain = (pv.OrthorhombicCell(((1., 11.), (-2., 8.), (.5, 10.5)))
                  if kind == 'rectangular' else pv.PeriodicCell(basis, origin=origin))
        local = origin + fractions @ basis
        translated = local + lifts @ basis
        # These are exact dyadic cohorts, not rounded generic random translations.
        for i in range(n):
            for j in range(3):
                assert F(translated[i, j]) == F(local[i, j]) + sum(
                    int(lifts[i, a]) * F(basis[a, j]) for a in range(3))
        geometry = geometry3d(domain)
        for cohort, points in (('ordinary', local), ('translated', translated)):
            def prepare():
                snapshot = geometry.native_periodic_snapshot() if kind == 'oblique' else None
                return prepare_generators(
                    points, geometry=geometry, operation='compute', external_ids=None,
                    backend_radii=None, duplicate_check='off', duplicate_threshold=1e-5,
                    duplicate_wrap=True, duplicate_max_pairs=10, periodic_snapshot=snapshot)

            def compute():
                return pv.compute(points, domain=domain, duplicate_check='off',
                                  return_vertices=False, return_adjacency=False,
                                  return_faces=False)

            samples = {}
            for name, function in [('preparation', prepare), ('compute', compute)]:
                function()
                values = timeit.repeat(function, number=1, repeat=3)
                samples[name] = dict(seconds=values, median_seconds=statistics.median(values))
            measurements.append(dict(n=n, domain=kind, cohort=cohort, timings=samples))
            print(f'{n} {kind} {cohort} done', file=sys.stderr, flush=True)

provenance = dict(
    revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    python=sys.version, numpy=np.__version__, platform=platform.platform(),
    source=pv.__file__,
    native=[dict(path=m.__file__, sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
            for m in (_core, _core2d)],
    native_inputs=subprocess.check_output(
        ['git', '-C', str(root), 'ls-tree', 'HEAD', 'cpp', 'vendor', 'CMakeLists.txt', 'pyproject.toml'],
        text=True),
)
print(json.dumps(dict(provenance=provenance, seed='166+n', warmups=1,
                      repeats=3, number=1, measurements=measurements), indent=2))
