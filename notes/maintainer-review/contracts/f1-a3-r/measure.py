"""Bounded local timing experiment; not a benchmark/CI framework."""
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
from pyvoro2._internal.spatial.domain_geometry import geometry3d
from pyvoro2._internal.generator_preparation import prepare_generators


def provenance():
    root = str(Path(__file__).resolve().parents[4])
    return dict(
        revision=subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip(),
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        python_source=pv.__file__,
        native=[dict(path=m.__file__, sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
                for m in (_core, _core2d)],
        native_source_tree=subprocess.check_output(
            ['git', '-C', root, 'ls-tree', 'HEAD', 'cpp', 'vendor', 'CMakeLists.txt'], text=True),
    )


rows = []
for n in (1000, 4000):
    rng = np.random.default_rng(20261127 + n)
    for kind in ('rectangular', 'triclinic'):
        domain = (pv.OrthorhombicCell(((1., 11.), (-2., 8.), (.5, 10.5)))
                  if kind == 'rectangular' else
                  pv.PeriodicCell.from_params(10, 2.5, 10, -1.25, .75, 10, origin=(1., -2., .5)))
        basis = np.array(domain.lattice_vectors if kind == 'rectangular' else domain.vectors)
        points = np.array([1., -2., .5]) + (rng.random((n, 3)) + rng.integers(-2, 3, (n, 3))) @ basis
        geom = geometry3d(domain)

        def prepare():
            snapshot = geom.native_periodic_snapshot() if kind == 'triclinic' else None
            return prepare_generators(
                points, geometry=geom, operation='compute', external_ids=None,
                backend_radii=None, duplicate_check='off', duplicate_threshold=1e-5,
                duplicate_wrap=True, duplicate_max_pairs=10, periodic_snapshot=snapshot)

        tasks = {
            'public_remap': (lambda: domain.remap_cart(points), 5, 5),
            'full_preparation': (prepare, 3, 1),
            'compute': (lambda: pv.compute(points, domain=domain, duplicate_check='off',
                                          return_vertices=False, return_adjacency=False,
                                          return_faces=False), 3, 1),
        }
        timings = {}
        for name, (fn, repeats, number) in tasks.items():
            fn()
            samples = [t / number for t in timeit.repeat(fn, number=number, repeat=repeats)]
            timings[name] = dict(median_seconds=statistics.median(samples), samples_seconds=samples)
        rows.append(dict(kind=kind, n=n, timings=timings))
print(json.dumps(dict(provenance=provenance(), seed='20261127+n', measurements=rows), indent=2))
