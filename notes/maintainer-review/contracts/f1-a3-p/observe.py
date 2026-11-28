"""Bounded R1 native readback; records baseline failures without changing oracles."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyvoro2 as pv
import pyvoro2.api as api
from pyvoro2 import _core, _core2d


root = Path(__file__).resolve().parents[4]
basis = np.array([[0., -1., -1.], [1., 0., 0.], [-1., -1., 0.]])
origin = np.array([1.25, -.5, .125])
points = origin + np.array([[.25, .5, .5], [.5, .5, .5], [.75, .5, .5]])
domain = pv.PeriodicCell(basis, origin=origin)
rows = []
for lift in (0, 7, 2**50):
    source = points.copy()
    source[1] += lift * basis[0]
    for axis in range(3):
        assert F(source[1, axis]) == F(points[1, axis]) + lift * F(basis[0, axis])
    for kind, kwargs in [
        ('standard', {}), ('weights', dict(mode='power', weights=[3/32, 0, 0])),
        ('radii', dict(mode='power', radii=[1/4, 0, 0])),
    ]:
        cells, reference = api._compute_for_matching(
            source, domain=domain, ids=[11, 13, 17], output='cells', **kwargs)
        rows.append(dict(
            lift=lift, kind=kind,
            volumes={c['id']: c['volume'] for c in cells},
            submitted_hex=[[float(v).hex() for v in row] for row in reference.submitted],
            K=reference.preparation_shifts.tolist(),
            L_hex=[[float(v).hex() for v in row] for row in reference.lattice],
            anchors_hex=[[float(v).hex() for v in row] for row in reference.anchors],
        ))
    owners = pv.locate(source, points, domain=domain, ids=[11, 13, 17],
                       return_owner_position=True)
    ghosts = []
    for ghost_lift in (0, 2**50):
        query = origin + [9/32, .5, .5] + ghost_lift * basis[0]
        cells = pv.ghost_cells(source, [query], domain=domain, ids=[11, 13, 17])
        ghosts.append(dict(lift=ghost_lift, volume=cells[0]['volume'],
                           query=cells[0]['query']))
    rows.append(dict(lift=lift, kind='query-controls',
                     owners=owners['owner_id'].tolist(),
                     owner_positions=owners['owner_pos'].tolist(), ghosts=ghosts))

print(json.dumps(dict(
    revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True).strip(),
    source=pv.__file__,
    native=[dict(path=m.__file__, sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest())
            for m in (_core, _core2d)],
    measurements=rows), indent=2))
