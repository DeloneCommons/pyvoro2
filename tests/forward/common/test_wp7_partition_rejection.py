"""Independent queries must never become an ordinary tessellation partition."""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('operation', ['vertices', 'topology', 'validation',
                                       'diagnostics'])
def test_partition_helpers_explicitly_reject_independent_ghosts(dim, operation):
    prefix = 'pyvoro2.planar' if dim == 2 else 'pyvoro2'
    package = importlib.import_module(prefix)
    normalize = importlib.import_module(prefix + '.normalize')
    validation = importlib.import_module(prefix + '.validation')
    diagnostics = importlib.import_module(prefix + '.diagnostics')
    domain = package.Box(((0., 1.),) * dim)
    cells = [dict(id=-1, query_index=0, site=[.5] * dim, empty=True,
                  vertices=[], adjacency=[], **{
                      'edges' if dim == 2 else 'faces': [],
                      'area' if dim == 2 else 'volume': 0.,
                  })]
    with pytest.raises(ValueError, match='[Gg]host'):
        if operation == 'vertices':
            normalize.normalize_vertices(cells, domain=domain)
        elif operation == 'topology':
            normalize.normalize_topology(cells, domain=domain)
        elif operation == 'validation':
            normalized = normalize.NormalizedVertices(np.empty((0, dim)), cells)
            validation.validate_normalized_topology(normalized, domain=domain)
        else:
            diagnostics.analyze_tessellation(cells, domain)
