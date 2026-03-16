"""Edge-level geometric properties for planar cells."""

from __future__ import annotations

from typing import Any

import numpy as np

from .planar._domain_geometry import geometry2d
from .planar.domains import Box, RectangularCell


def annotate_edge_properties(
    cells: list[dict[str, Any]],
    domain: Box | RectangularCell,
    *,
    tol: float = 1e-12,
) -> None:
    """Annotate 2D edges with basic geometric descriptors in-place.

    Added edge fields (when computable):
      - midpoint: [x, y]
      - tangent: [tx, ty] unit tangent from vertex[0] -> vertex[1]
      - normal: [nx, ny] unit normal oriented from site -> edge
      - length: float
      - other_site: [x, y] if the neighboring site can be resolved
    """

    sites: dict[int, np.ndarray] = {}
    for cell in cells:
        pid = int(cell.get('id', -1))
        site = np.asarray(cell.get('site', []), dtype=np.float64)
        if pid >= 0 and site.size == 2:
            sites[pid] = site.reshape(2)

    geom = geometry2d(domain)
    periodic = geom.has_any_periodic_axis

    def _other_site(edge: dict[str, Any]) -> np.ndarray | None:
        nid = int(edge.get('adjacent_cell', -999999))
        if nid < 0:
            return None
        other = sites.get(nid)
        if other is None:
            return None
        if periodic and 'adjacent_shift' in edge:
            shift = np.asarray(edge.get('adjacent_shift', (0, 0)), dtype=np.int64)
            if shift.shape == (2,):
                other = other + geom.shift_vector(shift)
        return other

    eps = float(max(tol, 1e-15))
    for cell in cells:
        pid = int(cell.get('id', -1))
        site = sites.get(pid)
        if site is None:
            continue
        vertices = np.asarray(cell.get('vertices', []), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            continue

        for edge in cell.get('edges') or []:
            idx = np.asarray(edge.get('vertices', []), dtype=np.int64)
            if idx.shape != (2,):
                edge['midpoint'] = None
                edge['tangent'] = None
                edge['normal'] = None
                edge['length'] = 0.0
                edge['other_site'] = None
                continue

            v0 = vertices[idx[0]]
            v1 = vertices[idx[1]]
            dv = v1 - v0
            length = float(np.linalg.norm(dv))
            midpoint = 0.5 * (v0 + v1)
            edge['midpoint'] = midpoint.tolist()
            edge['length'] = length

            if length <= eps:
                edge['tangent'] = None
                edge['normal'] = None
            else:
                tangent = dv / length
                normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
                if float(np.dot(normal, midpoint - site)) < 0.0:
                    normal = -normal
                edge['tangent'] = tangent.tolist()
                edge['normal'] = normal.tolist()

            other = _other_site(edge)
            edge['other_site'] = other.tolist() if other is not None else None
