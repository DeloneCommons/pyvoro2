from __future__ import annotations

import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec('pyvoro2._core2d') is None:
    pytest.skip('pyvoro2._core2d is not available', allow_module_level=True)

import pyvoro2.planar as pv2


def _periodic_cells() -> tuple[list[dict], pv2.RectangularCell]:
    pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    cells = pv2.compute(
        pts,
        domain=domain,
        return_vertices=True,
        return_edges=True,
        return_edge_shifts=True,
    )
    return cells, domain


def test_planar_normalize_vertices_box() -> None:
    pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    box = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
    cells = pv2.compute(pts, domain=box, return_vertices=True, return_edges=True)

    nv = pv2.normalize_vertices(cells, domain=box)

    assert nv.global_vertices.shape == (6, 2)
    assert {int(cell['id']) for cell in nv.cells} == {0, 1}
    for cell in nv.cells:
        assert len(cell['vertex_global_id']) == len(cell['vertices'])
        assert len(cell['vertex_shift']) == len(cell['vertices'])
        assert all(tuple(shift) == (0, 0) for shift in cell['vertex_shift'])


def test_planar_normalize_topology_periodic_ok() -> None:
    cells, domain = _periodic_cells()

    topo = pv2.normalize_topology(cells, domain=domain)
    diag = pv2.validate_normalized_topology(topo, domain, level='strict')

    assert topo.global_vertices.shape == (6, 2)
    assert len(topo.global_edges) == 9
    assert diag.ok is True
    assert diag.has_wall_edges is False
    assert diag.fully_periodic_domain is True


def test_planar_normalize_vertices_requires_edge_shifts_in_periodic_domains() -> None:
    pts = np.array([[0.2, 0.2], [0.8, 0.25], [0.4, 0.8]], dtype=float)
    domain = pv2.RectangularCell(((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    cells = pv2.compute(pts, domain=domain, return_vertices=True, return_edges=True)

    with pytest.raises(ValueError, match='return_edge_shifts=True'):
        pv2.normalize_vertices(cells, domain=domain)


def test_planar_validate_normalized_topology_strict_raises_on_tampering() -> None:
    cells, domain = _periodic_cells()
    topo = pv2.normalize_topology(cells, domain=domain)

    topo.cells[1]['vertex_shift'][2] = (0, 0)

    with pytest.raises(pv2.NormalizationError):
        pv2.validate_normalized_topology(topo, domain, level='strict')
