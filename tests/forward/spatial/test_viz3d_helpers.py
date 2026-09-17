import importlib

import numpy as np

import pyvoro2


class _DummyView:
    def __init__(self):
        self.lines = []
        self.spheres = []
        self.labels = []

    def setBackgroundColor(self, _c):
        return None

    def addLine(self, spec):
        self.lines.append(spec)

    def addSphere(self, spec):
        self.spheres.append(spec)

    def addLabel(self, text, spec):
        self.labels.append((text, spec))

    def zoomTo(self):
        return None


class _DummyPy3Dmol:
    def view(self, **_kwargs):
        return _DummyView()


def test_add_cell_wireframe_accepts_numpy_inputs(monkeypatch):
    viz = importlib.import_module('pyvoro2.viz3d')
    monkeypatch.setattr(viz, '_py3Dmol', _DummyPy3Dmol(), raising=False)

    v = _DummyView()
    cell = {
        'vertices': np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        'faces': [
            {
                # Intentionally a numpy array to exercise truthiness handling.
                'vertices': np.array([0, 1, 2], dtype=int)
            }
        ],
    }

    viz.add_cell_wireframe(v, cell)
    # A single triangle face -> 3 unique edges.
    assert len(v.lines) == 3


def test_dedup_vertices_uses_tuple_keys_and_preserves_order(monkeypatch):
    viz = importlib.import_module('pyvoro2.viz3d')
    monkeypatch.setattr(viz, '_py3Dmol', _DummyPy3Dmol(), raising=False)

    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + 1e-9, 0.0, 0.0],  # within tol of the previous vertex
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    out = viz._dedup_vertices(verts, tol=1e-6)
    assert out.shape == (3, 3)
    assert np.allclose(out[0], [0.0, 0.0, 0.0])
    assert np.allclose(out[1], [1.0, 0.0, 0.0])
    assert np.allclose(out[2], [0.0, 1.0, 0.0])


def test_periodic_visualization_wrap_delegates_to_user_lattice(monkeypatch):
    viz = importlib.import_module('pyvoro2.viz3d')
    monkeypatch.setattr(viz, '_py3Dmol', _DummyPy3Dmol(), raising=False)
    calls = []
    original = pyvoro2.PeriodicCell.wrap_cart

    def recorded(self, points, *, return_shifts=False):
        calls.append((np.array(points, copy=True), return_shifts))
        return original(self, points, return_shifts=return_shifts)

    monkeypatch.setattr(pyvoro2.PeriodicCell, 'wrap_cart', recorded)
    cell = pyvoro2.PeriodicCell(
        ((1.0, 0.0, 0.0), (0.25, 1.0, 0.0), (0.0, 0.0, -1.0))
    )
    cells = [{
        'id': 0,
        'site': [1.25, 0.25, -0.25],
        'vertices': [[1.2, 0.2, -0.2], [1.3, 0.2, -0.2], [1.2, 0.3, -0.2]],
        'faces': [{'vertices': [0, 1, 2], 'adjacent_cell': 0}],
    }]

    viz.view_tessellation(
        cells,
        domain=cell,
        wrap_cells=True,
        show_domain=False,
        show_axes=False,
        show_vertices=False,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [[1.25, 0.25, -0.25]])
    assert calls[0][1] is True
