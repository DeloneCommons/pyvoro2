"""Regenerate the Home inverse figure with pyvoro2[viz2d] and Matplotlib.

Run from the repository root: python tools/landing/render_separator_cycle.py
The ticks in both panels are input observations. Only the solid interior
edges on the right are computed boundaries.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyvoro2.planar as planar
from pyvoro2.inverse import (
    fit_weights_from_separators,
    resolve_separator_observations,
)


def main():
    points = np.array([[0.22, 0.24], [0.80, 0.30], [0.45, 0.79]])
    pairs = [(0, 1, 0.40), (1, 2, 0.58), (2, 0, 0.72)]
    domain = planar.Box(((0.0, 1.0), (0.0, 1.0)))
    observations = resolve_separator_observations(
        points, pairs, measurement='fraction', domain=domain,
    )
    fit = fit_weights_from_separators(points, observations)
    assert fit.status == 'optimal'
    targets = np.array([fraction for _, _, fraction in pairs])
    assert np.all(np.abs(fit.predicted_fraction - targets) > 0.05)

    result = planar.compute(
        points, domain=domain, mode='power', weights=fit.weights,
        return_vertices=True, return_edges=True,
    )
    np.testing.assert_allclose(result.cell_measures.sum(), 1.0)
    fig, (observed, fitted) = plt.subplots(1, 2, figsize=(4.1, 2.22))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.82, wspace=0.15)
    panels = [(observed, 'Observed positions'), (fitted, 'Computed cells')]
    for axis, title in panels:
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xticks=[], yticks=[])
        axis.set_title(title, fontsize=11, color='#283e40')
        for spine in axis.spines.values():
            spine.set_color('#758788')
        axis.scatter(*points.T, s=18, color='#182f35', zorder=4)

    for i, j, fraction in pairs:
        observed.plot(*points[[i, j]].T, '--', color='#758788', linewidth=1)
        direction = points[j] - points[i]
        position = points[i] + fraction * direction
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        tick = np.array([position - 0.045 * normal, position + 0.045 * normal])
        for axis in (observed, fitted):
            axis.plot(*tick.T, color='#a34e25', linewidth=2, zorder=3)

    colors = ('#e1ecec', '#e5ede5', '#eeece5')
    contacts = set()
    for cell in result.cells:
        vertices = np.asarray(cell['vertices'])
        # Planar cells are convex; order returned vertices around their centroid.
        center = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        polygon = vertices[np.argsort(angles)]
        x, y = polygon.T
        area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
        np.testing.assert_allclose(area, cell['area'], rtol=1e-12, atol=1e-12)
        fitted.fill(*polygon.T, color=colors[cell['id']], linewidth=0, zorder=0)
        for edge in cell['edges']:
            # Draw each interior edge once. Negative neighbors are box walls.
            i, j = cell['id'], edge['adjacent_cell']
            if j > i:
                segment = vertices[edge['vertices']]
                fitted.plot(*segment.T, color='#235e64', linewidth=1.6)
                contacts.add((i, j))
    assert contacts == {(0, 1), (1, 2), (0, 2)}

    root = Path(__file__).resolve().parents[2]
    output = root / 'docs/assets/landing/separator-cycle.png'
    fig.savefig(output, dpi=200, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
