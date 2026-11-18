"""Regenerate the Home inverse figure with pyvoro2[viz2d] and Matplotlib.

Run from the repository root: python tools/landing/render_separator_cycle.py
The ticks in both panels are input observations. Only the solid interior
edges on the right are computed boundaries.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvoro2
import pyvoro2.planar as planar
from pyvoro2.inverse import (
    fit_weights_from_separators,
    resolve_separator_observations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--theme', choices=('light', 'dark'), default='light')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    light = root / 'docs/assets/landing/separator-cycle.png'
    dark_asset = root / 'docs/assets/landing/separator-cycle-dark.png'
    dark = args.theme == 'dark'
    output = args.output or (dark_asset if dark else light)
    if output.resolve() == (light if dark else dark_asset):
        parser.error('Output would overwrite the other theme asset')
    if not __debug__:
        parser.error('Run with assertions enabled to check the scientific geometry')
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
        axis.set_title(title, fontsize=11, color='#e0e3df' if dark else '#283e40')
        for spine in axis.spines.values():
            spine.set_color('#90a4a4' if dark else '#758788')
        axis.scatter(*points.T, s=18, color='#e0e3df' if dark else '#182f35', zorder=4)

    for i, j, fraction in pairs:
        observed.plot(*points[[i, j]].T, '--',
                      color='#90a4a4' if dark else '#758788', linewidth=1)
        direction = points[j] - points[i]
        position = points[i] + fraction * direction
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        tick = np.array([position - 0.045 * normal, position + 0.045 * normal])
        for axis in (observed, fitted):
            axis.plot(*tick.T, color='#e9ad81' if dark else '#a34e25',
                      linewidth=2, zorder=3)

    colors = (('#2c4850', '#3d514a', '#535044') if dark
              else ('#e1ecec', '#e5ede5', '#eeece5'))
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
                fitted.plot(*segment.T, color='#acd0cb' if dark else '#235e64',
                            linewidth=1.8 if dark else 1.6)
                contacts.add((i, j))
    assert contacts == {(0, 1), (1, 2), (0, 2)}

    report = dict(
        theme='carbon' if dark else 'light', points=points.tolist(), pairs=pairs,
        weights=fit.weights.tolist(),
        predicted_fraction=fit.predicted_fraction.tolist(),
        status=fit.status, areas=result.cell_measures.tolist(),
        contacts=sorted(contacts),
        geometry_sha256=hashlib.sha256(json.dumps(
            result.cells, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest(),
        colors=dict(cells=colors, label='#e0e3df' if dark else '#283e40',
                    site='#e0e3df' if dark else '#182f35',
                    frame='#90a4a4' if dark else '#758788',
                    observed='#e9ad81' if dark else '#a34e25',
                    boundary='#acd0cb' if dark else '#235e64'),
        linewidth=dict(connector=1, observed=2, boundary=1.8 if dark else 1.6),
        transparent=True, dpi=200, figure_inches=(4.1, 2.22),
        versions=dict(python=platform.python_version(), pyvoro2=pyvoro2.__version__,
                      numpy=np.__version__,
                      matplotlib=matplotlib.__version__),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, transparent=True)
        report['png_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
