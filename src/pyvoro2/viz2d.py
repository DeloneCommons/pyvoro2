"""Optional matplotlib-based visualization helpers for planar tessellations."""

from __future__ import annotations

from typing import Iterable


def plot_tessellation(
    cells: Iterable[dict],
    *,
    ax=None,
    annotate_ids: bool = False,
):
    """Plot planar cells using matplotlib.

    Args:
        cells: Iterable of raw 2D cell dictionaries as returned by
            ``pyvoro2.planar.compute`` or ``pyvoro2.planar.ghost_cells``.
        ax: Optional existing matplotlib axes.
        annotate_ids: If True, label cell IDs at their reported sites.

    Returns:
        ``(fig, ax)``.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for cell in cells:
        vertices = cell.get('vertices') or []
        edges = cell.get('edges') or []
        if not vertices or not edges:
            continue
        for edge in edges:
            vids = edge.get('vertices', ())
            if len(vids) != 2:
                continue
            i, j = int(vids[0]), int(vids[1])
            if i < 0 or j < 0 or i >= len(vertices) or j >= len(vertices):
                continue
            vi = vertices[i]
            vj = vertices[j]
            ax.plot([vi[0], vj[0]], [vi[1], vj[1]])

        if annotate_ids:
            site = cell.get('site')
            if site is not None:
                ax.text(float(site[0]), float(site[1]), str(cell.get('id', '?')))

    ax.set_aspect('equal', adjustable='box')
    return fig, ax
