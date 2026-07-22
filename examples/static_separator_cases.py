"""Deterministic static locality-graph inputs shared by examples and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StaticSeparatorInputs:
    """Inputs for one static quadratic separator fit.

    Observation rows use the external IDs in :attr:`ids`.  The reference
    weights define the compatible part of each target; an optional deterministic
    perturbation can make the benchmark problem mildly inconsistent.
    """

    points: np.ndarray
    ids: np.ndarray
    observations: tuple[tuple[int, int, float], ...]
    confidence: np.ndarray
    reference_weights: np.ndarray
    components: tuple[tuple[int, ...], ...]


def molecular_locality_inputs(
    n_sites: int,
    *,
    neighbors: int = 4,
    n_components: int = 1,
    target_perturbation: float = 0.0,
) -> StaticSeparatorInputs:
    """Generate a deterministic wavy-chain k-nearest-neighbor problem.

    The geometry is chemistry-neutral.  It resembles a local, branched or
    polymer-like scientific graph without assigning atom types, physical
    radii, or chemical meaning.  SciPy is imported only when this generator is
    called, matching the optional sparse-workflow boundary.
    """

    n_sites = int(n_sites)
    neighbors = int(neighbors)
    n_components = int(n_components)
    perturbation = float(target_perturbation)
    if n_sites <= 0:
        raise ValueError('n_sites must be positive')
    if neighbors <= 0:
        raise ValueError('neighbors must be positive')
    if n_components <= 0 or n_components > n_sites:
        raise ValueError('n_components must lie in [1, n_sites]')
    if not np.isfinite(perturbation):
        raise ValueError('target_perturbation must be finite')

    points, components = _wavy_chain_points(n_sites, n_components)
    pairs = _knn_pairs(points, neighbors, components)
    ids = 10_000 + 7 * np.arange(n_sites, dtype=np.int64)
    reference_weights = 0.01 * np.sin(
        0.07 * np.arange(n_sites, dtype=np.float64)
    )
    rows: list[tuple[int, int, float]] = []
    for row_index, (i, j) in enumerate(pairs):
        delta = points[j] - points[i]
        distance2 = float(np.dot(delta, delta))
        fraction = 0.5 + (
            reference_weights[i] - reference_weights[j]
        ) / (2.0 * distance2)
        fraction += perturbation * np.sin(0.13 * row_index)
        rows.append((int(ids[i]), int(ids[j]), float(fraction)))

    return StaticSeparatorInputs(
        points=points,
        ids=ids,
        observations=tuple(rows),
        confidence=np.ones(len(rows), dtype=np.float64),
        reference_weights=reference_weights,
        components=components,
    )


def _wavy_chain_points(
    n_sites: int,
    n_components: int,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    counts = np.full(n_components, n_sites // n_components, dtype=np.int64)
    counts[: n_sites % n_components] += 1
    blocks: list[np.ndarray] = []
    components: list[tuple[int, ...]] = []
    start = 0
    for component, count in enumerate(counts.tolist()):
        stop = start + count
        t = np.arange(count, dtype=np.float64)
        block = np.column_stack(
            (
                0.20 * t,
                np.sin(0.31 * t),
                np.cos(0.17 * t),
            )
        )
        block[:, 1] += 100.0 * component
        blocks.append(block)
        components.append(tuple(range(start, stop)))
        start = stop
    return np.vstack(blocks), tuple(components)


def _knn_pairs(
    points: np.ndarray,
    neighbors: int,
    components: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], ...]:
    from scipy.spatial import cKDTree

    pairs: set[tuple[int, int]] = set()
    for component in components:
        indices = np.asarray(component, dtype=np.int64)
        block = points[indices]
        k = min(neighbors + 1, len(component))
        _, local_neighbors = cKDTree(block).query(block, k=k)
        local_neighbors = np.asarray(local_neighbors, dtype=np.int64)
        if local_neighbors.ndim == 1:
            local_neighbors = local_neighbors[:, None]
        for local_i, row in enumerate(local_neighbors):
            for local_j in row.tolist():
                if local_i == local_j:
                    continue
                i = int(indices[local_i])
                j = int(indices[int(local_j)])
                pairs.add((min(i, j), max(i, j)))
    return tuple(sorted(pairs))
