#!/usr/bin/env python3
"""Deterministic work/timing report for certified periodic image geometry."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from time import perf_counter

import numpy as np

from pyvoro2._internal.periodic_images import (
    _basis_cache_clear,
    _basis_cache_info,
    minimum_image_displacements,
)


DEFAULT_SIZES = (1, 10, 100, 1000, 10000)


def _median_seconds(call, repeats: int):
    call()
    durations = []
    last = None
    for _ in range(repeats):
        start = perf_counter()
        last = call()
        durations.append(perf_counter() - start)
    return statistics.median(durations), last


def _legacy_triclinic_search_one(
    pi: np.ndarray,
    pj: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Former bounded search, retained here only as timing context."""

    grid = np.arange(-1, 2, dtype=np.int64)
    candidates = np.array(
        np.meshgrid(grid, grid, grid, indexing='ij')
    ).reshape(3, -1).T
    translations = candidates @ lattice
    displacement = pj - pi
    images = displacement[:, None, :] + translations[None, :, :]
    distance_squared = np.einsum('mki,mki->mk', images, images)
    return candidates[np.argmin(distance_squared, axis=1)]


def _batch_points(size: int) -> tuple[np.ndarray, np.ndarray]:
    pi_row = np.array([0.63696169, 0.26978671, 0.04097352])
    pj_row = np.array([0.01652764, 0.81327024, 0.91275558])
    pi = np.tile(pi_row, (size, 1))
    pj = np.tile(pj_row, (size, 1))
    return pi, pj


def benchmark(
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    repeats: int = 3,
) -> dict[str, object]:
    """Return timings together with deterministic certification work counts."""

    orthogonal = np.diag([1.0, 1.5, 2.0])
    triclinic = np.array(
        [[1.0, 0.0, 0.0], [1.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rows: dict[str, object] = {}
    _basis_cache_clear()
    for size in sizes:
        pi, pj = _batch_points(size)
        orientation = np.ones(size, dtype=np.int8)

        orth_seconds, orth_result = _median_seconds(
            lambda: minimum_image_displacements(
                pi,
                pj,
                lattice_vectors=orthogonal,
                periodic_axes=(True, True, True),
                tie_orientation=orientation,
                image_search=1,
            ),
            repeats,
        )
        tric_seconds, tric_result = _median_seconds(
            lambda: minimum_image_displacements(
                pi,
                pj,
                lattice_vectors=triclinic,
                periodic_axes=(True, True, True),
                tie_orientation=orientation,
                image_search=1,
            ),
            repeats,
        )
        legacy_seconds, _ = _median_seconds(
            lambda: _legacy_triclinic_search_one(pi, pj, triclinic),
            repeats,
        )
        rows[str(size)] = {
            'orthogonal_seconds': orth_seconds,
            'orthogonal_exact_candidates': int(
                np.sum(orth_result.candidate_count, dtype=np.int64)
            ),
            'triclinic_seconds': tric_seconds,
            'triclinic_exact_candidates': int(
                np.sum(tric_result.candidate_count, dtype=np.int64)
            ),
            'triclinic_seed_candidates': int(
                np.sum(tric_result.seed_count, dtype=np.int64)
            ),
            'triclinic_max_candidates_per_pair': int(
                np.max(tric_result.candidate_count, initial=0)
            ),
            'legacy_search_one_seconds_context_only': legacy_seconds,
        }
    cache = _basis_cache_info()
    return {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'platform': platform.platform(),
        'repeats': repeats,
        'sizes': sizes,
        'basis_cache': {
            'hits': cache.hits,
            'misses': cache.misses,
            'maxsize': cache.maxsize,
            'currsize': cache.currsize,
        },
        'rows': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=list(DEFAULT_SIZES),
    )
    args = parser.parse_args()
    if args.repeats <= 0 or any(size <= 0 for size in args.sizes):
        parser.error('repeats and every size must be positive')
    report = benchmark(
        sizes=tuple(args.sizes),
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
