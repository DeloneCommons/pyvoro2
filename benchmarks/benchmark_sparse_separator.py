#!/usr/bin/env python3
"""Benchmark dense and sparse quadratic separator solves on static graphs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, TypeVar

import numpy as np

import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.static_separator_cases import molecular_locality_inputs  # noqa: E402


T = TypeVar('T')


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    n_sites: int
    neighbors: int
    components: int
    run_dense: bool


@dataclass(frozen=True)
class BenchmarkResult:
    case: str
    n_sites: int
    n_observations: int
    n_components: int
    dense_matrix_mib_estimate: float
    dense_matrix_mib_actual: float | None
    sparse_matrix_mib: float
    dense_assembly_seconds: float | None
    sparse_assembly_seconds: float
    dense_solve_seconds: float | None
    sparse_solve_seconds: float
    dense_fit_seconds: float | None
    sparse_fit_seconds: float
    max_edge_difference_disagreement: float | None
    objective_disagreement: float | None
    sparse_normal_residual_inf: float


CASES = (
    BenchmarkCase('small_knn', 64, 8, 1, True),
    BenchmarkCase('medium_knn', 384, 8, 1, True),
    BenchmarkCase('large_knn', 4096, 8, 1, False),
    BenchmarkCase('disconnected_knn', 1024, 8, 4, True),
)


def _static_problem(case: BenchmarkCase):
    inputs = molecular_locality_inputs(
        case.n_sites,
        neighbors=case.neighbors,
        n_components=case.components,
        target_perturbation=2.0e-3,
    )
    observations = inverse.resolve_separator_observations(
        inputs.points,
        inputs.observations,
        ids=inputs.ids,
        index_mode='id',
        confidence=inputs.confidence,
    )
    problem = separator.build_power_fit_problem(observations)
    return inputs.points, observations, problem


def _best_time(call: Callable[[], T], repeat: int) -> tuple[T, float]:
    best_value: T | None = None
    best_seconds = float('inf')
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        value = call()
        elapsed = perf_counter() - start
        if elapsed < best_seconds:
            best_value = value
            best_seconds = elapsed
    assert best_value is not None
    return best_value, best_seconds


def _free_indices(problem) -> np.ndarray:
    n_sites = int(problem.constraints.n_points)
    anchored = np.zeros(n_sites, dtype=bool)
    for component in problem.observation_graph.informative_components:
        anchored[int(component[0])] = True
    return np.flatnonzero(~anchored)


def _sparse_storage_bytes(matrix) -> int:
    return int(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )


def _weights_from_solution(
    problem,
    free: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(int(problem.constraints.n_points), dtype=np.float64)
    weights[free] = np.asarray(solution, dtype=np.float64)
    return problem.canonicalize_gauge(weights)


def _require_optimal_fit(fit, *, backend: str) -> np.ndarray:
    if fit.status != 'optimal' or not fit.converged or fit.weights is None:
        raise RuntimeError(
            f'{backend} public fit failed during benchmark: '
            f'status={fit.status!r}, detail={fit.status_detail!r}'
        )
    if fit.solver != backend:
        raise RuntimeError(
            f'{backend} public fit reported unexpected backend {fit.solver!r}'
        )
    return np.asarray(fit.weights, dtype=np.float64)


def run_case(case: BenchmarkCase, *, repeat: int) -> BenchmarkResult:
    from scipy.sparse.linalg import spsolve

    points, observations, problem = _static_problem(case)
    operator = problem.quadratic_operator
    free = _free_indices(problem)
    rhs = operator.regularized_normal_rhs

    sparse_matrix, sparse_assembly = _best_time(
        lambda: operator.regularized_normal_matrix_sparse(format='csc'),
        repeat,
    )
    sparse_reduced = sparse_matrix[free, :][:, free]
    sparse_solution, sparse_solve = _best_time(
        lambda: spsolve(sparse_reduced, rhs[free]),
        repeat,
    )
    sparse_weights = _weights_from_solution(problem, free, sparse_solution)
    sparse_fit, sparse_fit_seconds = _best_time(
        lambda: inverse.fit_weights_from_separators(
            points,
            observations,
            solver='sparse',
            connectivity_check='diagnose',
        ),
        repeat,
    )
    sparse_fit_weights = _require_optimal_fit(sparse_fit, backend='sparse')
    sparse_fit_disagreement = float(
        np.max(
            np.abs(
                problem.predict_difference(sparse_weights)
                - problem.predict_difference(sparse_fit_weights)
            )
        )
    )
    if sparse_fit_disagreement > 1.0e-9:
        raise RuntimeError(
            'sparse public fit disagrees with the direct benchmark solve: '
            f'{sparse_fit_disagreement:.3e}'
        )

    dense_matrix = None
    dense_assembly = None
    dense_solve = None
    dense_fit_seconds = None
    max_disagreement = None
    objective_disagreement = None
    dense_fit = None
    if case.run_dense:
        dense_matrix, dense_assembly = _best_time(
            operator.regularized_normal_matrix_dense,
            repeat,
        )
        dense_reduced = dense_matrix[np.ix_(free, free)]
        dense_solution, dense_solve = _best_time(
            lambda: np.linalg.solve(dense_reduced, rhs[free]),
            repeat,
        )
        dense_weights = _weights_from_solution(problem, free, dense_solution)
        max_disagreement = float(
            np.max(
                np.abs(
                    problem.predict_difference(dense_weights)
                    - problem.predict_difference(sparse_weights)
                )
            )
        )
        dense_fit, dense_fit_seconds = _best_time(
            lambda: inverse.fit_weights_from_separators(
                points,
                observations,
                solver='analytic',
                connectivity_check='diagnose',
            ),
            repeat,
        )
        _require_optimal_fit(dense_fit, backend='analytic')
        objective_disagreement = abs(
            float(dense_fit.objective_breakdown.total)
            - float(sparse_fit.objective_breakdown.total)
        )

    normal_residual = (
        operator.regularized_normal_matvec(sparse_fit_weights) - rhs
    )
    sparse_storage = _sparse_storage_bytes(sparse_matrix)
    dense_estimate = 8 * case.n_sites * case.n_sites
    return BenchmarkResult(
        case=case.name,
        n_sites=case.n_sites,
        n_observations=int(observations.n_constraints),
        n_components=problem.observation_graph.informative_graph.n_components,
        dense_matrix_mib_estimate=dense_estimate / (1024**2),
        dense_matrix_mib_actual=(
            None if dense_matrix is None else dense_matrix.nbytes / (1024**2)
        ),
        sparse_matrix_mib=sparse_storage / (1024**2),
        dense_assembly_seconds=dense_assembly,
        sparse_assembly_seconds=sparse_assembly,
        dense_solve_seconds=dense_solve,
        sparse_solve_seconds=sparse_solve,
        dense_fit_seconds=dense_fit_seconds,
        sparse_fit_seconds=sparse_fit_seconds,
        max_edge_difference_disagreement=max_disagreement,
        objective_disagreement=objective_disagreement,
        sparse_normal_residual_inf=float(np.max(np.abs(normal_residual))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case',
        action='append',
        choices=[case.name for case in CASES],
        help='case to run; repeat the option to select multiple cases',
    )
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error('--repeat must be positive')

    selected = (
        CASES
        if not args.case
        else tuple(case for case in CASES if case.name in args.case)
    )
    results = [asdict(run_case(case, repeat=args.repeat)) for case in selected]
    payload = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
