#!/usr/bin/env python3
"""Canonical performance gate for the positive-penalty scalar proximal path."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import platform
import statistics
from time import perf_counter

import numpy as np

from pyvoro2.inverse import separator
from pyvoro2.inverse.separator import solver as separator_solver
from pyvoro2.inverse.separator._scalar_prox import (
    _compile_scalar_prox_spec,
    _ScalarProxError,
    _solve_scalar_prox_coordinate,
)


SIZES = (1, 10, 100, 1000)


@dataclass(frozen=True, slots=True)
class _BenchmarkCase:
    run: Callable[[], np.ndarray]
    unique_coordinate_keys: int
    measure_branches: Callable[[np.ndarray], tuple[str, ...]]


@dataclass(slots=True)
class _RoutingCounters:
    batch_invocations: int = 0
    batch_eligible_rows: int = 0
    batch_certified_rows: int = 0
    batch_arithmetic_exceptions: int = 0
    scalar_dispatch_calls: int = 0
    high_precision_fallback_decisions: int = 0
    structured_failures: int = 0


def _coordinate_key_count(
    target: np.ndarray,
    confidence: np.ndarray,
    v: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> int:
    return len({
        (
            float(target[index]),
            float(confidence[index]),
            float(v[index]),
            float(lower[index]),
            float(upper[index]),
        )
        for index in range(target.size)
    })


def _timings(
    *,
    sizes: tuple[int, ...],
    repeats: int,
    make_case: Callable[[int], _BenchmarkCase],
) -> tuple[dict[int, float], dict[int, dict[str, object]]]:
    timings: dict[int, float] = {}
    routing: dict[int, dict[str, object]] = {}
    original_scalar = separator_solver._solve_scalar_prox_coordinate
    original_batch = separator_solver._solve_scalar_prox_batch_ordinary
    for n_rows in sizes:
        case = make_case(n_rows)
        counters = _RoutingCounters()

        def counted_scalar(*args: object, **kwargs: object) -> object:
            counters.scalar_dispatch_calls += 1
            try:
                result = original_scalar(*args, **kwargs)
            except _ScalarProxError as exc:
                counters.structured_failures += 1
                counters.high_precision_fallback_decisions += (
                    exc.failure.fallback_count
                )
                raise
            counters.high_precision_fallback_decisions += (
                result.high_precision_fallback_count
            )
            return result

        def counted_batch(*args: object, **kwargs: object) -> object:
            counters.batch_invocations += 1
            outcome = original_batch(*args, **kwargs)
            counters.batch_eligible_rows += int(np.count_nonzero(
                outcome.eligible
            ))
            counters.batch_certified_rows += int(np.count_nonzero(
                outcome.certified
            ))
            counters.batch_arithmetic_exceptions += int(
                outcome.arithmetic_exception
            )
            return outcome

        separator_solver._solve_scalar_prox_coordinate = counted_scalar
        separator_solver._solve_scalar_prox_batch_ordinary = counted_batch
        try:
            case.run()
            durations = []
            snapshots = []
            for _ in range(repeats):
                counters = _RoutingCounters()
                start = perf_counter()
                values = case.run()
                durations.append(perf_counter() - start)
                snapshots.append({
                    'actual_unique_coordinate_keys': (
                        case.unique_coordinate_keys
                    ),
                    'active_branches': case.measure_branches(values),
                    'batch_invocations': counters.batch_invocations,
                    'batch_eligible_rows': counters.batch_eligible_rows,
                    'batch_certified_rows': counters.batch_certified_rows,
                    'batch_arithmetic_exceptions': (
                        counters.batch_arithmetic_exceptions
                    ),
                    'scalar_dispatch_calls': counters.scalar_dispatch_calls,
                    'high_precision_fallback_decisions': (
                        counters.high_precision_fallback_decisions
                    ),
                    'structured_failures': counters.structured_failures,
                })
            if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
                raise AssertionError('benchmark routing changed across repeats')
            timings[n_rows] = statistics.median(durations)
            routing[n_rows] = snapshots[0]
        finally:
            separator_solver._solve_scalar_prox_coordinate = original_scalar
            separator_solver._solve_scalar_prox_batch_ordinary = original_batch
    return timings, routing


def benchmark(*, repeats: int = 5) -> dict[str, object]:
    """Report cache and heterogeneous certified-proximal performance."""

    identical_model = separator.FitModel(
        penalties=(separator.ExponentialBoundaryPenalty(),)
    )
    identical_spec = _compile_scalar_prox_spec(identical_model)
    coordinate = _solve_scalar_prox_coordinate(
        spec=identical_spec,
        target=0.0,
        confidence=1.0,
        v=0.0,
        rho=1.0,
    )

    def identical_case(n_rows: int) -> _BenchmarkCase:
        zeros = np.zeros(n_rows, dtype=np.float64)
        ones = np.ones(n_rows, dtype=np.float64)
        lower = np.full(n_rows, -np.inf, dtype=np.float64)
        upper = np.full(n_rows, np.inf, dtype=np.float64)

        def run() -> np.ndarray:
            return separator_solver._prox_measurement_objective(
                zeros,
                zeros,
                ones,
                model=identical_model,
                rho=1.0,
                y_lo=None,
                y_hi=None,
                spec=identical_spec,
            )

        return _BenchmarkCase(
            run=run,
            unique_coordinate_keys=_coordinate_key_count(
                zeros,
                ones,
                zeros,
                lower,
                upper,
            ),
            measure_branches=(
                lambda values: ('exponential',) if values.size else ()
            ),
        )

    heterogeneous_model = separator.FitModel(
        mismatch=separator.HuberLoss(delta=0.3),
        penalties=(
            separator.SoftIntervalPenalty(-0.2, 0.7, 0.2),
            separator.ExponentialBoundaryPenalty(
                lower=-1.0,
                upper=1.2,
                margin=0.1,
                strength=0.03,
                tau=0.2,
            ),
        ),
    )
    heterogeneous_spec = _compile_scalar_prox_spec(heterogeneous_model)

    def heterogeneous_case(n_rows: int) -> _BenchmarkCase:
        target = np.linspace(-0.15, 0.25, n_rows, dtype=np.float64)
        confidence = np.linspace(0.7, 1.3, n_rows, dtype=np.float64)
        fraction = np.linspace(0.0, 1.0, n_rows, dtype=np.float64)
        v = np.where(
            fraction < 0.3,
            -0.65 + 0.15 * fraction / 0.3,
            0.1 + 0.4 * (fraction - 0.3) / 0.7,
        )
        lower = np.linspace(-1.1, -0.9, n_rows, dtype=np.float64)
        upper = np.linspace(0.9, 1.1, n_rows, dtype=np.float64)

        def run() -> np.ndarray:
            return separator_solver._prox_measurement_objective(
                v,
                target,
                confidence,
                model=heterogeneous_model,
                rho=1.0,
                y_lo=lower,
                y_hi=upper,
                spec=heterogeneous_spec,
            )

        def measure_branches(values: np.ndarray) -> tuple[str, ...]:
            branches = {'exponential'} if values.size else set()
            if np.any(values < -0.2):
                branches.add('soft_lower')
            if np.any((values >= -0.2) & (values <= 0.7)):
                branches.add('soft_inactive')
            if np.any(values > 0.7):
                branches.add('soft_upper')
            residual = values - target
            if np.any(residual < -0.3):
                branches.add('huber_lower')
            if np.any((residual >= -0.3) & (residual <= 0.3)):
                branches.add('huber_quadratic')
            if np.any(residual > 0.3):
                branches.add('huber_upper')
            return tuple(sorted(branches))

        return _BenchmarkCase(
            run=run,
            unique_coordinate_keys=_coordinate_key_count(
                target,
                confidence,
                v,
                lower,
                upper,
            ),
            measure_branches=measure_branches,
        )

    identical, identical_routing = _timings(
        sizes=SIZES,
        repeats=repeats,
        make_case=identical_case,
    )
    heterogeneous, heterogeneous_routing = _timings(
        sizes=SIZES,
        repeats=repeats,
        make_case=heterogeneous_case,
    )
    passed = identical[1000] <= 1.0
    passed &= identical[1000] <= 12.0 * identical[100] + 0.10
    passed &= heterogeneous[100] <= 0.35
    passed &= heterogeneous[1000] <= 3.0
    passed &= heterogeneous[1000] <= 12.0 * heterogeneous[100] + 0.30
    for n_rows in SIZES:
        identical_route = identical_routing[n_rows]
        passed &= identical_route['actual_unique_coordinate_keys'] == 1
        passed &= identical_route['batch_invocations'] == 0
        passed &= identical_route['scalar_dispatch_calls'] == 1
        passed &= (
            identical_route['high_precision_fallback_decisions'] == 0
        )
        passed &= identical_route['structured_failures'] == 0
    for n_rows in SIZES:
        heterogeneous_route = heterogeneous_routing[n_rows]
        passed &= (
            heterogeneous_route['actual_unique_coordinate_keys'] == n_rows
        )
        passed &= (
            heterogeneous_route['high_precision_fallback_decisions'] == 0
        )
        passed &= heterogeneous_route['structured_failures'] == 0
        passed &= heterogeneous_route['batch_arithmetic_exceptions'] == 0
        if n_rows == 1:
            passed &= heterogeneous_route['batch_invocations'] == 0
            passed &= heterogeneous_route['scalar_dispatch_calls'] == 1
        else:
            passed &= heterogeneous_route['batch_invocations'] == 1
            passed &= heterogeneous_route['batch_eligible_rows'] == n_rows
            passed &= heterogeneous_route['batch_certified_rows'] == n_rows
            passed &= heterogeneous_route['scalar_dispatch_calls'] == 0
            passed &= {
                'soft_lower',
                'soft_inactive',
                'exponential',
            }.issubset(heterogeneous_route['active_branches'])
    passed &= coordinate.fallback_count == 0
    return {
        'environment': {
            'cpu': platform.processor() or platform.machine(),
            'python': platform.python_version(),
            'numpy': np.__version__,
        },
        'identical': {
            'timings_seconds': {
                str(key): value for key, value in identical.items()
            },
            'routing': {
                str(key): value for key, value in identical_routing.items()
            },
            'cache_contract': (
                'one scalar dispatch for one repeated coordinate key'
            ),
            'limit_1000_seconds': 1.0,
            'scaling_limit_seconds': 12.0 * identical[100] + 0.10,
        },
        'heterogeneous': {
            'timings_seconds': {
                str(key): value for key, value in heterogeneous.items()
            },
            'routing': {
                str(key): value
                for key, value in heterogeneous_routing.items()
            },
            'limit_100_seconds': 0.35,
            'limit_1000_seconds': 3.0,
            'scaling_limit_seconds': 12.0 * heterogeneous[100] + 0.30,
            'routing_contract': (
                '10/100/1000 unique rows are batch eligible and certified; '
                'no scalar or high-precision fallback dispatch'
            ),
        },
        'canonical_coordinate_iterations': coordinate.scalar_iterations,
        'canonical_coordinate_fallback_count': coordinate.fallback_count,
        'passed': passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('--repeats must be positive')
    result = benchmark(repeats=args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
