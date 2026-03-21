"""Public API for inverse fitting of power weights from pairwise constraints."""

from __future__ import annotations

from .constraints import PairBisectorConstraints, resolve_pair_bisector_constraints
from .model import (
    ExponentialBoundaryPenalty,
    FitModel,
    FixedValue,
    HuberLoss,
    Interval,
    L2Regularization,
    ReciprocalBoundaryPenalty,
    SoftIntervalPenalty,
    SquaredLoss,
)
from .active import (
    ActiveSetIteration,
    ActiveSetOptions,
    ActiveSetPathSummary,
    PairConstraintDiagnostics,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)
from .problem import (
    PowerFitProblem,
    build_power_fit_problem,
    build_power_fit_result,
)
from .realize import (
    RealizedPairDiagnostics,
    UnaccountedRealizedPair,
    UnaccountedRealizedPairError,
    match_realized_pairs,
)
from .report import (
    build_active_set_report,
    build_fit_report,
    build_realized_report,
    dumps_report_json,
    write_report_json,
)
from .solver import ConnectivityDiagnosticsError, fit_power_weights
from .transforms import radii_to_weights, weights_to_radii
from .types import (
    AlgebraicEdgeDiagnostics,
    ConnectivityDiagnostics,
    ConstraintGraphDiagnostics,
    HardConstraintConflict,
    HardConstraintConflictTerm,
    PowerFitBounds,
    PowerFitObjectiveBreakdown,
    PowerFitPredictions,
    PowerWeightFitResult,
)

__all__ = [
    'PairBisectorConstraints',
    'resolve_pair_bisector_constraints',
    'SquaredLoss',
    'HuberLoss',
    'Interval',
    'FixedValue',
    'SoftIntervalPenalty',
    'ExponentialBoundaryPenalty',
    'ReciprocalBoundaryPenalty',
    'L2Regularization',
    'FitModel',
    'AlgebraicEdgeDiagnostics',
    'ConstraintGraphDiagnostics',
    'ConnectivityDiagnostics',
    'ConnectivityDiagnosticsError',
    'HardConstraintConflictTerm',
    'HardConstraintConflict',
    'PowerFitBounds',
    'PowerFitPredictions',
    'PowerFitObjectiveBreakdown',
    'PowerFitProblem',
    'PowerWeightFitResult',
    'build_power_fit_problem',
    'build_power_fit_result',
    'RealizedPairDiagnostics',
    'UnaccountedRealizedPair',
    'UnaccountedRealizedPairError',
    'build_fit_report',
    'build_realized_report',
    'build_active_set_report',
    'dumps_report_json',
    'write_report_json',
    'ActiveSetOptions',
    'ActiveSetIteration',
    'ActiveSetPathSummary',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'fit_power_weights',
    'match_realized_pairs',
    'solve_self_consistent_power_weights',
    'radii_to_weights',
    'weights_to_radii',
]
