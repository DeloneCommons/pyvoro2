"""Separator-specific inverse fitting, diagnostics, and advanced workflows.

The fixed-observation solver is exact for its selected convex model. The
realization-aware active-set workflow is a separate experimental outer
algorithm and does not carry a universal convergence guarantee.
"""

from __future__ import annotations

from ..._weight_transforms import radii_to_weights, weights_to_radii
from .constraints import (
    PairBisectorConstraints,
    SeparatorObservations,
    resolve_pair_bisector_constraints,
    resolve_separator_observations,
)
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
    ActiveSetPathView,
    ActiveSetIteration,
    ActiveSetOptions,
    ActiveSetPathSummary,
    ActiveSetTerminationView,
    PairConstraintDiagnostics,
    SelfConsistentPowerFitResult,
    solve_self_consistent_power_weights,
)
from .problem import (
    PowerFitProblem,
    SeparatorFitProblem,
    build_power_fit_problem,
    build_power_fit_result,
)
from .realize import (
    RealizedGeometryView,
    RealizedPairDiagnostics,
    RequestedImageMatchView,
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
from .solver import (
    ConnectivityDiagnosticsError,
    fit_power_weights,
    fit_weights_from_separators,
)
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
    SeparatorAlgebraicView,
    SeparatorFitResult,
    SeparatorFitStateView,
    SeparatorIdentificationView,
    SeparatorObservationView,
    SeparatorSolverTerminationView,
)

__all__ = [
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitProblem',
    'SeparatorFitResult',
    'fit_weights_from_separators',
    'SeparatorFitStateView',
    'SeparatorIdentificationView',
    'SeparatorObservationView',
    'SeparatorAlgebraicView',
    'SeparatorSolverTerminationView',
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
    'RequestedImageMatchView',
    'RealizedGeometryView',
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
    'ActiveSetTerminationView',
    'ActiveSetPathView',
    'PairConstraintDiagnostics',
    'SelfConsistentPowerFitResult',
    'fit_power_weights',
    'match_realized_pairs',
    'solve_self_consistent_power_weights',
    'radii_to_weights',
    'weights_to_radii',
]
