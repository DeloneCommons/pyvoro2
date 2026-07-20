"""Shared public value objects for inverse power fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import PairBisectorConstraints


def _plain_value(value: object) -> object:
    return value.item() if hasattr(value, 'item') else value


def _readonly_array(
    value: np.ndarray | None,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.array(value, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class ConstraintGraphDiagnostics:
    """Connectivity summary for a graph induced by constraint rows."""

    n_points: int
    n_constraints: int
    n_edges: int
    isolated_points: tuple[int, ...]
    connected_components: tuple[tuple[int, ...], ...]
    fully_connected: bool

    @property
    def n_components(self) -> int:
        return int(len(self.connected_components))


@dataclass(frozen=True, slots=True)
class ConnectivityDiagnostics:
    """Structured connectivity diagnostics for the inverse-fit graph."""

    unconstrained_points: tuple[int, ...]
    candidate_graph: ConstraintGraphDiagnostics
    effective_graph: ConstraintGraphDiagnostics
    active_graph: ConstraintGraphDiagnostics | None = None
    active_effective_graph: ConstraintGraphDiagnostics | None = None
    candidate_offsets_identified_by_data: bool = False
    active_offsets_identified_by_data: bool | None = None
    offsets_identified_in_objective: bool = False
    gauge_policy: str = ''
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlgebraicEdgeDiagnostics:
    """Edge-space diagnostics matching the paper-side algebraic formulas."""

    alpha: np.ndarray
    beta: np.ndarray
    z_obs: np.ndarray
    z_fit: np.ndarray | None
    residual: np.ndarray | None
    edge_weight: np.ndarray
    weighted_l2: float | None
    weighted_rmse: float | None
    rmse: float | None
    mae: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'alpha', _readonly_array(self.alpha, dtype=np.float64))
        object.__setattr__(self, 'beta', _readonly_array(self.beta, dtype=np.float64))
        object.__setattr__(self, 'z_obs', _readonly_array(self.z_obs, dtype=np.float64))
        object.__setattr__(self, 'z_fit', _readonly_array(self.z_fit, dtype=np.float64))
        object.__setattr__(
            self,
            'residual',
            _readonly_array(self.residual, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'edge_weight',
            _readonly_array(self.edge_weight, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class PowerFitBounds:
    measurement_lower: np.ndarray | None
    measurement_upper: np.ndarray | None
    difference_lower: np.ndarray | None
    difference_upper: np.ndarray | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'measurement_lower',
            _readonly_array(self.measurement_lower, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'measurement_upper',
            _readonly_array(self.measurement_upper, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'difference_lower',
            _readonly_array(self.difference_lower, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'difference_upper',
            _readonly_array(self.difference_upper, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class PowerFitPredictions:
    difference: np.ndarray
    fraction: np.ndarray
    position: np.ndarray
    measurement: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'difference',
            _readonly_array(self.difference, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'fraction',
            _readonly_array(self.fraction, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'position',
            _readonly_array(self.position, dtype=np.float64),
        )
        object.__setattr__(
            self,
            'measurement',
            _readonly_array(self.measurement, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class PowerFitObjectiveBreakdown:
    total: float
    mismatch: float
    penalties_total: float
    penalty_terms: tuple[tuple[str, float], ...]
    regularization: float
    hard_constraints_satisfied: bool
    hard_max_violation: float


@dataclass(frozen=True, slots=True)
class HardConstraintConflictTerm:
    """One bound relation participating in an infeasibility witness."""

    constraint_index: int
    site_i: int
    site_j: int
    relation: str
    bound_value: float

    def to_record(self, *, ids: np.ndarray | None = None) -> dict[str, object]:
        site_i = int(self.site_i) if ids is None else ids[self.site_i].item()
        site_j = int(self.site_j) if ids is None else ids[self.site_j].item()
        return {
            'constraint_index': int(self.constraint_index),
            'site_i': site_i,
            'site_j': site_j,
            'relation': self.relation,
            'bound_value': float(self.bound_value),
        }


@dataclass(frozen=True, slots=True)
class HardConstraintConflict:
    """Compact witness for inconsistent hard separator restrictions."""

    component_nodes: tuple[int, ...]
    cycle_nodes: tuple[int, ...]
    terms: tuple[HardConstraintConflictTerm, ...]
    message: str

    @property
    def constraint_indices(self) -> tuple[int, ...]:
        return tuple(sorted({int(term.constraint_index) for term in self.terms}))

    def to_records(
        self,
        *,
        ids: np.ndarray | None = None,
    ) -> tuple[dict[str, object], ...]:
        return tuple(term.to_record(ids=ids) for term in self.terms)


@dataclass(frozen=True, slots=True)
class PowerWeightFitResult:
    """Result of inverse fitting of power weights."""

    status: str
    hard_feasible: bool
    weights: np.ndarray | None
    radii: np.ndarray | None
    weight_shift: float | None
    measurement: str
    target: np.ndarray
    predicted: np.ndarray | None
    predicted_fraction: np.ndarray | None
    predicted_position: np.ndarray | None
    residuals: np.ndarray | None
    rms_residual: float | None
    max_residual: float | None
    used_shifts: np.ndarray
    solver: str
    n_iter: int
    converged: bool
    conflict: HardConstraintConflict | None
    warnings: tuple[str, ...]
    status_detail: str | None = None
    connectivity: ConnectivityDiagnostics | None = None
    edge_diagnostics: AlgebraicEdgeDiagnostics | None = None
    objective_breakdown: PowerFitObjectiveBreakdown | None = None

    @property
    def is_optimal(self) -> bool:
        return self.status == 'optimal'

    @property
    def is_infeasible(self) -> bool:
        return self.status == 'infeasible_hard_constraints'

    @property
    def conflicting_constraint_indices(self) -> tuple[int, ...]:
        if self.conflict is None:
            return tuple()
        return self.conflict.constraint_indices

    def to_records(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> tuple[dict[str, object], ...]:
        if constraints.n_constraints != int(self.target.shape[0]):
            raise ValueError('constraints do not match the fit result length')
        left, right = constraints.pair_labels(use_ids=use_ids)
        from .problem import _edge_diagnostics_for_result

        edge_diag = _edge_diagnostics_for_result(self, constraints)
        rows: list[dict[str, object]] = []
        left_is_int = np.issubdtype(np.asarray(left).dtype, np.integer)
        right_is_int = np.issubdtype(np.asarray(right).dtype, np.integer)
        for k in range(constraints.n_constraints):
            site_i = int(left[k]) if left_is_int else _plain_value(left[k])
            site_j = int(right[k]) if right_is_int else _plain_value(right[k])
            rows.append(
                {
                    'constraint_index': int(k),
                    'site_i': site_i,
                    'site_j': site_j,
                    'shift': tuple(int(v) for v in constraints.shifts[k]),
                    'measurement': self.measurement,
                    'target': float(self.target[k]),
                    'predicted': (
                        None if self.predicted is None else float(self.predicted[k])
                    ),
                    'predicted_fraction': (
                        None
                        if self.predicted_fraction is None
                        else float(self.predicted_fraction[k])
                    ),
                    'predicted_position': (
                        None
                        if self.predicted_position is None
                        else float(self.predicted_position[k])
                    ),
                    'residual': (
                        None if self.residuals is None else float(self.residuals[k])
                    ),
                    'alpha': float(edge_diag.alpha[k]),
                    'beta': float(edge_diag.beta[k]),
                    'z_obs': float(edge_diag.z_obs[k]),
                    'z_fit': (
                        None if edge_diag.z_fit is None else float(edge_diag.z_fit[k])
                    ),
                    'algebraic_residual': (
                        None
                        if edge_diag.residual is None
                        else float(edge_diag.residual[k])
                    ),
                    'edge_weight': float(edge_diag.edge_weight[k]),
                }
            )
        return tuple(rows)

    def to_report(
        self,
        constraints: PairBisectorConstraints,
        *,
        use_ids: bool = False,
    ) -> dict[str, object]:
        from .report import build_fit_report

        return build_fit_report(self, constraints, use_ids=use_ids)
