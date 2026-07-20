"""Shared public value objects for inverse power fitting."""

from __future__ import annotations

from dataclasses import InitVar, KW_ONLY, dataclass, fields

import numpy as np

from .constraints import SeparatorObservations


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
    """Structured connectivity diagnostics for the inverse-fit graph.

    ``effective_graph`` and ``active_effective_graph`` are the informative
    observation graphs induced only by positive-confidence rows.  The
    historical ``unconstrained_points`` field remains candidate-graph based.
    ``offsets_identified_in_objective`` conservatively reports complete
    relative-offset determination by informative data or positive L2
    regularization; arbitrary penalties and hard restrictions do not set it.
    """

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
class SeparatorFitStateView:
    """Fitted mathematical state and its backend representation.

    ``global_representation_shift`` is the one common additive shift used to
    convert ``mathematical_weights`` to non-negative ``backend_radii``.  It
    selects a backend representation within the global geometric gauge; it is
    distinct from component offsets and is not recovered from the separator
    observations.
    """

    mathematical_weights: np.ndarray | None
    backend_radii: np.ndarray | None
    global_representation_shift: float | None


@dataclass(frozen=True, slots=True)
class SeparatorIdentificationView:
    """Identification and component-alignment meaning of a separator fit.

    The informative observation graph contains positive-confidence separator
    rows only.  ``relative_component_offsets_identified_by_data`` is true
    exactly when that graph is connected.  Hard restrictions and scalar
    penalties may constrain offsets but are not separator-observation data.

    ``component_offsets_selected_by_objective`` is conservative: it is true
    only when otherwise free informative-component offsets are guaranteed to
    be selected by a supported additional objective, currently positive L2
    regularization.  Exact hard equalities are not summarized by this view.
    """

    global_geometric_gauge_identified_by_data: bool
    effective_observation_components: tuple[tuple[int, ...], ...] | None
    relative_component_offsets_identified_by_data: bool | None
    component_offsets_selected_by_objective: bool | None
    component_alignment_policy: str | None
    unconstrained_sites: tuple[int, ...] | None
    connectivity: ConnectivityDiagnostics | None


@dataclass(frozen=True, slots=True)
class SeparatorObservationView:
    """Observation-space targets, predictions, residuals, and confidence.

    Arrays are references to the fit result or supplied resolved observations;
    constructing the view does not copy them.
    """

    measurement: str
    targets: np.ndarray
    target_fraction: np.ndarray
    target_position: np.ndarray
    confidence: np.ndarray
    predictions: np.ndarray | None
    predicted_fraction: np.ndarray | None
    predicted_position: np.ndarray | None
    residuals: np.ndarray | None
    rms_residual: float | None
    max_residual: float | None
    requested_shifts: np.ndarray


@dataclass(frozen=True, slots=True)
class SeparatorAlgebraicView:
    """Existing difference-space and connectivity diagnostics.

    Public incidence, Laplacian, and normal-operator representations are
    deliberately not part of this view.
    """

    edge_diagnostics: AlgebraicEdgeDiagnostics | None
    connectivity: ConnectivityDiagnostics | None


@dataclass(frozen=True, slots=True)
class SeparatorSolverTerminationView:
    """Termination state of the fixed-observation separator solver."""

    status: str
    status_detail: str | None
    backend: str
    n_iter: int
    converged: bool
    hard_feasible: bool
    conflict: HardConstraintConflict | None
    warnings: tuple[str, ...]


class _ObservationBoundResult:
    """Private storage that does not alter the public dataclass field surface."""

    __slots__ = ('_originating_observations',)


class _ObservationBindingInit:
    """Expose a private slot to ``dataclasses.replace`` as an init-only value."""

    def __get__(
        self,
        instance: object | None,
        owner: type[object] | None = None,
    ) -> SeparatorObservations | None:
        if instance is None:
            return None
        return getattr(instance, '_originating_observations', None)


_OBSERVATION_ARRAY_FIELDS = (
    'i',
    'j',
    'shifts',
    'target',
    'confidence',
    'distance',
    'distance2',
    'delta',
    'target_fraction',
    'target_position',
    'input_index',
    'explicit_shift',
    'ids',
)


def _observation_mismatch(
    originating: SeparatorObservations,
    supplied: SeparatorObservations,
) -> str | None:
    """Return the first field that differs between resolved observation sets."""

    if originating is supplied:
        return None
    for name in ('n_points', 'measurement', 'warnings'):
        if getattr(originating, name) != getattr(supplied, name):
            return name
    for name in _OBSERVATION_ARRAY_FIELDS:
        expected = getattr(originating, name)
        actual = getattr(supplied, name)
        if expected is None or actual is None:
            if expected is not actual:
                return name
        elif not np.array_equal(expected, actual):
            return name
    return None


def _bind_originating_observations(
    result: SeparatorFitResult,
    observations: SeparatorObservations,
) -> SeparatorFitResult:
    """Associate a result with its resolved inputs without copying their arrays."""

    object.__setattr__(result, '_originating_observations', observations)
    return result


@dataclass(frozen=True, slots=True)
class SeparatorFitResult(_ObservationBoundResult):
    """Result of fitting power weights from separator observations."""

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
    _: KW_ONLY
    _originating_observations_init: InitVar[
        SeparatorObservations | None
    ] = _ObservationBindingInit()

    def __post_init__(
        self,
        _originating_observations_init: SeparatorObservations | None,
    ) -> None:
        if _originating_observations_init is not None:
            object.__setattr__(
                self,
                '_originating_observations',
                _originating_observations_init,
            )

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

    @property
    def state(self) -> SeparatorFitStateView:
        """Return fitted weights and their backend-radius representation."""

        return SeparatorFitStateView(
            mathematical_weights=self.weights,
            backend_radii=self.radii,
            global_representation_shift=self.weight_shift,
        )

    @property
    def identification(self) -> SeparatorIdentificationView:
        """Return data-identification and component-alignment metadata."""

        connectivity = self.connectivity
        if connectivity is None:
            return SeparatorIdentificationView(
                global_geometric_gauge_identified_by_data=False,
                effective_observation_components=None,
                relative_component_offsets_identified_by_data=None,
                component_offsets_selected_by_objective=None,
                component_alignment_policy=None,
                unconstrained_sites=None,
                connectivity=None,
            )

        effective = connectivity.effective_graph
        has_relative_offsets = effective.n_components > 1
        return SeparatorIdentificationView(
            global_geometric_gauge_identified_by_data=False,
            effective_observation_components=effective.connected_components,
            relative_component_offsets_identified_by_data=bool(
                connectivity.candidate_offsets_identified_by_data
            ),
            component_offsets_selected_by_objective=bool(
                has_relative_offsets
                and connectivity.offsets_identified_in_objective
            ),
            component_alignment_policy=connectivity.gauge_policy,
            unconstrained_sites=effective.isolated_points,
            connectivity=connectivity,
        )

    def observation_view(
        self,
        observations: SeparatorObservations,
    ) -> SeparatorObservationView:
        """Return diagnostics for the observations that produced this result.

        Solver and problem-builder results retain a private reference to their
        resolved source observations.  The supplied object may be that source
        or an independently resolved, fully equivalent observation set.  The
        returned view shares arrays and does not copy them.
        """

        originating = getattr(self, '_originating_observations', None)
        if originating is None:
            raise ValueError(
                'fit result is not bound to originating observations; use a '
                'solver- or problem-builder-produced result'
            )
        mismatch = _observation_mismatch(originating, observations)
        if mismatch is not None:
            raise ValueError(
                'observations do not match the fit result originating '
                f'observations ({mismatch})'
            )

        if self.measurement != originating.measurement:
            raise ValueError(
                'fit result measurement does not match its originating '
                'observations'
            )
        expected_target = (
            originating.target_fraction
            if self.measurement == 'fraction'
            else originating.target_position
        )
        if not np.array_equal(self.target, expected_target):
            raise ValueError(
                'fit result target does not match its originating observations'
            )
        if not np.array_equal(self.used_shifts, originating.shifts):
            raise ValueError(
                'fit result requested shifts do not match its originating '
                'observations'
            )

        n_observations = int(observations.n_constraints)

        for name in (
            'predicted',
            'predicted_fraction',
            'predicted_position',
            'residuals',
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (n_observations,):
                raise ValueError(
                    f'fit result {name} must have shape (n_observations,)'
                )

        return SeparatorObservationView(
            measurement=self.measurement,
            targets=self.target,
            target_fraction=observations.target_fraction,
            target_position=observations.target_position,
            confidence=observations.confidence,
            predictions=self.predicted,
            predicted_fraction=self.predicted_fraction,
            predicted_position=self.predicted_position,
            residuals=self.residuals,
            rms_residual=self.rms_residual,
            max_residual=self.max_residual,
            requested_shifts=self.used_shifts,
        )

    @property
    def objective(self) -> PowerFitObjectiveBreakdown | None:
        """Return the existing objective-contribution breakdown, if available."""

        return self.objective_breakdown

    @property
    def algebraic(self) -> SeparatorAlgebraicView:
        """Return difference-space and connectivity diagnostics."""

        return SeparatorAlgebraicView(
            edge_diagnostics=self.edge_diagnostics,
            connectivity=self.connectivity,
        )

    @property
    def solver_termination(self) -> SeparatorSolverTerminationView:
        """Return termination metadata for the fixed-observation solver."""

        return SeparatorSolverTerminationView(
            status=self.status,
            status_detail=self.status_detail,
            backend=self.solver,
            n_iter=self.n_iter,
            converged=self.converged,
            hard_feasible=self.hard_feasible,
            conflict=self.conflict,
            warnings=self.warnings,
        )

    def to_records(
        self,
        constraints: SeparatorObservations,
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
        constraints: SeparatorObservations,
        *,
        use_ids: bool = False,
    ) -> dict[str, object]:
        from .report import build_fit_report

        return build_fit_report(self, constraints, use_ids=use_ids)


def _separator_fit_result_getstate(
    result: SeparatorFitResult,
) -> list[object]:
    """Preserve private observation identity in copies and pickle state."""

    values = [getattr(result, field.name) for field in fields(result)]
    values.append(getattr(result, '_originating_observations', None))
    return values


def _separator_fit_result_setstate(
    result: SeparatorFitResult,
    state: list[object],
) -> None:
    """Restore current state while accepting older field-only snapshots."""

    result_fields = fields(result)
    values = list(state)
    if len(values) == len(result_fields) + 1:
        originating = values.pop()
    elif len(values) == len(result_fields):
        originating = None
    else:
        raise ValueError('invalid SeparatorFitResult reconstruction state')
    for field, value in zip(result_fields, values):
        object.__setattr__(result, field.name, value)
    if originating is not None:
        object.__setattr__(
            result,
            '_originating_observations',
            originating,
        )


# Python 3.10's frozen/slotted dataclass decorator replaces state hooks declared
# in the class body. Install these after decoration on every supported version.
SeparatorFitResult.__getstate__ = _separator_fit_result_getstate
SeparatorFitResult.__setstate__ = _separator_fit_result_setstate


# Historical v0.6 name retained as an identity alias during v0.7.
# Planned removal: v0.8.
PowerWeightFitResult = SeparatorFitResult
