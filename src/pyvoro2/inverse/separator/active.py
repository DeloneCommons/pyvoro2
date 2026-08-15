"""Self-consistent active-set refinement for pairwise separator constraints."""

from __future__ import annotations

from dataclasses import InitVar, KW_ONLY, dataclass, fields, replace
import inspect
import sys
from typing import Literal, Sequence

import numpy as np

from ..._internal.inputs import coerce_point_array
from ..._internal.validation import (
    require_bool,
    require_bool_mask,
    require_nonnegative_finite_real,
    require_nonnegative_index,
    require_positive_finite_real,
    require_positive_index,
    require_real_in_interval,
    require_string_choice,
    require_string_tuple,
)
from ..._internal.weight_transforms import (
    validate_weight_representation_options,
    weights_to_radii,
)
from ._numerics import (
    _stable_norm,
    _stable_rms,
    _stable_sum,
    _stable_sum_products_sign,
    _stable_sum_scalar,
)
from .constraints import (
    _external_id_label,
    _validated_ids_array,
    SeparatorObservations,
    resolve_separator_observations,
)
from .model import FitModel
from .realize import (
    _require_realization_domain,
    RealizedPairDiagnostics,
    match_realized_pairs,
)
from .problem import (
    _build_active_set_connectivity_diagnostics,
    _standalone_gauge_policy_description,
    build_power_fit_problem,
    build_power_fit_result,
)
from .solver import (
    ConnectivityDiagnostics,
    SeparatorFitResult,
    _apply_connectivity_policy,
    fit_weights_from_separators,
)
from ...diagnostics import TessellationDiagnostics as TessellationDiagnostics3D
from ...domains import Box as Box3D, OrthorhombicCell, PeriodicCell
from ...planar.diagnostics import TessellationDiagnostics as TessellationDiagnostics2D
from ...planar.domains import Box as Box2D, RectangularCell
from ._identity import (
    _ObservationBindingInit,
    _ObservationBoundResult,
    _bind_full_source,
    _bind_originating_observations,
    _originating_observations,
    _require_observation_association,
    _require_observation_row_data,
    _row_ids,
)

ShiftTuple = tuple[int, ...]
_ActiveTermination = Literal[
    'self_consistent',
    'cycle_detected',
    'max_outer_iter',
    'infeasible_active_set',
    'numerical_failure',
]
_ActiveStateGeneration = Literal['outer_failure', 'final_refit']


def _label_value(
    values: np.ndarray,
    index: int,
    ids: np.ndarray | None,
) -> object:
    if ids is None:
        return int(values[index])
    return _external_id_label(ids, int(values[index]))


def _boundary_value(values: np.ndarray | None, index: int) -> float | None:
    if values is None or np.isnan(values[index]):
        return None
    return float(values[index])


def _require_self_consistent_supported_dim(
    constraints: SeparatorObservations,
) -> None:
    if constraints.dim not in (2, 3):
        raise ValueError(
            'solve_self_consistent_power_weights currently supports only 2D '
            'and 3D resolved constraints'
        )


@dataclass(frozen=True, slots=True)
class ActiveSetOptions:
    add_after: int = 1
    drop_after: int = 2
    relax: float = 1.0
    max_iter: int = 25
    cycle_window: int = 8
    weight_step_tol: float = 1e-8

    def __post_init__(self) -> None:
        add_after = require_positive_index(
            self.add_after,
            name='ActiveSetOptions.add_after',
            maximum=sys.maxsize,
        )
        drop_after = require_positive_index(
            self.drop_after,
            name='ActiveSetOptions.drop_after',
            maximum=sys.maxsize,
        )
        relax = require_real_in_interval(
            self.relax,
            name='ActiveSetOptions.relax',
            lower=0.0,
            upper=1.0,
        )
        if relax <= 0.0:
            raise ValueError('ActiveSetOptions.relax must lie in (0, 1]')
        max_iter = require_positive_index(
            self.max_iter,
            name='ActiveSetOptions.max_iter',
            maximum=sys.maxsize,
        )
        cycle_window = require_positive_index(
            self.cycle_window,
            name='ActiveSetOptions.cycle_window',
            maximum=sys.maxsize,
        )
        weight_step_tol = require_nonnegative_finite_real(
            self.weight_step_tol,
            name='ActiveSetOptions.weight_step_tol',
        )
        object.__setattr__(self, 'add_after', add_after)
        object.__setattr__(self, 'drop_after', drop_after)
        object.__setattr__(self, 'relax', relax)
        object.__setattr__(self, 'max_iter', max_iter)
        object.__setattr__(self, 'cycle_window', cycle_window)
        object.__setattr__(self, 'weight_step_tol', weight_step_tol)


@dataclass(frozen=True, slots=True)
class ActiveSetIteration:
    iteration: int
    n_active: int
    n_realized: int
    n_added: int
    n_removed: int
    rms_residual_all: float
    max_residual_all: float
    weight_step_norm: float
    n_active_fit: int | None = None
    fit_active_graph_n_components: int | None = None
    fit_active_effective_graph_n_components: int | None = None
    fit_active_offsets_identified_by_data: bool | None = None
    n_unaccounted_pairs: int | None = None


@dataclass(frozen=True, slots=True)
class ActiveSetPathSummary:
    """Compact summary of transient active-set path diagnostics."""

    n_iterations: int
    ever_fit_active_graph_disconnected: bool
    ever_fit_active_effective_graph_disconnected: bool
    ever_fit_active_offsets_unidentified_by_data: bool
    ever_unaccounted_pairs: bool
    max_fit_active_graph_components: int
    max_fit_active_effective_graph_components: int
    max_n_unaccounted_pairs: int
    first_fit_active_graph_disconnected_iter: int | None = None
    first_fit_active_effective_graph_disconnected_iter: int | None = None
    first_unaccounted_pairs_iter: int | None = None


@dataclass(slots=True)
class _ActiveSetPathAccumulator:
    n_iterations: int = 0
    ever_fit_active_graph_disconnected: bool = False
    ever_fit_active_effective_graph_disconnected: bool = False
    ever_fit_active_offsets_unidentified_by_data: bool = False
    ever_unaccounted_pairs: bool = False
    max_fit_active_graph_components: int = 0
    max_fit_active_effective_graph_components: int = 0
    max_n_unaccounted_pairs: int = 0
    first_fit_active_graph_disconnected_iter: int | None = None
    first_fit_active_effective_graph_disconnected_iter: int | None = None
    first_unaccounted_pairs_iter: int | None = None


@dataclass(frozen=True, slots=True)
class PairConstraintDiagnostics(_ObservationBoundResult):
    site_i: np.ndarray
    site_j: np.ndarray
    shift: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    predicted: np.ndarray
    predicted_fraction: np.ndarray
    predicted_position: np.ndarray
    residuals: np.ndarray
    active: np.ndarray
    realized: np.ndarray
    realized_same_shift: np.ndarray
    realized_other_shift: np.ndarray
    realized_shifts: tuple[tuple[ShiftTuple, ...], ...]
    endpoint_i_empty: np.ndarray
    endpoint_j_empty: np.ndarray
    boundary_measure: np.ndarray | None
    toggle_count: np.ndarray
    realized_toggle_count: np.ndarray
    first_realized_iter: np.ndarray
    last_realized_iter: np.ndarray
    marginal: np.ndarray
    status: tuple[str, ...]
    _: KW_ONLY
    _originating_observations_init: InitVar[
        SeparatorObservations | None
    ] = _ObservationBindingInit()

    def __post_init__(
        self,
        _originating_observations_init: SeparatorObservations | None,
    ) -> None:
        object.__setattr__(
            self,
            'status',
            require_string_tuple(self.status, name='status'),
        )
        if _originating_observations_init is not None:
            _bind_originating_observations(
                self,
                _originating_observations_init,
            )
            _require_observation_row_data(
                _originating_observations_init,
                i=self.site_i,
                j=self.site_j,
                shifts=self.shift,
                target=self.target,
                confidence=self.confidence,
                context='active constraint diagnostics',
            )

    def to_records(
        self,
        *,
        ids: Sequence[int | np.integer] | np.ndarray | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Return one plain-Python record per candidate pair."""

        ids_array = None if ids is None else _validated_ids_array(ids)
        originating = _originating_observations(
            self,
            context='active constraint diagnostics records',
        )
        _require_observation_row_data(
            originating,
            i=self.site_i,
            j=self.site_j,
            shifts=self.shift,
            target=self.target,
            confidence=self.confidence,
            context='active constraint diagnostics records',
        )
        row_ids = _row_ids(originating)
        rows: list[dict[str, object]] = []
        for k in range(int(self.site_i.shape[0])):
            realized_shifts = tuple(
                tuple(int(v) for v in shift)
                for shift in self.realized_shifts[k]
            )
            rows.append(
                {
                    'constraint_index': int(k),
                    'row_id': row_ids[k],
                    'site_i': _label_value(self.site_i, k, ids_array),
                    'site_j': _label_value(self.site_j, k, ids_array),
                    'shift': tuple(int(v) for v in self.shift[k]),
                    'target': float(self.target[k]),
                    'confidence': float(self.confidence[k]),
                    'predicted': float(self.predicted[k]),
                    'predicted_fraction': float(self.predicted_fraction[k]),
                    'predicted_position': float(self.predicted_position[k]),
                    'residual': float(self.residuals[k]),
                    'active': bool(self.active[k]),
                    'realized': bool(self.realized[k]),
                    'realized_same_shift': bool(self.realized_same_shift[k]),
                    'realized_other_shift': bool(self.realized_other_shift[k]),
                    'realized_shifts': realized_shifts,
                    'endpoint_i_empty': bool(self.endpoint_i_empty[k]),
                    'endpoint_j_empty': bool(self.endpoint_j_empty[k]),
                    'boundary_measure': _boundary_value(self.boundary_measure, k),
                    'toggle_count': int(self.toggle_count[k]),
                    'realized_toggle_count': int(self.realized_toggle_count[k]),
                    'first_realized_iter': int(self.first_realized_iter[k]),
                    'last_realized_iter': int(self.last_realized_iter[k]),
                    'marginal': bool(self.marginal[k]),
                    'status': self.status[k],
                }
            )
        return tuple(rows)


def _pair_constraint_diagnostics_getstate(
    diagnostics: PairConstraintDiagnostics,
) -> list[object]:
    values = [getattr(diagnostics, field.name) for field in fields(diagnostics)]
    values.append(getattr(diagnostics, '_originating_observations', None))
    return values


def _pair_constraint_diagnostics_setstate(
    diagnostics: PairConstraintDiagnostics,
    state: list[object],
) -> None:
    diagnostic_fields = fields(diagnostics)
    values = list(state)
    if len(values) == len(diagnostic_fields) + 1:
        originating = values.pop()
    elif len(values) == len(diagnostic_fields):
        originating = None
    else:
        raise ValueError('invalid PairConstraintDiagnostics reconstruction state')
    for field, value in zip(diagnostic_fields, values):
        object.__setattr__(diagnostics, field.name, value)
    diagnostics.__post_init__(originating)


PairConstraintDiagnostics.__getstate__ = _pair_constraint_diagnostics_getstate
PairConstraintDiagnostics.__setstate__ = _pair_constraint_diagnostics_setstate
_pair_diagnostics_signature = inspect.signature(PairConstraintDiagnostics)
PairConstraintDiagnostics.__signature__ = _pair_diagnostics_signature.replace(
    parameters=tuple(
        parameter
        for parameter in _pair_diagnostics_signature.parameters.values()
        if parameter.name != '_originating_observations_init'
    )
)


@dataclass(frozen=True, slots=True)
class ActiveSetTerminationView:
    """Termination state of the experimental active-set outer loop."""

    status: str
    converged: bool
    n_outer_iter: int
    cycle_length: int | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'status',
            require_string_choice(
                self.status,
                name='status',
                choices=(
                    'self_consistent',
                    'cycle_detected',
                    'max_outer_iter',
                    'infeasible_active_set',
                    'numerical_failure',
                ),
            ),
        )
        object.__setattr__(
            self,
            'warnings',
            require_string_tuple(self.warnings, name='warnings'),
        )


@dataclass(frozen=True, slots=True)
class ActiveSetPathView:
    """Final active state and optional outer-loop path diagnostics."""

    active_mask: np.ndarray
    marginal_constraint_indices: tuple[int, ...]
    history: tuple[ActiveSetIteration, ...] | None
    summary: ActiveSetPathSummary | None


@dataclass(frozen=True, slots=True)
class _ActiveStateOrigin:
    """R6 observation identity plus private active-state generation data."""

    candidate_observations: SeparatorObservations
    active_observations: SeparatorObservations
    active_mask: np.ndarray
    candidate_row_ids: tuple[str, ...]
    active_row_ids: tuple[str, ...]
    accepted_outer_iteration: int
    generation: _ActiveStateGeneration

    def __post_init__(self) -> None:
        active_mask = require_bool_mask(
            self.active_mask,
            name='accepted active-state mask',
            length=self.candidate_observations.n_constraints,
        ).copy()
        object.__setattr__(self, 'active_mask', active_mask)
        object.__setattr__(
            self,
            'accepted_outer_iteration',
            require_nonnegative_index(
                self.accepted_outer_iteration,
                name='accepted active-state outer iteration',
                maximum=sys.maxsize,
            ),
        )
        object.__setattr__(
            self,
            'generation',
            require_string_choice(
                self.generation,
                name='accepted active-state generation',
                choices=('outer_failure', 'final_refit'),
            ),
        )

        expected_active = self.candidate_observations.subset(active_mask)
        _require_observation_association(
            expected_active,
            self.active_observations,
            context='accepted active-state subset',
        )
        expected_candidate_row_ids = _row_ids(self.candidate_observations)
        expected_active_row_ids = tuple(
            row_id
            for row_id, is_active in zip(
                expected_candidate_row_ids,
                active_mask,
            )
            if bool(is_active)
        )
        if self.candidate_row_ids != expected_candidate_row_ids:
            raise ValueError(
                'accepted active-state candidate row IDs do not match its '
                'candidate observations'
            )
        if self.active_row_ids != expected_active_row_ids:
            raise ValueError(
                'accepted active-state row IDs do not match its active mask'
            )


@dataclass(frozen=True, slots=True)
class _AcceptedActiveSetState:
    """One atomic active-set state accepted for public result assembly."""

    origin: _ActiveStateOrigin
    fit: SeparatorFitResult
    accepted_weights: np.ndarray | None
    realized: RealizedPairDiagnostics | None
    diagnostics: PairConstraintDiagnostics | None
    n_outer_iter: int
    converged: bool
    termination: _ActiveTermination
    cycle_length: int | None
    marginal_constraints: tuple[int, ...]
    rms_residual_all: float | None
    max_residual_all: float | None
    tessellation_diagnostics: (
        TessellationDiagnostics2D | TessellationDiagnostics3D | None
    )
    history: tuple[ActiveSetIteration, ...] | None
    path_summary: ActiveSetPathSummary | None
    warnings: tuple[str, ...]
    connectivity: ConnectivityDiagnostics | None

    def __post_init__(self) -> None:
        termination = require_string_choice(
            self.termination,
            name='accepted active-state termination',
            choices=(
                'self_consistent',
                'cycle_detected',
                'max_outer_iter',
                'infeasible_active_set',
                'numerical_failure',
            ),
        )
        object.__setattr__(self, 'termination', termination)
        object.__setattr__(
            self,
            'warnings',
            require_string_tuple(self.warnings, name='warnings'),
        )
        if bool(self.converged) != (termination == 'self_consistent'):
            raise ValueError(
                'active result converged must be true exactly for '
                "termination='self_consistent'"
            )

        fit_origin = _originating_observations(
            self.fit,
            context='accepted active-state fit',
        )
        _require_observation_association(
            self.origin.active_observations,
            fit_origin,
            context='accepted active-state fit',
        )
        if _row_ids(fit_origin) != self.origin.active_row_ids:
            raise ValueError(
                'accepted active-state fit row IDs do not match the active '
                'mask'
            )
        self.fit.observation_view(self.origin.active_observations)

        if self.accepted_weights is None:
            self._require_unavailable_mode()
        else:
            self._require_available_mode()

    def _require_unavailable_mode(self) -> None:
        if self.fit.status in ('optimal', 'max_iter'):
            raise ValueError(
                'a weighted final fit status cannot be represented without '
                'complete finite weights and radii'
            )
        if self.fit.weights is not None or self.fit.radii is not None:
            raise ValueError(
                'unavailable active state cannot retain fitted weights or radii'
            )
        if self.fit.converged:
            raise ValueError(
                'an unavailable final fit cannot claim inner convergence'
            )
        if any(
            value is not None
            for value in (
                self.realized,
                self.diagnostics,
                self.rms_residual_all,
                self.max_residual_all,
                self.tessellation_diagnostics,
            )
        ):
            raise ValueError(
                'unavailable active state cannot retain weights-dependent '
                'final layers'
            )

    def _require_available_mode(self) -> None:
        weights = np.asarray(self.accepted_weights, dtype=np.float64)
        fit_weights = self.fit.weights
        fit_radii = self.fit.radii
        n_points = self.origin.candidate_observations.n_points
        if self.fit.status not in ('optimal', 'max_iter'):
            raise ValueError(
                'available active state requires a weighted optimal or '
                'max_iter final fit'
            )
        if self.fit.converged != (self.fit.status == 'optimal'):
            raise ValueError(
                'weighted active final fit convergence does not match its '
                'optimal/max_iter status'
            )
        if (
            fit_weights is None
            or fit_radii is None
            or weights.shape != (n_points,)
            or np.asarray(fit_weights).shape != (n_points,)
            or np.asarray(fit_radii).shape != (n_points,)
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(fit_weights))
            or not np.all(np.isfinite(fit_radii))
            or not np.array_equal(weights, fit_weights)
            or self.fit.weight_shift is None
            or not np.isfinite(self.fit.weight_shift)
        ):
            raise ValueError(
                'available active state requires one complete finite final '
                'weight/radius vector'
            )
        try:
            expected_radii, expected_shift = weights_to_radii(
                weights,
                weight_shift=float(self.fit.weight_shift),
            )
        except ValueError as exc:
            raise ValueError(
                'available active state has an inconsistent final '
                'weight/radius representation'
            ) from exc
        if expected_shift != self.fit.weight_shift or not np.array_equal(
            expected_radii,
            fit_radii,
        ):
            raise ValueError(
                'available active state realization radii do not represent '
                'the final weights'
            )
        if (
            self.realized is None
            or self.diagnostics is None
            or self.rms_residual_all is None
            or self.max_residual_all is None
        ):
            raise ValueError(
                'available active state requires realization, diagnostics, '
                'and residual summaries'
            )
        if not np.isfinite(self.rms_residual_all) or not np.isfinite(
            self.max_residual_all
        ):
            raise ValueError(
                'available active-state residual summaries must be finite'
            )

        candidate = self.origin.candidate_observations
        realized_origin = _originating_observations(
            self.realized,
            context='accepted active-state realization',
        )
        _require_observation_association(
            candidate,
            realized_origin,
            context='accepted active-state realization',
        )
        diagnostics_origin = _originating_observations(
            self.diagnostics,
            context='accepted active-state diagnostics',
        )
        _require_observation_association(
            candidate,
            diagnostics_origin,
            context='accepted active-state diagnostics',
        )
        _require_observation_row_data(
            diagnostics_origin,
            i=self.diagnostics.site_i,
            j=self.diagnostics.site_j,
            shifts=self.diagnostics.shift,
            target=self.diagnostics.target,
            confidence=self.diagnostics.confidence,
            context='accepted active-state diagnostics',
        )
        if _row_ids(diagnostics_origin) != self.origin.candidate_row_ids:
            raise ValueError(
                'accepted active-state diagnostic row IDs do not match the '
                'candidate observations'
            )
        if not np.array_equal(
            self.diagnostics.active,
            self.origin.active_mask,
        ):
            raise ValueError(
                'accepted active-state diagnostics do not match the active mask'
            )
        for name in (
            'realized',
            'realized_same_shift',
            'realized_other_shift',
            'endpoint_i_empty',
            'endpoint_j_empty',
        ):
            if not np.array_equal(
                getattr(self.diagnostics, name),
                getattr(self.realized, name),
            ):
                raise ValueError(
                    'accepted active-state diagnostics and realization differ '
                    f'for {name}'
                )
        if self.diagnostics.realized_shifts != self.realized.realized_shifts:
            raise ValueError(
                'accepted active-state diagnostics and realization use '
                'different realized shifts'
            )
        diagnostic_boundary = self.diagnostics.boundary_measure
        realized_boundary = self.realized.boundary_measure
        if (diagnostic_boundary is None) != (realized_boundary is None) or (
            diagnostic_boundary is not None
            and realized_boundary is not None
            and not np.array_equal(
                diagnostic_boundary,
                realized_boundary,
                equal_nan=True,
            )
        ):
            raise ValueError(
                'accepted active-state diagnostics and realization use '
                'different boundary measures'
            )
        if self.tessellation_diagnostics is not self.realized.tessellation_diagnostics:
            raise ValueError(
                'accepted active-state tessellation diagnostics do not match '
                'the final realization'
            )
        expected_prediction = build_power_fit_problem(candidate).predict(weights)
        expected_target = (
            candidate.target_fraction
            if candidate.measurement == 'fraction'
            else candidate.target_position
        )
        expected_residuals = expected_prediction.measurement - expected_target
        for name, actual, expected in (
            ('predicted', self.diagnostics.predicted, expected_prediction.measurement),
            (
                'predicted_fraction',
                self.diagnostics.predicted_fraction,
                expected_prediction.fraction,
            ),
            (
                'predicted_position',
                self.diagnostics.predicted_position,
                expected_prediction.position,
            ),
            ('residuals', self.diagnostics.residuals, expected_residuals),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(
                    'accepted active-state diagnostics were not derived from '
                    f'the final weights ({name})'
                )
        active_mask = self.origin.active_mask
        for name, actual, expected in (
            ('predicted', self.fit.predicted, expected_prediction.measurement),
            (
                'predicted_fraction',
                self.fit.predicted_fraction,
                expected_prediction.fraction,
            ),
            (
                'predicted_position',
                self.fit.predicted_position,
                expected_prediction.position,
            ),
        ):
            if actual is None or not np.array_equal(actual, expected[active_mask]):
                raise ValueError(
                    'accepted active-state final fit was not rebuilt from '
                    f'the final weights ({name})'
                )
        expected_rms = _stable_rms(expected_residuals)
        expected_max = (
            float(np.max(np.abs(expected_residuals)))
            if expected_residuals.size
            else 0.0
        )
        if (
            self.rms_residual_all != expected_rms
            or self.max_residual_all != expected_max
        ):
            raise ValueError(
                'accepted active-state residual summaries were not derived '
                'from the final weights'
            )
        if tuple(np.flatnonzero(self.diagnostics.marginal).tolist()) != (
            self.marginal_constraints
        ):
            raise ValueError(
                'accepted active-state marginal indices do not match final '
                'candidate diagnostics'
            )

    def to_result(self) -> SelfConsistentPowerFitResult:
        return SelfConsistentPowerFitResult(
            constraints=self.origin.candidate_observations,
            fit=self.fit,
            realized=self.realized,
            diagnostics=self.diagnostics,
            active_mask=self.origin.active_mask.copy(),
            n_outer_iter=self.n_outer_iter,
            converged=self.converged,
            termination=self.termination,
            cycle_length=self.cycle_length,
            marginal_constraints=self.marginal_constraints,
            rms_residual_all=self.rms_residual_all,
            max_residual_all=self.max_residual_all,
            tessellation_diagnostics=self.tessellation_diagnostics,
            history=self.history,
            path_summary=self.path_summary,
            warnings=self.warnings,
            connectivity=self.connectivity,
        )


@dataclass(frozen=True, slots=True)
class SelfConsistentPowerFitResult:
    constraints: SeparatorObservations
    fit: SeparatorFitResult
    realized: RealizedPairDiagnostics | None
    diagnostics: PairConstraintDiagnostics | None
    active_mask: np.ndarray
    n_outer_iter: int
    converged: bool
    termination: Literal[
        'self_consistent',
        'cycle_detected',
        'max_outer_iter',
        'infeasible_active_set',
        'numerical_failure',
    ]
    cycle_length: int | None
    marginal_constraints: tuple[int, ...]
    rms_residual_all: float | None
    max_residual_all: float | None
    tessellation_diagnostics: (
        TessellationDiagnostics2D | TessellationDiagnostics3D | None
    )
    history: tuple[ActiveSetIteration, ...] | None
    path_summary: ActiveSetPathSummary | None = None
    warnings: tuple[str, ...] = ()
    connectivity: ConnectivityDiagnostics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'termination',
            require_string_choice(
                self.termination,
                name='termination',
                choices=(
                    'self_consistent',
                    'cycle_detected',
                    'max_outer_iter',
                    'infeasible_active_set',
                    'numerical_failure',
                ),
            ),
        )
        object.__setattr__(
            self,
            'warnings',
            require_string_tuple(self.warnings, name='warnings'),
        )
        _accepted_state_from_result(self)

    @property
    def inner_fit(self) -> SeparatorFitResult:
        """Return the final fixed-observation inner fit."""

        return self.fit

    @property
    def final_realization(self) -> RealizedPairDiagnostics | None:
        """Return final realization diagnostics when weights are available."""

        return self.realized

    @property
    def candidate_diagnostics(self) -> PairConstraintDiagnostics | None:
        """Return final candidate diagnostics when weights are available."""

        return self.diagnostics

    @property
    def outer_termination(self) -> ActiveSetTerminationView:
        """Return termination metadata for the experimental outer loop."""

        return ActiveSetTerminationView(
            status=self.termination,
            converged=self.converged,
            n_outer_iter=self.n_outer_iter,
            cycle_length=self.cycle_length,
            warnings=self.warnings,
        )

    @property
    def path(self) -> ActiveSetPathView:
        """Return the final active state and optional path history."""

        return ActiveSetPathView(
            active_mask=self.active_mask,
            marginal_constraint_indices=self.marginal_constraints,
            history=self.history,
            summary=self.path_summary,
        )

    @property
    def final_state_available(self) -> bool:
        """Whether all required weights-dependent final layers are available."""

        return self.fit.weights is not None

    @property
    def final_state_unavailable_reason(self) -> str | None:
        """Return the final fit status when final layers are unavailable."""

        return None if self.final_state_available else self.fit.status

    @property
    def final_refit_converged(self) -> bool:
        """Return convergence of the final accepted inner fit/refit."""

        return bool(self.fit.converged)

    def to_records(
        self,
        *,
        use_ids: bool = False,
    ) -> tuple[dict[str, object], ...] | None:
        """Return candidate records, or ``None`` without final weights."""

        use_ids_value = require_bool(use_ids, name='use_ids')
        if self.diagnostics is None:
            return None
        ids = self.constraints.ids if use_ids_value else None
        return self.diagnostics.to_records(ids=ids)

    def to_report(self, *, use_ids: bool = False) -> dict[str, object]:
        """Return a JSON-friendly report for this active-set solve."""

        from .report import build_active_set_report

        use_ids_value = require_bool(use_ids, name='use_ids')
        return build_active_set_report(self, use_ids=use_ids_value)


def _active_state_origin(
    constraints: SeparatorObservations,
    active_constraints: SeparatorObservations,
    active_mask: np.ndarray,
    *,
    accepted_outer_iteration: int,
    generation: _ActiveStateGeneration,
) -> _ActiveStateOrigin:
    candidate_row_ids = _row_ids(constraints)
    return _ActiveStateOrigin(
        candidate_observations=constraints,
        active_observations=active_constraints,
        active_mask=active_mask,
        candidate_row_ids=candidate_row_ids,
        active_row_ids=tuple(
            row_id
            for row_id, is_active in zip(candidate_row_ids, active_mask)
            if bool(is_active)
        ),
        accepted_outer_iteration=accepted_outer_iteration,
        generation=generation,
    )


def _accepted_state_from_result(
    result: SelfConsistentPowerFitResult,
) -> _AcceptedActiveSetState:
    """Reconstitute and validate the private accepted state for public views."""

    active_constraints = result.constraints.subset(result.active_mask)
    generation: _ActiveStateGeneration = (
        'outer_failure'
        if result.termination in ('infeasible_active_set', 'numerical_failure')
        else 'final_refit'
    )
    origin = _active_state_origin(
        result.constraints,
        active_constraints,
        result.active_mask,
        accepted_outer_iteration=result.n_outer_iter,
        generation=generation,
    )
    return _AcceptedActiveSetState(
        origin=origin,
        fit=result.fit,
        accepted_weights=(
            None
            if result.fit.weights is None
            else np.asarray(result.fit.weights, dtype=np.float64).copy()
        ),
        realized=result.realized,
        diagnostics=result.diagnostics,
        n_outer_iter=result.n_outer_iter,
        converged=result.converged,
        termination=result.termination,
        cycle_length=result.cycle_length,
        marginal_constraints=result.marginal_constraints,
        rms_residual_all=result.rms_residual_all,
        max_residual_all=result.max_residual_all,
        tessellation_diagnostics=result.tessellation_diagnostics,
        history=result.history,
        path_summary=result.path_summary,
        warnings=result.warnings,
        connectivity=result.connectivity,
    )


def _normalize_fit_for_accepted_state(
    fit: SeparatorFitResult,
    constraints: SeparatorObservations,
) -> SeparatorFitResult:
    """Normalize a malformed claimed weighted fit to structured failure."""

    _bind_originating_observations(fit, constraints)
    n_points = constraints.n_points
    weights = fit.weights
    radii = fit.radii
    weighted_status = fit.status in ('optimal', 'max_iter')
    usable = False
    if weights is not None and radii is not None:
        try:
            weight_values = np.asarray(weights, dtype=np.float64)
            radius_values = np.asarray(radii, dtype=np.float64)
            usable = bool(
                weight_values.shape == (n_points,)
                and radius_values.shape == (n_points,)
                and np.all(np.isfinite(weight_values))
                and np.all(np.isfinite(radius_values))
                and fit.weight_shift is not None
                and np.isfinite(fit.weight_shift)
            )
            if usable:
                expected_radii, expected_shift = weights_to_radii(
                    weight_values,
                    weight_shift=float(fit.weight_shift),
                )
                usable = bool(
                    expected_shift == fit.weight_shift
                    and np.array_equal(expected_radii, radius_values)
                )
        except (TypeError, ValueError):
            usable = False

    if weighted_status and not usable:
        message = (
            f"final fit status {fit.status!r} did not provide one complete "
            'finite weight/radius vector'
        )
        normalized = replace(
            fit,
            status='numerical_failure',
            status_detail=message,
            weights=None,
            radii=None,
            weight_shift=None,
            predicted=None,
            predicted_fraction=None,
            predicted_position=None,
            residuals=None,
            rms_residual=None,
            max_residual=None,
            converged=False,
            warnings=fit.warnings + (message,),
            edge_diagnostics=None,
            objective_breakdown=None,
        )
        return _bind_originating_observations(normalized, constraints)

    if not weighted_status and (weights is not None or radii is not None):
        raise ValueError(
            'a non-weighted final fit status cannot retain weights or radii'
        )
    if not weighted_status and fit.weight_shift is not None:
        raise ValueError(
            'a no-weights final fit cannot retain a backend weight shift'
        )
    return fit


def _assemble_accepted_active_set_state(
    *,
    points: np.ndarray,
    domain: Box2D | RectangularCell | Box3D | OrthorhombicCell | PeriodicCell,
    constraints: SeparatorObservations,
    active_constraints: SeparatorObservations,
    active_mask: np.ndarray,
    fit: SeparatorFitResult,
    full_problem: object,
    toggle_count: np.ndarray,
    realized_toggle_count: np.ndarray,
    first_realized_iter: np.ndarray,
    last_realized_iter: np.ndarray,
    n_outer_iter: int,
    accepted_outer_iteration: int,
    generation: _ActiveStateGeneration,
    converged: bool,
    termination: _ActiveTermination,
    cycle_length: int | None,
    history_rows: list[ActiveSetIteration],
    path_acc: _ActiveSetPathAccumulator,
    return_history: bool,
    return_boundary_measure: bool,
    return_cells: bool,
    return_tessellation_diagnostics: bool,
    tessellation_check: str,
    connectivity_check: str,
    unaccounted_pair_check: str,
    gauge_policy: str,
    model: FitModel,
    warnings_list: list[str],
) -> _AcceptedActiveSetState:
    """Assemble the sole state from which an active public result is built."""

    fit = _normalize_fit_for_accepted_state(fit, active_constraints)
    warnings = list(warnings_list)
    warnings.extend(fit.warnings)
    origin = _active_state_origin(
        constraints,
        active_constraints,
        active_mask,
        accepted_outer_iteration=accepted_outer_iteration,
        generation=generation,
    )

    realized: RealizedPairDiagnostics | None = None
    diagnostics: PairConstraintDiagnostics | None = None
    rms_residual_all: float | None = None
    max_residual_all: float | None = None
    tessellation_diagnostics = None
    accepted_weights: np.ndarray | None = None

    path_marginal = toggle_count > 0
    if termination == 'cycle_detected':
        path_marginal = path_marginal | (realized_toggle_count > 0)

    if fit.weights is not None:
        accepted_weights = np.asarray(fit.weights, dtype=np.float64).copy()
        realized = match_realized_pairs(
            points,
            domain=domain,
            radii=fit.radii,
            constraints=constraints,
            return_boundary_measure=return_boundary_measure,
            return_cells=return_cells,
            return_tessellation_diagnostics=return_tessellation_diagnostics,
            tessellation_check=tessellation_check,
            unaccounted_pair_check=unaccounted_pair_check,
        )
        _bind_originating_observations(realized, constraints)
        warnings.extend(realized.warnings)

        prediction = full_problem.predict(accepted_weights)
        predicted_fraction = np.asarray(
            prediction.fraction,
            dtype=np.float64,
        )
        predicted_position = np.asarray(
            prediction.position,
            dtype=np.float64,
        )
        predicted = np.asarray(prediction.measurement, dtype=np.float64)
        target = (
            constraints.target_fraction
            if constraints.measurement == 'fraction'
            else constraints.target_position
        )
        residuals = predicted - target
        if not all(
            np.all(np.isfinite(values))
            for values in (
                predicted,
                predicted_fraction,
                predicted_position,
                residuals,
            )
        ):
            raise ValueError(
                'finite final weights produced non-finite active candidate '
                'diagnostics'
            )
        rms_residual_all = _stable_rms(residuals)
        max_residual_all = (
            float(np.max(np.abs(residuals))) if residuals.size else 0.0
        )

        marginal = path_marginal | realized.realized_other_shift
        status = _build_constraint_statuses(
            active=active_mask,
            realized=realized,
            toggle_count=toggle_count,
            realized_toggle_count=realized_toggle_count,
            termination=termination,
        )
        diagnostics = PairConstraintDiagnostics(
            site_i=constraints.i.copy(),
            site_j=constraints.j.copy(),
            shift=constraints.shifts.copy(),
            target=constraints.target.copy(),
            confidence=constraints.confidence.copy(),
            predicted=predicted,
            predicted_fraction=predicted_fraction,
            predicted_position=predicted_position,
            residuals=residuals,
            active=active_mask.copy(),
            realized=realized.realized.copy(),
            realized_same_shift=realized.realized_same_shift.copy(),
            realized_other_shift=realized.realized_other_shift.copy(),
            realized_shifts=realized.realized_shifts,
            endpoint_i_empty=realized.endpoint_i_empty.copy(),
            endpoint_j_empty=realized.endpoint_j_empty.copy(),
            boundary_measure=(
                None
                if realized.boundary_measure is None
                else realized.boundary_measure.copy()
            ),
            toggle_count=toggle_count.copy(),
            realized_toggle_count=realized_toggle_count.copy(),
            first_realized_iter=first_realized_iter.copy(),
            last_realized_iter=last_realized_iter.copy(),
            marginal=marginal.copy(),
            status=status,
        )
        _bind_originating_observations(diagnostics, constraints)
        tessellation_diagnostics = realized.tessellation_diagnostics
    else:
        marginal = path_marginal

    marginal_constraints = tuple(np.flatnonzero(marginal).tolist())
    connectivity = None
    if connectivity_check != 'none':
        connectivity = _build_active_set_connectivity_diagnostics(
            constraints,
            active_mask,
            model=model,
            gauge_policy=gauge_policy,
        )
        _apply_connectivity_policy(
            connectivity_check,
            connectivity,
            warnings,
        )

    return _AcceptedActiveSetState(
        origin=origin,
        fit=fit,
        accepted_weights=accepted_weights,
        realized=realized,
        diagnostics=diagnostics,
        n_outer_iter=n_outer_iter,
        converged=converged,
        termination=termination,
        cycle_length=cycle_length,
        marginal_constraints=marginal_constraints,
        rms_residual_all=rms_residual_all,
        max_residual_all=max_residual_all,
        tessellation_diagnostics=tessellation_diagnostics,
        history=tuple(history_rows) if return_history else None,
        path_summary=_finalize_path_summary(path_acc),
        warnings=tuple(warnings),
        connectivity=connectivity,
    )


def solve_self_consistent_power_weights(
    points: np.ndarray,
    constraints: SeparatorObservations | list[tuple] | tuple[tuple, ...],
    *,
    measurement: Literal['fraction', 'position'] = 'fraction',
    domain: Box2D | RectangularCell | Box3D | OrthorhombicCell | PeriodicCell,
    ids: Sequence[int | np.integer] | np.ndarray | None = None,
    index_mode: Literal['index', 'id'] = 'index',
    image: Literal['nearest', 'given_only'] = 'nearest',
    image_search: int = 1,
    confidence: list[float] | tuple[float, ...] | np.ndarray | None = None,
    model: FitModel | None = None,
    active0: np.ndarray | None = None,
    options: ActiveSetOptions | None = None,
    r_min: float = 0.0,
    weight_shift: float | None = None,
    fit_solver: Literal['direct', 'admm'] = 'direct',
    fit_linear_backend: Literal['dense', 'sparse'] = 'dense',
    fit_admm_max_iter: int = 2000,
    fit_admm_rho: float = 1.0,
    fit_admm_abs_tol: float = 1e-6,
    fit_admm_rel_tol: float = 1e-5,
    return_history: bool = False,
    return_cells: bool = False,
    return_boundary_measure: bool = False,
    return_tessellation_diagnostics: bool = False,
    tessellation_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'diagnose',
    connectivity_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
    unaccounted_pair_check: Literal['none', 'diagnose', 'warn', 'raise'] = 'warn',
) -> SelfConsistentPowerFitResult:
    """Iteratively refine an active pair set against realized power-diagram
    boundaries."""
    measurement = require_string_choice(
        measurement,
        name='measurement',
        choices=('fraction', 'position'),
    )
    index_mode = require_string_choice(
        index_mode,
        name='index_mode',
        choices=('index', 'id'),
    )
    image = require_string_choice(
        image,
        name='image',
        choices=('nearest', 'given_only'),
    )
    fit_solver = require_string_choice(
        fit_solver,
        name='fit_solver',
        choices=('direct', 'admm'),
    )
    fit_linear_backend = require_string_choice(
        fit_linear_backend,
        name='fit_linear_backend',
        choices=('dense', 'sparse'),
    )
    connectivity_check = require_string_choice(
        connectivity_check,
        name='connectivity_check',
        choices=('none', 'diagnose', 'warn', 'raise'),
    )
    unaccounted_pair_check = require_string_choice(
        unaccounted_pair_check,
        name='unaccounted_pair_check',
        choices=('none', 'diagnose', 'warn', 'raise'),
    )
    tessellation_check = require_string_choice(
        tessellation_check,
        name='tessellation_check',
        choices=('none', 'diagnose', 'warn', 'raise'),
    )

    image_search_value = require_nonnegative_index(
        image_search,
        name='image_search',
        maximum=sys.maxsize,
    )
    fit_admm_max_iter_value = require_positive_index(
        fit_admm_max_iter,
        name='fit_admm_max_iter',
        maximum=sys.maxsize,
    )
    fit_admm_rho_value = require_positive_finite_real(
        fit_admm_rho,
        name='fit_admm_rho',
    )
    fit_admm_abs_tol_value = require_positive_finite_real(
        fit_admm_abs_tol,
        name='fit_admm_abs_tol',
    )
    fit_admm_rel_tol_value = require_positive_finite_real(
        fit_admm_rel_tol,
        name='fit_admm_rel_tol',
    )
    r_min_value, weight_shift_value = validate_weight_representation_options(
        r_min,
        weight_shift,
    )
    return_history_value = require_bool(return_history, name='return_history')
    return_cells_value = require_bool(return_cells, name='return_cells')
    return_boundary_measure_value = require_bool(
        return_boundary_measure,
        name='return_boundary_measure',
    )
    return_tessellation_diagnostics_value = require_bool(
        return_tessellation_diagnostics,
        name='return_tessellation_diagnostics',
    )

    raw_points = np.asarray(points, dtype=object)
    if raw_points.ndim != 2 or raw_points.shape[1] <= 0:
        raise ValueError('points must have shape (n, d) with d >= 1')
    pts = coerce_point_array(
        raw_points,
        name='points',
        dim=int(raw_points.shape[1]),
    )
    if pts.shape[1] in (2, 3):
        _require_realization_domain(int(pts.shape[1]), domain)

    if model is None:
        model = FitModel()
    if options is None:
        options = ActiveSetOptions()

    if isinstance(constraints, SeparatorObservations):
        resolved = constraints
        if resolved.n_points != pts.shape[0]:
            raise ValueError('resolved constraints do not match the number of points')
        if resolved.dim != pts.shape[1]:
            raise ValueError('resolved constraints do not match the point dimension')
        _require_self_consistent_supported_dim(resolved)
    else:
        if pts.shape[1] not in (2, 3):
            raise ValueError(
                'solve_self_consistent_power_weights currently supports only '
                '2D and 3D points'
            )
        resolved = resolve_separator_observations(
            pts,
            constraints,
            measurement=measurement,
            domain=domain,
            ids=ids,
            index_mode=index_mode,
            image=image,
            image_search=image_search_value,
            confidence=confidence,
            allow_empty=True,
        )

    _bind_full_source(resolved, pts, domain)

    m = resolved.n_constraints
    if active0 is None:
        active = np.ones(m, dtype=bool)
    else:
        active = require_bool_mask(
            active0,
            name='active0',
            length=m,
        ).copy()

    warnings_list = list(resolved.warnings)
    full_problem = build_power_fit_problem(resolved, model=model)
    add_streak = np.zeros(m, dtype=np.int64)
    drop_streak = np.zeros(m, dtype=np.int64)
    toggle_count = np.zeros(m, dtype=np.int64)
    realized_toggle_count = np.zeros(m, dtype=np.int64)
    first_realized_iter = np.full(m, -1, dtype=np.int64)
    last_realized_iter = np.full(m, -1, dtype=np.int64)
    history_rows: list[ActiveSetIteration] = []
    path_acc = _ActiveSetPathAccumulator()
    gauge_policy = _self_consistent_gauge_policy_description(model)
    prev_weights_eval: np.ndarray | None = None
    prev_realized_same: np.ndarray | None = None
    seen_masks: dict[bytes, int] = {active.tobytes(): 0}

    termination: _ActiveTermination = 'max_outer_iter'
    cycle_length: int | None = None
    converged = False

    for outer_iter in range(1, options.max_iter + 1):
        active_constraints = resolved.subset(active)
        fit = fit_weights_from_separators(
            pts,
            active_constraints,
            model=model,
            r_min=r_min_value,
            weight_shift=weight_shift_value,
            solver=fit_solver,
            linear_backend=fit_linear_backend,
            admm_max_iter=fit_admm_max_iter_value,
            admm_rho=fit_admm_rho_value,
            admm_abs_tol=fit_admm_abs_tol_value,
            admm_rel_tol=fit_admm_rel_tol_value,
            connectivity_check='diagnose',
        )
        _bind_originating_observations(fit, active_constraints)
        fit = _normalize_fit_for_accepted_state(fit, active_constraints)
        if fit.weights is None:
            termination = (
                'numerical_failure'
                if fit.status == 'numerical_failure'
                else 'infeasible_active_set'
            )
            state = _assemble_accepted_active_set_state(
                points=pts,
                domain=domain,
                constraints=resolved,
                active_constraints=active_constraints,
                active_mask=active,
                fit=fit,
                full_problem=full_problem,
                toggle_count=toggle_count,
                realized_toggle_count=realized_toggle_count,
                first_realized_iter=first_realized_iter,
                last_realized_iter=last_realized_iter,
                n_outer_iter=outer_iter,
                accepted_outer_iteration=outer_iter,
                generation='outer_failure',
                converged=False,
                termination=termination,
                cycle_length=None,
                history_rows=history_rows,
                path_acc=path_acc,
                return_history=return_history_value,
                return_boundary_measure=return_boundary_measure_value,
                return_cells=return_cells_value,
                return_tessellation_diagnostics=(
                    return_tessellation_diagnostics_value
                ),
                tessellation_check=tessellation_check,
                connectivity_check=connectivity_check,
                unaccounted_pair_check=unaccounted_pair_check,
                gauge_policy=gauge_policy,
                model=model,
                warnings_list=warnings_list,
            )
            return state.to_result()

        weights_exact = fit.weights.copy()
        if prev_weights_eval is not None:
            weights_exact = _align_weights_to_reference(
                weights_exact,
                prev_weights_eval,
                _active_alignment_components(active_constraints, model),
            )
            weights_eval = (
                (1.0 - float(options.relax)) * prev_weights_eval
                + float(options.relax) * weights_exact
            )
            weight_step = _stable_sum(weights_eval, -prev_weights_eval)
            step_norm = _stable_norm(weight_step)
        else:
            weights_eval = weights_exact
            step_norm = 0.0

        fit_active_connectivity = _build_active_set_connectivity_diagnostics(
            resolved,
            active,
            model=model,
            gauge_policy=gauge_policy,
        )
        radii_eval, _ = weights_to_radii(
            weights_eval,
            r_min=r_min_value,
            weight_shift=weight_shift_value,
        )
        diag = match_realized_pairs(
            pts,
            domain=domain,
            radii=radii_eval,
            constraints=resolved,
            return_boundary_measure=False,
            return_cells=False,
            return_tessellation_diagnostics=False,
            tessellation_check='none',
            unaccounted_pair_check='diagnose',
        )
        _bind_originating_observations(diag, resolved)
        n_unaccounted_pairs = len(diag.unaccounted_pairs)
        _record_path_iteration(
            path_acc,
            iteration=outer_iter,
            connectivity=fit_active_connectivity,
            n_unaccounted_pairs=n_unaccounted_pairs,
        )
        realized_same = diag.realized_same_shift
        if prev_realized_same is not None:
            realized_toggle_count += prev_realized_same != realized_same
        newly_realized = realized_same & (first_realized_iter < 0)
        first_realized_iter[newly_realized] = outer_iter
        last_realized_iter[realized_same] = outer_iter

        new_active = active.copy()
        for k in range(m):
            if realized_same[k]:
                add_streak[k] += 1
                drop_streak[k] = 0
            else:
                drop_streak[k] += 1
                add_streak[k] = 0

            if active[k]:
                if drop_streak[k] >= options.drop_after:
                    new_active[k] = False
            else:
                if add_streak[k] >= options.add_after:
                    new_active[k] = True

        toggled = new_active != active
        toggle_count += toggled
        n_added = int(np.count_nonzero((~active) & new_active))
        n_removed = int(np.count_nonzero(active & (~new_active)))

        pred_all = full_problem.predict(weights_eval)
        pred = np.asarray(pred_all.measurement, dtype=np.float64)
        target = (
            resolved.target_fraction
            if resolved.measurement == 'fraction'
            else resolved.target_position
        )
        residuals = pred - target
        history_rows.append(
            ActiveSetIteration(
                iteration=outer_iter,
                n_active=int(np.count_nonzero(new_active)),
                n_realized=int(np.count_nonzero(realized_same)),
                n_added=n_added,
                n_removed=n_removed,
                rms_residual_all=_stable_rms(residuals),
                max_residual_all=float(np.max(np.abs(residuals)))
                if residuals.size
                else 0.0,
                weight_step_norm=step_norm,
                n_active_fit=int(np.count_nonzero(active)),
                fit_active_graph_n_components=(
                    fit_active_connectivity.active_graph.n_components
                    if fit_active_connectivity.active_graph is not None
                    else None
                ),
                fit_active_effective_graph_n_components=(
                    fit_active_connectivity.active_effective_graph.n_components
                    if fit_active_connectivity.active_effective_graph is not None
                    else None
                ),
                fit_active_offsets_identified_by_data=(
                    fit_active_connectivity.active_offsets_identified_by_data
                ),
                n_unaccounted_pairs=n_unaccounted_pairs,
            )
        )

        if (
            np.array_equal(new_active, active)
            and np.array_equal(realized_same, active)
            and step_norm <= float(options.weight_step_tol)
        ):
            active = new_active
            prev_weights_eval = weights_eval
            prev_realized_same = realized_same.copy()
            termination = 'self_consistent'
            converged = True
            break

        active_key = new_active.tobytes()
        if np.any(toggled):
            if (
                active_key in seen_masks
                and outer_iter - seen_masks[active_key] <= options.cycle_window
            ):
                cycle_length = outer_iter - seen_masks[active_key]
                active = new_active
                prev_weights_eval = weights_eval
                prev_realized_same = realized_same.copy()
                termination = 'cycle_detected'
                converged = False
                break
            seen_masks[active_key] = outer_iter

        active = new_active
        prev_weights_eval = weights_eval
        prev_realized_same = realized_same.copy()
    else:
        termination = 'max_outer_iter'

    active_constraints = resolved.subset(active)
    final_fit = fit_weights_from_separators(
        pts,
        active_constraints,
        model=model,
        r_min=r_min_value,
        weight_shift=weight_shift_value,
        solver=fit_solver,
        linear_backend=fit_linear_backend,
        admm_max_iter=fit_admm_max_iter_value,
        admm_rho=fit_admm_rho_value,
        admm_abs_tol=fit_admm_abs_tol_value,
        admm_rel_tol=fit_admm_rel_tol_value,
        connectivity_check='diagnose',
    )
    _bind_originating_observations(final_fit, active_constraints)
    final_fit = _normalize_fit_for_accepted_state(
        final_fit,
        active_constraints,
    )

    if final_fit.weights is not None:
        final_weights = final_fit.weights.copy()
        if prev_weights_eval is not None:
            final_weights = _align_weights_to_reference(
                final_weights,
                prev_weights_eval,
                _active_alignment_components(active_constraints, model),
            )
        final_fit = _rebuild_fit_with_weights(
            final_fit,
            active_constraints,
            final_weights,
            model=model,
            r_min=r_min_value,
            weight_shift=weight_shift_value,
        )
        _bind_originating_observations(final_fit, active_constraints)

    state = _assemble_accepted_active_set_state(
        points=pts,
        domain=domain,
        constraints=resolved,
        active_constraints=active_constraints,
        active_mask=active,
        fit=final_fit,
        full_problem=full_problem,
        toggle_count=toggle_count,
        realized_toggle_count=realized_toggle_count,
        first_realized_iter=first_realized_iter,
        last_realized_iter=last_realized_iter,
        n_outer_iter=len(history_rows),
        accepted_outer_iteration=len(history_rows),
        generation='final_refit',
        converged=converged,
        termination=termination,
        cycle_length=cycle_length,
        history_rows=history_rows,
        path_acc=path_acc,
        return_history=return_history_value,
        return_boundary_measure=return_boundary_measure_value,
        return_cells=return_cells_value,
        return_tessellation_diagnostics=return_tessellation_diagnostics_value,
        tessellation_check=tessellation_check,
        connectivity_check=connectivity_check,
        unaccounted_pair_check=unaccounted_pair_check,
        gauge_policy=gauge_policy,
        model=model,
        warnings_list=warnings_list,
    )
    return state.to_result()


def _align_weights_to_reference(
    weights: np.ndarray, reference: np.ndarray, comps: list[list[int]]
) -> np.ndarray:
    """Align true gauge components without changing binary64 differences.

    A mathematically uniform shift can round differently at distinct
    coordinates.  The final low-level fit has already been certified, so an
    alignment is accepted only when exact binary64-input arithmetic proves
    that every component contrast to one anchor is unchanged.  Otherwise the
    certified component is retained verbatim.
    """

    aligned = np.asarray(weights, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64)
    if aligned.shape != ref.shape:
        raise ValueError('weights and reference must have the same shape')
    for comp in comps:
        idx = np.asarray(comp, dtype=np.int64)
        if idx.size == 0:
            continue
        current = aligned[idx].copy()
        difference = _stable_sum(current, -ref[idx])
        if not np.all(np.isfinite(difference)):
            continue
        shift = _stable_sum_scalar(
            *(float(value) / int(idx.size) for value in difference)
        )
        if not np.isfinite(shift):
            continue
        proposed = _stable_sum(current, -shift)
        if not np.all(np.isfinite(proposed)):
            continue
        anchor_before = float(current[0])
        anchor_after = float(proposed[0])
        contrast_change_sign = _stable_sum_products_sign(
            (
                (proposed,),
                (-1.0, anchor_after),
                (-1.0, current),
                (anchor_before,),
            )
        )
        if np.all(contrast_change_sign == 0):
            aligned[idx] = proposed
    return aligned


def _active_alignment_components(
    constraints: SeparatorObservations,
    model: FitModel,
) -> list[list[int]]:
    problem = build_power_fit_problem(constraints, model=model)
    if problem.regularization_strength > 0.0:
        # Positive L2 fixes every component mean and removes the additive
        # gauge.  Aligning such a solution to a previous iterate changes the
        # authoritative objective and invalidates its optimality certificate.
        return []
    return problem._model_coupling_components()


def _self_consistent_gauge_policy_description(model: FitModel) -> str:
    if float(model.regularization.strength) > 0.0:
        return _standalone_gauge_policy_description(model)
    return (
        'each active model-coupling component is aligned to the previous '
        'iterate; the first iterate falls back to the standalone solver and '
        'component-alignment conventions'
    )


def _record_path_iteration(
    acc: _ActiveSetPathAccumulator,
    *,
    iteration: int,
    connectivity: ConnectivityDiagnostics,
    n_unaccounted_pairs: int,
) -> None:
    active_graph = connectivity.active_graph
    active_effective_graph = connectivity.active_effective_graph
    active_offsets_identified = connectivity.active_offsets_identified_by_data

    active_graph_components = (
        0 if active_graph is None else int(active_graph.n_components)
    )
    active_effective_components = (
        0
        if active_effective_graph is None
        else int(active_effective_graph.n_components)
    )

    acc.n_iterations += 1
    acc.max_fit_active_graph_components = max(
        acc.max_fit_active_graph_components,
        active_graph_components,
    )
    acc.max_fit_active_effective_graph_components = max(
        acc.max_fit_active_effective_graph_components,
        active_effective_components,
    )
    acc.max_n_unaccounted_pairs = max(
        acc.max_n_unaccounted_pairs,
        int(n_unaccounted_pairs),
    )

    if active_graph_components > 1:
        acc.ever_fit_active_graph_disconnected = True
        if acc.first_fit_active_graph_disconnected_iter is None:
            acc.first_fit_active_graph_disconnected_iter = int(iteration)
    if active_effective_components > 1:
        acc.ever_fit_active_effective_graph_disconnected = True
        if acc.first_fit_active_effective_graph_disconnected_iter is None:
            acc.first_fit_active_effective_graph_disconnected_iter = int(iteration)
    if active_offsets_identified is False:
        acc.ever_fit_active_offsets_unidentified_by_data = True
    if int(n_unaccounted_pairs) > 0:
        acc.ever_unaccounted_pairs = True
        if acc.first_unaccounted_pairs_iter is None:
            acc.first_unaccounted_pairs_iter = int(iteration)


def _finalize_path_summary(
    acc: _ActiveSetPathAccumulator,
) -> ActiveSetPathSummary:
    return ActiveSetPathSummary(
        n_iterations=int(acc.n_iterations),
        ever_fit_active_graph_disconnected=bool(
            acc.ever_fit_active_graph_disconnected
        ),
        ever_fit_active_effective_graph_disconnected=bool(
            acc.ever_fit_active_effective_graph_disconnected
        ),
        ever_fit_active_offsets_unidentified_by_data=bool(
            acc.ever_fit_active_offsets_unidentified_by_data
        ),
        ever_unaccounted_pairs=bool(acc.ever_unaccounted_pairs),
        max_fit_active_graph_components=int(acc.max_fit_active_graph_components),
        max_fit_active_effective_graph_components=int(
            acc.max_fit_active_effective_graph_components
        ),
        max_n_unaccounted_pairs=int(acc.max_n_unaccounted_pairs),
        first_fit_active_graph_disconnected_iter=(
            None
            if acc.first_fit_active_graph_disconnected_iter is None
            else int(acc.first_fit_active_graph_disconnected_iter)
        ),
        first_fit_active_effective_graph_disconnected_iter=(
            None
            if acc.first_fit_active_effective_graph_disconnected_iter is None
            else int(acc.first_fit_active_effective_graph_disconnected_iter)
        ),
        first_unaccounted_pairs_iter=(
            None
            if acc.first_unaccounted_pairs_iter is None
            else int(acc.first_unaccounted_pairs_iter)
        ),
    )


def _rebuild_fit_with_weights(
    fit: SeparatorFitResult,
    constraints: SeparatorObservations,
    weights: np.ndarray,
    *,
    model: FitModel,
    r_min: float,
    weight_shift: float | None,
) -> SeparatorFitResult:
    problem = build_power_fit_problem(constraints, model=model)
    return build_power_fit_result(
        problem,
        weights,
        solver=fit.solver,
        linear_backend=fit.linear_backend,
        status=fit.status,
        status_detail=fit.status_detail,
        converged=fit.converged,
        n_iter=fit.n_iter,
        warnings=fit.warnings,
        canonicalize_gauge=False,
        r_min=r_min,
        weight_shift=weight_shift,
    )


def _build_constraint_statuses(
    *,
    active: np.ndarray,
    realized: RealizedPairDiagnostics,
    toggle_count: np.ndarray,
    realized_toggle_count: np.ndarray,
    termination: str,
) -> tuple[str, ...]:
    rows: list[str] = []
    for k in range(active.shape[0]):
        if termination == 'numerical_failure':
            rows.append('numerical_failure')
            continue
        if termination == 'cycle_detected' and (
            bool(toggle_count[k] > 0) or bool(realized_toggle_count[k] > 0)
        ):
            rows.append('cycle_member')
            continue
        if bool(realized.realized_other_shift[k]):
            rows.append('realized_other_shift')
            continue
        if bool(realized.endpoint_i_empty[k] or realized.endpoint_j_empty[k]):
            rows.append('endpoint_empty')
            continue
        if bool(active[k]) and bool(realized.realized_same_shift[k]):
            rows.append(
                'toggled_active'
                if bool(toggle_count[k] > 0)
                else 'stable_active'
            )
            continue
        if (not bool(active[k])) and (not bool(realized.realized[k])):
            rows.append(
                'toggled_inactive'
                if bool(toggle_count[k] > 0)
                else 'stable_inactive'
            )
            continue
        if bool(active[k]) and (not bool(realized.realized_same_shift[k])):
            rows.append('active_unrealized')
            continue
        if (not bool(active[k])) and bool(realized.realized_same_shift[k]):
            rows.append('inactive_realized')
            continue
        rows.append('unresolved')
    return tuple(rows)
