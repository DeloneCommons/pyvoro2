"""Private evaluation values, availability and owned diagnostic provenance.

Only typed producers establish reasons. Exporters compare supplied values with
those producers before converting individual eligible leaves to JSON null.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._identity import (
    _ObservationBoundResult, _require_observation_association, _row_ids,
)
from ._numerics import (
    _stable_affine_residual, _stable_norm, _stable_rms,
    _stable_scaled_affine_residual,
)

_RANGE = 'out_of_binary64_range'
_DEPENDENCY = 'unavailable_dependency'


@dataclass(frozen=True, slots=True)
class _DiagnosticValue:
    value: object
    reasons: object

    def require(self, actual, *, context):
        if self.value is None:
            matches = actual is None
        else:
            matches = actual is not None and np.array_equal(
                np.asarray(actual), np.asarray(self.value), equal_nan=True,
            )
        if not matches:
            raise ValueError(f'{context}: diagnostic differs from evaluation')

    def item(self, index):
        if self.value is None:
            return self
        return _DiagnosticValue(float(self.value[index]), self.reasons[index])

    def json_value(self, path, unavailable):
        if self.value is None:
            return None
        if isinstance(self.value, np.ndarray):
            return [self.item(k).json_value(f'{path}/{k}', unavailable)
                    for k in range(self.value.size)]
        if self.reasons is not None:
            unavailable[path] = self.reasons
            return None
        return float(self.value)


def _derived(value, *, operands_available=True):
    """Finish a known producer evaluation, never classify caller metadata.

    The availability mask comes from required accepted operands or exact
    absence. A nonfinite output with finite operands must be a signed overflow;
    an unexplained NaN is a producer error rather than an export permission.
    """
    if value is None:
        return _DiagnosticValue(None, None)
    array = np.asarray(value, dtype=np.float64)
    available = np.broadcast_to(
        np.asarray(operands_available, dtype=bool), array.shape,
    )
    if np.any(available & np.isnan(array)):
        raise ValueError('diagnostic evaluation produced unexplained NaN')
    owned = array.copy()
    owned[~available] = np.nan
    reasons = np.full(array.shape, None, dtype=object)
    reasons[~available] = _DEPENDENCY
    reasons[available & np.isinf(array)] = _RANGE
    if array.ndim == 0:
        return _DiagnosticValue(float(owned), reasons.item())
    owned.setflags(write=False)
    return _DiagnosticValue(owned, tuple(reasons.tolist()))


def _ordinary(value, *, context):
    if value is not None and not np.all(np.isfinite(value)):
        raise ValueError(f'{context}: non-finite diagnostic lacks provenance')
    return _derived(value)


def _affine_diagnostic(beta, alpha, left, right, target, *scales):
    operands = np.broadcast_arrays(*[
        np.asarray(v, dtype=np.float64)
        for v in (beta, alpha, left, right, target, *scales)
    ])
    available = np.logical_and.reduce([np.isfinite(v) for v in operands])
    absent = np.zeros(available.shape, dtype=bool)
    for scale in operands[5:]:
        absent |= scale == 0.
    work = available & ~absent
    if scales:
        value = _stable_scaled_affine_residual(
            *operands[:5], *operands[5:], active=work,
        )
    else:
        value = _stable_affine_residual(*operands[:5], active=work)
    return _derived(value, operands_available=available | absent)


def _has_dependency(value):
    return _DEPENDENCY in value.reasons


def _norm_diagnostic(rows):
    if _has_dependency(rows):
        return _derived(np.nan, operands_available=False)
    # A proven outside-range row is a lower bound for the complete norm. This
    # rule is not valid for RMS until each complete row has been scaled.
    return _derived(_stable_norm(rows.value))


def _affine_rms(beta, alpha, left, right, target, rows, *scales):
    if _has_dependency(rows):
        return _derived(np.nan, operands_available=False)
    if np.all(np.isfinite(rows.value)):
        return _derived(_stable_rms(rows.value))
    count = rows.value.size
    scaled = _affine_diagnostic(
        beta, alpha, left, right, target, *scales, 1./math.sqrt(count),
    )
    return _norm_diagnostic(scaled)


def _max_diagnostic(rows):
    if _has_dependency(rows):
        return _derived(np.nan, operands_available=False)
    value = float(np.max(np.abs(rows.value))) if rows.value.size else 0.
    return _derived(value)


@dataclass(frozen=True, slots=True)
class _RowSnapshot:
    observations: object
    row_ids: tuple
    values: tuple

    @classmethod
    def produced(cls, observations, diagnostics):
        values = tuple(
            (name, tuple(float(v) for v in diagnostic.value),
             diagnostic.reasons)
            for name, diagnostic in diagnostics.items()
        )
        return cls(observations, _row_ids(observations), values)

    def checked(self, owner, observations):
        _require_observation_association(
            self.observations, observations,
            context='candidate diagnostic provenance',
        )
        if self.row_ids != _row_ids(observations):
            raise ValueError('candidate diagnostic provenance has stale rows')
        result = {}
        for name, values, reasons in self.values:
            expected = _DiagnosticValue(np.asarray(values), reasons)
            expected.require(getattr(owner, name), context=name)
            result[name] = expected
        return result


class _CandidateDiagnosticStorage(_ObservationBoundResult):
    __slots__ = ('_diagnostic_snapshot',)


class _DiagnosticBindingInit:
    def __get__(self, instance, owner=None):
        return (None if instance is None
                else getattr(instance, '_diagnostic_snapshot', None))


def _candidate_values(owner, observations):
    snapshot = getattr(owner, '_diagnostic_snapshot', None)
    if snapshot is not None:
        if not isinstance(snapshot, _RowSnapshot):
            raise ValueError('invalid candidate diagnostic provenance')
        return snapshot.checked(owner, observations)
    return {name: _ordinary(getattr(owner, name), context=name) for name in (
        'predicted', 'predicted_fraction', 'predicted_position', 'residuals',
    )}


@dataclass(frozen=True, slots=True)
class _HistorySnapshot:
    observations: object
    row_ids: tuple
    iteration: int
    rms: float
    maximum: float
    rms_reason: str | None
    max_reason: str | None

    @classmethod
    def produced(cls, observations, iteration, rms, maximum):
        return cls(observations, _row_ids(observations), iteration,
                   float(rms.value), float(maximum.value),
                   rms.reasons, maximum.reasons)


class _HistoryStorage:
    __slots__ = ('_history_snapshot',)


class _HistoryBindingInit:
    def __get__(self, instance, owner=None):
        return (None if instance is None
                else getattr(instance, '_history_snapshot', None))


def _history_values(row, observations):
    snapshot = getattr(row, '_history_snapshot', None)
    if snapshot is None:
        return {name: _ordinary(getattr(row, name), context='history')
                for name in ('rms_residual_all', 'max_residual_all')}
    if not isinstance(snapshot, _HistorySnapshot):
        raise ValueError('invalid history provenance')
    _require_observation_association(
        snapshot.observations, observations, context='history provenance',
    )
    if (snapshot.iteration != row.iteration
            or snapshot.row_ids != _row_ids(observations)):
        raise ValueError('history provenance has stale iteration or rows')
    values = {
        'rms_residual_all': _DiagnosticValue(snapshot.rms, snapshot.rms_reason),
        'max_residual_all': _DiagnosticValue(
            snapshot.maximum, snapshot.max_reason,
        ),
    }
    for name, value in values.items():
        value.require(getattr(row, name), context='history provenance')
    return values


def _rebase_diagnostic_maps(report):
    """Copy exhaustive nested maps into the containing typed report map."""
    unavailable = dict(report['unavailable_diagnostics'])

    def visit(value, prefix):
        if isinstance(value, dict):
            for path, reason in value.get('unavailable_diagnostics', {}).items():
                rebased = prefix + path
                previous = unavailable.setdefault(rebased, reason)
                if previous != reason:
                    raise ValueError('inconsistent nested diagnostic reasons')
            for key, child in value.items():
                if key != 'unavailable_diagnostics':
                    escaped = key.replace('~', '~0').replace('/', '~1')
                    visit(child, prefix + '/' + escaped)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, prefix + '/' + str(index))

    for key, value in report.items():
        if key != 'unavailable_diagnostics':
            visit(value, '/' + key.replace('~', '~0').replace('/', '~1'))
    report['unavailable_diagnostics'] = unavailable
    return report
