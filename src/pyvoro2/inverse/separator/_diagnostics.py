"""Private evaluation values, availability and owned diagnostic provenance.

Only typed producers establish reasons. Exporters compare supplied values with
those producers before converting individual eligible leaves to JSON null.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import sys

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
_MAX_EXACT = Fraction(sys.float_info.max)


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


def _affine_operands(beta, alpha, left, right, target, *scales):
    operands = np.broadcast_arrays(*[
        np.asarray(v, dtype=np.float64)
        for v in (beta, alpha, left, right, target, *scales)
    ])
    available = np.logical_and.reduce([np.isfinite(v) for v in operands])
    absent = np.zeros(available.shape, dtype=bool)
    for scale in operands[5:]:
        absent |= scale == 0.
    return operands, available, absent


def _affine_terms_bounded(operands, absent, limit):
    """Prove each original scaled term is strictly below 2**limit."""
    scale_exponent = np.zeros(absent.shape, dtype=np.int64)
    for scale in operands[5:]:
        scale_exponent += np.frexp(scale)[1]
    bounded = np.ones(absent.shape, dtype=bool)
    for indices in ((0,), (1, 2), (1, 3), (4,)):
        exponent = scale_exponent
        nonzero = ~absent
        for index in indices:
            exponent = exponent + np.frexp(operands[index])[1]
            nonzero &= operands[index] != 0.
        bounded &= ~nonzero | (exponent <= limit)
    return bounded


def _exact_affine_row(values):
    if any(v == 0. for v in values[5:]):
        return Fraction()
    b, a, l, r, t, *factors = (Fraction(float(v)) for v in values)
    residual = b + a * l - a * r - t
    for factor in factors:
        residual *= factor
    return residual


def _affine_diagnostic(beta, alpha, left, right, target, *scales):
    operands, available, absent = _affine_operands(
        beta, alpha, left, right, target, *scales,
    )
    work = available & ~absent
    # Four terms below 2**1021 give an exact row below 2**1023 < MAX.
    # Otherwise retain the exact range decision until producer materialization:
    # an exact MAX + tiny can round to finite MAX even in a stable evaluator.
    ordinary = _affine_terms_bounded(operands, absent, 1021)
    if scales:
        value = _stable_scaled_affine_residual(
            *operands[:5], *operands[5:], active=work,
        )
    else:
        value = _stable_affine_residual(*operands[:5], active=work)
    for index in np.flatnonzero(work & ~ordinary):
        exact = _exact_affine_row([v.flat[index] for v in operands])
        value.flat[index] = (
            (-np.inf if exact < 0 else np.inf)
            if abs(exact) > _MAX_EXACT else float(exact)
        )
    return _derived(value, operands_available=available | absent)


def _has_dependency(value):
    return _DEPENDENCY in value.reasons


def _weighted_affine_norms(beta, alpha, left, right, target, confidence):
    """Classify both norms from original c*r**2, never rounded sqrt(c)."""
    operands, available, absent = _affine_operands(
        beta, alpha, left, right, target, confidence,
    )
    if np.any(~available & ~absent):
        unavailable = _derived(np.nan, operands_available=False)
        return unavailable, unavailable
    count = operands[0].size
    if count == 0:
        return _derived(0.), _derived(0.)
    confidence = operands[5]
    # If each affine term is <2**e, |r| <2**(e+2). With c <2**ce
    # and count <=2**k, sum(c*r**2) <2**(2*e+4+ce+k). Requiring
    # that exponent <=2046 proves both norms <2**1023, safely below MAX.
    # sqrt(c) is allowed for the ordinary VALUE only after this proof.
    count_exponent = (count - 1).bit_length()
    limit = (2042 - np.frexp(confidence)[1] - count_exponent) // 2
    if np.all(_affine_terms_bounded(operands[:5], absent, limit)):
        weighted = _stable_scaled_affine_residual(
            *operands[:5], np.sqrt(confidence), active=~absent,
        )
        if np.all(np.isfinite(weighted)):
            return _derived(_stable_norm(weighted)), _derived(_stable_rms(weighted))

    # All required dependencies were checked before any early range exit.
    # Zero-confidence rows contribute zero but still count in the RMSE.
    total = Fraction()
    for values in zip(*(v.ravel() for v in operands)):
        if values[5] == 0.:
            continue
        residual = _exact_affine_row(values[:5])
        total += Fraction(float(values[5])) * residual**2
        if total > count * _MAX_EXACT**2:
            return _derived(np.inf), _derived(np.inf)
    return (_exact_average_diagnostic(total, 1, mean_abs=False),
            _exact_average_diagnostic(total, count, mean_abs=False))


def _affine_rms(beta, alpha, left, right, target, rows, *scales):
    """Reduce complete scaled affine rows, deciding range before rounding."""
    if _has_dependency(rows):
        return _derived(np.nan, operands_available=False)
    count = rows.value.size
    if count == 0:
        return _derived(0.)
    operands, available, absent = _affine_operands(
        beta, alpha, left, right, target, *scales,
    )
    if np.any(~available & ~absent):
        return _derived(np.nan, operands_available=False)

    # Bound all four ORIGINAL scaled terms by 2**1021, hence each exact
    # row and its RMS by 2**1023 < MAX. Integer exponent sums never form
    # overflowing products. Finite rounded rows alone cannot prove range:
    # even MAX + a subnormal term rounds to finite MAX.
    ordinary = np.all(_affine_terms_bounded(operands, absent, 1021))
    if ordinary and np.all(np.isfinite(rows.value)):
        return _derived(_stable_rms(rows.value))

    def exact_rows():
        for values in zip(*(v.ravel() for v in operands)):
            yield _exact_affine_row(values)

    return _exact_mean_diagnostic(exact_rows(), count, mean_abs=False)


def _exact_mean_diagnostic(rows, count, *, mean_abs):
    """Reduce exact rows after the owner has checked ALL required operands."""
    if count == 0:
        return _derived(0.)
    power = 1 if mean_abs else 2
    limit = count * _MAX_EXACT**power
    total = Fraction()
    for residual in rows:
        total += abs(residual) if mean_abs else residual * residual
        # Contributions are nonnegative, so this proves the complete sum
        # exceeds count*MAX or count*MAX**2, without rounded normalization.
        if total > limit:
            return _derived(np.inf)
    return _exact_average_diagnostic(total, count, mean_abs=mean_abs)


def _exact_average_diagnostic(total, count, *, mean_abs):
    """Decide final range from the exact sum, then round the finite value."""
    if total > count * _MAX_EXACT**(1 if mean_abs else 2):
        return _derived(np.inf)
    mean = total / count
    if mean_abs:
        return _derived(float(mean))
    # Range was decided exactly, before sqrt. Extra decimal precision is
    # only for the finite binary64 value, not a new correct-rounding promise.
    # A private context also avoids inheriting caller decimal policy.
    with localcontext(Context(
        prec=80, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
        clamp=0, flags=[], traps=[],
    )):
        value = (Decimal(mean.numerator) / Decimal(mean.denominator)).sqrt()
        return _derived(float(value))


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
