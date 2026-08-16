"""Shared private policy helpers for tessellation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Sequence

import numpy as np


Severity = Literal['info', 'warning', 'error']


@dataclass(frozen=True, slots=True)
class MeasureValidation:
    """Validated cell measure and any deterministic category failures."""

    value: float | None
    issue_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issue_codes


@dataclass(frozen=True, slots=True)
class ExpectedIdClassification:
    """Mode-aware classification of absent expected cell IDs."""

    missing_ids: tuple[int, ...]
    hidden_ids: tuple[int, ...]
    issue_code: str | None
    severity: Severity | None


def validate_cell_measure(
    cell: Mapping[str, Any],
    *,
    field: str,
    empty: bool,
) -> MeasureValidation:
    """Validate one raw area/volume value without lossy category coercion."""

    if field not in cell:
        if empty:
            return MeasureValidation(None)
        return MeasureValidation(None, ('MISSING_CELL_MEASURE',))

    value = cell[field]
    if (
        isinstance(
            value,
            (bool, np.bool_, complex, np.complexfloating, str, bytes, np.ndarray),
        )
        or not isinstance(value, (Real, np.integer, np.floating))
    ):
        return MeasureValidation(None, ('INVALID_CELL_MEASURE',))

    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return MeasureValidation(None, ('NONFINITE_CELL_MEASURE',))
    if not math.isfinite(converted):
        return MeasureValidation(None, ('NONFINITE_CELL_MEASURE',))

    issue_codes: list[str] = []
    if converted < 0.0:
        issue_codes.append('NEGATIVE_CELL_MEASURE')
    if empty and converted != 0.0:
        issue_codes.append('EMPTY_CELL_NONZERO_MEASURE')
    return MeasureValidation(converted, tuple(issue_codes))


def stable_measure_sum(values: Iterable[float]) -> float:
    """Reduce validated finite non-negative measures with stable summation.

    If their mathematical sum exceeds binary64 range, return positive infinity
    so the analyzers can report the aggregate as a closure overlap. Individual
    finite terms remain valid cell measures.
    """

    try:
        return float(math.fsum(values))
    except OverflowError:
        return math.inf


def classify_expected_ids(
    expected_ids: Sequence[int],
    present_ids: Iterable[int],
    *,
    mode: str | None,
) -> ExpectedIdClassification:
    """Classify absent expected IDs under standard/power/raw semantics."""

    missing = tuple(sorted(set(int(x) for x in expected_ids) - set(present_ids)))
    if not missing:
        return ExpectedIdClassification((), (), None, None)
    if mode == 'standard':
        return ExpectedIdClassification(missing, (), 'MISSING_IDS', 'error')
    if mode == 'power':
        return ExpectedIdClassification(missing, missing, 'HIDDEN_IDS', 'info')
    return ExpectedIdClassification(missing, (), 'MISSING_IDS', 'warning')


def reciprocity_issue_severity(
    *,
    required: bool,
    missing_shifts: bool = False,
) -> Severity:
    """Return the frozen required/optional reciprocity issue severity."""

    if required:
        return 'error'
    return 'info' if missing_shifts else 'warning'


def diagnostics_ok(issues: Iterable[object]) -> bool:
    """Return true exactly when no diagnostic issue is error-severity."""

    return not any(getattr(issue, 'severity', None) == 'error' for issue in issues)


def reset_owned_annotations(
    records: Iterable[MutableMapping[str, Any]],
    *,
    fields: Sequence[str],
) -> None:
    """Reset mutable annotations owned by a rerunnable analyzer."""

    for record in records:
        for field in fields:
            record[field] = False
