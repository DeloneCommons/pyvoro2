"""Private WP5 failures and refusal-only proof resource accounting."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class WP5Failure(Exception):
    """A certificate reason, converted to TessellationError by the API boundary."""

    def __init__(self, code: str, message: str, **context: object) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


@dataclass(frozen=True, slots=True)
class WP5Limits:
    """Private refusal limits; none changes a successful proof's region."""

    candidate_limit: int = 1_000_000
    work_limit: int = 16_000_000
    bit_limit: int = 16_384


class WP5Budget:
    """Deterministic per-call guards shared by source replay and exact ideals.

    ``candidates`` checks a complete family before iteration. ``charge`` and
    bit checks may abort exact work at any point; callers must propagate the
    failure and may never publish the already processed prefix.
    """

    def __init__(self, limits: WP5Limits | None = None) -> None:
        self.limits = limits if limits is not None else WP5Limits()
        self.work = 0

    def _refuse(self, kind: str, required: int, limit: int, stage: str) -> None:
        raise WP5Failure(
            'WP5_RESOURCE_LIMIT',
            f'WP5 {kind} limit exceeded during {stage}',
            resource=kind, required=required, limit=limit, stage=stage,
        )

    def candidates(self, count: int, *, stage: str = 'candidates') -> None:
        if count > self.limits.candidate_limit:
            self._refuse('candidate', count, self.limits.candidate_limit, stage)

    def charge(self, amount: int = 1, *, stage: str = 'exact') -> None:
        required = self.work + amount
        if required > self.limits.work_limit:
            self._refuse('work', required, self.limits.work_limit, stage)
        self.work = required

    def integer(self, value: int, *, stage: str = 'exact') -> None:
        bits = abs(value).bit_length()
        if bits > self.limits.bit_limit:
            self._refuse('integer bits', bits, self.limits.bit_limit, stage)

    def fraction(self, value: Fraction, *, stage: str = 'exact') -> None:
        self.integer(value.numerator, stage=stage)
        self.integer(value.denominator, stage=stage)
