"""Same-operation image labels for the existing observed numerical contacts.

This is relative-coordinate membership in the represented native lattice,
not source/native geometry equivalence or an insertion-history certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .inputs import owned_readonly_array
from .native_remap import _align
from .validation import require_index

if TYPE_CHECKING:
    from .generator_preparation import PreparedGenerators


@dataclass(frozen=True, slots=True)
class RealizationReference:
    """Owned consumed fields, indexed by internal site ID for one compute."""

    submitted: np.ndarray
    preparation_shifts: np.ndarray
    lattice: np.ndarray
    anchors: np.ndarray
    anchor_present: np.ndarray
    periodic_axes: tuple[bool, ...]

    def shift_offset(self, i: int, j: int) -> tuple[int, ...]:
        """Return ``K_i - K_j - h`` with exact relative membership for h."""
        prefix = f'observed pair ({i}, {j}): '
        n, dim = self.submitted.shape
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(prefix + 'missing internal site reference')
        if not (self.anchor_present[i] and self.anchor_present[j]):
            raise ValueError(prefix + 'missing genuine native anchor')
        h = [0] * dim
        if not (np.array_equal(self.submitted[i], self.anchors[i])
                and np.array_equal(self.submitted[j], self.anchors[j])):
            values = (*self.submitted[i], *self.submitted[j],
                      *self.anchors[i], *self.anchors[j], *self.lattice.flat)
            aligned, _ = _align(tuple(float(v).as_integer_ratio() for v in values))
            vi, vj, ai, aj = [aligned[k*dim:(k+1)*dim] for k in range(4)]
            rows = [aligned[(4+k)*dim:(5+k)*dim] for k in range(dim)]
            difference = [vj[k] - vi[k] - aj[k] + ai[k] for k in range(dim)]
            # Both native rectangular and triangular lattices are lower
            # triangular. Reverse substitution also checks inactive axes.
            for axis in reversed(range(dim)):
                residual = difference[axis] - sum(
                    h[k] * rows[k][axis] for k in range(axis + 1, dim))
                if not self.periodic_axes[axis]:
                    if residual:
                        raise ValueError(
                            prefix + f'inactive-axis mismatch on axis {axis}')
                    continue
                h[axis], remainder = divmod(residual, rows[axis][axis])
                if remainder:
                    raise ValueError(
                        prefix + f'nonintegral anchor correspondence on axis {axis}')
        return tuple(int(ki) - int(kj) - h_axis for ki, kj, h_axis in zip(
            self.preparation_shifts[i], self.preparation_shifts[j], h))

    def convert_shift(
        self, i: int, j: int, raw_shift, *, offset: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Convert a raw anchor-relative label without a fixed-width cast."""
        prefix = f'observed pair ({i}, {j}): '
        raw = np.asarray(raw_shift, dtype=object)
        if raw.shape != (len(self.periodic_axes),):
            raise ValueError(prefix + 'invalid adjacent_shift shape')
        shift = tuple(require_index(v, name=prefix + 'adjacent_shift') for v in raw)
        if any(s and not active for s, active in zip(shift, self.periodic_axes)):
            raise ValueError(prefix + 'adjacent_shift has an inactive-axis coefficient')
        return tuple(s + correction for s, correction in zip(shift, offset))


def capture_reference(
    prepared: PreparedGenerators, cells: list[dict], *, lattice,
) -> RealizationReference:
    """Capture native readback before empty insertion, ID and Cartesian changes.

    The compute worker has validated native IDs. Missing sites are explicitly
    absent; synthetic empty records are never readback evidence. Membership is
    checked lazily, only when an observed contact needs comparison.
    """
    anchors = np.zeros_like(prepared.native_points)
    present = np.zeros(len(anchors), dtype=bool)
    for cell in cells:
        if cell.get('empty', False) or 'site' not in cell:
            continue
        index = int(cell['id'])
        anchors[index] = cell['site']
        present[index] = True
    return RealizationReference(
        submitted=owned_readonly_array(prepared.native_points, dtype=np.float64),
        preparation_shifts=owned_readonly_array(prepared.remap_shifts, dtype=np.int64),
        lattice=owned_readonly_array(lattice, dtype=np.float64),
        anchors=owned_readonly_array(anchors, dtype=np.float64),
        anchor_present=owned_readonly_array(present, dtype=bool),
        periodic_axes=tuple(prepared.periodic_axes),
    )
