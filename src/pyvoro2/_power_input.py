"""Shared resolution of public power weights and backend radii."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ._inputs import coerce_nonnegative_vector
from ._weight_transforms import weights_to_radii


@dataclass(frozen=True, slots=True)
class ResolvedPowerInput:
    """Validated power input and its backend representation.

    The three fields are kept together so the common forward result can later
    reuse the exact validated representation without repeating conversion.
    """

    input_weights: np.ndarray | None
    backend_radii: np.ndarray | None
    representation_shift: float | None


def resolve_power_input(
    *,
    mode: Literal['standard', 'power'] | str,
    weights: Sequence[float] | np.ndarray | None,
    radii: Sequence[float] | np.ndarray | None,
    n: int,
) -> ResolvedPowerInput:
    """Resolve the mutually exclusive public power representations.

    Standard mode rejects both public power representations because neither
    weights nor radii have meaning for an unweighted Voronoi diagram.
    """

    if mode == 'standard':
        if weights is not None:
            raise ValueError('weights is not supported for mode="standard"')
        if radii is not None:
            raise ValueError('radii is not supported for mode="standard"')
        return ResolvedPowerInput(None, None, None)

    if mode != 'power':
        # The operation wrapper retains responsibility for its established
        # ``unknown mode`` error.
        return ResolvedPowerInput(None, None, None)

    if weights is not None and radii is not None:
        raise ValueError(
            'weights and radii are mutually exclusive for mode="power"'
        )
    if weights is None and radii is None:
        raise ValueError(
            'exactly one of weights or radii is required for mode="power"'
        )

    if weights is not None:
        input_weights = np.asarray(weights, dtype=np.float64)
        if input_weights.shape != (n,):
            raise ValueError('weights must have shape (n,)')
        if not np.all(np.isfinite(input_weights)):
            raise ValueError('weights must contain only finite values')
        backend_radii, representation_shift = weights_to_radii(input_weights)
        return ResolvedPowerInput(
            input_weights,
            backend_radii,
            representation_shift,
        )

    assert radii is not None
    backend_radii = coerce_nonnegative_vector(radii, name='radii', n=n)
    return ResolvedPowerInput(None, backend_radii, None)
