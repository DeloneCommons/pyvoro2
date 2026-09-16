"""Shared resolution of public power weights and backend radii."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .inputs import (
    coerce_finite_scalar_or_vector,
    coerce_finite_vector,
    coerce_nonnegative_scalar_or_vector,
    coerce_nonnegative_vector,
)
from .weight_transforms import weights_to_radii


@dataclass(frozen=True, slots=True)
class ResolvedPowerInput:
    """Validated power input and its backend representation.

    The three fields are kept together so the common forward result can later
    reuse the exact validated representation without repeating conversion.
    """

    input_weights: np.ndarray | None
    backend_radii: np.ndarray | None
    representation_shift: float | None


@dataclass(frozen=True, slots=True)
class ResolvedGhostPowerInput:
    """Validated persistent and temporary backend power representations."""

    backend_radii: np.ndarray | None
    backend_ghost_radii: np.ndarray | None


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
        input_weights = coerce_finite_vector(weights, name='weights', n=n)
        backend_radii, representation_shift = weights_to_radii(input_weights)
        return ResolvedPowerInput(
            input_weights,
            backend_radii,
            representation_shift,
        )

    assert radii is not None
    backend_radii = coerce_nonnegative_vector(radii, name='radii', n=n)
    return ResolvedPowerInput(None, backend_radii, None)


def resolve_ghost_power_input(
    *,
    mode: Literal['standard', 'power'] | str,
    weights: Sequence[float] | np.ndarray | None,
    radii: Sequence[float] | np.ndarray | None,
    ghost_weights: float | Sequence[float] | np.ndarray | None,
    ghost_radii: float | Sequence[float] | np.ndarray | None,
    n: int,
    m: int,
) -> ResolvedGhostPowerInput:
    """Resolve one complete persistent/temporary power-input family.

    Mathematical weights are converted together so persistent generators and
    temporary ghost generators share one global representation gauge.
    """

    supplied = (
        ('weights', weights),
        ('radii', radii),
        ('ghost_weights', ghost_weights),
        ('ghost_radii', ghost_radii),
    )
    if mode == 'standard':
        for name, value in supplied:
            if value is not None:
                raise ValueError(
                    f'{name} is not supported for mode="standard"'
                )
        return ResolvedGhostPowerInput(None, None)

    if mode != 'power':
        return ResolvedGhostPowerInput(None, None)

    weight_family = (
        weights is not None
        and ghost_weights is not None
        and radii is None
        and ghost_radii is None
    )
    radius_family = (
        radii is not None
        and ghost_radii is not None
        and weights is None
        and ghost_weights is None
    )
    if not (weight_family or radius_family):
        raise ValueError(
            'mode="power" requires exactly one complete representation family: '
            'weights with ghost_weights, or radii with ghost_radii'
        )

    if weight_family:
        assert weights is not None
        assert ghost_weights is not None
        persistent_weights = coerce_finite_vector(
            weights,
            name='weights',
            n=n,
        )
        temporary_weights = coerce_finite_scalar_or_vector(
            ghost_weights,
            name='ghost_weights',
            n=m,
            length_name='m',
        )
        combined_weights = np.concatenate(
            (persistent_weights, temporary_weights)
        )
        combined_radii, _representation_shift = weights_to_radii(
            combined_weights
        )
        return ResolvedGhostPowerInput(
            combined_radii[:n],
            combined_radii[n:],
        )

    assert radii is not None
    assert ghost_radii is not None
    backend_radii = coerce_nonnegative_vector(radii, name='radii', n=n)
    backend_ghost_radii = coerce_nonnegative_scalar_or_vector(
        ghost_radii,
        name='ghost_radii',
        n=m,
        length_name='m',
    )
    return ResolvedGhostPowerInput(
        backend_radii,
        backend_ghost_radii,
    )
