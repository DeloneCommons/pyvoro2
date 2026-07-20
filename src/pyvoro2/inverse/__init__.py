"""High-level inverse fitting for weighted tessellations.

The package root intentionally exposes only the fixed-observation separator
workflow and neutral weight/radius transforms. Advanced separator models,
realization diagnostics, reports, and experimental active-set refinement live
in :mod:`pyvoro2.inverse.separator`.
"""

from __future__ import annotations

from .._weight_transforms import radii_to_weights, weights_to_radii
from .separator import (
    SeparatorFitResult,
    SeparatorObservations,
    fit_weights_from_separators,
    resolve_separator_observations,
)

__all__ = [
    'SeparatorObservations',
    'resolve_separator_observations',
    'SeparatorFitResult',
    'fit_weights_from_separators',
    'weights_to_radii',
    'radii_to_weights',
]
