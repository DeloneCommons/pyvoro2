"""pyvoro2 package.

This package provides Python bindings to the Voro++ cell-based Voronoi
and power (Laguerre) tessellation library, including the planar
``pyvoro2.planar`` namespace for 2D workflows.
"""

from __future__ import annotations

from .__about__ import __version__
from .result import TessellationResult
from . import planar

from .domains import Box, OrthorhombicCell, PeriodicCell
from .api import compute, locate, ghost_cells
from .diagnostics import (
    TessellationDiagnostics,
    TessellationIssue,
    TessellationError,
    analyze_tessellation,
    validate_tessellation,
)

from .validation import (
    NormalizationDiagnostics,
    NormalizationIssue,
    NormalizationError,
    validate_normalized_topology,
)

from .duplicates import (
    DuplicatePair,
    DuplicateError,
    duplicate_check,
)
from .face_properties import annotate_face_properties
from .normalize import (
    NormalizedVertices,
    NormalizedTopology,
    normalize_vertices,
    normalize_edges_faces,
    normalize_topology,
)
from ._weight_transforms import radii_to_weights, weights_to_radii


__all__ = [
    'Box',
    'OrthorhombicCell',
    'PeriodicCell',
    'TessellationResult',
    'compute',
    'locate',
    'ghost_cells',
    'TessellationDiagnostics',
    'TessellationIssue',
    'TessellationError',
    'analyze_tessellation',
    'validate_tessellation',
    'NormalizationDiagnostics',
    'NormalizationIssue',
    'NormalizationError',
    'validate_normalized_topology',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_face_properties',
    'NormalizedVertices',
    'NormalizedTopology',
    'normalize_vertices',
    'normalize_edges_faces',
    'normalize_topology',
    'radii_to_weights',
    'weights_to_radii',
    '__version__',
    'planar',
]
