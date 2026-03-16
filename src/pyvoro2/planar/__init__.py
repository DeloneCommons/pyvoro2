"""Planar 2D namespace for pyvoro2."""

from __future__ import annotations

from ..duplicates import DuplicateError, DuplicatePair
from ..edge_properties import annotate_edge_properties
from ..viz2d import plot_tessellation
from .api import compute, ghost_cells, locate
from .diagnostics import (
    TessellationDiagnostics,
    TessellationError,
    TessellationIssue,
    analyze_tessellation,
    validate_tessellation,
)
from .domains import Box, RectangularCell
from .result import PlanarComputeResult
from .duplicates import duplicate_check
from .normalize import (
    NormalizedTopology,
    NormalizedVertices,
    normalize_edges,
    normalize_topology,
    normalize_vertices,
)
from .validation import (
    NormalizationDiagnostics,
    NormalizationError,
    NormalizationIssue,
    validate_normalized_topology,
)

__all__ = [
    'Box',
    'RectangularCell',
    'PlanarComputeResult',
    'compute',
    'locate',
    'ghost_cells',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_edge_properties',
    'plot_tessellation',
    'TessellationIssue',
    'TessellationDiagnostics',
    'TessellationError',
    'analyze_tessellation',
    'validate_tessellation',
    'NormalizedVertices',
    'NormalizedTopology',
    'normalize_vertices',
    'normalize_edges',
    'normalize_topology',
    'NormalizationIssue',
    'NormalizationDiagnostics',
    'NormalizationError',
    'validate_normalized_topology',
]
