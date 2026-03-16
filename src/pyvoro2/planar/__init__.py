"""Planar 2D namespace for pyvoro2."""

from __future__ import annotations

from ..duplicates import DuplicateError, DuplicatePair
from ..edge_properties import annotate_edge_properties
from ..viz2d import plot_tessellation
from .api import compute, ghost_cells, locate
from .domains import Box, RectangularCell
from .duplicates import duplicate_check

__all__ = [
    'Box',
    'RectangularCell',
    'compute',
    'locate',
    'ghost_cells',
    'DuplicatePair',
    'DuplicateError',
    'duplicate_check',
    'annotate_edge_properties',
    'plot_tessellation',
]
